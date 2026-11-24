import torch
import torch.nn.functional as F
from .partial_base_model import PartialBaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.fmap_util import nn_query, fmap2pointmap, get_gt_fmap_with_regularization

@MODEL_REGISTRY.register()
class Echo_Match_Model(PartialBaseModel):
    def __init__(self, opt):
        super(Echo_Match_Model, self).__init__(opt)
        self.prune_strategy = opt.get('prune_strategy', 'f2_only')
        self.lambda_ = opt.get('lambda', 0)
        print(f"prune_strategy: {self.prune_strategy}")

    def feed_data(self, data):
        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        net = self.networks['echo_match_net']

        # extract features
        feat_x = net.feature_extractor(data=data_x)
        feat_y = net.feature_extractor(data=data_y)

        overlap_score12, overlap_score21 = None, None
        if net.overlap_predictor is not None:
            overlap_score12, overlap_score21 = net.overlap_predictor(
                data_x, data_y, F.normalize(feat_x, dim=-1, p=2), F.normalize(feat_y, dim=-1, p=2)
            )

        gt_partiality_mask12 = data['first'].get('partiality_mask', torch.ones((evecs_x.shape[0])).long().to(self.device)).squeeze()
        gt_partiality_mask21 = data['second'].get('partiality_mask', torch.ones((evecs_y.shape[0])).long().to(self.device)).squeeze()

        if self.prune_strategy == "f2_only":
            if net.training:
                use_feat_x = feat_x
                use_feat_y = feat_y * gt_partiality_mask21.unsqueeze(0).unsqueeze(-1)
            else:
                use_feat_x = feat_x
                use_feat_y = feat_y
        elif self.prune_strategy == "both":
            if net.training:
                use_feat_x = feat_x * gt_partiality_mask12.unsqueeze(0).unsqueeze(-1)
                use_feat_y = feat_y * gt_partiality_mask21.unsqueeze(0).unsqueeze(-1)
            else:
                use_feat_x = feat_x
                use_feat_y = feat_y
        elif self.prune_strategy == "none":
            use_feat_x = feat_x
            use_feat_y = feat_y
        else:
            raise ValueError(f"Unknown prune strategy: {self.prune_strategy}")

        # predict fmap
        Cxy = net.fmreg_net(use_feat_x, use_feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)

        # gt correspondence
        corr_x = data['first']['corr'][0]
        corr_y = data['second']['corr'][0]

        # gt functional map
        if self.lambda_ == 0:
            P = torch.zeros((evecs_y.shape[0], evecs_x.shape[0])).to(self.device) # [Ny, Nx]
            P[corr_y, corr_x] = 1
            C_gt = evecs_trans_y @ P @ evecs_x
            C_gt = C_gt.unsqueeze(0)
        else:
            C_gt = get_gt_fmap_with_regularization(corr_x, corr_y, evecs_x, evecs_y, evals_x, evals_y, lambda_=self.lambda_, resolvant_gamma=0.5, device=self.device)

        # loss
        fmap_loss, acc_loss, nce_loss = self.losses["echo_match_loss"](C_gt, Cxy, corr_x, corr_y, feat_x, feat_y,
                             overlap_score12, overlap_score21, gt_partiality_mask12, gt_partiality_mask21)

        self.loss_metrics = {'fmap_loss': fmap_loss}
        if acc_loss is not None:
            self.loss_metrics['acc_loss'] = acc_loss
        self.loss_metrics['nce_loss'] = nce_loss

    def validate_single(self, data, timer):
        # start record
        timer.start()
        
        # get data pair
        data = to_device(data, self.device)
        data_x, data_y = data['first'], data['second']

        # get spectral operators
        evecs_x, evecs_y = data_x['evecs'][0], data_y['evecs'][0]
        evals_x, evals_y = data_x['evals'][0], data_y['evals'][0]
        evecs_trans_x, evecs_trans_y = data_x['evecs_trans'][0], data_y['evecs_trans'][0]

        net = self.networks['echo_match_net']

        # extract features
        feat_x = net.feature_extractor(data=data_x)
        feat_y = net.feature_extractor(data=data_y)

        feat_x_normed = F.normalize(feat_x, dim=-1, p=2)
        feat_y_normed = F.normalize(feat_y, dim=-1, p=2)
        overlap_scores = {}
        if net.overlap_predictor is not None:
            overlap_x, overlap_y = net.overlap_predictor(data_x, data_y, feat_x_normed, feat_y_normed)
            overlap_scores = {'overlap_score12': overlap_x, 'overlap_score21': overlap_y}

        # predict fmap
        Cxy = net.fmreg_net(feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
        Cxy = Cxy.squeeze()

        # convert functional map to point-to-point map
        p2p_fmap = fmap2pointmap(Cxy, evecs_x, evecs_y)

        # direct point-to-point map
        p2p_point = nn_query(feat_x_normed, feat_y_normed)

        p2p_variants = {'fmap': p2p_fmap, 'point': p2p_point}
        if not self.opt['is_train'] and self.opt.get('val', {}).get('oracle_overlap', False):
            # The oracle masks Y's descriptors before spectral projection.
            mask_y = data_y.get('partiality_mask')
            if mask_y is None or mask_y.shape != feat_y.shape[:2]:
                raise ValueError("The overlap oracle requires a GT mask of shape [1, num_vertices_y].")
            feat_y_oracle = feat_y * mask_y.unsqueeze(-1)
            Cxy_oracle = net.fmreg_net(
                feat_x, feat_y_oracle, evals_x, evals_y, evecs_trans_x, evecs_trans_y)
            p2p_variants['fmap_oracle'] = fmap2pointmap(Cxy_oracle.squeeze(), evecs_x, evecs_y)

        # finish record
        timer.record()

        return {
            'p2p': p2p_point,
            **overlap_scores,
            'p2p_variants': p2p_variants,
        }
