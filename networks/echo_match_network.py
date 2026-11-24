import torch
import torch.nn as nn
from networks.diffusion_a_network import DiffusionNet_A
from networks.overlap_predictor import CycleConsistencyOverlapPredictor
from utils.fmap_util import get_mask
from utils.registry import NETWORK_REGISTRY

class RegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        # compute linear operator matrix representation C1 and C2
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)

        F_hat = torch.bmm(evecs_trans_x, feat_x)
        G_hat = torch.bmm(evecs_trans_y, feat_y)
        A, B = F_hat, G_hat

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C
class FasterRegularizedFMNet(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, lambda_=1e-3, resolvant_gamma=0.5):
        super().__init__()
        self.lambda_ = lambda_
        self.resolvant_gamma = resolvant_gamma

    def forward(self, feat_x, feat_y, evals_x, evals_y, evecs_trans_x, evecs_trans_y):
        evecs_trans_x, evecs_trans_y = evecs_trans_x.unsqueeze(0), evecs_trans_y.unsqueeze(0)
        evals_x, evals_y = evals_x.unsqueeze(0), evals_y.unsqueeze(0)
        A = torch.bmm(evecs_trans_x, feat_x)  # [B, K, C]
        B = torch.bmm(evecs_trans_y, feat_y)  # [B, K, C]

        D = get_mask(evals_x.flatten(), evals_y.flatten(), self.resolvant_gamma, feat_x.device).unsqueeze(0)

        A_t = A.transpose(1, 2)  # [B, C, K]
        A_A_t = torch.bmm(A, A_t)  # [B, K, K]
        B_A_t = torch.bmm(B, A_t)  # [B, K, K]

        DiagD = torch.diag_embed(D)       # [B, K, K, K]
        Coeff = A_A_t.unsqueeze(1) + self.lambda_ * DiagD   # [B, K, K, K]

        # Use ROW i of BAt as RHS (matches the loop): [B, K, K, 1]
        RHS = B_A_t.unsqueeze(-1)           # no transpose here

        # Option 1: numerically stable
        X = torch.linalg.solve(Coeff, RHS)      # [B, K, K, 1]
        Cxy = X.squeeze(-1)   

        return Cxy


@NETWORK_REGISTRY.register()
class Echo_Match_Net(nn.Module):
    """Compute the functional map matrix representation."""

    def __init__(self, cfg, input_type='xyz', augmentation={'train': {}, 'test': {}},
                 overlap_predictor='cycle_consistency'):
        super().__init__()
        self.cfg = cfg

        # feature extractor
        self.feature_extractor = DiffusionNet_A(
            in_channels=cfg["feature_extractor"]["in_channels"],
            out_channels=cfg["feature_extractor"]["out_channels"],
            hidden_channels=cfg["feature_extractor"]["out_channels"],
            n_block=4,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            input_type=input_type,
            augmentation=augmentation
        )

        if overlap_predictor == 'cycle_consistency':
            self.overlap_predictor = CycleConsistencyOverlapPredictor(
                neighbor_size=cfg['overlap']['neighbor_size'],
                hidden_channels=cfg['overlap']['hidden_channels'],
                blocks=cfg['overlap']['blocks'],
            )
        elif overlap_predictor == 'none':
            self.overlap_predictor = None
        else:
            raise ValueError(f'Unknown EchoMatch overlap predictor: {overlap_predictor!r}')

        # regularized fmap
        self.fmreg_net = FasterRegularizedFMNet(lambda_=cfg["fmap"]["lambda_"], resolvant_gamma=cfg["fmap"]["resolvant_gamma"])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Published EchoMatch checkpoints stored both predictor modules directly on the net.
        # Leave duplicate or disabled-predictor keys for strict loading to reject.
        if isinstance(self.overlap_predictor, CycleConsistencyOverlapPredictor):
            legacy_prefixes = (prefix + 'permutation_network.', prefix + 'overlap_diffusion_net.')
            for key in list(state_dict):
                if key.startswith(legacy_prefixes):
                    new_key = prefix + 'overlap_predictor.' + key[len(prefix):]
                    if new_key not in state_dict:
                        state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
