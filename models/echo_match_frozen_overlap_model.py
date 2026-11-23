from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from .echo_match_model import Echo_Match_Model
from utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Echo_Match_Frozen_Overlap_Model(Echo_Match_Model):
    """Learn matching descriptors using a fixed published overlap predictor."""

    def __init__(self, opt):
        teacher_path = Path(opt['path']['overlap_teacher_checkpoint']).expanduser()
        if not teacher_path.is_file():
            raise FileNotFoundError(f"Missing published overlap teacher checkpoint: {teacher_path}")
        if opt['is_train'] and opt['train']['losses']['echo_match_loss'].get('w_acc', 1) != 0:
            raise ValueError("Frozen overlap training requires w_acc: 0.")

        super().__init__(opt)
        student = self._get_bare_net(self.networks['echo_match_net'])
        teacher = deepcopy(student)
        checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=True)
        weights = checkpoint['networks']['echo_match_net']
        teacher.load_state_dict({name.removeprefix('module.'): value for name, value in weights.items()})
        # BaseModel only switches modes and saves parameters in self.networks.
        self.overlap_teacher = teacher.requires_grad_(False).eval()

    def _setup_networks(self):
        super()._setup_networks()
        student = self.networks['echo_match_net']
        student.overlap_diffusion_net.requires_grad_(False)
        student.permutation_network.requires_grad_(False)

    @torch.no_grad()
    def _predict_overlap(self, data_x, data_y, feat_x_normed, feat_y_normed):
        teacher = self.overlap_teacher
        teacher_x = F.normalize(teacher.feature_extractor(data=data_x), dim=-1, p=2)
        teacher_y = F.normalize(teacher.feature_extractor(data=data_y), dim=-1, p=2)
        return self._overlap_from_descriptors(teacher, data_x, data_y, teacher_x, teacher_y)
