import torch
from torch import nn

from networks.diffusion_a_network import DiffusionNet_A
from networks.permutation_network import Similarity


class CycleConsistencyOverlapPredictor(nn.Module):
    """Predict overlap from the local entries of each soft round-trip map."""

    def __init__(self, neighbor_size, hidden_channels, blocks):
        super().__init__()
        self.neighbor_size = neighbor_size
        learned_tau = nn.Parameter(torch.Tensor([0.10]))
        self.permutation_network = Similarity(tau=learned_tau)
        self.overlap_diffusion_net = DiffusionNet_A(
            in_channels=neighbor_size,
            out_channels=1,
            hidden_channels=hidden_channels,
            n_block=blocks,
            dropout=True,
            with_gradient_features=True,
            with_gradient_rotations=True,
            last_activation=nn.Sigmoid(),
        )

    def forward(self, data_x, data_y, feat_x_normed, feat_y_normed):
        """Return one score per vertex for each shape in a single batched pair."""
        similarity = torch.bmm(feat_x_normed, feat_y_normed.transpose(1, 2))
        Pxy = self.permutation_network(similarity).squeeze(0)
        Pyx = self.permutation_network(similarity.transpose(1, 2)).squeeze(0)

        # Use the dataset's distances only when both shapes provide them.
        if 'dist' in data_x and 'dist' in data_y:
            dists_x, dists_y = data_x['dist'][0], data_y['dist'][0]
        else:
            verts_x, verts_y = data_x['verts'][0], data_y['verts'][0]
            dists_x = torch.cdist(verts_x, verts_x)
            dists_y = torch.cdist(verts_y, verts_y)

        score_x = self._neighborhood_scores(Pxy @ Pyx, dists_x)
        score_y = self._neighborhood_scores(Pyx @ Pxy, dists_y)
        overlap_x = self.overlap_diffusion_net(x_in=score_x.unsqueeze(0), data=data_x).flatten()
        overlap_y = self.overlap_diffusion_net(x_in=score_y.unsqueeze(0), data=data_y).flatten()
        return overlap_x, overlap_y

    def _neighborhood_scores(self, cycle, distances):
        count = min(self.neighbor_size, distances.shape[1])
        _, indices = torch.topk(distances, count, largest=False, sorted=True)
        scores = cycle.gather(1, indices)
        scores = scores / scores.max()

        if count < self.neighbor_size:
            padding = torch.zeros(scores.shape[0], self.neighbor_size - count, device=scores.device)
            scores = torch.cat([scores, padding], dim=1)
        return scores
