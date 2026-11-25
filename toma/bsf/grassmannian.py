import torch
import torch.nn as nn

from .base import BSF, group_topk


class GrassmannianBSF(BSF):
    def __init__(self, d, n_groups, group_size=3, l0=16):
        super().__init__(d, n_groups, group_size)
        self.l0 = l0
        B = torch.randn(n_groups, d, group_size)
        # orthonormal columns
        B, _ = torch.linalg.qr(B)
        self.B_raw = nn.Parameter(B)
        # learned per-block scale (linear, init 1.0), one gamma per concept/frame,
        # compensating the energy lost by tying encoder = decoder
        self.gamma = nn.Parameter(torch.ones(n_groups))

    # (n_groups * group_size, d), orthonormal within each concept
    def decoder_atoms(self):
        # (n_groups, d, group_size)
        B, _ = torch.linalg.qr(self.B_raw)
        return B.permute(0, 2, 1).reshape(-1, self.d)

    def encode(self, x):
        # (n_groups * group_size, d)
        atoms = self.decoder_atoms()
        z = (x @ atoms.t()).reshape(-1, self.n_groups, self.group_size)
        # tied encoder = per-block gamma * D
        z = self.gamma.view(1, -1, 1) * z
        mask = group_topk(z.norm(dim=-1), self.l0)
        return z * mask.unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        x_hat, _ = self(x)
        recon = (target - x_hat).pow(2).mean()
        return recon, {'recon': recon.item()}
