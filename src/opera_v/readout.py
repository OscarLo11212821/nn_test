from __future__ import annotations

from typing import List

import torch
from torch import nn

from .butterfly import PRButterfly
from .norms import RMSNorm


class ReadoutHead(nn.Module):
    """
    Multi-scale pooled readout with PBCM pre-mix (Section I).
    """

    def __init__(self, scale_dims: List[int], num_classes: int, p_mean: float = 3.0) -> None:
        super().__init__()
        self.scale_dims = scale_dims
        self.num_classes = num_classes
        self.p_mean = p_mean
        pooled_dim = sum(dim * 3 for dim in scale_dims)
        self.pre_mix = PRButterfly(pooled_dim)
        hidden = max(num_classes * 2, pooled_dim)
        self.head = nn.Sequential(
            RMSNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=(2, 3))
        max_val = torch.amax(x, dim=(2, 3))
        p = torch.clamp(x.abs(), min=1e-6).pow(self.p_mean)
        p_mean = p.mean(dim=(2, 3)).pow(1.0 / self.p_mean)
        return torch.cat([mean, max_val, p_mean], dim=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pooled = [self._pool(feat) for feat in features]
        concat = torch.cat(pooled, dim=1)
        mixed = self.pre_mix(concat.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        return self.head(mixed)
