from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Lightweight RMS normalization used throughout OPERA‑V.

    Compared to LN/BN it keeps the design paraunitary-friendly and inexpensive.
    """

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x.pow(2), dim=1, keepdim=True)
        normed = x * torch.rsqrt(rms + self.eps)
        if self.affine:
            w = self.weight.view(1, -1, 1, 1)
        else:
            w = self.weight.view(1, -1, 1, 1)
        return normed * w
