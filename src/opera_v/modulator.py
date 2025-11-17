from __future__ import annotations

from typing import List

import torch
from torch import nn


class KYPCodebookModulator(nn.Module):
    """
    Tiny hypernetwork that produces stability-preserving deltas (Section G).
    """

    def __init__(
        self,
        num_scales: int,
        stats_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> None:
        super().__init__()
        self.stats_dim = stats_dim
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.02)
        self.scale_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(stats_dim, stats_dim),
                    nn.GELU(),
                    nn.Linear(stats_dim, codebook_size),
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, stats: List[torch.Tensor]) -> List[torch.Tensor]:
        latents: List[torch.Tensor] = []
        for head, s in zip(self.scale_heads, stats):
            logits = head(s)
            weights = torch.softmax(logits, dim=-1)
            latent = torch.matmul(weights, self.codebook)
            latents.append(latent)
        return latents
