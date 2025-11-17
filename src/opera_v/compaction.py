from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .lifting import HouseholderProjection


class OrthonormalCompactor(nn.Module):
    """
    Small orthonormal projector used to decorrelate PSLF bands per scale (Section B).
    Optionally truncates the last channels to mimic energy-based pruning without sorting.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reflections: int = 2,
        dropout_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self.mix = HouseholderProjection(in_channels, in_channels, reflections=reflections)
        self.out_channels = out_channels
        self.dropout_ratio = dropout_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.mix(x)
        keep = min(self.out_channels, mixed.shape[1])
        truncated = mixed[:, :keep]
        if self.dropout_ratio > 0.0 and keep > 1:
            cheap_keep = max(int(keep * (1.0 - self.dropout_ratio)), 1)
            truncated = truncated[:, :cheap_keep]
        return truncated
