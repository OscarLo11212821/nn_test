from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .utils import pad_to_power_of_two, undo_padding


class PRButterfly(nn.Module):
    """
    Orthonormal butterfly mixer (Section F) built from stacked Givens rotations.
    """

    def __init__(
        self,
        channels: int,
        diagonal_cap: float = 0.99,
        modulation_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dim = 1 << (channels - 1).bit_length()
        self.stages = int(math.log2(self.dim))
        self.diagonal_cap = diagonal_cap
        self.total_pairs = self.stages * (self.dim // 2)
        self.angles = nn.Parameter(torch.zeros(self.total_pairs))
        self.diagonal = nn.Parameter(torch.zeros(self.dim))
        self.mod_dim = modulation_dim or channels
        self.angle_delta = nn.Linear(self.mod_dim, self.total_pairs, bias=False)
        self.diag_delta = nn.Linear(self.mod_dim, self.dim, bias=False)

    def _apply_stage(self, x: torch.Tensor, stage_angles: torch.Tensor, block: int) -> torch.Tensor:
        half = block // 2
        b, c, h, w = x.shape
        x = x.view(b, self.dim // block, block, h, w)
        angles = stage_angles.view(b, self.dim // block, half)
        cos = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)
        sin = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)
        left = x[:, :, :half, :, :]
        right = x[:, :, half:, :, :]
        new_left = cos * left + sin * right
        new_right = -sin * left + cos * right
        x = torch.cat([new_left, new_right], dim=2)
        return x.view(b, self.dim, h, w)

    def forward(self, x: torch.Tensor, modulation: Optional[torch.Tensor] = None) -> torch.Tensor:
        y, original = pad_to_power_of_two(x)
        batch = y.shape[0]
        base_angles = self.angles.view(1, -1)
        base_diag = torch.tanh(self.diagonal).clamp(-self.diagonal_cap, self.diagonal_cap).view(1, -1)

        if modulation is not None:
            angle_delta = torch.tanh(self.angle_delta(modulation))
            diag_delta = torch.tanh(self.diag_delta(modulation))
            angles = base_angles + angle_delta
            diag = base_diag + diag_delta
        else:
            angles = base_angles.expand(batch, -1)
            diag = base_diag.expand(batch, -1)

        offset = 0
        for stage in range(self.stages):
            per_stage = self.dim // 2
            block = 2 ** (stage + 1)
            stage_angles = angles[:, offset : offset + per_stage]
            y = self._apply_stage(y, stage_angles, block)
            offset += per_stage

        diag_tensor = diag.view(batch, self.dim).clamp(-self.diagonal_cap, self.diagonal_cap)
        y = y * diag_tensor.unsqueeze(-1).unsqueeze(-1)
        return undo_padding(y, original)
