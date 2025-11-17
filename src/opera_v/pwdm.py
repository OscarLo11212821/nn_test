from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .utils import clamp_preserve_grad, positive_param


class PassiveWaveDigitalMixer(nn.Module):
    """
    Passive wave-digital mixer (Section D). Implements local scattering with positive
    resistances and bounded gains to keep the operator strictly dissipative.
    """

    def __init__(
        self,
        channels: int,
        orientation_channels: int,
        use_diagonals: bool = True,
        modulation_dim: int = 32,
        gain_cap: float = 0.99,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_diagonals = use_diagonals
        self.gain_cap = gain_cap
        self.eps = eps

        self.r_center = nn.Parameter(torch.zeros(channels))
        self.r_horizontal = nn.Parameter(torch.zeros(channels))
        self.r_vertical = nn.Parameter(torch.zeros(channels))
        self.r_diag = nn.Parameter(torch.zeros(channels))

        self.gain_horizontal = nn.Parameter(torch.zeros(channels))
        self.gain_vertical = nn.Parameter(torch.zeros(channels))
        self.gain_diag = nn.Parameter(torch.zeros(channels))

        if orientation_channels > 0:
            proj_out = 4 if use_diagonals else 2
            self.steer_proj = nn.Conv2d(orientation_channels, proj_out, kernel_size=1, bias=False)
        else:
            self.steer_proj = None

        self.mod_linear = nn.Linear(modulation_dim, 3 if use_diagonals else 2, bias=False)

    def _gains(self, modulation: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gh = torch.tanh(self.gain_horizontal).view(1, -1, 1, 1)
        gv = torch.tanh(self.gain_vertical).view(1, -1, 1, 1)
        gd = torch.tanh(self.gain_diag).view(1, -1, 1, 1)

        if modulation is not None:
            delta = torch.tanh(self.mod_linear(modulation)).unsqueeze(-1).unsqueeze(-1)
            gh = clamp_preserve_grad(gh * (1.0 + 0.1 * delta[:, 0:1]), 0.0, self.gain_cap)
            gv = clamp_preserve_grad(gv * (1.0 + 0.1 * delta[:, 1:2]), 0.0, self.gain_cap)
            if self.use_diagonals:
                gd = clamp_preserve_grad(gd * (1.0 + 0.1 * delta[:, 2:3]), 0.0, self.gain_cap)
        return gh, gv, gd

    def forward(
        self,
        x: torch.Tensor,
        steer: Optional[torch.Tensor] = None,
        modulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        center = x
        right = torch.roll(x, shifts=-1, dims=-1)
        left = torch.roll(x, shifts=1, dims=-1)
        up = torch.roll(x, shifts=1, dims=-2)
        down = torch.roll(x, shifts=-1, dims=-2)

        diag_ne = torch.roll(x, shifts=(-1, -1), dims=(-2, -1))
        diag_sw = torch.roll(x, shifts=(1, 1), dims=(-2, -1))
        diag_nw = torch.roll(x, shifts=(1, -1), dims=(-2, -1))
        diag_se = torch.roll(x, shifts=(-1, 1), dims=(-2, -1))

        r_c = positive_param(self.r_center, self.eps).view(1, -1, 1, 1)
        r_h = positive_param(self.r_horizontal, self.eps).view(1, -1, 1, 1)
        r_v = positive_param(self.r_vertical, self.eps).view(1, -1, 1, 1)
        r_d = positive_param(self.r_diag, self.eps).view(1, -1, 1, 1)

        denom_h = r_c + r_h
        denom_v = r_c + r_v
        denom_d = r_c + r_d

        flow_h = ((right - center) + (left - center)) / denom_h
        flow_v = ((up - center) + (down - center)) / denom_v
        flow_d = torch.zeros_like(flow_h)
        if self.use_diagonals:
            flow_d = ((diag_ne - center) + (diag_sw - center) + (diag_nw - center) + (diag_se - center)) / (2.0 * denom_d)

        gh, gv, gd = self._gains(modulation)

        if self.steer_proj is not None and steer is not None:
            steer_fields = torch.tanh(self.steer_proj(steer))
            gh = clamp_preserve_grad(gh * (1.0 + 0.2 * steer_fields[:, 0:1]), 0.0, self.gain_cap)
            gv = clamp_preserve_grad(gv * (1.0 + 0.2 * steer_fields[:, 1:2]), 0.0, self.gain_cap)
            if self.use_diagonals:
                gd = clamp_preserve_grad(gd * (1.0 + 0.2 * steer_fields[:, 2:3]), 0.0, self.gain_cap)

        update = gh * flow_h + gv * flow_v
        if self.use_diagonals:
            update = update + gd * flow_d
        return center + update
