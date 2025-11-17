from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .butterfly import PRButterfly
from .multigrid import DepthwiseLaplacian, MultigridPreconditioner
from .pwdm import PassiveWaveDigitalMixer
from .router import ProximalSparseRouter
from .utils import depthwise_unit_norm


class BoundedDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(channels, 1, kernel_size, kernel_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        weight = depthwise_unit_norm(self.weight)
        return F.conv2d(x, weight, padding=padding, groups=self.channels)


class CheapResidualPath(nn.Module):
    def __init__(self, channels: int, gain: float = 0.25) -> None:
        super().__init__()
        self.channels = channels
        self.gain = gain
        self.weight = nn.Parameter(torch.randn(channels, 1, 1, 1) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = depthwise_unit_norm(self.weight)
        return x + self.gain * F.conv2d(x, weight, groups=self.channels)


class MonotoneImplicitBlock(nn.Module):
    """
    Douglas–Rachford implicit block with PWDM, PBCM, and proximal routing.
    """

    def __init__(
        self,
        channels: int,
        router: ProximalSparseRouter,
        pwm_digital: PassiveWaveDigitalMixer,
        butterfly: PRButterfly,
        preconditioner: MultigridPreconditioner,
        tau: float,
        eta: float,
        rezero_init: float,
        cheap_gain: float = 0.25,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.router = router
        self.pwdm = pwm_digital
        self.butterfly = butterfly
        self.preconditioner = preconditioner
        self.local_conv = BoundedDepthwiseConv(channels)
        self.cheap_path = CheapResidualPath(channels, gain=cheap_gain)
        self.laplacian = DepthwiseLaplacian(channels)
        self.tau = tau
        self.eta = eta
        self.alpha = nn.Parameter(torch.tensor(rezero_init))

    def forward(
        self,
        z: torch.Tensor,
        steer: Optional[torch.Tensor] = None,
        modulation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        u, diagnostics = self.router(z)
        mask = diagnostics["token_mask"] * diagnostics["channel_mask"]

        pwm = self.pwdm(u, steer=steer, modulation=modulation)
        local = self.local_conv(u)
        mixed = self.butterfly(u, modulation=modulation)
        f_linear = pwm + local + mixed

        cheap = self.cheap_path(u)
        f_u = mask * f_linear + (1.0 - mask) * cheap

        lap_u = self.laplacian(u)
        rezero = torch.tanh(self.alpha)
        rhs = u - self.tau * lap_u + rezero * f_u
        v = self.preconditioner.solve(rhs)
        return u + self.eta * (v - u)
