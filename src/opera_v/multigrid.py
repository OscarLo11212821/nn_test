from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


class DepthwiseLaplacian(nn.Module):
    """
    Depthwise discrete Laplacian used as SPD operator G.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel = torch.zeros(channels, 1, 3, 3)
        kernel[:, 0, 1, 1] = -4.0
        kernel[:, 0, 0, 1] = 1.0
        kernel[:, 0, 2, 1] = 1.0
        kernel[:, 0, 1, 0] = 1.0
        kernel[:, 0, 1, 2] = 1.0
        self.register_buffer("kernel", kernel, persistent=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1, groups=self.channels)


class MultigridPreconditioner(nn.Module):
    """
    Single V-cycle multigrid solver using PSLF restriction/prolongation.
    """

    def __init__(
        self,
        channels: int,
        restrict_fn: Callable[[torch.Tensor], torch.Tensor],
        prolong_fn: Callable[[torch.Tensor], torch.Tensor],
        tau: float = 0.075,
        levels: int = 2,
        smoothing_steps: int = 2,
        damping: float = 0.7,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.restrict_fn = restrict_fn
        self.prolong_fn = prolong_fn
        self.tau = tau
        self.levels = levels
        self.smoothing_steps = smoothing_steps
        self.damping = damping
        self.laplacian = DepthwiseLaplacian(channels)

    def _apply_operator(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.tau * self.laplacian(x)

    def _smooth(self, estimate: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        residual = rhs - self._apply_operator(estimate)
        return estimate + self.damping * residual

    def _v_cycle(self, rhs: torch.Tensor, depth: int) -> torch.Tensor:
        v = torch.zeros_like(rhs)
        for _ in range(self.smoothing_steps):
            v = self._smooth(v, rhs)

        if depth >= self.levels or min(rhs.shape[2:]) <= 2:
            return v

        residual = rhs - self._apply_operator(v)
        coarse_rhs = self.restrict_fn(residual)
        coarse_sol = self._v_cycle(coarse_rhs, depth + 1)
        correction = self.prolong_fn(coarse_sol)
        v = v + correction
        for _ in range(self.smoothing_steps):
            v = self._smooth(v, rhs)
        return v

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        return self._v_cycle(rhs, depth=0)
