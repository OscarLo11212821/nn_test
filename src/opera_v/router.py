from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .norms import RMSNorm
from .utils import depthwise_unit_norm


class ProximalSparseRouter(nn.Module):
    """
    Deterministic proximal router (Section C) that keeps the mapping 1-Lipschitz.
    """

    def __init__(
        self,
        channels: int,
        lambda_init: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.lambda_param = nn.Parameter(torch.full((channels, 1, 1), lambda_init))
        self.channel_lambda = nn.Parameter(torch.full((channels,), lambda_init))
        self.analysis_depthwise = nn.Parameter(torch.randn(channels, 1, 3, 3) * 0.01)
        self.analysis_pointwise = nn.Parameter(torch.randn(channels, channels, 1, 1) * 0.01)
        self.channel_analyzer = nn.Parameter(torch.randn(channels, channels, 1, 1) * 0.01)
        self.norm = RMSNorm(channels)

    def _analysis(self, x: torch.Tensor) -> torch.Tensor:
        dw = depthwise_unit_norm(self.analysis_depthwise)
        pw = depthwise_unit_norm(self.analysis_pointwise.view(self.channels, -1)).view_as(self.analysis_pointwise)
        y = F.conv2d(x, dw, padding=1, groups=self.channels)
        return F.conv2d(y, pw)

    def _soft_threshold(self, x: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(x.abs() - lamb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        normed = self.norm(x)
        saliency = self._analysis(normed)
        threshold = torch.nn.functional.softplus(self.lambda_param).view(1, self.channels, 1, 1)
        prox = self._soft_threshold(saliency, threshold)
        delta = prox - saliency

        # Synthesis via analysis transpose keeps the prox map nonexpansive.
        dw = depthwise_unit_norm(self.analysis_depthwise)
        pw = depthwise_unit_norm(self.analysis_pointwise.view(self.channels, -1)).view_as(self.analysis_pointwise)
        back = F.conv_transpose2d(delta, pw)
        back = F.conv_transpose2d(back, dw, padding=1, groups=self.channels)
        u = x + back

        mask = (prox.abs() > self.eps).to(x.dtype)
        channel_scores = F.conv2d(
            prox,
            depthwise_unit_norm(self.channel_analyzer.view(self.channels, -1)).view_as(self.channel_analyzer),
        )
        channel_threshold = torch.nn.functional.softplus(self.channel_lambda).view(1, self.channels, 1, 1)
        channel_prox = self._soft_threshold(channel_scores, channel_threshold)
        channel_mask = (channel_prox.abs() > self.eps).to(x.dtype)

        diagnostics = {
            "token_mask": mask,
            "channel_mask": channel_mask,
            "prox": prox,
            "channel_prox": channel_prox,
        }
        return u, diagnostics
