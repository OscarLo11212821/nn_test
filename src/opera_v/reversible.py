from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn


class ReversibleMacroBlock(nn.Module):
    """
    Reversible macro-block composed of paired monotone implicit blocks (Section H).
    """

    def __init__(
        self,
        channels: int,
        block_factory: Callable[[], nn.Module],
        depth: int,
    ) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("Reversible blocks require an even number of channels.")
        self.channels = channels
        self.depth = depth
        self.f_blocks = nn.ModuleList([block_factory() for _ in range(depth)])
        self.g_blocks = nn.ModuleList([block_factory() for _ in range(depth)])

    def forward(self, x: torch.Tensor, **context: Any) -> torch.Tensor:
        a, b = torch.chunk(x, 2, dim=1)
        for f_block, g_block in zip(self.f_blocks, self.g_blocks):
            a = a + f_block(b, **context)
            b = b + g_block(a, **context)
        return torch.cat([a, b], dim=1)
