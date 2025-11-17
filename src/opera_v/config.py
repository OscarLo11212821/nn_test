from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
from typing import Any, Tuple


@dataclass
class OperaVConfig:
    """
    Canonical configuration dataclass for OPERA‑V.

    The defaults correspond to a lightweight “B” scale model intended for 224×224 inputs.
    """

    image_channels: int = 3
    base_channels: int = 64
    scales: int = 4
    channels_per_scale: Tuple[int, ...] = (64, 128, 192, 256)
    orientations_per_scale: Tuple[int, ...] = (6, 6, 4, 2)
    macro_blocks_per_scale: Tuple[int, ...] = (1, 1, 1, 1)
    reversible_block_depth: int = 2
    implicit_steps_per_block: int = 1
    num_classes: int = 1000
    codebook_size: int = 12
    codebook_dim: int = 48
    router_lambda_init: float = 0.05
    cheap_path_gain: float = 0.25
    multigrid_levels: int = 2
    multigrid_smoothing_steps: int = 2
    multigrid_damping: float = 0.7
    tau: float = 0.075
    eta: float = 0.95
    rezero_init: float = 0.0
    max_diagonal_gain: float = 0.99
    p_readout: float = 3.0
    dropout_channels: float = 0.0
    spectral_clip: float = 1.0
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        if len(self.channels_per_scale) != self.scales:
            raise ValueError("channels_per_scale must have `scales` elements")
        if len(self.orientations_per_scale) != self.scales:
            raise ValueError("orientations_per_scale must have `scales` elements")
        if len(self.macro_blocks_per_scale) != self.scales:
            raise ValueError("macro_blocks_per_scale must have `scales` elements")
        for c in self.channels_per_scale:
            if c % 2 != 0:
                raise ValueError("Each channels_per_scale entry must be even to support reversible splits.")
        if self.channels_per_scale[0] < self.image_channels:
            raise ValueError("First scale must have at least as many channels as the input.")
        for prev, curr in zip(self.channels_per_scale, self.channels_per_scale[1:]):
            if curr < prev:
                raise ValueError("channels_per_scale must be non-decreasing to preserve paraunitary projections.")
        if not (0.0 < self.eta < 2.0):
            raise ValueError("η must lie in (0, 2) for Douglas–Rachford convergence.")
        if not (0.0 < self.max_diagonal_gain <= 1.0):
            raise ValueError("Diagonal gain cap must be in (0, 1].")
        if not (0.0 <= self.dropout_channels < 1.0):
            raise ValueError("Channel dropout ratio must be in [0, 1).")


def default_config(**overrides: Any) -> OperaVConfig:
    """
    Convenience helper that returns an OperaVConfig with optional keyword overrides.
    """

    params = asdict(OperaVConfig())
    params.update(overrides)
    return OperaVConfig(**params)
