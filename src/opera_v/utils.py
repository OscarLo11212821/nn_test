from __future__ import annotations

from typing import Iterable, Tuple

import torch


def clamp_preserve_grad(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """
    Clamp with gradient pass-through (identity gradient in unclamped region, zero outside).
    """

    if min_value is None and max_value is None:
        return x
    if min_value is None:
        return x - (x - max_value).relu()
    if max_value is None:
        return x + (min_value - x).relu()
    return max_value - (max_value - min_value - (x - min_value).relu()).relu()


def positive_param(param: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """
    Map an unconstrained parameter to (floor, ∞) via softplus with a stability floor.
    """

    return torch.nn.functional.softplus(param) + floor


def normalized_vector(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Return a unit vector with gradient-safe normalization.
    """

    return v * torch.rsqrt(torch.clamp(v.pow(2).sum(dim=-1, keepdim=True), min=eps))


def unitary_two_tap(angle: torch.Tensor) -> torch.Tensor:
    """
    Return 2-tap coefficients parameterized by a single Givens angle.
    """

    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def depthwise_unit_norm(weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize depthwise convolution kernels channel-wise to keep spectral norm ≤ 1.
    """

    flat = weight.view(weight.shape[0], -1)
    scale = torch.rsqrt(torch.clamp((flat.pow(2)).sum(dim=1, keepdim=True), min=eps))
    return weight * scale.view(-1, 1, 1, 1)


def log_safe_exp(delta: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    """
    Map unconstrained deltas to bounded exponentials used for resistance modulation.
    """

    return torch.exp(torch.clamp(delta, min=-cap, max=cap))


def pad_to_power_of_two(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Pad the channel dimension to the next power-of-two for butterfly operations.
    Returns padded tensor and the original channel count.
    """

    c = x.shape[1]
    target = 1 << (c - 1).bit_length()
    if target == c:
        return x, c
    pad = target - c
    zeros = torch.zeros(x.shape[0], pad, *x.shape[2:], device=x.device, dtype=x.dtype)
    return torch.cat((x, zeros), dim=1), c


def undo_padding(x: torch.Tensor, original_channels: int) -> torch.Tensor:
    """
    Remove butterfly padding along the channel dimension.
    """

    if x.shape[1] == original_channels:
        return x
    return x[:, :original_channels]


def pair_indices(dim: int, offset: int = 0) -> Iterable[Tuple[int, int]]:
    """
    Yield index pairs for butterfly/Givens sweeps.
    """

    start = offset % 2
    for i in range(start, dim - 1, 2):
        yield i, i + 1
