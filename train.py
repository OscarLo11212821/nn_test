#!/usr/bin/env python3
"""
Single-file OPERA-V reference implementation with a built-in MNIST training loop.

This script inlines all core modules from the original package so it can be
distributed and executed as a standalone Python file. It exposes the exact same
API shape as the multi-file version (`build_opera_v`) and adds a CLI entrypoint
that performs a quick MNIST training run for smoke testing.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def clamp_preserve_grad(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    if min_value is None and max_value is None:
        return x
    if min_value is None:
        return x - (x - max_value).relu()
    if max_value is None:
        return x + (min_value - x).relu()
    return max_value - (max_value - min_value - (x - min_value).relu()).relu()


def positive_param(param: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(param) + floor


def normalized_vector(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v * torch.rsqrt(torch.clamp(v.pow(2).sum(dim=-1, keepdim=True), min=eps))


def unitary_two_tap(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def depthwise_unit_norm(weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = weight.view(weight.shape[0], -1)
    scale = torch.rsqrt(torch.clamp((flat.pow(2)).sum(dim=1, keepdim=True), min=eps))
    view_shape = [weight.shape[0], 1] + [1] * (weight.dim() - 2)
    return weight * scale.view(*view_shape)


def pad_to_power_of_two(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    c = x.shape[1]
    target = 1 << (c - 1).bit_length()
    if target == c:
        return x, c
    pad = target - c
    zeros = torch.zeros(x.shape[0], pad, *x.shape[2:], device=x.device, dtype=x.dtype)
    return torch.cat((x, zeros), dim=1), c


def undo_padding(x: torch.Tensor, original_channels: int) -> torch.Tensor:
    if x.shape[1] == original_channels:
        return x
    return x[:, :original_channels]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.mean(x.pow(2), dim=1, keepdim=True)
        normed = x * torch.rsqrt(rms + self.eps)
        extra_dims = max(0, x.dim() - 2)
        view_shape = [1, -1] + [1] * extra_dims
        return normed * self.weight.view(*view_shape)


# ---------------------------------------------------------------------------
# Paraunitary Steerable Lifting
# ---------------------------------------------------------------------------


class HouseholderProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reflections: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = max(in_channels, out_channels)
        self.reflections = reflections
        self.vectors = nn.Parameter(torch.randn(reflections, self.dim))

    def _orthonormal_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q = torch.eye(self.dim, device=device, dtype=dtype)
        for v in self.vectors:
            v_n = normalized_vector(v)
            q = q - 2.0 * torch.outer(v_n, v_n) @ q
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._orthonormal_matrix(x.device, x.dtype)
        if self.in_channels < self.dim:
            pad = self.dim - self.in_channels
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        weight = q[: self.out_channels, : x.shape[1]].unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, weight)


class SteerableHarmonicBank(nn.Module):
    def __init__(self, max_orientations: int, scales: int) -> None:
        super().__init__()
        base_angles = torch.linspace(0.0, math.pi, steps=max_orientations)
        self.angles = nn.Parameter(base_angles)
        self.diag_weights = nn.Parameter(torch.zeros(max_orientations))
        self.scale_gains = nn.Parameter(torch.ones(scales, max_orientations))

    def forward(
        self,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        grad_diag: torch.Tensor,
        scale_idx: int,
        orientations: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = self.angles[:orientations]
        diag = torch.tanh(self.diag_weights[:orientations])
        gains = torch.nn.functional.softplus(self.scale_gains[scale_idx, :orientations])

        oriented = []
        for k in range(orientations):
            cos_k = torch.cos(angles[k])
            sin_k = torch.sin(angles[k])
            response = cos_k * grad_x + sin_k * grad_y + diag[k] * grad_diag
            oriented.append(response * gains[k])

        stack = torch.stack(oriented, dim=2)
        orientation_field = stack.mean(dim=1)
        detail = stack.permute(0, 2, 1, 3, 4).reshape(
            grad_x.shape[0], orientations * grad_x.shape[1], grad_x.shape[2], grad_x.shape[3]
        )
        return detail, orientation_field


def _vertical_kernel(angle: torch.Tensor) -> torch.Tensor:
    taps = unitary_two_tap(angle)
    kernel = torch.zeros(angle.shape[0], 1, 3, 1, device=angle.device, dtype=angle.dtype)
    kernel[:, 0, 0, 0] = taps[:, 0]
    kernel[:, 0, 2, 0] = taps[:, 1]
    return kernel


def _horizontal_kernel(angle: torch.Tensor) -> torch.Tensor:
    taps = unitary_two_tap(angle)
    kernel = torch.zeros(angle.shape[0], 1, 1, 3, device=angle.device, dtype=angle.dtype)
    kernel[:, 0, 0, 0] = taps[:, 0]
    kernel[:, 0, 0, 2] = taps[:, 1]
    return kernel


def _interleave_rows(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        even.shape[0],
        even.shape[1],
        even.shape[2] + odd.shape[2],
        even.shape[3],
        device=even.device,
        dtype=even.dtype,
    )
    out[:, :, 0::2, :] = even
    out[:, :, 1::2, :] = odd
    return out


def _interleave_cols(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        even.shape[0],
        even.shape[1],
        even.shape[2],
        even.shape[3] + odd.shape[3],
        device=even.device,
        dtype=even.dtype,
    )
    out[:, :, :, 0::2] = even
    out[:, :, :, 1::2] = odd
    return out


class PSLFLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        orientations: int,
        scale_idx: int,
        harmonic_bank: SteerableHarmonicBank,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.orientations = orientations
        self.scale_idx = scale_idx
        self.projection = HouseholderProjection(in_channels, out_channels)
        self.predict_vertical = nn.Parameter(torch.zeros(out_channels))
        self.update_vertical = nn.Parameter(torch.zeros(out_channels))
        self.predict_horizontal = nn.Parameter(torch.zeros(out_channels))
        self.update_horizontal = nn.Parameter(torch.zeros(out_channels))
        self.harmonic_bank = harmonic_bank

    def _lift(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.projection(x)
        even_rows = y[:, :, 0::2, :]
        odd_rows = y[:, :, 1::2, :]

        pred_v = F.conv2d(even_rows, _vertical_kernel(self.predict_vertical), padding=(1, 0), groups=self.out_channels)
        detail_rows = odd_rows - pred_v
        upd_v = F.conv2d(detail_rows, _vertical_kernel(self.update_vertical), padding=(1, 0), groups=self.out_channels)
        low_rows = even_rows + upd_v

        even_cols_low = low_rows[:, :, :, 0::2]
        odd_cols_low = low_rows[:, :, :, 1::2]
        pred_h_low = F.conv2d(
            even_cols_low, _horizontal_kernel(self.predict_horizontal), padding=(0, 1), groups=self.out_channels
        )
        lh = odd_cols_low - pred_h_low
        upd_h_low = F.conv2d(
            lh, _horizontal_kernel(self.update_horizontal), padding=(0, 1), groups=self.out_channels
        )
        ll = even_cols_low + upd_h_low

        even_cols_high = detail_rows[:, :, :, 0::2]
        odd_cols_high = detail_rows[:, :, :, 1::2]
        pred_h_high = F.conv2d(
            even_cols_high, _horizontal_kernel(self.predict_horizontal), padding=(0, 1), groups=self.out_channels
        )
        hh = odd_cols_high - pred_h_high
        upd_h_high = F.conv2d(
            hh, _horizontal_kernel(self.update_horizontal), padding=(0, 1), groups=self.out_channels
        )
        hl = even_cols_high + upd_h_high
        return ll, lh, hl, hh

    def analysis(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ll, lh, hl, hh = self._lift(x)
        detail, orientation_field = self.harmonic_bank(hl, lh, hh, self.scale_idx, self.orientations)
        return ll, detail, orientation_field

    def restrict(self, x: torch.Tensor) -> torch.Tensor:
        ll, _, _, _ = self._lift(x)
        return ll

    def synthesis(self, low: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        even_cols_low = low - F.conv2d(lh, _horizontal_kernel(self.update_horizontal), padding=(0, 1), groups=self.out_channels)
        odd_cols_low = lh + F.conv2d(even_cols_low, _horizontal_kernel(self.predict_horizontal), padding=(0, 1), groups=self.out_channels)
        low_rows = _interleave_cols(even_cols_low, odd_cols_low)

        even_cols_high = hl - F.conv2d(hh, _horizontal_kernel(self.update_horizontal), padding=(0, 1), groups=self.out_channels)
        odd_cols_high = hh + F.conv2d(
            even_cols_high, _horizontal_kernel(self.predict_horizontal), padding=(0, 1), groups=self.out_channels
        )
        detail_rows = _interleave_cols(even_cols_high, odd_cols_high)

        even_rows = low_rows - F.conv2d(
            detail_rows, _vertical_kernel(self.update_vertical), padding=(1, 0), groups=self.out_channels
        )
        odd_rows = detail_rows + F.conv2d(
            even_rows, _vertical_kernel(self.predict_vertical), padding=(1, 0), groups=self.out_channels
        )
        recon = _interleave_rows(even_rows, odd_rows)
        if self.in_channels == self.out_channels:
            return recon
        q = self.projection._orthonormal_matrix(recon.device, recon.dtype)
        weight = q.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        back = F.conv2d(recon, weight)
        return back[:, : self.in_channels]

    def prolongate_lowpass(self, low: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(low)
        return self.synthesis(low, zeros, zeros, zeros)


class ParaunitarySteerableLifting(nn.Module):
    def __init__(
        self,
        in_channels: int,
        lowpass_channels: List[int],
        orientations: List[int],
    ) -> None:
        super().__init__()
        if len(lowpass_channels) != len(orientations):
            raise ValueError("lowpass_channels and orientations must have the same length.")
        self.levels = nn.ModuleList()
        self.scales = len(lowpass_channels)
        harmonic_bank = SteerableHarmonicBank(max_orientations=max(orientations), scales=self.scales)

        prev = in_channels
        for idx, (c, o) in enumerate(zip(lowpass_channels, orientations)):
            self.levels.append(PSLFLevel(prev, c, o, idx, harmonic_bank))
            prev = c
        self.lowpass_channels = lowpass_channels

    def analysis(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        details: List[torch.Tensor] = []
        orientations: List[torch.Tensor] = []
        current = x
        for level in self.levels:
            current, detail, orientation = level.analysis(current)
            details.append(detail)
            orientations.append(orientation)
        return details, orientations, current

    def restrict(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        current = x
        for idx in range(depth):
            current = self.levels[idx].restrict(current)
        return current

    def prolongate(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        current = x
        for idx in reversed(range(depth)):
            current = self.levels[idx].prolongate_lowpass(current)
        return current


# ---------------------------------------------------------------------------
# Orthonormal Compaction
# ---------------------------------------------------------------------------


class OrthonormalCompactor(nn.Module):
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


# ---------------------------------------------------------------------------
# Proximal Router
# ---------------------------------------------------------------------------


class ProximalSparseRouter(nn.Module):
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
        pw = depthwise_unit_norm(self.analysis_pointwise)
        y = F.conv2d(x, dw, padding=1, groups=self.channels)
        return F.conv2d(y, pw)

    def _soft_threshold(self, x: torch.Tensor, lamb: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(x.abs() - lamb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        normed = self.norm(x)
        saliency = self._analysis(normed)
        threshold = torch.nn.functional.softplus(self.lambda_param).view(1, self.channels, 1, 1)
        prox = self._soft_threshold(saliency, threshold)
        delta = prox - saliency

        dw = depthwise_unit_norm(self.analysis_depthwise)
        pw = depthwise_unit_norm(self.analysis_pointwise)
        back = F.conv_transpose2d(delta, pw)
        back = F.conv_transpose2d(back, dw, padding=1, groups=self.channels)
        u = x + back

        mask = (prox.abs() > self.eps).to(x.dtype)
        channel_scores = F.conv2d(prox, depthwise_unit_norm(self.channel_analyzer))
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


# ---------------------------------------------------------------------------
# Passive Wave-Digital Mixer
# ---------------------------------------------------------------------------


class PassiveWaveDigitalMixer(nn.Module):
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


# ---------------------------------------------------------------------------
# Butterfly mixer
# ---------------------------------------------------------------------------


class PRButterfly(nn.Module):
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
        b, _, h, w = x.shape
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


# ---------------------------------------------------------------------------
# Multigrid preconditioner
# ---------------------------------------------------------------------------


class DepthwiseLaplacian(nn.Module):
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


# ---------------------------------------------------------------------------
# Monotone implicit block
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reversible macro-block
# ---------------------------------------------------------------------------


class ReversibleMacroBlock(nn.Module):
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


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class OperaVConfig:
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
    params = asdict(OperaVConfig())
    params.update(overrides)
    return OperaVConfig(**params)


# ---------------------------------------------------------------------------
# Modulator and readout
# ---------------------------------------------------------------------------


class KYPCodebookModulator(nn.Module):
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


class ReadoutHead(nn.Module):
    def __init__(self, scale_dims: List[int], num_classes: int, p_mean: float = 3.0) -> None:
        super().__init__()
        self.scale_dims = scale_dims
        self.num_classes = num_classes
        self.p_mean = p_mean
        pooled_dim = sum(dim * 3 for dim in scale_dims)
        self.pre_mix = PRButterfly(pooled_dim)
        hidden = max(num_classes * 2, pooled_dim)
        self.head = nn.Sequential(
            RMSNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=(2, 3))
        max_val = torch.amax(x, dim=(2, 3))
        p = torch.clamp(x.abs(), min=1e-6).pow(self.p_mean)
        p_mean = p.mean(dim=(2, 3)).pow(1.0 / self.p_mean)
        return torch.cat([mean, max_val, p_mean], dim=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        pooled = [self._pool(feat) for feat in features]
        concat = torch.cat(pooled, dim=1)
        mixed = self.pre_mix(concat.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        return self.head(mixed)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


class ScaleProcessor(nn.Module):
    def __init__(
        self,
        scale_idx: int,
        in_channels: int,
        out_channels: int,
        orientations: int,
        config: OperaVConfig,
        lifting_level,
    ) -> None:
        super().__init__()
        self.scale_idx = scale_idx
        self.out_channels = out_channels
        self.pre_norm = RMSNorm(in_channels)
        self.compactor = OrthonormalCompactor(in_channels, out_channels, dropout_ratio=config.dropout_channels)
        self.post_norm = RMSNorm(out_channels)
        self.block_channels = out_channels // 2

        def restrict_fn(t: torch.Tensor) -> torch.Tensor:
            return lifting_level.restrict(t)

        def prolong_fn(t: torch.Tensor) -> torch.Tensor:
            return lifting_level.prolongate_lowpass(t)

        def make_block() -> MonotoneImplicitBlock:
            router = ProximalSparseRouter(self.block_channels, lambda_init=config.router_lambda_init)
            pwdm = PassiveWaveDigitalMixer(
                self.block_channels,
                orientation_channels=orientations,
                modulation_dim=config.codebook_dim,
                gain_cap=config.max_diagonal_gain,
            )
            butterfly = PRButterfly(
                self.block_channels,
                diagonal_cap=config.max_diagonal_gain,
                modulation_dim=config.codebook_dim,
            )
            preconditioner = MultigridPreconditioner(
                self.block_channels,
                restrict_fn=restrict_fn,
                prolong_fn=prolong_fn,
                tau=config.tau,
                levels=config.multigrid_levels,
                smoothing_steps=config.multigrid_smoothing_steps,
                damping=config.multigrid_damping,
            )
            return MonotoneImplicitBlock(
                self.block_channels,
                router=router,
                pwm_digital=pwdm,
                butterfly=butterfly,
                preconditioner=preconditioner,
                tau=config.tau,
                eta=config.eta,
                rezero_init=config.rezero_init,
                cheap_gain=config.cheap_path_gain,
            )

        self.reversible = ReversibleMacroBlock(
            out_channels,
            block_factory=make_block,
            depth=config.macro_blocks_per_scale[scale_idx],
        )

    def forward(self, detail: torch.Tensor, orientation: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(detail)
        x = self.compactor(x)
        x = self.post_norm(x)
        return self.reversible(x, steer=orientation, modulation=latent)


class OperaV(nn.Module):
    def __init__(self, config: OperaVConfig) -> None:
        super().__init__()
        self.config = config
        lowpass_channels = list(config.channels_per_scale)
        self.pslf = ParaunitarySteerableLifting(
            config.image_channels, lowpass_channels, list(config.orientations_per_scale)
        )

        self.scales = nn.ModuleList()
        for idx, (out_c, orient) in enumerate(zip(config.channels_per_scale, config.orientations_per_scale)):
            detail_c = out_c * orient
            level = self.pslf.levels[idx]
            self.scales.append(ScaleProcessor(idx, detail_c, out_c, orient, config, level))

        stats_dim = 2 * max(config.channels_per_scale)
        self.modulator = KYPCodebookModulator(
            num_scales=config.scales,
            stats_dim=stats_dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
        )
        self.readout = ReadoutHead(list(config.channels_per_scale), config.num_classes, p_mean=config.p_readout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        details, orientations, _ = self.pslf.analysis(x)
        scale_stats = []
        for detail in details:
            mean = torch.mean(detail, dim=(2, 3))
            var = torch.var(detail, dim=(2, 3))
            stats = torch.cat([mean, var], dim=1)
            target_dim = self.modulator.stats_dim
            if stats.shape[1] < target_dim:
                stats = F.pad(stats, (0, target_dim - stats.shape[1]))
            elif stats.shape[1] > target_dim:
                stats = stats[:, :target_dim]
            scale_stats.append(stats)
        latents = self.modulator(scale_stats)

        processed_features = []
        for processor, detail, orientation, latent in zip(self.scales, details, orientations, latents):
            processed = processor(detail, orientation, latent)
            processed_features.append(processed)

        logits = self.readout(processed_features)
        return logits


def build_opera_v(**overrides: Any) -> OperaV:
    config = OperaVConfig(**overrides) if overrides else OperaVConfig()
    return OperaV(config)


# ---------------------------------------------------------------------------
# MNIST training entrypoint
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_mnist_model() -> OperaV:
    return build_opera_v(
        image_channels=1,
        num_classes=10,
        scales=2,
        channels_per_scale=(16, 24),
        orientations_per_scale=(4, 2),
        macro_blocks_per_scale=(1, 1),
        multigrid_levels=0,
        codebook_dim=32,
        codebook_size=8,
    )


def get_mnist_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    max_train: Optional[int],
    max_eval: Optional[int],
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
        ]
    )
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if max_train is not None and max_train < len(train_ds):
        train_ds = Subset(train_ds, list(range(max_train)))
    if max_eval is not None and max_eval < len(test_ds):
        test_ds = Subset(test_ds, list(range(max_eval)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def train_one_epoch(model: OperaV, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: OperaV, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone OPERA-V MNIST trainer.")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store MNIST.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay.")
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"), help="Execution device.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes.")
    parser.add_argument("--max-train", type=int, default=2048, help="Limit training samples for a quick run (None = full).")
    parser.add_argument("--max-eval", type=int, default=512, help="Limit eval samples for a quick run (None = full).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = build_mnist_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train=None if args.max_train <= 0 else args.max_train,
        max_eval=None if args.max_eval <= 0 else args.max_eval,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params:,} trainable parameters.")

    print("Compiling...")
    model = torch.compile(model)
    print("Compiled.")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        eval_loss, eval_acc = evaluate(model, test_loader, device)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
