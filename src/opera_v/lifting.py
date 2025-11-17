from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .utils import normalized_vector, unitary_two_tap


class HouseholderProjection(nn.Module):
    """
    Stack of Householder reflections that implements an orthonormal projection between
    channel dimensions without introducing gain. This is the backbone for the paraunitary
    predict/update filters.
    """

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
    """
    Shared steerable mixing bank using circular harmonics.
    Parameters are shared across scales with per-scale gains, fulfilling the specification.
    """

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

        stack = torch.stack(oriented, dim=2)  # (B, C, O, H, W)
        orientation_field = stack.mean(dim=1)  # (B, O, H, W)
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
    """
    Single paraunitary lifting level (analysis + synthesis) with steerable details.
    """

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
    """
    Multi-level PSLF module that exposes analysis + synthesis for multigrid coupling.
    """

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
