from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from .butterfly import PRButterfly
from .compaction import OrthonormalCompactor
from .config import OperaVConfig
from .implicit_block import MonotoneImplicitBlock
from .lifting import ParaunitarySteerableLifting
from .modulator import KYPCodebookModulator
from .multigrid import MultigridPreconditioner
from .norms import RMSNorm
from .pwdm import PassiveWaveDigitalMixer
from .readout import ReadoutHead
from .reversible import ReversibleMacroBlock
from .router import ProximalSparseRouter


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
    """
    Reference OPERA-V implementation faithful to the architecture specification.
    """

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
        details, orientations, lowpass = self.pslf.analysis(x)
        # Prepare stats for modulation (mean/variance per scale).
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
        for idx, (processor, detail, orientation, latent) in enumerate(
            zip(self.scales, details, orientations, latents)
        ):
            processed = processor(detail, orientation, latent)
            processed_features.append(processed)

        logits = self.readout(processed_features)
        return logits


def build_opera_v(**overrides) -> OperaV:
    config = OperaVConfig(**overrides) if overrides else OperaVConfig()
    return OperaV(config)
