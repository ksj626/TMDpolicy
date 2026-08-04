from __future__ import annotations

from typing import Any

from .base import ResearchMethod


def build_method(name: str, **kwargs: Any) -> ResearchMethod:
    if name == "flow_sft":
        from .flow_sft.method import FlowSFTMethod

        return FlowSFTMethod(**kwargs)
    if name in {"tmd", "tmd_stage1"}:
        from .tmd.method import TMDMethod

        return TMDMethod(**kwargs)
    if name == "tmd_stage2":
        from .tmd.method import TMDStage2Method

        return TMDStage2Method(**kwargs)
    if name == "dmd2_flow":
        from .dmd2_flow.method import DMD2FlowMethod

        return DMD2FlowMethod(**kwargs)
    if name in {"opd_categorical", "continuous_flow_opd"}:
        from .opd_on_policy.method import OPDMethod

        return OPDMethod(mode=name, **kwargs)
    if name == "occupancy_tmd":
        from .occupancy_tmd.method import OccupancyTMDMethod

        return OccupancyTMDMethod(**kwargs)
    if name == "occupancy_discriminator":
        from .occupancy_tmd.method import OccupancyDiscriminatorMethod

        return OccupancyDiscriminatorMethod(**kwargs)
    if name == "tmd_plain_gaussian_ablation":
        raise RuntimeError(
            "the legacy ablation remains under tmd_policy.models and is intentionally not a full research method"
        )
    raise KeyError(f"unknown method: {name}")
