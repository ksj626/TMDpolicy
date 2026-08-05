"""Concrete retained research methods with cycle-safe lazy public exports.

Backends depend on leaf utilities such as :mod:`flow_objectives`.  Importing a
leaf must not eagerly construct every method package, because DMD/TMD programs
in turn import those backends.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dmd2_flow import DMD2FlowProgram
    from .flow_sft import FlowSFTProgram
    from .occupancy_tmd import OccupancyDiscriminatorProgram, OccupancyWeightedTMDProgram
    from .tmd import TMDStage1Program, TMDStage2Program

__all__ = [
    "DMD2FlowProgram",
    "FlowSFTProgram",
    "OccupancyDiscriminatorProgram",
    "OccupancyWeightedTMDProgram",
    "TMDStage1Program",
    "TMDStage2Program",
]

_EXPORT_MODULES = {
    "DMD2FlowProgram": ".dmd2_flow",
    "FlowSFTProgram": ".flow_sft",
    "OccupancyDiscriminatorProgram": ".occupancy_tmd",
    "OccupancyWeightedTMDProgram": ".occupancy_tmd",
    "TMDStage1Program": ".tmd",
    "TMDStage2Program": ".tmd",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
