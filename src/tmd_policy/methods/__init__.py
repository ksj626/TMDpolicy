"""DMD2 objectives and implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["DMD2FlowProgram"]


def __getattr__(name: str) -> Any:
    if name != "DMD2FlowProgram":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".dmd2_flow", __name__), name)
    globals()[name] = value
    return value
