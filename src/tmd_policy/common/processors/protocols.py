from __future__ import annotations

from typing import Any, Protocol


class BatchPreprocessor(Protocol):
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]: ...


class BatchPostprocessor(Protocol):
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]: ...
