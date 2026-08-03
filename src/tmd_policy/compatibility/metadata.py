from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CompatibilityReport:
    dataset_id: str
    dataset_revision: str
    student_id: str
    student_revision: str
    teacher_id: str
    teacher_revision: str
    prediction_horizon: int
    action_dim: int
    canonical_state_dim: int
    student_declared_state_dim: int
    student_state_dim: int
    teacher_state_dim: int
    image_keys: tuple[str, ...]
    compatible: bool
    adaptations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    with urllib.request.urlopen(str(source), timeout=30) as response:
        return json.load(response)


def inspect_compatibility(
    dataset_info: str | Path,
    student_config: str | Path,
    teacher_config: str | Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    student_id: str,
    student_revision: str,
    teacher_id: str,
    teacher_revision: str,
    student_effective_state_dim: int | None = None,
) -> CompatibilityReport:
    dataset = _read_json(dataset_info)
    student = _read_json(student_config)
    teacher = _read_json(teacher_config)
    dataset_features = dataset["features"]
    action_dim = int(dataset_features["action"]["shape"][0])
    state_dim = int(dataset_features["observation.state"]["shape"][0])
    student_action = int(student["output_features"]["action"]["shape"][0])
    teacher_action = int(teacher["output_features"]["action"]["shape"][0])
    student_declared_state = int(student["input_features"]["observation.state"]["shape"][0])
    student_state = student_effective_state_dim or student_declared_state
    teacher_state = int(teacher["input_features"]["observation.state"]["shape"][0])
    prediction = int(student["chunk_size"])
    compatible = (
        action_dim == student_action == teacher_action
        and prediction == int(teacher["chunk_size"])
        and state_dim == teacher_state
        and student_state <= state_dim
    )
    adaptations: list[str] = []
    if student_declared_state != student_state:
        adaptations.append(
            f"student config declares {student_declared_state}D but its official normalizer is "
            f"{student_state}D; keep the canonical state"
        )
    elif student_state != state_dim:
        adaptations.append(f"student state projection: canonical {state_dim}D -> first {student_state}D")
    student_images = set(student["input_features"]) - {"observation.state"}
    dataset_images = tuple(sorted(key for key in dataset_features if key.startswith("observation.images.")))
    if not set(dataset_images).issubset(student_images):
        adaptations.append("apply the SmolVLA checkpoint's official camera rename map")
    return CompatibilityReport(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        student_id=student_id,
        student_revision=student_revision,
        teacher_id=teacher_id,
        teacher_revision=teacher_revision,
        prediction_horizon=prediction,
        action_dim=action_dim,
        canonical_state_dim=state_dim,
        student_declared_state_dim=student_declared_state,
        student_state_dim=student_state,
        teacher_state_dim=teacher_state,
        image_keys=dataset_images,
        compatible=compatible,
        adaptations=tuple(adaptations),
    )
