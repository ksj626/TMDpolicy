"""Lightweight LIBERO-Plus installation check; does not construct MuJoCo environments."""

from __future__ import annotations

import inspect
import contextlib
import io
import json
from pathlib import Path

from wand.api import library as _wand_library  # noqa: F401
from lerobot.envs.configs import LiberoPlusEnv
from libero.libero import benchmark


EXPECTED = {
    "libero_spatial": 2_402,
    "libero_object": 2_518,
    "libero_goal": 2_591,
    "libero_10": 2_519,
}


def main() -> None:
    classification_path = Path(inspect.getfile(benchmark)).resolve().with_name(
        "task_classification.json"
    )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    mapping = benchmark.get_benchmark_dict()
    for suite, count in EXPECTED.items():
        with contextlib.redirect_stdout(io.StringIO()):
            instance = mapping[suite]()
        assert len(instance.tasks) == count
        assert len(classification[suite]) == count
    assert sum(EXPECTED.values()) == 10_030
    assert LiberoPlusEnv().is_libero_plus
    print(
        json.dumps(
            {
                "libero_plus_source": str(classification_path.parent),
                "suite_counts": EXPECTED,
                "total_tasks": 10_030,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
