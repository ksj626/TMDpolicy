"""Asynchronous, task-balanced student-state replay for DMD2 training."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import torch

from tmd_policy.config import save_resolved_config
from tmd_policy.libero_protocol import (
    LIBERO_SUITE_MAX_EPISODE_STEPS,
    init_state_index_for_trial,
)
from tmd_policy.rollout import RolloutStore


_ALL40 = [
    {"suite": suite, "task_ids": list(range(10))}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
]


class BalancedStudentReplay:
    """Bounded per-task queues sampled round-robin across available tasks."""

    def __init__(self, capacity_replans: int, *, seed: int) -> None:
        if capacity_replans < 40:
            raise ValueError("student replay capacity must hold at least one replan per LIBERO task")
        self.capacity_per_task = max(1, capacity_replans // 40)
        self._records: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.capacity_per_task)
        )
        self._rng = random.Random(seed)
        self._task_cursor = 0
        self.rounds: list[str] = []

    @property
    def size(self) -> int:
        return sum(len(values) for values in self._records.values())

    @property
    def task_support(self) -> list[int]:
        return sorted(task for task, values in self._records.items() if values)

    def ingest(self, root: str | Path) -> int:
        store = RolloutStore(root)
        added = 0
        for row in store.records():
            if row.get("split") != "train":
                continue
            task = int(row["global_task_index"])
            for record in store.load_replans(row):
                self._records[task].append(record)
                added += 1
        resolved = str(Path(root).resolve())
        if resolved not in self.rounds:
            self.rounds.append(resolved)
        return added

    @staticmethod
    def _without_singleton_batch(value: torch.Tensor) -> torch.Tensor:
        return value[0] if value.ndim >= 1 and value.shape[0] == 1 else value

    def sample_like(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        tasks = self.task_support
        if not tasks:
            return None
        batch_size = int(torch.as_tensor(batch["observation.state"]).shape[0])
        selected: list[dict[str, Any]] = []
        for offset in range(batch_size):
            task = tasks[(self._task_cursor + offset) % len(tasks)]
            records = self._records[task]
            selected.append(records[self._rng.randrange(len(records))])
        self._task_cursor = (self._task_cursor + batch_size) % len(tasks)

        image_keys = set(selected[0]["observations"])
        if any(set(record["observations"]) != image_keys for record in selected):
            raise RuntimeError("student replay camera keys differ across tasks")
        replay_batch: dict[str, Any] = {
            "observation.state": torch.stack(
                [torch.as_tensor(record["state"]).float() for record in selected]
            ),
            "task": [str(record["instruction"]) for record in selected],
            "task_index": torch.tensor(
                [int(record["global_task_index"]) for record in selected], dtype=torch.long
            ),
            "action_is_pad": torch.zeros(batch_size, 50, dtype=torch.bool),
            "_student_replay_batch": True,
        }
        for key in sorted(image_keys):
            replay_batch[key] = torch.stack(
                [
                    self._without_singleton_batch(torch.as_tensor(record["observations"][key]))
                    for record in selected
                ]
            )
        return replay_batch

    def metrics(self) -> dict[str, Any]:
        return {
            "replay/size": self.size,
            "replay/task_support": len(self.task_support),
            "replay/rounds_ingested": len(self.rounds),
        }


class AsyncStudentRolloutManager:
    """Collect snapshots out-of-process and atomically promote completed rounds."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        output: Path,
        initial_round: str | Path | None = None,
    ) -> None:
        settings = config["dmd2"]["student_rollout_replay"]
        self.config = config
        self.settings = settings
        self.root = output / "student_rollout_replay"
        self.root.mkdir(parents=True, exist_ok=True)
        self.replay = BalancedStudentReplay(
            int(settings["capacity_replans"]), seed=int(config["training"]["seed"])
        )
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._active_output: Path | None = None
        self._last_launch_update = -1
        self._round = 0
        self._last_status = "idle"
        self._bootstrap_source: str | None = None
        if initial_round is not None:
            self._ingest_bootstrap(Path(initial_round).expanduser().resolve())
        self._load_completed_rounds()

    def _ingest_bootstrap(self, path: Path) -> None:
        if not path.is_dir() or not (path / "collection_report.json").is_file():
            raise FileNotFoundError(
                f"initial student rollout replay is not a completed round: {path}"
            )
        added = self.replay.ingest(path)
        if added == 0:
            raise RuntimeError(f"initial student rollout replay contains no train replans: {path}")
        if len(self.replay.task_support) != 40:
            raise RuntimeError(
                "initial student rollout replay is not task-balanced over all 40 LIBERO tasks: "
                f"{self.replay.task_support}"
            )
        self._bootstrap_source = str(path)
        # Match a freshly collected blocking round: the next refresh is due at
        # 500 actual generator updates, not immediately before optimization.
        self._last_launch_update = 0
        self._last_status = "ready:bootstrap"
        print(
            f"[student replay] bootstrapped {added} replans from {path}; "
            "skipping blocking pre-training rollout",
            flush=True,
        )

    def _load_completed_rounds(self) -> None:
        for path in sorted(self.root.glob("round-*")):
            try:
                self._round = max(self._round, int(path.name.split("-")[-1]) + 1)
            except ValueError:
                continue
            if (path / "collection_report.json").exists():
                self.replay.ingest(path)

    def _rollout_config(self, checkpoint: Path, output: Path) -> dict[str, Any]:
        round_index = self._round
        simulator_seed = int(self.settings.get("base_reset_seed", 100_000)) + round_index
        init_state_index = init_state_index_for_trial(round_index)
        validation_videos = {
            "enabled": False,
            "task_ids": [],
            "simulator_seed": 200_000,
            "init_state_index": 49,
            "camera": "image",
            **dict(self.settings.get("validation_videos", {})),
        }
        devices = [
            str(value)
            for value in self.settings.get("devices", [self.settings["device"]])
        ]
        return {
            "method": "collect_student",
            "classification": "DMD2 asynchronous balanced on-policy state refresh",
            "backend": self.config["backend"],
            "models": self.config["models"],
            "dataset": self.config["dataset"],
            "horizons": self.config["horizons"],
            "policy": {
                "method": "dmd2_flow",
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": "auto",
                "device": devices[0],
            },
            "collection": {
                "fps": int(self.settings.get("fps", 10)),
                "benchmark": _ALL40,
                "train_reset_seeds": [
                    simulator_seed
                ],
                "train_init_state_indices": [init_state_index],
                "validation_reset_seeds": (
                    [int(validation_videos["simulator_seed"])]
                    if bool(validation_videos["enabled"])
                    else []
                ),
                "validation_init_state_indices": (
                    [int(validation_videos["init_state_index"])]
                    if bool(validation_videos["enabled"])
                    else []
                ),
                "validation_task_ids": (
                    [int(value) for value in validation_videos["task_ids"]]
                    if bool(validation_videos["enabled"])
                    else []
                ),
                "save_validation_videos": bool(validation_videos["enabled"]),
                "video_camera": str(validation_videos["camera"]),
                "suite_max_episode_steps": dict(
                    self.settings.get(
                        "suite_max_episode_steps",
                        LIBERO_SUITE_MAX_EPISODE_STEPS,
                    )
                ),
                "batch_size": 1,
                "devices": devices,
                "collection_round": round_index,
            },
            "output": {"directory": str(output)},
        }

    def launch(self, checkpoint: Path, *, generator_updates: int, blocking: bool) -> None:
        if self.process is not None:
            raise RuntimeError("student rollout refresh is already running")
        round_output = self.root / f"round-{self._round:06d}"
        runtime_config = self.root / f"round-{self._round:06d}.yaml"
        save_resolved_config(self._rollout_config(checkpoint, round_output), runtime_config)
        log_path = self.root / f"round-{self._round:06d}.log"
        # The blocking bootstrap inherits the terminal so the user sees its
        # episode/NFE bars. Later refreshes write to a file and remain truly
        # asynchronous with respect to optimization.
        self._log_handle = None if blocking else log_path.open("a", encoding="utf-8")
        environment = dict(os.environ)
        environment.setdefault("MUJOCO_GL", "egl")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tmd_policy.cli",
                "rollout",
                "collect-student",
                "--config",
                str(runtime_config),
                "--output",
                str(round_output),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT if self._log_handle is not None else None,
            text=True,
        )
        self._active_output = round_output
        self._last_launch_update = generator_updates
        self._last_status = "collecting"
        destination = "streaming to terminal" if blocking else f"log={log_path}"
        print(
            f"[student replay] launched round {self._round} from generator update "
            f"{generator_updates}; {destination}",
            flush=True,
        )
        self._round += 1
        if blocking:
            self.wait()

    def poll(self) -> None:
        if self.process is None or self.process.poll() is None:
            return
        return_code = int(self.process.returncode)
        if self._log_handle is not None:
            self._log_handle.close()
        completed = self._active_output
        self.process = None
        self._log_handle = None
        self._active_output = None
        if return_code != 0 or completed is None:
            self._last_status = f"failed:{return_code}"
            print(
                f"[student replay] collector failed with exit code {return_code}; "
                "training continues and the next refresh may retry",
                flush=True,
            )
            return
        added = self.replay.ingest(completed)
        self._last_status = "ready"
        print(
            f"[student replay] ingested {added} replans from {completed}; "
            f"buffer={self.replay.size}",
            flush=True,
        )

    def wait(self) -> None:
        if self.process is None:
            return
        self.process.wait()
        self.poll()
        if self.replay.size == 0:
            raise RuntimeError("initial student rollout produced no replay records")
        if len(self.replay.task_support) != 40:
            raise RuntimeError(
                "initial student rollout is not task-balanced over all 40 LIBERO tasks: "
                f"{self.replay.task_support}"
            )

    def should_refresh(self, generator_updates: int) -> bool:
        interval = int(self.settings["refresh_generator_updates"])
        return (
            generator_updates > 0
            and generator_updates - self._last_launch_update >= interval
            and self.process is None
        )

    def metrics(self) -> dict[str, Any]:
        return {
            **self.replay.metrics(),
            "replay/collector_active": int(self.process is not None),
            "replay/status": self._last_status,
            "replay/last_launch_generator_update": self._last_launch_update,
            "replay/bootstrap_source": self._bootstrap_source,
        }

    def write_status(self) -> None:
        target = self.root / "status.json"
        temporary = target.with_suffix(".partial")
        temporary.write_text(json.dumps(self.metrics(), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, target)


__all__ = ["AsyncStudentRolloutManager", "BalancedStudentReplay"]
