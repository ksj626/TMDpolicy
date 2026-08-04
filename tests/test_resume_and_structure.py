from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from tmd_policy.training.engine import TrainingProgram, run_training


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 16

    def __getitem__(self, index: int):
        x = torch.tensor([float(index % 5)])
        return {"x": x, "y": 3 * x - 1}


class _Program(TrainingProgram):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def phase_schedule(self):
        return ("generator",)

    def loss(self, batch, phase):
        prediction = self.linear(batch["x"])
        loss = (prediction - batch["y"]).square().mean()
        return loss, {}

    def make_optimizers(self, training):
        return {"generator": torch.optim.AdamW(self.parameters(), lr=training["learning_rate"])}


def _config(manifest: Path) -> dict:
    return {
        "method": "unit_resume",
        "backend": {"lerobot_version": "0.6.1", "expected_source_hashes": None},
        "models": {
            "student": {"id": "unit", "revision": "a" * 40, "processor_revision": "a" * 40},
            "teacher": {"id": "unit", "revision": "b" * 40, "processor_revision": "b" * 40},
        },
        "dataset": {"id": "lerobot/libero", "revision": "c" * 40, "manifest": str(manifest)},
        "training": {
            "seed": 5,
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "mixed_precision": "no",
            "gradient_accumulation": 1,
            "gradient_clip_norm": 10.0,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "minimum_lr_scale": 1.0,
            "max_steps": 4,
            "validation_interval": 2,
            "validation_batches": 1,
            "checkpoint_interval": 2,
        },
    }


def test_uninterrupted_and_resumed_training_match(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"unit": True}), encoding="utf-8")
    config = _config(manifest)
    data = _Dataset()
    uninterrupted = _Program()
    run_training(uninterrupted, data, data, config=config, output_dir=tmp_path / "full")
    checkpoint = tmp_path / "full" / "checkpoints" / "step-00000002.pt"
    resumed = _Program()
    run_training(
        resumed,
        data,
        data,
        config=config,
        output_dir=tmp_path / "resumed",
        resume=checkpoint,
    )
    full_state = torch.load(tmp_path / "full" / "checkpoints" / "final.pt", weights_only=False)["program"]
    resumed_state = torch.load(tmp_path / "resumed" / "checkpoints" / "final.pt", weights_only=False)["program"]
    assert full_state.keys() == resumed_state.keys()
    assert all(torch.equal(full_state[key], resumed_state[key]) for key in full_state)


def test_removed_paths_and_training_scripts_are_concrete() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("*opd*"))
    assert not (root / "src/tmd_policy/research_cli.py").exists()
    assert not (root / "src/tmd_policy/common/density/cnf.py").exists()
    scripts = list((root / "scripts/train").glob("*.sh"))
    required = {
        "train_dmd2_flow_paper.sh",
        "train_tmd_stage1.sh",
        "train_tmd_stage2_paper.sh",
        "run_tmd_pipeline.sh",
    }
    assert required <= {script.name for script in scripts}
    for script in scripts:
        text = script.read_text(encoding="utf-8").lower()
        assert "tmd-policy train" in text
        assert "dry-run" not in text and "--execute" not in text
