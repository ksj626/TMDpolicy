from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .store import ResearchStore


class ManifestDataset(Dataset[dict[str, Any]]):
    """Loads all records in one explicit split; never repeats one selected row."""

    def __init__(self, root: str | Path, *, split: str) -> None:
        self.store = ResearchStore(root)
        self.rows = list(self.store.records(split=split))
        if not self.rows:
            raise ValueError(f"no {split!r} records in {root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"metadata": row, "arrays": self.store.load_arrays(row)}


def collate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty record list")
    keys = set(records[0]["arrays"])
    if any(set(record["arrays"]) != keys for record in records):
        raise ValueError("batch records have inconsistent array keys")
    arrays = {
        key: torch.from_numpy(np.stack([record["arrays"][key] for record in records]))
        for key in sorted(keys)
    }
    return {"metadata": [record["metadata"] for record in records], "arrays": arrays}


def make_dataloader(
    root: str | Path,
    *,
    split: str,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ManifestDataset(root, split=split),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_records,
        drop_last=False,
    )


__all__ = ["ManifestDataset", "collate_records", "make_dataloader"]
