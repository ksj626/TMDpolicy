from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tmd_policy.common.tasks import TaskIdentity

from .records import ResearchRecord
from .splits import assert_episode_disjoint


class ResearchStoreError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_hash(row: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    reserved = {
        "schema_version", "kind", "sample_id", "canonical_task_uid", "task_identity",
        "episode_index", "frame_index", "split", "content_hash", "payload", "payload_sha256",
    }
    stable = {
        "schema_version": 3,
        "kind": row["kind"],
        "sample_id": row["sample_id"],
        "task_identity": row["task_identity"],
        "episode_index": row["episode_index"],
        "frame_index": row["frame_index"],
        "split": row["split"],
        "metadata_extra": {key: value for key, value in row.items() if key not in reserved},
    }
    digest = hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode())
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class ResearchStore:
    """Append-only schema-v3 NPZ store with semantic and byte hashes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.payloads = self.root / "payloads"
        self.manifest = self.root / "manifest.jsonl"
        self.lock = self.root / ".writer.lock"
        self.payloads.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict[str, Any]]:
        if not self.manifest.exists():
            return []
        raw = self.manifest.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ResearchStoreError("manifest has a partial final line")
        rows = [json.loads(line) for line in raw.splitlines()]
        ids = [row.get("sample_id") for row in rows]
        if len(ids) != len(set(ids)) or None in ids:
            raise ResearchStoreError("manifest contains missing or duplicate sample IDs")
        return rows

    def _path(self, row: Mapping[str, Any]) -> Path:
        relative = Path(str(row.get("payload", "")))
        if relative.is_absolute() or relative.parts[:1] != ("payloads",) or ".." in relative.parts:
            raise ResearchStoreError(f"unsafe payload path: {relative}")
        path = (self.root / relative).resolve()
        if path.parent != self.payloads.resolve():
            raise ResearchStoreError(f"payload escapes store: {relative}")
        return path

    def _append_unlocked(self, record: ResearchRecord) -> Path:
        rows = self._rows()
        if any(row["sample_id"] == record.sample_id for row in rows):
            raise ValueError(f"duplicate sample ID: {record.sample_id}")
        target = self.payloads / f"{record.sample_id}.npz"
        if target.exists():
            raise FileExistsError(target)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{record.sample_id}-", suffix=".npz", dir=self.payloads
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **record.arrays)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory_descriptor = os.open(self.payloads, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        row = {
            **record.metadata,
            "payload": str(target.relative_to(self.root)),
            "payload_sha256": file_sha256(target),
        }
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return target

    def append(self, record: ResearchRecord) -> Path:
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise ResearchStoreError(f"research store already has a writer: {self.lock}") from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            return self._append_unlocked(record)
        finally:
            os.close(descriptor)
            self.lock.unlink(missing_ok=True)

    def records(self, **filters: Any) -> Iterator[dict[str, Any]]:
        for row in self._rows():
            if all(row.get(key) == value for key, value in filters.items()):
                yield row

    def load_arrays(self, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = self._path(row)
        if file_sha256(path) != row.get("payload_sha256"):
            raise ResearchStoreError(f"payload byte hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}

    def audit(self) -> dict[str, Any]:
        issues: list[str] = []
        rows = self._rows()
        referenced = set()
        for row in rows:
            path = self._path(row)
            referenced.add(path)
            if not path.is_file():
                issues.append(f"missing payload: {path}")
                continue
            arrays = None
            try:
                arrays = self.load_arrays(row)
            except (OSError, ValueError, ResearchStoreError) as error:
                issues.append(str(error))
            if not row.get("content_hash"):
                issues.append(f"missing semantic content hash: {row.get('sample_id')}")
            elif arrays is not None and semantic_hash(row, arrays) != row["content_hash"]:
                issues.append(f"semantic content hash mismatch: {row.get('sample_id')}")
        orphans = [str(path) for path in self.payloads.glob("*.npz") if path.resolve() not in referenced]
        for row in rows:
            try:
                identity = TaskIdentity.from_dict(row["task_identity"])
                if identity.canonical_task_uid != row["canonical_task_uid"]:
                    raise ValueError("manifest canonical UID differs from task identity")
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"invalid task identity for {row.get('sample_id')}: {error}")
        try:
            assert_episode_disjoint(
                (int(row["episode_index"]), int(row["frame_index"]), str(row["split"]))
                for row in rows
                if row.get("split") in {"train", "validation", "test"}
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(str(error))
        return {"records": len(rows), "issues": issues, "orphan_payloads": sorted(orphans)}


__all__ = ["ResearchStore", "ResearchStoreError", "file_sha256", "semantic_hash"]
