from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import SCHEMA_VERSION


class StorageIntegrityError(RuntimeError):
    pass


class ChunkStore:
    """Single-writer, append-only JSONL + immutable NPZ record store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.payload_dir = self.root / "payloads"
        self.manifest_path = self.root / "manifest.jsonl"
        self.lock_path = self.root / ".writer.lock"
        self.payload_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _writer(self) -> Iterator[None]:
        try:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as error:
            raise RuntimeError(
                f"store already has an active writer lock: {self.lock_path}; "
                "inspect the owning process before explicit lock recovery"
            ) from error
        try:
            os.write(descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(descriptor)
            yield
        finally:
            os.close(descriptor)
            self.lock_path.unlink(missing_ok=True)

    def _safe_payload_path(self, relative: Any) -> Path:
        value = Path(str(relative))
        if value.is_absolute() or ".." in value.parts or value.parts[:1] != ("payloads",):
            raise StorageIntegrityError(f"invalid relative payload path: {relative!r}")
        resolved = (self.root / value).resolve()
        payload_root = self.payload_dir.resolve()
        if resolved.parent != payload_root:
            raise StorageIntegrityError(f"payload escapes store directory: {relative!r}")
        return resolved

    def _read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        raw = self.manifest_path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise StorageIntegrityError("manifest has a partially written final line")
        records: list[dict[str, Any]] = []
        identifiers: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise StorageIntegrityError(f"invalid manifest JSON at line {line_number}") from error
            if record.get("schema_version") != SCHEMA_VERSION:
                raise StorageIntegrityError(
                    f"incompatible schema at line {line_number}: "
                    f"expected {SCHEMA_VERSION}, got {record.get('schema_version')!r}"
                )
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in identifiers:
                raise StorageIntegrityError(f"missing or duplicate sample_id at line {line_number}")
            identifiers.add(sample_id)
            self._safe_payload_path(record.get("payload"))
            records.append(record)
        return records

    def append(self, sample: Any) -> Path:
        metadata = dict(sample.metadata)
        arrays = dict(sample.arrays)
        sample_id = str(metadata["sample_id"])
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"sample schema must be version {SCHEMA_VERSION}")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id) is None:
            raise ValueError("sample_id must contain only letters, digits, '.', '_', or '-'")
        target = self.payload_dir / f"{sample_id}.npz"
        with self._writer():
            records = self._read_manifest()
            if any(record["sample_id"] == sample_id for record in records):
                raise ValueError(f"duplicate sample_id: {sample_id}")
            if target.exists():
                raise FileExistsError(target)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{sample_id}-", suffix=".npz", dir=self.payload_dir
            )
            os.close(descriptor)
            temp_path = Path(temp_name)
            try:
                np.savez_compressed(temp_path, **arrays)
                with temp_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temp_path, target)
                directory_descriptor = os.open(self.payload_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
                metadata["payload"] = str(target.relative_to(self.root))
                encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode()
                manifest_descriptor = os.open(
                    self.manifest_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
                )
                try:
                    written = os.write(manifest_descriptor, encoded)
                    if written != len(encoded):
                        raise OSError("short manifest append")
                    os.fsync(manifest_descriptor)
                finally:
                    os.close(manifest_descriptor)
            finally:
                temp_path.unlink(missing_ok=True)
        return target

    def records(self, **filters: Any) -> Iterator[dict[str, Any]]:
        for record in self._read_manifest():
            path = self._safe_payload_path(record["payload"])
            if not path.is_file():
                raise StorageIntegrityError(f"manifest payload is missing: {path}")
            if all(record.get(key) == value for key, value in filters.items()):
                yield record

    def load_arrays(self, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = self._safe_payload_path(record["payload"])
        try:
            with np.load(path, allow_pickle=False) as payload:
                return {key: payload[key] for key in payload.files}
        except (OSError, ValueError) as error:
            raise StorageIntegrityError(f"invalid NPZ payload: {path}") from error

    def audit(self) -> dict[str, list[str]]:
        """Inspect corruption without mutating evidence."""

        issues: dict[str, list[str]] = {
            "manifest": [],
            "missing_payloads": [],
            "orphan_payloads": [],
            "temporary_payloads": [],
        }
        try:
            records = self._read_manifest()
        except StorageIntegrityError as error:
            issues["manifest"].append(str(error))
            records = []
        referenced: set[Path] = set()
        for record in records:
            path = self._safe_payload_path(record["payload"])
            referenced.add(path)
            if not path.is_file():
                issues["missing_payloads"].append(str(path))
                continue
            try:
                self.load_arrays(record)
            except StorageIntegrityError as error:
                issues["manifest"].append(str(error))
        for path in sorted(self.payload_dir.glob("*.npz")):
            if path.resolve() not in referenced:
                issues["orphan_payloads"].append(str(path))
        issues["temporary_payloads"] = [str(path) for path in sorted(self.payload_dir.glob(".*.npz"))]
        return issues

    def recover(self, *, remove_orphans: bool = False, remove_stale_lock: bool = False) -> dict[str, Any]:
        """Explicitly repair only trailing JSON and optionally remove named orphans.

        The caller must first establish that no writer is alive. Recovery never
        invents a manifest record for an orphan payload.
        """

        if self.lock_path.exists() and not remove_stale_lock:
            raise RuntimeError("writer lock exists; pass remove_stale_lock only after checking its PID")
        if remove_stale_lock:
            self.lock_path.unlink(missing_ok=True)
        recovered: dict[str, Any] = {"truncated_manifest_bytes": 0, "removed_orphans": []}
        if self.manifest_path.exists():
            raw = self.manifest_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                complete = raw[: raw.rfind(b"\n") + 1] if b"\n" in raw else b""
                recovered["truncated_manifest_bytes"] = len(raw) - len(complete)
                temporary = self.manifest_path.with_suffix(".jsonl.recovering")
                temporary.write_bytes(complete)
                os.replace(temporary, self.manifest_path)
        if remove_orphans:
            issues = self.audit()
            if issues["manifest"]:
                raise StorageIntegrityError("manifest remains invalid after trailing-line recovery")
            for name in (*issues["orphan_payloads"], *issues["temporary_payloads"]):
                path = Path(name)
                path.unlink()
                recovered["removed_orphans"].append(str(path))
        return recovered

    def __len__(self) -> int:
        return sum(1 for _ in self.records())


__all__ = ["ChunkStore", "StorageIntegrityError"]
