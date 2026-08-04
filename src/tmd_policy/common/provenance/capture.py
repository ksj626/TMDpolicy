from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


def _run(directory: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=directory,
        check=check,
        text=True,
        capture_output=True,
    )


@dataclass(frozen=True)
class GitProvenance:
    path: str
    commit: str
    dirty: bool
    status: str
    patch_sha256: str
    patch: str


def capture_git(path: str | Path, *, include_patch: bool = True) -> GitProvenance:
    directory = Path(path).resolve()
    commit = _run(directory, "git", "rev-parse", "HEAD").stdout.strip()
    status = _run(directory, "git", "status", "--porcelain=v1", "--untracked-files=all").stdout
    tracked_patch = _run(directory, "git", "diff", "--binary", "HEAD").stdout
    untracked_patches: list[str] = []
    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        relative = line[3:]
        result = _run(directory, "git", "diff", "--no-index", "--binary", "/dev/null", relative, check=False)
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"could not capture untracked dependency file {relative}: {result.stderr}")
        untracked_patches.append(result.stdout)
    patch = tracked_patch + "".join(untracked_patches)
    digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    return GitProvenance(
        path=str(directory),
        commit=commit,
        dirty=bool(status),
        status=status,
        patch_sha256=digest,
        patch=patch if include_patch else "",
    )


@dataclass(frozen=True)
class RunProvenance:
    repository: GitProvenance
    dependencies: tuple[GitProvenance, ...]
    python: str
    packages: dict[str, str]
    torch: str
    cuda_runtime: str | None
    cudnn: int | None
    hostname: str
    gpu: tuple[dict[str, Any], ...]
    command: str
    resolved_config: dict[str, Any]
    seeds: dict[str, int]
    model_revisions: dict[str, str]
    processor_revisions: dict[str, str]
    dataset_revisions: dict[str, str]
    task_registry: dict[str, Any]

    def write(self, output: str | Path) -> Path:
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        for label, git in (("repository", self.repository), *[(f"dependency_{i}", d) for i, d in enumerate(self.dependencies)]):
            if git.dirty:
                patch_path = directory / f"{label}.patch"
                patch_path.write_text(git.patch, encoding="utf-8")
        target = directory / "provenance.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def _package_versions(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def capture_run_provenance(
    *,
    repository: str | Path,
    dependencies: Iterable[str | Path],
    resolved_config: dict[str, Any],
    seeds: dict[str, int],
    task_registry: dict[str, Any],
    model_revisions: dict[str, str],
    processor_revisions: dict[str, str],
    dataset_revisions: dict[str, str],
    command: Iterable[str] | None = None,
) -> RunProvenance:
    gpu = tuple(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    )
    return RunProvenance(
        repository=capture_git(repository),
        dependencies=tuple(capture_git(path) for path in dependencies),
        python=sys.version,
        packages=_package_versions(("numpy", "torch", "transformers", "lerobot", "datasets")),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        hostname=socket.gethostname(),
        gpu=gpu,
        command=shlex.join(command or sys.argv),
        resolved_config=resolved_config,
        seeds=seeds,
        model_revisions=model_revisions,
        processor_revisions=processor_revisions,
        dataset_revisions=dataset_revisions,
        task_registry=task_registry,
    )
