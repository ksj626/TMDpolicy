from __future__ import annotations

import argparse
import json
import shlex
import traceback
from pathlib import Path
from typing import Any

from tmd_policy.common.config import load_research_config
from tmd_policy.common.provenance import capture_run_provenance
from tmd_policy.common.tasks import TaskRegistry, inspect_cached_libero

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {
    "flow_sft": {"expert_action_chunks", "flow_velocity"},
    "tmd_stage1": {"expert_action_chunks", "flow_velocity"},
    "tmd_stage2": {"expert_action_chunks", "flow_score", "teacher_at_student_action"},
    "tmd_plain_gaussian_ablation": {"expert_action_chunks", "flow_velocity"},
    "dmd2_flow": {"expert_action_chunks", "flow_score", "teacher_at_student_action"},
    "opd_categorical": {"on_policy_rollouts", "token_log_probability"},
    "continuous_flow_opd": {"on_policy_rollouts", "exact_log_density", "teacher_at_student_action"},
    "occupancy_discriminator": {"path_windows", "on_policy_rollouts"},
    "occupancy_tmd": {"path_windows", "on_policy_rollouts", "flow_score"},
    "data": set(),
    "evaluation": set(),
}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    config = load_research_config(args.config)
    value = config.value
    registry_path = (PROJECT_ROOT / value["tasks"]["registry"]).resolve()
    mappings: list[dict[str, Any]] = []
    registry_error = None
    try:
        mappings = [task.to_dict() for task in TaskRegistry.from_json(registry_path).tasks]
    except (OSError, ValueError, KeyError) as error:
        registry_error = str(error)
    supplied, required = set(value.get("capabilities", [])), REQUIRED[config.method]
    missing = sorted(required - supplied)
    if registry_error:
        missing.append("unambiguous_task_registry")
    output = (PROJECT_ROOT / value["output"]["directory"]).resolve()
    exact = [
        "conda", "run", "-n", "lerobot", "python", "-m", "tmd_policy.research_cli",
        args.operation, "--config", str(config.path), "--execute",
    ]
    if args.resume:
        exact.extend(("--resume", args.resume))
    notes = []
    if config.method == "continuous_flow_opd" and "exact_log_density" not in supplied:
        notes.append("pi0.5 public API has no supported log_prob or invertible vector-field API")
    if config.method in {"tmd_stage2", "dmd2_flow", "occupancy_tmd"} and "flow_score" not in supplied:
        notes.append("pinned pi0.5 has no supported score-at-student-action API")
    report = {
        "dry_run": not args.execute,
        "operation": args.operation,
        "method": config.method,
        "classification": value.get("classification"),
        "executable": not missing,
        "required_capabilities": sorted(required),
        "configured_capabilities": sorted(supplied),
        "missing_capabilities": missing,
        "resolved_config": value,
        "dataset_selection": value["dataset"],
        "checkpoint_revisions": value["models"],
        "task_registry": str(registry_path),
        "task_mappings": mappings,
        "task_uids": [item["canonical_task_uid"] for item in mappings],
        "expected_output_directory": str(output),
        "resource_estimate": value["resources"],
        "resume": args.resume,
        "exact_real_command": shlex.join(exact),
        "notes": notes + ([f"task registry error: {registry_error}"] if registry_error else []),
    }
    if args.operation == "inspect-libero-tasks":
        report["cached_task_inspection"] = inspect_cached_libero(
            PROJECT_ROOT / value["dataset"]["cache"], registry_path
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TMDpolicy research operation preflight")
    parser.add_argument("operation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        return 0 if report["executable"] else 2
    if not report["executable"]:
        raise RuntimeError("operation failed closed: " + ", ".join(report["missing_capabilities"]))
    output = Path(report["expected_output_directory"])
    if output.exists() and args.resume is None:
        raise FileExistsError(f"refusing output overwrite: {output}")
    if args.resume is not None and not Path(args.resume).is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
    if args.resume is not None:
        attempt_root = output / "resume_attempts"
        attempt_index = 0
        while (attempt_root / f"attempt-{attempt_index:04d}").exists():
            attempt_index += 1
        output = attempt_root / f"attempt-{attempt_index:04d}"
        report["actual_resume_output_directory"] = str(output)
    output.mkdir(parents=True, exist_ok=args.resume is not None)
    config = report["resolved_config"]
    from yaml import safe_dump

    (output / "resolved_config.yaml").write_text(safe_dump(config, sort_keys=True), encoding="utf-8")
    registry = TaskRegistry.from_json(report["task_registry"])
    provenance = capture_run_provenance(
        repository=PROJECT_ROOT,
        dependencies=(PROJECT_ROOT.parent / "lerobot",),
        resolved_config=config,
        seeds={"training": int(config["training"]["seed"])},
        task_registry=registry.to_dict(),
        model_revisions=config["models"]["revisions"],
        processor_revisions={
            key: value for key, value in config["models"]["revisions"].items() if "processor" in key
        },
        dataset_revisions={config["dataset"]["id"]: config["dataset"]["revision"]},
    )
    provenance.write(output / "provenance")
    config["_provenance_summary"] = {
        "repository": {
            "commit": provenance.repository.commit,
            "dirty": provenance.repository.dirty,
            "patch_sha256": provenance.repository.patch_sha256,
        },
        "dependencies": [
            {"path": item.path, "commit": item.commit, "dirty": item.dirty, "patch_sha256": item.patch_sha256}
            for item in provenance.dependencies
        ],
        "task_registry_schema": provenance.task_registry.get("schema_version"),
    }
    try:
        if args.operation == "build-libero-expert":
            from tmd_policy.common.data.materialize import materialize_libero_expert

            result = materialize_libero_expert(
                config=config, registry=registry,
                output=PROJECT_ROOT / config["dataset"]["expert_store"],
            )
        elif args.operation == "audit-dataset":
            from tmd_policy.common.data import ResearchStore

            result = {
                name: ResearchStore(PROJECT_ROOT / config["dataset"][name]).audit()
                for name in ("expert_store", "rollout_store")
            }
        elif args.operation == "train-flow-sft":
            from tmd_policy.operations import train_flow_sft

            config["dataset"]["expert_store"] = str(PROJECT_ROOT / config["dataset"]["expert_store"])
            result = train_flow_sft(config, output=output, resume=args.resume)
        else:
            raise RuntimeError(
                f"{args.operation} has a complete mathematical component but no safe end-to-end adapter yet; "
                "the operation is recorded as a failed attempt, not silently replaced"
            )
        (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        failure = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        (output / "failed_attempt.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
