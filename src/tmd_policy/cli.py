from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tmd_policy.compatibility.lerobot_api import verify_lerobot_api
from tmd_policy.compatibility.metadata import inspect_compatibility
from tmd_policy.config import config_runtime_report, load_config, save_resolved_config
from tmd_policy.data.expert import (
    build_expert_chunks,
    episode_split_three_way,
    load_lerobot_expert_dataset,
)
from tmd_policy.data.storage import ChunkStore
from tmd_policy.evaluation.policy_runner import evaluate_policy_arm
from tmd_policy.smoke import run_synthetic_smoke
from tmd_policy.training.diagnostics import run_npz_chunk_overfit
from tmd_policy.training.runner import train_expert_chunk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tiny.yaml"
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(PROJECT_ROOT / ".cache" / "lerobot"))


def _write_report(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2, sort_keys=True))


def _audit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    report = {
        "config_consumers": config_runtime_report(),
        "lerobot": verify_lerobot_api(config.checkpoints.lerobot_commit),
        "stores": {},
    }
    for store_path in args.store:
        report["stores"][str(Path(store_path).resolve())] = ChunkStore(store_path).audit()
    _write_report(output / "audit.json", report)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    checkpoints = config.checkpoints
    dataset = config.dataset
    report = inspect_compatibility(
        f"https://huggingface.co/datasets/{dataset.repo_id}/resolve/{dataset.revision}/meta/info.json",
        f"https://huggingface.co/{checkpoints.student_id}/resolve/{checkpoints.student_revision}/config.json",
        f"https://huggingface.co/{checkpoints.teacher_id}/resolve/{checkpoints.teacher_revision}/config.json",
        dataset_id=dataset.repo_id,
        dataset_revision=dataset.revision,
        student_id=checkpoints.student_id,
        student_revision=checkpoints.student_revision,
        teacher_id=checkpoints.teacher_id,
        teacher_revision=checkpoints.teacher_revision,
        student_effective_state_dim=config.canonical.state_dim,
    )
    target = Path(args.output)
    save_resolved_config(config, target.parent / "resolved_config.yaml")
    _write_report(target, report.to_dict())
    return 0 if report.compatible else 2


def _smoke(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    report = run_synthetic_smoke(output, args.seed)
    print(json.dumps(report, indent=2, sort_keys=True))
    checks = report["tmd"]
    return 0 if checks["loss_decreased"] and checks["deterministic_fixed_noise"] else 1


def _build_expert(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    root = PROJECT_ROOT / ".cache" / "datasets" / config.dataset.repo_id.replace("/", "--")
    dataset = load_lerobot_expert_dataset(
        config.dataset.repo_id,
        config.dataset.revision,
        root,
        config.dataset.episodes,
        config.horizons.prediction_horizon,
        config.horizons.execution_horizon,
        download_videos=not args.no_videos,
    )
    episode_to_task = {}
    for episode in config.dataset.episodes:
        metadata = dataset.meta.episodes[episode]
        tasks = metadata["tasks"]
        if not tasks:
            raise RuntimeError(f"episode {episode} has no task provenance")
        episode_to_task[episode] = int(dataset.meta.get_task_index(tasks[0]))
    train, validation, test = episode_split_three_way(
        episode_to_task,
        validation_fraction=config.dataset.validation_fraction,
        test_fraction=config.dataset.test_fraction,
        seed=config.dataset.split_seed,
    )
    episode_splits = {
        **{episode: "train" for episode in train},
        **{episode: "validation" for episode in validation},
        **{episode: "test" for episode in test},
    }
    _write_report(
        output / "episode_splits.json",
        {"train": train, "validation": validation, "test": test, "episode_to_task": episode_to_task},
    )
    store = ChunkStore(output)
    for sample in build_expert_chunks(
        dataset,
        dataset_id=config.dataset.repo_id,
        dataset_revision=config.dataset.revision,
        prediction_horizon=config.horizons.prediction_horizon,
        execution_horizon=config.horizons.execution_horizon,
        stride=config.dataset.stride,
        max_chunks=args.max_chunks,
        episode_splits=episode_splits,
    ):
        store.append(sample)
    _write_report(output / "build_report.json", {"records": len(store), "output": str(output.resolve())})
    return 0


def _overfit_chunk(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = run_npz_chunk_overfit(
        args.payload, args.output, steps=args.steps, seed=args.seed, device=args.device
    )
    save_resolved_config(config, Path(args.output) / "resolved_config.yaml")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["loss_decreased"] else 1


def _train_expert(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = train_expert_chunk(
        config,
        expert_manifest=args.expert_manifest,
        output_dir=args.output,
        record_index=args.record_index,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["loss_decreased"] else 1


def _evaluate_policy(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    report = evaluate_policy_arm(
        config, arm=args.arm, output_dir=args.output, checkpoint=args.checkpoint
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _plot_motivation(args: argparse.Namespace) -> int:
    from experiments.motivation.runner import run_experiments

    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.yaml"
    if not resolved.exists():
        save_resolved_config(config, resolved)
    summary = run_experiments(
        output,
        args.experiments,
        seed=args.seed,
        model_kwargs={
            "num_tasks": config.discriminator.num_tasks,
            "model_dim": config.discriminator.model_dim,
            "num_layers": config.discriminator.num_layers,
            "num_heads": config.discriminator.num_heads,
            "feedforward_dim": config.discriminator.feedforward_dim,
            "dropout": config.discriminator.dropout,
        },
        training_steps=config.training.discriminator_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _train_discriminator(args: argparse.Namespace) -> int:
    from experiments.motivation.runner import run_experiments

    config = load_config(args.config)
    output = Path(args.output)
    save_resolved_config(config, output / "resolved_config.yaml")
    model_kwargs = {
        "num_tasks": config.discriminator.num_tasks,
        "model_dim": config.discriminator.model_dim,
        "num_layers": config.discriminator.num_layers,
        "num_heads": config.discriminator.num_heads,
        "feedforward_dim": config.discriminator.feedforward_dim,
        "dropout": config.discriminator.dropout,
    }
    first = run_experiments(
        output,
        ["M0"],
        seed=args.seed,
        model_kwargs=model_kwargs,
        training_steps=config.training.discriminator_steps,
    )
    second = run_experiments(
        output,
        ["M1"],
        seed=args.seed,
        model_kwargs=model_kwargs,
        training_steps=config.training.discriminator_steps,
    )
    print(json.dumps({"M0": first, "M1": second}, indent=2, sort_keys=True))
    return 0


def _evaluate_discriminator(args: argparse.Namespace) -> int:
    path = Path(args.metrics)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _gated(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "executed": False,
                "reason": (
                    "B3/B4 are intentionally gated. Produce saved complete-episode B0, B1, and B2 "
                    "reports before teacher querying, distillation, or replay."
                ),
                "requested_distillation_steps": config.training.distillation_steps,
                "requested_expert_weight": config.training.expert_weight,
                "requested_teacher_weight": config.training.teacher_weight,
                "mismatch_weighting": {
                    "minimum": config.discriminator.weighting_min,
                    "maximum": config.discriminator.weighting_max,
                    "temperature": config.discriminator.weighting_temperature,
                },
                "replay_mode": config.training.replay_mode,
                "minimum_fresh_current": config.training.minimum_fresh_current,
            },
            indent=2,
        )
    )
    return 2


def _run_experiment(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    reports = {}
    for arm in ("B0", "B1", "B2"):
        reports[arm] = evaluate_policy_arm(
            config,
            arm=arm,
            output_dir=root / arm,
            checkpoint=args.b2_checkpoint if arm == "B2" else None,
        )
    _write_report(root / "comparison.json", reports)
    return 0


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))


def _add_policy_eval(parser: argparse.ArgumentParser) -> None:
    _add_config(parser)
    parser.add_argument("--arm", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.set_defaults(handler=_evaluate_policy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmd-policy")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit", help="validate config, LeRobot API, and optional stores")
    _add_config(audit)
    audit.add_argument("--output", required=True)
    audit.add_argument("--store", action="append", default=[])
    audit.set_defaults(handler=_audit)

    inspect_parser = commands.add_parser("inspect", help="verify pinned Hub metadata compatibility")
    _add_config(inspect_parser)
    inspect_parser.add_argument("--output", required=True)
    inspect_parser.set_defaults(handler=_inspect)

    smoke = commands.add_parser("synthetic-smoke", help="run the tiny diagnostic pipeline")
    _add_config(smoke)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--seed", type=int, default=7)
    smoke.set_defaults(handler=_smoke)

    expert = commands.add_parser("build-expert", help="materialize strict expert records")
    _add_config(expert)
    expert.add_argument("--output", required=True)
    expert.add_argument("--max-chunks", type=int, default=8)
    expert.add_argument("--no-videos", action="store_true")
    expert.set_defaults(handler=_build_expert)

    overfit = commands.add_parser("overfit-chunk", help="audit Gaussian TM on one NPZ action chunk")
    _add_config(overfit)
    overfit.add_argument("--payload", required=True)
    overfit.add_argument("--output", required=True)
    overfit.add_argument("--steps", type=int, default=80)
    overfit.add_argument("--seed", type=int, default=7)
    overfit.add_argument("--device", default="cpu")
    overfit.set_defaults(handler=_overfit_chunk)

    train_expert = commands.add_parser("train-expert", help="train B2 on stored expert chunks")
    _add_config(train_expert)
    train_expert.add_argument("--expert-manifest", required=True)
    train_expert.add_argument("--output", required=True)
    train_expert.add_argument("--record-index", type=int, default=0)
    train_expert.set_defaults(handler=_train_expert)

    collect = commands.add_parser("collect-rollouts", help="collect complete episodes for B0/B1/B2")
    _add_policy_eval(collect)
    evaluate = commands.add_parser("evaluate-policy", help="evaluate complete episodes for B0/B1/B2")
    _add_policy_eval(evaluate)

    train_disc = commands.add_parser(
        "train-discriminator", help="run gated M0 then train held-out M1 discriminators"
    )
    _add_config(train_disc)
    train_disc.add_argument("--output", required=True)
    train_disc.add_argument("--seed", type=int, default=101)
    train_disc.set_defaults(handler=_train_discriminator)

    eval_disc = commands.add_parser("evaluate-discriminator", help="inspect a saved metrics artifact")
    eval_disc.add_argument("--metrics", required=True)
    eval_disc.set_defaults(handler=_evaluate_discriminator)

    motivation = commands.add_parser("plot-motivation", help="run and plot M0-M5 diagnostics")
    _add_config(motivation)
    motivation.add_argument("--experiments", nargs="+", default=["M0"])
    motivation.add_argument("--output", required=True)
    motivation.add_argument("--seed", type=int, default=101)
    motivation.set_defaults(handler=_plot_motivation)

    for name in ("query-teacher", "distill"):
        gated = commands.add_parser(name, help="B3/B4 command, gated until B0-B2 pass")
        _add_config(gated)
        gated.set_defaults(handler=_gated)

    run = commands.add_parser("run-experiment", help="evaluate B0, B1, and B2 sequentially")
    _add_config(run)
    run.add_argument("--b2-checkpoint", required=True)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=_run_experiment)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
