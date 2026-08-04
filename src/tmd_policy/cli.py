"""Command-line entry points; every training verb constructs real assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tmd_policy.config import load_config, project_path


def _output(args: argparse.Namespace, config: dict) -> Path:
    return Path(args.output).resolve() if args.output else project_path(config["output"]["directory"])


def _data_build(args: argparse.Namespace) -> int:
    from tmd_policy.data.libero import build_episode_manifest

    config = load_config(args.config, expected_method="data_build_expert")
    output = _output(args, config)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite expert output: {output}")
    manifest = build_episode_manifest(
        repo_id=config["dataset"]["id"],
        revision=config["dataset"]["revision"],
        root=project_path(config["dataset"]["cache"]) / "datasets" / "lerobot--libero",
        output=output / "episode_manifest.json",
        validation_fraction=float(config["dataset"]["validation_fraction"]),
        test_fraction=float(config["dataset"]["test_fraction"]),
        seed=int(config["dataset"]["split_seed"]),
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _teacher_parity(args: argparse.Namespace) -> int:
    from tmd_policy.integration.pi05_flow_parity import run

    report = run(args.config, _output(args, load_config(args.config)), sample_index=args.sample_index)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["cache_unchanged"] else 1


_TRAIN_METHODS = {
    "flow-sft": "flow_sft",
    "tmd-stage1": "tmd_stage1",
    "dmd2-flow": "dmd2_flow",
    "tmd-stage2": "tmd_stage2",
    "occupancy-discriminator": "occupancy_discriminator",
    "occupancy-tmd": "occupancy_tmd",
}


def _train(args: argparse.Namespace) -> int:
    from tmd_policy.training.builders import build_training_bundle
    from tmd_policy.training.engine import run_training

    expected = _TRAIN_METHODS[args.train_method]
    config = load_config(args.config, expected_method=expected)
    bundle = build_training_bundle(config)
    report = run_training(
        bundle.program,
        bundle.train_dataset,
        bundle.validation_dataset,
        config=config,
        output_dir=_output(args, config),
        resume=args.resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _rollout(args: argparse.Namespace) -> int:
    from tmd_policy.evaluation.libero import collect_student_rollouts

    config = load_config(args.config, expected_method="collect_student")
    report = collect_student_rollouts(config, _output(args, config))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from tmd_policy.evaluation.libero import evaluate_libero

    config = load_config(args.config, expected_method="evaluate_libero")
    if args.policy_method is not None:
        config["policy"]["method"] = args.policy_method
    if args.checkpoint is not None:
        config["policy"]["checkpoint"] = args.checkpoint
    if args.checkpoint_sha256 is not None:
        config["policy"]["checkpoint_sha256"] = args.checkpoint_sha256
    if args.outer_steps is not None:
        config["policy"]["outer_steps"] = args.outer_steps
    if args.inner_steps is not None:
        config["policy"]["inner_steps"] = args.inner_steps
    report = evaluate_libero(config, _output(args, config))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    from tmd_policy.evaluation.compare import compare

    report = compare(args.config, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tmd-policy")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="real expert data operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    build = data_commands.add_parser("build-expert")
    build.add_argument("--config", default="configs/data/libero.yaml")
    build.add_argument("--output")
    build.set_defaults(handler=_data_build)

    teacher = commands.add_parser("teacher", help="frozen PI0.5 validation")
    teacher_commands = teacher.add_subparsers(dest="teacher_command", required=True)
    parity = teacher_commands.add_parser("validate-pi05-flow")
    parity.add_argument("--config", default="configs/teacher/pi05_flow_parity.yaml")
    parity.add_argument("--output")
    parity.add_argument("--sample-index", type=int, default=0)
    parity.set_defaults(handler=_teacher_parity)

    train = commands.add_parser("train", help="real DataLoader/model training")
    train_commands = train.add_subparsers(dest="train_method", required=True)
    for command, expected in _TRAIN_METHODS.items():
        sub = train_commands.add_parser(command)
        sub.add_argument("--config", default=f"configs/methods/{expected}.yaml")
        sub.add_argument("--output")
        sub.add_argument("--resume")
        sub.set_defaults(handler=_train)

    rollout = commands.add_parser("rollout", help="real LIBERO student rollouts")
    rollout_commands = rollout.add_subparsers(dest="rollout_command", required=True)
    collect = rollout_commands.add_parser("collect-student")
    collect.add_argument("--config", default="configs/rollout/student.yaml")
    collect.add_argument("--output")
    collect.set_defaults(handler=_rollout)

    evaluate = commands.add_parser("evaluate", help="complete-episode LIBERO evaluation")
    evaluate_commands = evaluate.add_subparsers(dest="evaluate_command", required=True)
    libero = evaluate_commands.add_parser("libero")
    libero.add_argument("--config", default="configs/evaluation/libero_motivation.yaml")
    libero.add_argument("--output")
    libero.add_argument(
        "--policy-method",
        choices=("smolvla", "flow_sft", "tmd_stage1", "dmd2_flow", "tmd_stage2", "occupancy_tmd"),
    )
    libero.add_argument("--checkpoint")
    libero.add_argument("--checkpoint-sha256")
    libero.add_argument("--outer-steps", type=int)
    libero.add_argument("--inner-steps", type=int)
    libero.set_defaults(handler=_evaluate)
    comparison = evaluate_commands.add_parser("compare")
    comparison.add_argument("--config", default="configs/experiments/motivation.yaml")
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
