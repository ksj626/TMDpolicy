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
    initial_rollout_replay = getattr(args, "initial_rollout_replay", None)
    if initial_rollout_replay is not None:
        initial_rollout_replay = str(Path(initial_rollout_replay).expanduser().resolve())
        initial_path = Path(initial_rollout_replay)
        if not initial_path.is_dir() or not (initial_path / "collection_report.json").is_file():
            raise FileNotFoundError(
                "--initial-rollout-replay must be a completed rollout round containing "
                f"collection_report.json: {initial_path}"
            )
    from tmd_policy.training.preflight import preflight

    preflight(config)
    bundle = build_training_bundle(config)
    report = run_training(
        bundle.program,
        bundle.train_dataset,
        bundle.validation_dataset,
        config=config,
        output_dir=_output(args, config),
        resume=args.resume,
        train_batch_sampler=bundle.train_batch_sampler,
        initial_rollout_replay=initial_rollout_replay,
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
    if args.device is not None:
        config["policy"]["device"] = args.device
    if args.inner_steps is not None and config["policy"]["method"] in {
        "dmd2_flow", "tmd_stage1", "tmd_stage2", "occupancy_tmd"
    }:
        raise ValueError(
            "this distilled policy's inner step count comes from the checkpoint"
        )
    if args.outer_steps is not None and config["policy"]["method"] in {
        "tmd_stage1", "tmd_stage2", "occupancy_tmd"
    }:
        raise ValueError("TMD outer step counts come from the checkpoint")
    if args.outer_steps is not None:
        config["policy"]["outer_steps"] = args.outer_steps
    if args.inner_steps is not None:
        config["policy"]["inner_steps"] = args.inner_steps
    evaluation = config["evaluation"]
    sampled = False
    if args.suite:
        requested_suites = set(args.suite)
        available_suites = {str(value["suite"]) for value in evaluation["benchmark"]}
        missing_suites = requested_suites.difference(available_suites)
        if missing_suites:
            raise ValueError(f"evaluation suites are not configured: {sorted(missing_suites)}")
        evaluation["benchmark"] = [
            value
            for value in evaluation["benchmark"]
            if str(value["suite"]) in requested_suites
        ]
        sampled = True
    if args.task_ids is not None:
        task_ids = [int(value) for value in args.task_ids]
        if not task_ids or len(task_ids) != len(set(task_ids)) or not all(
            0 <= value <= 9 for value in task_ids
        ):
            raise ValueError("LIBERO --task-ids must be unique integers in [0,9]")
        for suite_spec in evaluation["benchmark"]:
            suite_spec["task_ids"] = task_ids
        sampled = True
    if args.reset_seeds is not None:
        reset_seeds = [int(value) for value in args.reset_seeds]
        if not reset_seeds or len(reset_seeds) != len(set(reset_seeds)) or min(reset_seeds) < 0:
            raise ValueError("--reset-seeds must be unique nonnegative integers")
        evaluation["reset_seeds"] = reset_seeds
        sampled = True
    if args.max_episode_steps is not None:
        if args.max_episode_steps < 1:
            raise ValueError("--max-episode-steps must be positive")
        evaluation["suite_max_episode_steps"] = {
            suite: int(args.max_episode_steps)
            for suite in evaluation["suite_max_episode_steps"]
        }
        sampled = True
    if sampled:
        config["classification"] = f"{config['classification']}; CLI-scoped sampled evaluation"
    report = evaluate_libero(config, _output(args, config))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _evaluate_libero_plus(args: argparse.Namespace) -> int:
    from tmd_policy.evaluation.libero_plus import evaluate_libero_plus

    config = load_config(args.config, expected_method="evaluate_libero_plus")
    if args.checkpoint is not None:
        config["policy"]["checkpoint"] = args.checkpoint
    if args.checkpoint_sha256 is not None:
        config["policy"]["checkpoint_sha256"] = args.checkpoint_sha256
    if args.device is not None:
        if args.devices is not None:
            raise ValueError("--device and --devices are mutually exclusive")
        config["policy"]["device"] = args.device
    if args.devices is not None:
        devices = [str(value) for value in args.devices]
        if not devices or len(devices) != len(set(devices)):
            raise ValueError("--devices must contain unique device names")
        config["evaluation"]["devices"] = devices
        config["policy"]["device"] = devices[0]
    if args.outer_steps is not None:
        if config["policy"]["method"] not in {"dmd2_flow", "flow_sft"}:
            raise ValueError("only DMD2/Flow-SFT support a LIBERO-Plus outer-step override")
        if args.outer_steps < 1:
            raise ValueError("--outer-steps must be positive")
        config["policy"]["outer_steps"] = int(args.outer_steps)
    evaluation = config["evaluation"]
    if args.suite:
        evaluation["suites"] = list(args.suite)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        evaluation["batch_size"] = int(args.batch_size)
    if args.save_videos is not None:
        evaluation["save_videos"] = bool(args.save_videos)
    if args.task_ids is not None:
        if args.sample_per_category is not None:
            raise ValueError("--task-ids and --sample-per-category are mutually exclusive")
        ids = [int(value) for value in args.task_ids]
        if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
            raise ValueError("--task-ids must be unique nonnegative integers")
        evaluation["task_ids"] = ids
        evaluation["sample_per_category"] = None
    if args.sample_per_category is not None:
        if args.sample_per_category < 1:
            raise ValueError("--sample-per-category must be positive")
        if args.sample_seed < 0:
            raise ValueError("--sample-seed must be nonnegative")
        evaluation["task_ids"] = None
        evaluation["sample_per_category"] = int(args.sample_per_category)
        evaluation["sample_seed"] = int(args.sample_seed)
        config["classification"] = (
            f"{config['classification']}; deterministic category-balanced sampled evaluation"
        )
    elif args.sample_seed != 0:
        raise ValueError("--sample-seed requires --sample-per-category")
    if args.reset_seeds is not None:
        seeds = [int(value) for value in args.reset_seeds]
        if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
            raise ValueError("--reset-seeds must be unique nonnegative integers")
        evaluation["reset_seeds"] = seeds
    if args.max_episode_steps is not None:
        if args.max_episode_steps < 1:
            raise ValueError("--max-episode-steps must be positive")
        evaluation["suite_max_episode_steps"] = {
            suite: int(args.max_episode_steps)
            for suite in evaluation["suite_max_episode_steps"]
        }
    report = evaluate_libero_plus(config, _output(args, config), resume=bool(args.resume))
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
        if command == "dmd2-flow":
            sub.add_argument(
                "--initial-rollout-replay",
                help=(
                    "completed balanced student-rollout round used to bootstrap replay; "
                    "skips the blocking pre-training rollout"
                ),
            )
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
        choices=("pi05", "smolvla", "flow_sft", "tmd_stage1", "dmd2_flow", "tmd_stage2", "occupancy_tmd"),
    )
    libero.add_argument("--checkpoint")
    libero.add_argument("--checkpoint-sha256")
    libero.add_argument(
        "--device",
        help="inference device override, for example cuda:2",
    )
    libero.add_argument("--outer-steps", type=int)
    libero.add_argument("--inner-steps", type=int)
    libero.add_argument(
        "--suite",
        action="append",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        help="restrict evaluation to one or more suites; repeat this flag for multiple suites",
    )
    libero.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        help="suite-local LIBERO task IDs to evaluate (0-9)",
    )
    libero.add_argument(
        "--reset-seeds",
        type=int,
        nargs="+",
        help="episode reset seeds to evaluate",
    )
    libero.add_argument(
        "--max-episode-steps",
        type=int,
        help="optional shortened horizon for a smoke test; use 600 for reportable evaluation",
    )
    libero.set_defaults(handler=_evaluate)
    libero_plus = evaluate_commands.add_parser(
        "libero-plus", help="resumable evaluation over the 10,030-task robustness benchmark"
    )
    libero_plus.add_argument(
        "--config", default="configs/evaluation/libero_plus_dmd2.yaml"
    )
    libero_plus.add_argument("--output")
    libero_plus.add_argument("--checkpoint")
    libero_plus.add_argument("--checkpoint-sha256")
    libero_plus.add_argument("--device")
    libero_plus.add_argument(
        "--devices",
        nargs="+",
        help="run independent serial task shards concurrently, one process per GPU",
    )
    libero_plus.add_argument("--outer-steps", type=int)
    libero_plus.add_argument(
        "--suite",
        action="extend",
        nargs="+",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_10"),
        help="fully evaluate one or more selected suites; accepts one flag or repeated flags",
    )
    libero_plus.add_argument("--task-ids", type=int, nargs="+")
    libero_plus.add_argument(
        "--sample-per-category",
        type=int,
        help="deterministically sample this many tasks from each of the seven categories",
    )
    libero_plus.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="nonnegative seed for --sample-per-category (default: 0)",
    )
    libero_plus.add_argument(
        "--batch-size",
        type=int,
        help="episodes per policy call (default: 1; prefer --devices for acceleration)",
    )
    libero_plus.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="save agent-view MP4s; defaults on for category-sampled runs",
    )
    libero_plus.add_argument("--reset-seeds", type=int, nargs="+")
    libero_plus.add_argument("--max-episode-steps", type=int)
    libero_plus.add_argument(
        "--resume", action="store_true", help="continue a matching output directory"
    )
    libero_plus.set_defaults(handler=_evaluate_libero_plus)
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
