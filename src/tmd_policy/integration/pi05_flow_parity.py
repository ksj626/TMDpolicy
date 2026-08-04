"""Compare repository Euler integration with official PI0.5 on one real batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from tmd_policy.backends.action_coordinates import ActionNormalizer
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, cache_fingerprint
from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot
from tmd_policy.config import load_config, project_path, save_resolved_config
from tmd_policy.data.libero import LeRobotLiberoChunks


def _error(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    difference = (left - right).abs()
    return {"maximum_absolute": float(difference.max()), "mean_absolute": float(difference.mean())}


def run(config_path: str | Path, output_dir: str | Path, *, sample_index: int) -> dict:
    config = load_config(config_path, expected_method="pi05_flow_parity")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    teacher_asset = config["models"]["teacher"]
    teacher = LeRobotPI05Teacher.from_pretrained(
        teacher_asset["id"],
        revision=teacher_asset["revision"],
        processor_revision=teacher_asset["processor_revision"],
        device=config["parity"]["device"],
        dtype=config["parity"]["dtype"],
        minimum_score_time=float(config["parity"]["minimum_score_time"]),
        cache_dir=project_path(config["dataset"]["cache"]) / "hub",
        local_files_only=bool(config["backend"].get("local_files_only", False)),
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )
    dataset = LeRobotLiberoChunks(
        project_path(config["dataset"]["manifest"]),
        "validation",
        root=project_path(config["dataset"]["cache"]) / "datasets" / "lerobot--libero",
        download_videos=True,
    )
    batch = next(iter(DataLoader(Subset(dataset, [sample_index]), batch_size=1)))
    processed = teacher.preprocess_observation(batch)
    condition = teacher.encode_condition(processed)
    seed = int(config["parity"]["noise_seed"])
    generator = torch.Generator(device=teacher.device).manual_seed(seed)
    noise = torch.randn(1, 50, 32, generator=generator, device=teacher.device, dtype=torch.float32)
    steps = int(config["parity"]["num_steps"])
    grid = torch.linspace(1.0, 0.0, steps + 1, device=teacher.device, dtype=torch.float32)
    before = cache_fingerprint(condition.past_key_values)
    wrapper = teacher.sample(condition, noise.clone(), steps, grid)
    after = cache_fingerprint(condition.past_key_values)
    repeat = teacher.sample(condition, noise.clone(), steps, grid)

    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, image_masks = teacher.policy._preprocess_images(processed)
    official = teacher.policy.model.sample_actions(
        images,
        image_masks,
        processed[OBS_LANGUAGE_TOKENS],
        processed[OBS_LANGUAGE_ATTENTION_MASK],
        noise=noise.clone(),
        num_steps=steps,
    )
    wrapper_post = teacher.postprocess_action(wrapper)
    official_post = teacher.postprocess_action(official)
    time = torch.full((1,), 0.5, device=teacher.device)
    velocity = teacher.velocity(condition, noise, time)
    score = teacher.score(condition, noise, time)
    normalizer = ActionNormalizer.from_pipeline(teacher.preprocessor).to(teacher.device)
    canonical = torch.as_tensor(batch["action"], device=teacher.device)[..., :7]
    round_trip = normalizer.unnormalize(normalizer.normalize(canonical))
    report = {
        "data": "real lerobot/libero validation batch",
        "sample_index": sample_index,
        "noise_seed": seed,
        "num_steps": steps,
        "normalized_32d": _error(wrapper, official),
        "normalized_valid_7d": _error(wrapper[..., :7], official[..., :7]),
        "official_postprocessed_7d": _error(wrapper_post, official_post),
        "cache_unchanged": before == after == condition.fingerprint,
        "deterministic_repeatability": _error(wrapper, repeat),
        "raw_velocity": {"shape": list(velocity.shape), "dtype": str(velocity.dtype), "device": str(velocity.device)},
        "score": {
            "finite_fraction": float(torch.isfinite(score).float().mean()),
            "minimum": float(score.min()),
            "maximum": float(score.max()),
            "mean": float(score.mean()),
            "minimum_query_time": teacher.minimum_score_time,
        },
        "coordinate_round_trip": _error(canonical, round_trip),
        "provenance": {
            "lerobot": verify_installed_lerobot(
                expected_source_hashes=config["backend"].get("expected_source_hashes")
            ),
            "models": config["models"],
            "dataset": config["dataset"],
        },
    }
    (output / "parity.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, sample_index=args.sample_index), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
