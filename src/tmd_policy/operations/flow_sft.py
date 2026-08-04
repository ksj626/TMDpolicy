from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from tmd_policy.common.data import make_dataloader
from tmd_policy.methods.flow_sft import FlowSFTConfig, FlowSFTMethod


def _canonical_batch(batch: dict[str, Any]) -> dict[str, Any]:
    arrays, metadata = batch["arrays"], batch["metadata"]
    result: dict[str, Any] = {
        "observation.state": arrays["state_sequence"][:, 0],
        "action": arrays["action_plan_canonical"],
        "action_is_pad": ~arrays["action_valid"].bool(),
        "task": [item["task_identity"]["instruction"] for item in metadata],
    }
    for key, value in arrays.items():
        if key.startswith("image::"):
            result[key.removeprefix("image::")] = value
    return result


def train_flow_sft(config: dict[str, Any], *, output: str | Path, resume: str | None) -> dict[str, Any]:
    """Real user-invoked Flow-SFT trainer; never called by dry-run."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    device = str(config["training"].get("device", "cuda"))
    revisions = config["models"]["revisions"]
    policy = SmolVLAPolicy.from_pretrained(
        config["models"]["student"], revision=revisions["student"]
    ).to(device)
    policy.config.device = device
    preprocessor, _ = make_pre_post_processors(
        policy.config, config["models"]["student"],
        pretrained_revision=revisions["student_processor"],
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    method_config = FlowSFTConfig(
        fine_tuning=str(config["training"].get("fine_tuning", "frozen_backbone")),
        mixed_precision=str(config["training"].get("mixed_precision", "bf16")),
        gradient_accumulation=int(config["training"].get("gradient_accumulation", 1)),
        learning_rate=float(config["training"].get("learning_rate", 1e-5)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    method = FlowSFTMethod(
        policy=policy, config=method_config, provenance=config.get("_provenance_summary")
    )
    if resume:
        method.load_method_state(resume)
    store = config["dataset"]["expert_store"]
    loaders = {
        split: make_dataloader(
            store, split=split, batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]), shuffle=split == "train",
        )
        for split in ("train", "validation", "test")
    }
    def train_loader(epoch: int) -> Any:
        return make_dataloader(
            store, split="train", batch_size=int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]) + epoch, shuffle=True,
        )
    def evaluate(loader: Any) -> float:
        values = []
        fixed_generator = torch.Generator(device=device).manual_seed(
            int(config["training"]["seed"]) + 991
        )
        policy.eval()
        with torch.no_grad():
            for raw in loader:
                processed = preprocessor(_canonical_batch(raw))
                actions = policy.prepare_action(processed)
                noise = torch.randn(
                    actions.shape, device=device, dtype=actions.dtype, generator=fixed_generator
                )
                time = torch.rand((actions.shape[0],), device=device, generator=fixed_generator)
                losses, _ = policy(processed, noise=noise, time=time, reduction="none")
                values.extend(losses.detach().cpu().tolist())
        return float(sum(values) / len(values))

    initial_validation = evaluate(loaders["validation"])
    policy.train()
    current_loader = train_loader(method.data_epoch)
    iterator = iter(current_loader)
    for _ in range(method.batch_in_epoch):
        next(iterator)
    global_steps = method.optimizer_steps
    accumulation = method_config.gradient_accumulation
    method.optimizer.zero_grad(set_to_none=True)
    autocast_dtype = torch.bfloat16 if method_config.mixed_precision == "bf16" else torch.float16
    use_autocast = method_config.mixed_precision != "none" and device.startswith("cuda")
    while global_steps < int(config["training"]["max_optimizer_steps"]):
        try:
            raw = next(iterator)
        except StopIteration:
            method.data_epoch += 1
            method.batch_in_epoch = 0
            current_loader = train_loader(method.data_epoch)
            iterator = iter(current_loader)
            raw = next(iterator)
        method.batch_in_epoch += 1
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            loss = method.training_step(preprocessor(_canonical_batch(raw)))["loss"] / accumulation
        method.scaler.scale(loss).backward()
        if method.step % accumulation:
            continue
        method.scaler.unscale_(method.optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 5.0)
        method.scaler.step(method.optimizer)
        method.scaler.update()
        method.optimizer.zero_grad(set_to_none=True)
        method.scheduler.step()
        global_steps += 1
        method.optimizer_steps = global_steps
    final_validation = evaluate(loaders["validation"])
    test_loss = evaluate(loaders["test"])
    checkpoint = method.save_method_state(Path(output) / "checkpoint.pt")
    report = {
        "method": method.name, "optimizer_steps": global_steps,
        "initial_fixed_validation_loss": initial_validation,
        "final_fixed_validation_loss": final_validation, "test_loss": test_loss,
        "checkpoint": str(checkpoint), "trainable_names": method.trainable_names,
    }
    (Path(output) / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
