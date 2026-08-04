# TMDpolicy

TMDpolicy is an executable research codebase for distilling a frozen
`lerobot/pi05_libero_finetuned` PI0.5 teacher into
`lerobot/smolvla_libero` on the immutable real `lerobot/libero` dataset. It
implements official-objective Flow-SFT, a SmolVLA Transition Matching
Distillation port, DMD2-flow, Stage-2 DMD2-v refinement, and an experimental
occupancy-weighted TMD method. VLA-OPD was deliberately removed; there is no
OPD registry entry, config, script, test, density estimator, or executable path.

## Environment and assets

The fixed target is native Linux x86_64, Python 3.12, NVIDIA CUDA/BF16,
PyTorch 2.11.0 from the explicit CUDA 12.8 wheel index, and PyPI
`lerobot[training,pi,smolvla,libero,evaluation]==0.6.1`.

```bash
cd /home/dmsdmswns/TMDpolicy
bash scripts/setup/create_environment.sh
conda activate tmdpolicy
huggingface-cli login
export MUJOCO_GL=egl
export HF_HOME=/home/dmsdmswns/TMDpolicy/.cache/huggingface
export HF_LEROBOT_HOME=/home/dmsdmswns/TMDpolicy/.cache/lerobot
```

Accept gated access to `google/paligemma-3b-pt-224` before PI0.5 loading. The
repository never modifies LeRobot or site-packages. Each run checks
`importlib.metadata.version("lerobot") == "0.6.1"` and records source paths and
SHA-256 hashes for PI0.5, SmolVLA, normalization, and processor-factory modules.
See [environment/README.md](environment/README.md).

Pinned assets:

- teacher: `lerobot/pi05_libero_finetuned` at
  `8e174154ef5f6c60a8da12ae99c303d8963138c1`;
- student: `lerobot/smolvla_libero` at
  `31d453f7edd78c839a8bbc39744a292686daf0de`;
- expert data: `lerobot/libero` at
  `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`.

## Real workflow

Build the one episode-disjoint data manifest, then validate PI0.5 raw flow:

```bash
conda run -n tmdpolicy tmd-policy data build-expert \
  --config configs/data/libero.yaml
MUJOCO_GL=egl conda run -n tmdpolicy tmd-policy teacher validate-pi05-flow \
  --config configs/teacher/pi05_flow_parity.yaml \
  --output artifacts/pi05_flow_parity
```

Every training script constructs a real LeRobot dataset, processors, and model
objects. None has a synthetic or dry execution mode:

```bash
bash scripts/train/train_flow_sft.sh
bash scripts/train/train_tmd_stage1.sh
bash scripts/train/train_dmd2_flow.sh
bash scripts/train/train_tmd_stage2.sh
bash scripts/data/collect_student_rollouts.sh
bash scripts/train/train_occupancy_discriminator.sh
bash scripts/train/train_occupancy_tmd.sh
```

Stage 2 and occupancy-TMD depend on upstream immutable checkpoints. Run
`sha256sum artifacts/.../checkpoints/final.pt` and replace the corresponding
`REPLACE_WITH_SHA256_FROM_sha256sum` field before launching. Resume exactly at a
saved boundary:

```bash
conda run -n tmdpolicy tmd-policy train tmd-stage1 \
  --config configs/methods/tmd_stage1.yaml \
  --output artifacts/training/tmd_stage1 \
  --resume artifacts/training/tmd_stage1/checkpoints/latest.pt
```

Checkpoints contain the full program, all optimizers/schedulers/scaler, counters,
sampler cursor, Python/NumPy/Torch/CUDA RNG, resolved config, exact trainable
names, and model/processor/dataset/LeRobot source provenance. New outputs refuse
to overwrite an existing directory.

## Motivation and main LIBERO experiments

The motivation protocol is intentionally larger than a ten-seed LIBERO-10
smoke test. `configs/evaluation/libero_motivation.yaml` evaluates four diagnostic
tasks from each of `libero_spatial`, `libero_object`, `libero_goal`, and
`libero_10`, using the same 20 reset seeds: 320 paired complete episodes per
method. `configs/evaluation/libero_main.yaml` evaluates all 40 suite/task entries
at 50 paired reset seeds: 2,000 episodes per method. This grid makes differences
visible and supports paired confidence intervals instead of relying on noisy
single-task outcomes.

The evaluation CLI accepts policy overrides, so the same paired protocol is
reused without duplicating YAML. The immutable SmolVLA baseline needs no local
checkpoint; trained arms require the checkpoint path and SHA-256:

```bash
MUJOCO_GL=egl conda run -n tmdpolicy tmd-policy evaluate libero \
  --config configs/evaluation/libero_motivation.yaml \
  --policy-method smolvla \
  --output artifacts/evaluation/motivation_smolvla

MUJOCO_GL=egl conda run -n tmdpolicy tmd-policy evaluate libero \
  --config configs/evaluation/libero_motivation.yaml \
  --policy-method tmd_stage1 \
  --checkpoint artifacts/training/tmd_stage1/checkpoints/final.pt \
  --checkpoint-sha256 REPLACE_WITH_SHA256_FROM_sha256sum \
  --outer-steps 2 --inner-steps 2 \
  --output artifacts/evaluation/motivation_tmd_stage1

conda run -n tmdpolicy tmd-policy evaluate compare \
  --config configs/experiments/motivation.yaml \
  --output artifacts/experiments/motivation_comparison
```

Comparison requires identical `(suite, task_id, reset_seed)` keys and reports
overall, per-suite, and per-task paired bootstrap success differences plus exact
McNemar tests. Results also include Wilson intervals, macro/micro success,
synchronized replan latency, action-path smoothness, and optional versioned
rollout paths. Substitute `configs/evaluation/libero_main.yaml` to execute the
2,000-episode main grid, then compare those outputs with
`configs/experiments/main.yaml`.

## Method fidelity and hardware

- Flow-SFT is the exact official LeRobot SmolVLA flow-matching objective.
- `tmd_split_transformer_head` is the primary, paper-closest SmolVLA
  action-space port. LeRobot 0.6.1 exposes no supported partial expert forward,
  so its repository-owned inner transformer is not claimed as an exact
  paper-native architecture. `tmd_gru_head` is a lightweight adaptation.
- Stage-2 uses the actual Stage-1 sampler and online PI0.5 DMD2-v updates.
- DMD2 `pi05_clone` is closest to the original fake-score architecture but very
  expensive; `smolvla_clone` and default `lightweight` are labeled adaptations.
- occupancy-weighted TMD is a proposed method, not a published-paper reproduction.

Flow/TMD head-only work generally needs a 24 GiB-class GPU. DMD2 with a frozen
4B PI0.5 teacher is configured for two 24+ GiB GPUs (`cuda:0` student/fake score,
`cuda:1` teacher); PI0.5-clone fake score needs materially more. Full/LoRA modes,
batch size, activation memory, prefix cache, and optimizer states change the
actual requirement. No large run is launched by setup or validation.

Architecture, tensor contracts, configuration fields, and limitations are in
[docs/architecture.md](docs/architecture.md),
[docs/experiment_protocol.md](docs/experiment_protocol.md), and each module's
README.
