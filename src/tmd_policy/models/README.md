# Models

This module owns model mathematics. It performs no dataset or environment I/O.

| File | Purpose | Public classes/functions | Caller → downstream | Inputs → outputs | Trainable parameters | Side effects / config / artifacts | Limitations |
|---|---|---|---|---|---|---|---|
| `transition_head.py` | Gaussian residual transition flow and explicit anchored ablation. | `InnerSourceMode`, `RecurrentTransitionHead`, `scalar_embedding` | `TMDActionGenerator` → refined velocity | `B,F,A_t,t,Z,s,mask` → transition/loss | projections, position embedding, GRUCells, norms, zero-init output | Uses `tmd.inner_steps`, mode, layers, width, dropout, loss; no files | MeanFlow is reserved, not implemented. |
| `tmd.py` | Outer Euler sampler and per-sample TM objective. | `MainBackboneOutput`, `TMDActionGenerator`, `gaussian_source_like`, `oracle_outer_integrate` | SmolVLA wrapper/training → actions/loss vectors | context/noises/actions → `[B,50,D]` or named losses | owns head/backbone modules | Updates evaluation counters; no files; uses outer/inner steps and loss weights | Euler only; no adaptive solver. |
| `smolvla_tmd.py` | Official-processor SmolVLA adapter, prefix KV reuse, trainable policy, audited loader. | `SmolVLAContext`, `SmolVLAMainBackbone`, `SmolVLATMDPolicy`, `load_smolvla_tmd` | runners → `TMDActionGenerator` | canonical processed batch → internal/canonical action chunk | head; optional action/state/time projections; VLM frozen | Downloads only into project cache; validates pinned LeRobot; checkpoint wrappers | Consumes pinned SmolVLA internals. |
| `discriminator.py` | Train-normalized pointwise/final/causal-prefix path classifier. | `DiscriminatorVariant`, `PathNormalizer`, `CausalPathDiscriminator` | discriminator trainer/motivation → metrics/weights | path/task/mask → logits | normalizer buffers (not trained), token projection, task/position embeddings, transformer, head | Normalizer may fit only with `split='train'`; no files | Low-dimensional paths only. |
| `__init__.py` | Stable model exports. | selected classes/functions | package callers → model files | n/a | none | none | no behavior |

## Gaussian and outer flows

```mermaid
flowchart LR
  Z[Z ~ N(0,I), s=1] --> Ys[Y_s]
  Y[Y=epsilon-A, s=0] --> Ys
  B[backbone B] --> D[Delta_theta recurrent]
  Ys --> D
  D --> Yh[Y_hat=B+Delta]
  Z --> U[u=Z-Y_hat]
  Yh --> U
  U -->|ds < 0| End[refined transition]
```

```mermaid
sequenceDiagram
  participant X as outer state A_t
  participant K as cached prefix KV
  participant B as SmolVLA backbone
  participant H as transition head
  loop outer steps N
    X->>B: A_t,t,K
    K-->>B: immutable prefix cache
    B-->>H: B,F
    loop inner steps M
      H->>H: Delta at s; recurrent hidden carried
    end
    H-->>X: refined velocity; dt=-1/N
  end
```

## Tensor dictionary

| Variable | Symbol / meaning | Shape | dtype/device | Coordinates / normalization / range | Mask / phase / gradients | Random producer → consumer |
|---|---|---|---|---|---|---|
| `actions` | `A`, clean plan | `[B,50,D]`, canonical `D=7`, internal padded `D=max_action_dim` | floating, model device | official student-normalized inside model; canonical after postprocess | valid mask `[B,50]`; training; no target gradient | expert/teacher → loss |
| `noise` | `epsilon`, outer source | same as `A` | same dtype/device | standard normal in internal action coordinates | train+infer; fixed seed for replay; no grad | `gaussian_source_like`/runner → outer path |
| `outer_state` | `A_t` | `[B,50,D]` | floating/model device | `(1-t)A+t epsilon` | train+infer; gradients allowed toward enabled main projections | analytic path/sampler → backbone |
| `outer_time` | `t` | `[B]` | float32/model device | dimensionless `[0,1]`, evaluated descending at inference | both; no mask/no grad | uniform train or Euler grid → backbone/head |
| `target_transition` | `Y=epsilon-A` | `[B,50,D]` | floating/model device | student-normalized velocity | training only; mask applies; detached target semantics | analytic producer → losses |
| `backbone_transition` | `B` | `[B,50,D]` | float32/model device | student-normalized velocity reference | both; gradient only for explicitly enabled projections | SmolVLA action output → head/outer loss |
| `backbone_features` | `F` | `[B,50,Fdim]` | float32/model device | latent coordinates, not normalized externally | both; VLM frozen, optional projections upstream | cached SmolVLA suffix → head hidden/input |
| `inner_noise` | `Z` | `[B,50,D]`; collection `[N,B,50,D]` | same as `Y` | independent standard normal | both; one seed per outer call; no grad | runner/generator → analytic path/refine |
| `inner_state` | `Y_s` | `[B,50,D]` | floating/model device | transition-velocity coordinates | both; padded positions excluded from loss | analytic path or Euler state → head |
| `inner_time` | `s` | `[B]` | float32/model device | dimensionless, `1→0` | both; no mask/no grad | loop grid → time embedding |
| `residual` | `Delta` | `[B,50,D]` | floating/model device | correction to `B`; zero at initialization | both; primary gradient-bearing output | recurrent head → `Y_hat` |
| `predicted_velocity` | `u=Z-(B+Delta)` | `[B,50,D]` | floating/model device | inner-flow velocity | both; masked training error; grad to head | residual parameterization → loss/Euler |
| `hidden` | recurrent state | list of `L×[B,50,Hdim]` | floating/model device | latent | carried across inner steps; grad allowed | feature projection/GRU → next GRU call |
| `states/actions` | discriminator path | `[B,11,8]`, `[B,10,7]` | float32/model device | canonical then train-normalized | prefix bool mask; discriminator gradient only | data collator → normalizer/tokenizer |
| `logits` | estimated `log rho_E/rho_C` orientation | `[B,10]` or `[B,1]` | float32/model device | uncalibrated logit | valid prefixes; no student gradient | discriminator → metrics/weight builder |

## Scalar and metadata variables

| Name | Type / valid / default | Config origin | Purpose / lifetime / provenance |
|---|---|---|---|
| `outer_steps` | positive int, 2 | `tmd.outer_steps` | Expensive calls per plan; checkpoint identity and inference lifetime. |
| `inner_steps` | positive int, 2 | `tmd.inner_steps` | Lightweight calls per outer step; checkpoint/seed identity. |
| `inner_source_mode` | `gaussian_tm` or explicit ablation | `tmd.inner_source_mode` | Selects mathematics; invalidates checkpoint comparison. |
| `transition_loss` | `huber`/`mse`, Huber | `tmd.loss` | Elementwise velocity regression during training. |
| `reduction` | `none`/`mean`, mean | function argument | Preserves `[B]` losses before weighting. |
| `variant` | pointwise/final/prefix | discriminator experiment | Controlled discriminator architecture; checkpoint identity. |
| `evaluation_counts` | mapping | runtime | Verifies exactly `N` backbone and `N×M` head calls. |

Public functions validate all non-obvious shapes. Inference APIs accept context and
noise, never target actions. Prefix cache structure is fingerprinted before and
after every suffix-only call and mutation is a hard error.
