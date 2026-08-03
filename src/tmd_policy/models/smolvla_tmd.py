from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .tmd import MainBackboneOutput, TMDActionGenerator
from .transition_head import InnerSourceMode, RecurrentTransitionHead


@dataclass
class SmolVLAContext:
    prefix_pad_masks: Tensor
    past_key_values: Any


class SmolVLAMainBackbone(nn.Module):
    """Exposes cached SmolVLA context and action-token features."""

    def __init__(self, base_policy: nn.Module) -> None:
        super().__init__()
        self.base_policy = base_policy

    @property
    def feature_dim(self) -> int:
        return int(self.base_policy.model.vlm_with_expert.expert_hidden_size)

    def build_context(self, batch: dict[str, Tensor]) -> SmolVLAContext:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        base = self.base_policy
        flow = base.model
        images, image_masks = base.prepare_images(batch)
        state = base.prepare_state(batch)
        tokens = batch["observation.language.tokens"]
        language_mask = batch["observation.language.attention_mask"]
        prefix, pad_mask, attention_mask = flow.embed_prefix(
            images, image_masks, tokens, language_mask, state=state
        )
        attention_2d = make_att_2d_masks(pad_mask, attention_mask)
        positions = torch.cumsum(pad_mask, dim=1) - 1
        _, cache = flow.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=flow.config.use_cache,
            fill_kv_cache=True,
        )
        return SmolVLAContext(prefix_pad_masks=pad_mask, past_key_values=cache)

    @staticmethod
    def _cache_signature(cache: Any) -> tuple[Any, ...]:
        """Return the structural state that suffix-only calls must not extend."""

        get_seq_length = getattr(cache, "get_seq_length", None)
        if callable(get_seq_length):
            return (type(cache).__qualname__, int(get_seq_length()))
        if isinstance(cache, (tuple, list)):
            shapes: list[tuple[int, ...]] = []
            for layer in cache:
                for value in layer if isinstance(layer, (tuple, list)) else (layer,):
                    if isinstance(value, Tensor):
                        shapes.append(tuple(value.shape))
            return (type(cache).__qualname__, *shapes)
        return (type(cache).__qualname__,)

    def forward(self, context: SmolVLAContext, action_state: Tensor, outer_time: Tensor) -> MainBackboneOutput:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        flow = self.base_policy.model
        suffix, suffix_pad, suffix_attention = flow.embed_suffix(action_state, outer_time)
        suffix_length = suffix_pad.shape[1]
        batch = suffix_pad.shape[0]
        prefix_length = context.prefix_pad_masks.shape[1]
        prefix_2d = context.prefix_pad_masks[:, None, :].expand(batch, suffix_length, prefix_length)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_attention)
        full_mask = torch.cat((prefix_2d, suffix_2d), dim=2)
        prefix_offsets = context.prefix_pad_masks.sum(dim=-1)[:, None]
        positions = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        cache_before = self._cache_signature(context.past_key_values)
        outputs, _ = flow.vlm_with_expert.forward(
            attention_mask=full_mask,
            position_ids=positions,
            past_key_values=context.past_key_values,
            inputs_embeds=[None, suffix],
            use_cache=flow.config.use_cache,
            fill_kv_cache=False,
        )
        cache_after = self._cache_signature(context.past_key_values)
        if cache_after != cache_before:
            raise RuntimeError(
                "SmolVLA prefix KV cache changed during a suffix-only evaluation; "
                f"before={cache_before}, after={cache_after}"
            )
        features = outputs[1][:, -flow.config.chunk_size :].float()
        transition = flow.action_out_proj(features)
        return MainBackboneOutput(transition=transition, features=features)


class SmolVLATMDPolicy(nn.Module):
    def __init__(
        self,
        base_policy: nn.Module,
        *,
        outer_steps: int = 2,
        inner_steps: int = 2,
        recurrent_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        main_loss_weight: float = 0.0,
        transition_loss: str = "huber",
        inner_source_mode: InnerSourceMode | str = InnerSourceMode.GAUSSIAN_TM,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.main_backbone = SmolVLAMainBackbone(base_policy)
        head = RecurrentTransitionHead(
            action_dim=int(self.config.max_action_dim),
            backbone_feature_dim=self.main_backbone.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=recurrent_layers,
            prediction_horizon=int(self.config.chunk_size),
            dropout=dropout,
        )
        self.generator = TMDActionGenerator(
            self.main_backbone,
            head,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            main_loss_weight=main_loss_weight,
            transition_loss=transition_loss,
            inner_source_mode=inner_source_mode,
        )

    @property
    def evaluation_counts(self) -> dict[str, int]:
        return dict(self.generator.last_counts)

    def reset(self) -> None:
        self.base_policy.reset()

    def train(self, mode: bool = True) -> SmolVLATMDPolicy:
        """Train the lightweight modules while keeping the pretrained base in eval mode."""

        super().train(mode)
        self.base_policy.eval()
        return self

    def configure_trainable(self, *, train_main_action_projections: bool = False) -> None:
        self.base_policy.requires_grad_(False)
        self.generator.transition_head.requires_grad_(True)
        if train_main_action_projections:
            flow = self.base_policy.model
            for module in (
                flow.action_in_proj,
                flow.action_out_proj,
                flow.action_time_mlp_in,
                flow.action_time_mlp_out,
                flow.state_proj,
            ):
                module.requires_grad_(True)

    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        inner_noises: Tensor | None = None,
    ) -> Tensor:
        self.eval()
        context = self.main_backbone.build_context(batch)
        if noise is None:
            batch_size = batch["observation.state"].shape[0]
            shape = (batch_size, self.config.chunk_size, self.config.max_action_dim)
            noise = self.base_policy.model.sample_noise(shape, batch["observation.state"].device)
        actions = self.generator.sample(context, noise, inner_noises=inner_noises)
        return actions[:, :, : self.config.action_feature.shape[0]]

    def transition_matching_loss(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        inner_noise: Tensor | None = None,
        outer_time: Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        context = self.main_backbone.build_context(batch)
        actions = self.base_policy.prepare_action(batch)
        valid = None
        if "action_is_pad" in batch:
            valid = ~batch["action_is_pad"].bool()
        return self.generator.transition_matching_loss(
            context,
            actions,
            valid,
            noise=noise,
            inner_noise=inner_noise,
            outer_time=outer_time,
            reduction=reduction,
        )

    def save_training_checkpoint(
        self,
        path: str | Path,
        *,
        discriminator: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
        metadata: dict[str, Any],
    ) -> Path:
        from tmd_policy.training.checkpoint import save_training_checkpoint

        return save_training_checkpoint(
            path,
            policy=self,
            discriminator=discriminator,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metadata=metadata,
        )

    def load_training_checkpoint(
        self,
        path: str | Path,
        *,
        discriminator: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scaler: Any | None,
    ) -> dict[str, Any]:
        from tmd_policy.training.checkpoint import load_training_checkpoint

        return load_training_checkpoint(
            path,
            policy=self,
            discriminator=discriminator,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )


def load_smolvla_tmd(
    checkpoint: str,
    *,
    revision: str | None = None,
    processor_revision: str | None = None,
    lerobot_commit: str = "3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6",
    device: str = "cuda",
    outer_steps: int = 2,
    inner_steps: int = 2,
    recurrent_layers: int = 2,
    hidden_dim: int = 256,
    main_loss_weight: float = 0.0,
    inner_source_mode: InnerSourceMode | str = InnerSourceMode.GAUSSIAN_TM,
) -> tuple[SmolVLATMDPolicy, Any, Any]:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    from tmd_policy.compatibility.lerobot_api import verify_lerobot_api

    verify_lerobot_api(lerobot_commit)
    kwargs: dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    base = SmolVLAPolicy.from_pretrained(checkpoint, **kwargs).to(device)
    verify_lerobot_api(lerobot_commit, base)
    base.config.device = device
    preprocess, postprocess = make_pre_post_processors(
        base.config,
        checkpoint,
        pretrained_revision=processor_revision or revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    policy = SmolVLATMDPolicy(
        base,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        recurrent_layers=recurrent_layers,
        hidden_dim=hidden_dim,
        main_loss_weight=main_loss_weight,
        inner_source_mode=inner_source_mode,
    ).to(device)
    policy.configure_trainable()
    return policy, preprocess, postprocess
