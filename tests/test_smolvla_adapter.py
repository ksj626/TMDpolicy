from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.models.smolvla_tmd import SmolVLAContext, SmolVLAMainBackbone


class FakeCache:
    def __init__(self):
        self.length = 3

    def get_seq_length(self):
        return self.length


class FakeVLM(nn.Module):
    expert_hidden_size = 8

    def __init__(self, mutate_cache: bool):
        super().__init__()
        self.mutate_cache = mutate_cache

    def forward(self, **kwargs):
        cache = kwargs["past_key_values"]
        if self.mutate_cache:
            cache.length += 1
        batch, suffix, _ = kwargs["inputs_embeds"][1].shape
        return [None, torch.zeros(batch, suffix, 8)], cache


class FakeFlow(nn.Module):
    def __init__(self, mutate_cache: bool):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True, chunk_size=5)
        self.vlm_with_expert = FakeVLM(mutate_cache)
        self.action_out_proj = nn.Linear(8, 7)

    def embed_suffix(self, action_state, outer_time):
        del outer_time
        batch, horizon, _ = action_state.shape
        suffix = torch.zeros(batch, horizon, 8)
        pad = torch.ones(batch, horizon, dtype=torch.bool)
        attention = torch.ones(batch, horizon, dtype=torch.bool)
        return suffix, pad, attention


class FakePolicy(nn.Module):
    def __init__(self, mutate_cache: bool):
        super().__init__()
        self.model = FakeFlow(mutate_cache)


def test_suffix_evaluation_does_not_extend_prefix_cache():
    backbone = SmolVLAMainBackbone(FakePolicy(mutate_cache=False))
    context = SmolVLAContext(torch.ones(1, 3, dtype=torch.bool), FakeCache())
    output = backbone(context, torch.randn(1, 5, 7), torch.ones(1))
    assert output.transition.shape == (1, 5, 7)
    assert context.past_key_values.length == 3


def test_suffix_evaluation_fails_when_dependency_mutates_prefix_cache():
    backbone = SmolVLAMainBackbone(FakePolicy(mutate_cache=True))
    context = SmolVLAContext(torch.ones(1, 3, dtype=torch.bool), FakeCache())
    with pytest.raises(RuntimeError, match="prefix KV cache changed"):
        backbone(context, torch.randn(1, 5, 7), torch.ones(1))
