import copy

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoModel, GptOssConfig, LlamaConfig, Qwen3Config

from joint_ttopt.factors import FactorMatrix
from joint_ttopt.policy import FrozenActionPolicy
from joint_ttopt.torch_layers import PersistentExperts, PersistentLinear


def tiny_config(kind):
    common = {"vocab_size": 32, "hidden_size": 8, "intermediate_size": 12,
              "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
              "max_position_embeddings": 64, "attention_dropout": 0.0}
    if kind == "qwen3":
        result = Qwen3Config(**common)
    elif kind == "llama":
        result = LlamaConfig(**common)
    else:
        result = GptOssConfig(**dict(common, num_local_experts=4, num_experts_per_tok=2,
                                    head_dim=4, layer_types=["full_attention"]))
    result._attn_implementation = "eager"
    return result


@pytest.mark.parametrize("kind", ["qwen3", "llama", "gpt_oss"])
def test_complete_score_factors_match_dense_autograd_on_backbone_interfaces(kind):
    torch.manual_seed(23)
    backbone = AutoModel.from_config(tiny_config(kind))
    backbone.eval()
    native = copy.deepcopy(backbone)
    policy = FrozenActionPolicy(backbone, nn.Linear(8, 4), seed=8)
    states = {(0, name): FactorMatrix.zeros(*block.matrix_shape) for name, block in policy.blocks.items()}
    policy.set_states(states, 0)
    identifiers = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected_hidden = native(input_ids=identifiers, use_cache=False).last_hidden_state
        actual_hidden = policy.backbone(input_ids=identifiers, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual_hidden, expected_hidden, atol=2e-6, rtol=2e-4)
    decision = policy.sample(0, 0, (1, 2, 3, 4), (0, 2, 3))
    captured = policy.score_gradients(decision)
    reference = copy.deepcopy(policy.backbone)
    native_blocks = {}
    for name, module in reference.named_modules():
        if isinstance(module, PersistentLinear):
            module.base.weight.requires_grad_(True)
            native_blocks[name] = module
        elif isinstance(module, PersistentExperts):
            module.base.down_proj.requires_grad_(True)
            native_blocks[name + ".down_proj"] = module
    identifiers = torch.tensor([decision.input_ids])
    hidden = reference(input_ids=identifiers, attention_mask=torch.ones_like(identifiers), use_cache=False).last_hidden_state[0, -1]
    logits = policy.head(hidden)[list(decision.legal_actions)] / policy.temperature
    score = torch.log_softmax(logits, dim=0)[decision.legal_actions.index(decision.action)]
    score.backward()
    assert len(captured) == 2
    for name, module in native_blocks.items():
        if isinstance(module, PersistentExperts):
            gradient = module.base.down_proj.grad.permute(2, 0, 1).reshape(module.matrix_shape)
        else:
            gradient = module.base.weight.grad
        np.testing.assert_allclose(captured[name].dense(), gradient.detach().numpy(), rtol=2e-4, atol=2e-6)
        assert captured[name].width >= len(decision.input_ids)


def test_fast_weights_are_persistent_and_backbone_weights_remain_frozen():
    torch.manual_seed(7)
    policy = FrozenActionPolicy(AutoModel.from_config(tiny_config("qwen3")), nn.Linear(8, 3))
    initial = {name: parameter.detach().clone() for name, parameter in policy.backbone.named_parameters()}
    states = {}
    for agent in (0, 1):
        for name, wrapper in policy.blocks.items():
            rows, columns = wrapper.matrix_shape
            states[(agent, name)] = FactorMatrix(np.full((rows, 1), 0.01 * (agent + 1)), np.full((1, columns), 0.02))
    policy.set_states(states, 4)
    first = policy.sample(0, 0, (1, 4, 5), (0, 1, 2))
    second = policy.sample(1, 0, (1, 4, 5), (0, 1, 2))
    policy.score_gradients(first)
    policy.score_gradients(second)
    for name, parameter in policy.backbone.named_parameters():
        assert not parameter.requires_grad
        assert torch.equal(parameter, initial[name])
    assert all(torch.equal(parameter, parameter.detach()) and not parameter.requires_grad for parameter in policy.head.parameters())
    assert policy.version == 4


def test_recorded_legal_distribution_and_version_are_checked():
    from dataclasses import replace
    policy = FrozenActionPolicy(AutoModel.from_config(tiny_config("llama")), nn.Linear(8, 3))
    policy.set_states({(0, name): FactorMatrix.zeros(*wrapper.matrix_shape) for name, wrapper in policy.blocks.items()}, 0)
    decision = policy.sample(0, 0, (1, 3, 5), (1,))
    assert decision.action == 1 and decision.log_probability == pytest.approx(0)
    assert all(np.linalg.norm(value.dense()) < 1e-8 for value in policy.score_gradients(decision).values())
    with pytest.raises(ValueError):
        policy.score_gradients(replace(decision, version=1))
    with pytest.raises(ValueError):
        policy.score_gradients(replace(decision, log_probability=-1.0))
