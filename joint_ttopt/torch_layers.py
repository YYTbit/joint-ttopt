import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .factors import FactorMatrix


class PersistentLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("Expected a linear backbone block")
        self.base = base
        self.base.requires_grad_(False)
        self.register_buffer("left", base.weight.new_empty((base.out_features, 0)), persistent=False)
        self.register_buffer("right", base.weight.new_empty((0, base.in_features)), persistent=False)
        self.capture = False
        self.parts = []

    @property
    def matrix_shape(self):
        return self.base.out_features, self.base.in_features

    def install(self, state):
        if state.shape != self.matrix_shape:
            raise ValueError("Fast-weight state has a different backbone shape")
        self.left = torch.tensor(state.left, device=self.base.weight.device, dtype=self.base.weight.dtype)
        self.right = torch.tensor(state.right, device=self.base.weight.device, dtype=self.base.weight.dtype)

    def start_capture(self):
        self.parts = []
        self.capture = True

    def end_capture(self):
        self.capture = False
        result = FactorMatrix.zeros(*self.matrix_shape)
        for part in self.parts:
            result = result.plus(part)
        self.parts = []
        return result

    def forward(self, inputs):
        result = self.base(inputs)
        if self.left.shape[1]:
            result = result + functional.linear(functional.linear(inputs, self.right), self.left)
        if self.capture:
            if not result.requires_grad:
                result.requires_grad_(True)
            activations = inputs.detach().reshape(-1, self.matrix_shape[1]).float().cpu().numpy().T.copy()
            def collect(derivative):
                sensitivity = derivative.detach().reshape(-1, self.matrix_shape[0]).float().cpu().numpy().T
                self.parts.append(FactorMatrix(sensitivity, activations.T))
            result.register_hook(collect)
        return result


class PersistentExperts(nn.Module):
    def __init__(self, base):
        super().__init__()
        required = ["gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias", "alpha", "limit"]
        if any(not hasattr(base, name) for name in required) or base.down_proj.ndim != 3:
            raise ValueError("Expected the released gpt-oss expert projection interface")
        self.base = base
        self.base.requires_grad_(False)
        self.experts, self.intermediate, self.hidden = base.down_proj.shape
        self.register_buffer("left", base.down_proj.new_empty((self.hidden, 0)), persistent=False)
        self.register_buffer("right", base.down_proj.new_empty((0, self.experts * self.intermediate)), persistent=False)
        self.capture = False
        self.parts = []

    @property
    def matrix_shape(self):
        return self.hidden, self.experts * self.intermediate

    def install(self, state):
        if state.shape != self.matrix_shape:
            raise ValueError("Expert fast-weight state has a different logical block shape")
        self.left = torch.tensor(state.left, device=self.base.down_proj.device, dtype=self.base.down_proj.dtype)
        self.right = torch.tensor(state.right, device=self.base.down_proj.device, dtype=self.base.down_proj.dtype)

    def start_capture(self):
        self.parts = []
        self.capture = True

    def end_capture(self):
        self.capture = False
        result = FactorMatrix.zeros(*self.matrix_shape)
        for part in self.parts:
            result = result.plus(part)
        self.parts = []
        return result

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        if router_indices is None or routing_weights is None:
            raise ValueError("Expert projection requires the native router output")
        shape = hidden_states.shape
        inputs = hidden_states.reshape(-1, self.hidden)
        if routing_weights.shape != (len(inputs), self.experts):
            raise ValueError("The pinned gpt-oss router supplies dense token-by-expert weights")
        if router_indices.ndim != 2 or router_indices.shape[0] != len(inputs):
            raise ValueError("Native expert routing indices have incompatible token dimensions")
        result = torch.zeros_like(inputs)
        for expert in range(self.experts):
            locations = torch.nonzero((router_indices == expert).any(dim=-1), as_tuple=False).flatten()
            if not len(locations):
                continue
            features = inputs[locations]
            gates = features @ self.base.gate_up_proj[expert] + self.base.gate_up_proj_bias[expert]
            gate = gates[..., 0::2].clamp(max=self.base.limit)
            up = gates[..., 1::2].clamp(min=-self.base.limit, max=self.base.limit)
            intermediate = (up + 1) * gate * torch.sigmoid(self.base.alpha * gate)
            contribution = intermediate @ self.base.down_proj[expert] + self.base.down_proj_bias[expert]
            start = expert * self.intermediate
            if self.left.shape[1]:
                contribution = contribution + functional.linear(
                    functional.linear(intermediate, self.right[:, start:start + self.intermediate]), self.left)
            if self.capture:
                if not contribution.requires_grad:
                    contribution.requires_grad_(True)
                activations = intermediate.detach().float().cpu().numpy().T.copy()
                def collect(derivative, values=activations, offset=start):
                    sensitivity = derivative.detach().float().cpu().numpy().T
                    right = np.zeros((sensitivity.shape[1], self.matrix_shape[1]))
                    right[:, offset:offset + self.intermediate] = values.T
                    self.parts.append(FactorMatrix(sensitivity, right))
                contribution.register_hook(collect)
            weighted = contribution * routing_weights[locations, expert, None]
            result = result.index_add(0, locations, weighted.to(result.dtype))
        return result.reshape(shape)


def install_blocks(backbone):
    blocks = {}
    modules = list(backbone.named_modules())
    for name, module in modules:
        if isinstance(module, nn.Linear) and name.endswith(("self_attn.o_proj", "mlp.down_proj")):
            parent_name, child = name.rsplit(".", 1)
            wrapper = PersistentLinear(module)
            setattr(backbone.get_submodule(parent_name), child, wrapper)
            blocks[name] = wrapper
        elif name.endswith("mlp.experts") and hasattr(module, "down_proj"):
            parent_name, child = name.rsplit(".", 1)
            wrapper = PersistentExperts(module)
            setattr(backbone.get_submodule(parent_name), child, wrapper)
            blocks[name + ".down_proj"] = wrapper
    if not blocks:
        raise ValueError("Backbone exposes no declared attention or down-projection blocks")
    return blocks
