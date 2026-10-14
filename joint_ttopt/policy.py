import json

import numpy as np
import torch
from torch import nn

from .renderer import ObservationRenderer
from .torch_layers import install_blocks
from .trajectories import Decision


class FrozenActionPolicy:
    def __init__(self, backbone, action_head, seed=0, temperature=1.0,
                 max_context=8192, replay_tolerance=1e-5):
        if not np.isfinite(temperature) or temperature <= 0 or max_context < 1:
            raise ValueError("Invalid action distribution or context limit")
        backbone.eval()
        backbone.requires_grad_(False)
        action_head.eval()
        action_head.requires_grad_(False)
        self.backbone = backbone
        self.head = action_head
        self.blocks = install_blocks(backbone)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.temperature = float(temperature)
        self.max_context = int(max_context)
        self.replay_tolerance = float(replay_tolerance)
        self.version = 0
        self.states = {}
        self.renderer = None
        self.metadata = {}

    @classmethod
    def from_pretrained(cls, checkpoint, head_path, seed=0, revision=None,
                        device_map="auto", dtype="bfloat16", **options):
        from transformers import AutoModel, AutoTokenizer, Mxfp4Config
        from .storage import file_digest
        with np.load(head_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            weight = np.array(archive["weight"], copy=True)
            bias = np.array(archive["bias"], copy=True) if "bias" in archive else None
        if revision is None:
            revision = metadata.get("revision")
        if metadata.get("checkpoint") != checkpoint or metadata.get("revision") != revision:
            raise ValueError("Action-head entry checkpoint or revision differs from the backbone")
        if weight.ndim != 2 or not np.isfinite(weight).all():
            raise ValueError("Action head has an invalid weight matrix")
        arguments = {"revision": revision, "device_map": device_map,
                     "torch_dtype": getattr(torch, dtype), "attn_implementation": "eager"}
        if metadata.get("model_type") == "gpt_oss":
            arguments["quantization_config"] = Mxfp4Config(dequantize=True)
        backbone = AutoModel.from_pretrained(checkpoint, **arguments)
        if any(parameter.is_meta for parameter in backbone.parameters()):
            raise ValueError("Deployment gradients require resident backbone tensors")
        if weight.shape[1] != backbone.config.hidden_size:
            raise ValueError("Action head has a different hidden dimension")
        head = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
        head.weight.data.copy_(torch.tensor(weight, dtype=head.weight.dtype))
        if bias is not None:
            if bias.shape != (weight.shape[0],) or not np.isfinite(bias).all():
                raise ValueError("Action head has an invalid bias")
            head.bias.data.copy_(torch.tensor(bias, dtype=head.bias.dtype))
        result = cls(backbone, head, seed, **options)
        expected = metadata.get("adapted_blocks")
        if expected is None or sorted(expected) != sorted(result.blocks):
            raise ValueError("Action-head record has a different adapted-block set")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        result.renderer = ObservationRenderer(tokenizer, **metadata.get("renderer", {}))
        result.metadata = dict(metadata, head_sha256=file_digest(head_path))
        return result

    def shapes(self, agents):
        return {(agent, block): wrapper.matrix_shape
                for agent in agents for block, wrapper in self.blocks.items()}

    def set_states(self, states, version):
        if version < 0:
            raise ValueError("Policy version is nonnegative")
        for (agent, block), state in states.items():
            if block not in self.blocks or state.shape != self.blocks[block].matrix_shape:
                raise ValueError("Policy state differs from the adapted backbone blocks")
        for agent in set(agent for agent, block in states):
            if {block for candidate, block in states if candidate == agent} != set(self.blocks):
                raise ValueError("Agent policy state omitted an adapted block")
        self.states = dict(states)
        self.version = int(version)

    def activate(self, agent):
        for block, wrapper in self.blocks.items():
            key = (agent, block)
            if key not in self.states:
                raise ValueError("Acting agent has no persistent state")
            wrapper.install(self.states[key])

    def log_probabilities(self, input_ids, legal_actions):
        if len(input_ids) > self.max_context:
            raise ValueError("Rendered observation exceeds the declared context limit")
        if not legal_actions or len(set(legal_actions)) != len(legal_actions):
            raise ValueError("Legal action list is nonempty and distinct")
        if any(value < 0 or value >= self.head.out_features for value in legal_actions):
            raise ValueError("A legal action lies outside the frozen action head")
        device = self.backbone.get_input_embeddings().weight.device
        identifiers = torch.tensor([input_ids], dtype=torch.long, device=device)
        output = self.backbone(input_ids=identifiers, attention_mask=torch.ones_like(identifiers), use_cache=False)
        hidden = output.last_hidden_state[:, -1].float()
        self.head.to(hidden.device)
        logits = self.head(hidden)[0] / self.temperature
        legal = torch.tensor(legal_actions, dtype=torch.long, device=logits.device)
        return torch.log_softmax(logits[legal], dim=-1)

    def sample(self, agent, time, input_ids, legal_actions, baseline_version=0):
        self.activate(agent)
        legal_actions = tuple(sorted(legal_actions))
        with torch.no_grad():
            log_probabilities = self.log_probabilities(input_ids, legal_actions)
            position = int(torch.multinomial(log_probabilities.exp().cpu(), 1, generator=self.generator).item())
        return Decision(self.version, agent, time, tuple(input_ids), legal_actions,
                        legal_actions[position], float(log_probabilities[position]), baseline_version)

    def score_gradients(self, decision):
        if decision.version != self.version:
            raise ValueError("Decision uses a different replay policy version")
        self.activate(decision.agent)
        for wrapper in self.blocks.values():
            wrapper.start_capture()
        try:
            log_probabilities = self.log_probabilities(decision.input_ids, decision.legal_actions)
            position = decision.legal_actions.index(decision.action)
            score = log_probabilities[position]
            if abs(float(score.detach()) - decision.log_probability) > self.replay_tolerance:
                raise ValueError("Recorded and replayed action likelihoods differ")
            score.backward()
            return {block: wrapper.end_capture() for block, wrapper in self.blocks.items()}
        finally:
            for wrapper in self.blocks.values():
                wrapper.capture = False
                wrapper.parts = []
