import json
from pathlib import Path

import numpy as np

from .storage import file_digest, write_archive


def train_action_head(checkpoint, revision, training_data, output, renderer_options,
                      seed=0, epochs=5, learning_rate=0.0003, device="cpu"):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, Mxfp4Config
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Invalid entry action-head training settings")
    examples = [json.loads(line) for line in Path(training_data).read_text().splitlines() if line.strip()]
    if not examples or any(row.get("split") != "entry_training" for row in examples):
        raise ValueError("Action-head labels are from public entry-training episodes")
    identifiers = {int(action) for row in examples for action in row["legal_actions"]}
    if min(identifiers) < 0:
        raise ValueError("Entry action identifiers are nonnegative")
    torch.manual_seed(seed)
    model_config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    arguments = {"revision": revision, "torch_dtype": torch.float32, "attn_implementation": "eager"}
    if model_config.model_type == "gpt_oss":
        arguments["quantization_config"] = Mxfp4Config(dequantize=True)
    backbone = AutoModel.from_pretrained(checkpoint, **arguments).to(device)
    backbone.eval()
    backbone.requires_grad_(False)
    head = nn.Linear(backbone.config.hidden_size, max(identifiers) + 1).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        for index in generator.permutation(len(examples)):
            row = examples[index]
            legal = tuple(int(value) for value in row["legal_actions"])
            action = int(row["action"])
            if len(set(legal)) != len(legal) or action not in legal:
                raise ValueError("Entry label is absent from its legal action set")
            inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
            with torch.no_grad():
                hidden = backbone(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).last_hidden_state[:, -1]
            logits = head(hidden)[0, list(legal)]
            loss = -torch.log_softmax(logits, dim=-1)[legal.index(action)]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    from .torch_layers import install_blocks
    blocks = install_blocks(backbone)
    metadata = {"checkpoint": checkpoint, "revision": getattr(backbone.config, "_commit_hash", None) or revision,
                "model_type": backbone.config.model_type,
                "adapted_blocks": sorted(blocks), "renderer": renderer_options,
                "training_data_sha256": file_digest(training_data), "training_seed": seed,
                "epochs": epochs, "learning_rate": learning_rate, "head_training": "supervised_legal_action_cross_entropy",
                "training_tasks": sorted({row["task_id"] for row in examples}),
                "training_partner_schedules": sorted({row["partner_schedule_id"] for row in examples})}
    write_archive(output, {"weight": head.weight.detach().cpu().numpy(),
                           "bias": head.bias.detach().cpu().numpy()}, metadata)
