import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .storage import file_digest, record_digest, write_archive
from .records import FileLock
from .model_io import entry_metadata, renderer_settings, tokenizer_digest


def train_action_head(checkpoint, revision, training_data, output, renderer_options,
                      seed=0, epochs=5, learning_rate=0.0003, device="cpu", resume=None,
                      checkpoint_every=100):
    output = Path(output)
    with FileLock(output.with_suffix(".lock")):
        return _train_action_head(checkpoint, revision, training_data, output, renderer_options,
                                  seed, epochs, learning_rate, device, resume, checkpoint_every)


def save_training_state(path, value):
    import torch
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _train_action_head(checkpoint, revision, training_data, output, renderer_options,
                       seed, epochs, learning_rate, device, resume, checkpoint_every):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from .model_io import quantization_options
    if (type(epochs) is not int or epochs < 1 or type(checkpoint_every) is not int or checkpoint_every < 1
            or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid entry action-head training settings")
    state_path = output.with_suffix(".training.pt")
    if resume is None and (output.exists() or state_path.exists()):
        raise FileExistsError("Action-head output already exists; use --resume")
    collection = entry_metadata(training_data)
    teacher = collection.get("teacher", {})
    if teacher.get("checkpoint") is not None and teacher["checkpoint"] != checkpoint:
        raise ValueError("Entry labels use a different backbone")
    if revision is None:
        revision = teacher.get("revision")
    elif teacher.get("revision") is not None and revision != teacher["revision"]:
        raise ValueError("Entry labels use a different backbone revision")
    examples = [json.loads(line) for line in Path(training_data).read_text().splitlines() if line.strip()]
    if not examples or any(row.get("split") != "entry_training" for row in examples):
        raise ValueError("Action-head labels are from public entry-training episodes")
    for row in examples:
        legal = row["legal_actions"]
        if (not legal or any(type(action) is not int or action < 0 for action in legal)
                or len(set(legal)) != len(legal) or type(row["action"]) is not int or row["action"] not in legal):
            raise ValueError("Entry labels require distinct nonnegative legal-action identifiers")
    identifiers = {int(action) for row in examples for action in row["legal_actions"]}
    if min(identifiers) < 0:
        raise ValueError("Entry action identifiers are nonnegative")
    action_count = collection.get("action_count")
    action_count = max(identifiers) + 1 if action_count is None else action_count
    if type(action_count) is not int or action_count <= max(identifiers):
        raise ValueError("The declared action vocabulary omits an entry action")
    torch.manual_seed(seed)
    model_config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    revision = getattr(model_config, "_commit_hash", None) or revision
    renderer_options = renderer_settings(renderer_options, model_config.model_type)
    if collection.get("renderer") is not None:
        from .renderer import ObservationRenderer
        # Compare complete effective settings after native model-specific defaults.
        tokenizer_path = Path(training_data).parent / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.is_dir() else checkpoint,
                                                  **({} if tokenizer_path.is_dir() else {"revision": revision}))
        renderer_options = ObservationRenderer(tokenizer, **renderer_options).configuration()
        if renderer_options != collection["renderer"]:
            raise ValueError("Action-head rendering differs from the entry collection")
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    arguments = {"revision": revision, "torch_dtype": torch.float32}
    arguments.update(quantization_options(model_config))
    identity = {"checkpoint": checkpoint, "revision": revision, "labels": file_digest(training_data),
                "renderer": renderer_options, "tokenizer": tokenizer_digest(tokenizer),
                "seed": seed, "epochs": epochs, "learning_rate": learning_rate, "action_count": action_count}
    cursor = 0
    state = None
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["identity"] != record_digest(identity):
            raise ValueError("Action-head resume requires the original labels, model and training settings")
    backbone = AutoModel.from_pretrained(checkpoint, **arguments).to(device)
    backbone.eval()
    backbone.requires_grad_(False)
    head = nn.Linear(backbone.config.hidden_size, action_count).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    if state is not None:
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        cursor = state["cursor"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def save(complete=False):
        save_training_state(state_path, {"identity": record_digest(identity), "cursor": cursor,
                                         "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                                         "torch_rng": torch.get_rng_state(), "complete": complete,
                                         "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None})

    if resume is None:
        save()
    generator = np.random.default_rng(seed)
    for epoch in range(epochs):
        for position, index in enumerate(generator.permutation(len(examples))):
            if epoch * len(examples) + position < cursor:
                continue
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
            cursor = epoch * len(examples) + position + 1
            if cursor % checkpoint_every == 0:
                save()
    from .torch_layers import install_blocks
    blocks = install_blocks(backbone)
    metadata = {"checkpoint": checkpoint, "revision": getattr(backbone.config, "_commit_hash", None) or revision,
                "model_type": backbone.config.model_type,
                "adapted_blocks": sorted(blocks), "renderer": renderer_options,
                "training_data_sha256": file_digest(training_data), "training_seed": seed,
                "epochs": epochs, "learning_rate": learning_rate, "head_training": "supervised_legal_action_cross_entropy",
                "training_complete": True, "tokenizer_sha256": tokenizer_digest(tokenizer),
                "tokenizer_directory": output.with_suffix(".tokenizer").name,
                "training_seeds": sorted({row["seed"] for row in examples if "seed" in row}),
                "training_tasks": sorted({row["task_id"] for row in examples}),
                "training_partner_schedules": sorted({row["partner_schedule_id"] for row in examples})}
    tokenizer.save_pretrained(output.with_suffix(".tokenizer"))
    write_archive(output, {"weight": head.weight.detach().cpu().numpy(),
                           "bias": head.bias.detach().cpu().numpy()}, metadata)
    save(complete=True)
