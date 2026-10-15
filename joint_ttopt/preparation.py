import json
import re
from pathlib import Path

import numpy as np

from .accounting import derive_seed
from .confidence import trajectory_norm_envelope
from .storage import file_digest, json_value, write_json
from .trajectories import returns_to_go


class GenerativeActionTeacher:
    def __init__(self, checkpoint, revision=None, device_map="auto", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, revision=revision, device_map=device_map, torch_dtype=getattr(torch, dtype))
        self.model.eval()
        self.model.requires_grad_(False)
        self.metadata = {"checkpoint": checkpoint,
                         "revision": getattr(self.model.config, "_commit_hash", None) or revision,
                         "label_source": "base_model_legal_action_selection"}

    def select(self, input_ids, legal_actions, tokenizer, seed):
        import torch
        inputs = torch.tensor([input_ids], device=self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                          max_new_tokens=16, do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        text = tokenizer.decode(output[0, len(input_ids):], skip_special_tokens=True).strip()
        if not re.fullmatch(r"\d+", text) or int(text) not in legal_actions:
            raise ValueError("The entry teacher returned an invalid legal-action identifier")
        return int(text)


def collect_entry(environment, teacher, renderer, specification, output, seeds,
                   max_steps=4096, discount=0.99, fallback_spans=None):
    if specification.get("split") != "entry_training":
        raise ValueError("Action-head collection uses entry-training tasks and partner schedules")
    if not specification.get("task_id") or not specification.get("partner_schedule_id"):
        raise ValueError("Entry collection records task and partner-schedule identifiers")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Entry collection uses distinct episode seeds")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Entry collection requires an empty output directory")
    labels, episodes = [], []
    for index, seed in enumerate(seeds):
        native_seed = derive_seed(seed, 10, 0, index)
        frame = environment.reset(native_seed)
        history_actions, decisions = {}, []
        rewards = {agent: [] for agent in environment.agents}
        for tick in range(max_steps):
            if frame.terminated:
                break
            actions = {}
            for agent in sorted(frame.observations):
                legal = frame.legal_actions[agent]
                identifiers = renderer.encode(agent, environment.name, frame.task, frame.observations[agent],
                                               legal, frame.histories.get(agent, ()), frame.roles.get(agent, ""),
                                               history_actions.get(agent), tick)
                action = teacher.select(identifiers, tuple(sorted(legal)), renderer.tokenizer,
                                          derive_seed(seed, 11, 0, index, tick))
                if action not in legal:
                    raise ValueError("The entry teacher selected an illegal native action")
                actions[agent] = action
                decisions.append({"split": "entry_training", "task_id": specification["task_id"],
                                  "partner_schedule_id": specification["partner_schedule_id"],
                                  "episode": index, "seed": native_seed, "agent": agent, "time": tick,
                                  "input_ids": list(identifiers), "legal_actions": sorted(legal), "action": action,
                                  "observation": frame.observations[agent], "native_actions": legal})
            if not actions:
                raise ValueError("A nonterminal entry frame has no acting participant")
            frame = environment.step(actions)
            if set(frame.rewards) != set(rewards) or not np.isfinite(list(frame.rewards.values())).all():
                raise ValueError("Entry trajectories retain all native participant rewards")
            for agent in rewards:
                rewards[agent].append(frame.rewards[agent])
            history_actions.update(actions)
        if not frame.terminated:
            raise ValueError("Entry collection reached the safety horizon before episode completion")
        episode = {"seed": native_seed, "rewards": rewards, "native_metrics": frame.metrics,
                   "decisions": decisions, "complete": True}
        labels.extend(decisions)
        episodes.append(episode)
        write_json(root / f"episode_{index:05d}.json", episode)
    label_path = root / "action_labels.jsonl"
    label_path.write_text("".join(json.dumps(json_value(row), allow_nan=False) + "\n" for row in labels))
    normalization = entry_normalization(episodes, discount, fallback_spans)
    write_json(root / "normalization.json", normalization)
    write_json(root / "metadata.json", {"split": "entry_training", "task_id": specification["task_id"],
                                        "partner_schedule_id": specification["partner_schedule_id"],
                                        "benchmark": environment.name, "source_revision": environment.version,
                                        "teacher": teacher.metadata, "seeds": list(seeds), "episodes": len(episodes),
                                        "labels": len(labels), "label_sha256": file_digest(label_path)})
    return labels, episodes


def entry_normalization(episodes, discount=0.99, fallback_spans=None):
    if not episodes or any(not episode.get("complete") for episode in episodes):
        raise ValueError("Entry normalization uses complete native episodes")
    agents = set(episodes[0]["rewards"])
    if any(set(episode["rewards"]) != agents for episode in episodes):
        raise ValueError("Entry trajectories have different participant sets")
    result = {}
    for agent in agents:
        values = [float(returns_to_go(episode["rewards"][agent], discount)[0]) for episode in episodes]
        lower, upper = min(values), max(values)
        record = {"lower": lower, "upper": upper}
        if lower == upper:
            span = (fallback_spans or {}).get(agent, (fallback_spans or {}).get(str(agent)))
            if span is None or not np.isfinite(span) or span <= 0:
                raise ValueError("A constant entry return requires the declared benchmark return span")
            record["fallback_span"] = float(span)
        result[str(agent)] = record
    return {"split": "entry_training", "discount": discount, "episodes": len(episodes), "agents": result}


def declared_envelopes(specification):
    bounds = {}
    for agent, record in specification["agents"].items():
        bounds[str(agent)] = trajectory_norm_envelope(
            record["advantage_bound"], record["score_bounds"], record["state_bounds"],
            record["span"], record.get("regularization", 0.45), record["horizon"],
            record.get("discount", 0.99))
    return {"sample_bounds": bounds, "second_moments": specification.get("second_moments", {}),
            "reference_scale": specification["reference_scale"],
            "validity_record": specification["validity_record"],
            "construction": "discounted_trajectory_score_and_regularization_envelope"}
