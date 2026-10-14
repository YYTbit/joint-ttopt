import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .allocation import rank_integer
from .factors import FactorMatrix, validate_field
from .optimizer import Coordinator


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {(str(key[0]) + "::" + key[1] if isinstance(key, tuple) else str(key)): json_value(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_value(value), indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="record-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def pack_field(arrays, metadata, name, field):
    metadata[name] = []
    for index, ((agent, block), matrix) in enumerate(sorted(field.items())):
        prefix = name + "_" + str(index)
        arrays[prefix + "_left"] = matrix.left
        arrays[prefix + "_right"] = matrix.right
        metadata[name].append([agent, block, prefix])


def unpack_field(archive, metadata, name):
    result = {}
    for agent, block, prefix in metadata[name]:
        key = (int(agent), str(block))
        if key in result:
            raise ValueError("Factor archive contains a duplicate block")
        result[key] = FactorMatrix(archive[prefix + "_left"], archive[prefix + "_right"])
    return result


def write_archive(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays, metadata=np.asarray(json.dumps(json_value(metadata), allow_nan=False)))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="state-", suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_factors(path, field, metadata=None):
    arrays, record = {}, dict(metadata or {})
    record["format"] = "joint_ttopt_factors_v1"
    pack_field(arrays, record, "factors", field)
    write_archive(path, arrays, record)


def load_factors(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != "joint_ttopt_factors_v1":
            raise ValueError("Unsupported factor archive format")
        return unpack_field(archive, metadata, "factors"), metadata


def save_checkpoint(path, coordinator, extra=None):
    with coordinator.lock:
        arrays = {}
        metadata = {"format": "joint_ttopt_checkpoint_v1", "version": coordinator.version,
                    "total_rank": coordinator.total_rank, "agent_caps": coordinator.agent_caps,
                    "margin": coordinator.margin, "error_budget": coordinator.error_budget,
                    "step_multipliers": coordinator.step_multipliers,
                    "ranks": [[agent, block, value] for (agent, block), value in sorted(coordinator.ranks.items())],
                    "extra": extra or {}}
        pack_field(arrays, metadata, "states", coordinator.states)
        if coordinator.preceding_states is not None:
            pack_field(arrays, metadata, "preceding_states", coordinator.preceding_states)
            pack_field(arrays, metadata, "preceding_gradients", coordinator.preceding_gradients)
        write_archive(path, arrays, metadata)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format") != "joint_ttopt_checkpoint_v1":
            raise ValueError("Unsupported checkpoint format")
        states = unpack_field(archive, metadata, "states")
        coordinator = Coordinator({key: state.shape for key, state in states.items()}, metadata["total_rank"],
                                  {int(agent): cap for agent, cap in metadata["agent_caps"].items()},
                                  metadata["margin"], metadata["error_budget"],
                                  {int(agent): value for agent, value in metadata["step_multipliers"].items()})
        ranks = {(int(agent), block): rank_integer(value) for agent, block, value in metadata["ranks"]}
        if set(ranks) != set(states) or any(state.width != ranks[key] for key, state in states.items()):
            raise ValueError("Checkpoint rank records differ from its states")
        if sum(ranks.values()) > coordinator.total_rank:
            raise ValueError("Checkpoint exceeds the shared rank budget")
        for agent, cap in coordinator.agent_caps.items():
            if sum(rank for key, rank in ranks.items() if key[0] == agent) > cap:
                raise ValueError("Checkpoint exceeds an agent rank cap")
        coordinator.states = states
        coordinator.ranks = ranks
        coordinator.version = rank_integer(metadata["version"])
        if "preceding_states" in metadata:
            coordinator.preceding_states = unpack_field(archive, metadata, "preceding_states")
            coordinator.preceding_gradients = unpack_field(archive, metadata, "preceding_gradients")
            validate_field(coordinator.preceding_states, states)
            validate_field(coordinator.preceding_gradients, states)
        return coordinator, metadata.get("extra", {})


def load_calibration(path):
    value = json.loads(Path(path).read_text())
    for name in ["sample_bounds", "second_moments"]:
        value[name] = {int(agent): bound for agent, bound in value.get(name, {}).items()}
    return Calibration(**value)
