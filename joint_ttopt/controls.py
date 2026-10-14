import numpy as np

from .allocation import rank_integer
from .calibration import joint_secant, select_step
from .factors import FactorMatrix, field_difference, joint_norm, validate_field
from .optimizer import Proposal


def fixed_ranks(coordinator, seed=0):
    ranks = dict.fromkeys(coordinator.states, 0)
    remaining = coordinator.total_rank
    usage = dict.fromkeys(coordinator.agent_caps, 0)
    generator = np.random.default_rng(seed)
    agents = [int(agent) for agent in generator.permutation(sorted(usage))]
    blocks = {agent: [key for key in sorted(ranks) if key[0] == agent] for agent in agents}
    blocks = {agent: [keys[index] for index in generator.permutation(len(keys))] for agent, keys in blocks.items()}
    while remaining:
        changed = False
        for agent in agents:
            if not remaining:
                break
            eligible = [key for key in blocks[agent] if ranks[key] < min(coordinator.states[key].shape)]
            if usage[agent] >= coordinator.agent_caps[agent] or not eligible:
                continue
            key = min(eligible, key=lambda candidate: ranks[candidate])
            ranks[key] += 1
            usage[agent] += 1
            remaining -= 1
            changed = True
        if not changed:
            break
    return ranks


def factor_update(coordinator, batch, calibration, ranks, seed=0):
    with coordinator.lock:
        if batch.version != coordinator.version or calibration.available_version > coordinator.version:
            raise ValueError("Factor update uses a different policy or calibration version")
        validate_field(batch.gradients, coordinator.states)
        if set(ranks) != set(coordinator.states):
            raise ValueError("Factor update has different adapted blocks")
        secant = joint_secant(coordinator.states, coordinator.preceding_states, batch.gradients,
                              coordinator.preceding_gradients, calibration.norm_threshold)
        common_step, penalty, scale = select_step(calibration, secant, coordinator.margin)
        steps = {agent: common_step * value for agent, value in coordinator.step_multipliers.items()}
        generator = np.random.default_rng(seed)
        states = {}
        for key in sorted(coordinator.states):
            state, gradient = coordinator.states[key], batch.gradients[key]
            rank = rank_integer(ranks[key])
            if rank > min(state.shape):
                raise ValueError("Assigned factor capacity exceeds its native block dimensions")
            if state.width == 0:
                left = np.zeros((state.shape[0], rank))
                right = np.linalg.qr(generator.normal(size=(state.shape[1], rank)), mode="reduced")[0].T
            else:
                if state.width != rank:
                    raise ValueError("Persistent factor capacity changed during the matched control")
                left, right = state.left, state.right
            left_gradient = gradient.left @ (gradient.right @ right.T)
            right_gradient = (left.T @ gradient.left) @ gradient.right
            states[key] = FactorMatrix(left + steps[key[0]] * left_gradient,
                                      right + steps[key[0]] * right_gradient)
        changes = field_difference(states, coordinator.states)
        proximal = sum(batch.gradients[key].inner(changes[key]) - changes[key].norm_squared() / (2 * steps[key[0]])
                       for key in states)
        diagnostics = {"update_rule": "simultaneous_factor_gradient_ascent", "joint_secant": secant,
                       "local_scale": scale, "common_step": common_step, "young_penalty": penalty,
                       "agent_steps": steps, "proximal_value": float(proximal),
                       "movement_norm": joint_norm(changes), "gradient_norm": joint_norm(batch.gradients),
                       "factor_entries": sum(state.storage for state in states.values()),
                       "sample_count": batch.sample_count,
                       "confidence": {"gap_lower_bound": None}, "allocation_refreshed": False}
        return Proposal(coordinator.version, states, dict(ranks), dict(batch.gradients), diagnostics)
