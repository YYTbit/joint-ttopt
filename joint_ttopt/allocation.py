import numpy as np


def rank_integer(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError("Rank constraints are finite nonnegative integers")
    if int(value) != value or value < 0:
        raise ValueError("Rank constraints are finite nonnegative integers")
    return int(value)


def allocate(spectra, total_rank, agent_caps, steps):
    total_rank = rank_integer(total_rank)
    agent_caps = {int(agent): rank_integer(cap) for agent, cap in agent_caps.items()}
    if set(steps) != set(agent_caps):
        raise ValueError("Steps and agent caps have different participants")
    if any(not np.isfinite(value) or value <= 0 for value in steps.values()):
        raise ValueError("Agent steps are finite and positive")
    if any(agent not in agent_caps for agent, block in spectra):
        raise ValueError("A spectral block has no agent cap")
    modes = []
    for (agent, block), spectrum in spectra.items():
        for index, singular in enumerate(spectrum.values):
            if singular > 0:
                # This contribution ranks each mode under the quadratic proximal objective.
                value = float(singular ** 2 / (2 * steps[agent]))
                if not np.isfinite(value):
                    raise ValueError("A spectral mode value is nonfinite")
                modes.append((-value, agent, block, index))
    modes.sort()
    ranks = dict.fromkeys(spectra, 0)
    used = dict.fromkeys(agent_caps, 0)
    selected = []
    excluded = []
    for negative_value, agent, block, index in modes:
        if len(selected) < total_rank and used[agent] < agent_caps[agent]:
            # A block retains leading singular modes, so selected indices must form a prefix.
            if index != ranks[(agent, block)]:
                raise ValueError("Selected singular modes do not form a prefix")
            ranks[(agent, block)] += 1
            used[agent] += 1
            selected.append((agent, block, index, -negative_value))
        else:
            excluded.append((agent, block, index, -negative_value))
    margins = {}
    for key, spectrum in spectra.items():
        rank = ranks[key]
        if 0 < rank < len(spectrum.values):
            margins[key] = float(spectrum.values[rank - 1] - spectrum.values[rank])
        else:
            margins[key] = None
    return {"ranks": ranks, "agent_ranks": used, "selected": selected,
            "retained_value": float(sum(item[3] for item in selected)),
            "excluded_value": float(sum(item[3] for item in excluded)),
            "eligible_modes": len(modes), "block_margins": margins}


def project(proposals, total_rank, agent_caps, steps):
    spectra = {key: matrix.spectrum() for key, matrix in proposals.items()}
    allocation = allocate(spectra, total_rank, agent_caps, steps)
    states = {key: spectrum.truncate(allocation["ranks"][key])
              for key, spectrum in spectra.items()}
    return states, allocation
