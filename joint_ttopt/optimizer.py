from dataclasses import dataclass, field
from threading import RLock

import numpy as np

from .allocation import project, rank_integer
from .calibration import joint_secant, select_step
from .confidence import confidence_diagnostic, gradient_radius
from .factors import FactorMatrix, field_difference, joint_norm, validate_field


@dataclass(frozen=True)
class GradientBatch:
    version: int
    gradients: dict
    sample_count: int
    sample_norms: dict = field(default_factory=dict)
    approximation_errors: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Proposal:
    version: int
    states: dict
    ranks: dict
    gradients: dict
    diagnostics: dict


class Coordinator:
    def __init__(self, shapes, total_rank, agent_caps, margin=0.10,
                 error_budget=0.05, step_multipliers=None):
        self.total_rank = rank_integer(total_rank)
        self.agent_caps = {int(agent): rank_integer(value) for agent, value in agent_caps.items()}
        if not shapes or set(agent for agent, block in shapes) != set(self.agent_caps):
            raise ValueError("Each updating agent has a declared block set and rank cap")
        if not 0 < margin < 1 or not 0 < error_budget < 1:
            raise ValueError("Invalid step margin or error budget")
        self.margin = float(margin)
        self.error_budget = float(error_budget)
        self.step_multipliers = (dict.fromkeys(self.agent_caps, 1.0) if step_multipliers is None
                                 else dict(step_multipliers))
        if set(self.step_multipliers) != set(self.agent_caps):
            raise ValueError("Step multipliers have different participants")
        if any(not np.isfinite(value) or not 0 < value <= 1
               for value in self.step_multipliers.values()):
            raise ValueError("Step multipliers lie in the interval (0, 1]")
        self.states = {key: FactorMatrix.zeros(*shape) for key, shape in shapes.items()}
        self.ranks = dict.fromkeys(shapes, 0)
        self.version = 0
        self.preceding_states = None
        self.preceding_gradients = None
        self.lock = RLock()

    def snapshot(self):
        with self.lock:
            return self.version, dict(self.states)

    def prepare(self, batch, calibration, block_caps=None, reset_incumbent=False):
        with self.lock:
            if batch.version != self.version:
                raise ValueError("Gradient batch uses a different policy version")
            if calibration.available_version > self.version:
                raise ValueError("Calibration was unavailable at this policy version")
            validate_field(batch.gradients, self.states)
            if isinstance(batch.sample_count, bool) or int(batch.sample_count) != batch.sample_count or batch.sample_count < 1:
                raise ValueError("Batch size counts complete independent trajectories")
            if set(calibration.sample_bounds) != set(self.agent_caps):
                raise ValueError("Calibration has different updating agents")
            if calibration.failure_probability >= self.error_budget:
                raise ValueError("Calibration consumes the round error budget")
            secant = joint_secant(self.states, self.preceding_states, batch.gradients,
                                  self.preceding_gradients, calibration.norm_threshold)
            common_step, penalty, scale = select_step(calibration, secant, self.margin)
            steps = {agent: common_step * multiplier
                     for agent, multiplier in self.step_multipliers.items()}
            anchors = ({key: FactorMatrix.zeros(*state.shape) for key, state in self.states.items()}
                       if reset_incumbent else self.states)
            proposals = {key: state.proposal(batch.gradients[key], steps[key[0]])
                         for key, state in anchors.items()}
            next_states, allocation = project(proposals, self.total_rank, self.agent_caps, steps)
            if block_caps is not None:
                if set(block_caps) != set(self.states):
                    raise ValueError("Block rank constraints have different adapted blocks")
                spectra = {key: matrix.spectrum() for key, matrix in proposals.items()}
                constrained = {key: min(rank_integer(block_caps[key]), len(spectra[key].values),
                                         int(np.sum(spectra[key].values > 0))) for key in spectra}
                if sum(constrained.values()) > self.total_rank:
                    raise ValueError("Block constraints exceed the shared rank budget")
                for agent, cap in self.agent_caps.items():
                    if sum(rank for key, rank in constrained.items() if key[0] == agent) > cap:
                        raise ValueError("Block constraints exceed an agent cap")
                next_states = {key: spectra[key].truncate(rank) for key, rank in constrained.items()}
                allocation["ranks"] = constrained
                allocation["agent_ranks"] = {agent: sum(rank for key, rank in constrained.items() if key[0] == agent)
                                               for agent in self.agent_caps}
                total_value = sum(np.sum(spectrum.values ** 2) / (2 * steps[key[0]]) for key, spectrum in spectra.items())
                retained = sum(np.sum(spectra[key].values[:rank] ** 2) / (2 * steps[key[0]])
                               for key, rank in constrained.items())
                allocation["retained_value"] = float(retained)
                allocation["excluded_value"] = float(total_value - retained)
                allocation["block_margins"] = {
                    key: float(spectrum.values[constrained[key] - 1] - spectrum.values[constrained[key]])
                    if 0 < constrained[key] < len(spectrum.values) else None for key, spectrum in spectra.items()}
            changes = field_difference(next_states, self.states)
            proximal = sum(batch.gradients[key].inner(changes[key])
                           - changes[key].norm_squared() / (2 * steps[key[0]])
                           for key in self.states)
            agent_level = (self.error_budget - calibration.failure_probability) / len(self.agent_caps)
            levels = dict.fromkeys(self.agent_caps, agent_level)
            radii = {agent: gradient_radius(calibration.sample_bounds[agent], batch.sample_count,
                                             levels[agent], calibration.second_moments.get(agent),
                                             batch.approximation_errors.get(agent, 0.0))
                     for agent in self.agent_caps}
            penalties = {agent: 1 / steps[agent] - scale for agent in self.agent_caps}
            certificate = confidence_diagnostic(radii, penalties, proximal, levels,
                                                  calibration.failure_probability, self.error_budget)
            bounds_valid = True
            observed_norms = dict(batch.sample_norms)
            if batch.sample_count == 1:
                for agent in self.agent_caps:
                    observed_norms.setdefault(agent, [joint_norm({key: value for key, value in batch.gradients.items()
                                                                  if key[0] == agent})])
            for agent, norms in observed_norms.items():
                if agent not in self.agent_caps or len(norms) != batch.sample_count:
                    raise ValueError("Trajectory norm records have incompatible dimensions")
                if not np.isfinite(norms).all() or np.any(np.asarray(norms) < 0):
                    raise ValueError("Trajectory norm records are finite and nonnegative")
                bounds_valid &= max(norms) <= calibration.sample_bounds[agent] * (1 + 1e-10) + 1e-12
            certificate["observed_samples_within_envelope"] = (bool(bounds_valid)
                                                               if set(observed_norms) == set(self.agent_caps) else None)
            if not bounds_valid:
                certificate["gap_lower_bound"] = None
            if reset_incumbent:
                certificate["gap_lower_bound"] = None
            ranks = allocation["ranks"]
            diagnostics = {"joint_secant": secant, "local_scale": scale,
                           "common_step": common_step, "young_penalty": penalty,
                           "agent_steps": steps, "proximal_value": float(proximal),
                           "movement_norm": joint_norm(changes),
                           "gradient_norm": joint_norm(batch.gradients),
                           "excluded_proximal_value": allocation["excluded_value"],
                           "retained_proximal_value": allocation["retained_value"],
                           "eligible_modes": allocation["eligible_modes"],
                           "agent_ranks": allocation["agent_ranks"],
                           "block_margins": allocation["block_margins"],
                           "factor_entries": sum(state.storage for state in next_states.values()),
                           "working_factor_entries": sum(state.storage for state in proposals.values()),
                           "rank_changes": sum(ranks[key] != self.ranks[key] for key in ranks),
                           "allocation_refreshed": block_caps is None,
                           "reset_incumbent_after_gradient": bool(reset_incumbent),
                           "sample_count": batch.sample_count, "confidence": certificate}
            return Proposal(self.version, next_states, ranks, dict(batch.gradients), diagnostics)

    def commit(self, proposal):
        with self.lock:
            if proposal.version != self.version:
                return {"committed": False, "reason": "stale_policy_version", "version": self.version}
            try:
                validate_field(proposal.states, self.states)
                validate_field(proposal.gradients, self.states)
                if set(proposal.ranks) != set(self.states):
                    raise ValueError("Candidate rank records have different blocks")
                ranks = {key: rank_integer(value) for key, value in proposal.ranks.items()}
                for key, state in proposal.states.items():
                    if state.width != ranks[key] or state.width > min(state.shape):
                        raise ValueError("Candidate factor width disagrees with its assigned rank")
                    if not np.isfinite(state.left).all() or not np.isfinite(state.right).all():
                        raise ValueError("Candidate state contains nonfinite factors")
                if sum(ranks.values()) > self.total_rank:
                    raise ValueError("Candidate exceeds the shared rank budget")
                for agent, cap in self.agent_caps.items():
                    if sum(value for key, value in ranks.items() if key[0] == agent) > cap:
                        raise ValueError("Candidate exceeds an agent rank cap")
            except (ValueError, TypeError, AttributeError) as error:
                return {"committed": False, "reason": str(error), "version": self.version}
            self.preceding_states = dict(self.states)
            self.preceding_gradients = dict(proposal.gradients)
            self.states = dict(proposal.states)
            self.ranks = ranks
            self.version += 1
            return dict(proposal.diagnostics, committed=True, version=self.version,
                        block_ranks=dict(self.ranks))

    def step(self, batch, calibration):
        return self.commit(self.prepare(batch, calibration))

    def clear_states(self):
        with self.lock:
            self.states = {key: FactorMatrix.zeros(*state.shape) for key, state in self.states.items()}
            self.ranks = dict.fromkeys(self.states, 0)
            self.preceding_states = None
            self.preceding_gradients = None
            self.version += 1
