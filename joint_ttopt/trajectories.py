from collections import deque
from dataclasses import dataclass

import numpy as np

from .factors import FactorMatrix, joint_norm
from .optimizer import GradientBatch


@dataclass(frozen=True)
class Decision:
    version: int
    agent: int
    time: int
    input_ids: tuple
    legal_actions: tuple
    action: int
    log_probability: float
    baseline_version: int = 0

    def __post_init__(self):
        if self.version < 0 or self.time < 0 or self.agent < 0:
            raise ValueError("Decision identifiers are nonnegative")
        if not self.input_ids or any(int(value) != value or value < 0 for value in self.input_ids):
            raise ValueError("Decision input contains invalid token identifiers")
        if not self.legal_actions or len(set(self.legal_actions)) != len(self.legal_actions):
            raise ValueError("Legal action identifiers are nonempty and distinct")
        if any(int(value) != value or value < 0 for value in self.legal_actions):
            raise ValueError("Legal action identifiers are nonnegative integers")
        if self.action not in self.legal_actions or not np.isfinite(self.log_probability):
            raise ValueError("Decision has an invalid action or log probability")
        if self.log_probability > 1e-6:
            raise ValueError("Action log probability exceeds zero")


@dataclass(frozen=True)
class Trajectory:
    version: int
    identifier: str
    decisions: tuple
    rewards: dict
    complete: bool = True

    def __post_init__(self):
        if not self.complete or not self.rewards:
            raise ValueError("Gradient samples are complete joint trajectories")
        lengths = {len(values) for values in self.rewards.values()}
        if len(lengths) != 1 or min(lengths) < 1:
            raise ValueError("Agent reward arrays have a common positive horizon")
        if any(not np.isfinite(values).all() for values in self.rewards.values()):
            raise ValueError("Trajectory rewards contain nonfinite values")
        seen = set()
        for decision in self.decisions:
            if decision.version != self.version or decision.agent not in self.rewards:
                raise ValueError("Trajectory decisions have inconsistent agents or versions")
            if decision.time >= min(lengths) or (decision.agent, decision.time) in seen:
                raise ValueError("Trajectory decision time is invalid or duplicated")
            seen.add((decision.agent, decision.time))


def returns_to_go(rewards, discount):
    if not np.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("Discount is in the interval [0, 1]")
    rewards = np.asarray(rewards, dtype=float)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("Expected finite one-dimensional rewards")
    result = np.empty_like(rewards)
    accumulated = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        accumulated = rewards[index] + discount * accumulated
        result[index] = accumulated
    return result


class HistoryBaseline:
    def __init__(self, agents, window=8, discount=0.99):
        if isinstance(window, bool) or int(window) != window or window < 1:
            raise ValueError("Baseline history window is a positive integer")
        if not 0 <= discount <= 1:
            raise ValueError("Invalid baseline discount")
        self.window = int(window)
        self.discount = float(discount)
        self.history = {agent: deque(maxlen=self.window) for agent in agents}
        self.version = 0

    def snapshot(self):
        result = {}
        for agent, records in self.history.items():
            for time in range(max((len(record) for record in records), default=0)):
                values = [record[time] for record in records if time < len(record)]
                result[(agent, time)] = float(np.mean(values))
        return self.version, result

    def observe(self, trajectories):
        for trajectory in trajectories:
            if set(trajectory.rewards) != set(self.history):
                raise ValueError("Baseline and trajectory have different participants")
            for agent in self.history:
                self.history[agent].append(returns_to_go(trajectory.rewards[agent], self.discount).tolist())
        self.version += 1

    def record(self):
        return {"version": self.version, "window": self.window, "discount": self.discount,
                "history": {str(agent): list(records) for agent, records in self.history.items()}}

    @classmethod
    def restore(cls, record):
        result = cls([int(agent) for agent in record["history"]], record["window"], record["discount"])
        for agent, values in record["history"].items():
            result.history[int(agent)].extend(values)
        result.version = int(record["version"])
        return result


def estimate_gradients(trajectories, policy, states, scales, regularization,
                       baseline, baseline_version, discount=0.99, shared_feedback=False):
    if not trajectories or len({trajectory.identifier for trajectory in trajectories}) != len(trajectories):
        raise ValueError("Gradient batch requires distinct complete trajectories")
    versions = {trajectory.version for trajectory in trajectories}
    if versions != {policy.version}:
        raise ValueError("Replay uses the incumbent trajectory policy version")
    agents = set(agent for agent, block in states)
    if set(scales) != agents or set(regularization) != agents:
        raise ValueError("Gradient utility fields have different updating agents")
    if any(not np.isfinite(value) or value < 0 for value in regularization.values()):
        raise ValueError("Regularization coefficients are finite and nonnegative")
    mean = {key: FactorMatrix.zeros(*state.shape) for key, state in states.items()}
    sample_norms = {agent: [] for agent in agents}
    count = len(trajectories)
    for trajectory in trajectories:
        rewards = trajectory.rewards
        if not agents.issubset(rewards):
            raise ValueError("Trajectory lacks an updating agent's reward")
        if shared_feedback:
            team_reward = np.mean([rewards[agent] for agent in sorted(rewards)], axis=0)
            returns = dict.fromkeys(agents, returns_to_go(team_reward, discount))
        else:
            returns = {agent: returns_to_go(rewards[agent], discount) for agent in agents}
        sample = {key: state.scaled(-regularization[key[0]]) for key, state in states.items()}
        for decision in trajectory.decisions:
            if decision.baseline_version != baseline_version:
                raise ValueError("Trajectory was sampled with a different baseline snapshot")
            if decision.agent not in agents:
                continue
            score = policy.score_gradients(decision)
            expected = {block: state for (agent, block), state in states.items() if agent == decision.agent}
            if set(score) != set(expected) or any(score[block].shape != expected[block].shape for block in score):
                raise ValueError("Policy score omitted or changed an adapted block")
            advantage = returns[decision.agent][decision.time] - baseline.get((decision.agent, decision.time), 0.0)
            coefficient = discount ** decision.time * advantage / scales[decision.agent].span
            for block, gradient in score.items():
                key = (decision.agent, block)
                sample[key] = sample[key].plus(gradient, coefficient)
        for agent in agents:
            sample_norms[agent].append(joint_norm({key: value for key, value in sample.items() if key[0] == agent}))
        for key in states:
            mean[key] = mean[key].plus(sample[key], 1 / count)
    return GradientBatch(policy.version, mean, count, sample_norms)
