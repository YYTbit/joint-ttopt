import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .optimizer import Coordinator
from .storage import json_value, load_checkpoint, save_checkpoint, write_json
from .trajectories import HistoryBaseline, Trajectory, estimate_gradients, returns_to_go


def collect_trajectory(environment, policy, seed, identifier, baseline_version, max_steps):
    frame = environment.reset(seed)
    expected_agents = set(environment.agents)
    decisions, reward_rows = [], {agent: [] for agent in environment.agents}
    preceding_actions = {}
    for tick in range(max_steps):
        if frame.terminated:
            break
        if set(frame.observations) != set(frame.legal_actions) or not set(frame.observations).issubset(expected_agents):
            raise ValueError("Native benchmark frame has inconsistent acting participants")
        if not frame.observations:
            raise ValueError("A nonterminal benchmark frame has no acting participant")
        actions = {}
        for agent in sorted(frame.observations):
            legal = frame.legal_actions[agent]
            input_ids = policy.renderer.encode(agent, environment.name, frame.task,
                                                 frame.observations[agent], legal,
                                                 frame.histories.get(agent, ()), frame.roles.get(agent, ""),
                                                 preceding_actions.get(agent), tick)
            decision = policy.sample(agent, tick, input_ids, tuple(legal), baseline_version)
            decisions.append(decision)
            actions[agent] = decision.action
        frame = environment.step(actions)
        if set(frame.rewards) != expected_agents or not np.isfinite(list(frame.rewards.values())).all():
            raise ValueError("Native benchmark frame omitted a finite participant reward")
        for agent in expected_agents:
            reward_rows[agent].append(frame.rewards[agent])
        preceding_actions.update(actions)
    if not frame.terminated:
        raise ValueError("Benchmark trajectory reached the safety horizon before completion")
    trajectory = Trajectory(policy.version, str(identifier), tuple(decisions), reward_rows)
    return trajectory, frame.metrics


def deployment_signature(config, policy, environment, calibration_digest, scales):
    record = {"algorithm": config, "checkpoint": policy.metadata.get("checkpoint"),
              "revision": policy.metadata.get("revision"), "head_sha256": policy.metadata.get("head_sha256"),
              "environment": environment.name, "environment_version": environment.version,
              "calibration_sha256": calibration_digest,
              "reward_scales": {str(agent): scale.__dict__ for agent, scale in scales.items()},
              "blocks": {name: wrapper.matrix_shape for name, wrapper in policy.blocks.items()}}
    return hashlib.sha256(json.dumps(json_value(record), sort_keys=True).encode()).hexdigest(), record


def run_deployment(environment, policy, calibration, calibration_digest, scales,
                   config, output, seed=0, method="joint", checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not checkpoint and any(output.iterdir()):
        raise ValueError("A new deployment requires an empty output directory")
    all_agents = tuple(environment.agents)
    updating = tuple(all_agents if config.get("updating_agents") is None else config["updating_agents"])
    if not set(updating).issubset(all_agents) or not updating or len(set(updating)) != len(updating):
        raise ValueError("Updating set is a nonempty subset of benchmark participants")
    if set(scales) != set(updating):
        raise ValueError("Entry normalization has different updating agents")
    budget = 8 * len(policy.blocks) if config.get("total_rank") is None else config["total_rank"]
    cap = int(np.ceil(config.get("agent_cap_multiplier", 3.0) * budget / len(updating)))
    caps = {agent: cap for agent in updating}
    if method == "equal_share":
        caps = {agent: budget // len(updating) + int(index < budget % len(updating))
                for index, agent in enumerate(sorted(updating))}
    coordinator = Coordinator(policy.shapes(updating), budget, caps,
                              config.get("step_margin", 0.10), config.get("error_budget", 0.05))
    baseline = HistoryBaseline(all_agents, config.get("baseline_window", 8), config.get("discount", 0.99))
    config = dict(config)
    if config.get("comparison_signature") is None:
        comparison = {"benchmark": environment.name, "environment_version": environment.version,
                      "task": config.get("task_id"), "partner_schedule": config.get("partner_schedule_id"),
                      "environment_configuration": config.get("environment_configuration_sha256"),
                      "checkpoint": policy.metadata.get("checkpoint"), "revision": policy.metadata.get("revision"),
                      "head_sha256": policy.metadata.get("head_sha256"),
                      "normalization_sha256": config.get("normalization_sha256"), "total_rank": budget,
                      "updating_agents": updating, "discount": config.get("discount", 0.99)}
        config["comparison_signature"] = hashlib.sha256(json.dumps(json_value(comparison), sort_keys=True).encode()).hexdigest()
    signature, metadata = deployment_signature(dict(config, method=method, seed=seed), policy,
                                                environment, calibration_digest, scales)
    completed = 0
    records = []
    if checkpoint:
        coordinator, extra = load_checkpoint(checkpoint)
        if extra.get("signature") != signature:
            raise ValueError("Checkpoint belongs to a different deployment configuration")
        baseline = HistoryBaseline.restore(extra["baseline"])
        completed = int(extra["episodes"])
        import torch
        policy.generator.set_state(torch.tensor(extra["policy_rng"], dtype=torch.uint8))
    write_json(output / "metadata.json", dict(metadata, signature=signature, method=method, seed=seed))
    horizon = int(config.get("episodes", 50))
    batch_size = int(config.get("trajectories_per_update", 1))
    if horizon < 1 or batch_size < 1 or horizon % batch_size:
        raise ValueError("Deployment horizon is a positive multiple of complete trajectories per update")
    intervals = int(config.get("allocation_interval", 1))
    if intervals < 1:
        raise ValueError("Allocation interval is a positive integer")
    for episode in range(completed, horizon, batch_size):
        version, states = coordinator.snapshot()
        policy_states = dict(states)
        for agent in set(all_agents) - set(updating):
            from .factors import FactorMatrix
            policy_states.update({(agent, block): FactorMatrix.zeros(*wrapper.matrix_shape)
                                  for block, wrapper in policy.blocks.items()})
        policy.set_states(policy_states, version)
        baseline_version, baseline_values = baseline.snapshot()
        trajectories, scores = [], []
        start = time.perf_counter()
        for offset in range(batch_size):
            trajectory, metrics = collect_trajectory(environment, policy, seed + episode + offset,
                                                       str(seed) + ":" + str(episode + offset),
                                                       baseline_version, int(config.get("max_episode_steps", 4096)))
            trajectories.append(trajectory)
            scores.append(metrics)
        interaction_time = time.perf_counter() - start
        start = time.perf_counter()
        if method == "frozen":
            update = {"committed": False, "reason": "frozen_policy", "version": version}
        else:
            gradients = estimate_gradients(trajectories, policy, states, scales,
                                             dict.fromkeys(updating, config.get("regularization", 0.45)),
                                             baseline_values, baseline_version, config.get("discount", 0.99),
                                             method == "shared_feedback")
            block_caps = dict(coordinator.ranks) if episode > 0 and episode % intervals else None
            if method == "rank_matched":
                from .controls import factor_update, fixed_ranks
                proposal = factor_update(coordinator, gradients, calibration, fixed_ranks(coordinator, seed), seed)
            else:
                proposal = coordinator.prepare(gradients, calibration, block_caps, method == "reset")
            update = coordinator.commit(proposal)
            if not update["committed"]:
                raise RuntimeError("Joint commit was rejected before advancing the deployment round")
        update_time = time.perf_counter() - start
        if method == "shared_feedback":
            baseline_trajectories = [Trajectory(trajectory.version, trajectory.identifier, trajectory.decisions,
                                                dict.fromkeys(all_agents, np.mean(list(trajectory.rewards.values()), axis=0)))
                                     for trajectory in trajectories]
            baseline.observe(baseline_trajectories)
        else:
            baseline.observe(trajectories)
        for offset, (trajectory, metrics) in enumerate(zip(trajectories, scores)):
            row = {"data_kind": getattr(environment, "data_kind", "measured_run"),
                   "benchmark": environment.name, "task": config.get("task_id"),
                   "backbone": policy.metadata.get("checkpoint"),
                   "comparison_signature": config.get("comparison_signature"),
                   "signature": signature, "method": method, "seed": seed,
                   "episode": episode + offset + 1, "policy_version": trajectory.version,
                   "native_metrics": metrics,
                   "discounted_returns": {agent: float(returns_to_go(values, config.get("discount", 0.99))[0])
                                            for agent, values in trajectory.rewards.items()},
                   "interaction_seconds": interaction_time / batch_size,
                   "update_seconds": update_time / batch_size,
                   "joint_trajectory_steps": len(next(iter(trajectory.rewards.values()))),
                   "update": update}
            records.append(row)
            write_json(output / ("episode_" + str(episode + offset + 1).zfill(5) + ".json"), row)
            if config.get("record_trajectories", True):
                write_json(output / ("trajectory_" + str(episode + offset + 1).zfill(5) + ".json"), trajectory.__dict__)
        extra = {"signature": signature, "episodes": episode + batch_size,
                 "baseline": baseline.record(), "policy_rng": policy.generator.get_state().tolist()}
        save_checkpoint(output / "state.npz", coordinator, extra)
    return records
