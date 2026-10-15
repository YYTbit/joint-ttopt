import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from joint_ttopt.calibration import Calibration, paired_second_moment
from joint_ttopt.confidence import RewardScale
from joint_ttopt.controls import factor_update, fixed_ranks
from joint_ttopt.deployment import run_deployment
from joint_ttopt.environments import Frame, serve
from joint_ttopt.factors import FactorMatrix
from joint_ttopt.optimizer import Coordinator, GradientBatch
from joint_ttopt.probes import load_probe_calibration, record_probes, unit_direction
from joint_ttopt.renderer import ObservationRenderer
from joint_ttopt.statistics import (aggregate_games, apply_anchors, build_anchors, deployment_metrics,
                                   normalized_gain, paired_bootstrap)
from joint_ttopt.storage import load_checkpoint
from joint_ttopt.trajectories import Decision


class TokenizerFixture:
    def get_vocab(self):
        return {"<state>": 0}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return [1, 2, 3]


class EnvironmentFixture:
    agents = (0,)
    name = "software_fixture"
    version = "fixture-v1"
    data_kind = "software_fixture"

    def reset(self, seed):
        return Frame({0: {"value": seed % 2}}, {0: {0: "action"}}, {0: 0.0}, False, "fixture", {}, {}, {})

    def step(self, actions):
        assert actions == {0: 0}
        return Frame({}, {}, {0: 1.0}, True, "fixture", {}, {}, {"fixture_metric": 1.0})

    def close(self):
        pass


class PolicyFixture:
    def __init__(self):
        self.generator = torch.Generator().manual_seed(0)
        self.blocks = {"a": SimpleNamespace(matrix_shape=(1, 1))}
        self.renderer = SimpleNamespace(encode=lambda *args: (1,))
        self.metadata = {"checkpoint": "software_fixture", "head_sha256": "fixture"}

    def shapes(self, agents):
        return {(agent, "a"): (1, 1) for agent in agents}

    def set_states(self, states, version):
        self.states, self.version = states, version

    def sample(self, agent, tick, input_ids, legal, baseline_version):
        return Decision(self.version, agent, tick, input_ids, legal, 0, 0.0, baseline_version)

    def score_gradients(self, decision):
        return {"a": FactorMatrix.from_dense(np.ones((1, 1)))}


def test_renderer_preserves_local_features_legal_actions_and_chat_record():
    tokenizer = TokenizerFixture()
    renderer = ObservationRenderer(tokenizer, {"state": {"ready": "<state>"}}, {"value": [0, 1]}, history_turns=1)
    assert renderer.features({"state": "ready", "value": np.float32(1)}) == {"state": "<state>", "value": 63}
    assert renderer.encode(0, "fixture", "task", {"state": "ready", "value": 0.0},
                           {2: "legal choice"}, ["earlier", "latest"], "private role", 2, 3) == (1, 2, 3)
    text = tokenizer.messages[1]["content"]
    assert "latest" in text and "earlier" not in text
    assert "legal choice" in text and "Previous action 2" in text
    with pytest.raises(ValueError):
        renderer.features({"state": "unknown"})


def test_native_service_serializes_exact_frame_fields():
    requests = io.StringIO('{"operation":"describe"}\n{"operation":"reset","seed":2}\n'
                          '{"operation":"step","actions":{"0":0}}\n')
    response = io.StringIO()
    serve(EnvironmentFixture(), requests, response)
    rows = [json.loads(row) for row in response.getvalue().splitlines()]
    assert rows[0]["agents"] == [0]
    assert Frame.from_record(rows[1]).legal_actions == {0: {0: "action"}}
    assert Frame.from_record(rows[2]).terminated


def test_factor_control_uses_factor_derivatives_and_preserves_capacity():
    coordinator = Coordinator({(0, "a"): (2, 3)}, 1, {0: 1})
    coordinator.states[(0, "a")] = FactorMatrix(np.array([[1.0], [2.0]]), np.array([[3.0, 4.0, 5.0]]))
    coordinator.ranks[(0, "a")] = 1
    gradient = FactorMatrix.from_dense(np.arange(6, dtype=float).reshape(2, 3))
    batch = GradientBatch(0, {(0, "a"): gradient}, 1)
    proposal = factor_update(coordinator, batch, Calibration(2.0, 0, {0: 100.0}), fixed_ranks(coordinator))
    state = coordinator.states[(0, "a")]
    np.testing.assert_allclose(proposal.states[(0, "a")].left, state.left + 0.45 * gradient.dense() @ state.right.T)
    np.testing.assert_allclose(proposal.states[(0, "a")].right, state.right + 0.45 * state.left.T @ gradient.dense())
    assert proposal.diagnostics["confidence"]["gap_lower_bound"] is None
    assert coordinator.commit(proposal)["committed"]


@pytest.mark.parametrize("agents", [5, 10])
def test_matched_capacity_is_balanced_across_agents_and_blocks(agents):
    shapes = {(agent, "block_" + str(block)): (4, 4) for agent in range(agents) for block in range(12)}
    coordinator = Coordinator(shapes, 96, dict.fromkeys(range(agents), 60))
    ranks = fixed_ranks(coordinator, seed=3)
    totals = [sum(rank for key, rank in ranks.items() if key[0] == agent) for agent in range(agents)]
    assert sum(totals) == 96 and min(totals) > 0
    assert max(totals) - min(totals) <= 1
    for agent in range(agents):
        values = [rank for key, rank in ranks.items() if key[0] == agent]
        assert max(values) - min(values) <= 1
    assert ranks == fixed_ranks(coordinator, seed=3)
    assert ranks != fixed_ranks(coordinator, seed=4)


def test_reset_changes_the_proposal_anchor_after_the_gradient_is_supplied():
    coordinator = Coordinator({(0, "a"): (1, 1)}, 1, {0: 1})
    calibration = Calibration(1.0, 0, {0: 10.0})
    coordinator.step(GradientBatch(0, {(0, "a"): FactorMatrix.from_dense(np.array([[1.0]]))}, 1), calibration)
    batch = GradientBatch(1, {(0, "a"): FactorMatrix.from_dense(np.array([[2.0]]))}, 1)
    reset = coordinator.prepare(batch, calibration, reset_incumbent=True)
    step = reset.diagnostics["common_step"]
    np.testing.assert_allclose(reset.states[(0, "a")].dense(), [[2 * step]])
    assert coordinator.states[(0, "a")].dense()[0, 0] == pytest.approx(0.9)


def test_zero_capacity_deployment_and_completed_checkpoint_resume(tmp_path):
    config = {"episodes": 2, "total_rank": 0, "record_trajectories": False, "task_id": "fixture"}
    calibration = Calibration(1.0, 0, {0: 10.0})
    output = tmp_path / "run"
    records = run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture",
                             {0: RewardScale(0, 1)}, config, output)
    assert len(records) == 2 and records[-1]["update"]["block_ranks"] == {(0, "a"): 0}
    state, extra = load_checkpoint(output / "state.npz")
    assert state.version == 2 and extra["episodes"] == 2
    assert run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture",
                          {0: RewardScale(0, 1)}, config, output, checkpoint=output / "state.npz") == []
    with pytest.raises(ValueError):
        run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture", {0: RewardScale(0, 1)}, config, output)
    with pytest.raises(ValueError):
        deployment_metrics([output], [2], "fixture_metric")


def test_rejected_commit_stops_before_baseline_or_episode_record(monkeypatch, tmp_path):
    monkeypatch.setattr(Coordinator, "commit", lambda self, proposal: {"committed": False, "reason": "fixture"})
    with pytest.raises(RuntimeError):
        run_deployment(EnvironmentFixture(), PolicyFixture(), Calibration(1.0, 0, {0: 10.0}), "fixture",
                        {0: RewardScale(0, 1)}, {"episodes": 2}, tmp_path / "run")
    assert not list((tmp_path / "run").glob("episode_*.json"))


def test_paired_probe_recordings_have_unit_native_norm_and_distinct_repeated_seeds(tmp_path):
    direction = unit_direction({(0, "a"): (2, 3), (1, "b"): (4, 2)}, np.random.default_rng(2))
    assert sum(value.norm_squared() for value in direction.values()) == pytest.approx(1.0)
    manifest = record_probes(EnvironmentFixture(), PolicyFixture(), {0: RewardScale(0, 1)},
                             {"probe_directions": 2}, {"sample_bounds": {"0": 10},
                                                       "reference_scale": 1, "validity_record": "software_fixture"}, tmp_path)
    assert len(manifest["probes"]) == 4
    assert len(manifest["secant_residuals"]) == 2
    assert all(probe["direction_norm"] == pytest.approx(1.0) for probe in manifest["probes"])
    assert manifest["work"]["joint_episodes"] == 8
    assert manifest["work"]["gradient_backward_calls"] == 8
    assert manifest["moment_pairs"] == 2
    assert manifest["moment_rule"] == "supplement_equation_19"
    assert set(manifest["second_moments"]) == {0}


def test_incumbent_calibration_restores_policy_and_uses_completed_transition(tmp_path):
    policy = PolicyFixture()
    states = {(0, "a"): FactorMatrix.from_dense(np.array([[2.0]]))}
    preceding = {(0, "a"): FactorMatrix.from_dense(np.array([[1.0]]))}
    policy.set_states(states, 3)
    rng = policy.generator.get_state().clone()
    settings = {"probe_directions": 2, "failure_probability": 0.02, "norm_threshold": 1e-9}
    manifest = record_probes(EnvironmentFixture(), policy, {0: RewardScale(0, 1)}, settings,
                             {"sample_bounds": {"0": 10}, "reference_scale": 1,
                              "validity_record": "software_fixture"}, tmp_path, 2, states, 3,
                             preceding_states=preceding)
    assert policy.version == 3
    np.testing.assert_allclose(policy.states[(0, "a")].dense(), [[2]])
    assert torch.equal(rng, policy.generator.get_state())
    assert manifest["available_version"] == 3
    assert len(manifest["transition_probes"]) == 4
    assert manifest["work"]["joint_episodes"] == 16
    assert manifest["secant_residuals"] == pytest.approx([0, 0])
    calibration = load_probe_calibration(tmp_path / "manifest.json")
    assert calibration.failure_probability == 0.02
    assert calibration.norm_threshold == 1e-9


def test_refresh_schedule_and_work_survive_resume(tmp_path):
    config = {"episodes": 2, "total_rank": 1, "task_id": "fixture",
              "calibration": {"refresh_interval": 1, "probe_directions": 1}}
    calibration = Calibration(1.0, 0, {0: 10.0})
    output = tmp_path / "run"
    records = run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture",
                             {0: RewardScale(0, 1)}, config, output, seed=3)
    assert [row["calibration_version"] for row in records] == [0, 1]
    assert records[0]["calibration_work"]["joint_episodes"] == 4
    assert records[1]["calibration_work"]["joint_episodes"] == 8
    assert records[0]["work"]["action_forward_calls"] == 1
    _, extra = load_checkpoint(output / "state.npz")
    assert extra["calibration"]["available_version"] == 1
    assert run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture", {0: RewardScale(0, 1)},
                          config, output, seed=3, checkpoint=output / "state.npz") == []
    other = run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture", {0: RewardScale(0, 1)},
                            dict(config, calibration={"refresh_interval": 0}), tmp_path / "other", seed=4)
    assert set(row["native_seed"] for row in records).isdisjoint(row["native_seed"] for row in other)


def test_paired_second_moment_matches_equation_19():
    differences, bound, scale, failure = [1.0, 4.0, 9.0], 5.0, 3.0, 0.01
    mean, logarithm = np.mean(differences), np.log(2 / failure)
    expected = mean + bound * np.sqrt(2 * mean * logarithm / 3) + 7 * scale ** 2 * logarithm / 6
    assert paired_second_moment(differences, bound, scale, failure) == pytest.approx(expected)
    with pytest.raises(ValueError):
        paired_second_moment([1], bound, scale)


@pytest.mark.parametrize("interrupted_version", [0, 1])
def test_incomplete_probe_directory_can_resume_without_losing_artifacts(monkeypatch, tmp_path, interrupted_version):
    from joint_ttopt import probes
    original = probes.record_probes
    def interrupted(*args, **kwargs):
        if args[8] == interrupted_version:
            directory = args[5]
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "partial.json").write_text('{}\n')
            raise RuntimeError("software fixture interruption")
        return original(*args, **kwargs)
    config = {"episodes": 2, "total_rank": 1, "task_id": "fixture",
              "calibration": {"refresh_interval": 1, "probe_directions": 2}}
    calibration = Calibration(1.0, 0, {0: 10.0})
    output = tmp_path / "run"
    monkeypatch.setattr(probes, "record_probes", interrupted)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture",
                        {0: RewardScale(0, 1)}, config, output)
    monkeypatch.setattr(probes, "record_probes", original)
    result = run_deployment(EnvironmentFixture(), PolicyFixture(), calibration, "fixture", {0: RewardScale(0, 1)},
                             config, output, checkpoint=output / "state.npz")
    assert len(result) == 2 - interrupted_version and result[-1]["episode"] == 2
    assert (output / f"calibration/version_{interrupted_version:05d}.incomplete_0/partial.json").is_file()


def test_normalization_anchors_are_shared_across_horizons_and_equal_weight_aggregation():
    rows = []
    for benchmark, tasks in {"one": ["a", "b"], "two": ["c"]}.items():
        for task in tasks:
            for method, score in {"frozen": 1.0, "baseline": 3.0, "joint": 5.0}.items():
                for seed in [0, 1]:
                    for horizon in [10, 50]:
                        rows.append({"data_kind": "measured_run", "benchmark": benchmark, "task": task,
                                     "backbone": "fixture", "method": method, "seed": seed, "episode": horizon,
                                     "native_score": score, "comparison_signature": "fixture"})
    anchors = build_anchors(rows, ["baseline"])
    normalized = apply_anchors(rows, anchors)
    aggregated = aggregate_games(normalized)
    assert all(row["normalized_gain"] == pytest.approx(2.0) for row in aggregated if row["method"] == "joint")
    assert all(row["anchor_horizon"] == 50 for row in anchors)
    altered = [dict(row, comparison_signature="other") for row in rows]
    with pytest.raises(ValueError):
        apply_anchors(altered, anchors)
    with pytest.raises(ValueError):
        normalized_gain(2, 1, 1)
    comparison = paired_bootstrap({0: 2, 1: 2}, {0: 1, 1: 1}, samples=100)
    assert comparison["lower"] == comparison["upper"] == 1
