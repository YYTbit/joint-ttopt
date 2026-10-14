from dataclasses import replace

import numpy as np
import pytest

from joint_ttopt.calibration import Calibration, calibration_from_probes, joint_secant, select_step
from joint_ttopt.confidence import RewardScale, confidence_diagnostic, gradient_radius
from joint_ttopt.factors import FactorMatrix, joint_norm, field_difference
from joint_ttopt.optimizer import Coordinator, GradientBatch
from joint_ttopt.storage import load_checkpoint, load_factors, save_checkpoint, save_factors
from joint_ttopt.theory import contraction_coefficient, regularized_game_gap, tracking_bounds


def fixture_coordinator():
    coordinator = Coordinator({(0, "a"): (3, 2), (1, "b"): (2, 4)}, 2, {0: 1, 1: 1})
    calibration = Calibration(2.0, 0.1, {0: 20.0, 1: 20.0})
    first = {(0, "a"): FactorMatrix.from_dense(np.asarray([[2, 0], [0, 1], [0, 0]])),
             (1, "b"): FactorMatrix.from_dense(np.asarray([[1, 0, 0, 0], [0, 0.5, 0, 0]]))}
    return coordinator, calibration, first


def test_lagged_joint_secant_uses_completed_transition():
    coordinator, calibration, first = fixture_coordinator()
    result = coordinator.step(GradientBatch(0, first, 1), calibration)
    assert result["joint_secant"] == 0
    second = {key: value.scaled(1.5) for key, value in first.items()}
    expected = joint_norm(field_difference(second, first)) / joint_norm(coordinator.states)
    proposal = coordinator.prepare(GradientBatch(1, second, 1), calibration)
    assert proposal.diagnostics["joint_secant"] == pytest.approx(expected)
    scale = max(2.0, expected + 0.1)
    assert proposal.diagnostics["common_step"] == pytest.approx(0.9 / scale)
    assert proposal.diagnostics["young_penalty"] == pytest.approx(scale / 9)


def test_joint_commit_rejects_one_invalid_participant_without_partial_change():
    coordinator, calibration, gradients = fixture_coordinator()
    proposal = coordinator.prepare(GradientBatch(0, gradients, 1), calibration)
    bad_states = dict(proposal.states)
    bad_states[(1, "b")] = FactorMatrix.zeros(7, 4)
    result = coordinator.commit(replace(proposal, states=bad_states))
    assert not result["committed"]
    assert coordinator.version == 0
    assert all(state.width == 0 for state in coordinator.states.values())
    assert coordinator.commit(proposal)["committed"]
    frozen = {key: state.dense().copy() for key, state in coordinator.states.items()}
    assert not coordinator.commit(proposal)["committed"]
    for key, state in coordinator.states.items():
        np.testing.assert_array_equal(state.dense(), frozen[key])


def test_zero_rank_releases_previous_agent_capacity():
    coordinator = Coordinator({(0, "a"): (2, 2), (1, "a"): (2, 2)}, 1, {0: 1, 1: 1})
    calibration = Calibration(1.0, 0, {0: 30.0, 1: 30.0})
    zero = FactorMatrix.zeros(2, 2)
    coordinator.step(GradientBatch(0, {(0, "a"): FactorMatrix.from_dense(np.diag([2.0, 0])), (1, "a"): zero}, 1), calibration)
    assert coordinator.ranks[(0, "a")] == 1
    gradients = {(0, "a"): FactorMatrix.from_dense(np.diag([-4.0, 0])),
                 (1, "a"): FactorMatrix.from_dense(np.diag([20.0, 0]))}
    coordinator.step(GradientBatch(1, gradients, 1), calibration)
    assert coordinator.states[(0, "a")].width == 0
    assert coordinator.ranks[(1, "a")] == 1


def test_version_and_calibration_causality_checks():
    coordinator, calibration, gradients = fixture_coordinator()
    with pytest.raises(ValueError):
        coordinator.prepare(GradientBatch(1, gradients, 1), calibration)
    with pytest.raises(ValueError):
        coordinator.prepare(GradientBatch(0, gradients, 1), replace(calibration, available_version=1))
    assert coordinator.version == 0
    assert joint_secant(coordinator.states, coordinator.states, gradients,
                        {key: value.scaled(2) for key, value in gradients.items()}) == 0


def test_reward_scale_and_entry_degeneracy():
    scale = RewardScale.from_interval(-3, 5)
    transformed = scale.affine(7, 2)
    assert transformed.normalize(7 * 4 + 2) == pytest.approx(scale.normalize(4))
    assert RewardScale.from_interval(1, 1, 5).span == 5
    with pytest.raises(ValueError):
        RewardScale.from_interval(1, 1)
    with pytest.raises(ValueError):
        RewardScale.from_interval(2, 1)


def test_single_trajectory_radius_and_joint_gap_correction():
    assert gradient_radius(3.0, 1, 0.02) == 6.0
    assert gradient_radius(3.0, 1, 0.02, 0.08) == pytest.approx(2.0)
    assert gradient_radius(3.0, 200, 0.02) < 6.0
    result = confidence_diagnostic({0: 2, 1: 3}, {0: 2, 1: 4}, 10,
                                  {0: 0.02, 1: 0.02}, 0.01, 0.05)
    assert result["gap_lower_bound"] == pytest.approx(10 - 1 - 9 / 8)
    assert result["joint_radius"] == pytest.approx(np.sqrt(13))
    assert result["round_failure_probability"] == pytest.approx(0.05)
    with pytest.raises(ValueError):
        confidence_diagnostic({0: 1}, {0: 1}, 1, {0: 0.05}, 0.01, 0.05)


def test_conditional_gap_bound_matches_an_exact_ideal_field():
    coordinator, calibration, gradients = fixture_coordinator()
    calibration = replace(calibration, second_moments={0: 0.0, 1: 0.0})
    proposal = coordinator.prepare(GradientBatch(0, gradients, 1), calibration)
    gap = regularized_game_gap(coordinator.states, gradients, 1 / proposal.diagnostics["local_scale"],
                               coordinator.total_rank, coordinator.agent_caps)
    assert gap + 1e-10 >= proposal.diagnostics["confidence"]["gap_lower_bound"]


def test_checkpoint_preserves_secant_history_and_factor_archive(tmp_path):
    coordinator, calibration, gradients = fixture_coordinator()
    coordinator.step(GradientBatch(0, gradients, 1), calibration)
    path = tmp_path / "state.npz"
    save_checkpoint(path, coordinator, {"signature": "fixture"})
    restored, extra = load_checkpoint(path)
    assert extra == {"signature": "fixture"}
    assert restored.version == coordinator.version
    second = {key: value.scaled(2) for key, value in gradients.items()}
    original = coordinator.prepare(GradientBatch(1, second, 1), calibration)
    resumed = restored.prepare(GradientBatch(1, second, 1), calibration)
    assert resumed.diagnostics["joint_secant"] == pytest.approx(original.diagnostics["joint_secant"])
    for key in original.states:
        np.testing.assert_allclose(original.states[key].dense(), resumed.states[key].dense())
    save_factors(tmp_path / "gradients.npz", gradients, {"sample_count": 1})
    loaded, record = load_factors(tmp_path / "gradients.npz")
    assert record["sample_count"] == 1
    for key in gradients:
        np.testing.assert_allclose(loaded[key].dense(), gradients[key].dense())


def test_paired_calibration_and_tracking_recursion():
    positive = {(0, "a"): FactorMatrix.from_dense(np.asarray([[2.0]]))}
    negative = {(0, "a"): FactorMatrix.from_dense(np.asarray([[-2.0]]))}
    calibration = calibration_from_probes([(0.5, positive, negative)], [0.1, 0.2, 0.3], {0: 5.0}, 1.0)
    assert calibration.local_scale == pytest.approx(4.0)
    assert calibration.secant_tolerance == pytest.approx(0.28)
    assert contraction_coefficient(0.1, 1, 2, 1) < 1
    bounds = tracking_bounds(2.0, [0.5, 0.5], [0.1, 0.2], [0.2, 0.2], [1, 1], [0.1, 0.1])
    np.testing.assert_allclose(bounds, [2.0, 1.12, 0.70])
