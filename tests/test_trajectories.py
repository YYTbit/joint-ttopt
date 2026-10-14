import numpy as np
import pytest

from joint_ttopt.confidence import RewardScale
from joint_ttopt.factors import FactorMatrix
from joint_ttopt.trajectories import Decision, HistoryBaseline, Trajectory, estimate_gradients, returns_to_go


class ScorePolicy:
    version = 0

    def score_gradients(self, decision):
        return {"a": FactorMatrix.from_dense(np.asarray([[decision.time + 1.0]]))}


def test_returns_discount_baseline_and_regularizer_follow_equation_14():
    decision_0 = Decision(0, 0, 0, (1,), (0,), 0, 0)
    decision_1 = Decision(0, 0, 1, (1,), (0,), 0, 0)
    trajectory = Trajectory(0, "fixture_0", (decision_0, decision_1), {0: [1.0, 2.0]})
    state = {(0, "a"): FactorMatrix.from_dense(np.asarray([[3.0]]))}
    gradients = estimate_gradients([trajectory], ScorePolicy(), state, {0: RewardScale(0, 2)},
                                  {0: 0.45}, {(0, 0): 0.5, (0, 1): 1.0}, 0, 0.5)
    expected = (2.0 - 0.5) / 2 + 0.5 * (2.0 - 1.0) / 2 * 2 - 0.45 * 3
    assert gradients.gradients[(0, "a")].dense().item() == pytest.approx(expected)
    assert gradients.sample_count == 1
    np.testing.assert_allclose(returns_to_go([1, 2, 3], 0.5), [2.75, 3.5, 3])


def test_regularizer_is_applied_once_after_batch_averaging():
    decisions = (Decision(0, 0, 0, (1,), (0,), 0, 0),)
    trajectories = [Trajectory(0, "fixture_0", decisions, {0: [2.0]}),
                    Trajectory(0, "fixture_1", decisions, {0: [4.0]})]
    states = {(0, "a"): FactorMatrix.from_dense(np.asarray([[2.0]]))}
    batch = estimate_gradients(trajectories, ScorePolicy(), states, {0: RewardScale(0, 1)}, {0: 0.5}, {}, 0)
    assert batch.gradients[(0, "a")].dense().item() == pytest.approx(2.0)
    assert batch.sample_count == 2
    assert batch.sample_norms[0] == pytest.approx([1.0, 3.0])


def test_action_independent_history_is_frozen_per_batch():
    baseline = HistoryBaseline([0], window=2, discount=0.5)
    version, snapshot = baseline.snapshot()
    assert version == 0 and snapshot == {}
    first = Trajectory(0, "fixture_0", (), {0: [2.0, 4.0]})
    baseline.observe([first])
    assert snapshot == {}
    assert baseline.snapshot()[1] == {(0, 0): 4.0, (0, 1): 4.0}
    restored = HistoryBaseline.restore(baseline.record())
    assert restored.snapshot() == baseline.snapshot()


def test_incomplete_duplicate_and_mixed_policy_trajectories_are_rejected():
    with pytest.raises(ValueError):
        Trajectory(0, "fixture_0", (), {0: [1.0]}, complete=False)
    with pytest.raises(ValueError):
        Trajectory(0, "fixture_0", (Decision(1, 0, 0, (1,), (0,), 0, 0),), {0: [1.0]})
    trajectory = Trajectory(0, "fixture_0", (), {0: [1.0]})
    with pytest.raises(ValueError):
        estimate_gradients([trajectory, trajectory], ScorePolicy(), {(0, "a"): FactorMatrix.zeros(1, 1)},
                           {0: RewardScale(0, 1)}, {0: 0}, {}, 0)
