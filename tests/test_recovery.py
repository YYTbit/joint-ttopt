import numpy as np
import pytest

from joint_ttopt.recovery import reconstruct_core, recover_in_subspace


def test_recovery_solves_a_common_gain_problem_and_retains_conflicting_state():
    pytest.importorskip("cvxpy")
    movement, gain, status = recover_in_subspace([[2.0], [1.0]], [0.0, 0.0], [2.0, 2.0], [1.0, 1.0], 2.0)
    assert status in {"optimal", "optimal_inaccurate"}
    np.testing.assert_allclose(movement, [0.5], atol=3e-5)
    assert gain == pytest.approx(0.25, abs=1e-6)
    movement, gain, status = recover_in_subspace([[1.0], [-1.0]], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], 1.0)
    np.testing.assert_array_equal(movement, [0.0])
    assert gain == 0 and status == "retain_state"


def test_recovery_uses_orthonormal_incumbent_subspaces():
    generator = np.random.default_rng(8)
    left = np.linalg.qr(generator.normal(size=(5, 2)))[0]
    right = np.linalg.qr(generator.normal(size=(4, 2)))[0]
    core = np.diag([2.0, 1.0])
    movement = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    result = reconstruct_core(left, core, right, movement)
    assert np.linalg.matrix_rank(result) <= 2
    np.testing.assert_allclose(np.linalg.norm(result - left @ core @ right.T), np.linalg.norm(movement))
    with pytest.raises(ValueError):
        reconstruct_core(2 * left, core, right, movement)
