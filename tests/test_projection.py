import itertools

import numpy as np
import pytest

from joint_ttopt.allocation import allocate, project
from joint_ttopt.factors import FactorMatrix, Spectrum, field_difference, joint_norm


def test_complete_factor_proposal_matches_dense_svd():
    random = np.random.default_rng(17)
    state = FactorMatrix(random.normal(size=(7, 2)), random.normal(size=(2, 5)))
    gradient = FactorMatrix(random.normal(size=(7, 9)), random.normal(size=(9, 5)))
    proposal = state.proposal(gradient, 0.13)
    dense = state.dense() + 0.13 * gradient.dense()
    spectrum = proposal.spectrum()
    np.testing.assert_allclose(proposal.dense(), dense, atol=1e-12)
    np.testing.assert_allclose(spectrum.values, np.linalg.svd(dense, compute_uv=False), atol=1e-12)
    left, values, right = np.linalg.svd(dense, full_matrices=False)
    np.testing.assert_allclose(spectrum.truncate(3).dense(), (left[:, :3] * values[:3]) @ right[:3], atol=1e-11)
    assert proposal.width == 11


def test_global_greedy_equals_exhaustive_rank_allocation():
    proposals = {(0, "a"): FactorMatrix.from_dense(np.diag([9.0, 3.0])),
                 (0, "b"): FactorMatrix.from_dense(np.diag([8.0, 2.0])),
                 (1, "a"): FactorMatrix.from_dense(np.diag([7.0, 6.0, 1.0])),
                 (2, "a"): FactorMatrix.from_dense(np.diag([20.0, 19.0]))}
    caps = {0: 2, 1: 3, 2: 0}
    steps = {0: 0.2, 1: 0.3, 2: 0.4}
    spectra = {key: value.spectrum() for key, value in proposals.items()}
    result = allocate(spectra, 4, caps, steps)
    optimum = 0.0
    keys = list(spectra)
    for ranks in itertools.product(*(range(len(spectra[key].values) + 1) for key in keys)):
        if sum(ranks) > 4 or any(sum(rank for key, rank in zip(keys, ranks) if key[0] == agent) > cap
                                 for agent, cap in caps.items()):
            continue
        value = sum(np.sum(spectra[key].values[:rank] ** 2) / (2 * steps[key[0]])
                    for key, rank in zip(keys, ranks))
        optimum = max(optimum, value)
    assert result["retained_value"] == pytest.approx(optimum)
    assert result["agent_ranks"][2] == 0
    assert sum(result["ranks"].values()) == 4
    states, _ = project(proposals, 4, caps, steps)
    assert all(state.width == result["ranks"][key] for key, state in states.items())


def test_unit_rank_allocation_ignores_block_storage_cost():
    proposals = {(0, "large"): FactorMatrix.from_dense(np.eye(20)),
                 (1, "small"): FactorMatrix.from_dense(np.diag([2.0, 0.1]))}
    states, allocation = project(proposals, 1, {0: 1, 1: 1}, {0: 0.5, 1: 0.5})
    assert states[(0, "large")].width == 0
    assert states[(1, "small")].width == 1
    assert allocation["excluded_value"] == pytest.approx(20.01)


def test_stable_ties_zero_budget_and_zero_proposals():
    spectra = {(1, "a"): Spectrum(np.eye(2), np.ones(2), np.eye(2)),
               (0, "b"): Spectrum(np.eye(2), np.ones(2), np.eye(2))}
    result = allocate(spectra, 1, {0: 2, 1: 2}, {0: 1.0, 1: 1.0})
    assert result["selected"][0][:3] == (0, "b", 0)
    assert sum(allocate(spectra, 0, {0: 2, 1: 2}, {0: 1.0, 1: 1.0})["ranks"].values()) == 0
    zero = FactorMatrix.zeros(4, 7)
    assert zero.spectrum().truncate(0).shape == (4, 7)
    assert zero.norm_squared() == 0


def test_native_matrix_norms_are_invariant_to_factor_basis():
    random = np.random.default_rng(2)
    left = random.normal(size=(6, 3))
    right = random.normal(size=(3, 5))
    transform = random.normal(size=(3, 3)) + 4 * np.eye(3)
    first = FactorMatrix(left, right)
    second = FactorMatrix(left @ transform, np.linalg.solve(transform, right))
    np.testing.assert_allclose(first.dense(), second.dense(), atol=1e-12)
    assert first.norm_squared() == pytest.approx(second.norm_squared(), rel=1e-12)
    field = {(0, "a"): first, (1, "b"): FactorMatrix.from_dense(np.ones((2, 9)))}
    assert joint_norm(field) == pytest.approx(np.sqrt(np.sum(first.dense() ** 2) + 18))
    assert joint_norm(field_difference({(0, "a"): first}, {(0, "a"): second})) < 1e-12


@pytest.mark.parametrize("rank", [-1, 1.5, True, float("inf")])
def test_invalid_rank_constraints_are_rejected(rank):
    with pytest.raises(ValueError):
        allocate({}, rank, {0: 1}, {0: 1.0})
