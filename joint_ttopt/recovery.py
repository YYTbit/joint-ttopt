import numpy as np


def recover_in_subspace(gradients, error_radii, curvature, utility_scales,
                         radius, tolerance=1e-7):
    import cvxpy as convex
    gradients = np.asarray(gradients, dtype=float)
    fields = [np.asarray(item, dtype=float) for item in (error_radii, curvature, utility_scales)]
    if gradients.ndim != 2 or gradients.shape[0] < 1:
        raise ValueError("Recovery requires one cross-utility gradient per protected participant")
    if any(item.shape != (gradients.shape[0],) for item in fields):
        raise ValueError("Recovery fields have incompatible participant dimensions")
    if not np.isfinite(gradients).all() or any(not np.isfinite(item).all() for item in fields):
        raise ValueError("Recovery inputs are finite")
    if np.any(fields[0] < 0) or np.any(fields[1] <= 0) or np.any(fields[2] <= 0):
        raise ValueError("Invalid recovery error, curvature, or utility scale")
    if not np.isfinite([radius, tolerance]).all() or radius <= 0 or tolerance <= 0:
        raise ValueError("Recovery radius and acceptance tolerance are positive")
    if gradients.shape[1] == 0:
        return np.empty(0), 0.0, "empty_subspace"
    movement = convex.Variable(gradients.shape[1])
    gain = convex.Variable(nonneg=True)
    norm = convex.norm(movement, 2)
    lower = gradients @ movement - convex.multiply(fields[0], norm)
    lower -= 0.5 * convex.multiply(fields[1], convex.sum_squares(movement))
    problem = convex.Problem(convex.Maximize(gain),
                             [lower >= convex.multiply(fields[2], gain), norm <= radius])
    try:
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
    except convex.error.SolverError:
        return np.zeros(gradients.shape[1]), 0.0, "solver_failure"
    if movement.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        return np.zeros(gradients.shape[1]), 0.0, str(problem.status)
    result = np.asarray(movement.value).reshape(-1)
    length = np.linalg.norm(result)
    if not np.isfinite(result).all():
        return np.zeros(gradients.shape[1]), 0.0, "invalid_candidate"
    if length > radius:
        result *= radius / length
        length = np.linalg.norm(result)
    lower = (gradients @ result - fields[0] * length - 0.5 * fields[1] * length ** 2) / fields[2]
    certified_gain = float(np.min(lower))
    if certified_gain <= tolerance:
        return np.zeros(gradients.shape[1]), 0.0, "retain_state"
    return result, certified_gain, str(problem.status)


def reconstruct_core(left_basis, incumbent_core, right_basis, movement):
    left, core, right, change = [np.asarray(item, dtype=float) for item in
                               (left_basis, incumbent_core, right_basis, movement)]
    if core.ndim != 2 or core.shape[0] != core.shape[1] or change.shape != core.shape:
        raise ValueError("Recovery core and movement have different square shapes")
    if any(item.ndim != 2 or item.shape[1] != core.shape[0] for item in (left, right)):
        raise ValueError("Recovery bases have different retained ranks")
    if any(not np.isfinite(item).all() for item in (left, core, right, change)):
        raise ValueError("Recovery state contains nonfinite entries")
    for basis in (left, right):
        if not np.allclose(basis.T @ basis, np.eye(core.shape[0]), atol=1e-8, rtol=0):
            raise ValueError("Recovery bases are orthonormal")
    return left @ (core + change) @ right.T
