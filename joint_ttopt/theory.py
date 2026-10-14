import numpy as np

from .allocation import project
from .factors import field_difference


def regularized_game_gap(states, gradients, step, total_rank, caps):
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Gap step is finite and positive")
    proposals = {key: state.proposal(gradients[key], step) for key, state in states.items()}
    candidate, _ = project(proposals, total_rank, caps, dict.fromkeys(caps, step))
    changes = field_difference(candidate, states)
    return float(sum(gradients[key].inner(changes[key])
                     - changes[key].norm_squared() / (2 * step) for key in states))


def contraction_coefficient(step, monotonicity, field_lipschitz, projection_sensitivity):
    values = [step, monotonicity, field_lipschitz, projection_sensitivity]
    if not np.isfinite(values).all() or min(values) <= 0 or field_lipschitz < monotonicity:
        raise ValueError("Local contraction constants are finite and positive with L >= mu")
    square = 1 - 2 * step * monotonicity + step ** 2 * field_lipschitz ** 2
    return float(projection_sensitivity * np.sqrt(max(0.0, square)))


def tracking_bounds(initial_error, coefficients, noise_radii, steps,
                    projection_sensitivities, stationary_drifts):
    arrays = [np.asarray(item, dtype=float) for item in
              (coefficients, noise_radii, steps, projection_sensitivities, stationary_drifts)]
    if len({item.shape for item in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError("Tracking records have different round dimensions")
    if not np.isfinite(initial_error) or initial_error < 0:
        raise ValueError("Initial tracking error is finite and nonnegative")
    if any(not np.isfinite(item).all() or np.any(item < 0) for item in arrays):
        raise ValueError("Tracking records are finite and nonnegative")
    if np.any(arrays[0] >= 1):
        raise ValueError("The local contraction coefficient is below one for this bound")
    bounds = [float(initial_error)]
    for coefficient, noise, step, sensitivity, drift in zip(*arrays):
        bounds.append(float(coefficient * bounds[-1] + sensitivity * step * noise + drift))
    return np.asarray(bounds)
