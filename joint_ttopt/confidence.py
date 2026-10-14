from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardScale:
    offset: float
    span: float

    def __post_init__(self):
        if not np.isfinite([self.offset, self.span]).all() or self.span <= 0:
            raise ValueError("Reward normalization requires a finite positive span")

    @classmethod
    def from_interval(cls, lower, upper, fallback_span=None):
        span = upper - lower
        if span == 0:
            if fallback_span is None:
                raise ValueError("A degenerate entry interval requires a benchmark return span")
            span = fallback_span
        if span < 0:
            raise ValueError("Entry return interval is reversed")
        return cls(float(lower), float(span))

    def normalize(self, value):
        return (value - self.offset) / self.span

    def affine(self, multiplier, shift):
        if not np.isfinite([multiplier, shift]).all() or multiplier <= 0:
            raise ValueError("Affine reward multiplier is finite and positive")
        return RewardScale(multiplier * self.offset + shift, multiplier * self.span)


def gradient_radius(sample_bound, sample_count, failure_probability,
                    second_moment_bound=None, approximation_error=0.0):
    if not np.isfinite([sample_bound, approximation_error]).all():
        raise ValueError("Gradient bounds are finite")
    if sample_bound < 0 or approximation_error < 0:
        raise ValueError("Gradient bounds are nonnegative")
    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 1:
        raise ValueError("Sample count is the positive number of complete trajectories")
    if not 0 < failure_probability < 1:
        raise ValueError("Sampling failure probability is between zero and one")
    radius = 2 * sample_bound
    if sample_count > 1:
        radius = min(radius, sample_bound / np.sqrt(sample_count)
                     * (1 + np.sqrt(2 * np.log(1 / failure_probability))))
    if second_moment_bound is not None:
        if not np.isfinite(second_moment_bound) or second_moment_bound < 0:
            raise ValueError("Second-moment bound is finite and nonnegative")
        radius = min(radius, np.sqrt(second_moment_bound / (sample_count * failure_probability)))
    return float(radius + approximation_error)


def confidence_diagnostic(radii, penalties, proximal_value, agent_levels,
                          calibration_level, total_level):
    if set(radii) != set(penalties) or set(radii) != set(agent_levels):
        raise ValueError("Confidence fields have different participants")
    if not 0 <= calibration_level < total_level < 1:
        raise ValueError("Invalid calibration or round error budget")
    if not np.isfinite(proximal_value) or any(not np.isfinite(value) or not 0 < value < 1
                                             for value in agent_levels.values()):
        raise ValueError("Invalid proximal value or agent sampling level")
    if sum(agent_levels.values()) > total_level - calibration_level + 1e-12:
        raise ValueError("Agent error levels exceed the round budget")
    if any(not np.isfinite(value) or value < 0 for value in radii.values()):
        raise ValueError("Gradient error radii are finite and nonnegative")
    if any(not np.isfinite(value) or value <= 0 for value in penalties.values()):
        raise ValueError("Young penalties are finite and positive")
    correction = sum(radii[agent] ** 2 / (2 * penalties[agent]) for agent in radii)
    return {"joint_radius": float(np.sqrt(sum(value ** 2 for value in radii.values()))),
            "gap_lower_bound": float(proximal_value - correction),
            "sampling_failure_probability": float(sum(agent_levels.values())),
            "calibration_failure_probability": float(calibration_level),
            "round_failure_probability": float(calibration_level + sum(agent_levels.values())),
            "conditional_on_calibration": True}


def trajectory_norm_envelope(advantage_bound, score_bounds, state_bounds,
                             span, regularization, horizon, discount):
    scores = np.asarray(score_bounds, dtype=float)
    states = np.asarray(state_bounds, dtype=float)
    values = [advantage_bound, span, regularization, discount]
    if scores.ndim != 1 or states.shape != scores.shape or not np.isfinite(values).all():
        raise ValueError("Trajectory envelope has incompatible bounds")
    if any(not np.isfinite(item).all() or np.any(item < 0) for item in (scores, states)):
        raise ValueError("Block score and state bounds are finite and nonnegative")
    if advantage_bound < 0 or span <= 0 or regularization < 0 or not 0 <= discount <= 1:
        raise ValueError("Invalid utility or discount bounds")
    if isinstance(horizon, bool) or int(horizon) != horizon or horizon < 1:
        raise ValueError("Envelope horizon is a positive integer")
    discounted_horizon = sum(discount ** time for time in range(int(horizon)))
    return float(np.linalg.norm(discounted_horizon * advantage_bound * scores / span
                                + regularization * states))
