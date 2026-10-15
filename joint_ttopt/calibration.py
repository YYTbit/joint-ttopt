from dataclasses import dataclass, field

import numpy as np

from .factors import field_difference, joint_norm


@dataclass(frozen=True)
class Calibration:
    local_scale: float
    secant_tolerance: float
    sample_bounds: dict
    second_moments: dict = field(default_factory=dict)
    available_version: int = 0
    norm_threshold: float = 1e-12
    failure_probability: float = 0.01
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        numbers = [self.local_scale, self.secant_tolerance, self.norm_threshold,
                   self.failure_probability]
        if not np.isfinite(numbers).all() or self.local_scale <= 0:
            raise ValueError("Calibration scale is finite and positive")
        if min(self.secant_tolerance, self.norm_threshold) < 0:
            raise ValueError("Calibration tolerances are nonnegative")
        if (not 0 <= self.failure_probability < 1 or isinstance(self.available_version, bool)
                or int(self.available_version) != self.available_version or self.available_version < 0):
            raise ValueError("Invalid calibration probability or policy version")
        if not self.sample_bounds or not set(self.second_moments).issubset(self.sample_bounds):
            raise ValueError("Calibration moments refer to declared participant envelopes")
        for values in (self.sample_bounds, self.second_moments):
            if any(not np.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError("Calibration bounds are finite and nonnegative")


def joint_secant(current_states, preceding_states, current_gradients,
                 preceding_gradients, norm_threshold=1e-12):
    if preceding_states is None or preceding_gradients is None:
        return 0.0
    if not np.isfinite(norm_threshold) or norm_threshold < 0:
        raise ValueError("Secant norm threshold is finite and nonnegative")
    movement = joint_norm(field_difference(current_states, preceding_states))
    if movement <= norm_threshold:
        return 0.0
    change = joint_norm(field_difference(current_gradients, preceding_gradients))
    return float(change / movement)


def select_step(calibration, secant, margin=0.10):
    if not np.isfinite([secant, margin]).all() or secant < 0 or not 0 < margin < 1:
        raise ValueError("Invalid joint secant or step margin")
    # The step leaves a fixed margin below the calibrated local gradient-change bound.
    scale = max(calibration.local_scale, secant + calibration.secant_tolerance)
    step = (1 - margin) / scale
    penalty = 1 / step - scale
    return float(step), float(penalty), float(scale)


def calibration_from_probes(probes, residuals, sample_bounds, reference_scale,
                            second_moments=None, quantile=0.90, floor_fraction=0.05,
                            available_version=0, provenance=None):
    if not probes or not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Calibration requires paired probes and a positive reference scale")
    if not 0 <= quantile <= 1 or not 0 < floor_fraction <= 1:
        raise ValueError("Invalid calibration quantile or scale floor")
    values = []
    for radius, positive, negative in probes:
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Probe radius is finite and positive")
        values.append(joint_norm(field_difference(positive, negative)) / (2 * radius))
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 1 or not len(residuals) or not np.isfinite(residuals).all():
        raise ValueError("Calibration requires recorded finite secant residuals")
    return Calibration(max(floor_fraction * reference_scale, max(values)),
                       float(np.quantile(np.abs(residuals), quantile)),
                       dict(sample_bounds), dict(second_moments or {}), available_version,
                       provenance=dict(provenance or {}, probe_count=len(probes),
                                       probe_radii=[float(item[0]) for item in probes],
                                       residual_count=len(residuals), residual_quantile=quantile,
                                       reference_scale=float(reference_scale)))
