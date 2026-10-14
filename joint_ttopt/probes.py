from pathlib import Path

import numpy as np

from .deployment import collect_trajectory
from .factors import FactorMatrix, field_difference, joint_norm
from .storage import save_factors, write_json
from .trajectories import HistoryBaseline, estimate_gradients


def unit_direction(shapes, generator):
    direction = {key: FactorMatrix(generator.normal(size=(rows, 1)), generator.normal(size=(1, columns)))
                 for key, (rows, columns) in sorted(shapes.items())}
    norm = joint_norm(direction)
    if norm == 0:
        raise ValueError("Calibration direction has zero product norm")
    return {key: value.scaled(1 / norm) for key, value in direction.items()}


def record_probes(environment, policy, scales, settings, bounds, output, seed=0):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("A new calibration recording requires an empty output directory")
    agents = tuple(sorted(scales))
    shapes = policy.shapes(agents)
    all_shapes = policy.shapes(environment.agents)
    count = int(settings.get("probe_directions", 32))
    radius = float(settings.get("radius", 0.01))
    if count < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Calibration uses a positive probe count and radius")
    if set(map(int, bounds["sample_bounds"])) != set(agents):
        raise ValueError("Declared calibration envelopes have different updating agents")
    baseline = HistoryBaseline(environment.agents, settings.get("baseline_window", 8), settings.get("discount", 0.99))
    baseline_version, baseline_values = baseline.snapshot()
    generator = np.random.default_rng(seed)
    probes, residuals = [], []
    for index in range(count):
        direction = unit_direction(shapes, generator)
        ratios = []
        for repeat in range(2):
            pair_seed = seed + 2 * index + repeat
            fields = []
            for sign, label in ((1, "positive"), (-1, "negative")):
                states = {key: value.scaled(sign * radius) for key, value in direction.items()}
                all_states = {key: states.get(key, FactorMatrix.zeros(*shape)) for key, shape in all_shapes.items()}
                policy.set_states(all_states, 0)
                policy.generator.manual_seed(pair_seed)
                trajectory, metrics = collect_trajectory(environment, policy, pair_seed,
                                                          "probe:" + str(index) + ":" + str(repeat) + ":" + label,
                                                          baseline_version, int(settings.get("max_episode_steps", 4096)))
                batch = estimate_gradients([trajectory], policy, states, scales,
                                             dict.fromkeys(agents, settings.get("regularization", 0.45)),
                                             baseline_values, baseline_version, settings.get("discount", 0.99))
                name = "probe_" + str(index).zfill(3) + "_" + str(repeat) + "_" + label + ".npz"
                save_factors(root / name, batch.gradients, {"paired_seed": pair_seed, "policy_version": 0,
                                                            "sample_count": 1, "sample_norms": batch.sample_norms})
                write_json(root / (name.replace(".npz", "_trajectory.json")), trajectory)
                fields.append((name, batch.gradients))
            ratios.append(joint_norm(field_difference(fields[0][1], fields[1][1])) / (2 * radius))
            probes.append({"positive": fields[0][0], "negative": fields[1][0], "radius": radius,
                           "direction_norm": joint_norm(direction)})
        residuals.append(abs(ratios[0] - ratios[1]))
    manifest = {"available_version": 0, "probes": probes, "secant_residuals": residuals,
                "sample_bounds": bounds["sample_bounds"], "second_moments": bounds.get("second_moments", {}),
                "reference_scale": bounds["reference_scale"], "validity_record": bounds["validity_record"],
                "residual_definition": "absolute_difference_between_repeated_paired_secant_estimates",
                "direction_distribution": "normalized_gaussian_rank_one_product_direction",
                "base_state": "zero_fast_weights", "repeat_pairs": 2}
    write_json(root / "manifest.json", manifest)
    return manifest
