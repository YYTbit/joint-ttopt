import time
from pathlib import Path

import numpy as np

from .accounting import add_work, derive_seed, trajectory_work
from .calibration import calibration_from_probes, paired_second_moment
from .deployment import collect_trajectory
from .factors import FactorMatrix, field_difference, joint_norm, validate_field
from .storage import file_digest, load_factors, save_factors, write_json
from .trajectories import HistoryBaseline, estimate_gradients


def unit_direction(shapes, generator):
    direction = {key: FactorMatrix(generator.normal(size=(rows, 1)), generator.normal(size=(1, columns)))
                 for key, (rows, columns) in sorted(shapes.items())}
    norm = joint_norm(direction)
    if norm == 0:
        raise ValueError("Calibration direction has zero product norm")
    return {key: value.scaled(1 / norm) for key, value in direction.items()}


def record_probes(environment, policy, scales, settings, bounds, output, seed=0,
                  base_states=None, version=0, baseline=None, preceding_states=None):
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
    base = ({key: FactorMatrix.zeros(*shape) for key, shape in shapes.items()}
            if base_states is None else dict(base_states))
    validate_field(base, {key: FactorMatrix.zeros(*shape) for key, shape in shapes.items()})
    if preceding_states is not None:
        validate_field(preceding_states, base)
    baseline = baseline or HistoryBaseline(environment.agents, settings.get("baseline_window", 8),
                                            settings.get("discount", 0.99))
    baseline_version, baseline_values = baseline.snapshot()
    generator = np.random.default_rng(seed)
    probes, residuals, work, transition_probes = [], [], [], []
    empirical_norms = {agent: [] for agent in agents}
    paired_differences = {agent: [] for agent in agents}
    saved_states = getattr(policy, "states", None)
    saved_version = getattr(policy, "version", 0)
    saved_rng = policy.generator.get_state().clone()
    save_factors(root / "base_state.npz", base, {"policy_version": version})
    if preceding_states is not None:
        save_factors(root / "preceding_state.npz", preceding_states, {"policy_version": max(0, version - 1)})
    start = time.perf_counter()

    def measure(states, pair_seed, name):
        all_states = {key: states.get(key, (saved_states or {}).get(key, FactorMatrix.zeros(*shape)))
                      for key, shape in all_shapes.items()}
        policy.set_states(all_states, version)
        policy.generator.manual_seed(pair_seed)
        trajectory, metrics = collect_trajectory(environment, policy, pair_seed, name,
                                                  baseline_version, int(settings.get("max_episode_steps", 4096)))
        batch = estimate_gradients([trajectory], policy, states, scales,
                                     dict.fromkeys(agents, settings.get("regularization", 0.45)),
                                     baseline_values, baseline_version, settings.get("discount", 0.99))
        save_factors(root / (name + ".npz"), batch.gradients,
                     {"paired_seed": pair_seed, "policy_version": version, "sample_count": 1,
                      "baseline_version": baseline_version, "sample_norms": batch.sample_norms})
        write_json(root / (name + "_trajectory.json"), trajectory)
        work.append(trajectory_work(trajectory, agents))
        for agent in agents:
            empirical_norms[agent].extend(batch.sample_norms[agent])
        return name + ".npz", batch.gradients

    try:
        for index in range(count):
            direction = unit_direction(shapes, generator)
            ratios = []
            for repeat in range(2):
                pair_seed = derive_seed(seed, 2, version, index, repeat)
                fields = []
                for sign, label in ((1, "positive"), (-1, "negative")):
                    states = {key: base[key].plus(value, sign * radius) for key, value in direction.items()}
                    name = "probe_" + str(index).zfill(3) + "_" + str(repeat) + "_" + label
                    fields.append(measure(states, pair_seed, name))
                ratios.append(joint_norm(field_difference(fields[0][1], fields[1][1])) / (2 * radius))
                if repeat == 0:
                    difference = field_difference(fields[0][1], fields[1][1])
                    for agent in agents:
                        paired_differences[agent].append(sum(value.norm_squared() for key, value in difference.items()
                                                              if key[0] == agent))
                probes.append({"positive": fields[0][0], "negative": fields[1][0], "radius": radius,
                               "paired_seed": pair_seed, "direction_norm": joint_norm(direction)})
            residuals.append(abs(ratios[0] - ratios[1]))
        movement = joint_norm(field_difference(base, preceding_states)) if preceding_states is not None else 0.0
        if movement > settings.get("norm_threshold", 1e-12):
            residuals = []
            for index in range(count):
                ratios = []
                for repeat in range(2):
                    pair_seed = derive_seed(seed, 3, version, index, repeat)
                    first = measure(base, pair_seed, f"transition_{index:03d}_{repeat}_current")
                    second = measure(preceding_states, pair_seed, f"transition_{index:03d}_{repeat}_preceding")
                    ratios.append(joint_norm(field_difference(first[1], second[1])) / movement)
                    transition_probes.append({"current": first[0], "preceding": second[0],
                                              "paired_seed": pair_seed, "movement_norm": movement,
                                              "secant": ratios[-1]})
                residuals.append(abs(ratios[0] - ratios[1]))
    finally:
        policy.set_states(saved_states or {key: base.get(key, FactorMatrix.zeros(*shape))
                                          for key, shape in all_shapes.items()}, saved_version)
        policy.generator.set_state(saved_rng)
    moment_bounds, moment_scales = {}, {}
    if count >= 2:
        for agent in agents:
            declared_scale = bounds.get("probe_difference_scales", {}).get(str(agent),
                               bounds.get("probe_difference_scales", {}).get(agent))
            scale = float(declared_scale) if declared_scale is not None else float(np.sqrt(max(paired_differences[agent])))
            envelope = bounds["sample_bounds"].get(str(agent), bounds["sample_bounds"].get(agent))
            moment_bounds[agent] = paired_second_moment(paired_differences[agent], envelope, scale,
                                                         settings.get("failure_probability", 0.01))
            moment_scales[agent] = scale
    else:
        moment_bounds = bounds.get("second_moments", {})
    manifest = {"available_version": version, "probes": probes, "secant_residuals": residuals,
                "transition_probes": transition_probes,
                "sample_bounds": bounds["sample_bounds"], "second_moments": moment_bounds,
                "declared_second_moments": bounds.get("second_moments", {}),
                "paired_squared_gradient_differences": paired_differences,
                "probe_difference_scales": moment_scales, "moment_pairs": count,
                "moment_rule": "supplement_equation_19" if count >= 2 else "declared_or_envelope_fallback",
                "reference_scale": bounds["reference_scale"], "validity_record": bounds["validity_record"],
                "empirical_second_moments": {agent: float(np.mean(np.square(norms)))
                                              for agent, norms in empirical_norms.items()},
                "empirical_sample_maxima": {agent: max(norms) for agent, norms in empirical_norms.items()},
                "residual_definition": ("repeated_paired_seed_completed_transition_secants" if transition_probes
                                         else "repeated_paired_seed_entry_probe_transition_secants"),
                "direction_distribution": "normalized_gaussian_rank_one_product_direction",
                "base_state": "base_state.npz", "baseline": baseline.record(), "repeat_pairs": 2,
                "settings": dict(settings), "seed": seed, "work": add_work(work),
                "seconds": time.perf_counter() - start}
    write_json(root / "manifest.json", manifest)
    return manifest


def load_probe_calibration(path, settings=None):
    import json
    path = Path(path)
    manifest = json.loads(path.read_text())
    settings = dict(manifest.get("settings", {}), **(settings or {}))
    probes = []
    for item in manifest["probes"]:
        if abs(float(item["direction_norm"]) - 1) > 1e-8:
            raise ValueError("Calibration probe direction has unit product norm")
        positive, first = load_factors(path.parent / item["positive"])
        negative, second = load_factors(path.parent / item["negative"])
        if (first.get("paired_seed") != second.get("paired_seed")
                or first.get("policy_version") != manifest["available_version"]
                or second.get("policy_version") != manifest["available_version"]):
            raise ValueError("Calibration probes require matching seeds and incumbent versions")
        probes.append((float(item["radius"]), positive, negative))
    return calibration_from_probes(
        probes, manifest["secant_residuals"],
        {int(agent): value for agent, value in manifest["sample_bounds"].items()}, manifest["reference_scale"],
        {int(agent): value for agent, value in manifest.get("second_moments", {}).items()},
        quantile=settings.get("residual_quantile", 0.90), floor_fraction=settings.get("scale_floor_fraction", 0.05),
        available_version=int(manifest["available_version"]),
        norm_threshold=settings.get("norm_threshold", 1e-12),
        failure_probability=settings.get("failure_probability", 0.01),
        provenance={"probe_manifest_sha256": file_digest(path), "validity_record": manifest["validity_record"],
                    "empirical_second_moments": manifest.get("empirical_second_moments", {}),
                    "residual_definition": manifest.get("residual_definition")})
