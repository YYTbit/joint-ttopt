import csv
import json
from pathlib import Path

import numpy as np


def seed_summary(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Seed summary requires at least two finite independent run values")
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "seeds": len(values)}


def normalized_gain(score, frozen_anchor, baseline_anchor):
    if not np.isfinite([score, frozen_anchor, baseline_anchor]).all() or baseline_anchor <= frozen_anchor:
        raise ValueError("Normalized gain requires a positive baseline improvement over Frozen")
    return float((score - frozen_anchor) / (baseline_anchor - frozen_anchor))


def deployment_metrics(directories, horizons, metric):
    rows = []
    for directory in directories:
        episodes = [json.loads(path.read_text()) for path in sorted(Path(directory).glob("episode_*.json"))]
        if not episodes or any(row.get("data_kind") != "measured_run" for row in episodes):
            raise ValueError("Metric export requires measured native deployment records")
        first = episodes[0]
        for horizon in horizons:
            prefix = [row for row in episodes if row["episode"] <= horizon]
            if sorted(row["episode"] for row in prefix) != list(range(1, horizon + 1)):
                raise ValueError("Metric export requires each episode in the requested horizon")
            fields = ("signature", "comparison_signature", "benchmark", "task", "backbone", "method", "seed")
            if any(any(row.get(field) != first.get(field) for field in fields) for row in prefix):
                raise ValueError("A deployment directory mixes distinct comparison configurations")
            values = np.asarray([row["native_metrics"][metric] for row in prefix], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Native metric export contains a nonfinite score")
            rows.append(dict({field: first[field] for field in fields}, data_kind="measured_run",
                             episode=horizon, native_score=float(np.mean(values)), native_metric=metric))
    return rows


def build_anchors(rows, baseline_methods, anchor_horizon=50):
    if not baseline_methods or "joint" in baseline_methods or "frozen" in baseline_methods:
        raise ValueError("Anchor baselines are a declared set of comparison methods")
    groups = {}
    for row in rows:
        if row.get("data_kind") != "measured_run":
            raise ValueError("Normalization requires measured native run records")
        if row["episode"] == anchor_horizon and row["method"] in {"frozen", *baseline_methods}:
            key = (row["benchmark"], row["task"], row["backbone"])
            group = groups.setdefault(key, {})
            seeds = group.setdefault(row["method"], {})
            if row["seed"] in seeds:
                raise ValueError("Anchor input repeats a method and seed")
            seeds[row["seed"]] = row
    anchors = []
    for key, methods in sorted(groups.items()):
        if set(methods) != {"frozen", *baseline_methods}:
            raise ValueError("Anchor input omitted Frozen or a declared baseline")
        seeds = set(methods["frozen"])
        if len(seeds) < 2 or any(set(values) != seeds for values in methods.values()):
            raise ValueError("Anchors use matched independent seeds for each method")
        signatures = {row["comparison_signature"] for values in methods.values() for row in values.values()}
        if len(signatures) != 1 or None in signatures:
            raise ValueError("Anchor methods use different comparison configurations")
        scores = {method: float(np.mean([row["native_score"] for row in values.values()]))
                  for method, values in methods.items()}
        best = max(sorted(baseline_methods), key=lambda method: scores[method])
        normalized_gain(scores[best], scores["frozen"], scores[best])
        anchors.append(dict(zip(("benchmark", "task", "backbone"), key),
                            frozen_anchor=scores["frozen"], baseline_anchor=scores[best],
                            strongest_baseline=best, anchor_horizon=anchor_horizon,
                            comparison_signature=next(iter(signatures)), seeds=sorted(seeds)))
    if not anchors:
        raise ValueError("Anchor input has no complete game and backbone comparison")
    return anchors


def apply_anchors(rows, anchors):
    lookup = {(row["benchmark"], row["task"], row["backbone"]): row for row in anchors}
    result = []
    for row in rows:
        anchor = lookup[(row["benchmark"], row["task"], row["backbone"])]
        if row["comparison_signature"] != anchor["comparison_signature"]:
            raise ValueError("Score and normalization anchors use different comparison configurations")
        result.append(dict(row, normalized_gain=normalized_gain(row["native_score"], anchor["frozen_anchor"],
                                                                 anchor["baseline_anchor"])))
    return result


def aggregate_games(rows):
    seen, groups = set(), {}
    for row in rows:
        identity = (row["benchmark"], row["task"], row["backbone"], row["method"], row["seed"], row["episode"])
        if identity in seen:
            raise ValueError("Comparison contains a duplicate game, seed, and horizon value")
        seen.add(identity)
        key = (row["benchmark"], row["backbone"], row["method"], row["seed"], row["episode"])
        groups.setdefault(key, {})[row["task"]] = float(row["normalized_gain"])
    task_sets = {}
    benchmark_means = {}
    for key, games in groups.items():
        benchmark, backbone, method, seed, episode = key
        reference = (benchmark, backbone, episode)
        if reference in task_sets and task_sets[reference] != set(games):
            raise ValueError("Compared runs have different benchmark game sets")
        task_sets[reference] = set(games)
        benchmark_means.setdefault((backbone, method, seed, episode), {})[benchmark] = float(np.mean(list(games.values())))
    benchmark_sets = {}
    backbone_means = {}
    for key, benchmarks in benchmark_means.items():
        backbone, method, seed, episode = key
        if episode in benchmark_sets and benchmark_sets[episode] != set(benchmarks):
            raise ValueError("Compared runs have different benchmark sets")
        benchmark_sets[episode] = set(benchmarks)
        backbone_means.setdefault((method, seed, episode), {})[backbone] = float(np.mean(list(benchmarks.values())))
    backbone_sets = {}
    result = []
    for (method, seed, episode), backbones in backbone_means.items():
        if episode in backbone_sets and backbone_sets[episode] != set(backbones):
            raise ValueError("Compared runs have different backbone sets")
        backbone_sets[episode] = set(backbones)
        result.append({"method": method, "seed": seed, "episode": episode,
                       "normalized_gain": float(np.mean(list(backbones.values())))})
    return result


def paired_bootstrap(first, second, level=0.05, comparisons=15, samples=10000, seed=0):
    if set(first) != set(second) or len(first) < 2:
        raise ValueError("Paired comparison requires the same set of at least two seeds")
    if not 0 < level < 1 or comparisons < 1 or samples < 1:
        raise ValueError("Invalid paired confidence settings")
    differences = np.asarray([first[key] - second[key] for key in sorted(first)], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Paired comparison contains nonfinite differences")
    generator = np.random.default_rng(seed)
    estimates = generator.choice(differences, size=(samples, len(differences)), replace=True).mean(axis=1)
    tail = level / (2 * comparisons)
    lower, upper = np.quantile(estimates, [tail, 1 - tail])
    return {"difference": float(np.mean(differences)), "lower": float(lower), "upper": float(upper),
            "paired_seeds": len(differences), "family_level": level, "comparisons": comparisons}


def read_metric_rows(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("data_kind") != "measured_run":
                    raise ValueError("Statistical summaries require measured run records")
                rows.append(row)
    return rows


def write_summary(path, rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["episode"]), {})[row["seed"]] = row["normalized_gain"]
    output = []
    for (method, episode), values in sorted(groups.items()):
        output.append(dict(seed_summary(list(values.values())), method=method, episode=episode))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "episode", "mean", "std", "seeds"])
        writer.writeheader()
        writer.writerows(output)
    return output
