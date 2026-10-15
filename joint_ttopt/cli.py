import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from .confidence import RewardScale
from .optimizer import Coordinator, GradientBatch
from .storage import (file_digest, json_value, load_calibration, load_checkpoint,
                      load_factors, save_checkpoint, write_json)


def read_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration is a mapping")
    return value


def update_command(args):
    gradients, record = load_factors(args.gradients)
    settings = read_config(args.config)["optimization"]
    calibration = load_calibration(args.calibration)
    if args.state:
        coordinator, extra = load_checkpoint(args.state)
    else:
        if args.rank_budget is None:
            raise ValueError("A new state requires an explicit shared rank budget")
        agents = sorted({agent for agent, block in gradients})
        caps = ({int(agent): int(value) for agent, value in json.loads(Path(args.agent_caps).read_text()).items()}
                if args.agent_caps else dict.fromkeys(agents, math.ceil(3 * args.rank_budget / len(agents))))
        coordinator = Coordinator({key: value.shape for key, value in gradients.items()}, args.rank_budget, caps,
                                  settings["step_margin"], settings["error_budget"])
        extra = {}
    version = int(record["policy_version"])
    count = int(record["sample_count"])
    norms = {int(agent): values for agent, values in record.get("sample_norms", {}).items()}
    errors = {int(agent): value for agent, value in record.get("approximation_errors", {}).items()}
    result = coordinator.step(GradientBatch(version, gradients, count, norms, errors), calibration)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    save_checkpoint(destination / "state.npz", coordinator, extra)
    write_json(destination / "update.json", result)
    print(json.dumps(json_value(result), indent=2, allow_nan=False))


def calibrate_command(args):
    from .probes import load_probe_calibration
    result = load_probe_calibration(args.manifest, read_config(args.config).get("calibration", {}))
    write_json(args.output, asdict(result))
    print(json.dumps(json_value(asdict(result)), indent=2))


def probes_command(args):
    from .environments import load_environment
    from .policy import FrozenActionPolicy
    from .probes import record_probes
    config = read_config(args.config)
    configuration = read_config(args.environment)
    environment = load_environment(configuration["factory"], configuration.get("options", {}))
    try:
        profile = config["models"][args.model]
        policy = FrozenActionPolicy.from_pretrained(profile["checkpoint"], args.head, args.seed,
                                                    profile.get("revision"), args.device_map,
                                                    config["model_execution"]["dtype"],
                                                    temperature=config["model_execution"]["action_temperature"],
                                                    max_context=config["model_execution"]["context_tokens"])
        normalization = json.loads(Path(args.normalization).read_text())
        scales = {int(agent): RewardScale.from_interval(value["lower"], value["upper"], value.get("fallback_span"))
                  for agent, value in normalization["agents"].items()}
        settings = dict(config["deployment"], **config["calibration"],
                        regularization=config["optimization"]["regularization"])
        result = record_probes(environment, policy, scales, settings,
                               json.loads(Path(args.bounds).read_text()), args.output, args.seed)
        print(json.dumps({"paired_probes": len(result["probes"])}))
    finally:
        environment.close()


def deploy_command(args):
    from .deployment import run_deployment
    from .environments import load_environment
    from .policy import FrozenActionPolicy
    config = read_config(args.config)
    environment_config = read_config(args.environment)
    profile = config["models"][args.model]
    environment = load_environment(environment_config["factory"], environment_config.get("options", {}))
    try:
        policy = FrozenActionPolicy.from_pretrained(profile["checkpoint"], args.head, args.seed,
                                                    profile.get("revision"), args.device_map,
                                                    config["model_execution"]["dtype"],
                                                    temperature=config["model_execution"]["action_temperature"],
                                                    max_context=config["model_execution"]["context_tokens"])
        if len(policy.blocks) != profile["adapted_blocks"]:
            raise ValueError("Backbone block count differs from the paper configuration")
        normalization = json.loads(Path(args.normalization).read_text())
        scales = {int(agent): RewardScale.from_interval(value["lower"], value["upper"], value.get("fallback_span"))
                  for agent, value in normalization["agents"].items()}
        settings = dict(config["optimization"], **config["deployment"],
                        calibration=config["calibration"],
                        task_id=environment_config["task_id"], benchmark=environment.name,
                        partner_schedule_id=environment_config["partner_schedule_id"],
                        environment_configuration_sha256=file_digest(args.environment),
                        normalization_sha256=file_digest(args.normalization))
        if args.calibration_bounds:
            settings["calibration_bounds"] = json.loads(Path(args.calibration_bounds).read_text())
        if args.variant:
            variants = read_config(args.ablations)
            entries = dict(variants["coarse"], **variants["fine"])
            override = dict(entries[args.variant])
            args.method = override.pop("method", args.method)
            if "rank_multiplier" in override:
                settings["total_rank"] = int(profile["reference_rank"] * override.pop("rank_multiplier"))
            if "history_turns" in override:
                policy.renderer.history_turns = int(override.pop("history_turns"))
            settings.update(override)
            settings["variant"] = args.variant
        if args.episodes is not None:
            settings["episodes"] = args.episodes
        if settings["task_id"] in policy.metadata.get("training_tasks", []):
            raise ValueError("Deployment task overlaps the action-head entry-training split")
        if settings["partner_schedule_id"] in policy.metadata.get("training_partner_schedules", []):
            raise ValueError("Deployment partner schedule overlaps the entry-training split")
        result = run_deployment(environment, policy, load_calibration(args.calibration),
                                file_digest(args.calibration), scales, settings, args.output,
                                args.seed, args.method, args.resume)
        print(json.dumps({"completed_episodes": len(result), "method": args.method}))
    finally:
        environment.close()


def fetch_command(args):
    entries = json.loads(Path(args.sources).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.names or sorted(entries):
        item = entries[name]
        destination = root / item["directory"]
        if not destination.exists():
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", item["repository"], str(destination)], check=True)
        subprocess.run(["git", "fetch", "origin", item["revision"]], cwd=destination, check=True)
        subprocess.run(["git", "checkout", "--detach", item["revision"]], cwd=destination, check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination, text=True).strip()
        if revision != item["revision"]:
            raise ValueError("Fetched benchmark revision differs from its source pin")
        print(name + " " + revision)


def external_command(args):
    request = read_config(args.request)
    request["checkpoint_hashes"] = {name: file_digest(path) for name, path in request.pop("checkpoint_files", {}).items()}
    response = subprocess.run(shlex.split(args.command), input=json.dumps(request, allow_nan=False),
                              text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in response.stdout.splitlines() if line.strip()]
    if not rows or any(row.get("data_kind") != "measured_run" for row in rows):
        raise ValueError("External baseline returns measured native run records")
    required = {"benchmark", "task", "backbone", "method", "seed", "episode", "native_score", "comparison_signature"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("External baseline record omitted comparison fields")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Joint Test-Time Optimization from Multi-Agent Interaction of Large Language Models")
    commands = parser.add_subparsers(dest="operation", required=True)
    update = commands.add_parser("update")
    update.add_argument("--gradients", required=True)
    update.add_argument("--calibration", required=True)
    update.add_argument("--config", default="configs/paper.yaml")
    update.add_argument("--state")
    update.add_argument("--rank-budget", type=int)
    update.add_argument("--agent-caps")
    update.add_argument("--output", required=True)
    update.set_defaults(function=update_command)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--manifest", required=True)
    calibration.add_argument("--config", default="configs/paper.yaml")
    calibration.add_argument("--output", required=True)
    calibration.set_defaults(function=calibrate_command)
    probes = commands.add_parser("record-probes")
    probes.add_argument("--config", default="configs/paper.yaml")
    probes.add_argument("--environment", required=True)
    probes.add_argument("--model", choices=["Qwen3-8B", "Llama-3.1-8B", "gpt-oss-20b"], required=True)
    probes.add_argument("--head", required=True)
    probes.add_argument("--normalization", required=True)
    probes.add_argument("--bounds", required=True)
    probes.add_argument("--seed", type=int, default=0)
    probes.add_argument("--device-map", default="auto")
    probes.add_argument("--output", required=True)
    probes.set_defaults(function=probes_command)
    deployment = commands.add_parser("deploy")
    deployment.add_argument("--config", default="configs/paper.yaml")
    deployment.add_argument("--environment", required=True)
    deployment.add_argument("--model", choices=["Qwen3-8B", "Llama-3.1-8B", "gpt-oss-20b"], required=True)
    deployment.add_argument("--head", required=True)
    deployment.add_argument("--calibration", required=True)
    deployment.add_argument("--calibration-bounds")
    deployment.add_argument("--normalization", required=True)
    deployment.add_argument("--method", choices=["joint", "frozen", "equal_share", "rank_matched", "reset", "shared_feedback"], default="joint")
    deployment.add_argument("--seed", type=int, default=0)
    deployment.add_argument("--episodes", type=int)
    deployment.add_argument("--variant")
    deployment.add_argument("--ablations", default="configs/ablations.yaml")
    deployment.add_argument("--device-map", default="auto")
    deployment.add_argument("--resume")
    deployment.add_argument("--output", required=True)
    deployment.set_defaults(function=deploy_command)
    source = commands.add_parser("fetch")
    source.add_argument("--sources", default="configs/sources.json")
    source.add_argument("--names", nargs="+")
    source.add_argument("--output", default="external")
    source.set_defaults(function=fetch_command)
    inspection = commands.add_parser("inspect")
    inspection.add_argument("--state", required=True)
    def inspect_command(args):
        coordinator, extra = load_checkpoint(args.state)
        print(json.dumps(json_value({"version": coordinator.version, "total_rank": coordinator.total_rank,
                                     "agent_caps": coordinator.agent_caps, "ranks": coordinator.ranks,
                                     "factor_entries": sum(state.storage for state in coordinator.states.values()),
                                     "metadata": extra}), indent=2))
    inspection.set_defaults(function=inspect_command)
    external = commands.add_parser("external")
    external.add_argument("--command", required=True)
    external.add_argument("--request", required=True)
    external.add_argument("--output", required=True)
    external.set_defaults(function=external_command)
    service = commands.add_parser("serve")
    service.add_argument("--environment", required=True)
    def serve_command(args):
        from .environments import load_environment, serve
        configuration = read_config(args.environment)
        environment = load_environment(configuration["factory"], configuration.get("options", {}))
        try:
            serve(environment, sys.stdin, sys.stdout)
        finally:
            environment.close()
    service.set_defaults(function=serve_command)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("runs", nargs="+")
    metrics.add_argument("--horizons", nargs="+", type=int, default=[10, 25, 50, 100])
    metrics.add_argument("--metric", required=True)
    metrics.add_argument("--output", required=True)
    def write_rows(path, rows):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(json_value(row), allow_nan=False) + "\n" for row in rows))
    def metrics_command(args):
        from .statistics import deployment_metrics
        write_rows(args.output, deployment_metrics(args.runs, args.horizons, args.metric))
    metrics.set_defaults(function=metrics_command)
    normalization = commands.add_parser("normalize")
    normalization.add_argument("inputs", nargs="+")
    normalization.add_argument("--baselines", nargs="+", required=True)
    normalization.add_argument("--anchors", required=True)
    normalization.add_argument("--output", required=True)
    def normalize_command(args):
        from .statistics import apply_anchors, build_anchors, read_metric_rows
        rows = read_metric_rows(args.inputs)
        anchors = build_anchors(rows, args.baselines)
        write_json(args.anchors, anchors)
        write_rows(args.output, apply_anchors(rows, anchors))
    normalization.set_defaults(function=normalize_command)
    summary = commands.add_parser("summarize")
    summary.add_argument("inputs", nargs="+")
    summary.add_argument("--output", required=True)
    def summarize_command(args):
        from .statistics import aggregate_games, read_metric_rows, write_summary
        result = write_summary(args.output, aggregate_games(read_metric_rows(args.inputs)))
        print(json.dumps(result, indent=2))
    summary.set_defaults(function=summarize_command)
    head = commands.add_parser("train-head")
    head.add_argument("--checkpoint", required=True)
    head.add_argument("--revision")
    head.add_argument("--training-data", required=True)
    head.add_argument("--renderer", required=True)
    head.add_argument("--output", required=True)
    head.add_argument("--seed", type=int, default=0)
    head.add_argument("--epochs", type=int, default=5)
    head.add_argument("--learning-rate", type=float, default=0.0003)
    head.add_argument("--device", default="cpu")
    def head_command(args):
        from .heads import train_action_head
        train_action_head(args.checkpoint, args.revision, args.training_data, args.output,
                          json.loads(Path(args.renderer).read_text()), args.seed,
                          args.epochs, args.learning_rate, args.device)
    head.set_defaults(function=head_command)
    entry = commands.add_parser("collect-entry")
    entry.add_argument("--environment", required=True)
    entry.add_argument("--checkpoint", required=True)
    entry.add_argument("--revision")
    entry.add_argument("--renderer", required=True)
    entry.add_argument("--episodes", type=int, default=100)
    entry.add_argument("--seed", type=int, default=0)
    entry.add_argument("--fallback-spans", required=True)
    entry.add_argument("--device-map", default="auto")
    entry.add_argument("--output", required=True)
    def entry_command(args):
        from transformers import AutoTokenizer
        from .environments import load_environment
        from .preparation import GenerativeActionTeacher, collect_entry
        from .renderer import ObservationRenderer
        specification = read_config(args.environment)
        if specification.get("split") != "entry_training" or args.episodes < 1:
            raise ValueError("Collection requires an entry-training task and a positive episode count")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
        renderer = ObservationRenderer(tokenizer, **json.loads(Path(args.renderer).read_text()))
        teacher = GenerativeActionTeacher(args.checkpoint, args.revision, args.device_map)
        environment = load_environment(specification["factory"], specification.get("options", {}))
        try:
            labels, episodes = collect_entry(environment, teacher, renderer, specification, args.output,
                                              list(range(args.seed, args.seed + args.episodes)),
                                              fallback_spans=json.loads(Path(args.fallback_spans).read_text()))
            print(json.dumps({"episodes": len(episodes), "labels": len(labels)}))
        finally:
            environment.close()
    entry.set_defaults(function=entry_command)
    envelopes = commands.add_parser("prepare-envelopes")
    envelopes.add_argument("--specification", required=True)
    envelopes.add_argument("--output", required=True)
    def envelopes_command(args):
        from .preparation import declared_envelopes
        write_json(args.output, declared_envelopes(json.loads(Path(args.specification).read_text())))
    envelopes.set_defaults(function=envelopes_command)
    args = parser.parse_args(argv)
    args.function(args)
