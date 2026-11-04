# Input preparation

## Required inputs

Deployment uses an action-head checkpoint, entry normalization, calibration records, and a benchmark environment configuration. Download model and benchmark dependencies separately. `configs/sources.json` pins benchmark source revisions.

The action-head NPZ contains `weight`, optional `bias`, and JSON `metadata`. `collect-entry` generates labels from complete native training episodes using the base model's legal-action selections. Its environment configuration declares `split: entry_training`, a training task identifier, and a training partner-schedule identifier. Deployment tasks, schedules, and seeds use separate identifiers. The collector also writes the entry return normalization. `train-head` then fits the frozen action head from these labels.

```bash
python -m joint_ttopt collect-entry \
  --environment data/entry_environment.yaml --checkpoint Qwen/Qwen3-8B \
  --renderer data/renderer.json --fallback-spans data/benchmark_return_spans.json \
  --output data/entry
python -m joint_ttopt train-head \
  --checkpoint Qwen/Qwen3-8B --revision COMMIT \
  --training-data data/entry/action_labels.jsonl \
  --renderer data/renderer.json --output checkpoints/action_head.npz
```

Renderer settings define `discrete_tokens`, `numerical_ranges`, `bins`, `history_turns`, and `chat_template`. Reserved tokens are present in the tokenizer vocabulary. Normalization contains an `agents` mapping with per-agent `lower` and `upper` return bounds. A degenerate interval supplies a positive `fallback_span`.

## Calibration

The bounds input contains `sample_bounds`, optional `second_moments`, `reference_scale`, and `validity_record`. `prepare-envelopes` computes trajectory norm envelopes from declared advantage, per-block score, state, horizon, and utility-span bounds. Probe pairs supply the squared gradient differences and recorded difference scale used in supplementary Equation (19). The moment scale enters the conditional confidence calculation; its calibration-validity condition is retained in the record. Bounds are supplied before deployment sampling.

```bash
python -m joint_ttopt record-probes \
  --environment data/environment.yaml --model Qwen3-8B \
  --head checkpoints/action_head.npz --normalization data/entry/normalization.json \
  --bounds data/gradient_bounds.json --output calibration/probes
python -m joint_ttopt calibrate \
  --manifest calibration/probes/manifest.json --output calibration/reference.json
```

The paper configuration refreshes calibration at each update round. Probes use the incumbent fast weights and a frozen history-baseline snapshot; paired seeds are reused across the two states. Completed transitions provide the repeated secant residuals after the entry round. Probe recording restores the policy and action-sampling random state. `calibration.refresh_interval: 0` reuses the supplied record. Every episode records action forwards, gradient replay forwards and backwards, calibration trajectories, and elapsed stage times; these counts come from executed trajectories.

## Environment integration

An environment factory exposes `agents`, `name`, `version`, `reset(seed)`, `step(actions)`, and `close()`. `agents` contains integer participant identifiers, and `version` identifies the environment release and task configuration. Use the benchmark's released observations, legal actions and native scoring rules.

Both `reset` and `step` return a `Frame` from `joint_ttopt.environments`. It contains participant-local `observations`, `legal_actions`, `rewards`, `roles`, and visible `histories`, together with the public `task`, `terminated`, and native `metrics`. Legal actions map integer action identifiers to their meanings. The action head and renderer use the same action vocabulary during entry training and deployment. Terminal frames retain every participant's reward.

The configuration declares `factory` in `module:function` form, constructor `options`, `task_id`, and `partner_schedule_id`. Entry collection additionally declares `split: entry_training`; use separate task, partner-schedule, and seed identifiers for deployment.

Configure the benchmark factory in `data/environment.yaml` before the preparation and deployment commands above.

`JsonEnvironment` connects a worker in a separate environment. Set its factory to `joint_ttopt.environments:JsonEnvironment` and pass a `command` argument list through `options`. Each request has `operation` equal to `describe`, `reset`, or `step`. `describe` returns `agents`, `name`, `version`, and optional `data_kind`; `reset` receives `seed`; `step` receives `actions`. The latter two operations return serialized `Frame` fields. `serve` provides the corresponding worker loop. Diagnostic output goes to stderr.

`ParallelEnvironment` connects environments using the standard parallel reset/step interface. Its action catalog maps integer identifiers to `meaning` and the complete `native` action, including language or tool arguments when applicable. Its metric callback returns the official terminal metrics.

## Evaluation

Native scores come from the benchmark evaluator. Use the names and units in [benchmarks.yaml](../configs/benchmarks.yaml) when exporting measured runs. `metrics`, `normalize`, and `summarize` process those run outputs. Normalization uses measured frozen and strongest-baseline anchors under matched tasks and configurations.

Use `--resume results/joint/state.npz` with the original deployment command to continue a saved run.
