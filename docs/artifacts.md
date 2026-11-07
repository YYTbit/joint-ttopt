# Input preparation

## Required inputs

Deployment uses an action-head checkpoint, entry normalization, calibration records, and a benchmark environment configuration. Download model and benchmark dependencies separately. `configs/sources.json` pins benchmark source revisions.

The action-head NPZ contains `weight`, optional `bias`, and JSON `metadata`. `collect-entry` generates labels from complete native training episodes using the base model's legal-action selections. Its environment configuration declares `split: entry_training`, a training task identifier, and a training partner-schedule identifier. Deployment tasks, schedules, and seeds use separate identifiers. The collector also writes the entry return normalization. `train-head` then fits the frozen action head from these labels.

```bash
python -m joint_ttopt collect-entry --model "${MODEL}" \
  --environment data/entry_environment.yaml \
  --renderer data/renderer.json --fallback-spans data/benchmark_return_spans.json \
  --output data/entry/${MODEL}
python -m joint_ttopt train-head --model "${MODEL}" \
  --training-data data/entry/${MODEL}/action_labels.jsonl \
  --renderer data/renderer.json --output checkpoints/${MODEL}/action_head.npz
```

Use `MODEL=Qwen3-8B`, `MODEL=Llama-3.1-8B`, or `MODEL=gpt-oss-20b` throughout the workflow. Each profile resolves its native checkpoint from `configs/paper.yaml`. Qwen3 uses non-thinking mode; gpt-oss retains its released reasoning-effort default. Collection retains the resolved model revision and native tokenizer. The trained head uses that same revision and tokenizer.

Renderer settings define `discrete_tokens`, `numerical_ranges`, `bins`, `history_turns`, and `chat_template`. Reserved tokens are present in the tokenizer vocabulary. Normalization contains an `agents` mapping with per-agent `lower` and `upper` return bounds. A degenerate interval supplies a positive `fallback_span`.

## Calibration

The bounds input contains `sample_bounds`, optional `second_moments`, `reference_scale`, and `validity_record`. `prepare-envelopes` computes trajectory norm envelopes from declared advantage, per-block score, state, horizon, and utility-span bounds. Probe pairs supply the squared gradient differences and recorded difference scale used in supplementary Equation (19). The moment scale enters the conditional confidence calculation; its calibration-validity condition is retained in the record. Bounds are supplied before deployment sampling.

```bash
python -m joint_ttopt prepare-envelopes \
  --specification data/envelope_specification.json --output data/gradient_bounds.json
python -m joint_ttopt record-probes \
  --environment data/environment.yaml --model "${MODEL}" \
  --head checkpoints/${MODEL}/action_head.npz --normalization data/entry/${MODEL}/normalization.json \
  --bounds data/gradient_bounds.json --output calibration/${MODEL}/probes
python -m joint_ttopt calibrate \
  --manifest calibration/${MODEL}/probes/manifest.json --output calibration/${MODEL}/reference.json
```

The envelope specification declares `reference_scale`, `validity_record`, and an `agents` mapping. Each agent supplies `advantage_bound`, per-block `score_bounds` and `state_bounds`, the normalization `span`, `horizon`, and `discount`; `regularization` defaults to the paper value. The bounds describe the deployment neighborhood. `prepare-envelopes` evaluates supplementary Equation (21).

The paper configuration refreshes calibration at each update round. Probes use the incumbent fast weights and a frozen history-baseline snapshot; paired seeds are reused across the two states. Completed transitions provide the repeated secant residuals after the entry round. Probe recording restores the policy and action-sampling random state. `calibration.refresh_interval: 0` reuses the supplied record. Every episode records action forwards, gradient replay forwards and backwards, calibration trajectories, and elapsed stage times; these counts come from executed trajectories.

## Environment integration

An environment factory exposes `agents`, `name`, `version`, `reset(seed)`, `step(actions)`, and `close()`. `agents` contains integer participant identifiers, and `version` identifies the environment release and task configuration. Use the benchmark's released observations, legal actions and native scoring rules.

Both `reset` and `step` return a `Frame` from `joint_ttopt.environments`. It contains participant-local `observations`, `legal_actions`, `rewards`, `roles`, and visible `histories`, together with the public `task`, `terminated`, and native `metrics`. Legal actions map integer action identifiers to their meanings. The action head and renderer use the same action vocabulary during entry training and deployment. Terminal frames retain every participant's reward.

An environment can declare `action_count` for its complete action vocabulary, including actions absent from the collected trajectories. `ParallelEnvironment` derives this value from its action catalog, and the JSON worker carries it in `describe`. Entry training retains this vocabulary size.

The configuration declares `factory` in `module:function` form, constructor `options`, `task_id`, and `partner_schedule_id`. Entry collection additionally declares `split: entry_training`; use separate task, partner-schedule, and seed identifiers for deployment.

Create `data/environment.yaml` with these fields for deployment. Create
`data/entry_environment.yaml` with the entry-training identifiers and split for
entry collection. The `factory` value names the benchmark adapter's importable
constructor; `options` supplies its keyword arguments.

`JsonEnvironment` connects a worker in a separate environment. Set its factory to `joint_ttopt.environments:JsonEnvironment` and pass a `command` argument list through `options`. Each request has `operation` equal to `describe`, `reset`, or `step`. `describe` returns `agents`, `name`, `version`, and optional `data_kind`; `reset` receives `seed`; `step` receives `actions`. The latter two operations return serialized `Frame` fields. `serve` provides the corresponding worker loop. Diagnostic output goes to stderr.

`ParallelEnvironment` connects environments using the standard parallel reset/step interface. Its action catalog maps integer identifiers to `meaning` and the complete `native` action, including language or tool arguments when applicable. Its metric callback returns the official terminal metrics.

## Evaluation

Native scores come from the benchmark evaluator. Use the names and units in [benchmarks.yaml](../configs/benchmarks.yaml) when exporting measured runs. `metrics`, `normalize`, and `summarize` process those run outputs. Normalization uses measured frozen and strongest-baseline anchors under matched tasks and configurations.

Use `--resume results/${MODEL}/joint/state.npz` with the original deployment command to continue a saved run. Metric export reads episodes committed by that checkpoint.

## Resuming training

Repeat entry collection or probe recording with the original inputs and output directory to continue from the committed cursor. Each complete trajectory is saved before collection advances. Native model and tokenizer identities, environment versions, seeds, and execution settings remain bound to the collection.

Resume action-head training with the same data and training settings:

```bash
python -m joint_ttopt train-head --model "${MODEL}" \
  --training-data data/entry/${MODEL}/action_labels.jsonl \
  --renderer data/renderer.json --output checkpoints/${MODEL}/action_head.npz \
  --resume checkpoints/${MODEL}/action_head.training.pt
```

The saved training state restores the head, native optimizer, RNG state, and data position. Deployment loads the completed head and its tokenizer. Calibration retains paired probe measurements and restores the incumbent policy and sampling state. A matching entry calibration is reused at the first round, and its recorded work enters the deployment accounting. Subsequent refreshes follow `configs/paper.yaml`.
