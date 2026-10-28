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
  --environment configs/avalon.yaml --model Qwen3-8B \
  --head checkpoints/action_head.npz --normalization data/entry_normalization.json \
  --bounds data/gradient_bounds.json --output calibration/probes
python -m joint_ttopt calibrate \
  --manifest calibration/probes/manifest.json --output calibration/reference.json
```

The paper configuration refreshes calibration at each update round. Probes use the incumbent fast weights and a frozen history-baseline snapshot; paired seeds are reused across the two states. Completed transitions provide the repeated secant residuals after the entry round. Probe recording restores the policy and action-sampling random state. `calibration.refresh_interval: 0` reuses the supplied record. Every episode records action forwards, gradient replay forwards and backwards, calibration trajectories, and elapsed stage times; these counts come from executed trajectories.

Existing gradient records can be processed with `update`. NPZ inputs use the archive format implemented in `joint_ttopt.storage`.

## Benchmark integration

Avalon uses the released engine. Decrypto requires a clue catalog mapping action identifiers to `meaning` and a three-clue `native` action. Other benchmark adapters expose `agents`, `name`, `version`, `reset(seed)`, `step(actions)`, and `close()`, with a `module:function` factory or a JSONL subprocess interface.

A frame provides local observations, legal actions, rewards, termination status, task rules, private roles, visible histories, and native metrics. Evaluation forwards actions to the corresponding benchmark role and uses its released scoring rules.

Use `--resume results/joint/state.npz` with the original run configuration to restore a checkpoint. `metrics`, `normalize`, and `summarize` process measured run outputs.
