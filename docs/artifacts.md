# Input preparation

## Required inputs

Deployment uses an action-head checkpoint, entry normalization, calibration records, and a benchmark environment configuration. Download model and benchmark dependencies separately. `configs/sources.json` pins benchmark source revisions.

The action-head NPZ contains `weight`, optional `bias`, and JSON `metadata`. Use `train-head` with JSONL entry-training episodes containing `split`, `input_ids`, `legal_actions`, `action`, `task_id`, and `partner_schedule_id`. The split label is `entry_training`.

```bash
python -m joint_ttopt train-head \
  --checkpoint Qwen/Qwen3-8B --revision COMMIT \
  --training-data data/entry_training.jsonl \
  --renderer data/renderer.json --output checkpoints/action_head.npz
```

Renderer settings define `discrete_tokens`, `numerical_ranges`, `bins`, `history_turns`, and `chat_template`. Reserved tokens are present in the tokenizer vocabulary. Normalization contains an `agents` mapping with per-agent `lower` and `upper` return bounds. A degenerate interval supplies a positive `fallback_span`.

## Calibration

The bounds input contains `sample_bounds`, optional `second_moments`, `reference_scale`, and `validity_record`. Bounds are supplied before deployment sampling.

```bash
python -m joint_ttopt record-probes \
  --environment configs/avalon.yaml --model Qwen3-8B \
  --head checkpoints/action_head.npz --normalization data/entry_normalization.json \
  --bounds data/gradient_bounds.json --output calibration/probes
python -m joint_ttopt calibrate \
  --manifest calibration/probes/manifest.json --output calibration/reference.json
```

Existing gradient records can be processed with `update`. NPZ inputs use the archive format implemented in `joint_ttopt.storage`.

## Benchmark integration

Avalon uses the released engine. Decrypto requires a clue catalog mapping action identifiers to `meaning` and a three-clue `native` action. Other benchmark adapters expose `agents`, `name`, `version`, `reset(seed)`, `step(actions)`, and `close()`, with a `module:function` factory or a JSONL subprocess interface.

A frame provides local observations, legal actions, rewards, termination status, task rules, private roles, visible histories, and native metrics. Evaluation forwards actions to the corresponding benchmark role and uses its released scoring rules.

Use `--resume results/joint/state.npz` with the original run configuration to restore a checkpoint. `metrics`, `normalize`, and `summarize` process measured run outputs. Runtime checkpoints, datasets, and results remain outside version control.
