# Joint Test-Time Optimization from Multi-Agent Interaction of Large Language Models

Large language model (LLM) agents increasingly update their parameters during
deployment to adapt to unfamiliar tasks and changing interactions. Existing
test-time training methods improve individual agents, yet simultaneous
adaptation couples their updates through shared computational resources and
evolving learning signals. To address this problem, we formulate joint test-time
optimization over persistent low-rank weights under a shared rank budget.
Specifically, we derive the exact solution of a per-round rank-constrained
proximal problem, which allocates spectral directions across agents and
parameter blocks according to their marginal values. The resulting algorithm
retains parameter states across rounds, reallocates rank as interaction feedback
changes, and adjusts update steps using observed joint gradient variation. We
establish a finite-sample lower bound on the regularized joint game gap and
prove invariance to positive affine reward rescaling. We evaluate the method on
five public multi-agent benchmarks and three LLM backbones, obtaining the
highest mean performance in all 15 settings. At 50 episodes, its normalized gain
exceeds the strongest baseline by 0.93 while using 65% of its computation. The
advantage increases from 0.42 to 1.01 as deployment extends from 10 to 100
episodes, supporting persistent joint adaptation.

## Algorithm Flow

To implement each shared-rank decision, Algorithm 1 combines interaction sampling, gradient estimation, step selection, spectral allocation, and joint commitment.

![Joint adaptation in five-player AvalonBench](figures/figure1_overview.png)

**Figure 1.** Joint adaptation in five-player AvalonBench (schematic). (a) Context-dependent experience reuse. (b) Equal per-agent rank budgets. (c) Changing learning signals. (d) Shared spectral allocation, interaction-sensitive steps, and persistent weights. Optimality holds for the per-round proximal objective. Tile heights indicate update values; solid tiles mark retained directions, totaling ten in both allocations. Agent colors identify players, whose roles remain private.

**Execution order for a deployment update.**

Initialize the persistent parameter states and the joint secant to zero. For each
deployment round, execute the following steps.

| Step | Operation | Information and output |
| --- | --- | --- |
| 1 | Set policy version | Incumbent fast weights and an action-independent baseline |
| 2 | Collect interaction | Complete joint trajectories under the incumbent policy |
| 3 | Form gradients | Recorded actions, rewards, and regularizer contributions |
| 4 | Measure secant | Current and preceding gradients with the completed joint state change |
| 5 | Set common scale | Calibrated local scale, secant value, tolerance, and step margin |
| 6 | Form proposals | Incumbent state plus the scaled empirical gradient |
| 7 | Allocate rank | Descending mode values under shared and agent capacity limits |
| 8 | Validate and commit | Finite factors, consistent policy versions, and feasible ranks |
| 9 | Record diagnostics | Proximal value, spectral margins, and conditional confidence bound |

If validation succeeds, commit the candidate states jointly. Otherwise, retain
the incumbent states. The committed state enters the next episode through the
persistent parameter updates. Return the adapted joint policy after the final round.

![Algorithm 1](figures/algorithm.png)

## Results

### Main Results

**Table 1.** Performance across five benchmarks and three backbones after 50 episodes (mean ± standard deviation over ten paired seeds). All metrics are higher-is-better; bold marks the best mean per backbone. Native metrics complement normalized gain. Paired comparisons and their confidence intervals are detailed in the supplement.

![Main results from Table 1](figures/main_results.png)

**Joint adaptation improves all benchmark-backbone settings.** Table 1 reports the highest mean for the full method in all 15 settings. Averaged over backbones, MultiAgentBench KPI rises from TEMPO’s 60.9 to 63.8, Decrypto success from 44.4 to 51.0, and AvalonBench win rate from 38.9 to 44.6. The corresponding GAMA-Bench payoff and SOTOPIA score increase from 0.617 to 0.657 and from 7.27 to 8.14.

![Performance versus horizon and compute](figures/analysis.png)

**Figure 2.** Performance versus (a) horizon and (b) compute at 50 episodes. Bars show seed standard deviations. Retr., Plan., and LoRA denote Retrieval, Planning, and LoRA TT-RL.

**Persistent adaptation benefits from continued interaction.** Figure 2(a) shows the lead over TEMPO growing from 0.42 at ten episodes to 1.01 at 100. Persistent states retain earlier feedback, while repeated allocation responds to changing update values. The common 50-episode normalization anchors keep this comparison on the same scale.

**The gain remains at comparable computation.** Figure 2(b) relates performance to cumulative compute. At 50 episodes, Ours and LoRA TT-RL use 50.12 ± 2.76 and 49.84 ± 2.74 PFLOPs, with a 1.16 difference in normalized gain; Ours uses 65% of TEMPO’s compute.

## Getting Started

### Environment Setup

```bash
conda create -n joint-ttopt python=3.10 -y
conda activate joint-ttopt
pip install -e '.[models]'
```

### Data Preparation

Download the benchmark sources:

```bash
python -m joint_ttopt fetch --output external
```

Install the corresponding environment dependencies using their released instructions.
The paper evaluates MultiAgentBench, GAMA-Bench, SOTOPIA, Decrypto, and AvalonBench.
Configure the benchmark in `data/environment.yaml`. Prepare the action head,
reward normalization, and calibration inputs with the commands in
[input preparation](docs/artifacts.md). The deployment command below uses the
paper settings from `configs/paper.yaml`.

Select one of the three backbones for preparation and deployment:

```bash
MODEL=Qwen3-8B
# MODEL=Llama-3.1-8B
# MODEL=gpt-oss-20b
```

## Training

Collect entry-training trajectories and train the frozen action head:

```bash
python -m joint_ttopt collect-entry --model "${MODEL}" \
  --environment data/entry_environment.yaml --renderer data/renderer.json \
  --fallback-spans data/benchmark_return_spans.json --output data/entry/${MODEL}

python -m joint_ttopt train-head --model "${MODEL}" \
  --training-data data/entry/${MODEL}/action_labels.jsonl \
  --renderer data/renderer.json --output checkpoints/${MODEL}/action_head.npz
```

Record calibration probes using the prepared gradient bounds:

```bash
python -m joint_ttopt record-probes --environment data/environment.yaml \
  --model "${MODEL}" --head checkpoints/${MODEL}/action_head.npz \
  --normalization data/entry/${MODEL}/normalization.json \
  --bounds data/gradient_bounds.json --output calibration/${MODEL}/probes

python -m joint_ttopt calibrate --manifest calibration/${MODEL}/probes/manifest.json \
  --output calibration/${MODEL}/reference.json
```

Run joint test-time training in the configured environment:

```bash
python -m joint_ttopt deploy --environment data/environment.yaml \
  --model "${MODEL}" --head checkpoints/${MODEL}/action_head.npz \
  --normalization data/entry/${MODEL}/normalization.json \
  --calibration calibration/${MODEL}/reference.json \
  --calibration-bounds data/gradient_bounds.json --output results/${MODEL}/joint
```

Add `--resume results/${MODEL}/joint/state.npz` to continue a saved deployment.
Entry collection and probe recording continue when their commands use the same
output directory. Action-head recovery is described in the
[preparation guide](docs/artifacts.md#resuming-training).

## Evaluation

Export the environment's native metric after the 50-episode deployment:

```bash
python -m joint_ttopt metrics results/${MODEL}/joint --horizons 50 \
  --metric METRIC --output results/${MODEL}/metrics.jsonl
```

Set `METRIC` to the native metric returned by the benchmark evaluator.
[Score aggregation](docs/artifacts.md#evaluation) describes normalization across runs.

## License

The code uses the [MIT License](LICENSE).
