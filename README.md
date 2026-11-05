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

To adapt interacting LLM agents under a shared rank budget, we formulate test-time parameter training as joint optimization over persistent low-rank weights.

1. Sample joint trajectories under the current policy version. Estimate the update field and confidence radii.
2. Set the secant estimate and step size from observed and calibrated scales. Form the block proposals and spectral mode values.
3. Greedily allocate ranks under the shared budget and agent caps. Retain the leading singular directions for each block.
4. Validate factor values, budgets, and policy versions. Commit the proposals jointly if validation succeeds; otherwise retain the incumbent state.
5. Record ranks, spectral margins, and the joint gap bound. The committed state enters the next episode.

![Joint adaptation in five-player AvalonBench](figures/figure1_overview.png)

**Figure 1.** Joint adaptation in five-player AvalonBench (schematic). (a) Context-dependent experience reuse. (b) Equal per-agent rank budgets. (c) Changing learning signals. (d) Shared spectral allocation, interaction-sensitive steps, and persistent weights. Optimality holds for the per-round proximal objective. Tile heights indicate update values; solid tiles mark retained directions, totaling ten in both allocations. Agent colors identify players, whose roles remain private.

## Results

**Table 1.** Performance across five benchmarks and three backbones after 50 episodes (mean ± standard deviation over ten paired seeds). All metrics are higher-is-better; bold marks the best mean per backbone. Native metrics complement normalized gain. Paired comparisons and their confidence intervals are detailed in the supplement.

![Main results from Table 1](figures/main_results.png)

**Joint adaptation improves all benchmark-backbone settings.** Table 1 reports the highest mean for the full method in all 15 settings. Averaged over backbones, MultiAgentBench KPI rises from TEMPO’s 60.9 to 63.8, Decrypto success from 44.4 to 51.0, and AvalonBench win rate from 38.9 to 44.6. The corresponding GAMA-Bench payoff and SOTOPIA score increase from 0.617 to 0.657 and from 7.27 to 8.14.

**The gain remains at comparable computation.** At 50 episodes, Ours and LoRA TT-RL use 50.12 ± 2.76 and 49.84 ± 2.74 PFLOPs, with a 1.16 difference in normalized gain; Ours uses 65% of TEMPO’s compute. Accounting includes 2,000 trajectory tokens and 256,000 calibration token-passes per round for both methods. Including this dominant calibration cost keeps the comparison sensitive to the complete adaptation procedure.

## Getting Started

### Environment Setup

Use Python 3.10 or later and install PyTorch for your system.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models]' \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

### Data Preparation

Download the benchmark sources and install the environment dependencies:

```bash
python -m joint_ttopt fetch --output external
```

The paper evaluates MultiAgentBench, GAMA-Bench, SOTOPIA, Decrypto, and AvalonBench.
Their upstream sources are listed in [configs/sources.json](configs/sources.json).
Prepare the action head, reward normalization, and calibration inputs using the
[preparation guide](docs/artifacts.md). Configure `data/environment.yaml` through
the [environment interface](docs/artifacts.md#environment-integration).
Model and optimizer settings are in [configs/paper.yaml](configs/paper.yaml).

### Training and Evaluation

Run joint test-time training in the configured environment:

```bash
python -m joint_ttopt deploy --environment data/environment.yaml \
  --model Qwen3-8B --head checkpoints/action_head.npz \
  --normalization data/entry/normalization.json \
  --calibration calibration/reference.json \
  --calibration-bounds data/gradient_bounds.json --output results/joint
```

Export the environment's native metric after the 50-episode deployment:

```bash
python -m joint_ttopt metrics results/joint --horizons 50 \
  --metric METRIC --output results/metrics.jsonl
```

Set `METRIC` to the evaluator's metric name from [configs/benchmarks.yaml](configs/benchmarks.yaml).
The [evaluation guide](docs/artifacts.md#evaluation) describes normalization and aggregation.

## License

The code uses the [MIT License](LICENSE).
