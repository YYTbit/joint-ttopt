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

![Joint adaptation in five-player AvalonBench](figures/figure1_overview.png)

**Figure 1.** Joint adaptation in five-player AvalonBench (schematic). (a) Context-dependent experience reuse. (b) Equal per-agent rank budgets. (c) Changing learning signals. (d) Shared spectral allocation, interaction-sensitive steps, and persistent weights. Optimality holds for the per-round proximal objective. Tile heights indicate update values; solid tiles mark retained directions, totaling ten in both allocations. Agent colors identify players, whose roles remain private.

![Algorithm 1](figures/algorithm.png)

## Results

### Main Results

**Table 1.** Performance across five benchmarks and three backbones after 50 episodes (mean ± standard deviation over ten paired seeds). All metrics are higher-is-better; bold marks the best mean per backbone. Native metrics complement normalized gain. Paired comparisons and their confidence intervals are detailed in the supplement.

![Main results from Table 1](figures/main_results.png)

### Ablation Study

**Table 2.** Component ablation across backbones and benchmarks (mean ± standard deviation over ten paired seeds).

![Component ablations from Table 2](figures/ablations.png)

**Shared allocation benefits unequal update values.** Per-agent persistent LoRA uses separate agent budgets; Equal-split persistent LoRA divides the common budget evenly. Both preserve the state and common step control. Joint allocation improves their KPI by 0.7 and 1.2 points and Decrypto success by 2.0 and 2.3 points. Value-proportional allocates capacity in proportion to update values; its KPI averages 62.3 against 63.8 for the full method. Spectral selection accounts for each direction’s marginal value.

### Analysis

![Analysis from Figure 2](figures/analysis.png)

**Figure 2.** Performance versus (a) horizon and (b) compute at 50 episodes. Bars show seed standard deviations. Retr., Plan., and LoRA denote Retrieval, Planning, and LoRA TT-RL.

**Persistent adaptation benefits from continued interaction.** Figure 2(a) shows the lead over TEMPO growing from 0.42 at ten episodes to 1.01 at 100. Persistent states retain earlier feedback, while repeated allocation responds to changing update values. The common 50-episode normalization anchors keep this comparison on the same scale.

## Getting Started

### Environment Setup

```bash
conda create -n joint-ttopt python=3.10 -y
conda activate joint-ttopt
pip install -e '.[models]'
```

### Data Preparation

Download the benchmark sources and install the environment dependencies:

```bash
python -m joint_ttopt fetch --output external
```

The paper evaluates MultiAgentBench, GAMA-Bench, SOTOPIA, Decrypto, and AvalonBench.
Configure the benchmark in `data/environment.yaml`. Prepare the action head,
reward normalization, and calibration inputs with the commands in
[input preparation](docs/artifacts.md). The deployment command below uses the
paper settings from `configs/paper.yaml`.

## Training

Run joint test-time training in the configured environment:

```bash
python -m joint_ttopt deploy --environment data/environment.yaml \
  --model Qwen3-8B --head checkpoints/action_head.npz \
  --normalization data/entry/normalization.json \
  --calibration calibration/reference.json \
  --calibration-bounds data/gradient_bounds.json --output results/joint
```

## Evaluation

Export the environment's native metric after the 50-episode deployment:

```bash
python -m joint_ttopt metrics results/joint --horizons 50 \
  --metric METRIC --output results/metrics.jsonl
```

Set `METRIC` to the native metric returned by the benchmark evaluator.
[Score aggregation](docs/artifacts.md#evaluation) describes normalization across runs.

## License

The code uses the [MIT License](LICENSE).
