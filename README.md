# Joint Test-Time Optimization from Multi-Agent Interaction of Large Language Models

Open-source code for the paper.

Existing test-time training methods improve individual agents, yet simultaneous adaptation couples their updates through shared computational resources and evolving learning signals. To address this problem, we formulate joint test-time optimization over persistent low-rank weights under a shared rank budget.

![Joint adaptation in five-player AvalonBench](figures/figure1_overview.png)

Joint adaptation in five-player AvalonBench (schematic).

## Installation

Python 3.10 or later.

```bash
pip install -e ".[models,test]"
```

## Deployment

Prepare the action head, normalization, and calibration inputs described in [Input preparation](docs/artifacts.md).

```bash
python -m joint_ttopt fetch
python -m joint_ttopt deploy \
  --environment configs/avalon.yaml --model Qwen3-8B \
  --head checkpoints/action_head.npz \
  --normalization data/entry_normalization.json \
  --calibration calibration/reference.json --output results/joint
```

Settings and ablation controls are in [configs](configs). Use `python -m joint_ttopt --help` for preparation, deployment, and evaluation commands.

## Software checks

```bash
pytest -q -p no:cacheprovider
```

The code uses the [MIT License](LICENSE). The paper image retains its published license.
