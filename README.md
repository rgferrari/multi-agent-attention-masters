# Multi-Agent Attention Masters

Comparison of two multi-agent reinforcement learning (MARL) algorithms trained on [MPE2](https://github.com/Farama-Foundation/PettingZoo) environments:

- **MAAC** — Multi-Agent Actor-Critic with attention-based centralized critic
- **MADDPG** — Multi-Agent Deep Deterministic Policy Gradient (PyTorch)

Both follow Centralized Training Decentralized Execution (CTDE): agents share information during training but act independently at inference.

## Environments

| Environment | Type |
|---|---|
| `simple_spread_v3` | Cooperative navigation |
| `simple_adversary_v3` | Physical deception |
| `simple_tag_v3` | Predator-prey |
| `simple_push_v2` | Keep-away |
| `simple_crypto_v3` | Covert communication |
| `simple_speaker_listener_v4` | Cooperative communication |

## Setup

```bash
pip install huggingface_hub  # only needed for checkpoint sync
```

The project depends on `torch`, `gymnasium`, `mpe2`, `pyyaml`, and `tensorboard`. Use the environment that has all of these available.

## Training

```bash
# Train with default config
python main.py --agent maac --config configs/maac.yaml --train
python main.py --agent maddpg --config configs/maddpg.yaml --train

# Override any config value (dot-path notation, repeat flag for multiple)
python main.py --agent maddpg --config configs/maddpg.yaml \
  --override "common.env=simple_spread_v3" \
  --override "common.episodes=10000" \
  --train

# Train all agents on all environments
./train_all.sh
```

Config files live in `configs/`. All hyperparameters are documented there. Common keys (`env`, `episodes`, `seed`, etc.) sit under `common:`; algorithm-specific keys under `maac:` or `maddpg:`.

## Evaluation

```bash
python scripts/test.py \
  --agent maac \
  --config configs/maac.yaml \
  --checkpoint checkpoints/maac/simple_spread_v3/run1/maac_best.pt \
  --episodes 20 \
  --render_mode human
```

## Monitoring

Training logs TensorBoard metrics to `runs/`:

```bash
tensorboard --logdir runs/
```

Logged metrics:

| Tag | Description |
|---|---|
| `episode/mean_reward` | Mean reward across agents |
| `episode/mean_reward_window` | Rolling average |
| `episode/length` | Steps per episode |
| `episode/agent_reward/{id}` | Per-agent reward |
| `agentN/losses/critic_loss` | Critic loss for agent N |
| `agentN/losses/actor_loss` / `pol_loss` | Policy loss for agent N |
| `agentN/policy_entropy` | Policy entropy (MAAC only) |
| `losses/q_loss`, `grad_norms/*` | Attention critic metrics (MAAC only) |

Run directories are created automatically under `runs/{algorithm}/{env}/run{N}/` and checkpoints under `checkpoints/{algorithm}/{env}/run{N}/`, with matching run numbers.

## Checkpoint Sync

Upload to or download from Hugging Face Hub:

```bash
# First-time setup — create a free account at huggingface.co, then:
huggingface-cli login

# Upload
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters

# Download on another machine
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters --download
```

Pass `--checkpoints` or `--runs` to sync only one folder. The private repo is created automatically on first upload.

## Project Structure

```
agents/
  maac/               MAAC algorithm (attention SAC)
  maddpg_torch/       MADDPG algorithm (PyTorch)
configs/              YAML hyperparameter files
scripts/
  train/              Training loops (maac.py, maddpg.py)
  test.py             Checkpoint evaluation
  hf_sync.py          HuggingFace Hub sync
reports/              Plot generation scripts
main.py               Entry point
config.py             YAML loader with CLI override support
train_all.sh          Train all agents × all environments
```
