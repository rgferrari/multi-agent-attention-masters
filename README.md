# Multi-Agent Attention Masters

Comparison of multi-agent reinforcement learning (MARL) algorithms — **MAAC**, **MADDPG**, and **MAPPO** — trained on [MPE2](https://github.com/Farama-Foundation/PettingZoo) cooperative/competitive environments, with continuous-control support on MAMuJoCo.

- **MAAC** — Multi-Agent Actor-Critic with attention-based centralized critic
- **MADDPG** — Multi-Agent Deep Deterministic Policy Gradient (PyTorch port of the original TF1 implementation)
- **MAPPO** — Multi-Agent PPO, on-policy with a per-agent state-value critic

All three follow Centralized Training, Decentralized Execution (CTDE): agents act on their own local observation, while the critic is centralized during training. MAAC and MADDPG are off-policy (replay buffer); MAPPO is on-policy (rollout buffer + GAE).

## Environments

| Environment | Type |
|---|---|
| `simple_spread_v3` | Cooperative navigation |
| `simple_adversary_v3` | Physical deception |
| `simple_tag_v3` | Predator-prey |
| `simple_push_v3` | Keep-away |
| `simple_crypto_v3` | Covert communication |
| `simple_speaker_listener_v4` | Cooperative communication |

MAMuJoCo (continuous control) is also supported — e.g. HalfCheetah (`2x3`, `6x1` factorizations) and Ant (`2x4`, `4x2`, `2x4d`). See Training below.

## Setup

Requires Python 3.10 (pinned by `pygame`/`numpy` constraints in `requirements.txt`).

```bash
conda create -n multiagents python=3.10
conda activate multiagents
pip install -r requirements.txt
```

## Training

```bash
# Train with default config
python main.py --agent maac --config configs/maac.yaml --train
python main.py --agent maddpg --config configs/maddpg.yaml --train
python main.py --agent mappo --config configs/mappo.yaml --train

# Override any config value (dot-path notation, repeat flag for multiple)
python main.py --agent maddpg --config configs/maddpg.yaml \
  --override "common.env=simple_spread_v3" \
  --override "common.episodes=10000" \
  --train

# Train all agents on all environments
./train_all.sh
```

Config files live in `configs/`. All hyperparameters are documented there. Common keys (`env`, `episodes`, `seed`, etc.) sit under `common:`; algorithm-specific keys under `maac:`, `maddpg:`, or `mappo:`.

### MAMuJoCo (continuous control)

```bash
python train_mamujoco.py --config configs/mamujoco_maddpg.yaml --train
python train_mamujoco_maac.py --config configs/maac_continuous.yaml --train
python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml --train

# Pick a scenario/factorization (defaults to HalfCheetah 2x3)
python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml --scenario Ant --agent_conf 2x4 --train
```

Run/checkpoint dirs land under `runs/mamujoco/<scenario>/{maddpg|maac_continuous|mappo}/runN/`.

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

Run directories are created automatically under `runs/{algorithm}/{env}/run{N}/` (MPE2) or `runs/mamujoco/<scenario>/{algorithm}/run{N}/` (MAMuJoCo), with matching checkpoint dirs under `checkpoints/` using the same run numbers.

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
  maac/               MAAC algorithm (attention SAC, discrete)
  maac_continuous/    MAAC for continuous control (MAMuJoCo)
  maddpg_torch/       MADDPG algorithm (PyTorch, discrete + continuous)
  maddpg/             Original TF1 MADDPG reference (ported to maddpg_torch/, kept for reference)
  mappo/              MAPPO algorithm (discrete + continuous)
configs/              YAML hyperparameter files
scripts/
  train/              Training loops (maac.py, maac_continuous.py, maddpg.py, mappo.py)
  test.py             Checkpoint evaluation
  hf_sync.py          HuggingFace Hub sync
reports/              Plot generation scripts (mpe2/, mamujoco/)
main.py               Entry point (MPE2)
train_mamujoco*.py    Entry points (MAMuJoCo, one per algorithm)
config.py             YAML loader with CLI override support
train_all.sh          Train all agents × all environments
```
