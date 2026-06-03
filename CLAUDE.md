# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A research framework comparing two multi-agent reinforcement learning (MARL) algorithms — **MAAC** (Multi-Agent Actor-Critic with attention) and **MADDPG** (Multi-Agent DDPG) — trained on [MPE2](https://github.com/Farama-Foundation/PettingZoo) cooperative/competitive multi-agent environments (simple_spread, simple_adversary, simple_tag, simple_push, simple_crypto, simple_speaker_listener).

## Commands

**Train a single agent:**
```bash
python main.py --agent maac --config configs/maac.yaml --train
python main.py --agent maddpg --config configs/maddpg.yaml --train
```

**Override config values (dot-path notation):**
```bash
python main.py --agent maac --config configs/maac.yaml \
  --override "common.env=simple_spread_v3" "maac.pi_lr=0.001" --train
```

**Train all agents on all environments:**
```bash
./train_all.sh
```

**Evaluate a trained checkpoint:**
```bash
python scripts/test.py --agent maac --config configs/maac.yaml \
  --checkpoint checkpoints/maac/simple_spread_v3/run1/maac_best.pt \
  --episodes 20 --render_mode human
```

**Generate report plots:**
```bash
python reports/generate_mean_reward_plots.py
python reports/validate_and_plot_mpe_runs.py
```

## Architecture

### Entry Points & Config

- [main.py](main.py) — dispatches to algorithm-specific training loops based on `--agent`
- [config.py](config.py) — loads a YAML file and applies CLI `--override` args via `_set_nested()`
- [configs/maac.yaml](configs/maac.yaml) / [configs/maddpg.yaml](configs/maddpg.yaml) — all hyperparameters; `common.*` keys are shared, algorithm-specific keys live under their own section

### Training Loops

Both loops follow the same structure: reset env → rollout with policy → store in replay buffer → sample batch → update networks → log to tensorboard → checkpoint.

- [scripts/train/maac.py](scripts/train/maac.py) — MAAC training loop
- [scripts/train/maddpg.py](scripts/train/maddpg.py) — MADDPG training loop
- [scripts/train/utils.py](scripts/train/utils.py) — shared run-directory management (`get_run_dir`); run numbers are shared between `runs/` (tensorboard) and `checkpoints/` so logs and weights stay paired

### Algorithm Implementations

**MAAC** ([agents/maac/](agents/maac/)):
- [attention_sac.py](agents/maac/attention_sac.py) — top-level `AttentionSAC` class; wraps per-agent policies and a shared attention critic
- [utils/critics.py](agents/maac/utils/critics.py) — multi-head scaled dot-product attention critic; each agent encodes its own state-action pair and attends to all others
- [utils/agents.py](agents/maac/utils/agents.py) — per-agent policy with discrete action output
- [utils/buffer.py](agents/maac/utils/buffer.py) — circular replay buffer using numpy object arrays (handles variable-length obs per agent)
- Discrete action spaces only. Attention entropy is regularized alongside policy entropy (SAC-style reward scaling).

**MADDPG** ([agents/maddpg_torch/](agents/maddpg_torch/)):
- [maddpg.py](agents/maddpg_torch/maddpg.py) — top-level `MADDPG` class
- [models.py](agents/maddpg_torch/models.py) — actor (logits for discrete / tanh for continuous) and critic (concatenated global obs+actions) networks
- [buffer.py](agents/maddpg_torch/buffer.py) — fixed-size float32 circular replay buffer
- Supports both discrete (Gumbel-softmax exploration) and continuous (Gaussian noise) action spaces.

### Key Patterns

**Device management** — call `prep_training()` before gradient updates and `prep_rollouts()` before inference; these move networks between devices and toggle `requires_grad`.

**Soft target updates** — `target = (1 - tau) * target + tau * source` used in both algorithms for stability.

**Checkpoint format:**
- MAAC saves `{'init_dict', 'agent_params', 'critic_params'}` so the full model can be reconstructed from file without a running instance
- MADDPG saves full `state_dict()` for actors, critics, and optimizers

**Action representation** — actions are passed around as `Dict[agent_id → int]` (discrete index) or `Dict[agent_id → np.ndarray]` (continuous) at environment boundaries, and converted to one-hot tensors inside training updates.

**Tensorboard logging** — runs written to `runs/{algorithm}/{env}/run{N}/`; per-agent rewards logged as `episode/agent_reward/{agent_id}`.
