# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A research framework comparing multi-agent reinforcement learning (MARL) algorithms — **MAAC** (Multi-Agent Actor-Critic with attention), **MADDPG** (Multi-Agent DDPG), and **MAPPO** (Multi-Agent PPO) — trained on [MPE2](https://github.com/Farama-Foundation/PettingZoo) cooperative/competitive multi-agent environments (simple_spread, simple_adversary, simple_tag, simple_push, simple_crypto, simple_speaker_listener). MAAC/MADDPG are off-policy (replay buffer); MAPPO is on-policy (rollout buffer + GAE + PPO).

## Research Log

[RESEARCH_LOG.md](RESEARCH_LOG.md) records *why* changes were made, not just what — kept for thesis traceability. Always add an entry there after making a change to the agents (`agents/`) or environments (MPE2/MAMuJoCo configs, training loops): what changed and its purpose. Follow the entry format documented at the top of that file.

## Setup

Requires Python 3.10 (pinned by `pygame` and `numpy` constraints in `requirements.txt`).

```bash
conda create -n multiagents python=3.10
conda activate multiagents
pip install -r requirements.txt
```

## Commands

**Train a single agent (MPE2):**
```bash
python main.py --agent maac --config configs/maac.yaml --train
python main.py --agent maddpg --config configs/maddpg.yaml --train
python main.py --agent mappo --config configs/mappo.yaml --train
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

**Train on MAMuJoCo (continuous control):**
```bash
# MADDPG
python train_mamujoco.py --config configs/mamujoco_maddpg.yaml --train
python train_mamujoco.py --config configs/mamujoco_maddpg.yaml --scenario Ant --agent_conf 2x4 --train

# MAAC continuous (attention critic + Gaussian policy)
python train_mamujoco_maac.py --config configs/maac_continuous.yaml --train
python train_mamujoco_maac.py --config configs/maac_continuous.yaml --scenario Ant --agent_conf 2x4 --train

# MAPPO (Gaussian policy handles continuous natively; same class as MPE2)
python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml --train
python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml --scenario Ant --agent_conf 2x4 --train
```
Run/checkpoint dirs land under `runs/mamujoco/<scenario>/{maddpg|maac_continuous|mappo}/runN/`.

**Evaluate a trained checkpoint:**
```bash
python scripts/test.py --agent maac --config configs/maac.yaml \
  --checkpoint checkpoints/maac/simple_spread_v3/run1/maac_best.pt \
  --episodes 20 --render_mode human
```

**Monitor training:**
```bash
tensorboard --logdir runs/
```

**Generate report plots:**
```bash
python reports/generate_mean_reward_plots.py
python reports/validate_and_plot_mpe_runs.py
python reports/plot_episode_length_by_env.py
python reports/plot_training_time_by_env.py
```

**Sync checkpoints/runs with Hugging Face Hub:**
```bash
huggingface-cli login  # once
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters           # upload both
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters --download # download
# --checkpoints / --runs to sync only one folder; --dry-run to preview upload
```

## Architecture

### Entry Points & Config

- [main.py](main.py) — dispatches to algorithm-specific training loops based on `--agent`
- [config.py](config.py) — loads a YAML file and applies CLI `--override` args via `_set_nested()`
- [configs/maac.yaml](configs/maac.yaml) / [configs/maddpg.yaml](configs/maddpg.yaml) — all hyperparameters; `common.*` keys are shared, algorithm-specific keys live under their own section

### Training Loops

The off-policy loops (MAAC/MADDPG) follow the same structure: reset env → rollout with policy → store in replay buffer → sample batch → update networks → log to tensorboard → checkpoint. MAPPO is on-policy: each "episode" collects one fresh rollout of `max_cycles` steps → compute GAE returns/advantages → run a PPO update (several epochs of clipped-surrogate minibatch SGD) → reset the rollout buffer.

- [scripts/train/maac.py](scripts/train/maac.py) — MAAC training loop
- [scripts/train/maddpg.py](scripts/train/maddpg.py) — MADDPG training loop
- [scripts/train/mappo.py](scripts/train/mappo.py) — MAPPO training loop (on-policy)
- [scripts/train/utils.py](scripts/train/utils.py) — shared run-directory management (`get_next_shared_run_dirs`); run numbers are shared between `runs/` (tensorboard) and `checkpoints/` so logs and weights stay paired

### Algorithm Implementations

**MAAC** ([agents/maac/](agents/maac/)):
- [attention_sac.py](agents/maac/attention_sac.py) — top-level `AttentionSAC` class; wraps per-agent policies and a shared attention critic
- [utils/critics.py](agents/maac/utils/critics.py) — multi-head scaled dot-product attention critic; each agent encodes its own state-action pair and attends to all others
- [utils/agents.py](agents/maac/utils/agents.py) — per-agent policy with discrete action output
- [utils/buffer.py](agents/maac/utils/buffer.py) — circular replay buffer using numpy object arrays (handles variable-length obs per agent)
- Discrete action spaces only. Attention entropy is regularized alongside policy entropy (SAC-style reward scaling).

**MAAC Continuous** ([agents/maac_continuous/](agents/maac_continuous/)):
- [attention_sac_continuous.py](agents/maac_continuous/attention_sac_continuous.py) — `AttentionSACContinuous`; same structure as MAAC but for continuous spaces
- [utils/policies.py](agents/maac_continuous/utils/policies.py) — `GaussianPolicy`: mean + log_std, tanh squashing, reparameterization trick
- [utils/critics.py](agents/maac_continuous/utils/critics.py) — `AttentionCriticContinuous`: scalar Q output; Q head receives the agent's own SA encoding (not state-only) so the reparameterization gradient flows through it
- Policy update is per-agent: agent `i` uses its reparameterized action; others use buffer actions (no cross-agent gradient leakage)

**MADDPG** ([agents/maddpg_torch/](agents/maddpg_torch/)):
- [maddpg.py](agents/maddpg_torch/maddpg.py) — top-level `MADDPG` class
- [models.py](agents/maddpg_torch/models.py) — actor (logits for discrete / tanh for continuous) and critic (concatenated global obs+actions) networks
- [buffer.py](agents/maddpg_torch/buffer.py) — fixed-size float32 circular replay buffer
- Supports both discrete (Gumbel-softmax exploration) and continuous (Gaussian noise) action spaces.

**MAPPO** ([agents/mappo/](agents/mappo/)):
- [mappo.py](agents/mappo/mappo.py) — top-level `MAPPO` class; per-agent actor + per-agent *centralized* critic (fed the concatenated global obs, value-only). A single class handles both discrete and continuous, branching on `ActionSpec` (like MADDPG) — no separate continuous variant.
- [models.py](agents/mappo/models.py) — `Actor` (Categorical for discrete / diagonal Gaussian with state-independent `log_std` for continuous) and `Critic` (scalar V(s)). Reuses `ActionSpec`/`mlp` from `agents/maddpg_torch/models.py`.
- [buffer.py](agents/mappo/buffer.py) — `RolloutBuffer`: on-policy, single rollout, per-agent typed arrays; `compute_returns` does per-agent GAE, `agent_minibatch_generator` yields per-agent shuffled minibatches (no cross-agent flattening) so heterogeneous obs/action dims work.
- Per-agent independent networks (not upstream MAPPO's parameter sharing) and feedforward-only (no recurrent policy). Single-env rollouts (no vectorized threads), so batches are one episode's worth of steps.

### CTDE: what each algorithm centralizes

All three follow **Centralized Training, Decentralized Execution (CTDE)**: every agent acts on its *own local observation* (decentralized execution, one policy per agent), while the critic is **centralized** — it conditions on all-agent information during training. None of them use a "decentralized" (local-obs-only) critic.

Where they differ is *what the critic sees* and *whether it is instantiated per-agent or shared*:

| | Actor | Critic instances | Critic input | Shares teammate *actions*? |
|---|---|---|---|---|
| **MADDPG** | per-agent, local obs | one **per agent** | global obs **+ all actions** → Q(s,a) | ✅ yes |
| **MAAC** | per-agent, local obs | one **shared** attention module (per-agent heads) | all agents' state-action pairs, via attention | ✅ yes |
| **MAPPO** | per-agent, local obs | one **per agent** | global obs only → V(s) | ❌ no (state-value baseline) |

The key subtlety: MADDPG and MAAC feed teammates' **actions** into the critic (action-value Q); MAPPO's critic is a **state-value** function V(s) that sees all observations but no actions — a defining property of PPO, not an omission. MADDPG and MAPPO instantiate a separate critic per agent; MAAC uses a single shared attention critic.

### Key Patterns

**Device management** — call `prep_training()` before gradient updates and `prep_rollouts()` before inference; these move networks between devices and toggle `requires_grad`.

**Soft target updates** — `target = (1 - tau) * target + tau * source` used in both algorithms for stability.

**Checkpoint format:**
- MAAC saves `{'init_dict', 'agent_params', 'critic_params'}` so the full model can be reconstructed from file without a running instance
- MADDPG saves full `state_dict()` for actors, critics, and optimizers
- MAPPO saves full `state_dict()` for actors, critics, and optimizers (same pattern as MADDPG); reload by rebuilding via `_build_mappo` then `load_state_dict`

**Action representation** — actions are passed around as `Dict[agent_id → int]` (discrete index) or `Dict[agent_id → np.ndarray]` (continuous) at environment boundaries, and converted to one-hot tensors inside training updates.

**Tensorboard logging** — runs written to `runs/{algorithm}/{env}/run{N}/`; per-agent rewards logged as `episode/agent_reward/{agent_id}`.
