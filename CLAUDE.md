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

**Every command below assumes the `multiagents` conda env is active.** Activate it first, or prefix one-off commands with `conda run -n multiagents`:

```bash
conda run -n multiagents python main.py --agent mat --config configs/mat.yaml --train
```

> **Two environments exist in this repo — they are not interchangeable.** A `.venv/` is also present (Python 3.10.12) and is what `train_all.sh` invokes via `./.venv/bin/python`. It has the MPE2 dependencies but **is missing `gymnasium_robotics`, so every MAMuJoCo entry point fails under it**. The conda env (`multiagents`, Python 3.10.20) has the full set and is the one to use. Either install `gymnasium_robotics` into `.venv`, or point `train_all.sh` at the conda interpreter, before running MAMuJoCo from that script.

## Commands

**Train a single agent (MPE2):**
```bash
python main.py --agent maac --config configs/maac.yaml --train
python main.py --agent maddpg --config configs/maddpg.yaml --train
python main.py --agent mappo --config configs/mappo.yaml --train
python main.py --agent mat --config configs/mat.yaml --train
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

# MAT (same class handles discrete + continuous). The default config targets
# HalfCheetah 6x1 with the MAT paper's hyperparameters, for reproduction.
python train_mamujoco_mat.py --config configs/mamujoco_mat.yaml --train
python train_mamujoco_mat.py --config configs/mamujoco_mat.yaml --scenario Ant --agent_conf 2x4 --train
```
Run/checkpoint dirs land under `runs/mamujoco/<scenario>/{maddpg|maac_continuous|mappo|mat}/runN/`.

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
python reports/report.py --env_type mpe2
python reports/report.py --env_type mamujoco
python reports/mpe2/validate_and_plot_mpe_runs.py
```
See [reports/CLAUDE.md](reports/CLAUDE.md) for the full set of scripts, including older per-env-type equivalents still present alongside `report.py`.

**Sync checkpoints/runs with Hugging Face Hub:**
```bash
huggingface-cli login  # once
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters           # upload both
python scripts/hf_sync.py --repo your-username/multi-agent-attention-masters --download # download
# --checkpoints / --runs to sync only one folder; --dry-run to preview upload
```

## Evaluation & Metrics

Three distinct things happen under this heading: continuous tensorboard logging during training, an automatic best/early-stop mechanism that runs on every training loop, and a separate post-hoc evaluation script. They are independent of each other.

### Tensorboard metrics (logged during training)

**Common to all four training loops** (`train_maac`, `train_maac_continuous`, `train_maddpg`, `train_mappo`), once per episode:

| Tag | Meaning |
|---|---|
| `episode/mean_reward` | This episode's reward, averaged across agents |
| `episode/mean_reward_window` | Rolling average of `mean_reward` over the last `early_stop_window` episodes — also the metric driving best-checkpoint selection and early stopping (see below) |
| `episode/length` | Steps before termination/truncation/`max_cycles` |
| `episode/agent_reward/{agent_id}` | Per-agent (not averaged) episode reward |

**MADDPG** ([scripts/train/maddpg.py](scripts/train/maddpg.py)), logged every update step:

| Tag | Meaning |
|---|---|
| `agent{i}/losses/critic_loss` | Agent i's critic TD loss |
| `agent{i}/losses/actor_loss` | Agent i's deterministic policy-gradient loss |

**MAPPO** ([scripts/train/mappo.py](scripts/train/mappo.py)), logged once per episode (averaged over that update's PPO epochs/minibatches):

| Tag | Meaning |
|---|---|
| `agent{i}/losses/actor_loss` | Clipped-surrogate PPO loss **minus** the entropy bonus (`entropy_coef * entropy`) — the actual term minimized |
| `agent{i}/losses/value_loss` | Clipped value loss (PPO2-style, since `use_clipped_value_loss` defaults `True`): `0.5 * max((V-return)², (V_clip-return)²)` |
| `agent{i}/losses/entropy` | Mean action-distribution entropy — a diagnostic, separate from the entropy term already folded into `actor_loss` |

**MAAC / MAAC Continuous** ([agents/maac/attention_sac.py](agents/maac/attention_sac.py), [agents/maac_continuous/attention_sac_continuous.py](agents/maac_continuous/attention_sac_continuous.py)), logged every critic/policy update:

| Tag | Meaning |
|---|---|
| `losses/q_loss` | Shared attention critic's Bellman MSE (SAC-style target, includes `-log_pi/reward_scale`), plus attention-magnitude regularization |
| `grad_norms/q` | Critic gradient norm *before* clipping (clip threshold `10 * n_agents`) |
| `agent{i}/losses/pol_loss` | Agent i's SAC-style policy loss, using the shared critic's Q as the advantage baseline |
| `agent{i}/grad_norms/pi` | Agent i's policy gradient norm before clipping (clip threshold `0.5`) |
| `agent{i}/policy_entropy` | Entropy of agent i's action distribution — exact categorical entropy (discrete) or `-log_pi.mean()` (continuous, a differential-entropy proxy for the Gaussian policy) |
| `attention/mean_entropy` (discrete) or `agent{i}/attention/mean_entropy` (continuous) | Entropy of the attention-weight distribution — high = attends broadly to all agents, low = focused on one/few. **Note the tag path differs between the two variants**: discrete pools all agents' attention into one shared value, continuous logs one per agent (matching its per-agent Q heads) |

**MAAC / MAAC Continuous attention visualization** ([scripts/train/maac.py:198-213](scripts/train/maac.py#L198-L213)), every `attn_heatmap_every` episodes (default 100), sampled from the replay buffer:

| Tag | Meaning |
|---|---|
| `attention/heatmap_mean`, `attention/heatmap_std` | Full attention-matrix heatmap images (mean / std across the sampled batch) |
| `attention/{src}_to_{dst}`, `attention/{src}_to_{dst}_std` | Same data, broken out as scalars: mean/std attention weight agent `src` places on agent `dst` |

### Best-checkpoint selection & early stopping

Identical logic in all four training loops (`scripts/train/{maac,maac_continuous,maddpg,mappo}.py`):

- The metric is always `episode/mean_reward_window` (the rolling window average). The `early_stop_metric` config key exists but **is not read by any training loop** — changing it currently has no effect.
- Starting at episode `early_stop_start_episode`, whenever the windowed metric improves by more than `early_stop_min_delta`, the current weights are saved as `{algorithm}_best.pt`.
- If `early_stop_patience > 0` and the windowed metric hasn't improved for that many episodes, training stops early. Default `early_stop_patience: 0` disables this — training always runs the full `episodes` budget unless stopped explicitly.
- `{algorithm}_final.pt` is saved at the end of training regardless (if `save_final: true`), whether or not it's the best-performing checkpoint.

### Post-training evaluation (`scripts/test.py`)

Loads one checkpoint and runs `--episodes` (default 20) fully deterministic episodes (no exploration noise/sampling), printing each episode's `mean_reward` to stdout. This is a standalone check: nothing is written to `runs/` or aggregated into a summary statistic across episodes — each line is one episode.

### Qualitative sanity checks (not evaluation metrics)

- [reports/mpe2/validate_and_plot_mpe_runs.py](reports/mpe2/validate_and_plot_mpe_runs.py) — renders a grid of frames from saved checkpoints at fixed training steps for visual inspection. Currently hardcoded to MAAC/MADDPG on 2 environments; not extended to MAPPO or MAMuJoCo.
- [scripts/test_mpe2_gpu_render.py](scripts/test_mpe2_gpu_render.py) — smoke-tests GPU-based action sampling and rendering with random actions. Unrelated to trained-policy evaluation.

There are no automated unit tests (no `pytest`/`unittest` in this repo) — "tests" here means the evaluation protocol above, not software tests.

## Architecture

### Entry Points & Config

- [main.py](main.py) — dispatches to algorithm-specific training loops based on `--agent`
- [config.py](config.py) — loads a YAML file and applies CLI `--override` args via `_set_nested()`
- [configs/maac.yaml](configs/maac.yaml) / [configs/maddpg.yaml](configs/maddpg.yaml) — all hyperparameters; `common.*` keys are shared, algorithm-specific keys live under their own section

### Training Loops

See [scripts/CLAUDE.md](scripts/CLAUDE.md) for the training loop scripts, the evaluation script (`scripts/test.py`), and other scripts (`scripts/hf_sync.py`, `scripts/test_mpe2_gpu_render.py`).

### Algorithm Implementations

See [agents/CLAUDE.md](agents/CLAUDE.md) for per-algorithm implementation details (MAAC, MAAC Continuous, MADDPG, MAPPO), the CTDE comparison table, and shared implementation patterns (device management, soft target updates, checkpoint formats, action representation).

**Tensorboard logging** — runs written to `runs/{algorithm}/{env}/run{N}/`. See Evaluation & Metrics above for the full metric list and what each one is for.
