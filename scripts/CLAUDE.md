# CLAUDE.md — scripts/

Documentation for the scripts in this directory. See the root [CLAUDE.md](../CLAUDE.md) for project-wide setup, commands, and evaluation/metrics; see [agents/CLAUDE.md](../agents/CLAUDE.md) for the algorithm implementations these scripts train and evaluate.

## Training Loops (`train/`)

The off-policy loops (MAAC/MAAC Continuous/MADDPG) follow the same structure: reset env → rollout with policy → store in replay buffer → sample batch → update networks → log to tensorboard → checkpoint. MAPPO is on-policy: each "episode" collects one fresh rollout of `max_cycles` steps → compute GAE returns/advantages → run a PPO update (several epochs of clipped-surrogate minibatch SGD) → reset the rollout buffer.

- [maac.py](train/maac.py) — MAAC training loop (MPE2, discrete). Also defines `_agent_label` and `_attention_heatmap_figure`, reused by `maac_continuous.py` for its own attention logging.
- [maac_continuous.py](train/maac_continuous.py) — MAAC Continuous training loop, driven by `train_mamujoco_maac.py` at the repo root for MAMuJoCo. Reuses `agents/maac/utils/buffer.py`'s `ReplayBuffer` (shared with discrete MAAC).
- [maddpg.py](train/maddpg.py) — MADDPG training loop (MPE2 and, via `train_mamujoco.py`, MAMuJoCo).
- [mappo.py](train/mappo.py) — MAPPO training loop (on-policy; MPE2 and, via `train_mamujoco_mappo.py`, MAMuJoCo — same class handles both since `MAPPO` branches on `ActionSpec`).
- [mat.py](train/mat.py) — MAT training loop (on-policy; MPE2 and, via `train_mamujoco_mat.py`, MAMuJoCo). Same class handles discrete and continuous.
- [evaluate.py](train/evaluate.py) — periodic **deterministic** evaluation during training, enabled by `common.eval_interval` (episodes; 0 = off, the default) and `common.eval_episodes`. Logs `eval/eval_average_episode_rewards` — deliberately the reference implementations' metric name, because this is the quantity the MAT and MAPPO papers report, as opposed to the exploring training-rollout reward the loops log per episode. The two are **not** interchangeable: a stochastic policy exploring scores lower than the same policy evaluated greedily, so comparing a training curve against a published eval curve understates the agent. Currently wired into the MAT loop only. Failures are swallowed so evaluation can never fail a training run.
- [report_hook.py](train/report_hook.py) — optional end-of-training report generation (`common.auto_report`). See [reports/CLAUDE.md](../reports/CLAUDE.md).
- [utils.py](train/utils.py) — shared run-directory management. `get_next_shared_run_dirs()` picks one shared run index across multiple base dirs (e.g. `runs/` and `checkpoints/`) so tensorboard logs and weights stay paired under the same `runN`.

Best-checkpoint selection, early stopping, and the full tensorboard metric list logged by each loop are documented in the root CLAUDE.md's "Evaluation & Metrics" section, not here.

## Other Scripts

- [test.py](test.py) — post-training evaluation (`python scripts/test.py --agent {maac,maddpg,mappo,mat} --config ... --checkpoint ... --episodes N`). Loads one checkpoint, rebuilds the matching model (`AttentionSAC.init_from_save`, or `_build_mappo`/`_build_maddpg` + `load_state_dict`), and runs fully deterministic episodes (no exploration noise/sampling) on an MPE2 env, printing each episode's `mean_reward` to stdout. Standalone check — nothing is written to `runs/` or aggregated across episodes.
- [hf_sync.py](hf_sync.py) — uploads/downloads `checkpoints/` and `runs/` to/from a Hugging Face Hub model repo (created private automatically if missing). Content-addressed skip on both sides (already-synced files are skipped). `--checkpoints`/`--runs` restrict to one folder; `--dry-run` previews without transferring; requires `huggingface-cli login` or `HF_TOKEN` first.
- [test_mpe2_gpu_render.py](test_mpe2_gpu_render.py) — smoke test only, unrelated to trained-policy evaluation. Renders an MPE2 env with `render_mode="human"` and steps it using randomly sampled actions generated on-device (`torch`, honors `--use_gpu`), to sanity-check GPU-based action sampling plus the rendering path itself.
