# Research Log

Chronological record of substantive changes to this repo and the reasoning behind them — kept so a design decision can be traced back to its motivation later (thesis writeup, or revisiting an old choice and asking "why did we do it this way?").

Newest entries at the bottom. Add an entry when a change is made, with firsthand knowledge of the reasoning — not reconstructed after the fact, which risks misattributing rationale.

This log was created 2026-07-06. Entries dated earlier are backfilled only where the user directly supplied the rationale (e.g. the MADDPG port and the hyperparameter-defaults note below) — the rest of the pre-2026-07-06 history (continuous MAAC for MAMuJoCo, the run-dir reorg, etc.) lives in `git log` but hasn't been backfilled; ask if you'd like those reconstructed from commit messages (they'd be marked as reconstructed, not firsthand).

**Entry format:**
```
## YYYY-MM-DD — Short title

**Change:** what was done, concretely.
**Purpose:** why — the motivation, the question it answers, alternatives rejected (if any).
```

---

## 2026-06-03 — Port MADDPG from TF1 to PyTorch

**Change:** `agents/maddpg_torch/` is a port of the original MADDPG reference implementation (TensorFlow 1) to PyTorch. Present since this repo's first commit, so the port itself predates the repo's git history.

**Purpose:** *(as told by user)* to make MADDPG usable in this project — needed a runnable PyTorch implementation, consistent with the rest of the (PyTorch-based) codebase, rather than depending on the unmaintained TF1 original.

---

## 2026-06-03 — Hyperparameters follow each algorithm's reference-repo defaults

**Change:** Config files (`configs/*.yaml`) for MAAC, MADDPG, and MAPPO use the default hyperparameter settings from each algorithm's own original/reference repo, rather than custom-tuning per environment.

**Purpose:** *(as told by user)* keeps each algorithm faithful to its published configuration — relevant to note since this is a comparison study, where selectively tuning one algorithm and not another would confound the comparison.

---

## 2026-07-06 — Add MAPPO algorithm (MPE2 + MAMuJoCo)

**Change:** Implemented MAPPO as a third algorithm (`agents/mappo/`), alongside MAAC and MADDPG — per-agent actor + per-agent centralized critic (state-value only, no actions), on-policy rollout buffer with GAE, clipped-surrogate PPO updates. Wired up for both MPE2 (`configs/mappo.yaml`, `scripts/train/mappo.py`) and MAMuJoCo (`configs/mamujoco_mappo.yaml`, `train_mamujoco_mappo.py`). CLAUDE.md updated with a CTDE comparison table contrasting how MADDPG/MAAC/MAPPO differ in what their critics condition on.

**Purpose:** *(inferred from CLAUDE.md's own framing — confirm or correct this)* rounds out the thesis comparison with an on-policy algorithm, so it isn't limited to two off-policy variants (MAAC vs MADDPG) that both feed teammate actions into the critic. MAPPO's state-value-only critic is the natural contrast point on that axis.

---

## 2026-07-06 — Add claude-md-updater subagent

**Change:** Added `.claude/agents/claude-md-updater.md` — a project-scoped subagent that diffs the repo's actual structure against CLAUDE.md's claims and edits only what's drifted, rather than requiring a fully manual doc update.

**Purpose:** CLAUDE.md's usefulness depends on staying accurate as algorithms are added; without a dedicated process it tends to lag behind (as happened with MAPPO above, which needed a manual update). Scoped to on-demand invocation only, not auto-triggered after every change, to keep doc edits deliberate.

---

## 2026-08-14 — Add MAT (Multi-Agent Transformer) algorithm

**Change:** Implemented MAT as a fourth algorithm (`agents/mat/`), ported from the vendored reference clone in `agents/Multi-Agent-Transformer/`. A single shared encoder-decoder transformer treats the agent axis as a sequence: the unmasked encoder produces per-agent values (this *is* the centralized critic — there is no separate critic network), and the causally-masked decoder emits actions autoregressively over agents, so agent `i` conditions on agents `0..i-1`'s actions. PPO-trained, so the training loop (`scripts/train/mat.py`) mirrors `train_mappo` — same run-dir layout, episode tags, best-checkpoint/early-stop block. Wired into `main.py` as `--agent mat` with `configs/mat.yaml`. Two deliberate departures from the reference are documented in the module docstrings:

1. **Heterogeneous space support.** Upstream sizes every agent from `observation_space[0]`, requiring uniform obs/action dims across agents. Of this repo's six MPE2 envs only `simple_spread_v3` satisfies that; the other five have per-agent dims (e.g. `simple_speaker_listener_v4` is obs 3/11, act 3/5). Observations are therefore zero-padded to `max_i obs_dim_i` and a per-agent action mask drives surplus logits to -inf so they can never be sampled.
2. **Dropped the dead `state_dim`/`encode_state` path.** The reference carries it but disables it in its own code by overwriting the state with `np.zeros((..., 37))` before use; the encoder here always runs on observations.

Verified: autoregressive and parallel action paths return identical log-probs (max diff 0.0, so the PPO ratio is valid), action masking holds under sampling (zero probability mass on invalid actions), all six MPE2 envs train, the continuous/MAMuJoCo branch runs, and reward improves over a short 600-episode run.

**Follow-up (2026-08-14):** added `train_mamujoco_mat.py` + `configs/mamujoco_mat.yaml` and a deterministic-evaluation hook (`scripts/train/evaluate.py`), specifically to make *reproducing the MAT paper* possible — see the 2026-08-14 entry below.

**Purpose:** adds a *sequence-modeling* point of comparison to a thesis lineup that was otherwise three variations on per-agent networks. MAAC, MADDPG, and MAPPO all instantiate one actor (and critic) per agent and choose actions independently given observations; MAT is the first here where a single shared model's parameter count is agent-count-independent and where actions are sequentially dependent. That makes it the natural contrast on the "are agents independent decision-makers or one joint sequence?" axis, alongside the existing critic-conditioning axis in CLAUDE.md's CTDE table. The padding/masking extension was chosen over restricting MAT to `simple_spread_v3` so it can be compared on the same six environments as the other three algorithms — a comparison restricted to one env would not have been meaningful.

---

## 2026-08-14 — MAMuJoCo entry point for MAT + deterministic evaluation (paper reproduction)

**Change:** Three additions enabling MAT to be validated against its published results:

1. **`train_mamujoco_mat.py`** — MAMuJoCo entry point for MAT, following the `train_mamujoco_mappo.py` pattern (reuses `_build_mat`/`_make_args` from `main.py`). MAT's continuous branch was already tested but had no CLI path, so MAT could previously only run on MPE2.
2. **`configs/mamujoco_mat.yaml`** — targets `HalfCheetah 6x1`, the MAT paper's Multi-Agent MuJoCo scenario, with hyperparameters taken verbatim from the reference repo's `mat/scripts/train_mujoco.sh` (`lr`/`critic_lr` 5e-5, `entropy_coef` 0.001, `max_grad_norm` 0.5, `clip_param` 0.05, `ppo_epoch` 10) plus `config.py` defaults (`gamma` 0.99, `gae_lambda` 0.95, `n_block`/`n_embd`/`n_head` 1/64/1). Verified `HalfCheetah 6x1` instantiates in the installed `gymnasium_robotics`.
3. **`scripts/train/evaluate.py`** — periodic deterministic evaluation (`common.eval_interval`, `common.eval_episodes`; 0 disables, off by default). Logs `eval/eval_average_episode_rewards`, matching the reference metric name. Wired into the MAT loop after the PPO update, before the next episode's `env.reset`. Also added MAT to `scripts/test.py` (`--agent mat`), which previously accepted only maac/maddpg/mappo.

**Purpose:** the earlier claim that MAT results here are "not comparable to published numbers" was too strong, and the user correctly pushed back. Most of the gap was *configuration* (scenario `6x1` vs `2x3`, step budget, thread count) — adjustable, not fundamental. Only one difference was methodological and real: **the paper reports `eval_average_episode_rewards` from separate deterministic episodes, while this repo logged the exploring training rollout's reward.** Those are different quantities — a stochastic policy exploring scores lower than the same policy evaluated greedily — so comparing them would have understated MAT. `evaluate.py` produces the comparable quantity.

Why this matters for the thesis: the port currently has only *structural* verification (autoregressive/parallel log-probs identical, action masking sound). A successful reproduction on `HalfCheetah 6x1` would add *behavioural* evidence that it learns like published MAT, which in turn anchors the padding/masking extension used on the heterogeneous MPE2 envs — `6x1` is homogeneous, so it exercises the port without that extension.

Residual differences to state if published numbers are cited: single-env rollouts vs. the paper's 40 vectorized threads (much smaller PPO batch; `num_mini_batch` is 1 here, not 40), and no `ValueNorm`/huber loss (dropped for consistency with how MAPPO was ported). Expect same order of magnitude and trend, not identical numbers. Measured throughput on this machine: ~190 env-steps/s single-env CPU on `HalfCheetah 6x1`, so the paper's 10M-step budget is ≈15 h on CPU — feasible, and less with `common.use_gpu: true`.

---

## 2026-08-17 — PPO batch size, not the port, was throttling MAT on MAMuJoCo

**Change:** Ran a controlled comparison of `max_cycles` on `HalfCheetah 6x1` (`runs/mamujoco/ma_halfcheetah_6x1/mat/run{1,2}`), holding every other hyperparameter fixed. Because MAT is on-policy with one PPO update per episode, `max_cycles` *is* the PPO batch size.

| | run1 (`max_cycles: 100`) | run2 (`max_cycles: 1000`) |
|---|---|---|
| Best `eval_average_episode_rewards` | 118.7, reached at ~2,145,000 steps | **709.8, reached at ~225,000 steps** |
| Plateau | ~64 | ~470–500 |
| Steps trained | 2,145,600 | 1,139,000 |

**~6x higher return with ~10x fewer environment steps.**

**Purpose:** an earlier run reached only ~64 eval reward against the paper's ~5631 (Table 3, [arXiv:2205.14953](https://arxiv.org/abs/2205.14953), 1e7 steps), which left two indistinguishable explanations: a subtle bug in the port, or insufficient training scale. This test separates them — **the architecture is fine; it was being throttled by 100-step PPO updates.**

The root cause was my own misreading when writing `configs/mamujoco_mat.yaml`: I copied `episode_length 100` from the reference's `train_mujoco.sh` without accounting for its `n_rollout_threads 40`, so upstream's effective batch is ~4000 steps per update, not 100. `max_cycles: 1000` is both MAMuJoCo's native episode horizon and 10x closer to that effective batch.

Residual gap to the paper (~12x) is now plausibly configuration rather than correctness: batch still 4x smaller than upstream; only 1.1M of 10M steps trained; and the port omits `ValueNorm` plus the reference's `--use_state_agent` / `--add_center_xy` / `--use_value_active_masks` flags. The run2 curve plateaued around 200-300k steps with high point-to-point variance (344-564), which is the signature of an unnormalised value target on a task whose returns reach thousands — **`ValueNorm` is the first thing to try if this is pursued further.**

**Implication beyond MAT:** any on-policy algorithm here is exposed to the same batch-size effect. `configs/mamujoco_mappo.yaml` should be reviewed on the same grounds. MAAC Continuous and MADDPG are unaffected, being off-policy with replay buffers whose batch size is set independently of episode length.

---

## 2026-08-18 — Add HAPPO algorithm (MPE2 + MAMuJoCo)

**Change:** Implemented HAPPO as a fifth algorithm (`agents/happo/`), ported from the reference in the gitignored `on-policy` clone (`onpolicy/algorithms/happo/` + the sequential loop in `runner/separated/base_runner.py:135-182`). Wired into `main.py` as `--agent happo` with `configs/happo.yaml`, plus `train_mamujoco_happo.py` / `configs/mamujoco_happo.yaml`, `scripts/train/happo.py`, and support in `scripts/test.py` and `reports/report.py`.

Implemented as a **subclass of `MAPPO` overriding only `update()`** — the networks, rollout, and checkpoint format are identical, so subclassing states that relationship in code rather than duplicating ~150 lines. The override is the algorithm: agents are updated sequentially in a random permutation, and each agent's clipped surrogate is scaled by a `factor` accumulating the policy change of the agents already updated (`factor *= exp(new_log_prob - old_log_prob)`). The factor lives in MAPPO's `RolloutBuffer` as `factor_buf`, all-ones unless HAPPO sets it, so MAPPO is unaffected.

Verified: factor starts at 1.0 (so the first agent in the order faces exactly MAPPO's objective) and varies in training (0.92–1.91, std 0.15 over 40 episodes — not silently pinned); update order is genuinely random; all six MPE2 envs train; MAMuJoCo entry point runs and writes to the correct run-dir layout; HAPPO and MAPPO both learn over 600 episodes on `simple_spread_v3` (HAPPO −27.5 → −22.4, MAPPO −32.8 → −24.5 — indicative only, single seed, far too short to rank them); and MAPPO's own buffer behaviour is unchanged.

**Purpose:** *(user asked for it after learning what HAPPO is from the MAT paper's baseline table)* HAPPO completes an ablation the thesis did not previously have. HAPPO and MAT rest on the **same** multi-agent advantage decomposition theorem but apply it in opposite places — HAPPO sequentialises the *optimisation procedure*, MAT sequentialises the *architecture* — while MAPPO applies it nowhere. With all three sharing this repo's conventions (per-agent networks where applicable, same buffer, same training loop template, same metrics), the comparison isolates one idea instead of contrasting unrelated algorithms.

Relevant context for interpreting results: in the MAT paper's own HalfCheetah table, MAT (~5631) and HAPPO (~5646) are effectively tied while MAPPO trails badly (~3304) — i.e. published evidence suggests the *theorem*, not the transformer, carries most of the benefit. If that reproduces here, the honest framing of MAT's advantage is scalability (one decoder pass vs. N sequential updates), not raw performance.
