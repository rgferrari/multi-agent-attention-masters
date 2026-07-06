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
