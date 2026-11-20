# CLAUDE.md — reports/

Documentation for the report/plot generation scripts in this directory. See the root [CLAUDE.md](../CLAUDE.md) for project-wide setup and commands.

All scripts read tensorboard event files from `runs/` and write figures under `reports/`. None of this is training or evaluation — it's post-hoc visualization of what's already in `runs/`.

## Metrics: what is reported, and what each one is for

Training writes many scalars (see the root [CLAUDE.md](../CLAUDE.md)'s "Evaluation & Metrics" for the full per-algorithm list). **Reports consume only a small subset.** This section defines that subset, so comparisons across algorithms stay on a common footing.

### The comparison metrics (what actually goes in a report)

| Metric | Source tag | What it measures | Use it for |
|---|---|---|---|
| **Mean episode reward** | `episode/mean_reward` | This episode's reward averaged over agents. The primary performance number. | **Cross-algorithm comparison.** Directly comparable between MAAC/MADDPG/MAPPO/MAT, because all four log it identically. Default for `--metric`. |
| **Windowed mean reward** | `episode/mean_reward_window` | Rolling average over `early_stop_window` episodes. **This is the quantity training used to select `*_best.pt` and to drive early stopping.** | Reporting best-checkpoint results, where the raw per-episode value describes a slightly different quantity than the one the checkpoint was chosen by. Select with `--metric mean_reward_window`. |
| **Rolling mean ± std / 95% CI** | derived from `episode/mean_reward` | Smoothed level plus run-to-run spread over a 25-episode window. | Judging whether a gap between algorithms exceeds noise. A difference inside overlapping bands is not a result. |
| **Episode length** | `episode/length` | Steps before termination/truncation/`max_cycles`. | Diagnostic, **not** performance. In fixed-horizon MPE2 (`max_cycles: 25`) it is near-constant and carries no signal; it matters in MAMuJoCo, where early termination means the robot fell. |
| **Wall-clock training time** | event file first/last timestamps | Real seconds elapsed for the run. | Cost comparison only. **Not a fair efficiency measure** — it conflates algorithm cost with hardware, CPU/GPU, and contention from other runs on the same machine. |

### Metrics logged but *not* currently reported

These exist in `runs/` and are available for deeper analysis, but no report script reads them:

- **`episode/agent_reward/{agent_id}`** — per-agent, un-averaged. Essential for the asymmetric envs (`simple_adversary`, `simple_crypto`, `simple_tag`, `simple_push`), where a rising team mean can hide one agent winning at another's expense. The averaged `mean_reward` is misleading on its own in adversarial settings.
- **Per-algorithm loss/diagnostic tags** — `agent{i}/losses/*`, `grad_norms/*`, `policy_entropy`, `attention/*`. **These are not comparable across algorithms** and must never be plotted on shared axes: MAAC's `pol_loss` is a SAC objective, MAPPO's `actor_loss` is a clipped surrogate minus entropy, MADDPG's is a deterministic policy gradient, and MAT logs one model-wide set instead of per-agent. A lower `critic_loss` in one algorithm than another says nothing about which learned better — they are different loss functions with different scales. **Only mean reward is overlaid across algorithms**; diagnostics stay within-algorithm debugging signals.
  - MAAC subsamples these via `maac.log_every` (default 25 ≈ once per episode), since its updates run ~25× per episode and logging every one made diagnostics 58% of the event file (now 5%). Set `log_every: 1` for fine-grained debugging of a short run.
  - MADDPG's losses are logged against `update_step`, a **different x-axis** from its own episode curves (verified: 14899 vs 59999 on a 60k run). Do not plot them against episode-indexed series. MAAC's `niter` happens to track episodes closely (59955 vs 59999), and MAPPO/MAT are exactly aligned.
- **`attention/*`** — MAAC only. Heatmap figures are the most expensive event type in the file; `common.attn_heatmap_every` (default 500 episodes) now controls the frequency, and was previously hardcoded and untunable.
- **MAT logs `losses/{policy_loss,value_loss,entropy,ratio}` at the top level**, not under `agent{i}/`, because it trains one shared model. Any script that assumes a per-agent loss namespace will silently find nothing for MAT.

### Standardizing comparisons

**Unequal training budgets — use `--common_horizon` or `--max_episodes`.** Runs were not all trained for the same number of episodes (on MPE2: MAAC/MADDPG ≈60k, MAPPO ≈25k). Plotting them untruncated compares a longer run's tail against nothing.

```bash
# Truncate every curve at the shortest run in the selection (prints the horizon)
python reports/report.py --env_type mpe2 --common_horizon

# Or pin an explicit budget
python reports/report.py --env_type mpe2 --max_episodes 25000
```

`--common_horizon` computes the shortest final step **per environment and then takes the minimum**, so one environment's short runs do not silently truncate every other environment. `--max_episodes` wins if both are given. Truncated figures get an `_ep<N>` filename suffix, so they never overwrite untruncated ones.

**Which reward metric — `--metric`.** Two tags are available, and they answer different questions:

```bash
python reports/report.py --env_type mpe2 --metric mean_reward         # default: raw per-episode
python reports/report.py --env_type mpe2 --metric mean_reward_window  # what selected *_best.pt
```

Use `mean_reward_window` when reporting best-checkpoint results, since that is the quantity training actually optimised against. Outputs are named `mean_reward_*` / `mean_reward_window_*` respectively.

**Remaining gap: single-run series.** Where several runs exist for one (env, agent), the ± band reflects the rolling window *within* a run, not variance *across seeds*. Genuine seed variance requires multiple runs per configuration aggregated together — no current script does this, so bands must not be read as seed variance.

## Automatic per-run reports

Set `common.auto_report: true` (already on in `configs/{maac,maac_continuous,maddpg,mappo,mat}.yaml`) and each training run writes its own figures when it finishes. The hook lives in [../scripts/train/report_hook.py](../scripts/train/report_hook.py) and is called after `writer.close()` in all five training loops.

It produces, into `reports/<env_type>/<env>/`:
- `mean_reward_{std,ci}` and `mean_reward_window_{std,ci}` — both reward metrics
- `episode_length` — **only when that series actually varies** (see `_episode_length_has_signal`); skipped on MPE2 and on MAMuJoCo HalfCheetah, emitted for Ant
- `run_meta.json` — episodes completed, final/best windowed reward per model, and the `git_rev` the run was produced at (thesis traceability)

Deliberate scope: **the report covers the environment that just finished, not a cross-algorithm comparison**, because the other algorithms may not have been trained yet and partial overlays mislead. The four-way comparison is generated by `train_all.sh` after every run completes.

Every failure is swallowed and printed as `[report] skipped ...` — a plotting error must never fail a finished training run. A non-default `--override common.log_dir=...` redirects both the source runs and the output figures, so scratch runs never overwrite `reports/`.

## Current: unified generator

- [report.py](report.py) — `python reports/report.py --env_type {mpe2,mamujoco,lugobots} [--env <env_name>]`. Scans `runs/<env_type>/*/*/run*/events.out.tfevents.*`, and for each run extracts `episode/mean_reward` and `episode/length` scalars (rolling-window EMA-smoothed) plus wall-clock training duration. Writes to `reports/<env_type>/` (or `reports/<env_type>/<env>/` if `--env` is given):
  - `mean_reward_std.pdf`/`.png` and `mean_reward_ci.pdf`/`.png` — rolling mean ± std band / 95% CI band, one subplot per agent (or overlaid per-agent lines if `--env` narrows to a single environment)
  - `episode_length.pdf`/`.png` — same layout, for episode length
  - `training_time.csv`/`.pdf`/`.png` — per-(env, agent) wall-clock training duration bar chart, computed from event first/last timestamps
  - Agent display names come from the `AGENT_LABELS` dict (`maac`, `maac_continuous`, `maac_adversarial`, `maddpg`, `mappo`, `mat`); anything else falls back to the raw model dir name. Note `maac` and `maac_continuous` both render as `"MAAC"` — intentional (same algorithm, different action space), but they are indistinguishable if both appear in one figure.
  - `--metric`, `--max_episodes`, and `--common_horizon` control which reward tag is plotted and over what horizon — see "Standardizing comparisons" above

This supersedes the older per-env-type scripts below (added 2026-07-05 in the "change reports organization" commit, which split MPE2's original top-level scripts into `mpe2/` and added `mamujoco/` equivalents without removing them). Prefer `report.py` for new reports.

## Legacy: per-env-type scripts

`reports/mpe2/` and `reports/mamujoco/` each contain their own near-duplicate trio, predating `report.py`'s unification. Still present and functional, but redundant with it:

- `generate_mean_reward_plots.py` — mean-reward plots only, hardcoded output paths (writes into `reports/` or the script's own directory rather than accepting `--env_type`/`--env`)
- `plot_episode_length_by_env.py` — episode-length plot only
- `plot_training_time_by_env.py` — training-time bar chart + CSV only
- `training_durations.csv` — checked-in output of the above, per env type

## MPE2-only: checkpoint visual inspection

- [mpe2/validate_and_plot_mpe_runs.py](mpe2/validate_and_plot_mpe_runs.py) — loads saved checkpoints (`checkpoints/{agent}/{env}/run1/{agent}_best.pt`, falling back to `_final.pt`) and renders a grid of environment frames at fixed step indices (`CAPTURE_STEPS = [0, 5, 25]`) for visual/qualitative inspection — not a metric, a sanity check. Hardcoded to `AGENTS = ['maac', 'maddpg']` on `ENVS = ['simple_speaker_listener_v4', 'simple_adversary_v3']`; not extended to MAPPO or MAMuJoCo.
  - Note: its `OUT_DIR` is computed as `reports/validation_grid/` (two parents up from the script, i.e. the repo's `reports/` root) rather than `reports/mpe2/validation_grid/` — a leftover from before the script moved into `mpe2/`. The checked-in `validation_grid.pdf`/`.png` currently live under `reports/mpe2/validation_grid/` from before that move; re-running the script today will instead write to `reports/validation_grid/`.

## Checked-in outputs

The `.pdf`/`.png`/`.csv` files alongside each script are committed report artifacts (figures for the thesis), not something to hand-edit — regenerate them by rerunning the corresponding script.
