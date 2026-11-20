"""Unified report generator for all env types.

Usage:
    python reports/report.py --env_type mpe2
    python reports/report.py --env_type mamujoco
    python reports/report.py --env_type lugobots
    python reports/report.py --env_type mpe2 --env simple_spread_v3

    # Compare algorithms fairly when they were trained for different budgets
    python reports/report.py --env_type mpe2 --common_horizon
    python reports/report.py --env_type mpe2 --max_episodes 25000

    # Plot the metric that actually selected *_best.pt / drove early stopping
    python reports/report.py --env_type mpe2 --metric mean_reward_window
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

RUNS_ROOT = Path('runs')
REPORTS_ROOT = Path('reports')

AGENT_LABELS: dict[str, str] = {
    'maac': 'MAAC',
    'maac_continuous': 'MAAC',
    'maac_adversarial': 'MAAC (adv)',
    'maddpg': 'MADDPG',
    'mappo': 'MAPPO',
    'happo': 'HAPPO',
    'mat': 'MAT',
}

# The metric training itself uses to select ``*_best.pt`` and drive early
# stopping. ``episode/mean_reward`` is the raw per-episode value; the windowed
# tag is the rolling average that actually decided which checkpoint was kept.
REWARD_TAGS: dict[str, str] = {
    'mean_reward': 'episode/mean_reward',
    'mean_reward_window': 'episode/mean_reward_window',
}


def ema(values: list[float], weight: float = 0.99) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    last = values[0]
    for v in values[1:]:
        last = last * weight + v * (1.0 - weight)
        smoothed.append(last)
    return smoothed


def rolling_stats(values: list[float], window: int = 25) -> tuple[list[float], list[float], list[int]]:
    means, stds, counts = [], [], []
    for i in range(len(values)):
        seg = values[max(0, i - window + 1): i + 1]
        n = len(seg)
        m = sum(seg) / n
        s = (sum((x - m) ** 2 for x in seg) / (n - 1)) ** 0.5 if n > 1 else 0.0
        means.append(m)
        stds.append(s)
        counts.append(n)
    return means, stds, counts


def _band(
    mode: str, means: list[float], stds: list[float], counts: list[int]
) -> tuple[list[float], list[float]]:
    if mode == 'std':
        return [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)]
    z = NormalDist().inv_cdf(0.975)
    hw = [z * s / (n ** 0.5) if n > 1 else 0.0 for s, n in zip(stds, counts)]
    return [m - h for m, h in zip(means, hw)], [m + h for m, h in zip(means, hw)]


def _load_scalar_series(
    env_type: str, tag: str, env_filter: str | None, max_episodes: int | None = None
) -> list[tuple[str, str, list[int], list[float]]]:
    """Return [(model, env, steps, values)] from runs/<env_type>/<env>/<model>/runN/.

    ``max_episodes`` truncates every series at that step, so algorithms trained
    for different budgets are compared over a common horizon rather than one
    curve's tail being plotted against nothing.
    """
    root = RUNS_ROOT / env_type
    series = []
    for p in sorted(root.glob('*/*/run*/events.out.tfevents.*')):
        if not p.parent.name.startswith('run'):
            continue
        parts = p.parts
        try:
            idx = next(i for i, part in enumerate(parts) if part == env_type)
            env = parts[idx + 1]
            model = parts[idx + 2]
        except (StopIteration, IndexError):
            continue
        if env_filter and env != env_filter:
            continue
        ea = event_accumulator.EventAccumulator(str(p))
        ea.Reload()
        if tag not in ea.Tags().get('scalars', []):
            continue
        vals = ea.Scalars(tag)
        if max_episodes is not None:
            vals = [v for v in vals if v.step <= max_episodes]
        if not vals:
            continue
        series.append((model, env, [v.step for v in vals], [v.value for v in vals]))
    return series


def common_horizon(env_type: str, tag: str, env_filter: str | None) -> int | None:
    """Shortest final step across runs — the largest fair comparison horizon.

    Computed per environment and then minimised, so one environment having only
    short runs does not silently truncate every other environment's curves.
    """
    series = _load_scalar_series(env_type, tag, env_filter)
    if not series:
        return None
    per_env: dict[str, list[int]] = {}
    for _, env, steps, _ in series:
        if steps:
            per_env.setdefault(env, []).append(steps[-1])
    if not per_env:
        return None
    return min(min(ends) for ends in per_env.values())


def _single_env_axes(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_subplot(1, 1, 1)
    return ax


def plot_mean_reward(
    env_type: str,
    mode: str,
    env_filter: str | None,
    out_dir: Path,
    window: int = 25,
    metric: str = 'mean_reward',
    max_episodes: int | None = None,
) -> Path:
    assert mode in {'std', 'ci'}
    tag = REWARD_TAGS[metric]
    series = _load_scalar_series(env_type, tag, env_filter, max_episodes)
    if not series:
        raise RuntimeError(f'No {tag} scalars found under runs/{env_type}')
    ylabel = 'Mean reward' if metric == 'mean_reward' else 'Mean reward (rolling window)'

    if env_filter:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for model, env, steps, rewards in series:
            means, stds, counts = rolling_stats(rewards, window)
            means = ema(means)
            lower, upper = _band(mode, means, stds, counts)
            ax.plot(steps, means, label=AGENT_LABELS.get(model, model), linewidth=1.3)
            ax.fill_between(steps, lower, upper, alpha=0.18)
        ax.set_title(env_filter, fontsize=10)
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
    else:
        models = sorted(set(m for m, _, _, _ in series))
        fig, axes = plt.subplots(1, len(models), figsize=(6.8, 2.8), sharey=True)
        if len(models) == 1:
            axes = [axes]
        y_min = y_max = None
        for ax, model in zip(axes, models):
            for m, env, steps, rewards in series:
                if m != model:
                    continue
                means, stds, counts = rolling_stats(rewards, window)
                means = ema(means)
                lower, upper = _band(mode, means, stds, counts)
                if means:
                    y_min = min(y_min if y_min is not None else means[0], min(means))
                    y_max = max(y_max if y_max is not None else means[0], max(means))
                ax.plot(steps, means, label=env, linewidth=1.3)
                ax.fill_between(steps, lower, upper, alpha=0.18)
            ax.set_title(AGENT_LABELS.get(model, model.upper()), fontsize=10)
            ax.set_xlabel('Episode', fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=6.5, ncol=1, frameon=False)
        axes[0].set_ylabel(ylabel, fontsize=9)
        if y_min is not None and y_max is not None:
            pad = max(1e-9, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
            for ax in axes:
                ax.set_ylim(y_min - pad, y_max + pad)
        fig.tight_layout()

    # Distinct stems so the two reward metrics (and truncated variants) do not
    # overwrite one another in the same output dir.
    stem = f'{metric}_{mode}'
    if max_episodes is not None:
        stem += f'_ep{max_episodes}'
    out = out_dir / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    return out


def plot_episode_length(
    env_type: str, env_filter: str | None, out_dir: Path, max_episodes: int | None = None
) -> Path:
    series = _load_scalar_series(env_type, 'episode/length', env_filter, max_episodes)
    if not series:
        raise RuntimeError(f'No episode/length scalars found under runs/{env_type}')

    if env_filter:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for model, env, steps, lengths in series:
            ax.plot(steps, ema(lengths), label=AGENT_LABELS.get(model, model), linewidth=1.3)
        ax.set_title(env_filter, fontsize=10)
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Episode length', fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
    else:
        models = sorted(set(m for m, _, _, _ in series))
        fig, axes = plt.subplots(1, len(models), figsize=(6.8, 2.8), sharey=True)
        if len(models) == 1:
            axes = [axes]
        for ax, model in zip(axes, models):
            for m, env, steps, lengths in series:
                if m != model:
                    continue
                ax.plot(steps, ema(lengths), label=env, linewidth=1.3)
            ax.set_title(AGENT_LABELS.get(model, model.upper()), fontsize=10)
            ax.set_xlabel('Episode', fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=6.5, ncol=1, frameon=False)
        axes[0].set_ylabel('Episode length', fontsize=9)
        fig.tight_layout()

    stem = 'episode_length' + (f'_ep{max_episodes}' if max_episodes is not None else '')
    out = out_dir / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    return out


def plot_training_time(
    env_type: str, env_filter: str | None, out_dir: Path
) -> tuple[Path, Path]:
    root = RUNS_ROOT / env_type
    durations: dict[str, dict[str, float]] = {}
    for p in sorted(root.glob('*/*/run*/events.out.tfevents.*')):
        if not p.parent.name.startswith('run'):
            continue
        parts = p.parts
        try:
            idx = next(i for i, part in enumerate(parts) if part == env_type)
            env = parts[idx + 1]
            model = parts[idx + 2]
        except (StopIteration, IndexError):
            continue
        if env_filter and env != env_filter:
            continue
        ea = event_accumulator.EventAccumulator(str(p))
        ea.Reload()
        tag = 'episode/mean_reward'
        if tag in ea.Tags().get('scalars', []):
            vals = ea.Scalars(tag)
            if len(vals) < 2:
                continue
            dur = vals[-1].wall_time - vals[0].wall_time
        else:
            events = ea.Events()
            if not events:
                continue
            dur = events[-1].wall_time - events[0].wall_time
        durations.setdefault(env, {})
        durations[env][model] = durations[env].get(model, 0.0) + dur

    envs = sorted(durations)
    models = sorted({m for d in durations.values() for m in d})
    minutes = {m: [durations.get(e, {}).get(m, 0.0) / 60.0 for e in envs] for m in models}

    csv_path = out_dir / 'training_time.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env', 'model', 'minutes'])
        for i, env in enumerate(envs):
            for m in models:
                w.writerow([env, m, f'{minutes[m][i]:.6f}'])

    x = np.arange(len(envs))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(6, len(envs) * 1.2), 3.2))
    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        rects = ax.bar(
            x + offset, minutes[model], width, label=AGENT_LABELS.get(model, model)
        )
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=7,
                )
    ax.set_ylabel('Training time (minutes)')
    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=30, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()

    pdf_path = out_dir / 'training_time.pdf'
    png_path = out_dir / 'training_time.png'
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    plt.close(fig)
    return csv_path, pdf_path


def _episode_length_has_signal(env_type: str, env: str) -> bool:
    """True if ``episode/length`` varies — i.e. episodes can end early.

    Data-driven rather than keyed on env_type: on MPE2 the length is pinned to
    ``max_cycles`` (verified constant across all 18 runs) and carries no
    information, but the same is true of MAMuJoCo HalfCheetah, which cannot
    fall. Ant *does* vary, and there the length encodes "the robot fell".
    """
    for _, _, _, values in _load_scalar_series(env_type, 'episode/length', env):
        if len(set(values)) > 1:
            return True
    return False


def _run_metadata(env_type: str, env: str) -> dict:
    """Summary of the runs backing this report, for thesis traceability."""
    meta: dict = {'env_type': env_type, 'env': env, 'runs': []}
    try:
        meta['git_rev'] = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        meta['git_rev'] = None

    for tag_key, tag in REWARD_TAGS.items():
        for model, _, steps, values in _load_scalar_series(env_type, tag, env):
            if tag_key != 'mean_reward_window':
                continue
            meta['runs'].append({
                'model': model,
                'episodes': steps[-1] if steps else 0,
                'final_mean_reward_window': values[-1] if values else None,
                'best_mean_reward_window': max(values) if values else None,
            })
    return meta


def generate_run_report(
    env_type: str,
    env: str,
    out_dir: Path | None = None,
    max_episodes: int | None = None,
    runs_root: str | Path | None = None,
) -> list[Path]:
    """Write the per-environment report figures. Used by the end-of-training hook.

    Emits both reward metrics (``mean_reward`` and the ``mean_reward_window``
    that actually selected ``*_best.pt``), and includes the episode-length plot
    only where that series carries signal. Individual failures are swallowed:
    a report must never fail a finished training run.
    """
    out_dir = out_dir or (REPORTS_ROOT / env_type / env)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Honour a non-default log_dir (the trainers accept one) so the report
    # describes the run that just finished rather than whatever is in ./runs.
    global RUNS_ROOT
    previous_root = RUNS_ROOT
    if runs_root is not None:
        RUNS_ROOT = Path(runs_root)
    try:
        return _generate_run_report_inner(env_type, env, out_dir, max_episodes, written)
    finally:
        RUNS_ROOT = previous_root


def _generate_run_report_inner(
    env_type: str,
    env: str,
    out_dir: Path,
    max_episodes: int | None,
    written: list[Path],
) -> list[Path]:
    for metric in REWARD_TAGS:
        for mode in ('std', 'ci'):
            try:
                written.append(
                    plot_mean_reward(env_type, mode, env, out_dir, metric=metric, max_episodes=max_episodes)
                )
            except (RuntimeError, ValueError, KeyError) as exc:
                print(f'  [report] skipped {metric}/{mode}: {exc}')

    if _episode_length_has_signal(env_type, env):
        try:
            written.append(plot_episode_length(env_type, env, out_dir, max_episodes=max_episodes))
        except (RuntimeError, ValueError) as exc:
            print(f'  [report] skipped episode_length: {exc}')

    try:
        meta_path = out_dir / 'run_meta.json'
        meta_path.write_text(json.dumps(_run_metadata(env_type, env), indent=2))
        written.append(meta_path)
    except (OSError, ValueError) as exc:
        print(f'  [report] skipped run_meta.json: {exc}')

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate training reports')
    parser.add_argument('--env_type', required=True, choices=['mpe2', 'mamujoco', 'lugobots'])
    parser.add_argument('--env', default=None, help='Filter to a single environment')
    parser.add_argument(
        '--metric',
        default='mean_reward',
        choices=sorted(REWARD_TAGS),
        help=(
            'Reward tag to plot. "mean_reward" is the raw per-episode value; '
            '"mean_reward_window" is the rolling average training used to pick '
            '*_best.pt and drive early stopping.'
        ),
    )
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=None,
        help=(
            'Truncate every curve at this episode, so algorithms trained for '
            'different budgets are compared over a common horizon.'
        ),
    )
    parser.add_argument(
        '--common_horizon',
        action='store_true',
        help=(
            'Derive --max_episodes automatically as the shortest run in the '
            'selection. Ignored if --max_episodes is given explicitly.'
        ),
    )
    args = parser.parse_args()

    out_dir = REPORTS_ROOT / args.env_type
    if args.env:
        out_dir = out_dir / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    max_episodes = args.max_episodes
    if max_episodes is None and args.common_horizon:
        max_episodes = common_horizon(args.env_type, REWARD_TAGS[args.metric], args.env)
        if max_episodes is None:
            raise SystemExit(f'No {REWARD_TAGS[args.metric]} scalars found under runs/{args.env_type}')
        print(f'common horizon: {max_episodes} episodes')

    print(plot_mean_reward(args.env_type, 'std', args.env, out_dir, metric=args.metric, max_episodes=max_episodes))
    print(plot_mean_reward(args.env_type, 'ci', args.env, out_dir, metric=args.metric, max_episodes=max_episodes))
    print(plot_episode_length(args.env_type, args.env, out_dir, max_episodes=max_episodes))
    csv_path, pdf_path = plot_training_time(args.env_type, args.env, out_dir)
    print(csv_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
