"""Unified report generator for all env types.

Usage:
    python reports/report.py --env_type mpe2
    python reports/report.py --env_type mamujoco
    python reports/report.py --env_type lugobots
    python reports/report.py --env_type mpe2 --env simple_spread_v3
"""
from __future__ import annotations

import argparse
import csv
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
    env_type: str, tag: str, env_filter: str | None
) -> list[tuple[str, str, list[int], list[float]]]:
    """Return [(model, env, steps, values)] from runs/<env_type>/<env>/<model>/runN/."""
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
        if not vals:
            continue
        series.append((model, env, [v.step for v in vals], [v.value for v in vals]))
    return series


def _single_env_axes(fig: plt.Figure) -> plt.Axes:
    ax = fig.add_subplot(1, 1, 1)
    return ax


def plot_mean_reward(
    env_type: str, mode: str, env_filter: str | None, out_dir: Path, window: int = 25
) -> Path:
    assert mode in {'std', 'ci'}
    series = _load_scalar_series(env_type, 'episode/mean_reward', env_filter)
    if not series:
        raise RuntimeError(f'No mean_reward scalars found under runs/{env_type}')

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
        ax.set_ylabel('Mean reward', fontsize=9)
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
        axes[0].set_ylabel('Mean reward', fontsize=9)
        if y_min is not None and y_max is not None:
            pad = max(1e-9, 0.05 * (y_max - y_min if y_max > y_min else 1.0))
            for ax in axes:
                ax.set_ylim(y_min - pad, y_max + pad)
        fig.tight_layout()

    stem = f'mean_reward_{mode}'
    out = out_dir / f'{stem}.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)
    return out


def plot_episode_length(env_type: str, env_filter: str | None, out_dir: Path) -> Path:
    series = _load_scalar_series(env_type, 'episode/length', env_filter)
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

    out = out_dir / 'episode_length.pdf'
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate training reports')
    parser.add_argument('--env_type', required=True, choices=['mpe2', 'mamujoco', 'lugobots'])
    parser.add_argument('--env', default=None, help='Filter to a single environment')
    args = parser.parse_args()

    out_dir = REPORTS_ROOT / args.env_type
    if args.env:
        out_dir = out_dir / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    print(plot_mean_reward(args.env_type, 'std', args.env, out_dir))
    print(plot_mean_reward(args.env_type, 'ci', args.env, out_dir))
    print(plot_episode_length(args.env_type, args.env, out_dir))
    csv_path, pdf_path = plot_training_time(args.env_type, args.env, out_dir)
    print(csv_path)
    print(pdf_path)


if __name__ == '__main__':
    main()
