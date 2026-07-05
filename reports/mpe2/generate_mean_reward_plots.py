from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


ROOT = Path('runs')
OUT_DIR = Path('reports')
OUT_DIR.mkdir(exist_ok=True)


def ema(values: list[float], weight: float = 0.99) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    last = values[0]
    for value in values[1:]:
        last = last * weight + value * (1.0 - weight)
        smoothed.append(last)
    return smoothed


def rolling_stats(values: list[float], window: int = 25) -> tuple[list[float], list[float], list[int]]:
    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start : i + 1]
        n = len(segment)
        mean = sum(segment) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in segment) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        means.append(mean)
        stds.append(std)
        counts.append(n)
    return means, stds, counts


def load_mean_reward_series() -> list[tuple[str, str, list[int], list[float]]]:
    series = []
    for p in sorted(ROOT.glob('*/*/run*/events.out.tfevents.*')):
        parts = p.parts
        try:
            agent = parts[parts.index('runs') + 1]
            env = parts[parts.index('runs') + 2]
        except ValueError:
            continue
        ea = event_accumulator.EventAccumulator(str(p))
        ea.Reload()
        if 'episode/mean_reward' not in ea.Tags().get('scalars', []):
            continue
        vals = ea.Scalars('episode/mean_reward')
        if not vals:
            continue
        steps = [v.step for v in vals]
        rewards = [v.value for v in vals]
        series.append((agent, env, steps, rewards))
    if not series:
        raise RuntimeError('No mean_reward scalars found')
    return series


def plot(mode: str, window: int = 25) -> Path:
    assert mode in {'std', 'ci'}
    series = load_mean_reward_series()
    agents = sorted(set(a for a, _, _, _ in series))
    fig, axes = plt.subplots(1, len(agents), figsize=(6.8, 2.8), sharey=True)
    if len(agents) == 1:
        axes = [axes]

    y_min = None
    y_max = None

    for ax, agent in zip(axes, agents):
        for a, env, steps, rewards in series:
            if a != agent:
                continue
            means, stds, counts = rolling_stats(rewards, window=window)
            means = ema(means, 0.99)
            if means:
                local_min = min(means)
                local_max = max(means)
                y_min = local_min if y_min is None else min(y_min, local_min)
                y_max = local_max if y_max is None else max(y_max, local_max)
            if mode == 'std':
                lower = [m - s for m, s in zip(means, stds)]
                upper = [m + s for m, s in zip(means, stds)]
            else:
                z = NormalDist().inv_cdf(0.975)
                half_width = [z * s / (n ** 0.5) if n > 1 else 0.0 for s, n in zip(stds, counts)]
                lower = [m - hw for m, hw in zip(means, half_width)]
                upper = [m + hw for m, hw in zip(means, half_width)]

            ax.plot(steps, means, label=env, linewidth=1.3)
            ax.fill_between(steps, lower, upper, alpha=0.18)

        ax.set_title(agent.upper(), fontsize=10)
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

    out_path = OUT_DIR / f'mean_episode_reward_{mode}_narrow.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.png'), bbox_inches='tight')
    return out_path


if __name__ == '__main__':
    print(plot('std'))
    print(plot('ci'))