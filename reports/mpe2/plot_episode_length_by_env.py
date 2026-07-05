from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('runs')
out_path = Path('reports/episode_length_by_env.pdf')
out_path.parent.mkdir(exist_ok=True)

# Strong EMA smoothing for a TensorBoard-like curve.
def smooth(values, weight=0.99):
    if not values:
        return []
    smoothed = [values[0]]
    last = values[0]
    for v in values[1:]:
        last = last * weight + v * (1.0 - weight)
        smoothed.append(last)
    return smoothed

series = []
for p in sorted(root.glob('*/*/run*/events.out.tfevents.*')):
    parts = p.parts
    try:
        agent = parts[parts.index('runs') + 1]
        env = parts[parts.index('runs') + 2]
    except ValueError:
        continue
    ea = event_accumulator.EventAccumulator(str(p))
    ea.Reload()
    if 'episode/length' not in ea.Tags().get('scalars', []):
        continue
    vals = ea.Scalars('episode/length')
    steps = [v.step for v in vals]
    lengths = [v.value for v in vals]
    series.append((agent, env, steps, smooth(lengths, 0.99)))

if not series:
    raise RuntimeError('No episode/length scalars found')

agents = sorted(set(a for a, _, _, _ in series))
fig, axes = plt.subplots(1, len(agents), figsize=(6.8, 2.8), sharey=True)
if len(agents) == 1:
    axes = [axes]

for ax, agent in zip(axes, agents):
    for a, env, steps, lengths in series:
        if a != agent:
            continue
        ax.plot(steps, lengths, label=env, linewidth=1.3)
    ax.set_title(agent.upper(), fontsize=10)
    ax.set_xlabel('Episode', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, ncol=1, frameon=False)

axes[0].set_ylabel('Episode length', fontsize=9)
fig.tight_layout()
fig.savefig(out_path, bbox_inches='tight')
print(out_path)
