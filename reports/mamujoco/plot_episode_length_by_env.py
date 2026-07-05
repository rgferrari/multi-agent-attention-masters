from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('runs/mamujoco')
OUT_DIR = Path('reports/mamujoco')
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_LABELS = {'maac_continuous': 'MAAC', 'maddpg': 'MADDPG'}


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
for p in sorted(ROOT.glob('*/*/run*/events.out.tfevents.*')):
    if not p.parent.name.startswith('run'):
        continue
    parts = p.parts
    try:
        idx = next(i for i, part in enumerate(parts) if part == 'mamujoco')
        scenario = parts[idx + 1]
        agent = parts[idx + 2]
    except (StopIteration, IndexError):
        continue
    ea = event_accumulator.EventAccumulator(str(p))
    ea.Reload()
    if 'episode/length' not in ea.Tags().get('scalars', []):
        continue
    vals = ea.Scalars('episode/length')
    steps = [v.step for v in vals]
    lengths = [v.value for v in vals]
    series.append((agent, scenario, steps, smooth(lengths, 0.99)))

if not series:
    raise RuntimeError('No episode/length scalars found under runs/mamujoco')

agents = sorted(set(a for a, _, _, _ in series))
fig, axes = plt.subplots(1, len(agents), figsize=(6.8, 2.8), sharey=True)
if len(agents) == 1:
    axes = [axes]

for ax, agent in zip(axes, agents):
    for a, scenario, steps, lengths in series:
        if a != agent:
            continue
        ax.plot(steps, lengths, label=scenario, linewidth=1.3)
    ax.set_title(AGENT_LABELS.get(agent, agent.upper()), fontsize=10)
    ax.set_xlabel('Episode', fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, ncol=1, frameon=False)

axes[0].set_ylabel('Episode length', fontsize=9)
fig.tight_layout()

pdf_path = OUT_DIR / 'episode_length_by_env.pdf'
png_path = OUT_DIR / 'episode_length_by_env.png'
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, bbox_inches='tight')
plt.close(fig)
print(pdf_path)
print(png_path)
