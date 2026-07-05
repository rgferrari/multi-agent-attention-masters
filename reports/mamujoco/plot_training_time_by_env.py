from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np

ROOT = Path('runs/mamujoco')
OUT_DIR = Path('reports/mamujoco')
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(ROOT.glob('*/*/run*/events.out.tfevents.*'))
files = [p for p in files if p.parent.name.startswith('run')]

durations = {}  # scenario -> {agent: duration_seconds}
for p in files:
    parts = p.parts
    try:
        idx = next(i for i, part in enumerate(parts) if part == 'mamujoco')
        scenario = parts[idx + 1]
        agent = parts[idx + 2]
    except (StopIteration, IndexError):
        continue
    ea = event_accumulator.EventAccumulator(str(p))
    ea.Reload()
    tag = 'episode/mean_reward'
    if tag in ea.Tags().get('scalars', []):
        vals = ea.Scalars(tag)
        if len(vals) < 2:
            continue
        start = vals[0].wall_time
        end = vals[-1].wall_time
    else:
        events = ea.Events()
        if not events:
            continue
        start = events[0].wall_time
        end = events[-1].wall_time
    dur_s = end - start
    durations.setdefault(scenario, {})[agent] = durations.get(scenario, {}).get(agent, 0.0) + dur_s

scenarios = sorted(durations.keys())
agents = ['maddpg', 'maac_continuous']
agent_labels = {'maddpg': 'MADDPG', 'maac_continuous': 'MAAC'}
minutes = {agent: [] for agent in agents}
for scenario in scenarios:
    env_dict = durations.get(scenario, {})
    for agent in agents:
        val = env_dict.get(agent, 0.0) / 60.0
        minutes[agent].append(val)

csv_path = OUT_DIR / 'training_durations.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['scenario', 'agent', 'minutes'])
    for i, scenario in enumerate(scenarios):
        for agent in agents:
            writer.writerow([scenario, agent, f"{minutes[agent][i]:.6f}"])

x = np.arange(len(scenarios))
width = 0.35
fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 1.2), 3.2))
rects1 = ax.bar(x - width / 2, minutes['maddpg'], width, label='MADDPG', color='C0')
rects2 = ax.bar(x + width / 2, minutes['maac_continuous'], width, label='MAAC', color='C1')

ax.set_ylabel('Training time (minutes)')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)


def autolabel(rects):
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


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
pdf_path = OUT_DIR / 'training_time_by_env.pdf'
png_path = OUT_DIR / 'training_time_by_env.png'
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, bbox_inches='tight')
plt.close(fig)
print(csv_path)
print(pdf_path)
print(png_path)
