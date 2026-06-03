from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

root = Path('runs')
files = sorted(root.glob('*/*/run*/events.out.tfevents.*'))

# Aggregate durations per (env, agent)
durations = {}  # env -> {agent: duration_seconds}
for p in files:
    parts = p.parts
    try:
        idx = parts.index('runs')
        agent = parts[idx + 1]
        env = parts[idx + 2]
    except ValueError:
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
    durations.setdefault(env, {})[agent] = durations.get(env, {}).get(agent, 0.0) + dur_s

# Convert to minutes and prepare lists
envs = sorted(durations.keys())
agents = ['maddpg', 'maac']
minutes = {agent: [] for agent in agents}
for env in envs:
    env_dict = durations.get(env, {})
    for agent in agents:
        val = env_dict.get(agent, 0.0) / 60.0
        minutes[agent].append(val)

out_dir = Path('reports')
out_dir.mkdir(exist_ok=True)
csv_path = out_dir / 'training_time_by_env.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['env', 'agent', 'minutes'])
    for i, env in enumerate(envs):
        for agent in agents:
            writer.writerow([env, agent, f"{minutes[agent][i]:.6f}"])

# Plot grouped bars: for each env, maddpg (blue) left, maac (red) right
import numpy as np
x = np.arange(len(envs))
width = 0.35
fig, ax = plt.subplots(figsize=(max(6, len(envs)*0.8), 3.2))
rects1 = ax.bar(x - width/2, minutes['maddpg'], width, label='MADDPG', color='C0')
rects2 = ax.bar(x + width/2, minutes['maac'], width, label='MAAC', color='C1')

ax.set_ylabel('Training time (minutes)')
ax.set_xticks(x)
ax.set_xticklabels(envs, rotation=30, ha='right', fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.2)

# Annotate bars with values
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        if h > 0:
            ax.annotate(f'{h:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
pdf_path = out_dir / 'training_time_by_env.pdf'
png_path = out_dir / 'training_time_by_env.png'
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, bbox_inches='tight')
print(csv_path)
print(pdf_path)
print(png_path)
