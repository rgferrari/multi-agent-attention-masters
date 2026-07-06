"""MAAC training loop for MPE2 parallel envs."""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from agents.maac.attention_sac import AttentionSAC
from agents.maac.utils.buffer import ReplayBuffer
from scripts.train.utils import get_next_shared_run_dirs


_AGENT_ABBREVS = {
    "adversary": "adv",
    "agent": "a",
    "good_agent": "good",
    "leadadversary": "lead_adv",
}


def _agent_label(agent_id: str) -> str:
    for prefix, abbrev in _AGENT_ABBREVS.items():
        if agent_id.startswith(prefix + "_"):
            return abbrev + "_" + agent_id[len(prefix) + 1:]
    return agent_id


def _compute_attention_matrix(agent: "AttentionSAC", sample, agent_ids, device):
    """Return (mean_attn, std_attn, labels) as [n, n] numpy arrays (diagonal = 0).

    ``mean_attn`` averages attention weights over heads and batch; ``std_attn``
    is their std over the batch (averaged over heads). High std means the
    connection's weight varies a lot across states -- i.e. attention is doing
    state-dependent work that a flat batch-mean would hide.
    """
    obs, acs, _, _, _ = sample
    n = len(agent_ids)
    critic_in = list(zip(obs, acs))

    agent.critic.eval()
    with torch.no_grad():
        attend_per_agent = agent.critic(critic_in, return_q=False, return_attend=True)
    agent.critic.train()

    if n == 1:
        attend_per_agent = [attend_per_agent]

    mean_attn = np.zeros((n, n))
    std_attn = np.zeros((n, n))
    for i, weights in enumerate(attend_per_agent):
        # weights: [n_heads, batch, 1, n_other]
        mean_w = weights.mean(axis=(0, 1, 2))           # [n_other]
        std_w = weights.std(axis=1).mean(axis=(0, 1))   # std over batch, avg heads
        others = [j for j in range(n) if j != i]
        for k, j in enumerate(others):
            mean_attn[i, j] = float(mean_w[k])
            std_attn[i, j] = float(std_w[k])

    labels = [_agent_label(a) for a in agent_ids]
    return mean_attn, std_attn, labels


def _attention_heatmap_figure(attn: np.ndarray, labels, title, vmax=1.0, cmap='Blues'):
    """Build a matplotlib Figure from a precomputed attention matrix."""
    from matplotlib.figure import Figure

    n = len(labels)
    fig = Figure(figsize=(max(4, n + 1), max(3, n)))
    ax = fig.add_subplot(111)
    im = ax.imshow(attn, vmin=0, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Attended agent')
    ax.set_ylabel('Attending agent')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    thresh = 0.6 * vmax
    for row in range(n):
        for col in range(n):
            if row != col:
                color = 'white' if attn[row, col] > thresh else 'black'
                ax.text(col, row, f'{attn[row, col]:.2f}',
                        ha='center', va='center', fontsize=8, color=color)
    fig.tight_layout()
    return fig


def _maac_actions_with_onehot(
    agent: AttentionSAC,
    obs: dict,
    device: torch.device,
    agent_ids: List[str],
) -> tuple[dict, List[torch.Tensor]]:
    agent.prep_rollouts("gpu" if device.type == "cuda" else "cpu")
    obs_list = [torch.as_tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0) for a in agent_ids]
    onehots = agent.step(obs_list, explore=True)
    actions = {agent_id: int(onehot.argmax(dim=1).item()) for agent_id, onehot in zip(agent_ids, onehots)}
    return actions, onehots


def train_maac(env, agent: AttentionSAC, device: torch.device, args) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Missing tensorboard. Install with: pip install tensorboard") from exc

    rel = getattr(args, "run_subpath", None) or os.path.join("mpe2", args.env, "maac")
    log_base = os.path.join(args.log_dir, rel)
    save_base = os.path.join(args.save_dir, rel)
    run_name = getattr(args, 'run_name', None)
    if run_name:
        log_dir = os.path.join(log_base, run_name)
        save_dir = os.path.join(save_base, run_name)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    else:
        log_dir, save_dir = get_next_shared_run_dirs(log_base, save_base)
    writer = SummaryWriter(log_dir=log_dir)

    agent_ids = list(env.agents)
    obs_spaces = [env.observation_space(a) for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]
    obs_dims = [sp.shape[0] for sp in obs_spaces]
    act_dims = [sp.n for sp in act_spaces]

    buffer = ReplayBuffer(args.maac_buffer_size, len(agent_ids), obs_dims, act_dims)

    if args.save_every > 0 or args.save_final:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    total_steps = args.episodes * args.max_cycles
    progress = tqdm(total=max(1, total_steps), desc="Training", dynamic_ncols=True, file=sys.stderr)

    best_metric = None
    epochs_no_improve = 0
    recent_metrics = deque(maxlen=max(1, getattr(args, "early_stop_window", 1)))

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        prev_obs = obs
        ep_rewards = np.zeros(len(agent_ids), dtype=np.float32)

        while not done and steps < args.max_cycles:
            actions, onehots = _maac_actions_with_onehot(agent, obs, device, agent_ids)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            for i, agent_id in enumerate(agent_ids):
                ep_rewards[i] += float(rewards[agent_id])

            # MAAC environments can expose different observation sizes per agent.
            # Keep them as object arrays so the replay buffer can store each agent's
            # native observation shape without forcing a rectangular stack.
            obs_arr = np.array([prev_obs[a] for a in agent_ids], dtype=object)[None, ...]
            next_obs_arr = np.array([obs[a] for a in agent_ids], dtype=object)[None, ...]
            rews_arr = [rewards[a] for a in agent_ids]
            dones_arr = [bool(terminations[a] or truncations[a]) for a in agent_ids]

            obs_np = obs_arr
            next_obs_np = next_obs_arr
            rewards_np = np.asarray(rews_arr, dtype=np.float32)[None, ...]
            dones_np = np.asarray(dones_arr, dtype=np.float32)[None, ...]
            actions_np = [onehot.detach().cpu().numpy() for onehot in onehots]

            buffer.push(obs_np, actions_np, rewards_np, next_obs_np, dones_np)
            if len(buffer) >= args.maac_batch_size and global_step % args.maac_steps_per_update == 0:
                agent.prep_training("gpu" if device.type == "cuda" else "cpu")
                for _ in range(args.maac_num_updates):
                    sample = buffer.sample(args.maac_batch_size, to_gpu=args.use_gpu)
                    agent.update_critic(sample, logger=writer)
                    agent.update_policies(sample, logger=writer)
                    agent.update_all_targets()
                agent.prep_rollouts("gpu" if device.type == "cuda" else "cpu")

            done = all(terminations.values()) or all(truncations.values())
            prev_obs = obs
            steps += 1
            global_step += 1
            progress.update(1)

        mean_ep_rew = float(ep_rewards.mean())
        recent_metrics.append(mean_ep_rew)
        writer.add_scalar("episode/mean_reward", mean_ep_rew, ep)
        writer.add_scalar("episode/mean_reward_window", float(np.mean(recent_metrics)), ep)
        writer.add_scalar("episode/length", steps, ep)
        for i, agent_id in enumerate(agent_ids):
            writer.add_scalar(f"episode/agent_reward/{agent_id}", float(ep_rewards[i]), ep)

        attn_heatmap_every = getattr(args, 'attn_heatmap_every', 100)
        if (ep + 1) % attn_heatmap_every == 0 and len(buffer) >= args.maac_batch_size:
            sample = buffer.sample(args.maac_batch_size, to_gpu=args.use_gpu)
            attn, attn_std, labels = _compute_attention_matrix(agent, sample, agent_ids, device)
            writer.add_figure('attention/heatmap_mean',
                              _attention_heatmap_figure(attn, labels, 'Mean attention weights'), ep)
            std_vmax = max(0.05, float(attn_std.max()))
            writer.add_figure('attention/heatmap_std',
                              _attention_heatmap_figure(attn_std, labels,
                                                        'Attention std across states',
                                                        vmax=std_vmax, cmap='Reds'), ep)
            for i, src in enumerate(labels):
                for j, dst in enumerate(labels):
                    if i != j:
                        writer.add_scalar(f'attention/{src}_to_{dst}', attn[i, j], ep)
                        writer.add_scalar(f'attention/{src}_to_{dst}_std', attn_std[i, j], ep)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"maac_ep{ep + 1}.pt")
            agent.save(ckpt_path)

        # Track best checkpoint independently from early stopping.
        start_episode = max(0, getattr(args, "early_stop_start_episode", 0))
        metric_value = float(np.mean(recent_metrics))
        can_track_best = (ep + 1) >= start_episode
        improved = False
        if can_track_best:
            if best_metric is None:
                improved = True
            elif args.early_stop_mode == "min":
                improved = metric_value < best_metric - args.early_stop_min_delta
            else:
                improved = metric_value > best_metric + args.early_stop_min_delta

            if improved:
                best_metric = metric_value
                epochs_no_improve = 0
                if args.save_final:
                    ckpt_path = os.path.join(save_dir, "maac_best.pt")
                    agent.save(ckpt_path)
            elif best_metric is not None:
                epochs_no_improve += 1

        # Early stopping remains optional.
        if (
            args.early_stop_patience
            and args.early_stop_patience > 0
            and can_track_best
            and best_metric is not None
            and epochs_no_improve >= args.early_stop_patience
        ):
            break

    if args.save_final:
        ckpt_path = os.path.join(save_dir, "maac_final.pt")
        agent.save(ckpt_path)

    progress.close()
    writer.close()
