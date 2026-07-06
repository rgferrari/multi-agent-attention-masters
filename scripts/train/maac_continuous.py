"""Training loop for continuous MAAC (AttentionSACContinuous)."""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from agents.maac_continuous.attention_sac_continuous import AttentionSACContinuous
from agents.maac.utils.buffer import ReplayBuffer
from scripts.train.maac import _agent_label, _attention_heatmap_figure
from scripts.train.utils import get_next_shared_run_dirs


def _to_device(dev: str) -> str:
    return "gpu" if dev == "cuda" else "cpu"


def _compute_attention_matrix(agent: AttentionSACContinuous, sample, agent_ids, device):
    """Return (mean_attn, std_attn, labels) as [n, n] numpy arrays (diagonal = 0).

    ``mean_attn`` averages attention weights over heads and batch; ``std_attn``
    is their std over the batch (averaged over heads).
    """
    obs, acs, _, _, _ = sample
    n = len(agent_ids)
    critic_in = list(zip(obs, acs))

    agent.critic.eval()
    with torch.no_grad():
        attend_per_agent = agent.critic(critic_in, return_attend=True)
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


def train_maac_continuous(env, agent: AttentionSACContinuous, device: torch.device, args) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Missing tensorboard. Install with: pip install tensorboard") from exc

    rel = getattr(args, "run_subpath", None) or os.path.join("maac_continuous", args.env)
    log_dir, save_dir = get_next_shared_run_dirs(
        os.path.join(args.log_dir, rel),
        os.path.join(args.save_dir, rel),
    )
    writer = SummaryWriter(log_dir=log_dir)

    agent_ids = list(env.agents)
    obs_dims = [env.observation_space(a).shape[0] for a in agent_ids]
    act_dims = [env.action_space(a).shape[0] for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]

    buffer = ReplayBuffer(args.buffer_size, len(agent_ids), obs_dims, act_dims)

    if args.save_every > 0 or args.save_final:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    update_step = 0
    total_steps = args.episodes * args.max_cycles
    progress = tqdm(total=max(1, total_steps), desc="Training", dynamic_ncols=True)

    best_metric = None
    epochs_no_improve = 0
    recent_metrics = deque(maxlen=max(1, getattr(args, "early_stop_window", 1)))

    dev_str = _to_device(device.type)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        prev_obs = obs
        ep_rewards = np.zeros(len(agent_ids), dtype=np.float32)

        while not done and steps < args.max_cycles:
            if global_step < args.start_steps:
                actions = {a: act_spaces[i].sample() for i, a in enumerate(agent_ids)}
                actions_np = [actions[a][None, :] for a in agent_ids]
            else:
                agent.prep_rollouts(dev_str)
                obs_tensors = [
                    torch.as_tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0)
                    for a in agent_ids
                ]
                with torch.no_grad():
                    step_results = agent.step(obs_tensors, explore=True)
                actions = {}
                actions_np = []
                for i, a_id in enumerate(agent_ids):
                    act_t, _ = step_results[i]
                    act_arr = act_t.squeeze(0).cpu().numpy()
                    actions[a_id] = act_arr
                    actions_np.append(act_arr[None, :])

            obs, rewards, terminations, truncations, infos = env.step(actions)
            for i, a_id in enumerate(agent_ids):
                ep_rewards[i] += float(rewards[a_id])

            obs_arr = np.array([prev_obs[a] for a in agent_ids], dtype=object)[None, :]
            next_obs_arr = np.array([obs[a] for a in agent_ids], dtype=object)[None, :]
            rewards_np = np.array([rewards[a] for a in agent_ids], dtype=np.float32)[None, :]
            dones_np = np.array(
                [bool(terminations[a] or truncations[a]) for a in agent_ids], dtype=np.float32
            )[None, :]

            buffer.push(obs_arr, actions_np, rewards_np, next_obs_arr, dones_np)

            if (
                global_step >= args.start_steps
                and len(buffer) >= args.batch_size
                and global_step % args.update_every == 0
            ):
                agent.prep_training(dev_str)
                for _ in range(args.num_updates):
                    sample = buffer.sample(args.batch_size, to_gpu=(device.type == "cuda"))
                    agent.update_critic(sample, logger=writer)
                    agent.update_policies(sample, logger=writer)
                    agent.update_all_targets()
                    update_step += 1
                agent.prep_rollouts(dev_str)

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
        for i, a_id in enumerate(agent_ids):
            writer.add_scalar(f"episode/agent_reward/{a_id}", float(ep_rewards[i]), ep)

        attn_heatmap_every = getattr(args, "attn_heatmap_every", 100)
        if (ep + 1) % attn_heatmap_every == 0 and len(buffer) >= args.batch_size:
            sample = buffer.sample(args.batch_size, to_gpu=(device.type == "cuda"))
            attn, attn_std, labels = _compute_attention_matrix(agent, sample, agent_ids, device)
            writer.add_figure("attention/heatmap_mean",
                              _attention_heatmap_figure(attn, labels, "Mean attention weights"), ep)
            std_vmax = max(0.05, float(attn_std.max()))
            writer.add_figure("attention/heatmap_std",
                              _attention_heatmap_figure(attn_std, labels,
                                                        "Attention std across states",
                                                        vmax=std_vmax, cmap="Reds"), ep)
            for i, src in enumerate(labels):
                for j, dst in enumerate(labels):
                    if i != j:
                        writer.add_scalar(f"attention/{src}_to_{dst}", attn[i, j], ep)
                        writer.add_scalar(f"attention/{src}_to_{dst}_std", attn_std[i, j], ep)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            agent.save(os.path.join(save_dir, f"maac_continuous_ep{ep + 1}.pt"))

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
                    agent.save(os.path.join(save_dir, "maac_continuous_best.pt"))
            elif best_metric is not None:
                epochs_no_improve += 1

        if (
            args.early_stop_patience
            and args.early_stop_patience > 0
            and can_track_best
            and best_metric is not None
            and epochs_no_improve >= args.early_stop_patience
        ):
            break

    if args.save_final:
        agent.save(os.path.join(save_dir, "maac_continuous_final.pt"))

    progress.close()
    writer.close()
