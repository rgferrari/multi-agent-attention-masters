"""MADDPG training loop for MPE2 parallel envs."""

from __future__ import annotations

import os
from collections import deque

import torch
from tqdm import tqdm

from agents.maddpg_torch import MADDPG
from scripts.train.utils import get_next_shared_run_dirs


def train_maddpg(env, maddpg: MADDPG, args) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Missing tensorboard. Install with: pip install tensorboard") from exc

    # Default layout is runs/maddpg/<env>/runN; callers (e.g. MAMuJoCo) may
    # override the relative path via args.run_subpath.
    rel = getattr(args, "run_subpath", None) or os.path.join("maddpg", args.env)
    log_base = os.path.join(args.log_dir, rel)
    save_base = os.path.join(args.save_dir, rel)
    log_dir, save_dir = get_next_shared_run_dirs(log_base, save_base)
    writer = SummaryWriter(log_dir=log_dir)

    if args.save_every > 0 or args.save_final:
        os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    update_step = 0
    total_steps = args.episodes * args.max_cycles
    progress = tqdm(total=max(1, total_steps), desc="Training", dynamic_ncols=True)

    best_metric = None
    epochs_no_improve = 0
    recent_metrics = deque(maxlen=max(1, getattr(args, "early_stop_window", 1)))

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        prev_obs = obs
        agent_ids = list(env.agents)
        ep_rewards = {agent_id: 0.0 for agent_id in agent_ids}

        while not done and steps < args.max_cycles:
            actions, actions_onehot = maddpg.select_actions(obs, explore=True)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in agent_ids:
                ep_rewards[agent_id] += float(rewards[agent_id])

            dones = {k: bool(terminations[k] or truncations[k]) for k in terminations}
            maddpg.store_transition(prev_obs, actions_onehot, rewards, obs, dones)
            if global_step >= args.start_steps and global_step % args.update_every == 0:
                for _ in range(args.updates_per_step):
                    metrics = maddpg.update()
                    if metrics:
                        for i, agent_id in enumerate(agent_ids):
                            writer.add_scalar(f"agent{i}/losses/critic_loss", metrics[f"critic_loss_{i}"], update_step)
                            writer.add_scalar(f"agent{i}/losses/actor_loss", metrics[f"actor_loss_{i}"], update_step)
                        update_step += 1

            done = all(terminations.values()) or all(truncations.values())
            prev_obs = obs
            steps += 1
            global_step += 1
            progress.update(1)

        ep_rewards_arr = [ep_rewards[a] for a in agent_ids]
        mean_ep_rew = float(sum(ep_rewards_arr) / max(1, len(ep_rewards_arr)))
        recent_metrics.append(mean_ep_rew)
        writer.add_scalar("episode/mean_reward", mean_ep_rew, ep)
        writer.add_scalar("episode/mean_reward_window", float(sum(recent_metrics) / max(1, len(recent_metrics))), ep)
        writer.add_scalar("episode/length", steps, ep)
        for agent_id in agent_ids:
            writer.add_scalar(f"episode/agent_reward/{agent_id}", float(ep_rewards[agent_id]), ep)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"maddpg_ep{ep + 1}.pt")
            torch.save(maddpg.state_dict(), ckpt_path)

        # Track best checkpoint independently from early stopping.
        start_episode = max(0, getattr(args, "early_stop_start_episode", 0))
        metric_value = float(sum(recent_metrics) / max(1, len(recent_metrics)))
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
                    ckpt_path = os.path.join(save_dir, "maddpg_best.pt")
                    torch.save(maddpg.state_dict(), ckpt_path)
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
        ckpt_path = os.path.join(save_dir, "maddpg_final.pt")
        torch.save(maddpg.state_dict(), ckpt_path)

    progress.close()
    writer.close()
