"""MAPPO training loop for parallel envs (on-policy).

Each "episode" collects one fresh rollout of up to ``max_cycles`` steps, computes
GAE returns/advantages, then runs a PPO update over it -- in contrast to the
off-policy replay-buffer cadence of MAAC/MADDPG. Run-dir layout, tensorboard
scalar names, checkpointing, and early stopping all follow ``train_maddpg``.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import torch
from tqdm import tqdm

from agents.mappo import MAPPO
from agents.mappo.buffer import RolloutBuffer
from scripts.train.utils import get_next_shared_run_dirs


def train_mappo(env, mappo: MAPPO, args) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Missing tensorboard. Install with: pip install tensorboard") from exc

    # Default layout is runs/mpe2/<env>/mappo/runN; callers (e.g. MAMuJoCo) may
    # override the relative path via args.run_subpath.
    rel = getattr(args, "run_subpath", None) or os.path.join("mpe2", args.env, "mappo")
    log_base = os.path.join(args.log_dir, rel)
    save_base = os.path.join(args.save_dir, rel)
    log_dir, save_dir = get_next_shared_run_dirs(log_base, save_base)
    writer = SummaryWriter(log_dir=log_dir)

    if args.save_every > 0 or args.save_final:
        os.makedirs(save_dir, exist_ok=True)

    buffer = RolloutBuffer(
        max_steps=args.max_cycles,
        obs_dims=mappo.obs_dims,
        act_dims=mappo.act_store_dims,
        global_obs_dim=mappo.total_obs_dim,
        num_agents=mappo.num_agents,
        device=mappo.device,
    )

    global_step = 0
    total_steps = args.episodes * args.max_cycles
    progress = tqdm(total=max(1, total_steps), desc="Training", dynamic_ncols=True, file=sys.stderr)

    best_metric = None
    epochs_no_improve = 0
    recent_metrics = deque(maxlen=max(1, getattr(args, "early_stop_window", 1)))

    for ep in range(args.episodes):
        buffer.reset()
        obs, _ = env.reset(seed=args.seed + ep)
        agent_ids = list(env.agents)
        ep_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        steps = 0

        for _ in range(args.max_cycles):
            actions, actions_store, log_probs, values = mappo.select_actions(obs, deterministic=False)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in agent_ids:
                ep_rewards[agent_id] += float(rewards[agent_id])

            obs_list = [obs[a] for a in agent_ids]
            rew_list = [float(rewards[a]) for a in agent_ids]
            # mask = 1 - terminated: keep the bootstrap on time-limit truncations,
            # cut it on true terminations.
            masks = [1.0 - float(terminations[a]) for a in agent_ids]
            buffer.insert(obs_list, actions_store, log_probs, values, rew_list, masks)

            obs = next_obs
            steps += 1
            global_step += 1
            progress.update(1)
            if all(terminations.values()) or all(truncations.values()):
                break

        next_value = mappo.get_values(obs)
        buffer.compute_returns(next_value, args.gamma, args.gae_lambda)
        metrics = mappo.update(buffer)

        for i, agent_id in enumerate(agent_ids):
            writer.add_scalar(f"agent{i}/losses/actor_loss", metrics[f"actor_loss_{i}"], ep)
            writer.add_scalar(f"agent{i}/losses/value_loss", metrics[f"value_loss_{i}"], ep)
            writer.add_scalar(f"agent{i}/losses/entropy", metrics[f"entropy_{i}"], ep)

        ep_rewards_arr = [ep_rewards[a] for a in agent_ids]
        mean_ep_rew = float(sum(ep_rewards_arr) / max(1, len(ep_rewards_arr)))
        recent_metrics.append(mean_ep_rew)
        writer.add_scalar("episode/mean_reward", mean_ep_rew, ep)
        writer.add_scalar("episode/mean_reward_window", float(sum(recent_metrics) / max(1, len(recent_metrics))), ep)
        writer.add_scalar("episode/length", steps, ep)
        for agent_id in agent_ids:
            writer.add_scalar(f"episode/agent_reward/{agent_id}", float(ep_rewards[agent_id]), ep)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"mappo_ep{ep + 1}.pt")
            torch.save(mappo.state_dict(), ckpt_path)

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
                    ckpt_path = os.path.join(save_dir, "mappo_best.pt")
                    torch.save(mappo.state_dict(), ckpt_path)
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
        ckpt_path = os.path.join(save_dir, "mappo_final.pt")
        torch.save(mappo.state_dict(), ckpt_path)

    progress.close()
    writer.close()
