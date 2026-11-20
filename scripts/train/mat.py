"""MAT training loop for parallel envs (on-policy).

Structurally identical to ``train_mappo`` -- collect one rollout of up to
``max_cycles`` steps, compute GAE, run a PPO update, reset the buffer -- since
MAT is PPO-trained. The difference is that a single shared transformer is
updated, so the per-agent loss scalars MAPPO logs become one set of model-wide
scalars here.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

from agents.mat import MAT
from agents.mat.buffer import MATRolloutBuffer
from scripts.train.evaluate import maybe_evaluate
from scripts.train.report_hook import maybe_generate_report
from scripts.train.utils import get_next_shared_run_dirs


def train_mat(env, mat: MAT, args) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("Missing tensorboard. Install with: pip install tensorboard") from exc

    # Default layout is runs/mpe2/<env>/mat/runN; callers (e.g. MAMuJoCo) may
    # override the relative path via args.run_subpath.
    rel = getattr(args, "run_subpath", None) or os.path.join("mpe2", args.env, "mat")
    log_base = os.path.join(args.log_dir, rel)
    save_base = os.path.join(args.save_dir, rel)
    log_dir, save_dir = get_next_shared_run_dirs(log_base, save_base)
    writer = SummaryWriter(log_dir=log_dir)

    if args.save_every > 0 or args.save_final:
        os.makedirs(save_dir, exist_ok=True)

    buffer = MATRolloutBuffer(
        max_steps=args.max_cycles,
        obs_dim=mat.obs_dim,
        act_dim=mat.action_dim,
        num_agents=mat.num_agents,
        device=mat.device,
        is_discrete=mat.is_discrete,
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
            actions, actions_store, log_probs, values = mat.select_actions(obs, deterministic=False)
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in agent_ids:
                ep_rewards[agent_id] += float(rewards[agent_id])

            obs_stacked = mat._stack_obs(obs)
            rew_arr = np.array([[float(rewards[a])] for a in mat.agent_ids], dtype=np.float32)
            # mask = 1 - terminated: keep the bootstrap on time-limit truncations,
            # cut it on true terminations.
            mask_arr = np.array([[1.0 - float(terminations[a])] for a in mat.agent_ids], dtype=np.float32)
            buffer.insert(obs_stacked, actions_store, log_probs, values, rew_arr, mask_arr)

            obs = next_obs
            steps += 1
            global_step += 1
            progress.update(1)
            if all(terminations.values()) or all(truncations.values()):
                break

        next_value = mat.get_values(obs)
        buffer.compute_returns(next_value, args.gamma, args.gae_lambda)
        metrics = mat.update(buffer)

        # MAT shares one model across agents, so these are model-wide rather
        # than the per-agent losses MAPPO/MADDPG log.
        writer.add_scalar("losses/policy_loss", metrics["policy_loss"], ep)
        writer.add_scalar("losses/value_loss", metrics["value_loss"], ep)
        writer.add_scalar("losses/entropy", metrics["entropy"], ep)
        writer.add_scalar("losses/ratio", metrics["ratio"], ep)

        ep_rewards_arr = [ep_rewards[a] for a in agent_ids]
        mean_ep_rew = float(sum(ep_rewards_arr) / max(1, len(ep_rewards_arr)))
        recent_metrics.append(mean_ep_rew)
        writer.add_scalar("episode/mean_reward", mean_ep_rew, ep)
        writer.add_scalar("episode/mean_reward_window", float(sum(recent_metrics) / max(1, len(recent_metrics))), ep)
        writer.add_scalar("episode/length", steps, ep)
        for agent_id in agent_ids:
            writer.add_scalar(f"episode/agent_reward/{agent_id}", float(ep_rewards[agent_id]), ep)

        # Deterministic evaluation -- the quantity the MAT paper reports.
        maybe_evaluate(env, mat.select_actions, args, ep, writer)

        if args.save_every > 0 and (ep + 1) % args.save_every == 0:
            ckpt_path = os.path.join(save_dir, f"mat_ep{ep + 1}.pt")
            torch.save(mat.state_dict(), ckpt_path)

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
                    ckpt_path = os.path.join(save_dir, "mat_best.pt")
                    torch.save(mat.state_dict(), ckpt_path)
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
        ckpt_path = os.path.join(save_dir, "mat_final.pt")
        torch.save(mat.state_dict(), ckpt_path)

    progress.close()
    writer.close()

    maybe_generate_report(args, log_dir)
