#!/usr/bin/env python3
"""Evaluate a trained agent on an MPE2 env."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

import gymnasium as gym
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.maac.attention_sac import AttentionSAC
from agents.maddpg_torch import MADDPG
from agents.maddpg_torch.maddpg import MADDPGConfig
from config import load_config


def _warn_if_no_display(render_mode: str) -> None:
    if render_mode != "human":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        "Warning: render_mode=human but no DISPLAY/WAYLAND detected. "
        "Use X/WSLg or run with xvfb if needed."
    )


def _with_defaults(user_cfg: Dict[str, object] | None, defaults: Dict[str, object]) -> Dict[str, object]:
    merged = defaults.copy()
    if user_cfg:
        merged.update(user_cfg)
    return merged


def _make_args(agent: str, cfg: Dict[str, object]) -> argparse.Namespace:
    common_defaults = {
        "env": "simple_spread_v3",
        "episodes": 20,
        "max_cycles": 200,
        "seed": 0,
        "render_mode": "human",
        "continuous_actions": True,
        "use_gpu": False,
    }
    maddpg_defaults = {
        "gamma": 0.95,
        "tau": 0.01,
        "actor_lr": 1e-3,
        "critic_lr": 1e-3,
        "hidden_dim": 128,
        "batch_size": 1024,
        "buffer_size": 1_000_000,
        "start_steps": 10_000,
        "update_every": 100,
        "updates_per_step": 1,
        "discrete_temperature": 1.0,
        "expl_noise": 0.1,
        "epsilon": 0.1,
    }
    maac_defaults = {
        "gamma": 0.95,
        "tau": 0.01,
        "pi_lr": 0.01,
        "q_lr": 0.01,
        "reward_scale": 10.0,
        "pol_hidden_dim": 128,
        "critic_hidden_dim": 128,
        "attend_heads": 4,
    }

    common = _with_defaults(cfg.get("common"), common_defaults)

    if agent == "maddpg":
        maddpg_cfg = _with_defaults(cfg.get("maddpg"), maddpg_defaults)
        merged = {**common, **maddpg_cfg}
        return SimpleNamespace(**merged)
    if agent == "maac":
        maac_cfg = _with_defaults(cfg.get("maac"), maac_defaults)
        merged = {**common, **{f"maac_{k}": v for k, v in maac_cfg.items()}}
        return SimpleNamespace(**merged)

    return SimpleNamespace(**common)


def _build_maddpg(env, device: torch.device, args: argparse.Namespace) -> MADDPG:
    agent_ids = list(getattr(env, "agents", env.possible_agents))
    obs_spaces = [env.observation_space(a) for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]
    cfg = MADDPGConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        start_steps=args.start_steps,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        discrete_temperature=args.discrete_temperature,
        expl_noise=args.expl_noise,
        epsilon=args.epsilon,
    )
    return MADDPG(agent_ids, obs_spaces, act_spaces, device=device, config=cfg)


def _maac_actions(agent: AttentionSAC, obs: Dict[str, torch.Tensor], device: torch.device, agent_ids: List[str]) -> Dict[str, int]:
    agent.prep_rollouts("gpu" if device.type == "cuda" else "cpu")
    obs_list = [torch.as_tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0) for a in agent_ids]
    onehots = agent.step(obs_list, explore=False)
    return {agent_id: int(onehot.argmax(dim=1).item()) for agent_id, onehot in zip(agent_ids, onehots)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["maac", "maddpg"], help="Agent type")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--override", action="append", default=[], help="Override config key=value (dot paths)")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--episodes", type=int, default=None, help="Override episodes")
    parser.add_argument("--render_mode", default=None, help="Override render mode")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    run_args = _make_args(args.agent, cfg)
    if args.episodes is not None:
        run_args.episodes = args.episodes
    if args.render_mode is not None:
        run_args.render_mode = args.render_mode

    env_module = importlib.import_module(f"mpe2.{run_args.env}")
    env = env_module.parallel_env(
        render_mode=run_args.render_mode,
        max_cycles=run_args.max_cycles,
        continuous_actions=run_args.continuous_actions,
    )
    _warn_if_no_display(run_args.render_mode)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")
    agent_ids = list(getattr(env, "agents", env.possible_agents))

    if args.agent == "maac":
        model = AttentionSAC.init_from_save(args.checkpoint, load_critic=True)
        model.prep_rollouts("gpu" if device.type == "cuda" else "cpu")
        for ep in range(run_args.episodes):
            obs, _ = env.reset(seed=run_args.seed + ep)
            done = False
            steps = 0
            ep_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            while not done and steps < run_args.max_cycles:
                actions = _maac_actions(model, obs, device, agent_ids)
                obs, rewards, terminations, truncations, infos = env.step(actions)
                for agent_id in agent_ids:
                    ep_rewards[agent_id] += float(rewards[agent_id])
                done = all(terminations.values()) or all(truncations.values())
                steps += 1
            mean_ep = sum(ep_rewards.values()) / max(1, len(ep_rewards))
            print(f"Episode {ep + 1}: mean_reward={mean_ep:.4f}")
    else:
        model = _build_maddpg(env, device, run_args)
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        for ep in range(run_args.episodes):
            obs, _ = env.reset(seed=run_args.seed + ep)
            done = False
            steps = 0
            ep_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            while not done and steps < run_args.max_cycles:
                actions, _ = model.select_actions(obs, explore=False)
                obs, rewards, terminations, truncations, infos = env.step(actions)
                for agent_id in agent_ids:
                    ep_rewards[agent_id] += float(rewards[agent_id])
                done = all(terminations.values()) or all(truncations.values())
                steps += 1
            mean_ep = sum(ep_rewards.values()) / max(1, len(ep_rewards))
            print(f"Episode {ep + 1}: mean_reward={mean_ep:.4f}")

    env.close()


if __name__ == "__main__":
    main()
