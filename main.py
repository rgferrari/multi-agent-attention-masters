#!/usr/bin/env python3
"""Entry point to run a selected agent in a selected MPE2 environment."""

import argparse
import importlib
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from types import SimpleNamespace
from typing import Dict, List

import gymnasium as gym
import torch

from agents.maac.attention_sac import AttentionSAC
from agents.maddpg_torch import MADDPG
from agents.maddpg_torch.maddpg import MADDPGConfig
from config import load_config
from scripts.train.maac import train_maac
from scripts.train.maddpg import train_maddpg


def _warn_if_no_display(render_mode: str) -> None:
    if render_mode != "human":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        "Warning: render_mode=human but no DISPLAY/WAYLAND detected. "
        "Use X/WSLg or run with xvfb if needed."
    )


def _build_maac(env, args: argparse.Namespace) -> AttentionSAC:
    agent_ids = list(env.agents)
    obs_spaces = [env.observation_space(a) for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]

    agent_init_params = []
    sa_size = []
    for obsp, acsp in zip(obs_spaces, act_spaces):
        if not isinstance(acsp, gym.spaces.Discrete):
            raise ValueError("MAAC runner supports Discrete action spaces only")
        agent_init_params.append(
            {
                "num_in_pol": obsp.shape[0],
                "num_out_pol": acsp.n,
            }
        )
        sa_size.append((obsp.shape[0], acsp.n))

    init_dict = {
        "gamma": args.maac_gamma,
        "tau": args.maac_tau,
        "pi_lr": args.maac_pi_lr,
        "q_lr": args.maac_q_lr,
        "reward_scale": args.maac_reward_scale,
        "pol_hidden_dim": args.maac_pol_hidden_dim,
        "critic_hidden_dim": args.maac_critic_hidden_dim,
        "attend_heads": args.maac_attend_heads,
        "agent_init_params": agent_init_params,
        "sa_size": sa_size,
    }

    model = AttentionSAC(
        agent_init_params,
        sa_size,
        gamma=args.maac_gamma,
        tau=args.maac_tau,
        pi_lr=args.maac_pi_lr,
        q_lr=args.maac_q_lr,
        reward_scale=args.maac_reward_scale,
        pol_hidden_dim=args.maac_pol_hidden_dim,
        critic_hidden_dim=args.maac_critic_hidden_dim,
        attend_heads=args.maac_attend_heads,
    )
    model.init_dict = init_dict
    return model


def _maac_actions(agent: AttentionSAC, obs: Dict[str, torch.Tensor], device: torch.device, agent_ids: List[str]) -> Dict[str, int]:
    agent.prep_rollouts("gpu" if device.type == "cuda" else "cpu")
    obs_list = [torch.as_tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0) for a in agent_ids]
    onehots = agent.step(obs_list, explore=True)
    return {agent_id: int(onehot.argmax(dim=1).item()) for agent_id, onehot in zip(agent_ids, onehots)}


def _random_actions(env, agent_ids: List[str]) -> Dict[str, int]:
    return {agent_id: env.action_space(agent_id).sample() for agent_id in agent_ids}


def _build_maddpg(env, device: torch.device, args: argparse.Namespace) -> MADDPG:
    agent_ids = list(env.agents)
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


def _with_defaults(user_cfg: Dict[str, object] | None, defaults: Dict[str, object]) -> Dict[str, object]:
    merged = defaults.copy()
    if user_cfg:
        merged.update(user_cfg)
    return merged


def _make_args(agent: str, cfg: Dict[str, object], cli_train: bool) -> argparse.Namespace:
    common_defaults = {
        "env": "simple_spread_v3",
        "episodes": 1,
        "max_cycles": 200,
        "seed": 0,
        "render_mode": "human",
        "continuous_actions": True,
        "use_gpu": False,
        "run_name": None,
        "log_dir": "runs",
        "save_dir": "checkpoints",
        "save_every": 10,
        "save_final": False,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_metric": "mean_reward",
        "early_stop_mode": "max",
        "train": False,
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
        "buffer_size": 1_000_000,
        "batch_size": 1024,
        "steps_per_update": 100,
        "num_updates": 4,
        "gamma": 0.95,
        "tau": 0.01,
        "pi_lr": 0.01,
        "q_lr": 0.01,
        "reward_scale": 10.0,
        "pol_hidden_dim": 128,
        "critic_hidden_dim": 128,
        "attend_heads": 4,
        "attend_mag_reg": 1e-3,
    }

    common = _with_defaults(cfg.get("common"), common_defaults)
    common["train"] = bool(cli_train or common.get("train"))

    if agent == "maddpg":
        maddpg_cfg = _with_defaults(cfg.get("maddpg"), maddpg_defaults)
        merged = {**common, **maddpg_cfg}
        return SimpleNamespace(**merged)
    if agent == "maac":
        maac_cfg = _with_defaults(cfg.get("maac"), maac_defaults)
        merged = {**common, **{f"maac_{k}": v for k, v in maac_cfg.items()}}
        return SimpleNamespace(**merged)

    return SimpleNamespace(**common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["maac", "maddpg", "random"], help="Agent type")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--override", nargs='+', action="append", default=[], help="Override config key=value (dot paths)")
    parser.add_argument("--train", action="store_true", help="Force training mode")
    args = parser.parse_args()

    overrides = [item for group in args.override for item in group]
    cfg = load_config(args.config, overrides)
    run_args = _make_args(args.agent, cfg, args.train)

    env_module = importlib.import_module(f"mpe2.{run_args.env}")
    env = env_module.parallel_env(
        render_mode=run_args.render_mode,
        max_cycles=run_args.max_cycles,
        continuous_actions=run_args.continuous_actions,
    )
    _warn_if_no_display(run_args.render_mode)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")
    agent = None
    maddpg = None
    if args.agent == "maac":
        env.reset(seed=run_args.seed)
        agent = _build_maac(env, run_args)
        agent.prep_rollouts("gpu" if device.type == "cuda" else "cpu")
    elif args.agent == "maddpg":
        env.reset(seed=run_args.seed)
        maddpg = _build_maddpg(env, device, run_args)

    if run_args.train and args.agent == "maac" and agent is not None:
        train_maac(env, agent, device, run_args)
        env.close()
        return
    if run_args.train and args.agent == "maddpg" and maddpg is not None:
        train_maddpg(env, maddpg, run_args)
        env.close()
        return

    for ep in range(run_args.episodes):
        obs, _ = env.reset(seed=run_args.seed + ep)
        done = False
        steps = 0
        prev_obs = obs
        agent_ids = list(env.agents)
        while not done and steps < run_args.max_cycles:
            if args.agent == "maac" and agent is not None:
                actions = _maac_actions(agent, obs, device, agent_ids)
            elif args.agent == "maddpg" and maddpg is not None:
                actions, actions_onehot = maddpg.select_actions(obs, explore=False)
            else:
                actions = _random_actions(env, agent_ids)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())
            prev_obs = obs
            steps += 1

    env.close()


if __name__ == "__main__":
    main()
