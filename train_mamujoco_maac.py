#!/usr/bin/env python3
"""Train continuous MAAC (AttentionSACContinuous) on MAMuJoCo environments.

Examples
--------
    # HalfCheetah 2x3 (config defaults)
    python train_mamujoco_maac.py --config configs/maac_continuous.yaml --train

    # Ant 2x4
    python train_mamujoco_maac.py --config configs/maac_continuous.yaml \
        --scenario Ant --agent_conf 2x4 --train

    # Override hyperparameters
    python train_mamujoco_maac.py --config configs/maac_continuous.yaml \
        --override "maac_continuous.pi_lr=0.0005" "common.episodes=5000" --train
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from types import SimpleNamespace

import torch
from gymnasium_robotics import mamujoco_v1

from agents.maac_continuous.attention_sac_continuous import AttentionSACContinuous
from config import load_config
from scripts.train.maac_continuous import train_maac_continuous


def _build_env(scenario: str, agent_conf: str, agent_obsk=None, render_mode=None):
    kwargs = {"scenario": scenario, "agent_conf": agent_conf, "render_mode": render_mode}
    if agent_obsk is not None:
        kwargs["agent_obsk"] = agent_obsk
    return mamujoco_v1.parallel_env(**kwargs)


def _make_args(cfg: dict, cli_train: bool) -> SimpleNamespace:
    common_defaults = {
        "env": "ma_halfcheetah_2x3",
        "episodes": 20000,
        "max_cycles": 1000,
        "seed": 0,
        "use_gpu": False,
        "log_dir": "runs",
        "save_dir": "checkpoints",
        "save_every": 0,
        "save_final": True,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.01,
        "early_stop_start_episode": 100,
        "early_stop_window": 10,
        "early_stop_metric": "mean_reward",
        "early_stop_mode": "max",
        "train": False,
    }
    maac_continuous_defaults = {
        "buffer_size": 1_000_000,
        "batch_size": 1024,
        "tau": 0.01,
        "gamma": 0.95,
        "pi_lr": 0.001,
        "q_lr": 0.001,
        "reward_scale": 10.0,
        "pol_hidden_dim": 256,
        "critic_hidden_dim": 256,
        "attend_heads": 4,
        "start_steps": 25_000,
        "update_every": 100,
        "num_updates": 1,
    }

    common = {**common_defaults, **(cfg.get("common") or {})}
    common["train"] = bool(cli_train or common.get("train"))
    maac_cfg = {**maac_continuous_defaults, **(cfg.get("maac_continuous") or {})}

    return SimpleNamespace(**common, **maac_cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuous MAAC on MAMuJoCo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--agent_conf", default=None)
    parser.add_argument("--override", nargs='+', action="append", default=[])
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    overrides = [item for group in args.override for item in group]
    cfg = load_config(args.config, overrides)
    run_args = _make_args(cfg, args.train)

    if not run_args.train:
        raise SystemExit("Nothing to do: pass --train (or set common.train: true in the config).")

    mamujoco_cfg = cfg.get("mamujoco") or {}
    scenario = args.scenario or mamujoco_cfg.get("scenario", "HalfCheetah")
    agent_conf = args.agent_conf or mamujoco_cfg.get("agent_conf", "2x3")
    agent_obsk = mamujoco_cfg.get("agent_obsk")

    run_args.env = f"ma_{scenario.lower()}_{agent_conf}"
    run_args.run_subpath = os.path.join("mamujoco", run_args.env, "maac_continuous")

    env = _build_env(scenario, agent_conf, agent_obsk=agent_obsk)
    env.reset(seed=run_args.seed)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")

    agent = AttentionSACContinuous.init_from_env(
        env,
        gamma=run_args.gamma,
        tau=run_args.tau,
        pi_lr=run_args.pi_lr,
        q_lr=run_args.q_lr,
        reward_scale=run_args.reward_scale,
        pol_hidden_dim=run_args.pol_hidden_dim,
        critic_hidden_dim=run_args.critic_hidden_dim,
        attend_heads=run_args.attend_heads,
    )

    train_maac_continuous(env, agent, device, run_args)
    env.close()


if __name__ == "__main__":
    main()
