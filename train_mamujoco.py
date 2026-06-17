#!/usr/bin/env python3
"""Entry point to train MADDPG on MAMuJoCo (Multi-Agent MuJoCo) environments.

MAMuJoCo (via ``gymnasium_robotics.mamujoco_v1``) provides continuous-control
cooperative multi-agent tasks by factorizing a single MuJoCo robot into several
agents, each controlling a subset of the joints. Two scenarios are wired up here:

  * Multi-Agent HalfCheetah  (default factorization ``2x3``: 2 agents x 3 joints)
  * Multi-Agent Ant          (default factorization ``2x4``: 2 agents x 4 joints)

Only MADDPG is supported — it handles continuous action spaces. MAAC is
discrete-action only, so it cannot be used here.

Examples
--------
    # HalfCheetah 2x3 (config defaults)
    python train_mamujoco.py --config configs/mamujoco_maddpg.yaml --train

    # Ant 2x4
    python train_mamujoco.py --config configs/mamujoco_maddpg.yaml \
        --scenario Ant --agent_conf 2x4 --train

    # Override hyperparameters (dot-path notation, same as main.py)
    python train_mamujoco.py --config configs/mamujoco_maddpg.yaml \
        --override "maddpg.actor_lr=0.0005" "common.episodes=5000" --train
"""

import argparse
import os

import torch
from gymnasium_robotics import mamujoco_v1

from config import load_config
from main import _build_maddpg, _make_args
from scripts.train.maddpg import train_maddpg


def build_env(scenario: str, agent_conf: str, agent_obsk=None, render_mode=None):
    """Create a MAMuJoCo PettingZoo-style parallel env.

    ``agent_obsk`` controls how many nearest bodies each agent observes; ``None``
    keeps the gymnasium-robotics default.
    """
    kwargs = {"scenario": scenario, "agent_conf": agent_conf, "render_mode": render_mode}
    if agent_obsk is not None:
        kwargs["agent_obsk"] = agent_obsk
    return mamujoco_v1.parallel_env(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MADDPG on MAMuJoCo envs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scenario", default=None, help="MAMuJoCo scenario, e.g. HalfCheetah, Ant")
    parser.add_argument("--agent_conf", default=None, help="Joint factorization, e.g. 2x3 (HalfCheetah), 2x4 (Ant)")
    parser.add_argument("--override", action="append", default=[], help="Override config key=value (dot paths)")
    parser.add_argument("--train", action="store_true", help="Force training mode")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    run_args = _make_args("maddpg", cfg, args.train)

    # scenario / agent_conf: CLI flag overrides config, which overrides built-in default.
    mamujoco_cfg = cfg.get("mamujoco") or {}
    scenario = args.scenario or mamujoco_cfg.get("scenario", "HalfCheetah")
    agent_conf = args.agent_conf or mamujoco_cfg.get("agent_conf", "2x3")
    agent_obsk = mamujoco_cfg.get("agent_obsk")

    # Label used for runs/ and checkpoints/ directories, e.g. "ma_halfcheetah_2x3".
    run_args.env = f"ma_{scenario.lower()}_{agent_conf}"
    # Layout: runs/mamujoco/<scenario>/maddpg/runN (and same under checkpoints/).
    run_args.run_subpath = os.path.join("mamujoco", run_args.env, "maddpg")

    env = build_env(scenario, agent_conf, agent_obsk=agent_obsk, render_mode=None)
    env.reset(seed=run_args.seed)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")
    maddpg = _build_maddpg(env, device, run_args)

    if not run_args.train:
        raise SystemExit("Nothing to do: pass --train (or set common.train: true in the config).")

    train_maddpg(env, maddpg, run_args)
    env.close()


if __name__ == "__main__":
    main()
