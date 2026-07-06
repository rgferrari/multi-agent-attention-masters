#!/usr/bin/env python3
"""Entry point to train MAPPO on MAMuJoCo (Multi-Agent MuJoCo) environments.

MAMuJoCo (via ``gymnasium_robotics.mamujoco_v1``) provides continuous-control
cooperative multi-agent tasks by factorizing a single MuJoCo robot into several
agents, each controlling a subset of the joints. MAPPO handles continuous action
spaces natively via its Gaussian actor, so the same ``MAPPO`` class used for the
discrete MPE2 envs is reused here unchanged (the MADDPG reuse pattern).

Examples
--------
    # HalfCheetah 2x3 (config defaults)
    python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml --train

    # Ant 2x4
    python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml \
        --scenario Ant --agent_conf 2x4 --train

    # Override hyperparameters (dot-path notation, same as main.py)
    python train_mamujoco_mappo.py --config configs/mamujoco_mappo.yaml \
        --override "mappo.actor_lr=0.0003" "common.episodes=5000" --train
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from gymnasium_robotics import mamujoco_v1

from config import load_config
from main import _build_mappo, _make_args
from scripts.train.mappo import train_mappo


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
    parser = argparse.ArgumentParser(description="Train MAPPO on MAMuJoCo envs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scenario", default=None, help="MAMuJoCo scenario, e.g. HalfCheetah, Ant")
    parser.add_argument("--agent_conf", default=None, help="Joint factorization, e.g. 2x3 (HalfCheetah), 2x4 (Ant)")
    parser.add_argument("--override", nargs='+', action="append", default=[], help="Override config key=value (dot paths)")
    parser.add_argument("--train", action="store_true", help="Force training mode")
    args = parser.parse_args()

    overrides = [item for group in args.override for item in group]
    cfg = load_config(args.config, overrides)
    run_args = _make_args("mappo", cfg, args.train)

    # scenario / agent_conf: CLI flag overrides config, which overrides built-in default.
    mamujoco_cfg = cfg.get("mamujoco") or {}
    scenario = args.scenario or mamujoco_cfg.get("scenario", "HalfCheetah")
    agent_conf = args.agent_conf or mamujoco_cfg.get("agent_conf", "2x3")
    agent_obsk = mamujoco_cfg.get("agent_obsk")

    # Label used for runs/ and checkpoints/ directories, e.g. "ma_halfcheetah_2x3".
    run_args.env = f"ma_{scenario.lower()}_{agent_conf}"
    # Layout: runs/mamujoco/<scenario>/mappo/runN (and same under checkpoints/).
    run_args.run_subpath = os.path.join("mamujoco", run_args.env, "mappo")

    env = build_env(scenario, agent_conf, agent_obsk=agent_obsk, render_mode=None)
    env.reset(seed=run_args.seed)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")
    mappo = _build_mappo(env, device, run_args)

    if not run_args.train:
        raise SystemExit("Nothing to do: pass --train (or set common.train: true in the config).")

    train_mappo(env, mappo, run_args)
    env.close()


if __name__ == "__main__":
    main()
