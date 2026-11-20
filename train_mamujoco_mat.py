#!/usr/bin/env python3
"""Entry point to train MAT on MAMuJoCo (Multi-Agent MuJoCo) environments.

MAT handles continuous action spaces natively via the Gaussian branch of its
decoder, so the same ``MAT`` class used for the discrete MPE2 envs is reused
here unchanged (the MADDPG/MAPPO reuse pattern).

This entry point also exists to support *reproducing the MAT paper*: the
reference implementation reports Multi-Agent MuJoCo results on scenarios such
as ``HalfCheetah 6x1``, which this file can target directly. See
``configs/mamujoco_mat.yaml`` for the paper's hyperparameters and
``RESEARCH_LOG.md`` for what does and does not transfer.

Examples
--------
    # HalfCheetah 6x1 -- the MAT paper's scenario
    python train_mamujoco_mat.py --config configs/mamujoco_mat.yaml --train

    # Ant 2x4 (matches the other algorithms' runs in this repo)
    python train_mamujoco_mat.py --config configs/mamujoco_mat.yaml \
        --scenario Ant --agent_conf 2x4 --train

    # Override hyperparameters (dot-path notation, same as main.py)
    python train_mamujoco_mat.py --config configs/mamujoco_mat.yaml \
        --override "mat.lr=0.0001" "common.episodes=5000" --train
"""

import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from gymnasium_robotics import mamujoco_v1

from config import load_config
from main import _build_mat, _make_args
from scripts.train.mat import train_mat


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
    parser = argparse.ArgumentParser(description="Train MAT on MAMuJoCo envs")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--scenario", default=None, help="MAMuJoCo scenario, e.g. HalfCheetah, Ant")
    parser.add_argument("--agent_conf", default=None, help="Joint factorization, e.g. 6x1 (paper), 2x3, 2x4")
    parser.add_argument("--override", nargs='+', action="append", default=[], help="Override config key=value (dot paths)")
    parser.add_argument("--train", action="store_true", help="Force training mode")
    args = parser.parse_args()

    overrides = [item for group in args.override for item in group]
    cfg = load_config(args.config, overrides)
    run_args = _make_args("mat", cfg, args.train)

    # scenario / agent_conf: CLI flag overrides config, which overrides built-in default.
    mamujoco_cfg = cfg.get("mamujoco") or {}
    scenario = args.scenario or mamujoco_cfg.get("scenario", "HalfCheetah")
    agent_conf = args.agent_conf or mamujoco_cfg.get("agent_conf", "6x1")
    agent_obsk = mamujoco_cfg.get("agent_obsk")

    # Label used for runs/ and checkpoints/ directories, e.g. "ma_halfcheetah_6x1".
    run_args.env = f"ma_{scenario.lower()}_{agent_conf}"
    # Layout: runs/mamujoco/<scenario>/mat/runN (and same under checkpoints/).
    run_args.run_subpath = os.path.join("mamujoco", run_args.env, "mat")

    env = build_env(scenario, agent_conf, agent_obsk=agent_obsk, render_mode=None)
    env.reset(seed=run_args.seed)

    device = torch.device("cuda" if run_args.use_gpu and torch.cuda.is_available() else "cpu")
    mat = _build_mat(env, device, run_args)

    if not run_args.train:
        raise SystemExit("Nothing to do: pass --train (or set common.train: true in the config).")

    train_mat(env, mat, run_args)
    env.close()


if __name__ == "__main__":
    main()
