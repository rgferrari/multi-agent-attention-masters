#!/usr/bin/env python3
"""Run an MPE2 environment with human rendering and GPU-based action sampling."""

import argparse
import importlib
import os
from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _warn_if_no_display(render_mode: str) -> None:
    if render_mode != "human":
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    print(
        "Warning: render_mode=human but no DISPLAY/WAYLAND detected. "
        "Use X/WSLg or run with xvfb if needed."
    )


def _sample_action(space: gym.Space, device: torch.device) -> Any:
    if isinstance(space, gym.spaces.Box):
        low = torch.as_tensor(space.low, device=device, dtype=torch.float32)
        high = torch.as_tensor(space.high, device=device, dtype=torch.float32)
        action = low + (high - low) * torch.rand(space.shape, device=device)
        return action.cpu().numpy()
    if isinstance(space, gym.spaces.Discrete):
        return int(torch.randint(space.n, (1,), device=device).item())
    if isinstance(space, gym.spaces.MultiDiscrete):
        nvec = torch.as_tensor(space.nvec, device=device)
        return torch.randint(nvec, (len(nvec),), device=device).cpu().numpy()
    if isinstance(space, gym.spaces.MultiBinary):
        return torch.randint(2, space.n, device=device, dtype=torch.int8).cpu().numpy()

    # Fallback to environment sampling on CPU.
    return space.sample()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="simple_spread_v3", help="MPE2 env module name")
    parser.add_argument("--steps", type=int, default=200, help="Max steps")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--use_gpu", action="store_true", help="Use CUDA for sampling")
    parser.add_argument("--render_mode", default="human", help="Render mode")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env_module = importlib.import_module(f"mpe2.{args.env}")
    env = env_module.parallel_env(render_mode=args.render_mode, max_cycles=args.steps)

    _warn_if_no_display(args.render_mode)

    observations = env.reset(seed=args.seed)
    done = False
    step = 0
    while not done and step < args.steps:
        actions = {
            agent: _sample_action(env.action_space(agent), device)
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = all(terminations.values()) or all(truncations.values())
        step += 1

    env.close()
    _ = observations, rewards, infos  # keep lints quiet if unused


if __name__ == "__main__":
    main()
