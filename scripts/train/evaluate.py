"""Periodic deterministic evaluation during training.

Training loops log the reward of the *exploring* rollout. The MAT paper (and
the MAPPO benchmark it builds on) instead report
``eval_average_episode_rewards``: the mean return of separate episodes run with
a deterministic policy and no exploration noise. Those are different
quantities -- a stochastic policy exploring scores lower than the same policy
evaluated greedily -- so comparing a training-rollout curve against a published
eval curve understates the agent.

This module runs that separate evaluation so the comparable number exists.
Enable it with ``common.eval_interval`` (episodes; 0 disables) and
``common.eval_episodes``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def evaluate_deterministic(env, select_actions: Callable, episodes: int, max_cycles: int, seed: int) -> dict:
    """Run ``episodes`` greedy episodes and return summary statistics.

    ``select_actions`` must accept ``(obs, deterministic=True)`` and return a
    tuple whose first element is the env-ready action dict -- the signature
    shared by ``MAT``, ``MAPPO`` and ``MADDPG`` in this repo.

    Uses a seed offset well clear of the training range so evaluation episodes
    are reproducible without colliding with training resets.
    """
    returns: list[float] = []
    lengths: list[int] = []

    for i in range(max(1, episodes)):
        obs, _ = env.reset(seed=seed + 1_000_000 + i)
        agent_ids = list(env.agents)
        totals = {a: 0.0 for a in agent_ids}
        steps = 0

        for _ in range(max_cycles):
            result = select_actions(obs, deterministic=True)
            actions = result[0] if isinstance(result, tuple) else result
            obs, rewards, terminations, truncations, _ = env.step(actions)
            for a in agent_ids:
                totals[a] += float(rewards[a])
            steps += 1
            if all(terminations.values()) or all(truncations.values()):
                break

        returns.append(sum(totals.values()) / max(1, len(totals)))
        lengths.append(steps)

    arr = np.asarray(returns, dtype=np.float64)
    return {
        # Named to match the reference implementation's metric.
        'eval_average_episode_rewards': float(arr.mean()),
        'eval_episode_rewards_std': float(arr.std()),
        'eval_average_episode_length': float(np.mean(lengths)),
    }


def maybe_evaluate(env, select_actions: Callable, args, ep: int, writer) -> None:
    """Run evaluation on the configured interval and log it. Never raises."""
    interval = int(getattr(args, 'eval_interval', 0) or 0)
    if interval <= 0 or (ep + 1) % interval != 0:
        return

    try:
        stats = evaluate_deterministic(
            env,
            select_actions,
            episodes=int(getattr(args, 'eval_episodes', 5) or 5),
            max_cycles=args.max_cycles,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must not fail training
        print(f'[eval] skipped ({type(exc).__name__}): {exc}')
        return

    for key, value in stats.items():
        writer.add_scalar(f'eval/{key}', value, ep)
