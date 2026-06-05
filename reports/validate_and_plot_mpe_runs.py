from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.maac.attention_sac import AttentionSAC
from agents.maddpg_torch import MADDPG
from agents.maddpg_torch.maddpg import MADDPGConfig

CHECKPOINTS = ROOT / 'checkpoints'
OUT_DIR = ROOT / 'reports' / 'validation_grid'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENVS = [
    'simple_speaker_listener_v4',
    'simple_adversary_v3',
]
AGENTS = ['maac', 'maddpg']
CAPTURE_STEPS = [0, 5, 25]


def _load_env(env_name: str, max_cycles: int = 25):
    mod = importlib.import_module(f'mpe2.{env_name}')
    return mod.parallel_env(render_mode='rgb_array', max_cycles=max_cycles)


def _checkpoint_path(agent: str, env: str) -> Path:
    base = CHECKPOINTS / agent / env / 'run1'
    best = base / f'{agent}_best.pt'
    final = base / f'{agent}_final.pt'
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(f'No checkpoint found for {agent} on {env}: expected {best} or {final}')


def _render_frame(env) -> np.ndarray:
    frame = env.render()
    if not isinstance(frame, np.ndarray):
        raise RuntimeError(f'Expected RGB array from env.render(), got {type(frame)}')
    return frame.copy()


_COLOR_HEX = {'RED': '#CC0000', 'GREEN': '#007700', 'BLUE': '#0000CC'}


def _speaker_listener_annotation(env) -> tuple[str, str] | None:
    """Return (label, hex_color) for the color the speaker is signalling.

    Landmark colors: A=RED (idx 0), B=GREEN (idx 1), C=BLUE (idx 2).
    The dominant RGB channel of the goal landmark's color gives the color name.
    The speaker is the non-silent agent (silent=False) that has goal_b set.
    """
    world = env.unwrapped.world
    for agent in world.agents:
        if not agent.silent:
            goal = getattr(agent, 'goal_b', None)
            if goal is None:
                return None
            color = np.asarray(goal.color, dtype=float)
            name = ['RED', 'GREEN', 'BLUE'][int(np.argmax(color))]
            return f'→ {name}', _COLOR_HEX[name]
    return None


def _maac_action(agent: AttentionSAC, obs: Dict[str, np.ndarray], device: torch.device, agent_ids: List[str]) -> Dict[str, int]:
    agent.prep_rollouts('gpu' if device.type == 'cuda' else 'cpu')
    obs_list = [torch.as_tensor(obs[a], dtype=torch.float32, device=device).unsqueeze(0) for a in agent_ids]
    onehots = agent.step(obs_list, explore=False)
    return {agent_id: int(onehot.argmax(dim=1).item()) for agent_id, onehot in zip(agent_ids, onehots)}


def _maddpg_action(maddpg: MADDPG, obs: Dict[str, np.ndarray], explore: bool = False) -> Dict[str, int | np.ndarray]:
    actions, _ = maddpg.select_actions(obs, explore=explore)
    return actions


def _load_maac(env_name: str) -> AttentionSAC:
    ckpt = _checkpoint_path('maac', env_name)
    model = AttentionSAC.init_from_save(str(ckpt), load_critic=True)
    return model


def _load_maddpg(env_name: str, env) -> MADDPG:
    ckpt = _checkpoint_path('maddpg', env_name)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    agent_ids = list(env.agents)
    obs_spaces = [env.observation_space(a) for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]
    cfg_dict = state.get('config', {}) or {}
    config = MADDPGConfig(**cfg_dict)
    model = MADDPG(agent_ids, obs_spaces, act_spaces, device=torch.device('cpu'), config=config)
    model.load_state_dict(state)
    return model


def _capture_episode_frames(agent_name: str, env_name: str) -> Tuple[List[np.ndarray], str | None]:
    env = _load_env(env_name, max_cycles=max(CAPTURE_STEPS))
    obs, _ = env.reset(seed=0)
    agent_ids = list(env.agents)
    device = torch.device('cpu')

    if agent_name == 'maac':
        model = _load_maac(env_name)
        model.prep_rollouts('cpu')
        act_fn = lambda current_obs: _maac_action(model, current_obs, device, agent_ids)
    elif agent_name == 'maddpg':
        model = _load_maddpg(env_name, env)
        model.device = device
        act_fn = lambda current_obs: _maddpg_action(model, current_obs, explore=False)
    else:
        raise ValueError(agent_name)

    frames: Dict[int, np.ndarray] = {}
    frames[0] = _render_frame(env)
    current_step = 0
    while current_step < max(CAPTURE_STEPS):
        actions = act_fn(obs)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        current_step += 1
        if current_step in CAPTURE_STEPS:
            frames[current_step] = _render_frame(env)
        if all(terminations.values()) or all(truncations.values()):
            break

    speaker_note = _speaker_listener_annotation(env) if env_name == 'simple_speaker_listener_v4' else None
    env.close()
    ordered_frames = [frames.get(step, next(iter(frames.values()))) for step in CAPTURE_STEPS]
    return ordered_frames, speaker_note


def build_validation_grid() -> Path:
    """Single grid: rows=environments, cols=steps×agents (MAAC|MADDPG per step)."""
    # Pre-capture so each (agent, env) episode runs exactly once
    all_frames: Dict[str, Dict[str, Tuple[List[np.ndarray], str | None]]] = {}
    for agent_name in AGENTS:
        all_frames[agent_name] = {}
        for env_name in ENVS:
            all_frames[agent_name][env_name] = _capture_episode_frames(agent_name, env_name)

    n_envs = len(ENVS)
    n_steps = len(CAPTURE_STEPS)

    # GridSpec: n_steps cols per agent, with a narrow spacer between the two agent groups
    SPACER = 0.12  # relative width of the gap column between model groups
    width_ratios = [1] * n_steps + [SPACER] + [1] * n_steps
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(
        n_envs, n_steps * 2 + 1,
        width_ratios=width_ratios,
        left=0.10, right=0.99, top=0.94, bottom=0.01,
        wspace=0.02, hspace=0.03,
    )

    # col mapping: agent 0 → cols 0..n_steps-1, agent 1 → cols n_steps+1..2*n_steps
    def grid_col(a_idx: int, c_step: int) -> int:
        return a_idx * (n_steps + 1) + c_step

    for r, env_name in enumerate(ENVS):
        for a_idx, agent_name in enumerate(AGENTS):
            for c_step in range(n_steps):
                col = grid_col(a_idx, c_step)
                ax = fig.add_subplot(gs[r, col])
                frames, speaker_note = all_frames[agent_name][env_name]

                ax.imshow(frames[c_step])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.8)
                    spine.set_edgecolor('black')

                # Agent label above first step of each group, top row only
                if r == 0 and c_step == n_steps // 2:
                    ax.set_title(agent_name.upper(), fontsize=10, fontweight='bold', pad=5)

                # Env label on the very first column only
                if a_idx == 0 and c_step == 0:
                    ax.set_ylabel(
                        env_name.replace('_v4', '').replace('_v3', '').replace('_', ' '),
                        fontsize=9, fontweight='bold', labelpad=4,
                    )

                if env_name == 'simple_speaker_listener_v4' and speaker_note is not None:
                    label, hex_color = speaker_note
                    ax.text(
                        0.97, 0.03, label,
                        transform=ax.transAxes,
                        ha='right', va='bottom',
                        fontsize=9, color=hex_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
                    )

    out_pdf = OUT_DIR / 'validation_grid.pdf'
    out_png = OUT_DIR / 'validation_grid.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    return out_pdf


if __name__ == '__main__':
    print(build_validation_grid())
