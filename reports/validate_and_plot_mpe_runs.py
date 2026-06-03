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


def _speaker_listener_annotation(env) -> str | None:
    """Get the name of the color the speaker is sending to the listener.
    Landmarks: 0=RED (A), 1=GREEN (B), 2=BLUE (C)
    """
    world = env.unwrapped.world
    speaker = None
    for agent in world.agents:
        if hasattr(agent, 'listener') and getattr(agent, 'listener', False) is False:
            speaker = agent
            break
    if speaker is None:
        return None
    goal = getattr(speaker, 'goal_b', None)
    if goal is None:
        return None
    
    goal_color = np.asarray(goal.color, dtype=float)
    landmarks = world.landmarks
    
    # Map landmark index to color name
    color_names = ['RED', 'GREEN', 'BLUE']
    for i, landmark in enumerate(landmarks):
        landmark_color = np.asarray(landmark.color, dtype=float)
        if np.allclose(landmark_color, goal_color):
            if i < len(color_names):
                return f'sending {color_names[i]}'
    
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
    state = torch.load(ckpt, map_location='cpu')
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


def _label_row(ax, text: str) -> None:
    ax.text(
        0.5,
        1.02,
        text,
        transform=ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
    )


def build_validation_grid() -> Path:
    """Build separate validation grids for MADDPG and MAAC.
    Each grid: rows=environments, columns=steps.
    Environment names written vertically on the left.
    """
    out_paths = []
    
    for agent_name in AGENTS:
        fig, axes = plt.subplots(len(ENVS), len(CAPTURE_STEPS), figsize=(12, 8))
        if len(ENVS) == 1:
            axes = np.array([axes])
        if len(CAPTURE_STEPS) == 1:
            axes = np.array([[axes]])

        fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.05, wspace=0.08, hspace=0.12)

        for r, env_name in enumerate(ENVS):
            for c, step in enumerate(CAPTURE_STEPS):
                ax = axes[r, c]
                frames, speaker_note = _capture_episode_frames(agent_name, env_name)
                ax.set_axis_off()
                
                # Get the frame for this step
                frame = frames[c]
                
                # Create a nested gridspec for this cell to add framing
                inner = ax.get_subplotspec().subgridspec(1, 1, wspace=0.0, hspace=0.0)
                subax = fig.add_subplot(inner[0, 0])
                subax.imshow(frame)
                subax.set_xticks([])
                subax.set_yticks([])
                subax.set_frame_on(True)
                for spine in subax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(1.0)
                    spine.set_edgecolor('black')
                
                # Add step label above image
                subax.set_title(f'step {step}', fontsize=8, pad=3)
                
                # Add vertical environment label on the Y-axis for leftmost images
                if c == 0:
                    subax.set_ylabel(env_name, fontsize=10, fontweight='bold')
                
                # Add "sending <color>" at bottom right for speaker-listener env
                if env_name == 'simple_speaker_listener_v4' and speaker_note is not None:
                    subax.text(
                        0.98,
                        0.02,
                        speaker_note,
                        transform=subax.transAxes,
                        ha='right',
                        va='bottom',
                        fontsize=9,
                        color='darkred',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                    )

        fig.tight_layout()
        out_pdf = OUT_DIR / f'validation_grid_{agent_name}.pdf'
        out_png = OUT_DIR / f'validation_grid_{agent_name}.png'
        fig.savefig(out_pdf, bbox_inches='tight')
        fig.savefig(out_png, bbox_inches='tight')
        out_paths.append(out_pdf)
        plt.close(fig)

    return out_paths[0]


if __name__ == '__main__':
    print(build_validation_grid())
