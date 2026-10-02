from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import cv2
import torch


import imageio

from src.flynn.agents.teacher_analytic_agent import PlannerAnalyticTeacher
from src.flynn.config import ARENA_HALF_EXTENT, DTYPE, MAX_EPISODE_STEPS, N_OBSTACLES, VIDEO_DIR
from src.flynn.core.utils import get_device, obs_to_torch
from src.flynn.envs.mujoco_two_cam_env import MuJoCoTwoCamEnv
from src.flynn.training.regimes import build_fresh_agent

def make_frame(obs, step):
    frame = np.ascontiguousarray(np.hstack([obs["cam_left"], obs["cam_right"]]))
    cv2.putText(frame, f"step {step}  policy=real_expert", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def record_expert_episode(env, teacher, video_path, fps=30):
    obs, _ = env.reset()
    teacher.reset()
    frames, steps, done, trunc = [], 0, False, False
    while not (done or trunc):
        action = teacher.act(env)
        if action is None:
            action = np.zeros(2, dtype=np.float32)
        frames.append(make_frame(obs, steps))
        obs, reward, done, trunc, info = env.step(action)
        steps += 1
    dist = info.get("dist_to_goal", float("inf"))
    success = dist < env.goal_radius
    final = frames[-1].copy()
    label, color = ("SUCCESS", (0, 200, 0)) if success else ("FAILED", (0, 0, 220))
    cv2.putText(final, label, (final.shape[1] // 2 - 60, final.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    frames.extend([final] * (fps * 2))
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return success, steps, dist


VISION_CONDITIONS = {
    "full": [True, True],
    "right_only": [False, True],
    "left_only": [True, False],
    "blind": [False, False],
}

def make_eval_frame(obs, step, vision_label):
    frame = np.ascontiguousarray(np.hstack([obs["cam_left"], obs["cam_right"]]))
    cv2.putText(frame, f"step {step}  vision={vision_label}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

@torch.no_grad()
def record_checkpoint_episode(env, agent, vision, vision_label, video_path, device, dtype, fps=30, seed=None):
    obs, _ = env.reset(seed=seed)
    agent.reset_vision_state()
    h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
    frames, steps, done, trunc = [], 0, False, False
    while not (done or trunc):
        obs_t = obs_to_torch(obs, device=device, dtype=dtype, vision=vision)
        h, action = agent.step(h, obs_t)
        frames.append(make_eval_frame(obs, steps, vision_label))
        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, done, trunc, info = env.step(action_np)
        steps += 1
    final_xy = env._base_xy()
    final_dist = float(np.linalg.norm(env._goal_xy - final_xy))
    success = final_dist < env.goal_radius
    final = frames[-1].copy()
    label, color = ("SUCCESS", (0, 200, 0)) if success else ("FAILED", (0, 0, 220))
    cv2.putText(final, label, (final.shape[1] // 2 - 60, final.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    frames.extend([final] * (fps * 2))
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return success, steps, final_dist


def make_agent_frame(obs, step, checkpoint_name):
    frame = np.ascontiguousarray(np.hstack([obs["cam_left"], obs["cam_right"]]))
    cv2.putText(frame, f"step {step}  ckpt={checkpoint_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


@torch.no_grad()
def record_agent_episode(env, agent, device, dtype, video_path, checkpoint_name, fps=30):
    obs, _ = env.reset()
    agent.reset_vision_state()
    h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
    frames, steps, done, trunc, info = [], 0, False, False, {}
    while not (done or trunc):
        obs_t = obs_to_torch(obs, device=device, dtype=dtype)
        h, action = agent.step(h, obs_t)
        action_np = action.squeeze(0).cpu().numpy()
        frames.append(make_agent_frame(obs, steps, checkpoint_name))
        obs, reward, done, trunc, info = env.step(action_np)
        steps += 1
    dist = info.get("dist_to_goal", float(np.linalg.norm(env._goal_xy - env._base_xy())))
    success = dist < env.goal_radius

    final = frames[-1].copy()
    label, color = ("SUCCESS", (0, 200, 0)) if success else ("FAILED", (0, 0, 220))
    cv2.putText(final, label, (final.shape[1] // 2 - 60, final.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    frames.extend([final] * (fps * 2))

    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    return success, steps, dist


def run_checkpoint_videos(checkpoint_path, n_episodes=4, out_dir=VIDEO_DIR / "eval"):
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found: {checkpoint_path}\n"
        f"Available checkpoints: {sorted(os.listdir('checkpoints')) if os.path.isdir('checkpoints') else '(no checkpoints/ dir yet)'}"
    )
    device = get_device()
    dtype = DTYPE
    print(f"[device] Using {device} ({dtype})")


    agent, stats = build_fresh_agent(device, dtype, train_rnn_weights=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"[ckpt] Loaded {checkpoint_path} into a fresh agent (N={stats['N']}, "
          f"Nin={stats['Nin']}, Nout={stats['Nout']})")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = Path(checkpoint_path).stem

    env = MuJoCoTwoCamEnv(
        width=256, height=256, n_obstacles=N_OBSTACLES, max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=None, arena_half_extent=ARENA_HALF_EXTENT, goal_radius=0.8,
        contact_penalty=0.5, end_on_collision=False,
    )
    results = []
    try:
        for ep in range(n_episodes):
            video_path = out_dir / f"{checkpoint_name}_ep{ep}.mp4"
            success, steps, dist = record_agent_episode(env, agent, device, dtype, video_path, checkpoint_name)
            print(f"episode {ep}: success={success} steps={steps} final_dist={dist:.2f} -> {video_path}")
            results.append(dict(episode=ep, success=success, steps=steps, final_dist=dist, video=str(video_path)))
    finally:
        env.close()

    print(f"\n[done] {sum(r['success'] for r in results)}/{len(results)} episodes succeeded. "
          f"Videos saved to {out_dir}/")
    return results



def run_expert_videos(n_episodes: int = 5, out_dir=VIDEO_DIR / "expert", fps: int = 30):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = MuJoCoTwoCamEnv(
        width=256, height=256, n_obstacles=20, max_episode_steps=1000, render_mode=None,
        arena_half_extent=7.0, goal_radius=0.8, contact_penalty=0.5, end_on_collision=False,
    )
    teacher = PlannerAnalyticTeacher(
        arena_half_extent=env.arena, cell_size=0.1, robot_radius=0.2, safety_margin=0.1,
        obstacle_box_half=(0.4, 0.4), k_nearest_obs=5, device="cpu", include_collision_flag=True,
    )
    results = []
    try:
        for ep in range(n_episodes):
            video_path = out_dir / f"real_expert_ep{ep}.mp4"
            success, steps, dist = record_expert_episode(env, teacher, video_path, fps=fps)
            print(f"episode {ep}: success={success} steps={steps} final_dist={dist:.2f} -> {video_path}")
            results.append(dict(episode=ep, success=success, steps=steps, final_dist=dist, video=str(video_path)))
    finally:
        env.close()
    return results


def run_vision_condition_videos(checkpoint_path, out_dir=VIDEO_DIR / "vision_conditions", seed: int = 0):
    device = get_device()
    dtype = DTYPE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent, _stats = build_fresh_agent(device, dtype, train_rnn_weights=False)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    eval_env = MuJoCoTwoCamEnv(
        width=128, height=128, max_episode_steps=700, n_obstacles=20,
        arena_half_extent=7.0, render_mode=None, end_on_collision=False,
        goal_bonus=0, contact_penalty=0,
    )
    results = []
    try:
        for label, vision in VISION_CONDITIONS.items():
            video_path = out_dir / f"checkpoint_{label}.mp4"
            success, steps, dist = record_checkpoint_episode(
                eval_env, agent, vision, label, video_path, device, dtype, seed=seed)
            print(f"vision={label}: success={success} steps={steps} final_dist={dist:.2f} -> {video_path}")
            results.append(dict(vision=label, success=success, steps=steps, final_dist=dist, video=str(video_path)))
    finally:
        eval_env.close()
    return results
