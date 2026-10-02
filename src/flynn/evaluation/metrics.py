from __future__ import annotations


import numpy as np
import torch



from src.flynn.config import ARENA_HALF_EXTENT, ENV_HEIGHT, ENV_WIDTH, MAX_EPISODE_STEPS, N_OBSTACLES
from src.flynn.core.utils import obs_to_torch
from src.flynn.envs.mujoco_two_cam_env import MuJoCoTwoCamEnv

@torch.no_grad()
def evaluate_agent(
    agent, device, dtype, n_episodes: int = 10, seed_base: int = 42,
    max_episode_steps: int = MAX_EPISODE_STEPS, n_obstacles: int = N_OBSTACLES,
    arena_half_extent: float = ARENA_HALF_EXTENT,
):
    agent.eval()
    env = MuJoCoTwoCamEnv(
        width=ENV_WIDTH, height=ENV_HEIGHT, max_episode_steps=max_episode_steps,
        n_obstacles=n_obstacles, arena_half_extent=arena_half_extent,
        render_mode=None, end_on_collision=False,
    )
    successes, steps_list, final_dists, collision_counts, path_effs = [], [], [], [], []
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            agent.reset_vision_state()
            h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
            start_xy = env._base_xy().copy()
            goal_xy = env._goal_xy.copy()
            straight_line = float(np.linalg.norm(goal_xy - start_xy)) + 1e-6
            path_len = 0.0
            prev_xy = start_xy.copy()
            n_collisions = 0
            steps, done, trunc = 0, False, False
            while not (done or trunc):
                obs_t = obs_to_torch(obs, device=device, dtype=dtype)
                h, action = agent.step(h, obs_t)
                action_np = action.squeeze(0).cpu().numpy()
                obs, _reward, done, trunc, _info = env.step(action_np)
                xy = env._base_xy()
                path_len += float(np.linalg.norm(xy - prev_xy))
                prev_xy = xy
                if env._has_collision()[0]:
                    n_collisions += 1
                steps += 1
            final_dist = float(np.linalg.norm(env._goal_xy - env._base_xy()))
            successes.append(final_dist < env.goal_radius)
            steps_list.append(steps)
            final_dists.append(final_dist)
            collision_counts.append(n_collisions)
            path_effs.append(straight_line / max(path_len, 1e-6))
    finally:
        env.close()

    return dict(
        success_rate=float(np.mean(successes)) if successes else float("nan"),
        mean_steps=float(np.mean(steps_list)) if steps_list else float("nan"),
        mean_final_dist=float(np.mean(final_dists)) if final_dists else float("nan"),
        mean_collisions_per_episode=float(np.mean(collision_counts)) if collision_counts else float("nan"),
        mean_path_efficiency=float(np.mean(path_effs)) if path_effs else float("nan"),
        n_episodes=n_episodes,
    )
