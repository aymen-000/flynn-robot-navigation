from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch



from src.flynn.config import (
    ARENA_HALF_EXTENT, CHECKPOINT_DIR, DTYPE, ENV_HEIGHT, ENV_WIDTH, MAX_EPISODE_STEPS, N_OBSTACLES, RESULTS_DIR,
)
from src.flynn.core.utils import get_device, obs_to_torch
from src.flynn.envs.mujoco_two_cam_env import MuJoCoTwoCamEnv
from src.flynn.training.regimes import build_fresh_agent

CHECKPOINT_PATHS = {
    "dagger_finetuned": str(CHECKPOINT_DIR / "dagger__finetuned.pt"),
    "bc_finetuned": str(CHECKPOINT_DIR / "bc__finetuned.pt"),
}
N_EPISODES_PER_CONDITION = 5
OOD_SEED_BASE = 90000   # far from any seed used in training/eval -> novel obstacle layouts

OOD_CONDITIONS = {
    "in_distribution":   dict(n_obstacles=N_OBSTACLES, arena_half_extent=ARENA_HALF_EXTENT, texture_mode="checker"),
    "sparse_obstacles":  dict(n_obstacles=5,            arena_half_extent=ARENA_HALF_EXTENT, texture_mode="checker"),
    "dense_small_arena": dict(n_obstacles=12,           arena_half_extent=4.5,               texture_mode="checker"),
    "large_open_arena":  dict(n_obstacles=N_OBSTACLES,  arena_half_extent=12.0,              texture_mode="checker"),
    "unseen_texture":    dict(n_obstacles=N_OBSTACLES,  arena_half_extent=ARENA_HALF_EXTENT, texture_mode="realistic"),
}


def load_checkpoint_into_agent(agent, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"    [info] ignored {len(unexpected)} unused checkpoint key(s) not used at eval time: {unexpected}")
    real_missing = [k for k in missing if not k.startswith("value_head.")]
    if real_missing:
        print(f"    [warn] missing key(s) when loading {ckpt_path}: {real_missing}")
    return agent


def _safe_make_env(env_kwargs, seed, max_retries=5):
    last_err = None
    for attempt in range(max_retries):
        try:
            return MuJoCoTwoCamEnv(
                width=ENV_WIDTH, height=ENV_HEIGHT, max_episode_steps=MAX_EPISODE_STEPS,
                render_mode=None, end_on_collision=False, seed=seed + attempt * 10_000, **env_kwargs,
            )
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(
        f"Could not construct env with {env_kwargs} after {max_retries} seed retries "
        f"(last error: {last_err}). Lower n_obstacles or raise arena_half_extent."
    )


def _safe_reset(env, seed, max_retries=5):
    last_err = None
    for attempt in range(max_retries):
        try:
            return env.reset(seed=seed + attempt * 10_000)
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"env.reset failed after {max_retries} seed retries (last error: {last_err}).")


@torch.no_grad()
def evaluate_on_condition(agent, device, dtype, env_kwargs, n_episodes, seed_base):
    try:
        env = _safe_make_env(env_kwargs, seed=seed_base)
    except RuntimeError as e:
        print(f"    [SKIPPED] {e}")
        return None

    successes, steps_list, final_dists, collisions_list, path_effs = [], [], [], [], []
    try:
        for ep in range(n_episodes):
            try:
                obs, _ = _safe_reset(env, seed=seed_base + ep)
            except RuntimeError as e:
                print(f"    [episode {ep} SKIPPED] {e}")
                continue
            agent.reset_vision_state()
            h = torch.zeros(1, agent.cell.N, device=device, dtype=dtype)
            start_xy = env._base_xy().copy()
            goal_xy = env._goal_xy.copy()
            straight_line = float(np.linalg.norm(goal_xy - start_xy)) + 1e-6
            path_len, prev_xy, n_collisions, steps, done, trunc = 0.0, start_xy.copy(), 0, 0, False, False
            while not (done or trunc):
                obs_t = obs_to_torch(obs, device=device, dtype=dtype)
                h, action = agent.step(h, obs_t)
                obs, _reward, done, trunc, _info = env.step(action.squeeze(0).cpu().numpy())
                xy = env._base_xy()
                path_len += float(np.linalg.norm(xy - prev_xy)); prev_xy = xy
                if env._has_collision()[0]:
                    n_collisions += 1
                steps += 1
            final_dist = float(np.linalg.norm(env._goal_xy - env._base_xy()))
            successes.append(final_dist < env.goal_radius)
            steps_list.append(steps); final_dists.append(final_dist)
            collisions_list.append(n_collisions)
            path_effs.append(straight_line / max(path_len, 1e-6))
    finally:
        env.close()

    if not successes:
        print("    [SKIPPED] every episode failed to reset.")
        return None

    return dict(
        success_rate=float(np.mean(successes)),
        mean_steps=float(np.mean(steps_list)),
        mean_final_dist=float(np.mean(final_dists)),
        mean_collisions_per_episode=float(np.mean(collisions_list)),
        mean_path_efficiency=float(np.mean(path_effs)),
        n_episodes=len(successes),
    )


def run_ood_evaluation(checkpoint_paths=CHECKPOINT_PATHS, conditions=OOD_CONDITIONS,
                        n_episodes=N_EPISODES_PER_CONDITION, seed_base=OOD_SEED_BASE):
    device = get_device()
    dtype = DTYPE
    rows = []
    for ckpt_name, ckpt_path in checkpoint_paths.items():
        if not os.path.exists(ckpt_path):
            print(f"[skip] {ckpt_name}: checkpoint not found at {ckpt_path}")
            continue
        agent, stats = build_fresh_agent(device, dtype, train_rnn_weights=False)
        load_checkpoint_into_agent(agent, ckpt_path, device)
        agent.eval()
        print(f"\n=== {ckpt_name} ({ckpt_path}) ===")
        for cond_name, env_kwargs in conditions.items():
            metrics = evaluate_on_condition(agent, device, dtype, env_kwargs, n_episodes, seed_base)
            if metrics is None:
                rows.append(dict(checkpoint=ckpt_name, condition=cond_name, success_rate=np.nan,
                                  mean_steps=np.nan, mean_final_dist=np.nan,
                                  mean_collisions_per_episode=np.nan, mean_path_efficiency=np.nan, n_episodes=0))
                continue
            print(f"  [{cond_name}] success_rate={metrics['success_rate']:.2f}  "
                  f"collisions/ep={metrics['mean_collisions_per_episode']:.2f}  "
                  f"path_eff={metrics['mean_path_efficiency']:.2f}")
            rows.append(dict(checkpoint=ckpt_name, condition=cond_name, **metrics))
        del agent

    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "ood_generalization_results.csv"), index=False)
    return results_df


def plot_ood_results(results_df, save_path=None):
    """Print the generalization gap and plot in-distribution vs. OOD success rate per checkpoint."""
    if not len(results_df):
        return None
    pivot = results_df.pivot(index="checkpoint", columns="condition", values="success_rate")
    if "in_distribution" in pivot.columns:
        gap = pivot.drop(columns=["in_distribution"]).sub(pivot["in_distribution"], axis=0)
        print("\nGeneralization gap (OOD success_rate - in_distribution success_rate):")
        print(gap.round(3))

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("success_rate")
    ax.set_title("In-distribution vs. OOD success rate, per checkpoint")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    save_path = save_path or os.path.join(RESULTS_DIR, "ood_generalization.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path
