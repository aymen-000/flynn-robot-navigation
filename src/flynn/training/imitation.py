from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch



from src.flynn.config import CHECKPOINT_DIR, DTYPE, LOSS_DIR
from src.flynn.core.utils import get_device
from src.flynn.training import dagger as _dagger
from src.flynn.training.dagger import (
    BATCH_SIZE, CONSECUTIVE_SEGMENTS_N, DIRECTION_THRESHOLD_DEG, EPISODES_PER_ITER, GRAD_ACCUM_STEPS, LR,
    MAX_CHUNKS_PER_CATEGORY, N_ENVS, T_BURN, T_UNROLL, TRAIN_STEPS_PER_ITER,
    BalancedDAggerBuffer, PathAnalyzer, _make_env, _make_teacher, rollout_and_collect_balanced, train_step,
)
from src.flynn.training.optim import configure_optimizer

_DAGGER_OVERRIDABLE = ("EPISODES_PER_ITER", "T_BURN", "T_UNROLL", "BATCH_SIZE", "GRAD_ACCUM_STEPS")

def run_imitation_training(
    agent, algo: str = "dagger",
    n_iters: int = 1, episodes_per_iter: int = EPISODES_PER_ITER, train_steps_per_iter: int = TRAIN_STEPS_PER_ITER,
    n_envs: int = N_ENVS, batch_size: int = BATCH_SIZE, grad_accum_steps: int = GRAD_ACCUM_STEPS,
    t_unroll: int = T_UNROLL, t_burn: int = T_BURN, lr: float = LR,
    train_rnn_weights: bool = False, train_rnn_bias: bool = False,
    log_csv_path: str = str(LOSS_DIR / "imitation_loss.csv"), checkpoint_path: str = str(CHECKPOINT_DIR / "imitation_final.pt"),
    device=None, dtype=DTYPE, seed_base: int = 0,
):
    device = device or get_device()
    assert algo in ("dagger", "bc"), f"algo must be 'dagger' or 'bc', got {algo!r}"

    def beta_sched(it):
        if algo == "bc":
            return 1.0, 0.0
        beta = max(0.0, 1.0 - 0.5 * it)
        noise = max(0.0, 0.5 - 0.2 * it)
        return beta, noise

    envs = [_make_env(render_mode=None, seed=seed_base + i) for i in range(n_envs)]
    teachers = [_make_teacher() for _ in range(n_envs)]
    buffer = BalancedDAggerBuffer(capacity_per_category=MAX_CHUNKS_PER_CATEGORY)
    analyzer = PathAnalyzer(
        direction_threshold_deg=DIRECTION_THRESHOLD_DEG, consecutive_n=CONSECUTIVE_SEGMENTS_N,
        collision_lookback=t_burn + 50, collision_recovery=t_unroll + 50,
    )
    chunk_length = t_burn + t_unroll
    stride = t_burn

    opt = configure_optimizer(
        agent, lr=lr, train_rnn_weights=train_rnn_weights, train_rnn_bias=train_rnn_bias,
        train_readout_head=True, train_input_scale=True, train_policy_head=True,
        train_value_head=False, train_wind_mlp=True, train_retina=True,
    )

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_csv_path) or ".", exist_ok=True)
    history = []

    _saved_globals = {k: getattr(_dagger, k) for k in _DAGGER_OVERRIDABLE}
    try:
        for it in range(n_iters):
            beta, beta_noise = beta_sched(it)
            print(f"\n[{algo.upper()}] iter {it + 1}/{n_iters} | beta={beta:.3f} | noise={beta_noise:.3f}")

            for _name, _value in dict(
                EPISODES_PER_ITER=episodes_per_iter, T_BURN=t_burn, T_UNROLL=t_unroll,
                BATCH_SIZE=batch_size, GRAD_ACCUM_STEPS=grad_accum_steps,
            ).items():
                setattr(_dagger, _name, _value)

            rollout_and_collect_balanced(
                envs, teachers, agent, buffer, beta, device, dtype, beta_noise,
                analyzer, chunk_length, stride,
            )
            losses = []
            for step_idx in range(train_steps_per_iter):
                try:
                    loss = train_step(agent, buffer, opt, accum_steps=grad_accum_steps)
                    losses.append(loss)
                except ValueError:
                    break
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            history.append(dict(algo=algo, iter=it + 1, mean_loss=mean_loss, batches=len(losses)))
            print(f"[{algo.upper()}] iter {it + 1} mean_loss={mean_loss:.5f} (batches={len(losses)})")
    finally:
        for _name, _value in _saved_globals.items():
            setattr(_dagger, _name, _value)
        for e in envs:
            e.close()
        torch.save(agent.state_dict(), checkpoint_path)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(log_csv_path, index=False)
    return agent, hist_df
