from __future__ import annotations

import os
from typing import Dict, List, Any

import pandas as pd
import torch



from src.flynn.config import CHECKPOINT_DIR, DTYPE, LOSS_DIR, RESULTS_DIR
from src.flynn.core.utils import get_device
from src.flynn.evaluation.metrics import evaluate_agent
from src.flynn.training.imitation import run_imitation_training
from src.flynn.training.regimes import TRAINING_REGIMES, build_fresh_agent

MASTER_RESULTS: List[Dict[str, Any]] = []


def run_one_condition(
    algo: str, regime_key: str,
    n_iters: int = 1, episodes_per_iter: int = 200, train_steps_per_iter: int = 200,
     eval_episodes: int = 20,
    device=None, dtype=DTYPE,
):
    device = device or get_device()
    regime = TRAINING_REGIMES[regime_key]
    agent, stats = build_fresh_agent(
        device, dtype,
        train_rnn_weights=regime.get("train_rnn_weights", False) if regime["train"] else False,
        train_readout_head=True,
    )

    tag = f"{algo}__{regime_key}"
    if regime["train"]:
        if algo in ("dagger", "bc"):
            agent, _hist = run_imitation_training(
                agent, algo=algo, n_iters=n_iters, episodes_per_iter=episodes_per_iter,
                train_steps_per_iter=train_steps_per_iter,
                train_rnn_weights=regime.get("train_rnn_weights", False),
                train_rnn_bias=regime.get("train_rnn_bias", False),
                log_csv_path=str(LOSS_DIR / f"{tag}_loss.csv"), checkpoint_path=str(CHECKPOINT_DIR / f"{tag}.pt"),
                device=device, dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown algo {algo!r}")
    else:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(agent.state_dict(), CHECKPOINT_DIR / f"{tag}.pt")

    metrics = evaluate_agent(agent, device, dtype, n_episodes=eval_episodes)
    row = dict(algo=algo, regime=regime_key, **stats, **metrics)
    MASTER_RESULTS.append(row)
    print(f"[done] {tag}: {metrics}")
    return agent, row


def run_grid(
    algos=("dagger", "bc"), regimes=("finetuned",), include_untrained: bool = True,
    results_csv=None, **condition_kwargs,
):

    os.makedirs(LOSS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


    if include_untrained:
        try:
            run_one_condition(algo="none", regime_key="before_training", **condition_kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[FAILED] before_training: {e}")

    for regime in regimes:
        for algo in algos:
            try:
                run_one_condition(algo, regime, **condition_kwargs)
            except Exception as e:  # noqa: BLE001
                print(f"[FAILED] {algo}/{regime}: {e}")

    results_df = pd.DataFrame(MASTER_RESULTS)
    results_df.to_csv(results_csv or RESULTS_DIR / "master_comparison_results.csv", index=False)
    return results_df
