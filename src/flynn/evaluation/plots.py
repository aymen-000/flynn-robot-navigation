from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt



from src.flynn.config import LOSS_DIR, RESULTS_DIR

import glob


def plot_master_comparison(results_df, save_path=None):
    """Grouped bars of success rate / collisions / path efficiency per training regime, one bar per algorithm."""
    if not len(results_df):
        print("No results yet -- run the experiment grid first.")
        return None
    metrics_to_plot = ["success_rate", "mean_collisions_per_episode", "mean_path_efficiency"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
    for ax, metric in zip(axes, metrics_to_plot):
        pivot = results_df.pivot_table(index="regime", columns="algo", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20, labelsize=9)
    plt.tight_layout()
    save_path = Path(save_path or RESULTS_DIR / "master_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_training_curves(loss_dir=LOSS_DIR, out_dir=RESULTS_DIR):
    """DAgger-vs-BC training loss """
    loss_dir, out_dir = Path(loss_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    il_files = sorted(glob.glob(str(loss_dir / "*__dagger__*_loss.csv")) + glob.glob(str(loss_dir / "*__bc__*_loss.csv"))
                      + glob.glob(str(loss_dir / "dagger__*_loss.csv")) + glob.glob(str(loss_dir / "bc__*_loss.csv")))

    def _plot(files, y_col, xlabel, ylabel, title, out_name):
        fig = plt.figure(figsize=(8, 5))
        for f in dict.fromkeys(files):
            try:
                d = pd.read_csv(f)
                if y_col in d.columns and len(d):
                    plt.plot(d["iter"], d[y_col], marker="o", label=os.path.basename(f).replace("_loss.csv", ""))
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {f}: {e}")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(out_dir / out_name)
        plt.close(fig)
        written.append(out_dir / out_name)

    if il_files:
        _plot(il_files, "mean_loss", "Training iteration", "Mean training loss (imitation)",
              "DAgger vs. Behavior Cloning (IL) -- training loss", "dagger_vs_bc_loss.png")
    else:
        print("No DAgger/BC loss CSVs found yet -- run the experiment grid first.")
    return written
