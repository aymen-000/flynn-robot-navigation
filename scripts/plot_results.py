import pandas as pd

from src.flynn.config import RESULTS_DIR
from src.flynn.evaluation.plots import plot_master_comparison, plot_training_curves

if __name__ == "__main__":
    csv = RESULTS_DIR / "master_comparison_results.csv"
    if csv.exists():
        print("saved", plot_master_comparison(pd.read_csv(csv)))
    else:
        print(f"{csv} not found -- run scripts/run_experiments.py first")
    for path in plot_training_curves():
        print("saved", path)
