import argparse

from src.flynn.evaluation.plots import plot_master_comparison, plot_training_curves
from src.flynn.training.experiments import run_grid
from src.flynn.training.regimes import TRAINING_REGIMES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=["dagger", "bc"], choices=["dagger", "bc"])
    p.add_argument("--regimes", nargs="+", default=["finetuned"],
                   choices=[r for r in TRAINING_REGIMES if r != "before_training"])
    p.add_argument("--no-untrained-baseline", action="store_true")
    p.add_argument("--n-iters", type=int, default=1, help="DAgger/BC iterations")
    p.add_argument("--episodes-per-iter", type=int, default=200)
    p.add_argument("--train-steps-per-iter", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=20)
    args = p.parse_args()

    results = run_grid(
        algos=args.algos, regimes=args.regimes, include_untrained=not args.no_untrained_baseline,
        n_iters=args.n_iters, episodes_per_iter=args.episodes_per_iter,
        train_steps_per_iter=args.train_steps_per_iter, eval_episodes=args.eval_episodes,
    )
    print(results.to_string())
    plot_master_comparison(results)
    plot_training_curves()


if __name__ == "__main__":
    main()
