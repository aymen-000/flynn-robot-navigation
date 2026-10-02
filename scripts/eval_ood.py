"""Evaluate checkpoints on out-of-distribution environments (density, arena size, unseen texture)."""
import argparse

from src.flynn.evaluation.ood import OOD_SEED_BASE, plot_ood_results, run_ood_evaluation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="+", metavar="NAME=PATH",
                   help="e.g. dagger=outputs/checkpoints/dagger__finetuned.pt ")
    p.add_argument("--episodes", type=int, default=5, help="episodes per condition")
    p.add_argument("--seed-base", type=int, default=OOD_SEED_BASE)
    args = p.parse_args()

    kwargs = {}
    if args.checkpoints:
        kwargs["checkpoint_paths"] = dict(item.split("=", 1) for item in args.checkpoints)
    df = run_ood_evaluation(n_episodes=args.episodes, seed_base=args.seed_base, **kwargs)
    print(df.to_string())
    plot_ood_results(df)


if __name__ == "__main__":
    main()
