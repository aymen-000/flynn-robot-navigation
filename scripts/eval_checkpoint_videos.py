import argparse

from src.flynn.config import VIDEO_DIR
from src.flynn.evaluation.videos import run_checkpoint_videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a .pt checkpoint, e.g. outputs/checkpoints/bc__finetuned.pt")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--out-dir", default=str(VIDEO_DIR / "eval"))
    args = p.parse_args()
    run_checkpoint_videos(args.checkpoint, n_episodes=args.episodes, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
