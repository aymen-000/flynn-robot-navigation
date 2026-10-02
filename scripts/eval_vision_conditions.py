"""Record one episode per vision condition (full / right_only / left_only / blind) for a checkpoint."""
import argparse

from src.flynn.config import VIDEO_DIR
from src.flynn.evaluation.videos import run_vision_condition_videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a .pt checkpoint, e.g. outputs/checkpoints/dagger__finetuned.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(VIDEO_DIR / "vision_conditions"))
    args = p.parse_args()
    run_vision_condition_videos(args.checkpoint, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
