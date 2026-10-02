"""Record headless videos of the analytic VFH*+A* expert."""
import argparse

from src.flynn.config import VIDEO_DIR
from src.flynn.evaluation.videos import run_expert_videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out-dir", default=str(VIDEO_DIR / "expert"))
    args = p.parse_args()
    run_expert_videos(n_episodes=args.episodes, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
