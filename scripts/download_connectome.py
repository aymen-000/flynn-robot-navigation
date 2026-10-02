import argparse

from src.flynn.data import ensure_edge_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if the file already exists")
    args = p.parse_args()
    ensure_edge_list(force=args.force)


if __name__ == "__main__":
    main()
