
import argparse
import os

from src.flynn.config import LOSS_DIR
from src.flynn.training.dagger import main

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.makedirs(LOSS_DIR, exist_ok=True)
    main()
