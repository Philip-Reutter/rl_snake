"""
    Re-draw results/learning_curves.png from CSV logs

    python scripts/plot.py

    Arguments:
    --window  # moving-average window in episodes (default 50)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RESULTS_DIR
from src.plotting import plot_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=50, help="moving-average window in episodes")
    print(plot_curves(RESULTS_DIR, parser.parse_args().window))
