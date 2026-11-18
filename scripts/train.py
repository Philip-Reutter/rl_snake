"""
    Train a DQN agent on Snake

    python scripts/train.py  # defaults from src/config.py

    Arguments:
    --episodes  # number of training episodes (default config.episodes)
    --seed      # RNG seed for reproducibility (default config.seed)
    --device    # "cpu" or "cuda" (default "cpu")

    Writes:
        results/training_log.csv, results/eval_log.csv, results/learning_curves.png
        checkpoints/milestones/epNNNNN.pt  # greedy snapshots (replay with scripts/watch.py)
        checkpoints/best.pt, checkpoints/final.pt
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()

    cfg = Config()
    overrides = {k: v for k, v in (("episodes", args.episodes), ("seed", args.seed)) if v is not None}
    train(replace(cfg, **overrides), device=args.device)


if __name__ == "__main__":
    main()
