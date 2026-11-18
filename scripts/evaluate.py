"""
    Score a policy over many games

    python scripts/evaluate.py

    Arguments:
    --checkpoint  # evaluate a trained agent, omit to evaluate a random agent
    --games       # number of games to play (default 100)
    --seed        # RNG seed for reproducibility (default 1000)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.agent import greedy_action, load_checkpoint
from src.config import Config
from src.env import N_ACTIONS, SnakeEnv
from src.evaluation import random_policy, run_games


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="omit to evaluate a uniformly random agent")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    cfg = Config()
    size = cfg.grid_size
    if args.checkpoint:
        net, meta = load_checkpoint(args.checkpoint)
        size = meta.get("grid_size", size)
        policy, name = (lambda obs: greedy_action(net, obs)), args.checkpoint.name
    else:
        policy, name = random_policy(N_ACTIONS, args.seed), "random agent"

    env = SnakeEnv(size, cfg.max_idle_steps, seed=args.seed)
    scores, steps = run_games(env, policy, args.games)
    print(f"{name}: {args.games} games on a {size}x{size} grid")
    print(f"  score  mean {np.mean(scores):.2f} | median {np.median(scores):.0f} | max {max(scores)}")
    print(f"  steps  mean {np.mean(steps):.1f}")


if __name__ == "__main__":
    main()
