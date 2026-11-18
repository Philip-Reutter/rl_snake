"""
    Plays games and reports score, length and how each game ended.

    python scripts/diagnose.py

    Arguments:
    --checkpoint  # which model to load
    --games       # how many games to play (default 100)
    --epsilons    # probability of taking a random action instead of the greedy action (default 0.0 and 0.02)
    --seed        # random seed for reproducibility (default 2000)
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.agent import greedy_action, load_checkpoint
from src.config import CHECKPOINT_DIR, Config
from src.env import DIRS, N_ACTIONS, OBS_DIM, SnakeEnv


def how_it_ended(env, truncated):
    if truncated:
        return "timeout (no food for max_idle_steps)"
    if env.food is None:
        return "won (board full)"
    # On a fatal move snake does not move, but direction is updated, so the cell it tried to enter is head + direction
    hx, hy = env.snake[0]
    dx, dy = DIRS[env.direction]
    x, y = hx + dx, hy + dy
    return "wall" if not (0 <= x < env.size and 0 <= y < env.size) else "own body"


def play_games(net, size, max_idle_steps, games, epsilon, seed):
    env = SnakeEnv(size, max_idle_steps, seed=seed)
    rng = np.random.default_rng(seed)
    scores, steps, endings = [], [], Counter()
    for _ in range(games):
        obs = env.reset()
        while True:
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = greedy_action(net, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        scores.append(env.score)
        steps.append(env.steps)
        endings[how_it_ended(env, truncated)] += 1
    return np.array(scores), np.array(steps), endings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.02])
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()

    net, meta = load_checkpoint(args.checkpoint)
    size = meta.get("grid_size", Config().grid_size)

    print(f"{args.checkpoint.name}: {args.games} games on a {size}x{size} grid")
    for eps in args.epsilons:
        scores, steps, endings = play_games(net, size, Config().max_idle_steps, args.games, eps, args.seed)
        stderr = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else float("nan")
        print(f"\nepsilon = {eps}")
        print(f"  score  mean {scores.mean():.1f} +- {stderr:.1f} (std. error) | median {np.median(scores):.0f} "
              f"| max {scores.max()}")
        print(f"  steps  mean {steps.mean():.0f}")
        for reason, count in endings.most_common():
            print(f"  ended by {reason}: {count} ({100 * count / args.games:.0f}%)")


if __name__ == "__main__":
    main()
