"""
    Compares current config with the same config plus one or more changes

    python scripts/compare.py

    Arguments:
    --set NAME=VALUE  # config change for the variant, can be given several times (e.g. --set gamma=0.97 --set lr=0.0005)
    --seeds           # number of seeds to run (default 3)
    --episodes        # number of episodes to train (default config.episodes)
    --jobs            # number of parallel processes (one CPU core each)

    Both variants are trained for the full config.episodes with the same seeds (1000, 1001, like `confirm` in hyperparameter_opt.py)
    and scored on the same 100 held-out games, so each seed is a paired comparison
"""
import argparse
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.training import train

SEED_START = 1000
FINAL_GAMES = 100


def parse_overrides(pairs, cfg):
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not hasattr(cfg, key) or not value:
            raise SystemExit(f"--set expects NAME=VALUE with a Config field, got '{pair}'")
        try:
            overrides[key] = type(getattr(cfg, key))(value)
        except ValueError:
            raise SystemExit(f"cannot use '{value}' for Config.{key} (expected {type(getattr(cfg, key)).__name__})")
    return overrides


def run(cfg, seed):
    cfg = replace(cfg, seed=seed, milestones=())
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        result = train(cfg, out / "ckpt", out / "res", final_eval_games=FINAL_GAMES, plot=False, verbose=False)
    return float(np.mean(result["final_scores"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", required=True, metavar="NAME=VALUE",
                        help="Config change for the variant; can be given several times")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, help="default: Config.episodes")
    parser.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()

    base = Config()
    if args.episodes:
        base = replace(base, episodes=args.episodes)
    overrides = parse_overrides(args.set, base)
    variant = replace(base, **overrides)
    label = " ".join(f"{k}={v}" for k, v in overrides.items())
    seeds = [SEED_START + i for i in range(args.seeds)]
    print(f"baseline vs. {label}: {len(seeds)} seeds x {base.episodes} episodes, {args.jobs} in parallel", flush=True)

    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [(pool.submit(run, base, s), pool.submit(run, variant, s)) for s in seeds]
        scores = [(a.result(), b.result()) for a, b in futures]

    base_scores, var_scores = np.array(scores).T
    diff = var_scores - base_scores
    print(f"\nHeld-out score (mean of {FINAL_GAMES} games, same games for both)")
    print(f"  {'seed':>6} {'baseline':>9} {'variant':>9} {'diff':>7}")
    for s, b, v, d in zip(seeds, base_scores, var_scores, diff):
        print(f"  {s:>6} {b:9.1f} {v:9.1f} {d:+7.1f}")
    print(f"  {'mean':>6} {base_scores.mean():9.1f} {var_scores.mean():9.1f} {diff.mean():+7.1f}")
    print(f"\n'{label}' is better on {int((diff > 0).sum())} of {len(seeds)} seeds. "
          f"Seed-to-seed spread of the baseline: {base_scores.std(ddof=1) if len(seeds) > 1 else float('nan'):.1f}. "  # ddof=1 for sample stddev
          f"A mean difference below that spread is not conclusive, use more seeds.")


if __name__ == "__main__":
    main()
