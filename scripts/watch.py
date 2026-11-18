"""
    Watch trained agents play (greedy policy, pygame window)

    python scripts/watch.py

    Arguments:
    --checkpoint  # only watch a single .pt file (default: all milestones in checkpoints/milestones in order)
    --games       # number of games per snapshot (default 1)
    --fps         # frames per second (default 20)
    --cell        # cell size in pixels (default 30)
    --seed        # RNG seed for reproducibility (food order, default 123)

    Controls:
    - N or right arrow = skip to the next snapshot
    - ESC / Q / close window = quit.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import greedy_action, load_checkpoint
from src.config import MILESTONE_DIR, Config
from src.env import SnakeEnv
from src.renderer import Renderer


def play(renderer, env, net, title, games, fps):
    """Returns 'quit', 'skip' or None (finished normally)."""
    for game in range(1, games + 1):
        obs = env.reset()
        note = f"game {game}/{games}" if games > 1 else ""
        while True:
            renderer.draw(env, title, note)
            command = renderer.poll()
            if command:
                return command
            obs, _, terminated, truncated, _ = env.step(greedy_action(net, obs))
            renderer.tick(fps)
            if terminated or truncated:
                break
        renderer.draw(env, title, "finished")
        print(f"{title}: score {env.score}, {env.steps} steps")
        time.sleep(1.5)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="a single .pt file (default: all milestones)")
    parser.add_argument("--games", type=int, default=1, help="games per snapshot")
    parser.add_argument("--fps", type=int, default=20, help="frames per second")
    parser.add_argument("--cell", type=int, default=30, help="cell size in pixels")
    parser.add_argument("--seed", type=int, default=123, help="random seed for reproducibility")
    args = parser.parse_args()

    paths = [args.checkpoint] if args.checkpoint else sorted(MILESTONE_DIR.glob("ep*.pt"))
    if not paths:
        raise SystemExit("No checkpoints found - run scripts/train.py first.")

    cfg = Config()
    renderer = None
    try:
        for path in paths:
            net, meta = load_checkpoint(path)
            size = meta.get("grid_size", cfg.grid_size)
            if renderer is None:
                renderer = Renderer(size, args.cell)
            env = SnakeEnv(size, cfg.max_idle_steps, seed=args.seed)  # same seed for same food order for every snapshot
            title = f"after {meta['episode']} training episodes" if "episode" in meta else path.name
            if play(renderer, env, net, title, args.games, args.fps) == "quit":
                break
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
