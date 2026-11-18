"""
    Record trained agents as GIFs for README

    python scripts/record_gif.py

    Arguments:
    checkpoints        # .pt files to record (default: ep00001, ep00100, best.pt and final.pt, if they exist)
    --output           # directory to save the GIFs (default: assets/)
    --frames-per-game  # upper limit; longer games are sped up (default fps*120)
    --max-steps        # a game still running after this is cut off (default 10000)
    --fps              # playback speed of the GIF (default 20)
    --hold             # seconds to show the last frame of each game (default 1.2)
    --cell             # cell size in pixels (default 20)
    --seed             # RNG seed for the food sequence (default 123)

    Long games are sped up: at most --frames-per-game frames are drawn per game, and the speed-up factor is shown in the corner (x1 = every step)
"""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # draw without a window
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame
from PIL import Image

from src.agent import greedy_action, load_checkpoint
from src.config import CHECKPOINT_DIR, MILESTONE_DIR, ROOT, Config
from src.env import OBS_DIM, SnakeEnv
from src.renderer import Renderer


def snapshot(env):
    return SimpleNamespace(snake=list(env.snake), food=env.food, score=env.score, steps=env.steps)


def play(net, size, max_idle_steps, seed, max_steps):
    env = SnakeEnv(size, max_idle_steps, seed=seed)
    obs = env.reset()
    states = [snapshot(env)]
    while len(states) <= max_steps:
        obs, _, terminated, truncated, _ = env.step(greedy_action(net, obs))
        states.append(snapshot(env))
        if terminated or truncated:
            return states, "finished"
    return states, "cut off"


def capture(renderer):
    return Image.frombytes("RGB", renderer.screen.get_size(),
                           pygame.image.tostring(renderer.screen, "RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", type=Path,
                        help="in this order (default: ep00001, ep00100, best.pt and final.pt, if they exist)")
    parser.add_argument("--output", type=Path, default=ROOT / "assets", help="directory to save the GIFs (default: assets/)")
    parser.add_argument("--frames-per-game", type=int, default=None, help="upper limit, longer games are sped up")
    parser.add_argument("--max-steps", type=int, default=10000, help="a game still running after this is cut off")
    parser.add_argument("--fps", type=int, default=20, help="playback speed of the GIF")
    parser.add_argument("--hold", type=float, default=1.2, help="seconds to show the last frame of each game")
    parser.add_argument("--cell", type=int, default=20, help="cell size in pixels")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.frames_per_game is None:
        args.frames_per_game = args.fps * 120

    paths = args.checkpoints or [MILESTONE_DIR / "ep00001.pt", MILESTONE_DIR / "ep00100.pt",
                                 CHECKPOINT_DIR / "best.pt", CHECKPOINT_DIR / "final.pt"]
    existing = [p for p in paths if p.exists()]
    for p in paths:
        if p not in existing:
            print(f"skipping {p}: not found")
    if not existing:
        raise SystemExit("No checkpoints found - run scripts/train.py first or pass .pt files.")

    cfg = Config()
    renderer = None
    try:
        for path in existing:
            frames, durations = [], []
            net, meta = load_checkpoint(path)
            if meta["obs_dim"] != OBS_DIM:
                raise SystemExit(f"{path} was trained with {meta['obs_dim']} features, but the environment "
                                 f"now produces {OBS_DIM}. Train a new model with scripts/train.py.")
            size = meta.get("grid_size", cfg.grid_size)
            if renderer is None:
                renderer = Renderer(size, args.cell)
            title = f"episode {meta['episode']}" if "episode" in meta else path.stem

            states, ending = play(net, size, cfg.max_idle_steps, args.seed, args.max_steps)
            stride = max(1, math.ceil(len(states) / args.frames_per_game))
            speed = f"x{stride}  " if stride > 1 else ""
            for state in states[:-1:stride]:
                renderer.draw(state, title, speed.strip())
                frames.append(capture(renderer))
                durations.append(round(1000 / args.fps))
            renderer.draw(states[-1], title, ending)
            frames.append(capture(renderer))
            durations.append(round(args.hold * 1000))
            print(f"{path.name}: score {states[-1].score}, {states[-1].steps} steps, {ending}, "
                  f"speed x{stride}")
            args.output.mkdir(parents=True, exist_ok=True)
            output_path = args.output / f"{path.stem}.gif"
            paletted = [
                frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
                for frame in frames
            ]
            paletted[0].save(
                output_path,
                save_all=True,
                append_images=paletted[1:],
                duration=durations,
                loop=0,
                optimize=True,
            )
            print(f"Saved {output_path} ({len(frames)} frames)")
    finally:
        if renderer:
            renderer.close()


if __name__ == "__main__":
    main()
