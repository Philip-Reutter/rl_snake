"""Learning curves from the CSV logs written by scripts/train.py"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"      # train / greedy-eval series
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2de"


def _read(path):
    if not path.exists():
        raise SystemExit(f"{path} not found - run scripts/train.py first.")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    return np.array([np.nanmean(x[max(0, i - window + 1): i + 1]) if not np.all(np.isnan(x[max(0, i - window + 1): i + 1])) else np.nan
                     for i in range(len(x))])


def _style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("episode", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def plot_curves(results_dir, window=50):
    """Reads training_log.csv (+ eval_log.csv if present), writes learning_curves.png"""
    results_dir = Path(results_dir)
    log = _read(results_dir / "training_log.csv")
    if not log:
        raise SystemExit(f"No data in {results_dir / 'training_log.csv'} - run scripts/train.py first.")
    ev_path = results_dir / "eval_log.csv"
    ev = _read(ev_path) if ev_path.exists() else {}
    ep = log["episode"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    ax = axes[0, 0]
    ax.plot(ep, log["score"], color=BLUE, alpha=0.2, linewidth=1)
    ax.plot(ep, moving_average(log["score"], window), color=BLUE, linewidth=2,
            label=f"training (moving avg, {window})")
    if ev:
        ax.plot(ev["episode"], ev["mean_score"], color=ORANGE, linewidth=2, marker="o", markersize=5,
                label="greedy evaluation (mean)")
    ax.legend(frameon=False, labelcolor=MUTED)
    _style(ax, "Score (food eaten)", "score")

    ax = axes[0, 1]
    ax.plot(ep, log["steps"], color=BLUE, alpha=0.2, linewidth=1)
    ax.plot(ep, moving_average(log["steps"], window), color=BLUE, linewidth=2)
    _style(ax, "Episode length (steps survived)", "steps")

    ax = axes[1, 0]
    ax.plot(ep, moving_average(log["loss"], window), color=BLUE, linewidth=2)
    ax.set_yscale("log")
    _style(ax, "TD loss (Huber, moving avg)", "loss")

    ax = axes[1, 1]
    ax.plot(ep, log["epsilon"], color=BLUE, linewidth=2)
    _style(ax, "Exploration rate epsilon", "epsilon")

    fig.tight_layout()
    out = results_dir / "learning_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
