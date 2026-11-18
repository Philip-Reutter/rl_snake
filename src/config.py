"""All hyperparameters in one place."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
MILESTONE_DIR = CHECKPOINT_DIR / "milestones"
RESULTS_DIR = ROOT / "results"


# Hyperparameters for the environment, DQN, exploration and the training run
# After testing different values, most influential hyperparameters were optimized (see hyperparameter_opt.py)
@dataclass
class Config:
    # environment
    grid_size: int = 20
    max_idle_steps: int = 200      # episode is cut off after this many steps without food
    reward_food: float = 1.0
    reward_death: float = -1.0
    shaping: float = 0.02          # small reward for moving closer to the food
    #shaping: float = 0.00

    # DQN
    hidden: int = 128
    lr: float = 0.0003951
    gamma: float = 0.9945
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_steps: int = 1_000       # collect transitions before learning
    target_update_every: int = 459  # gradient steps between target-network copies
    grad_clip: float = 10.0

    # exploration (epsilon-greedy, decayed per episode)
    eps_start: float = 1.0
    eps_end: float = 0.005444
    eps_decay: float = 0.9896       # eps = max(eps_end, eps_start * eps_decay ** (episode - 1))

    # training run
    episodes: int = 1000
    #episodes: int = 600
    seed: int = 42
    eval_every: int = 50           # greedy evaluation every N episodes
    eval_games: int = 10
    log_every: int = 25
    # greedy snapshots
    milestones: tuple = (1, 10, 25, 50, 100, 200, 400, 600, 800 ,1000, 1500, 2000)  # save greedy snapshots after these episodes
