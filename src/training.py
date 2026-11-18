import csv
import time

import numpy as np
import torch

from src.agent import DQNAgent, greedy_action, save_checkpoint
from src.config import CHECKPOINT_DIR, RESULTS_DIR
from src.env import N_ACTIONS, OBS_DIM, SnakeEnv
from src.evaluation import run_games
from src.replay_buffer import ReplayBuffer
from src.plotting import plot_curves

# Food sequence of final evaluation, same for every model and never used during training
# Scores of different runs are measured on identical games
HELDOUT_SEED = 123_456


def make_env(cfg, seed):
    return SnakeEnv(cfg.grid_size, cfg.max_idle_steps, cfg.reward_food,
                    cfg.reward_death, cfg.shaping, seed=seed)


def train(cfg, checkpoint_dir=CHECKPOINT_DIR, results_dir=RESULTS_DIR, device="cpu",
          on_eval=None, final_eval_games=0, plot=True, verbose=True):
    """returns {"eval_means": mean greedy score of every evaluation, "final_scores": list or None}"""
    torch.manual_seed(cfg.seed)
    env, eval_env = make_env(cfg, cfg.seed), make_env(cfg, cfg.seed + 1)
    buffer = ReplayBuffer(cfg.buffer_size, OBS_DIM, seed=cfg.seed)
    agent = DQNAgent(OBS_DIM, N_ACTIONS, cfg, device, seed=cfg.seed)

    milestone_dir = checkpoint_dir / "milestones"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    if cfg.milestones:
        milestone_dir.mkdir(exist_ok=True)
        for old in milestone_dir.glob("ep*.pt"):  # don't mix snapshots of different runs
            old.unlink()

    def save(path, episode, **extra):
        save_checkpoint(path, agent.online, OBS_DIM, N_ACTIONS, cfg.hidden,
                        episode=episode, grid_size=cfg.grid_size, **extra)

    greedy = lambda o: greedy_action(agent.online, o, device)

    train_f = open(results_dir / "training_log.csv", "w", newline="")
    eval_f = open(results_dir / "eval_log.csv", "w", newline="")
    train_log, eval_log = csv.writer(train_f), csv.writer(eval_f)
    train_log.writerow(["episode", "score", "steps", "reward", "epsilon", "loss"])
    eval_log.writerow(["episode", "mean_score", "max_score"])

    best_eval, total_steps, start = -1.0, 0, time.time()
    scores, eval_means, final_scores = [], [], None
    episode = 0
    try:
        for episode in range(1, cfg.episodes + 1):
            epsilon = max(cfg.eps_end, cfg.eps_start * cfg.eps_decay ** (episode - 1))
            obs = env.reset()
            ep_reward, losses = 0.0, []
            while True:
                action = agent.act(obs, epsilon)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # Only a real death stops bootstrapping; a time-limit cut-off does not.
                buffer.add(obs, action, reward, next_obs, terminated)
                obs, ep_reward, total_steps = next_obs, ep_reward + reward, total_steps + 1
                if len(buffer) >= cfg.warmup_steps:
                    losses.append(agent.learn(buffer))
                if terminated or truncated:
                    break

            scores.append(env.score)
            train_log.writerow([episode, env.score, env.steps, f"{ep_reward:.4f}", f"{epsilon:.4f}",
                                f"{np.mean(losses):.6f}" if losses else "nan"])
            train_f.flush()

            if episode in cfg.milestones:
                save(milestone_dir / f"ep{episode:05d}.pt", episode)

            if episode % cfg.eval_every == 0 or episode == cfg.episodes:
                eval_scores, _ = run_games(eval_env, greedy, cfg.eval_games)
                mean_eval = float(np.mean(eval_scores))
                eval_means.append(mean_eval)
                eval_log.writerow([episode, f"{mean_eval:.2f}", max(eval_scores)])
                eval_f.flush()
                if mean_eval > best_eval:
                    best_eval = mean_eval
                    save(checkpoint_dir / "best.pt", episode, eval_mean_score=mean_eval)
                if on_eval is not None:
                    on_eval(episode, mean_eval)

            if verbose and episode % cfg.log_every == 0:
                print(f"ep {episode:5d} | avg score(last {cfg.log_every}) {np.mean(scores[-cfg.log_every:]):5.2f} "
                      f"| best greedy eval {best_eval:5.2f} | eps {epsilon:.3f} "
                      f"| steps {total_steps:7d} | {time.time() - start:5.0f}s", flush=True)

        if final_eval_games:
            final_scores, _ = run_games(make_env(cfg, HELDOUT_SEED), greedy, final_eval_games)
    except KeyboardInterrupt:
        print("\nInterrupted - saving current model and exiting...")
    finally:
        train_f.close()
        eval_f.close()
        if episode:
            save(checkpoint_dir / "final.pt", episode)
            if verbose:
                print(f"Saved checkpoints to {checkpoint_dir}")
            if plot:
                path = plot_curves(results_dir)
                if verbose:
                    print(f"Learning curves: {path}")
    return {"eval_means": eval_means, "final_scores": final_scores}
