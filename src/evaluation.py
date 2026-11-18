"""Play games with a policy"""
import numpy as np


def run_games(env, policy, games):
    """Returns (scores, steps), one entry per game."""
    scores, steps = [], []
    for _ in range(games):
        obs = env.reset()
        while True:
            obs, _, terminated, truncated, _ = env.step(policy(obs))
            if terminated or truncated:
                break
        scores.append(env.score)
        steps.append(env.steps)
    return scores, steps


def random_policy(n_actions, seed=None):
    rng = np.random.default_rng(seed)
    return lambda obs: int(rng.integers(n_actions))  # returns a function that takes an observation and returns a random action
