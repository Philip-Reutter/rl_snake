from collections import deque

import numpy as np
import pytest

from env import (DIRS, N_ACTIONS, OBS_DIM, OBS_NAMES, STRAIGHT, TURN_LEFT,
                          TURN_RIGHT, SnakeEnv)

UP, RIGHT, DOWN, LEFT = range(4)


def make_env(snake, direction, food, size=20, **kwargs):
    env = SnakeEnv(size, seed=0, **kwargs)
    env.snake = deque(snake)
    env.occupied = set(snake)
    env.direction = direction
    env.food = food
    return env


def obs_of(env):
    return dict(zip(OBS_NAMES, env.observe()))


def test_reset_gives_valid_observation():
    env = SnakeEnv(20, seed=0)
    obs = env.reset()
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert len(env.snake) == 3 and env.score == 0
    assert env.food not in env.occupied


def test_wall_collision_terminates():
    env = make_env([(0, 0), (1, 0), (2, 0)], LEFT, food=(10, 10))
    _, reward, terminated, truncated, _ = env.step(STRAIGHT)
    assert terminated and not truncated and reward == env.reward_death


def test_eating_grows_snake_and_rewards():
    env = make_env([(5, 5), (4, 5), (3, 5)], RIGHT, food=(6, 5))
    _, reward, terminated, _, _ = env.step(STRAIGHT)
    assert reward == env.reward_food and not terminated
    assert env.score == 1 and len(env.snake) == 4
    assert env.food not in env.occupied


def test_moving_without_food_keeps_length_and_shapes_reward():
    env = make_env([(5, 5), (4, 5), (3, 5)], RIGHT, food=(10, 5), shaping=0.02)
    _, reward, _, _, _ = env.step(STRAIGHT)
    assert len(env.snake) == 3 and reward == pytest.approx(0.02)
    _, reward, _, _, _ = env.step(TURN_RIGHT)
    assert len(env.snake) == 3 and reward == pytest.approx(-0.02)


def test_self_collision_terminates():
    env = make_env([(5, 5), (6, 5), (6, 6), (5, 6), (4, 6)], LEFT, food=(15, 15))
    _, reward, terminated, _, _ = env.step(TURN_LEFT)
    assert terminated and reward == env.reward_death


def test_moving_into_the_tail_cell_is_legal():
    env = make_env([(5, 5), (6, 5), (6, 6), (5, 6)], DOWN, food=(15, 15))
    _, _, terminated, _, _ = env.step(STRAIGHT)
    assert not terminated


def test_idle_limit_truncates_but_does_not_terminate():
    env = make_env([(10, 10), (9, 10), (8, 10)], RIGHT, food=(0, 19), max_idle_steps=5)
    flags = [env.step(STRAIGHT)[2:4] for _ in range(5)]
    assert flags[:4] == [(False, False)] * 4
    assert flags[4] == (False, True)  # limit reached, truncated but not terminated


def test_board_invariants_hold_over_random_play():
    env = SnakeEnv(8, seed=3, max_idle_steps=10_000)
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(int(rng.integers(N_ACTIONS)))
            done = terminated or truncated
            assert env.occupied == set(env.snake) or terminated
            if not terminated:
                assert len(env.occupied) == len(env.snake)
                assert env.food is None or env.food not in env.occupied
            assert obs.shape == (OBS_DIM,) and np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_observation_walls_and_food():
    env = make_env([(19, 5), (18, 5), (17, 5)], RIGHT, food=(19, 2))
    o = obs_of(env)
    assert o["obstacle_proximity_ahead"] == 1.0  # wall in front
    assert o["obstacle_proximity_left"] == pytest.approx(1 / 6)  # 5 free cells upwards
    assert o["food_ahead"] == 0.0 and o["food_right"] == pytest.approx(-3 / 20)
    assert o["food_is_left"] == 1.0 and o["food_is_right"] == 0.0
    assert o["food_is_ahead"] == 0.0 and o["food_is_behind"] == 0.0


def test_observation_is_egocentric():
    right = obs_of(make_env([(8, 10), (7, 10), (6, 10)], RIGHT, food=(12, 10)))
    up = obs_of(make_env([(10, 8), (10, 9), (10, 10)], UP, food=(10, 4)))
    for key in ("food_ahead", "food_right", "food_is_ahead", "food_is_behind",
                "food_is_right", "food_is_left"):
        assert right[key] == up[key]


# 5x5 board. Cell (3,2), right of the head, is a pocket enclosed by the snake's own body
# The tail is (2,1), above the head, so it does not touch the pocket
POCKET_SNAKE = [(2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (3, 1), (2, 1)]


def test_room_and_area_features_detect_dead_end():
    o = obs_of(make_env(POCKET_SNAKE, RIGHT, food=(0, 0), size=5))
    assert o["room_ahead"] == pytest.approx(1 / 8)  # 1 reachable cell, snake length 8
    assert o["area_ahead"] == pytest.approx(1 / 18)  # 1 of 17 empty cells + the tail cell
    assert o["room_right"] == 0.0 and o["area_right"] == 0.0  # moving down hits the body
    assert o["room_left"] == 1.0  # moving up (into the tail cell) is open
    assert o["area_left"] == pytest.approx(17 / 18)  # everything except the pocket


def test_tail_reachable_flag():
    o = obs_of(make_env(POCKET_SNAKE, RIGHT, food=(0, 0), size=5))
    assert o["tail_reachable_ahead"] == 0.0  # the pocket cannot reach the tail
    assert o["tail_reachable_right"] == 0.0  # blocked move
    assert o["tail_reachable_left"] == 1.0  # that move enters the tail cell itself


def test_tail_counts_as_free_for_the_observation():
    snake = [(2, 2), (2, 3), (3, 3), (4, 3), (4, 2), (4, 1), (3, 1)]
    o = obs_of(make_env(snake, RIGHT, food=(0, 0), size=5))
    assert o["room_ahead"] == 1.0
    assert o["tail_reachable_ahead"] == 1.0


def test_open_space_has_full_area_and_tail_access():
    o = obs_of(make_env([(10, 10), (9, 10), (8, 10)], RIGHT, food=(0, 0)))
    for side in ("ahead", "right", "left"):
        assert o[f"area_{side}"] == pytest.approx(1.0)
        assert o[f"tail_reachable_{side}"] == 1.0


def test_tail_offset_is_egocentric():
    right = obs_of(make_env([(10, 10), (9, 10), (8, 10)], RIGHT, food=(0, 0)))
    up = obs_of(make_env([(10, 10), (10, 11), (10, 12)], UP, food=(0, 0)))
    assert right["tail_ahead"] == up["tail_ahead"] == pytest.approx(-2 / 20)  # tail is behind
    assert right["tail_right"] == up["tail_right"] == 0.0
    corner = obs_of(make_env([(10, 10), (10, 11), (10, 12)], RIGHT, food=(0, 0)))
    assert corner["tail_ahead"] == 0.0 and corner["tail_right"] == pytest.approx(2 / 20)  # tail to the right
