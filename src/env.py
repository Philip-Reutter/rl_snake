"""
Snake environment

Grid coordinates: x to the right, y downwards, (0, 0) is the top-left cell.

Actions are relative to the snake's heading:
    0 = straight, 1 = turn right, 2 = turn left

Observation: 21 floats, egocentric, OBS_NAMES
"""
from collections import deque

import numpy as np

DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # up, right, down, left
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2 # index for TURN
N_ACTIONS = 3
TURN = (0, 1, -1)  # action, right = 1, left = - 1

OBS_NAMES = (
    "obstacle_proximity_ahead",   # 1 / (1 + free cells before wall/body) in that direction
    "obstacle_proximity_right",
    "obstacle_proximity_left",
    "food_ahead",                 # signed food offset along heading / grid_size
    "food_right",                 # signed food offset to the right / grid_size
    "food_is_ahead",              # 4 binary flags for the food's side
    "food_is_behind",
    "food_is_right",
    "food_is_left",
    "room_ahead",                 # min(1, cells reachable after this move / snake length)
    "room_right",                 # -> drops below 1 when  move leads into dead end
    "room_left",
    "snake_length",               # length / (grid_size ** 2)
    "area_ahead",                 # cells reachable after this move / all free cells (no cap)
    "area_right",
    "area_left",
    "tail_reachable_ahead",       # 1 if the region reachable after this move contains the tail
    "tail_reachable_right",       # -> a path to the tail means the snake is never fully trapped
    "tail_reachable_left",
    "tail_ahead",                 # signed tail offset along heading / grid_size
    "tail_right",                 # signed tail offset to the right / grid_size
)
OBS_DIM = len(OBS_NAMES)


class SnakeEnv:
    def __init__(self, size=20, max_idle_steps=200, reward_food=1.0,
                 reward_death=-1.0, shaping=0.02, seed=None):
        self.size = size                      # grid size
        self.max_idle_steps = max_idle_steps  # steps allowed without eating before truncation
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.shaping = shaping                # reward per cell moved closer to / away from food
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        cx = cy = self.size // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first, initial length 3
        self.occupied = set(self.snake)
        self.direction = 1  # heading right
        self.score = 0
        self.steps = 0
        self.idle = 0
        self.food = self._place_food()
        return self.observe()

    def step(self, action):
        """
        Returns (obs, reward, terminated, truncated, info)

        terminated: the snake died (or filled the board)
        truncated: idle limit hit
        """
        self.direction = (self.direction + TURN[action]) % 4
        dx, dy = DIRS[self.direction]
        hx, hy = self.snake[0]
        new = (hx + dx, hy + dy)
        self.steps += 1
        self.idle += 1

        if not self._is_free(*new) and new != self.snake[-1]:
            return self.observe(), self.reward_death, True, False, {"score": self.score}

        ate = new == self.food
        if not ate:
            dist_before = abs(hx - self.food[0]) + abs(hy - self.food[1])
            dist_after = abs(new[0] - self.food[0]) + abs(new[1] - self.food[1])
            self.occupied.discard(self.snake.pop())
        self.snake.appendleft(new)
        self.occupied.add(new)

        terminated = False
        if ate:
            self.score += 1
            self.idle = 0
            reward = self.reward_food
            self.food = self._place_food()
            terminated = self.food is None  # board full, game won
        else:
            reward = self.shaping * (dist_before - dist_after)

        truncated = (not terminated) and self.idle >= self.max_idle_steps
        return self.observe(), reward, terminated, truncated, {"score": self.score}

    def observe(self):
        hx, hy = self.snake[0]
        tx, ty = self.snake[-1]
        ahead = DIRS[self.direction]
        right = DIRS[(self.direction + 1) % 4]
        left = DIRS[(self.direction - 1) % 4]
        length = len(self.snake)
        free_total = self.size * self.size - length + 1  # empty cells + the tail cell (counts as free)

        proximity, room, area, tail_reachable = [], [], [], []
        for dx, dy in (ahead, right, left):
            first = (hx + dx, hy + dy)
            free_cells, (x, y) = 0, first
            while self._is_free(x, y):
                free_cells += 1
                x, y = x + dx, y + dy
            proximity.append(1.0 / (1 + free_cells))
            if free_cells:  # the first cell is free, so the move is possible
                reached, tail_in_region = self._flood(first)
                room.append(min(1.0, reached / length))
                area.append(reached / free_total)
                tail_reachable.append(float(tail_in_region))
            else:
                room.append(0.0)
                area.append(0.0)
                tail_reachable.append(0.0)

        if self.food is None:  # snake filled board, game won
            food = [0.0] * 6  # fill six food features with zeros
        else:
            fx, fy = self.food[0] - hx, self.food[1] - hy
            f_ahead = fx * ahead[0] + fy * ahead[1]
            f_right = fx * right[0] + fy * right[1]
            food = [f_ahead / self.size, f_right / self.size,
                    f_ahead > 0, f_ahead < 0, f_right > 0, f_right < 0]

        tail_ahead = ((tx - hx) * ahead[0] + (ty - hy) * ahead[1]) / self.size
        tail_right = ((tx - hx) * right[0] + (ty - hy) * right[1]) / self.size

        obs = (proximity + food + room + [length / (self.size * self.size)]
                + area + tail_reachable + [tail_ahead, tail_right])
        return np.array(obs, dtype=np.float32)

    def _is_free(self, x, y):
        return (0 <= x < self.size and 0 <= y < self.size
                and ((x, y) not in self.occupied or (x, y) == self.snake[-1]))


    def _flood(self, start):
        """Number of empty cells reachable from start"""
        seen, stack = {start}, [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in DIRS:
                cell = (x + dx, y + dy)
                if cell not in seen and self._is_free(*cell):
                    seen.add(cell)
                    stack.append(cell)
        return len(seen), self.snake[-1] in seen


    def _place_food(self):
        free = [(x, y) for x in range(self.size) for y in range(self.size)
                if (x, y) not in self.occupied]
        if not free:
            return None
        return free[self.rng.integers(len(free))]
