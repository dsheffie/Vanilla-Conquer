"""Adaptive curriculum over the built-in AI's difficulty.

Training starts against the easy AI and moves up a level once the win rate against the
current level reaches a threshold over its last `window` games. A `floor` share of games
stays on the easier levels, split evenly, so the policy keeps what it learned there and
doesn't come to rely on tactics that only work against one level.
"""

import collections

import numpy as np

LEVELS = ("easy", "normal", "hard")


class Curriculum:
    def __init__(self, threshold=0.5, window=100, floor=0.2, level=0, seed=0):
        self.threshold = threshold
        self.window = window
        self.floor = floor
        self.level = level
        self.results = {d: collections.deque(maxlen=window) for d in range(len(LEVELS))}
        self.rng = np.random.default_rng(seed)

    def mix(self):
        """Probability of each difficulty for the next game."""
        p = np.zeros(len(LEVELS))
        if self.level == 0:
            p[0] = 1.0
        else:
            p[self.level] = 1.0 - self.floor
            p[: self.level] = self.floor / self.level
        return p

    def sample(self):
        return int(self.rng.choice(len(LEVELS), p=self.mix()))

    def record(self, difficulty, won):
        """Count a finished game. Returns True if this moved the curriculum up a level."""
        self.results[difficulty].append(bool(won))
        current = self.results[self.level]
        if (
            self.level < len(LEVELS) - 1
            and difficulty == self.level
            and len(current) >= self.window
            and np.mean(current) >= self.threshold
        ):
            self.level += 1
            return True
        return False

    def win_rate(self, difficulty):
        games = self.results[difficulty]
        return float(np.mean(games)) if games else float("nan")

    def state(self):
        return {"level": self.level, "results": {d: list(r) for d, r in self.results.items()}}

    def load(self, state):
        self.level = state["level"]
        for d, games in state["results"].items():
            self.results[int(d)] = collections.deque(games, maxlen=self.window)
