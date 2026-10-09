"""Strategy-level actions for TiberianDawnEnv.

MacroEnv wraps the click-level environment with a small discrete set of decisions a
player makes: deploy the MCV, build or train something by role, send harvesters out,
attack, scout or defend. Finished buildings are placed automatically. Each decision
lasts decision_frames game frames.

The observation adds a normalised feature vector and the action mask, so masks travel
with observations through vector environments.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._native import AIRCRAFT, BUILDING, ENEMY, INFANTRY, SELF, UNIT
from .env import BUILDABLE_FEATURES, CATALOG, CATALOG_INDEX, GRID, PLANES, SCALARS, TiberianDawnEnv

# Roles, each a list of sidebar names in preference order: GDI and Nod build different
# things for the same job.
STRUCTURES = {
    "power": ["NUKE"],
    "advanced_power": ["NUK2"],
    "refinery": ["PROC"],
    "silo": ["SILO"],
    "barracks": ["PYLE", "HAND"],
    "factory": ["WEAP", "AFLD"],
    "radar": ["HQ"],
    "repair": ["FIX"],
    "helipad": ["HPAD"],
    "defense": ["GTWR", "GUN"],
    "advanced_defense": ["ATWR", "OBLI"],
    "anti_air": ["SAM"],
    "tech": ["EYE", "TMPL"],
}
UNITS = {
    "rifleman": ["E1"],
    "grenadier": ["E2", "E4"],
    "rocket_soldier": ["E3"],
    "harvester": ["HARV"],
    "light_vehicle": ["JEEP", "BGGY"],
    "bike": ["BIKE"],
    "apc": ["APC"],
    "tank": ["MTNK", "LTNK"],
    "heavy_tank": ["HTNK", "FTNK"],
    "artillery": ["MSAM", "ARTY"],
    "stealth_tank": ["STNK"],
    "rocket_launcher": ["MLRS"],
    "aircraft": ["ORCA", "HELI"],
}
ARMY = ("attack", "scout", "defend", "harvest")

ACTIONS = (
    ["noop", "deploy_mcv"]
    + ["build_" + role for role in STRUCTURES]
    + ["train_" + role for role in UNITS]
    + list(ARMY)
)
ACTION_INDEX = {name: i for i, name in enumerate(ACTIONS)}

# Units that don't fight.
NON_COMBAT = {b"HARV", b"MCV", b"E6"}

FEATURES = len(SCALARS) + len(CATALOG) * len(BUILDABLE_FEATURES)


def default_reward(previous, scalars, status, won, lost):
    """Win or lose, plus small shaping terms so there is signal before the first win."""
    delta = {k: scalars[k] - previous[k] for k in scalars}
    return (
        (1.0 if won else 0.0)
        - (1.0 if lost else 0.0)
        + 0.0002 * delta["harvested_credits"]
        + 0.02 * delta["units_killed"]
        + 0.05 * delta["buildings_killed"]
        - 0.02 * delta["units_lost"]
        - 0.05 * delta["buildings_lost"]
    )


class MacroEnv(gym.Wrapper):
    """
    Observation (dict):
      map:      uint8 (len(PLANES), 64, 64), as TiberianDawnEnv.
      features: float32 (FEATURES,), log-scaled scalars and buildables.
      mask:     int8 (len(ACTIONS),), 1 for actions that can apply now.
    Action: Discrete(len(ACTIONS)), see ACTIONS.
    """

    def __init__(self, env, decision_frames=30, reward_fn=default_reward):
        super().__init__(env)
        self.base = env.unwrapped
        self.base.frame_skip = decision_frames
        self.reward_fn = reward_fn
        self.observation_space = spaces.Dict(
            {
                "map": env.observation_space["map"],
                "features": spaces.Box(-np.inf, np.inf, (FEATURES,), np.float32),
                "mask": spaces.Box(0, 1, (len(ACTIONS),), np.int8),
            }
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._wrap(obs), info

    def step(self, action):
        previous = self.base._scalars
        self._apply(int(action))
        self._place_finished()
        obs, _, terminated, truncated, info = self.env.step(np.zeros(6, dtype=np.int64))
        won = terminated and info["status"] == 1
        lost = terminated and info["status"] == 2
        reward = self.reward_fn(previous, self.base._scalars, info["status"], won, lost)
        info["won"], info["lost"] = won, lost
        return self._wrap(obs), float(reward), terminated, truncated, info

    # Observation.

    def _wrap(self, obs):
        scalars = np.log1p(np.maximum(obs["scalars"], 0))
        buildables = obs["buildables"].copy()
        buildables[:, 1] = np.log1p(buildables[:, 1])  # cost
        features = np.concatenate([scalars, buildables.ravel()]).astype(np.float32)
        return {"map": obs["map"], "features": features, "mask": self.action_mask()}

    def action_mask(self):
        mask = np.zeros(len(ACTIONS), dtype=np.int8)
        mask[ACTION_INDEX["noop"]] = 1
        own = self._own()
        mask[ACTION_INDEX["deploy_mcv"]] = bool((own["name"] == b"MCV").any())
        for role, names in STRUCTURES.items():
            mask[ACTION_INDEX["build_" + role]] = self._buildable(names) is not None
        for role, names in UNITS.items():
            mask[ACTION_INDEX["train_" + role]] = self._buildable(names) is not None
        has_army = len(self._army()) > 0
        for name in ("attack", "scout", "defend"):
            mask[ACTION_INDEX[name]] = has_army
        mask[ACTION_INDEX["harvest"]] = bool((own["name"] == b"HARV").any()) and bool(self.base._tiberium.any())
        return mask

    # Helpers.

    def _own(self):
        objects = self.base.objects
        return objects[objects["relation"] == SELF]

    def _army(self):
        own = self._own()
        mobile = np.isin(own["type"], (INFANTRY, UNIT, AIRCRAFT)) & ~np.isin(own["name"], list(NON_COMBAT))
        return own[mobile]

    def _buildable(self, names):
        """First of 'names' that can be started now."""
        for name in names:
            b = self.base._buildables.get(name)
            if b is not None and not (b["busy"] or b["constructing"] or b["completed"]):
                return name
        return None

    def _base_center(self):
        own = self._own()
        anchors = own[own["type"] == BUILDING]
        if len(anchors) == 0:
            anchors = own
        if len(anchors) == 0:
            return GRID // 2, GRID // 2
        return int(np.mean(anchors["cell_x"])), int(np.mean(anchors["cell_y"]))

    def _command(self, group, x, y):
        if len(group):
            self.base._native.command(group, int(x), int(y))

    def _unexplored_target(self):
        h, w = self.base._native.height, self.base._native.width
        ys, xs = np.nonzero(self.base._explored[:h, :w] == 0)
        if len(xs) == 0:
            return None
        i = self.base.np_random.integers(len(xs))
        return xs[i], ys[i]

    def _nearest(self, cells_x, cells_y, x, y):
        d = (cells_x - x) ** 2 + (cells_y - y) ** 2
        i = int(np.argmin(d))
        return cells_x[i], cells_y[i]

    def _apply(self, action):
        name = ACTIONS[action]
        if not self.action_mask()[action]:
            return
        native = self.base._native
        if name == "deploy_mcv":
            own = self._own()
            for mcv in own[own["name"] == b"MCV"]:
                native.command(mcv[None], int(mcv["cell_x"]), int(mcv["cell_y"]))
        elif name.startswith("build_"):
            native.build(self._buildable(STRUCTURES[name[6:]]))
        elif name.startswith("train_"):
            native.build(self._buildable(UNITS[name[6:]]))
        elif name == "attack":
            army = self._army()
            objects = self.base.objects
            enemies = objects[objects["relation"] == ENEMY]
            cx, cy = np.mean(army["cell_x"]), np.mean(army["cell_y"])
            if len(enemies):
                buildings = enemies[enemies["type"] == BUILDING]
                targets = buildings if len(buildings) else enemies
                x, y = self._nearest(targets["cell_x"], targets["cell_y"], cx, cy)
            else:
                target = self._unexplored_target()
                if target is None:
                    return
                x, y = target
            self._command(army, x, y)
        elif name == "scout":
            army = self._army()
            target = self._unexplored_target()
            if target is not None:
                self._command(army[self.base.np_random.integers(len(army))][None], *target)
        elif name == "defend":
            self._command(self._army(), *self._base_center())
        elif name == "harvest":
            own = self._own()
            ys, xs = np.nonzero(self.base._tiberium)
            for harvester in own[own["name"] == b"HARV"]:
                if harvester["pips"] == 0:  # Not carrying a load back.
                    x, y = self._nearest(xs, ys, harvester["cell_x"], harvester["cell_y"])
                    native.command(harvester[None], int(x), int(y))

    def _place_finished(self):
        """Place any finished building: refineries near Tiberium, the rest near the base."""
        for name, b in self.base._buildables.items():
            if not (b["completed"] and b["type"] == BUILDING):
                continue
            ys, xs = np.nonzero(self.base.placement_mask(name))
            if len(xs) == 0:
                continue
            tx, ty = self._base_center()
            if name == "PROC":
                tys, txs = np.nonzero(self.base._tiberium)
                if len(txs):
                    tx, ty = self._nearest(txs, tys, tx, ty)
            x, y = self._nearest(xs, ys, tx, ty)
            self.base._native.place(name, int(x), int(y))


def make_macro_env(decision_frames=30, max_minutes=30, **kwargs):
    """A MacroEnv around a TiberianDawnEnv; kwargs go to TiberianDawnEnv."""
    env = TiberianDawnEnv(max_frames=int(max_minutes * 60 * 15), **kwargs)
    return MacroEnv(env, decision_frames=decision_frames)
