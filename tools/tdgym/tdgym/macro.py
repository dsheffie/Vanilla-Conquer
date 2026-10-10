"""Strategy-level actions for TiberianDawnEnv.

MacroEnv wraps the click-level environment with a small discrete set of decisions a
player makes: deploy the MCV, build or train something by role, send harvesters out,
attack, hunt, scout or defend. Finished buildings are placed automatically. Each decision
lasts decision_frames game frames.

Army orders:
- attack is an attack-move on the enemy's production: the army heads for the nearest known
  construction yard, factory or barracks (else any building, any enemy, or unexplored
  ground), and every decision each unit fights the nearest visible enemy within
  ENGAGE_RADIUS cells instead, if there is one. It stands until another army order.
- hunt puts the army into the game's own search-and-destroy mission, as the built-in AI
  attacks: each unit seeks out the greatest threat it can find.
- scout sends one unit to unexplored ground, defend brings the army home, and harvest
  sends idle harvesters to the nearest known Tiberium.

The observation adds a normalised feature vector and the action mask, so masks travel
with observations through vector environments.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ._native import AIRCRAFT, BUILDING, ENEMY, INFANTRY, SELF, UNIT
from .env import BUILDABLE_FEATURES, CATALOG, CATALOG_INDEX, GRID, PLANES, SCALARS, TiberianDawnEnv
from .expert import Expert

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
# hunt is last so the earlier actions keep their indices from before it was added.
ARMY = ("attack", "scout", "defend", "harvest", "hunt")
# Enemy buildings attack goes for first: what the enemy builds with.
PRODUCTION = {b"FACT", b"WEAP", b"AFLD", b"PYLE", b"HAND", b"HPAD"}
# Cells within which a unit on attack fights an enemy rather than walking past it.
ENGAGE_RADIUS = 5

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


@dataclass
class Reward:
    """
    Reward weights. Winning dominates: everything else is shaping toward it, and is capped
    well below the win reward over a whole game.

    An episode that hits the time limit scores almost like a loss and ends there, rather
    than being treated as cut short, so a policy can't score by stalling; destroying
    enemy buildings is what makes up the difference.
    """

    win: float = 10.0
    loss: float = -10.0
    timeout: float = -8.0
    enemy_building_destroyed: float = 0.2
    enemy_unit_killed: float = 0.05
    building_lost: float = -0.2
    unit_lost: float = -0.05
    # Spread over the whole map: exploring all of it is worth this much in total.
    explored: float = 1.0
    # Per credit harvested. Enough to get an economy going, too little to be the goal:
    # a strong 30 minute economy (~45k credits) is worth about 0.45.
    harvested: float = 0.00001

    def __call__(self, previous, scalars, explored_delta, won, lost, timed_out):
        delta = {k: scalars[k] - previous[k] for k in scalars}
        return (
            (self.win if won else 0.0)
            + (self.loss if lost else 0.0)
            + (self.timeout if timed_out else 0.0)
            + self.enemy_building_destroyed * delta["buildings_killed"]
            + self.enemy_unit_killed * delta["units_killed"]
            + self.building_lost * delta["buildings_lost"]
            + self.unit_lost * delta["units_lost"]
            + self.explored * explored_delta
            + self.harvested * delta["harvested_credits"]
        )


class MacroEnv(gym.Wrapper):
    """
    Observation (dict):
      map:      uint8 (len(PLANES), 64, 64), as TiberianDawnEnv.
      features: float32 (FEATURES,), log-scaled scalars and buildables.
      mask:     int8 (len(ACTIONS),), 1 for actions that can apply now.
      expert:   int8 (len(ACTIONS),), 1 for the actions the built-in AI would take now
                (see expert.py, expert= picks the variant); all 0 when it would wait.
    Action: Discrete(len(ACTIONS)), see ACTIONS.
    """

    def __init__(self, env, decision_frames=30, reward=None, expert="ai"):
        super().__init__(env)
        self.base = env.unwrapped
        self.base.frame_skip = decision_frames
        self.reward = reward if reward is not None else Reward()
        self.observation_space = spaces.Dict(
            {
                "map": env.observation_space["map"],
                "features": spaces.Box(-np.inf, np.inf, (FEATURES,), np.float32),
                "mask": spaces.Box(0, 1, (len(ACTIONS),), np.int8),
                "expert": spaces.Box(0, 1, (len(ACTIONS),), np.int8),
            }
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.expert = Expert(self, expert)

    @property
    def ai_difficulty(self):
        """The base env's AI difficulty, settable here so vector envs' set_attr reaches it."""
        return self.base.ai_difficulty

    @ai_difficulty.setter
    def ai_difficulty(self, value):
        self.base.ai_difficulty = value

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.expert.reset()
        self._attacking = False
        self._attack_goal = None  # Exploration goal while no enemy is known.
        self._orders = {}  # Unit id: the cell it was last sent to on attack.
        self._explored_fraction = self._explored_now()
        return self._wrap(obs), info

    def step(self, action):
        previous = self.base._scalars
        self._apply(int(action))
        if self._attacking:
            self._drive_attack()
        self.expert.note_action(ACTIONS[int(action)], previous["frame"])
        self._place_finished()
        obs, _, terminated, truncated, info = self.env.step(np.zeros(6, dtype=np.int64))
        won = terminated and info["status"] == 1
        lost = terminated and info["status"] == 2
        timed_out = truncated and not terminated
        explored = self._explored_now()
        reward = self.reward(previous, self.base._scalars, explored - self._explored_fraction, won, lost, timed_out)
        self._explored_fraction = explored
        info["won"], info["lost"], info["timed_out"] = won, lost, timed_out
        # A timeout is the end of the game for scoring, not an interruption: report it as
        # terminal so learners don't bootstrap a value past it and undo the penalty.
        return self._wrap(obs), float(reward), terminated or timed_out, False, info

    def _explored_now(self):
        return float(self.base._explored.mean())

    # Observation.

    def _wrap(self, obs):
        scalars = np.log1p(np.maximum(obs["scalars"], 0))
        buildables = obs["buildables"].copy()
        buildables[:, 1] = np.log1p(buildables[:, 1])  # cost
        features = np.concatenate([scalars, buildables.ravel()]).astype(np.float32)
        mask = self.action_mask()
        return {"map": obs["map"], "features": features, "mask": mask, "expert": self.expert.mask(mask)}

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
        for name in ("attack", "scout", "defend", "hunt"):
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
            self._attacking = True
            self._orders = {}
        elif name == "hunt":
            self._attacking = False
            native.hunt(self._army())
        elif name == "scout":
            self._attacking = False
            army = self._army()
            target = self._unexplored_target()
            if target is not None:
                self._command(army[self.base.np_random.integers(len(army))][None], *target)
        elif name == "defend":
            self._attacking = False
            self._command(self._army(), *self._base_center())
        elif name == "harvest":
            own = self._own()
            ys, xs = np.nonzero(self.base._tiberium)
            for harvester in own[own["name"] == b"HARV"]:
                if harvester["pips"] == 0:  # Not carrying a load back.
                    x, y = self._nearest(xs, ys, harvester["cell_x"], harvester["cell_y"])
                    native.command(harvester[None], int(x), int(y))

    def _goal(self, army):
        """Where attack heads: the nearest known enemy production, else building, else any
        enemy, else unexplored ground (kept until explored)."""
        objects = self.base.objects
        enemies = objects[objects["relation"] == ENEMY]
        cx, cy = np.mean(army["cell_x"]), np.mean(army["cell_y"])
        buildings = enemies[enemies["type"] == BUILDING]
        production = buildings[np.isin(buildings["name"], list(PRODUCTION))]
        for targets in (production, buildings, enemies):
            if len(targets):
                return self._nearest(targets["cell_x"], targets["cell_y"], cx, cy)
        goal = self._attack_goal
        if goal is None or self.base._explored[goal[1], goal[0]]:
            goal = self._attack_goal = self._unexplored_target()
        return goal

    def _drive_attack(self):
        """One decision of attack-move: each unit fights the nearest visible enemy within
        ENGAGE_RADIUS, or else heads for the goal. Orders are only re-sent when they change,
        so units aren't interrupted mid-fight."""
        army = self._army()
        if len(army) == 0:
            return
        goal = self._goal(army)
        objects = self.base.objects
        enemies = objects[objects["relation"] == ENEMY]
        orders = {}
        for unit in army:
            target = goal
            if len(enemies):
                d = (enemies["cell_x"] - unit["cell_x"]) ** 2 + (enemies["cell_y"] - unit["cell_y"]) ** 2
                i = int(np.argmin(d))
                if d[i] <= ENGAGE_RADIUS**2:
                    target = (enemies["cell_x"][i], enemies["cell_y"][i])
            if target is None:
                continue
            target = (int(target[0]), int(target[1]))
            uid = int(unit["id"])
            if self._orders.get(uid) != target:
                self.base._native.command(unit[None], *target)
            orders[uid] = target
        self._orders = orders

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


def make_macro_env(decision_frames=30, max_minutes=30, expert="ai", **kwargs):
    """A MacroEnv around a TiberianDawnEnv; kwargs go to TiberianDawnEnv."""
    env = TiberianDawnEnv(max_frames=int(max_minutes * 60 * 15), **kwargs)
    return MacroEnv(env, decision_frames=decision_frames, expert=expert)
