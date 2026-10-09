"""Gymnasium environment for Tiberian Dawn skirmishes against the built-in AI."""

import os
import tempfile
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import _native
from ._native import AIRCRAFT, BUILDING, ENEMY, INFANTRY, LOST, NEUTRAL, SELF, TERRAIN, UNIT, WON

# Everything the multiplayer sidebar can offer, GDI and Nod. Special weapons are left out.
CATALOG = (
    # Structures.
    "NUKE", "NUK2", "PROC", "SILO", "PYLE", "HAND", "WEAP", "AFLD", "FIX", "HQ", "HPAD", "EYE",
    "TMPL", "GTWR", "ATWR", "OBLI", "GUN", "SAM", "SBAG", "CYCL", "BRIK", "BARB", "WOOD",
    # Infantry.
    "E1", "E2", "E3", "E4", "E5", "E6", "RMBO",
    # Vehicles.
    "HARV", "MCV", "JEEP", "APC", "MTNK", "HTNK", "MSAM", "BGGY", "BIKE", "LTNK", "ARTY", "FTNK",
    "STNK", "MLRS", "MHQ", "LST",
    # Aircraft.
    "TRAN", "ORCA", "HELI",
)
CATALOG_INDEX = {name: i for i, name in enumerate(CATALOG)}

# Grids are padded to this size so every map has the same observation shape. The largest
# Tiberian Dawn maps are 62 cells plus the one cell border.
GRID = 64

# Map observation planes.
PLANES = (
    "explored",           # 1 where the agent has explored.
    "tiberium",           # Density 0-12.
    "own_buildings",      # 1 over each footprint.
    "own_vehicles",       # Count per cell, vehicles and aircraft.
    "own_infantry",       # Count per cell.
    "enemy_buildings",
    "enemy_vehicles",
    "enemy_infantry",
    "neutral",            # Trees, civilian buildings and the like.
    "own_health",         # Highest health fraction in the cell, 0-255.
    "enemy_health",
    "in_map",             # 1 inside the map, 0 in the padding.
)
PLANE = {name: i for i, name in enumerate(PLANES)}

SCALARS = (
    "frame", "credits", "tiberium", "max_tiberium", "power_produced", "power_drained",
    "units_killed", "buildings_killed", "units_lost", "buildings_lost", "harvested_credits",
)

# Per catalog item: whether it is offered now, and its production state.
BUILDABLE_FEATURES = ("available", "cost", "progress", "completed", "constructing", "on_hold", "busy")

# Action kinds.
NOOP, BUILD, PLACE, COMMAND, STOP, SELL, CANCEL = range(7)
ACTION_KINDS = ("noop", "build", "place", "command", "stop", "sell", "cancel")

MOBILE_TYPES = (INFANTRY, UNIT, AIRCRAFT)


def _env_path(value, variable):
    value = value or os.environ.get(variable)
    if not value:
        raise ValueError("pass a path or set $%s" % variable)
    return value


class TiberianDawnEnv(gym.Env):
    """
    The agent plays player 0 of a multiplayer skirmish against built-in AI players, with
    the information a human would have: other houses' objects only show in explored cells.

    Observation (dict):
      map:        uint8 (len(PLANES), 64, 64), see PLANES; grid cell [y, x].
      scalars:    float32 (len(SCALARS),), raw values.
      buildables: float32 (len(CATALOG), len(BUILDABLE_FEATURES)).

    Action: MultiDiscrete [kind, item, x, y, src_x, src_y], see ACTION_KINDS:
      build / cancel item; place a completed building item with its top left at (x, y);
      command your mobile units within group_radius cells of (src_x, src_y) at (x, y), as a
      human selecting them and clicking there (move, attack, harvest, enter, or deploy an
      MCV commanded onto itself); stop that group; sell your building covering (x, y).
    Invalid actions do nothing; info["action_valid"] reports whether it applied, and
    info["action_mask"] which kinds and items are currently possible. placement_mask(item)
    gives the cells where a completed building can go.

    Reward: +1 for winning, -1 for losing, unless reward_fn(previous_scalars, scalars,
    status) is given. Episodes are truncated after max_frames game frames.

    The game is a process-wide singleton: one environment per process. For parallel
    environments use gymnasium.vector.AsyncVectorEnv.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        game_lib=None,
        data=None,
        tdenv_lib=None,
        work_dir=None,
        disc="gdi",
        map_number=1,
        num_ais=1,
        agent_side=0,
        credits=5000,
        frame_skip=15,
        max_frames=15 * 60 * 60,
        group_radius=4,
        reward_fn=None,
        render_mode=None,
    ):
        self._native = _native.Native(
            _env_path(tdenv_lib or _native.default_library(), "TDGYM_TDENV"),
            _env_path(game_lib, "TDGYM_GAME_LIB"),
            _env_path(data, "TDGYM_DATA"),
            work_dir or tempfile.mkdtemp(prefix="tdgym-"),
            disc,
        )
        self.map_number = map_number
        self.num_ais = num_ais
        self.agent_side = agent_side
        self.credits = credits
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.group_radius = group_radius
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self._warned = set()

        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(0, 255, (len(PLANES), GRID, GRID), np.uint8),
                "scalars": spaces.Box(0, np.inf, (len(SCALARS),), np.float32),
                "buildables": spaces.Box(0, np.inf, (len(CATALOG), len(BUILDABLE_FEATURES)), np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([len(ACTION_KINDS), len(CATALOG), GRID, GRID, GRID, GRID])

    # Gymnasium interface.

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game_seed = int(self.np_random.integers(1, 2**31))
        self._native.reset(self.map_number, self.num_ais, game_seed, self.agent_side, self.credits)
        self._status = _native.RUNNING
        self._capture()
        return self._observation(), self._info(action_valid=True, game_seed=game_seed)

    def step(self, action):
        previous = self._scalars
        valid = self._apply(np.asarray(action, dtype=np.int64))
        self._status = self._native.step(self.frame_skip)
        self._capture()
        if self.reward_fn is not None:
            reward = float(self.reward_fn(previous, self._scalars, self._status))
        else:
            reward = {WON: 1.0, LOST: -1.0}.get(self._status, 0.0)
        terminated = self._status in (WON, LOST)
        truncated = not terminated and self._scalars["frame"] >= self.max_frames
        return self._observation(), reward, terminated, truncated, self._info(action_valid=valid)

    def render(self):
        if self.render_mode != "ansi":
            return None
        chars = np.full((self._native.height, self._native.width), " ", dtype="<U1")
        chars[self._explored.astype(bool)] = "."
        chars[self._tiberium > 0] = "~"
        for obj in self._objects:
            mark = {BUILDING: "B", UNIT: "V", AIRCRAFT: "A", INFANTRY: "i", TERRAIN: "t"}.get(int(obj["type"]), "?")
            if obj["relation"] == ENEMY:
                mark = mark.lower() if mark != "i" else "x"
            elif obj["relation"] == NEUTRAL:
                mark = "t"
            x0, y0 = int(obj["cell_x"]), int(obj["cell_y"])
            chars[y0 : y0 + int(obj["size_y"]), x0 : x0 + int(obj["size_x"])] = mark
        s = self._scalars
        header = "frame %d  credits %d  power %d/%d  kills %d/%d  losses %d/%d\n" % (
            s["frame"], s["credits"], s["power_produced"], s["power_drained"],
            s["units_killed"], s["buildings_killed"], s["units_lost"], s["buildings_lost"],
        )
        return header + "\n".join("".join(row) for row in chars)

    def close(self):
        self._native.close()
        super().close()

    # Helpers for agents.

    def placement_mask(self, item):
        """GRID x GRID bool array of cells where completed building 'item' (index or name) can go."""
        name = CATALOG[item] if isinstance(item, (int, np.integer)) else item
        mask = np.zeros((GRID, GRID), dtype=bool)
        mask[: self._native.height, : self._native.width] = self._native.placement(name).astype(bool)
        return mask

    @property
    def objects(self):
        """The observable objects as a structured array (see _native.TDObject)."""
        return self._objects

    # Internals.

    def _capture(self):
        n = self._native
        n.observe()
        self._scalars = n.scalars()
        self._objects = n.objects()
        self._explored = n.shroud()
        self._tiberium = n.tiberium()
        self._buildables = {}
        for b in n.buildables():
            name = b["name"].decode()
            if name in CATALOG_INDEX:
                self._buildables[name] = b
            elif name not in self._warned:
                self._warned.add(name)
                warnings.warn("tdgym: buildable %r is not in CATALOG and is ignored" % name)

    def _observation(self):
        h, w = self._native.height, self._native.width
        grid = np.zeros((len(PLANES), GRID, GRID), dtype=np.uint8)
        grid[PLANE["explored"], :h, :w] = self._explored
        grid[PLANE["tiberium"], :h, :w] = self._tiberium
        grid[PLANE["in_map"], :h, :w] = 1
        for obj in self._objects:
            x0, y0 = int(obj["cell_x"]), int(obj["cell_y"])
            if not (0 <= x0 < GRID and 0 <= y0 < GRID):
                continue
            x1, y1 = min(x0 + int(obj["size_x"]), GRID), min(y0 + int(obj["size_y"]), GRID)
            relation, kind = int(obj["relation"]), int(obj["type"])
            if relation == NEUTRAL or kind == TERRAIN:
                grid[PLANE["neutral"], y0:y1, x0:x1] = 1
                continue
            side = "own" if relation == SELF else "enemy"
            if kind == BUILDING:
                grid[PLANE[side + "_buildings"], y0:y1, x0:x1] = 1
            else:
                plane = PLANE[side + ("_infantry" if kind == INFANTRY else "_vehicles")]
                grid[plane, y0, x0] = min(int(grid[plane, y0, x0]) + 1, 255)
            health = int(255 * obj["strength"] / max(int(obj["max_strength"]), 1))
            plane = PLANE[side + "_health"]
            grid[plane, y0:y1, x0:x1] = np.maximum(grid[plane, y0:y1, x0:x1], health)

        scalars = np.array([self._scalars[name] for name in SCALARS], dtype=np.float32)

        buildables = np.zeros((len(CATALOG), len(BUILDABLE_FEATURES)), dtype=np.float32)
        for name, b in self._buildables.items():
            buildables[CATALOG_INDEX[name]] = (
                1, b["cost"], b["progress"], b["completed"], b["constructing"], b["on_hold"], b["busy"],
            )
        return {"map": grid, "scalars": scalars, "buildables": buildables}

    def _action_mask(self):
        build = np.zeros(len(CATALOG), dtype=bool)
        place = np.zeros(len(CATALOG), dtype=bool)
        cancel = np.zeros(len(CATALOG), dtype=bool)
        for name, b in self._buildables.items():
            i = CATALOG_INDEX[name]
            build[i] = not (b["busy"] or b["constructing"] or b["completed"])
            place[i] = bool(b["completed"]) and b["type"] == BUILDING
            cancel[i] = bool(b["constructing"] or b["completed"] or b["on_hold"])
        own = self._objects[self._objects["relation"] == SELF]
        has_mobile = bool(np.isin(own["type"], MOBILE_TYPES).any())
        has_building = bool((own["type"] == BUILDING).any())
        kind = np.array([True, build.any(), place.any(), has_mobile, has_mobile, has_building, cancel.any()])
        return {"kind": kind, "build": build, "place": place, "cancel": cancel}

    def _info(self, **extra):
        info = {"status": self._status, "scalars": dict(self._scalars), "action_mask": self._action_mask()}
        info.update(extra)
        return info

    def _group(self, x, y):
        own = self._objects[(self._objects["relation"] == SELF) & np.isin(self._objects["type"], MOBILE_TYPES)]
        near = (np.abs(own["cell_x"] - x) <= self.group_radius) & (np.abs(own["cell_y"] - y) <= self.group_radius)
        return own[near & (own["is_selectable"] != 0)]

    def _apply(self, action):
        kind, item, x, y, src_x, src_y = (int(a) for a in action)
        name = CATALOG[item]
        n = self._native
        if kind == NOOP:
            return True
        if kind == BUILD:
            return n.build(name)
        if kind == CANCEL:
            return n.cancel(name)
        if kind == PLACE:
            return n.place(name, x, y)
        if kind in (COMMAND, STOP):
            group = self._group(src_x, src_y)
            if len(group) == 0:
                return False
            return n.command(group, x, y) if kind == COMMAND else n.stop(group)
        if kind == SELL:
            own = self._objects[(self._objects["relation"] == SELF) & (self._objects["type"] == BUILDING)]
            covers = (
                (own["cell_x"] <= x) & (x < own["cell_x"] + own["size_x"])
                & (own["cell_y"] <= y) & (y < own["cell_y"] + own["size_y"])
            )
            return bool(covers.any()) and n.sell(int(own[covers][0]["id"]))
        return False
