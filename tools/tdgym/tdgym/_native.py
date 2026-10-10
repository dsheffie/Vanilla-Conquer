"""ctypes binding for libtdenv (see ../tdenv.h)."""

import ctypes
import os
import sys

import numpy as np

NAME_LENGTH = 16

RUNNING, WON, LOST, ERROR = 0, 1, 2, -1
SELF, ALLY, ENEMY, NEUTRAL = 0, 1, 2, 3
INFANTRY, UNIT, AIRCRAFT, BUILDING, TERRAIN = 1, 2, 3, 4, 5

# Room for every object a game can hold.
MAX_OBJECTS = 2048
MAX_BUILDABLES = 128


class TDObject(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("type", ctypes.c_int),
        ("name", ctypes.c_char * NAME_LENGTH),
        ("relation", ctypes.c_int),
        ("house", ctypes.c_int),
        ("cell_x", ctypes.c_int),
        ("cell_y", ctypes.c_int),
        ("size_x", ctypes.c_int),
        ("size_y", ctypes.c_int),
        ("pixel_x", ctypes.c_int),
        ("pixel_y", ctypes.c_int),
        ("strength", ctypes.c_int),
        ("max_strength", ctypes.c_int),
        ("can_move", ctypes.c_int),
        ("can_deploy", ctypes.c_int),
        ("can_harvest", ctypes.c_int),
        ("is_factory", ctypes.c_int),
        ("is_selectable", ctypes.c_int),
        ("pips", ctypes.c_int),
        ("max_pips", ctypes.c_int),
    ]


class TDBuildable(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * NAME_LENGTH),
        ("type", ctypes.c_int),
        ("cost", ctypes.c_int),
        ("build_time", ctypes.c_int),
        ("progress", ctypes.c_float),
        ("completed", ctypes.c_int),
        ("constructing", ctypes.c_int),
        ("on_hold", ctypes.c_int),
        ("busy", ctypes.c_int),
    ]


class TDScalars(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int)
        for name in (
            "frame",
            "credits",
            "tiberium",
            "max_tiberium",
            "power_produced",
            "power_drained",
            "units_killed",
            "buildings_killed",
            "units_lost",
            "buildings_lost",
            "harvested_credits",
            "defeated",
        )
    ]


def _numpy_dtype(struct):
    """numpy view of a structure, with char arrays as byte strings rather than arrays of bytes."""
    names, formats, offsets = [], [], []
    for name, ctype in struct._fields_:
        names.append(name)
        offsets.append(getattr(struct, name).offset)
        if issubclass(ctype, ctypes.Array) and ctype._type_ is ctypes.c_char:
            formats.append("S%d" % ctype._length_)
        else:
            formats.append(np.dtype(ctype))
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": ctypes.sizeof(struct)})


# numpy views of the structures, so object lists convert without Python loops.
OBJECT_DTYPE = _numpy_dtype(TDObject)
BUILDABLE_DTYPE = _numpy_dtype(TDBuildable)


def default_library():
    """libtdenv next to this package, as put there by the build, or $TDGYM_TDENV."""
    if "TDGYM_TDENV" in os.environ:
        return os.environ["TDGYM_TDENV"]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "libtdenv" + suffix)


class TDEnvError(RuntimeError):
    pass


class Native:
    """The C environment. The game is a process-wide singleton, so only one can exist."""

    _instance = None

    def __init__(self, tdenv_path, game_lib, data, work, disc="gdi"):
        if Native._instance is not None:
            raise TDEnvError(
                "only one Tiberian Dawn environment per process; use a vector env with subprocesses"
            )
        self._lib = ctypes.CDLL(tdenv_path)
        self._declare()
        Native._instance = self
        self._check(self._lib.tdenv_open(*(s.encode() for s in (game_lib, data, disc, work))))
        self._objects = (TDObject * MAX_OBJECTS)()
        self._buildables = (TDBuildable * MAX_BUILDABLES)()
        self._scalars = TDScalars()
        self.width = self.height = 0

    def close(self):
        """Let another environment be created in this process."""
        if Native._instance is self:
            Native._instance = None

    def _declare(self):
        lib = self._lib
        c_int, c_str, c_uint = ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
        int_p = ctypes.POINTER(c_int)
        byte_p = ctypes.POINTER(ctypes.c_ubyte)
        signatures = {
            "tdenv_last_error": ([], c_str),
            "tdenv_open": ([c_str, c_str, c_str, c_str], c_int),
            "tdenv_reset": ([c_int, c_int, c_uint, c_int, c_int, c_int], c_int),
            "tdenv_grid_size": ([int_p, int_p], c_int),
            "tdenv_set_rendering": ([c_int], c_int),
            "tdenv_frame": ([byte_p, c_int, int_p, int_p], c_int),
            "tdenv_step": ([c_int], c_int),
            "tdenv_observe": ([], c_int),
            "tdenv_scalars": ([ctypes.POINTER(TDScalars)], c_int),
            "tdenv_objects": ([ctypes.POINTER(TDObject), c_int], c_int),
            "tdenv_shroud": ([byte_p], c_int),
            "tdenv_tiberium": ([byte_p], c_int),
            "tdenv_buildables": ([ctypes.POINTER(TDBuildable), c_int], c_int),
            "tdenv_build": ([c_str], c_int),
            "tdenv_cancel": ([c_str], c_int),
            "tdenv_place": ([c_str, c_int, c_int], c_int),
            "tdenv_placement": ([c_str, byte_p], c_int),
            "tdenv_command": ([int_p, int_p, c_int, c_int, c_int], c_int),
            "tdenv_stop": ([int_p, int_p, c_int], c_int),
            "tdenv_hunt": ([int_p, int_p, c_int], c_int),
            "tdenv_sell": ([c_int], c_int),
        }
        for name, (args, result) in signatures.items():
            func = getattr(lib, name)
            func.argtypes = args
            func.restype = result

    def _check(self, result):
        if result == ERROR:
            raise TDEnvError(self._lib.tdenv_last_error().decode(errors="replace"))
        return result

    def reset(self, map_number, num_ais, seed, agent_side, credits, ai_difficulty=1):
        self._check(self._lib.tdenv_reset(map_number, num_ais, seed, agent_side, credits, ai_difficulty))
        w, h = ctypes.c_int(), ctypes.c_int()
        self._check(self._lib.tdenv_grid_size(ctypes.byref(w), ctypes.byref(h)))
        self.width, self.height = w.value, h.value

    def step(self, frames):
        return self._check(self._lib.tdenv_step(frames))

    def observe(self):
        self._check(self._lib.tdenv_observe())

    def scalars(self):
        self._lib.tdenv_scalars(ctypes.byref(self._scalars))
        return {name: getattr(self._scalars, name) for name, _ in TDScalars._fields_}

    def objects(self):
        """Structured numpy array of the observable objects (see TDObject)."""
        count = self._lib.tdenv_objects(self._objects, MAX_OBJECTS)
        return np.frombuffer(self._objects, dtype=OBJECT_DTYPE, count=count).copy()

    def buildables(self):
        count = self._lib.tdenv_buildables(self._buildables, MAX_BUILDABLES)
        return np.frombuffer(self._buildables, dtype=BUILDABLE_DTYPE, count=count).copy()

    def _grid(self, func):
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        func(out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        return out

    def set_rendering(self, enabled):
        self._check(self._lib.tdenv_set_rendering(int(enabled)))

    def frame(self):
        """The game screen as an (height, width, 3) uint8 RGB array; needs set_rendering(True)."""
        out = np.empty(self.height * self.width * 24 * 24 * 3, dtype=np.uint8)
        w, h = ctypes.c_int(), ctypes.c_int()
        self._check(
            self._lib.tdenv_frame(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), out.size, ctypes.byref(w), ctypes.byref(h)
            )
        )
        return out[: w.value * h.value * 3].reshape(h.value, w.value, 3)

    def shroud(self):
        """1 where the agent has explored, by [y, x]."""
        return self._grid(self._lib.tdenv_shroud)

    def tiberium(self):
        """Tiberium density 0-12 by [y, x], 0 where unexplored."""
        return self._grid(self._lib.tdenv_tiberium)

    def build(self, name):
        return bool(self._lib.tdenv_build(name.encode()))

    def cancel(self, name):
        return bool(self._lib.tdenv_cancel(name.encode()))

    def place(self, name, x, y):
        return bool(self._lib.tdenv_place(name.encode(), x, y))

    def placement(self, name):
        """Cells by [y, x] where a completed building can have its top left corner."""
        return self._grid(lambda out: self._lib.tdenv_placement(name.encode(), out))

    @staticmethod
    def _id_arrays(objects):
        types = np.ascontiguousarray(objects["type"], dtype=np.intc)
        ids = np.ascontiguousarray(objects["id"], dtype=np.intc)
        int_p = ctypes.POINTER(ctypes.c_int)
        return types, ids, types.ctypes.data_as(int_p), ids.ctypes.data_as(int_p)

    def command(self, objects, x, y):
        """Command own objects (rows of objects()) at cell (x, y)."""
        types, ids, tp, ip = self._id_arrays(objects)
        return bool(self._lib.tdenv_command(tp, ip, len(types), x, y))

    def stop(self, objects):
        types, ids, tp, ip = self._id_arrays(objects)
        return bool(self._lib.tdenv_stop(tp, ip, len(types)))

    def hunt(self, objects):
        """Send own objects (rows of objects()) to search and destroy."""
        types, ids, tp, ip = self._id_arrays(objects)
        return bool(self._lib.tdenv_hunt(tp, ip, len(types)))

    def sell(self, building_id):
        return bool(self._lib.tdenv_sell(building_id))
