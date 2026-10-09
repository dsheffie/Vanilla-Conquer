"""A scripted agent for tdgym: builds a small base, trains riflemen and sends them out.

Mostly a usage example, and a check that the environment's actions work end to end.
It also asserts that the observation never reveals other houses' objects in cells the
agent hasn't explored.

    TDGYM_GAME_LIB=... TDGYM_DATA=... TDGYM_TDENV=... python scripted_agent.py [--render]
"""

import argparse

import numpy as np

import tdgym
from tdgym import BUILD, CATALOG, COMMAND, NOOP, PLACE
from tdgym._native import SELF

BASE = ["NUKE", "PROC", "PYLE"]


def action(kind, item="NUKE", x=0, y=0, src_x=0, src_y=0):
    return np.array([kind, CATALOG.index(item), x, y, src_x, src_y])


def check_fog(env):
    explored = env._explored
    others = env.objects[env.objects["relation"] != SELF]
    for obj in others:
        x0, y0 = obj["cell_x"], obj["cell_y"]
        cells = explored[y0 : y0 + obj["size_y"], x0 : x0 + obj["size_x"]]
        assert cells.any(), "observed %s in unexplored cells" % obj["name"]


def choose(env, obs):
    own = env.objects[env.objects["relation"] == SELF]
    names = [o.decode() for o in own["name"]]
    buildables = obs["buildables"]

    # Deploy the MCV where it stands.
    if "MCV" in names and "FACT" not in names:
        mcv = own[names.index("MCV")]
        return action(COMMAND, x=mcv["cell_x"], y=mcv["cell_y"], src_x=mcv["cell_x"], src_y=mcv["cell_y"])

    # Place whatever building is finished, at the first legal spot.
    for item in BASE:
        if buildables[CATALOG.index(item), 3]:  # completed
            ys, xs = np.nonzero(env.placement_mask(item))
            if len(xs):
                return action(PLACE, item, xs[0], ys[0])

    # Build the next missing base building, then riflemen.
    for item in BASE:
        if item not in names:
            available, busy = buildables[CATALOG.index(item), 0], buildables[CATALOG.index(item), 6]
            constructing = buildables[CATALOG.index(item), 4]
            if available and not busy and not constructing:
                return action(BUILD, item)
            return action(NOOP)
    if buildables[CATALOG.index("E1"), 0] and not buildables[CATALOG.index("E1"), 4]:
        return action(BUILD, "E1")

    # Send idle riflemen to the middle of the map once there are a few.
    riflemen = own[own["name"] == b"E1"]
    if len(riflemen) >= 5:
        r = riflemen[0]
        return action(COMMAND, x=32, y=25, src_x=r["cell_x"], src_y=r["cell_y"])
    return action(NOOP)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=10, help="game minutes to play")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = tdgym.TiberianDawnEnv(render_mode="ansi", max_frames=int(args.minutes * 60 * 15))
    obs, info = env.reset(seed=args.seed)
    done = False
    steps = applied = 0
    while not done:
        act = choose(env, obs)
        obs, reward, terminated, truncated, info = env.step(act)
        check_fog(env)
        steps += 1
        applied += info["action_valid"] and act[0] != NOOP
        done = terminated or truncated
        if args.render and steps % 40 == 0:
            print(env.render(), "\n")

    own = env.objects[env.objects["relation"] == SELF]
    counts = {}
    for name in own["name"]:
        counts[name.decode()] = counts.get(name.decode(), 0) + 1
    print("steps %d, actions applied %d, reward %s, status %s" % (steps, applied, reward, info["status"]))
    print("scalars", info["scalars"])
    print("own objects", counts)
    print("fog check passed every step")


if __name__ == "__main__":
    main()
