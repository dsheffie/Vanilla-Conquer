"""Play MacroEnv episodes against the built-in AI and report results.

    python evaluate.py --policy random|scripted|CHECKPOINT [--episodes N] [--envs N]

random picks uniformly among valid actions; scripted follows a fixed build order and
attacks with a large enough army; anything else is a checkpoint saved by ppo.py.
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tdgym.macro import ACTION_INDEX, ACTIONS, make_macro_env  # noqa: E402

BUILD_ORDER = ["build_power", "build_refinery", "build_barracks", "build_power", "build_factory", "build_refinery"]


def scripted(obs, state):
    """A fixed plan: deploy, follow BUILD_ORDER, then tanks; attack with 8 or more units."""
    mask = obs["mask"]
    if mask[ACTION_INDEX["deploy_mcv"]]:
        return ACTION_INDEX["deploy_mcv"]
    step = state.setdefault("order", 0)
    if step < len(BUILD_ORDER) and mask[ACTION_INDEX[BUILD_ORDER[step]]]:
        state["order"] += 1
        return ACTION_INDEX[BUILD_ORDER[step]]
    state["tick"] = state.get("tick", 0) + 1
    if state["tick"] % 20 == 0 and mask[ACTION_INDEX["harvest"]]:
        return ACTION_INDEX["harvest"]
    if state["tick"] % 30 == 0 and mask[ACTION_INDEX["attack"]]:
        return ACTION_INDEX["attack"]
    for name in ("train_tank", "train_rocket_soldier", "train_rifleman"):
        if mask[ACTION_INDEX[name]]:
            return ACTION_INDEX[name]
    return ACTION_INDEX["noop"]


def random_policy(obs, state, rng=np.random.default_rng(0)):
    return int(rng.choice(np.nonzero(obs["mask"])[0]))


def model_policy(path, device="cpu"):
    import torch

    from policy import Policy

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    net = Policy(**checkpoint["shape"]).to(device)
    net.load_state_dict(checkpoint["model"])
    net.eval()

    def act(obs, state):
        with torch.no_grad():
            logits, _ = net(
                torch.as_tensor(obs["map"][None], device=device),
                torch.as_tensor(obs["features"][None], device=device),
                torch.as_tensor(obs["mask"][None], device=device),
            )
            return int(torch.distributions.Categorical(logits=logits).sample())

    return act


def evaluate(policy, episodes, seed, env_kwargs):
    env = make_macro_env(**env_kwargs)
    results = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        state, done, total, counts = {}, False, 0.0, np.zeros(len(ACTIONS), dtype=int)
        while not done:
            action = policy(obs, state)
            counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
        results.append(
            {
                "won": info["won"],
                "lost": info["lost"],
                "timed_out": info["timed_out"],
                "return": total,
                "minutes": info["scalars"]["frame"] / 900,
                "harvested": info["scalars"]["harvested_credits"],
                "kills": info["scalars"]["units_killed"] + info["scalars"]["buildings_killed"],
                "losses": info["scalars"]["units_lost"] + info["scalars"]["buildings_lost"],
                "actions": counts,
            }
        )
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--map", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--show-actions", action="store_true")
    args = parser.parse_args()

    if args.policy == "random":
        policy = random_policy
    elif args.policy == "scripted":
        policy = scripted
    else:
        policy = model_policy(args.policy)
    results = evaluate(policy, args.episodes, args.seed, {"map_number": args.map, "max_minutes": args.max_minutes})
    for r in results:
        outcome = "won" if r["won"] else "lost" if r["lost"] else "time limit"
        print(
            "%-10s %5.1f min  return %6.2f  harvested %6d  kills %3d  losses %3d"
            % (outcome, r["minutes"], r["return"], r["harvested"], r["kills"], r["losses"])
        )
        if args.show_actions:
            top = np.argsort(r["actions"])[::-1][:8]
            print("   ", ", ".join("%s %d" % (ACTIONS[i], r["actions"][i]) for i in top if r["actions"][i]))
    wins = sum(r["won"] for r in results)
    print("win rate %d/%d, mean return %.2f" % (wins, len(results), np.mean([r["return"] for r in results])))


if __name__ == "__main__":
    main()
