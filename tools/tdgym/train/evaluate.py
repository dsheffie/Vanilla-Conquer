"""Play MacroEnv episodes against the built-in AI and report results.

    python evaluate.py --policy random|scripted|CHECKPOINT [--episodes N]
    python evaluate.py --policy CHECKPOINT --episodes 1 --video game.mp4 [--video-speed 8] [--video-scale 0.5]
    python evaluate.py --policy CHECKPOINT --episodes 18 --maps 9 --side random

--maps and --side work as in ppo.py: each episode picks its map and side at random.

random picks uniformly among valid actions; scripted follows a fixed build order and
attacks with a large enough army; anything else is a checkpoint saved by ppo.py.

--video records each episode with the real game graphics, as the agent sees them (its own
shroud), to an MP4 through ffmpeg; with several episodes, game.mp4 becomes game-1.mp4 and
so on. Drawing the screen slows the game down, so recording takes longer than evaluating.
"""

import argparse
import os
import shutil
import subprocess
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tdgym.env import SIDES, parse_maps  # noqa: E402
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


class VideoWriter:
    """Streams RGB frames to ffmpeg as an H.264 MP4, starting it at the first frame."""

    def __init__(self, path, fps, scale):
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--video needs ffmpeg on the PATH")
        self.path, self.fps, self.scale = path, fps, scale
        self.process = None
        self.frames = 0

    def write(self, frame):
        if self.process is None:
            h, w, _ = frame.shape
            # H.264 in yuv420p needs even dimensions.
            scale = "scale=trunc(iw*%g/2)*2:trunc(ih*%g/2)*2:flags=neighbor" % (self.scale, self.scale)
            self.process = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (w, h),
                 "-r", str(self.fps), "-i", "-", "-vf", scale, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "20", self.path],
                stdin=subprocess.PIPE,
            )
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            if self.process.wait() != 0:
                raise SystemExit("ffmpeg failed writing %s" % self.path)


def evaluate(policy, episodes, seed, env_kwargs, video=None):
    """video: dict(path, speed, scale, fps) to record each episode."""
    writer = None
    if video:
        # Capture every 'every' game frames and play back at 'fps': speed x real time at 15 fps.
        every = max(1, round(15 * video["speed"] / video["fps"]))
        env_kwargs = dict(env_kwargs, render_mode="rgb_array", frame_every=every)
    env = make_macro_env(**env_kwargs)
    results = []
    for episode in range(episodes):
        if video:
            stem, ext = os.path.splitext(video["path"])
            path = video["path"] if episodes == 1 else "%s-%d%s" % (stem, episode + 1, ext)
            writer = VideoWriter(path, video["fps"], video["scale"])
            env.unwrapped.on_frame = writer.write
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
                "map": info["map_number"],
                "side": SIDES[info["agent_side"]],
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
        if writer is not None:
            writer.close()
            results[-1]["video"] = (writer.path, writer.frames)
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--policy", default="scripted")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--maps", type=parse_maps, default="1", help="maps to play, e.g. 1, 1-8, 1,3,5 or all")
    parser.add_argument("--side", default="gdi", choices=SIDES + ("random",))
    parser.add_argument("--ai-difficulty", default="normal", choices=["easy", "normal", "hard"])
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--show-actions", action="store_true")
    parser.add_argument("--video", help="record each episode to this MP4 (needs ffmpeg)")
    parser.add_argument("--video-speed", type=float, default=8, help="playback speed vs real time")
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-scale", type=float, default=0.5, help="size vs the full 24 px per cell map")
    args = parser.parse_args()

    if args.policy == "random":
        policy = random_policy
    elif args.policy == "scripted":
        policy = scripted
    else:
        policy = model_policy(args.policy)
    video = None
    if args.video:
        video = {"path": args.video, "speed": args.video_speed, "fps": args.video_fps, "scale": args.video_scale}
    results = evaluate(
        policy, args.episodes, args.seed, {"map_number": args.maps, "agent_side": args.side, "max_minutes": args.max_minutes, "ai_difficulty": args.ai_difficulty}, video
    )
    for r in results:
        outcome = "won" if r["won"] else "lost" if r["lost"] else "time limit"
        print(
            "map %d %-3s  %-10s %5.1f min  return %6.2f  harvested %6d  kills %3d  losses %3d"
            % (r["map"], r["side"], outcome, r["minutes"], r["return"], r["harvested"], r["kills"], r["losses"])
        )
        if "video" in r:
            print("    video %s: %d frames, %.0f s" % (r["video"][0], r["video"][1], r["video"][1] / args.video_fps))
        if args.show_actions:
            top = np.argsort(r["actions"])[::-1][:8]
            print("   ", ", ".join("%s %d" % (ACTIONS[i], r["actions"][i]) for i in top if r["actions"][i]))
    wins = sum(r["won"] for r in results)
    print("win rate %d/%d, mean return %.2f" % (wins, len(results), np.mean([r["return"] for r in results])))


if __name__ == "__main__":
    main()
