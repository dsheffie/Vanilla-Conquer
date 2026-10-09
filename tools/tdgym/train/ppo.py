"""PPO with action masking for MacroEnv, one game per subprocess.

    python ppo.py --run-dir runs/first [--envs 12] [--total-steps 2000000] [--device mps]
    python ppo.py --benchmark [--total-steps 98304] [--envs 12] [--run-dir DIR]
    python ppo.py --run-dir runs/curriculum --curriculum [--init-from runs/first/latest.pt]

Follows the single-file CleanRL style. Writes metrics.csv and checkpoints (latest.pt and
every --save-every updates) to the run directory; evaluate.py plays a checkpoint.

--ai-difficulty fixes the built-in AI's difficulty; --curriculum instead starts against the
easy AI and moves up a level once the win rate there reaches --curriculum-threshold over
the last --curriculum-window games at that level, keeping --curriculum-floor of games on
the easier levels (see curriculum.py). Each env's next game is assigned as its current one
starts. Win rates are logged per difficulty.

--benchmark trains for --total-steps with fixed seeds and no checkpoints, then reports the
wall time and where it went (game processes, policy inference, PPO updates), with the
machine and settings, to stdout and run-dir/benchmark.json if --run-dir is given.
Needs $TDGYM_GAME_LIB, $TDGYM_DATA and $TDGYM_TDENV.

torch is only imported inside functions: on macOS the game subprocesses start with
'spawn', which re-imports this file, and a top-level import would load torch into every
one of them (about 280 MB each).
"""

import argparse
import collections
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tdgym.env import GRID, PLANES  # noqa: E402
from tdgym.env import AI_DIFFICULTIES, ai_difficulty_level  # noqa: E402
from tdgym.macro import ACTIONS, FEATURES, make_macro_env  # noqa: E402

from curriculum import Curriculum  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", help="where metrics and checkpoints go; required unless --benchmark")
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--steps", type=int, default=128, help="rollout length per env")
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--device", default="auto", help="auto, mps, cuda or cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--map", type=int, default=1)
    p.add_argument("--ai-difficulty", default="normal", choices=AI_DIFFICULTIES)
    p.add_argument("--curriculum", action="store_true", help="adapt the AI difficulty, see above")
    p.add_argument("--curriculum-threshold", type=float, default=0.5)
    p.add_argument("--curriculum-window", type=int, default=100)
    p.add_argument("--curriculum-floor", type=float, default=0.2)
    p.add_argument("--max-minutes", type=float, default=30)
    p.add_argument("--decision-frames", type=int, default=30)
    p.add_argument("--save-every", type=int, default=50, help="updates between numbered checkpoints")
    p.add_argument("--resume", help="checkpoint to continue from: weights, optimizer, counters, curriculum")
    p.add_argument("--init-from", help="checkpoint to take the starting weights from, for a new run")
    p.add_argument("--benchmark", action="store_true", help="time a fixed amount of training, see above")
    args = p.parse_args()
    if not args.run_dir and not args.benchmark:
        p.error("--run-dir is required unless --benchmark")
    return args


def make_env(args):
    def thunk():
        return make_macro_env(
            decision_frames=args.decision_frames,
            max_minutes=args.max_minutes,
            map_number=args.map,
            ai_difficulty=args.ai_difficulty,
        )

    return thunk


def env_item(stacked, i):
    """Entry i of a vector env's final_obs or final_info: an array of per-env dicts, or a
    dict of per-env arrays, depending on the Gymnasium version."""
    if isinstance(stacked, dict):
        return {key: env_item(value, i) for key, value in stacked.items()}
    return stacked[i]


def to_tensors(obs, device):
    import torch

    return (
        torch.as_tensor(obs["map"], device=device),
        torch.as_tensor(obs["features"], device=device),
        torch.as_tensor(obs["mask"], device=device),
    )


def main():
    import torch
    from policy import Policy

    t_main = time.time()
    args = parse_args()
    benchmark_dir = args.run_dir
    if args.benchmark and not args.run_dir:
        args.run_dir = tempfile.mkdtemp(prefix="tdgym-benchmark-")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args) for _ in range(args.envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    shape = {"planes": len(PLANES), "grid": GRID, "features": FEATURES, "actions": len(ACTIONS)}
    net = Policy(**shape).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    global_step = update = 0
    if args.curriculum:
        curriculum = Curriculum(args.curriculum_threshold, args.curriculum_window, args.curriculum_floor, seed=args.seed)
    else:
        # A fixed difficulty: a curriculum that never moves, for the per-difficulty statistics.
        curriculum = Curriculum(float("inf"), 100, 0.0, level=ai_difficulty_level(args.ai_difficulty))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step, update = checkpoint["global_step"], checkpoint["update"]
        if args.curriculum and "curriculum" in checkpoint:
            curriculum.load(checkpoint["curriculum"])
    elif args.init_from:
        net.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["model"])

    n, t = args.envs, args.steps
    maps = torch.zeros((t, n, len(PLANES), GRID, GRID), dtype=torch.uint8, device=device)
    features = torch.zeros((t, n, FEATURES), device=device)
    masks = torch.zeros((t, n, len(ACTIONS)), dtype=torch.int8, device=device)
    actions = torch.zeros((t, n), dtype=torch.long, device=device)
    logprobs = torch.zeros((t, n), device=device)
    rewards = torch.zeros((t, n), device=device)
    dones = torch.zeros((t, n), device=device)
    values = torch.zeros((t, n), device=device)

    recent = collections.deque(maxlen=100)
    metrics_path = os.path.join(args.run_dir, "metrics.csv")
    new_file = not os.path.exists(metrics_path)
    metrics_file = open(metrics_path, "a", newline="")
    metrics = csv.writer(metrics_file)
    if new_file:
        metrics.writerow(
            ["update", "global_step", "sps", "episodes", "win_rate", "loss_rate", "timeout_rate", "return", "minutes",
             "harvested", "kills", "losses", "policy_loss", "value_loss", "entropy", "approx_kl",
             "level", "win_easy", "win_normal", "win_hard"]
        )

    # The difficulty each env's next game is played at.
    next_difficulty = [curriculum.sample() if args.curriculum else curriculum.level for _ in range(n)]
    envs.set_attr("ai_difficulty", next_difficulty)
    obs, _ = envs.reset(seed=args.seed + 1000 * update)
    if args.curriculum:
        next_difficulty = [curriculum.sample() for _ in range(n)]
        envs.set_attr("ai_difficulty", next_difficulty)
    next_done = torch.zeros(n, device=device)
    start = time.time()
    setup_s = start - t_main
    start_step = global_step
    batch = n * t
    minibatch = batch // args.minibatches
    timing = collections.Counter()  # Seconds spent in each phase.

    def sync():
        if args.benchmark and device.type == "cuda":
            torch.cuda.synchronize()
        elif args.benchmark and device.type == "mps":
            torch.mps.synchronize()

    while global_step < args.total_steps:
        update += 1
        net.eval()
        for step in range(t):
            t0 = time.time()
            m, f, k = to_tensors(obs, device)
            maps[step], features[step], masks[step], dones[step] = m, f, k, next_done
            with torch.no_grad():
                logits, value = net(m, f, k)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            actions[step], logprobs[step], values[step] = action, dist.log_prob(action), value
            action_np = action.cpu().numpy()  # Waits for the device.
            t1 = time.time()

            obs, reward, terminated, truncated, info = envs.step(action_np)
            t2 = time.time()
            timing["inference"] += t1 - t0
            timing["env_step"] += t2 - t1
            reward = np.asarray(reward, dtype=np.float32)
            # Episodes cut off by the time limit continue in principle: bootstrap from the
            # value of their final observation.
            if truncated.any() and "final_obs" in info:
                for i in np.nonzero(truncated & ~terminated)[0]:
                    final = {key: np.asarray(v)[None] for key, v in env_item(info["final_obs"], i).items()}
                    with torch.no_grad():
                        reward[i] += args.gamma * float(net(*to_tensors(final, device))[1])
            rewards[step] = torch.as_tensor(reward, device=device)
            done = terminated | truncated
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            global_step += n

            if done.any() and "final_info" in info:
                for i in np.nonzero(done)[0]:
                    final = env_item(info["final_info"], i)
                    scalars = final["scalars"]
                    difficulty = int(final["ai_difficulty"])
                    if curriculum.record(difficulty, final["won"]):
                        print("curriculum: moving up to the %s AI" % AI_DIFFICULTIES[curriculum.level], flush=True)
                    if args.curriculum:
                        # The game just started in env i was assigned before; pick the one after it.
                        next_difficulty[i] = curriculum.sample()
                    recent.append(
                        {
                            "ai_difficulty": difficulty,
                            "won": bool(final["won"]),
                            "lost": bool(final["lost"]),
                            "timed_out": bool(final["timed_out"]),
                            "minutes": scalars["frame"] / 900,
                            "harvested": scalars["harvested_credits"],
                            "kills": scalars["units_killed"] + scalars["buildings_killed"],
                            "losses": scalars["units_lost"] + scalars["buildings_lost"],
                        }
                    )
            episode_returns_add(recent, done, rewards[step])
            if args.curriculum and done.any():
                envs.set_attr("ai_difficulty", next_difficulty)
            timing["bookkeeping"] += time.time() - t2

        t_update = time.time()
        # Generalised advantage estimation.
        with torch.no_grad():
            next_value = net(*to_tensors(obs, device))[1]
            advantages = torch.zeros_like(rewards)
            last = 0
            for step in reversed(range(t)):
                if step == t - 1:
                    not_done, next_v = 1.0 - next_done, next_value
                else:
                    not_done, next_v = 1.0 - dones[step + 1], values[step + 1]
                delta = rewards[step] + args.gamma * next_v * not_done - values[step]
                last = delta + args.gamma * args.gae_lambda * not_done * last
                advantages[step] = last
            returns = advantages + values

        flat = lambda x: x.reshape((batch,) + x.shape[2:])  # noqa: E731
        b_maps, b_features, b_masks = flat(maps), flat(features), flat(masks)
        b_actions, b_logprobs, b_adv, b_returns = flat(actions), flat(logprobs), flat(advantages), flat(returns)

        net.train()
        for _ in range(args.epochs):
            order = torch.randperm(batch, device=device)
            for s in range(0, batch, minibatch):
                idx = order[s : s + minibatch]
                logits, value = net(b_maps[idx], b_features[idx], b_masks[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_ratio = dist.log_prob(b_actions[idx]) - b_logprobs[idx]
                ratio = log_ratio.exp()
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                policy_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                value_loss = 0.5 * (value - b_returns[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + args.value_coef * value_loss - args.entropy * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
        sync()
        timing["update"] += time.time() - t_update
        t_log = time.time()

        sps = (global_step - start_step) / (time.time() - start)
        episodes = [e for e in recent if "won" in e]
        def mean(key):  # noqa: E306
            return float(np.mean([e[key] for e in episodes])) if episodes else float("nan")
        returns_recent = [e["return"] for e in recent if "return" in e]
        row = [
            update, global_step, round(sps), len(episodes), mean("won"), mean("lost"), mean("timed_out"),
            float(np.mean(returns_recent)) if returns_recent else float("nan"), mean("minutes"),
            mean("harvested"), mean("kills"), mean("losses"), policy_loss.item(), value_loss.item(),
            entropy.item(), approx_kl.item(),
            AI_DIFFICULTIES[curriculum.level], *(curriculum.win_rate(d) for d in range(len(AI_DIFFICULTIES))),
        ]
        metrics.writerow(row)
        metrics_file.flush()
        print(
            "update %4d  step %8d  %5d sps  episodes %3d  win %.2f  loss %.2f  timeout %.2f  return %6.2f  "
            "minutes %4.1f  harvested %6.0f  kills %4.1f  losses %4.1f  entropy %.2f  "
            "AI %s  win e/n/h %.2f/%.2f/%.2f"
            % tuple(row[:12] + [row[14]] + row[16:]),
            flush=True,
        )

        if not args.benchmark:
            checkpoint = {
                "model": net.state_dict(), "optimizer": optimizer.state_dict(), "shape": shape,
                "args": vars(args), "update": update, "global_step": global_step,
                "curriculum": curriculum.state(),
            }
            torch.save(checkpoint, os.path.join(args.run_dir, "latest.pt"))
            if update % args.save_every == 0:
                torch.save(checkpoint, os.path.join(args.run_dir, "update%05d.pt" % update))
        timing["logging"] += time.time() - t_log

    wall_s = time.time() - start
    env_memory = env_process_memory(envs) if args.benchmark else None
    envs.close()
    if args.benchmark:
        report_benchmark(args, benchmark_dir, device, wall_s, setup_s, global_step - start_step, update, timing, env_memory)
        if not benchmark_dir:
            import shutil

            shutil.rmtree(args.run_dir, ignore_errors=True)  # The temporary one made above.


def env_process_memory(envs):
    """Memory of each live env process in MB: proportional set size on Linux, so pages
    shared with the trainer after fork are split rather than counted in full, and resident
    size elsewhere."""
    sizes = []
    for process in getattr(envs, "processes", []):
        try:
            if sys.platform.startswith("linux"):
                with open("/proc/%d/smaps_rollup" % process.pid) as f:
                    kb = next(int(line.split()[1]) for line in f if line.startswith("Pss:"))
            else:
                kb = int(subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True).stdout)
            sizes.append(kb // 1024)
        except (OSError, ValueError, StopIteration):
            pass
    return sizes


def machine_info(device):
    import torch

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    try:
        if sys.platform == "darwin":
            sysctl = lambda key: subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()  # noqa: E731
            info["cpu"] = sysctl("machdep.cpu.brand_string")
            info["memory_gb"] = round(int(sysctl("hw.memsize")) / 2**30, 1)
        else:
            with open("/proc/cpuinfo") as f:
                names = [line.split(":", 1)[1].strip() for line in f if line.lower().startswith("model name")]
            info["cpu"] = names[0] if names else platform.processor()
            with open("/proc/meminfo") as f:
                info["memory_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        pass
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        info["gpu"] = "Apple GPU (MPS)"
    return info


def report_benchmark(args, out_dir, device, wall_s, setup_s, decisions, updates, timing, env_memory):
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    scale = 1 if sys.platform == "darwin" else 1024
    peak = lambda who: round(resource.getrusage(who).ru_maxrss * scale / 2**20)  # noqa: E731
    env_memory = env_memory or [0]
    result = {
        "machine": machine_info(device),
        "settings": {
            "device": str(device), "envs": args.envs, "steps": args.steps, "total_steps": args.total_steps,
            "decision_frames": args.decision_frames, "epochs": args.epochs, "minibatches": args.minibatches,
            "map": args.map, "seed": args.seed, "ai_difficulty": args.ai_difficulty,
        },
        "results": {
            "wall_s": round(wall_s, 1),
            "setup_s": round(setup_s, 1),
            "decisions": decisions,
            "updates": updates,
            "decisions_per_s": round(decisions / wall_s),
            "game_frames_per_s": round(decisions * args.decision_frames / wall_s),
            "seconds": {k: round(v, 1) for k, v in timing.items()},
            "share": {k: round(v / wall_s, 3) for k, v in timing.items()},
            "update_ms": round(1000 * timing["update"] / max(updates, 1)),
            "memory_mb": {
                "trainer_peak": peak(resource.RUSAGE_SELF),
                "env_process_largest": max(env_memory),
                "env_processes_total": sum(env_memory),
            },
        },
    }
    r = result["results"]
    print("\nbenchmark: %d decisions in %.1f s (+%.1f s setup) on %s, %d envs"
          % (decisions, wall_s, setup_s, device, args.envs))
    print("  %d decisions/s, %d game frames/s, %d ms per PPO update"
          % (r["decisions_per_s"], r["game_frames_per_s"], r["update_ms"]))
    for phase in ("env_step", "inference", "update", "bookkeeping", "logging"):
        print("  %-12s %6.1f s  %5.1f%%" % (phase, timing[phase], 100 * timing[phase] / wall_s))
    print("  memory: trainer peak %d MB, env processes largest %d MB, total %d MB" % tuple(r["memory_mb"].values()))
    if out_dir:
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump(result, f, indent=2)
        print("  wrote", os.path.join(out_dir, "benchmark.json"))


# Running episode returns, kept per env between rollouts.
_running = {}


def episode_returns_add(recent, done, reward):
    reward = reward.cpu().numpy()
    for i in range(len(reward)):
        _running[i] = _running.get(i, 0.0) + float(reward[i])
        if done[i]:
            # The matching outcome entry was appended just before; attach the return to it.
            for e in reversed(recent):
                if "return" not in e:
                    e["return"] = _running[i]
                    break
            _running[i] = 0.0


if __name__ == "__main__":
    main()
