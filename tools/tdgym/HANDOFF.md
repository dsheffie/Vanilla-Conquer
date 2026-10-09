# Tiberian Dawn RL: handoff

How to pick this up on a bigger machine (more RAM, more cores, an NVIDIA GPU, e.g. an
aarch64 Linux box), what exists, what the first training run showed, and what to do next.

## What exists

All on branch `headless-rl`:

| Piece | Where | What it does |
|---|---|---|
| Portable remaster dll | `tiberiandawn/dllinterface.*`, `startup.cpp`, CMake | The game simulation as `TiberianDawn.so`/`.dylib`, built on macOS and Linux (`-DBUILD_REMASTERTD=ON`). |
| Engine additions | `CNC_Set_Random_Seed`, `CNC_Set_Headless`, `HouseClass::MPlayer_Defeated` | Seeded games, no screen drawing, AI-only matches that play until one side remains. |
| `tools/tdhost` | C++ static library | Prepares a work directory, loads the dll, starts skirmishes. |
| `tools/tdbench` | `tdbench`, `scale.sh` | Headless AI-vs-AI benchmark with a state hash for determinism checks. |
| `tools/tdgym/tdenv.*` | `libtdenv.so`/`.dylib` | C layer for RL: fog-filtered objects, shroud, Tiberium, sidebar, placement masks, actions. |
| `tools/tdgym/tdgym/env.py` | `TiberianDawnEnv` | Click-level Gymnasium env (`TiberianDawn-v0`). |
| `tools/tdgym/tdgym/macro.py` | `MacroEnv` | 31 strategy-level actions (deploy, build/train by role, harvest, attack, scout, defend), auto-placement, shaped reward. |
| `tools/tdgym/train/` | `policy.py`, `ppo.py`, `evaluate.py` | Patch-transformer actor-critic, masked PPO, evaluation against the built-in AI. |

## Setting up a new machine

### 1. Build

Only the dll and the tools are needed; no SDL or OpenAL.

```sh
cmake -G Ninja -B build-remaster -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DBUILD_REMASTERTD=ON -DBUILD_VANILLATD=OFF -DBUILD_VANILLARA=OFF \
  -DSDL2=OFF -DOPENAL=OFF -DNETWORKING=OFF .
cmake --build build-remaster
```

Produces `build-remaster/tiberiandawn/TiberianDawn.so`, `build-remaster/tools/tdgym/libtdenv.so`
and `build-remaster/tdbench`. Verified with GCC 13 on aarch64 Ubuntu 24.04 and Clang on macOS.
Windows is untested for these tools (they're excluded from Windows builds).

### 2. Game data

The data comes from EA's freeware Command & Conquer Gold CDs, one ISO per disc:

1. **Download the ISOs.** The GDI and NOD discs are linked from the top-level README
   ("Running"), on ModDB:
   [GDI](https://www.moddb.com/games/cc-gold/downloads/command-conquer-gold-free-game-gdi-iso),
   [NOD](https://www.moddb.com/games/cc-gold/downloads/command-conquer-gold-free-game-nod-iso).
   ModDB sits behind a Cloudflare check, so `curl`/`wget` get a challenge page instead of
   the file: download in a browser and copy the ISOs over (each is about 600 MB; the
   original files are `CnC_GDI95.iso` and `CnC_NOD95.iso`).

2. **Install the two tools** the script needs:

   ```sh
   sudo apt install libarchive-tools      # bsdtar, reads ISOs without mounting; built into macOS
   cargo install unshield                 # agrif's InstallShield 3 extractor, for INSTALL/SETUP.Z
   ```

   `cargo` comes from [rustup](https://rustup.rs) if the machine doesn't have it.

3. **Build the data directory:**

   ```sh
   tools/tdgym/setup_data.sh CnC_GDI95.iso CnC_NOD95.iso /path/to/data
   ```

   It needs about 2 GB of temporary space (in `$TMPDIR`) and writes about 900 MB. It ends
   with `Done: 15 shared files, gdi: GENERAL.MIX MOVIES.MIX nod: GENERAL.MIX MOVIES.MIX`.
   Then check it with the `tdbench` hash in step 4.

The script produces this layout, which you can also assemble by hand:

```
data/
  AUD.MIX CONQUER.MIX DESERT.MIX SCORES.MIX SOUNDS.MIX TEMPERAT.MIX WINTER.MIX   (GDI disc root)
  CCLOCAL.MIX UPDATE.MIX UPDATEC.MIX SPEECH.MIX TRANSIT.MIX                    (INSTALL/SETUP.Z)
  DESEICNH.MIX TEMPICNH.MIX WINTICNH.MIX                                       (INSTALL/SETUP.Z)
  gdi/GENERAL.MIX gdi/MOVIES.MIX                                               (GDI disc root)
  nod/GENERAL.MIX nod/MOVIES.MIX                                               (NOD disc root)
```

Notes for doing it by hand:

- Only GENERAL.MIX and MOVIES.MIX differ between the discs; the other root files are identical.
- `SETUP.Z` is an InstallShield 3 archive (`unshield extract INSTALL/SETUP.Z out/` puts
  the files under `out/C&C95/`). Use its `CCLOCAL.MIX`, not the one in `INSTALL/`.
- File name case doesn't matter; the game looks files up case-insensitively.
- The multiplayer maps (SCM01-09, 70-74, 77, 96) are inside GENERAL.MIX. tdhost presents
  this layout to the dll as one flat directory of links (`--disc` / `disc=` picks gdi or nod).

### 3. Python

Python 3.10 or newer:

```sh
python3 -m venv venv
venv/bin/pip install torch numpy gymnasium   # torch built for your CUDA version
export TDGYM_GAME_LIB=$PWD/build-remaster/tiberiandawn/TiberianDawn.so
export TDGYM_TDENV=$PWD/build-remaster/tools/tdgym/libtdenv.so
export TDGYM_DATA=/path/to/data
export PYTHONPATH=$PWD/tools/tdgym
```

The default Linux torch wheel targets the newest CUDA and needs a recent driver; check
`nvidia-smi` and pick a matching build if it is older. For example zen5's driver 550
supports CUDA 12.4, so: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
(CUDA 12.x builds run on any 12.x-era driver).

### 4. Check it works

```sh
build-remaster/tdbench --lib $TDGYM_GAME_LIB --data $TDGYM_DATA --work /tmp/tb --seed 1 --frames 9000
venv/bin/python tools/tdgym/examples/scripted_agent.py --minutes 10    # builds a base, asserts fog every step
venv/bin/python tools/tdgym/train/evaluate.py --policy scripted --episodes 4
```

`tdbench --seed 1 --frames 9000` must print `state=f51e317e7cf2df0d`, the same hash
on every platform. The scripted agent should end with PROC, NUKE, FACT, HARV and 24 E1 on seed 1.

### 5. Train

```sh
venv/bin/python tools/tdgym/train/ppo.py --run-dir runs/NAME --envs N --total-steps 20000000
```

`--ai-difficulty easy|normal|hard` (also on `evaluate.py`, `ai_difficulty=` on the envs and
`--ai-difficulty 0-2` on tdbench) sets the built-in AI's handicap, which scales its
firepower, armor, speed, rate of fire, costs and build speed. Normal is the default and
plays exactly as before; the scripted baseline averages -10.75 / -11.30 / -11.76 return
against easy / normal / hard over 6 games each. Remaster builds have no difficulty rules
of their own, so tdhost passes the vanilla game's easy and hard values through `CNC_Config`.

**Curriculum:** `--curriculum` starts against the easy AI and moves up a level once the
win rate at the current level reaches `--curriculum-threshold` (default 0.5) over its last
`--curriculum-window` games (default 100). `--curriculum-floor` (default 0.2) of games stay
on the easier levels, split evenly, so the policy keeps what it learned there. Each env's
next game is assigned when its current game starts, so a level change reaches each env one
game later. The log and `metrics.csv` show the current level and win rates per difficulty;
`--resume` continues the curriculum, and `--init-from CHECKPOINT` starts a new run from
another run's weights, e.g. to carry a policy trained against normal into a curriculum.

`--device auto` picks CUDA, then MPS, then CPU. `--envs` is one game per process; use
about one per core and leave a few cores for the learner. Resume with `--resume runs/NAME/latest.pt`.
Progress is in `runs/NAME/metrics.csv` (one row per update); watch `win_rate`,
`loss_rate`, `minutes`, `kills` and `losses`. Evaluate a checkpoint with
`evaluate.py --policy runs/NAME/latest.pt --episodes 20`.

### Videos

`evaluate.py --video` records games with the real game graphics, as the agent sees them
(its own shroud), against the built-in AI. Needs `ffmpeg`:

```sh
venv/bin/python tools/tdgym/train/evaluate.py --policy runs/NAME/latest.pt --episodes 1 --video game.mp4
```

Defaults: 8x real time at 30 fps, half size (696 x 588 on SCM01), so a 30-minute game is a
4-minute, ~12 MB video; `--video-speed`, `--video-fps` and `--video-scale` change that.
Recording one 30-minute game took 30 s on zen5. `--policy random` or `scripted` records
the baselines. With several episodes, `game.mp4` becomes `game-1.mp4` and so on.

### 6. Benchmark

Time a fixed amount of training, the same everywhere, to compare machines and settings:

```sh
venv/bin/python tools/tdgym/train/ppo.py --benchmark --total-steps 98304 --envs 12 --run-dir runs/bench-HOST
```

98,304 decisions is 64 PPO updates at the default 12 envs x 128 steps. It trains with
fixed seeds and writes no checkpoints, then prints and saves `benchmark.json`: machine,
settings, wall time, decisions/s, game frames/s, and the time split between waiting on
the games (`env_step`), choosing actions (`inference`), PPO updates (`update`) and
bookkeeping, plus peak memory. Keep `--total-steps` fixed and vary `--envs`/`--device` to
find the best configuration; the `share` fields show which part to scale.

Reference results, `--total-steps 98304`:

| Machine | Device | Envs | Wall | Decisions/s | env_step | inference | update | Memory |
|---|---|---|---|---|---|---|---|---|
| Apple M6, 12 cores, 16 GB | MPS | 12 | 128.5 s | 765 | 28.5% | 14.5% | 52.3% (1.05 s each) | trainer 513 MB, env 141 MB |
| zen5: Ryzen 9 9900X, 12C/24T, 60 GB | RTX 4060 Ti 8 GB | 12 | 67.8 s | 1,450 | 51.8% | 11.1% | 35.7% (378 ms each) | trainer 1.6 GB, env 143 MB |
| zen5 | RTX 4060 Ti 8 GB | 20 | 63.9 s | 1,563 | 50.0% | 9.7% | 39.0% (639 ms each) | |

(zen5 runs had another job using about 4 of its 24 threads.)

On the M6 the PPO update is the bottleneck; on zen5 the GPU is 2.8x faster at it and
waiting on the games becomes half the time. Going from 12 to 20 games there gains only 7%:
envs step in lockstep, so each step waits for the slowest game, and extra games share
cores with SMT siblings. The next speedup is structural rather than more envs: run PPO
updates while the games play the next rollout (they currently alternate), and step games
asynchronously.

Work directories (one per env: a copy of the library, its CONQUER.INI and links to the
data) go to `$TDGYM_WORK_ROOT`, else `/dev/shm` on Linux, else the temp directory, and are
deleted when the env closes or the process exits. Each game looks up files about 1,000
times a second; on a very high core count machine moving work directories and data to
`/dev/shm` gave a clear speedup, while on zen5 (12 envs) it made no measurable difference
(1,443 vs 1,454 decisions/s). With many envs, also put the game data in `/dev/shm`.

On Linux the env processes are forked from the trainer, so the kernel's peak-RSS for
children counts pages shared with it; the benchmark reports env memory as proportional
set size from `/proc/PID/smaps_rollup` instead.

## Memory sizing

The first run was killed for low memory on a 16 GB Mac. Two causes were found and fixed:

- **Reload leak (the big one):** each reset reloads the game library, and on non-Windows
  platforms nothing freed the game's allocations on unload, so every game process grew
  ~42 MB per episode; the first run's processes played ~90 episodes each. The dll now
  runs the same cleanup on unload as Windows' `DllMain` does. A drift of under 1 MB per
  episode remains, which only matters over thousands of episodes per process.
- **torch in game processes:** see below.

Two related Linux problems were also fixed: GCC marked `MixFileClass::MixList` as a GNU
unique symbol, which made `dlclose` a no-op so "reloads" restarted the game in place
(built with `-fno-gnu-unique` now), and freeing the game from a destructor double-freed it
at exit (tdhost calls the new `CNC_Free_Game` before `dlclose` instead).

Measured with both fixes:

- Each game process: about 120-140 MB.
- Trainer process: about 0.5 GB, plus GPU memory for the 1.2M parameter network.
- Rollout buffer, kept on the device: `steps x envs x (12*64*64 bytes + 2.4 KB)`, about
  75 MB at 128 x 12 and 400 MB at 128 x 64.

The game processes also used to import torch (about 280 MB each)
because macOS starts them with `spawn`, which re-imports `ppo.py`. Keep torch imports
inside functions in `ppo.py` and out of anything the env processes import.

On Linux, Python before 3.14 starts subprocesses with `fork`. `ppo.py` creates the vector
env before touching CUDA, which keeps that safe; keep it that way, or pass
`context="forkserver"` to `AsyncVectorEnv`.

## Speed reference (M6, 12 cores)

- Simulation alone: ~8,000-13,000 frames/s per game; 12 parallel AI-only games give
  ~5,400 complete games per hour (`scale.sh`).
- `TiberianDawnEnv`: ~1,700 steps/s per process (15 frames per step).
- `MacroEnv` + PPO, 12 envs, MPS: ~750-850 decisions/s, about 3,000 thirty-minute games per hour
  (see Benchmark above).
- Learner: on Apple silicon, torch CPU matmuls run on the SME unit through Accelerate
  (1.5 TFLOP/s single thread) but convolutions run NEON kernels; the policy uses no
  convolutions for that reason. MPS was still about 2x faster than CPU for training.

## First training run (macro-v1)

12 envs on map SCM01 against one built-in AI, 30-minute games, a decision every 2 game
seconds. Killed at 1.0M of 5M decisions; checkpoints are in `build-remaster/runs/macro-v1/`
on the original Mac only (not committed).

| Decisions | Win | Loss | Minutes | Harvested | Kills | Losses |
|---|---|---|---|---|---|---|
| 120k | 1% | 31% | 27.4 | 26k | 7.3 | 31.5 |
| 245k | 0% | 4% | 29.9 | 49k | 7.4 | 23.5 |
| 1.0M | 0% | 1% | 30.0 | 45k | 6.7 | 28.9 |

Baselines: random and the scripted build order lost 4 of 4 games each.

It learned to survive and farm, not to win. Reward shaping is the cause: 0.0002 per
harvested credit makes 45k credits worth about +9, against +1 for a win, so turtling pays.

## Next steps

1. **Train with the new reward** (`macro.Reward`, done after macro-v1): win +10, loss -10,
   timeout -8 and terminal, building kills +0.2, exploration up to +1, harvesting 20x
   smaller. Under it macro-v1's turtling scores about -6, losing about -12. If training
   stalls at turtling, raise `timeout` to `loss`; if it farms civilian buildings for the
   kill bonus, filter kills by owner.
2. **Evaluate macro-v1** to see what it does with its army (`evaluate.py --show-actions`).
3. **Scale up**: more envs and longer runs; then more maps (`--map`), both sides
   (`agent_side`) and 2-3 AI opponents (`num_ais`) for robustness.
4. **Self-play** needs two agent-controlled players; `tdenv` currently has one agent
   (player 0) and AI opponents.
5. **Environment gaps**: no terrain passability layer (the dll doesn't export it), no special
   weapons, masks don't cover command targets, buildings are auto-placed by a fixed rule in
   `MacroEnv`.

## Things that bit us

- **Game state is global**: one game per process. Restarting a scenario in place leaks
  state (seeded games diverge after ~3,600 frames), so `reset` reloads the library.
- **Library path**: the dll finds its data through a `CONQUER.INI` next to itself. macOS
  reports a loaded library by its resolved path, so tdhost copies the library into each
  work directory rather than symlinking. INI values over ~245 characters are ignored, so
  keep work directory paths short.
- **Data layout**: in dll mode the game only searches one directory, hence the flat view.
- **Coordinates**: the grid is the map plus a one-cell border; click pixels are offset one
  cell from exported positions; exported cells are object centers, so building footprints
  come from the occupancy export, which leaves an unused slot after each cell's occupiers.
- **First frame**: commands before the first frame are ignored; `reset` runs one frame.
- **Legacy rendering**: with fewer than two human players the dll draws the whole screen
  every frame unless `CNC_Set_Headless(true)`, which tdhost always sets.
- **AI-only games** used to end on frame 2 ("no humans left"); fixed in `MPlayer_Defeated`.
- **`char` signedness**: object owners are plain `char`, signed on macOS and unsigned on
  aarch64 Linux; always read them as unsigned.
- **`dlclose` may not unload**: check `/proc/self/maps` after it if reloads misbehave;
  any GNU unique symbol (`nm -D --defined-only lib.so | grep " u "`) pins a library in memory.
- **`pkill -f`** on a pattern that appears in your own shell command kills that shell; use
  `pkill -x vanillatd` or exact pids.
