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

`--device auto` picks CUDA, then MPS, then CPU. `--envs` is one game per process; use
about one per core and leave a few cores for the learner. Resume with `--resume runs/NAME/latest.pt`.
Progress is in `runs/NAME/metrics.csv` (one row per update); watch `win_rate`,
`loss_rate`, `minutes`, `kills` and `losses`. Evaluate a checkpoint with
`evaluate.py --policy runs/NAME/latest.pt --episodes 20`.

## Memory sizing

The first run was killed for low memory on a 16 GB Mac. Measured afterwards:

- Each game process: about 120 MB.
- Trainer process: about 0.5 GB, plus GPU memory for the 1.2M parameter network.
- Rollout buffer, kept on the device: `steps x envs x (12*64*64 bytes + 2.4 KB)`, about
  75 MB at 128 x 12 and 400 MB at 128 x 64.

One fix is already in: the game processes used to import torch (about 280 MB each)
because macOS starts them with `spawn`, which re-imports `ppo.py`. Keep torch imports
inside functions in `ppo.py` and out of anything the env processes import.

On Linux, Python before 3.14 starts subprocesses with `fork`. `ppo.py` creates the vector
env before touching CUDA, which keeps that safe; keep it that way, or pass
`context="forkserver"` to `AsyncVectorEnv`.

## Speed reference (M6, 12 cores)

- Simulation alone: ~8,000-13,000 frames/s per game; 12 parallel AI-only games give
  ~5,400 complete games per hour (`scale.sh`).
- `TiberianDawnEnv`: ~1,700 steps/s per process (15 frames per step).
- `MacroEnv` + PPO, 12 envs, MPS: ~650-850 decisions/s, about 3,000 thirty-minute games per hour.
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

1. **Fix the reward** (`macro.default_reward`): cut the harvest term by 10-100x or anneal it
   to zero, raise the win/loss reward, reward destroying enemy buildings more than units.
   Consider making a 30-minute timeout count partly as a loss.
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
- **`pkill -f`** on a pattern that appears in your own shell command kills that shell; use
  `pkill -x vanillatd` or exact pids.
