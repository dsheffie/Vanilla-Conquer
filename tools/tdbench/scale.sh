#!/bin/sh
#
# Measure tdbench throughput with several games running at once.
#
# Usage: tools/tdbench/scale.sh TDBENCH LIB DATA [FRAMES] [COUNTS...]
#   TDBENCH  path to the tdbench executable
#   LIB      path to TiberianDawn.so / TiberianDawn.dylib
#   DATA     game data directory
#   FRAMES   game frames per run (default 54000, one hour of game time)
#   COUNTS   parallel game counts to try (default: 1 2 4 6 8 12 16)
#
# Each parallel worker plays REPEAT games back to back (default 3, set in the environment)
# so that games ending early don't leave cores idle. Every game gets its own seed, and each
# worker its own work directory under $TMPDIR. game_hrs/hr is how many hours of game time all
# games together simulate per hour of wall-clock time; avg_min is the mean game length in
# minutes of game time.
#
set -eu

if [ $# -lt 3 ]; then
    sed -n '4,18p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

TDBENCH=$1
LIB=$2
DATA=$3
FRAMES=${4:-54000}
shift 3
[ $# -gt 0 ] && shift
COUNTS=${*:-1 2 4 6 8 12 16}
REPEAT=${REPEAT:-3}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/tdbench-scale.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# Seconds since the epoch with sub-second precision, portable to macOS and Linux.
now() {
    perl -MTime::HiRes=time -e 'printf "%.3f\n", time'
}

printf "%7s %7s %7s %13s %10s %10s %10s %9s\n" workers ended wall_s per_game_fps total_fps game_hrs/hr games/hr avg_min
for n in $COUNTS; do
    start=$(now)
    i=1
    while [ "$i" -le "$n" ]; do
        (
            k=0
            while [ "$k" -lt "$REPEAT" ]; do
                "$TDBENCH" --lib "$LIB" --data "$DATA" --work "$WORK/run$i" --frames "$FRAMES" \
                    --seed $((i + n * k)) 2> /dev/null
                k=$((k + 1))
            done
        ) > "$WORK/out$i" &
        i=$((i + 1))
    done
    wait
    end=$(now)
    # Games can end early when one side wins, so total throughput counts the frames each
    # game actually simulated, reported as hours of game time simulated per wall-clock hour.
    cat "$WORK"/out* | awk -v n="$n" -v repeat="$REPEAT" -v start="$start" -v end="$end" '
        {
            for (f = 1; f <= NF; f++) {
                split($f, kv, "=")
                if (kv[1] == "fps") { fps += kv[2]; runs++ }
                if (kv[1] == "frames") { frames += kv[2] }
            }
            if ($0 ~ /end="game over"/) { ended++ }
        }
        END {
            wall = end - start
            if (runs != n * repeat) { printf "only %d of %d runs reported\n", runs, n * repeat > "/dev/stderr"; exit 1 }
            printf "%7d %7s %7.1f %13.0f %10.0f %10.0f %10.0f %9.1f\n", n, ended "/" runs, wall, fps / runs, frames / wall, frames / 15 / wall, runs * 3600 / wall, frames / runs / 15 / 60
        }'
    rm -f "$WORK"/out*
done
