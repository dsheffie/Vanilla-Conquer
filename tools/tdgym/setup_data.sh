#!/bin/sh
#
# Build the Tiberian Dawn data directory from the freeware C&C Gold CD images.
#
# Usage: tools/tdgym/setup_data.sh GDI.iso NOD.iso OUTPUT_DIR
#
# Needs bsdtar (Debian/Ubuntu: libarchive-tools; built into macOS) to read the ISOs
# without mounting them, and unshield (cargo install unshield) for the installer
# archive INSTALL/SETUP.Z. Set UNSHIELD to use another unshield binary.
#
# Produces:
#   OUTPUT_DIR/*.MIX                 shared data: disc root files plus installer files
#   OUTPUT_DIR/gdi/GENERAL.MIX ...   per-disc files from the GDI disc
#   OUTPUT_DIR/nod/GENERAL.MIX ...   per-disc files from the NOD disc
#
set -eu

if [ $# -ne 3 ]; then
    sed -n '4,16p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
GDI_ISO=$1
NOD_ISO=$2
OUT=$3
UNSHIELD=${UNSHIELD:-unshield}

for tool in bsdtar "$UNSHIELD"; do
    if ! command -v "$tool" > /dev/null 2>&1; then
        echo "error: $tool not found" >&2
        exit 1
    fi
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/tdgym-data.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

# Copy the file named $2 (any case) found under $1 to $3. Fails if it is missing.
copy_file() {
    found=$(find "$1" -type f -iname "$2" | head -n 1)
    if [ -z "$found" ]; then
        echo "error: $2 not found in $1" >&2
        exit 1
    fi
    cp "$found" "$3/$2"
}

# Files that differ between the discs; everything else at the disc root is shared.
PER_DISC="GENERAL.MIX MOVIES.MIX"
# Files the installer unpacks from INSTALL/SETUP.Z. Its CCLOCAL.MIX, not the one in
# INSTALL/, is the one an installed game uses.
INSTALLER="CCLOCAL.MIX UPDATE.MIX UPDATEC.MIX SPEECH.MIX TRANSIT.MIX DESEICNH.MIX TEMPICNH.MIX WINTICNH.MIX"

mkdir -p "$OUT/gdi" "$OUT/nod"
for disc in gdi nod; do
    iso=$GDI_ISO
    [ "$disc" = nod ] && iso=$NOD_ISO
    echo "Reading $iso"
    mkdir -p "$WORK/$disc"
    bsdtar -xf "$iso" -C "$WORK/$disc"
    for name in $PER_DISC; do
        copy_file "$WORK/$disc" "$name" "$OUT/$disc"
    done
done

echo "Copying shared files"
find "$WORK/gdi" -maxdepth 1 -type f -iname '*.MIX' | while read -r path; do
    name=$(basename "$path" | tr '[:lower:]' '[:upper:]')
    case " $PER_DISC " in
    *" $name "*) ;;
    *) cp "$path" "$OUT/$name" ;;
    esac
done

echo "Unpacking INSTALL/SETUP.Z"
setup=$(find "$WORK/gdi" -type f -ipath '*/INSTALL/SETUP.Z' | head -n 1)
if [ -z "$setup" ]; then
    echo "error: INSTALL/SETUP.Z not found on $GDI_ISO" >&2
    exit 1
fi
"$UNSHIELD" extract "$setup" "$WORK/setup" > /dev/null
for name in $INSTALLER; do
    copy_file "$WORK/setup" "$name" "$OUT"
done

echo "Done: $(find "$OUT" -maxdepth 1 -type f -iname '*.MIX' | wc -l | tr -d ' ') shared files," \
    "gdi: $(ls "$OUT/gdi" | tr '\n' ' ')nod: $(ls "$OUT/nod" | tr '\n' ' ')"
