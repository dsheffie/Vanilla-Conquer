#!/bin/sh
#
# Copy SDL into the macOS app bundles so they run without Homebrew/MacPorts.
#
# Usage: scripts/bundle-macos.sh [build_dir]
#
# Re-run after every build: relinking points the executable back at the
# system SDL. Anything else in the bundle (e.g. game data under
# Contents/share) is left untouched.
#
set -eu

BUILD_DIR=${1:-build}

if [ ! -d "$BUILD_DIR" ]; then
    echo "error: build directory '$BUILD_DIR' not found" >&2
    exit 1
fi

# Prints the first dependency of $1 whose path contains $2.
find_dep() {
    otool -L "$1" | tail -n +2 | awk '{print $1}' | grep "$2" | head -n 1 || true
}

# Locates an SDL3 dylib for sdl2-compat to load at runtime.
find_sdl3() {
    if [ -n "${SDL3_LIB:-}" ]; then
        echo "$SDL3_LIB"
        return
    fi
    for prefix in "$(brew --prefix sdl3 2>/dev/null || true)" /opt/homebrew /opt/local /usr/local; do
        if [ -n "$prefix" ] && [ -f "$prefix/lib/libSDL3.0.dylib" ]; then
            echo "$prefix/lib/libSDL3.0.dylib"
            return
        fi
    done
}

found=0

for app in "$BUILD_DIR"/*.app; do
    [ -d "$app" ] || continue
    found=1

    name=$(basename "$app" .app)
    exe="$app/Contents/MacOS/$name"
    fw="$app/Contents/Frameworks"

    echo "== $name"

    sdl2_ref=$(find_dep "$exe" libSDL2)
    if [ -z "$sdl2_ref" ]; then
        echo "   not linked against SDL2, skipping"
        continue
    fi

    mkdir -p "$fw"

    case "$sdl2_ref" in
    @executable_path/*)
        echo "   SDL2 already bundled"
        ;;
    *)
        echo "   SDL2: $sdl2_ref"
        cp -f "$sdl2_ref" "$fw/libSDL2-2.0.0.dylib"
        chmod u+w "$fw/libSDL2-2.0.0.dylib"
        install_name_tool -id @rpath/libSDL2-2.0.0.dylib "$fw/libSDL2-2.0.0.dylib" 2>/dev/null
        install_name_tool -change "$sdl2_ref" @executable_path/../Frameworks/libSDL2-2.0.0.dylib "$exe" 2>/dev/null
        ;;
    esac

    # sdl2-compat is a shim that dlopen()s SDL3, looking next to itself first.
    if strings "$fw/libSDL2-2.0.0.dylib" | grep -q sdl2-compat; then
        sdl3=$(find_sdl3)
        if [ -z "$sdl3" ]; then
            echo "error: SDL2 is sdl2-compat but no SDL3 dylib was found; set SDL3_LIB" >&2
            exit 1
        fi
        echo "   SDL3: $sdl3"
        cp -f "$sdl3" "$fw/libSDL3.dylib"
        chmod u+w "$fw/libSDL3.dylib"
        install_name_tool -id @rpath/libSDL3.dylib "$fw/libSDL3.dylib" 2>/dev/null
    fi

    # Warn about any other dependency that would tie the app to this machine.
    for bin in "$exe" "$fw"/*.dylib; do
        otool -L "$bin" | tail -n +2 | awk '{print $1}' \
            | grep -E '^/(opt|usr/local)/' | sed "s|^|   warning: $(basename "$bin") still links |" || true
    done

    codesign --force -s - "$fw"/*.dylib 2>/dev/null
    codesign --force -s - "$app" 2>/dev/null
    codesign --verify --deep --strict "$app"
    echo "   ok"
done

if [ "$found" -eq 0 ]; then
    echo "error: no .app bundles in '$BUILD_DIR'" >&2
    exit 1
fi
