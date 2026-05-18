#!/usr/bin/env bash
# setup-llama.sh — fetch upstream llama.cpp at the exact commit used by PartoGuard
#
# This script clones llama.cpp at the verified commit (b9199-39cf5d619)
# that matches the VPS deployment (95.71% accuracy on the 350-image eval corpus)
# and creates the symlink expected by CMakeLists.txt.
#
# Run this once before building the Android app:
#   cd android/llama && bash setup-llama.sh

set -euo pipefail

LLAMA_COMMIT="39cf5d61915769124b7efbbfa69c46f19a6363ee"
LLAMA_DIR="/root/llama-upstream"
SYMLINK_PATH="$(dirname "$0")/src/main/cpp/llama.cpp"

echo "==> Checking for llama.cpp at $LLAMA_DIR ..."

if [ -d "$LLAMA_DIR/.git" ]; then
    current=$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
    if [ "$current" = "$LLAMA_COMMIT" ]; then
        echo "    Already at correct commit ($LLAMA_COMMIT). Nothing to do."
    else
        echo "    Wrong commit ($current). Fetching $LLAMA_COMMIT ..."
        git -C "$LLAMA_DIR" fetch origin
        git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
    fi
else
    echo "==> Cloning llama.cpp ..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
fi

echo "==> Creating symlink: $SYMLINK_PATH -> $LLAMA_DIR"
ln -sfn "$LLAMA_DIR" "$SYMLINK_PATH"

echo ""
echo "Done. You can now build the Android app:"
echo "    cd android && ./gradlew assembleDebug"
