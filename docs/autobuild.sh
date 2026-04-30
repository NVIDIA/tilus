#!/usr/bin/env bash
# Live-preview server for Sphinx documentation using sphinx-autobuild.
# Rebuilds and refreshes the browser automatically when source files change.
#
# Usage:
#   ./docs/autobuild.sh [--clean] [-- <sphinx-autobuild args>]
#
# Options:
#   --clean   Remove build and generated directories before starting
#
# Any arguments after "--" are forwarded to sphinx-autobuild.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"
BUILD_DIR="$SCRIPT_DIR/build/html"

CLEAN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            CLEAN=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! command -v sphinx-autobuild &>/dev/null; then
    echo "Error: sphinx-autobuild not found. Install with: pip install -e \".[dev]\"" >&2
    exit 1
fi

if [[ "$CLEAN" -eq 1 ]]; then
    echo "Cleaning build directory..."
    rm -rf "$SCRIPT_DIR/build"
    find "$SOURCE_DIR" -type d -name generated -exec rm -rf {} + 2>/dev/null || true
fi

exec sphinx-autobuild "$SOURCE_DIR" "$BUILD_DIR" \
    --watch "$SOURCE_DIR" \
    --open-browser \
    "${EXTRA_ARGS[@]}"
