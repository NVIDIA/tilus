#!/usr/bin/env bash
# Serve the built documentation locally.
# Usage: ./docs/serve.sh [port]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_DIR="$SCRIPT_DIR/build/html"
PORT="${1:-8000}"

if [ ! -d "$HTML_DIR" ]; then
    echo "Error: $HTML_DIR does not exist. Run 'make html' first." >&2
    exit 1
fi

echo "Serving docs at http://localhost:$PORT"
python3 -m http.server "$PORT" --directory "$HTML_DIR"
