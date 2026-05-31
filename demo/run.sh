#!/usr/bin/env bash
# COMODO Live Demo launcher.
#
# Usage:   bash demo/run.sh           (foreground)
#          bash demo/run.sh --open    (foreground + open browser)
#
# Requires the project's .venv to exist (already set up).
# DYLD_LIBRARY_PATH is needed on macOS to point Python at Homebrew's libexpat,
# because the system libexpat shipped with macOS 15+ no longer exports
# _XML_SetAllocTrackerActivationThreshold.

set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:${DYLD_LIBRARY_PATH:-}"

URL="http://127.0.0.1:8000"

echo "=============================================="
echo "  COMODO Live Demo"
echo "  → $URL"
echo "=============================================="

if [[ "${1:-}" == "--open" ]]; then
  (sleep 2 && open "$URL") &
fi

exec "$ROOT/.venv/bin/python" -m uvicorn demo.app:app \
  --host 127.0.0.1 --port 8000
