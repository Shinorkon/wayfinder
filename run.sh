#!/usr/bin/env bash
#
# Launch the Wayfinder dashboard.
#
# Uses whatever python3 is active (activate your virtualenv first if you use one),
# resolves paths relative to this script, and tells you if the analysis artefacts are
# missing instead of quietly starting with no data.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PORT="${WAYFINDER_PORT:-8000}"
HOST="${WAYFINDER_HOST:-127.0.0.1}"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: '$PYTHON' not found. Activate a virtualenv or set PYTHON=..." >&2
  exit 1
fi

if ! "$PYTHON" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "error: fastapi/uvicorn are not installed for $PYTHON." >&2
  echo "       run: $PYTHON -m pip install -r requirements.txt" >&2
  exit 1
fi

if [ ! -f "data/analysis/index.json" ]; then
  echo "warning: data/analysis/index.json is missing, so no stop has a measured arrival rate."
  echo "         generate it first:  $PYTHON scripts/analyze_footage.py --skip-sam3"
  echo "         the dashboard will start and report the data as unavailable."
  echo
fi

echo "=========================================================="
echo "  Wayfinder :: RTL stop demand & frequency modelling"
echo "  http://${HOST}:${PORT}"
echo "=========================================================="

exec "$PYTHON" -m uvicorn src.api.server:app --host "$HOST" --port "$PORT"
