#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:---demo}"
if [[ "$MODE" != "--demo" && "$MODE" != "--private" ]]; then
  echo "Usage: ./scripts/run_platform.sh [--demo|--private]" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv with Python 3.11 and install requirements-platform.txt first." >&2
  exit 1
fi
if ! .venv/bin/python -c 'import fastapi, uvicorn, multipart' >/dev/null 2>&1; then
  echo "Install dependencies: .venv/bin/python -m pip install -r requirements-platform.txt" >&2
  exit 1
fi
if [[ ! -f web/dist/index.html ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "Node.js 22.12+ is required to build the frontend." >&2
    exit 1
  fi
  (cd web && npm ci && npm run build)
fi
if [[ "$MODE" == "--demo" ]]; then
  export AWARE_DEMO_MODE=1
  export AWARE_STORAGE_DIR="${AWARE_STORAGE_DIR:-$PWD/var/demo}"
else
  export AWARE_DEMO_MODE=0
  export AWARE_STORAGE_DIR="${AWARE_STORAGE_DIR:-$PWD/var/private}"
fi
export AWARE_PORT="${AWARE_PORT:-8000}"
echo "AWARE $MODE: http://127.0.0.1:$AWARE_PORT"
echo "Private runtime directory: $AWARE_STORAGE_DIR"
exec .venv/bin/python -m uvicorn service.api:create_app --factory --host 127.0.0.1 --port "$AWARE_PORT" --workers 1
