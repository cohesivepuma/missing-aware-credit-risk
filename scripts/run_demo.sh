#!/usr/bin/env bash
set -euo pipefail
AWARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AWARE_ROOT"
AWARE_PYTHON="${AWARE_PYTHON:-$AWARE_ROOT/.venv/bin/python}"
if [[ ! -x "$AWARE_PYTHON" ]]; then
  echo "Python environment missing. Follow README.md to create .venv, or set AWARE_PYTHON."
  exit 1
fi
if [[ ! -f data/processed/german_credit/german_credit.csv ]]; then
  "$AWARE_PYTHON" scripts/prepare_german_credit.py
fi
exec "$AWARE_PYTHON" -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 "$@"
