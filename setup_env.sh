#!/usr/bin/env bash
# Create the clean .venv and install pinned dependencies.
# Avoids base anaconda (numpy 2.4.x ABI break with matplotlib/pandas/torch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBASE="${PYBASE:-/Users/markschutera/anaconda3/bin/python3}"

echo "Creating venv at ${ROOT}/.venv using ${PYBASE}"
"${PYBASE}" -m venv "${ROOT}/.venv"
"${ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements.txt"

echo "Done. Recorded versions:"
"${ROOT}/.venv/bin/python" -m pip freeze | grep -Ei 'numpy|torch|matplotlib|pandas|pytest|tqdm' || true
