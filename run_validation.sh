#!/usr/bin/env bash
# Produce the JS reference scores and run the Python fidelity tests.
#
#   RUNS / FRAMES env vars control the eval size (defaults: official 500 x 2000).
#   For a quick dev pass:  RUNS=50 bash run_validation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ROOT}/.venv/bin/python"
REF="${ROOT}/reference/node_ref/run_ref.js"
RUNS="${RUNS:-500}"
FRAMES="${FRAMES:-2000}"

echo "================================================================"
echo " JS REFERENCE SCORES (runs=${RUNS} frames=${FRAMES} deterministic)"
echo "================================================================"
for pol in nop accel brain; do
  echo "--- policy=${pol} ---"
  node "${REF}" eval --policy="${pol}" --runs="${RUNS}" --frames="${FRAMES}" --det=true
done

echo
echo "================================================================"
echo " PYTHON FIDELITY TESTS"
echo "================================================================"
DT_SCORE_RUNS="${RUNS}" DT_SCORE_FRAMES="${FRAMES}" \
  "${PY}" -m pytest "${ROOT}/tests/test_fidelity.py" -v
