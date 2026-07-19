#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[predict] Starting UK Lotto predictions (10 lines, 6 numbers each)"

# Activate conda environment
if [[ -d "$ROOT_DIR/miniconda" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/miniconda/bin/activate"
else
  echo "[predict] miniconda not found at $ROOT_DIR/miniconda. Install or adjust this script." >&2
  exit 1
fi

if conda env list | grep -q "conda-py311"; then
  conda activate "$ROOT_DIR/conda-py311"
else
  echo "[predict] conda-py311 env not found. Create with: conda create -y -p ./conda-py311 python=3.11" >&2
  exit 1
fi

# Determine best ensemble if available
ENSEMBLE=$(python - <<'PY'
import json, os
best='outputs/results/best_ensemble.json'
m='frequency'
try:
    if os.path.exists(best):
        d=json.load(open(best))
        mm=d.get('best_method')
        if mm in {'frequency','weighted','consensus','probmap'}:
            m=mm
except Exception:
    pass
print(m)
PY
)

echo "[predict] Using ensemble method: $ENSEMBLE"

# Allow extra args to override defaults
EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  EXTRA_ARGS=("$@")
fi

# Run prediction
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then EXTRA_ARGS=( ); fi

set +u
PYTHONPATH=. python scripts/new_predict.py \
  --count 10 \
  --diversity 0.4 \
  --ensemble "$ENSEMBLE" \
  --optimize-coverage \
  --cap-number-usage 6 \
  --require-high-per-line \
  --decade-balance \
  --wildcards 2 \
  --no-viz \
  "${EXTRA_ARGS[@]}"
set -u

echo "[predict] Finished. See outputs/predictions/ for files and terminal for portfolio analysis."


