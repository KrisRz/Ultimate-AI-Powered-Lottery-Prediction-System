#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[predict] Starting UK Lotto predictions (10 lines, 6 numbers each)"

# Save script args before sourcing conda's activate (it would swallow them)
EXTRA_ARGS=("$@")
set --

# Activate conda environment: prefer in-repo conda-py311, else the
# `lotto-predict` env from environment.yml (make setup)
if [[ -d "$ROOT_DIR/miniconda" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/miniconda/bin/activate"
fi

if [[ -d "$ROOT_DIR/conda-py311" ]] && command -v conda >/dev/null; then
  conda activate "$ROOT_DIR/conda-py311"
elif command -v conda >/dev/null && conda env list | grep -q "lotto-predict"; then
  conda activate lotto-predict
else
  echo "[predict] No conda env found. Run: make setup && conda activate lotto-predict" >&2
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

# Run prediction (EXTRA_ARGS captured at the top of the script)
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


