#!/usr/bin/env bash
# Post-draw routine - run after each UK Lotto draw (Wed/Sat evening).
# Fetches the official result (accumulates prize_tiers.csv for the EV model),
# settles any played lines in the ledger, refreshes the dashboard, and prints
# the EV verdict for the NEXT draw.
#
# Install as a launchd job with: make install-cron   (see ops/README)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -d "$ROOT_DIR/miniconda" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT_DIR/miniconda/bin/activate"
fi
if [[ -d "$ROOT_DIR/conda-py311" ]] && command -v conda >/dev/null; then
  conda activate "$ROOT_DIR/conda-py311"
elif command -v conda >/dev/null && conda env list | grep -q "lotto-predict"; then
  conda activate lotto-predict
fi

# Sync data committed by the cloud collector (GitHub Actions) first.
# A failed rebase must not leave the repo mid-rebase or every later run
# would fail on the pull too.
git pull --rebase --autostash origin main || {
  git rebase --abort 2>/dev/null || true
  echo "[post-draw] git pull failed - continuing with local data"
}

echo "[post-draw] $(date '+%Y-%m-%d %H:%M') fetching latest result..."
PYTHONPATH=. python -c "from scripts.fetch_data import download_fresh_data; download_fresh_data()"

echo "[post-draw] scoring the model if that was a Must-Be-Won draw..."
PYTHONPATH=. python scripts/monitoring/post_mbw_validation.py || true

echo "[post-draw] settling ledger..."
PYTHONPATH=. python scripts/roi_ledger.py settle
PYTHONPATH=. python scripts/roi_ledger.py report

echo "[post-draw] refreshing dashboard..."
PYTHONPATH=. python scripts/dashboard.py

echo "[post-draw] EV verdict for the next draw:"
PYTHONPATH=. python scripts/ev_play.py --lines 5 || true

# Optional email alert on +EV draws; SMTP credentials live in ~/.lotto_env
# (never in the repo)
if [[ -f "$HOME/.lotto_env" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.lotto_env"
fi
PYTHONPATH=. python scripts/monitoring/ev_alert.py || true

echo "[post-draw] done."
