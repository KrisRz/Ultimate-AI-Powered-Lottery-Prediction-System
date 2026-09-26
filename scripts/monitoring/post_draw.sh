#!/usr/bin/env bash
# Post-draw routine - run the morning after each UK Lotto draw (Thu/Sun).
# Fetches the official result (accumulates prize_tiers.csv for the EV model),
# settles any played lines in the ledger, refreshes the dashboard, and prints
# the EV verdict for the NEXT draw.
#
# It runs the morning AFTER the draw rather than racing the cloud collector:
# the old 22:30 slot fired 15 min before collect.yml and left the tracked CSVs
# dirty on every single draw. GitHub's cron has also been running 2-11 h late
# since 2026-08-26, so the morning slot is the one that still finds the draw
# already collected.
#
# Install as a launchd job with: make install-cron   (see ops/README)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# The project interpreter directly, as the Makefile does. Activating it used
# to need a second conda install (miniconda/, 673 MB) for nothing but the
# `conda` command; the environment's own python needs no activation.
PY="${PY:-$ROOT_DIR/conda-py311/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "[post-draw] no interpreter at $PY - set PY or run make setup" >&2
  exit 1
fi

# Sync data committed by the cloud collector (GitHub Actions) first. The
# collector owns those files; see sync_collector_data.sh for why merging the
# local copy instead of discarding it corrupted the CSVs on 2026-09-05.
# A non-zero exit means the tree is unsound - stop rather than feed the model
# half-merged data.
bash "$ROOT_DIR/scripts/monitoring/sync_collector_data.sh" || {
  echo "[post-draw] refusing to run on an unsound tree - see the [sync] lines above"
  exit 1
}

# Nudge the cloud collector before doing anything locally. GitHub's cron has
# been dropping and delaying scheduled runs since late August 2026, and a
# workflow_dispatch is honoured immediately - so if the scheduled run never
# fired, this repairs the cloud copy (which is canonical) rather than leaving
# the Mac as the only machine holding the draw. Best-effort: no gh, no auth, no
# network, and the local fetch below still does the job.
if command -v gh >/dev/null; then
  gh workflow run collect.yml --ref main 2>/dev/null \
    && echo "[post-draw] asked the collector to run" \
    || echo "[post-draw] could not dispatch collect.yml - carrying on locally"
fi

echo "[post-draw] $(date '+%Y-%m-%d %H:%M') fetching latest result..."
PYTHONPATH=. "$PY" -c "from scripts.fetch_data import download_fresh_data; download_fresh_data()"

echo "[post-draw] scoring the model if that was a Must-Be-Won draw..."
PYTHONPATH=. "$PY" scripts/monitoring/post_mbw_validation.py || true

echo "[post-draw] settling ledger..."
PYTHONPATH=. "$PY" scripts/roi_ledger.py settle
PYTHONPATH=. "$PY" scripts/roi_ledger.py report

echo "[post-draw] refreshing dashboard..."
PYTHONPATH=. "$PY" scripts/dashboard.py

echo "[post-draw] EV verdict for the next draw:"
PYTHONPATH=. "$PY" scripts/ev_play.py --lines 5 || true

# Optional email alert on +EV draws; SMTP credentials live in ~/.lotto_env
# (never in the repo)
if [[ -f "$HOME/.lotto_env" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.lotto_env"
fi
PYTHONPATH=. "$PY" scripts/monitoring/ev_alert.py || true

echo "[post-draw] done."
