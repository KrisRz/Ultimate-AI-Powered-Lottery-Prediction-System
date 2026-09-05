#!/usr/bin/env bash
# Bring the local checkout in line with the data the cloud collector committed.
#
# The collector is canonical for the files below: it fetches the same draw from
# the same feed and commits it, so the local copy is always a subset of the
# remote one and is never worth keeping. The old `git pull --rebase --autostash`
# assumed the opposite. It broke on 2026-09-05: the Mac fetches a draw into the
# tracked CSVs at 21:30 UTC, 15 min before the collector commits the same draw,
# so the tree is always dirty here; miss a run or two (this machine missed
# 08-29 and 09-02) and the local +1 draw and the remote +3 land on the same
# lines - "Applying autostash resulted in conflicts", CSVs left full of
# `<<<<<<<` markers, and `make play` reading them as "£2M, rollover 0 of 5".
#
# So: throw the local copies away first, fast-forward only, and refuse to hand
# a half-merged tree to the model.
#
# Exit 0 - the tree is sound (possibly stale; a warning says so).
# Exit 1 - the tree is NOT sound; the caller must not run the model on it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REMOTE="${1:-origin}"
BRANCH="${2:-main}"

# Everything collect.yml commits. Ledger and logs are local-only and stay put.
COLLECTOR_FILES=(
  data/prize_tiers.csv
  data/merged_lottery_data.csv
  data/lotto_full_history.csv
  data/mbw_validation.csv
  site/public/data/site.json
  site/src/__fixtures__/popularity-golden.json
)

unmerged() { test -n "$(git ls-files -u)"; }

if unmerged; then
  echo "[sync] unmerged files from an earlier run - refusing to touch them:"
  git ls-files -u --format='  %(path)' | sort -u
  echo "[sync] fix with: git restore --staged --worktree <files> (the collector's copy is canonical)"
  exit 1
fi

git fetch "$REMOTE" "$BRANCH"

# Discard the local copies BEFORE merging, so there is nothing to conflict with.
for f in "${COLLECTOR_FILES[@]}"; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git checkout HEAD -- "$f"
done

if ! git merge --ff-only "$REMOTE/$BRANCH"; then
  echo "[sync] cannot fast-forward onto $REMOTE/$BRANCH - local commits or a diverged branch."
  echo "[sync] continuing with the data already in the tree (it is intact, just possibly stale)."
  exit 0
fi

if unmerged; then
  echo "[sync] fast-forward left unmerged files - this should be impossible; refusing to continue"
  exit 1
fi

echo "[sync] in step with $REMOTE/$BRANCH ($(git rev-parse --short HEAD))"
