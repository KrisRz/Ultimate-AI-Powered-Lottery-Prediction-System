# Working in this repo

An EV toolkit for UK Lotto, not a predictor. It answers two questions — **when
to play** (is this draw's expected value above zero?) and **what to play**
(which lines share a jackpot with fewest people) — and its most common correct
answer is SKIP. Anything that makes the model look more promising than the data
supports is a bug, including in prose.

## Running things

```bash
PY=./conda-py311/bin/python        # what the Makefile uses; no conda activate needed
PYTHONPATH=. $PY scripts/ev_play.py
make test                          # pytest, the whole suite, ~5 s
cd site && npm run lint && npm run typecheck && npm test && npm run build && npm run size
```

Run **all five** site commands, in that order — they are exactly what Site CI
runs, and `lint` is the first. `tsc` and `vitest` alone will miss an ESLint
error (the React Compiler rules reject things that typecheck fine, such as
setting state in a mount effect).

`PYTHONPATH=.` is required for every script — they import `lottery.ev` and each
other. The Makefile sets it; a bare `python scripts/...` will not.

## Rules that are not obvious from the code

**The collector owns `data/`.** `collect.yml` fetches every draw and commits
`prize_tiers.csv`, `merged_lottery_data.csv`, `lotto_full_history.csv`,
`draw_pools.csv`, `mbw_validation.csv` and the site snapshot. Local copies of
those files are always a subset of the remote ones and are never worth keeping —
`scripts/monitoring/sync_collector_data.sh` discards them before pulling, and
refuses to run the model on a half-merged tree. Do not "fix" a conflict in them
by hand; take the collector's copy.

**`make site-data` belongs in the same commit as a model change.** The page
quotes the model, and CI runs `export_site_data.py --check` on every pull
request: a code change that moves any published figure cannot land without the
snapshot and the golden fixtures moving with it. That check now covers both
files — it silently skipped the fixtures until 2026-09-05.

**The popularity and sharing model exists twice**, in `lottery/ev.py` and again
in TypeScript under `site/src/data/`, because the page generates lines in the
visitor's browser. `site/src/__fixtures__/popularity-golden.json` is written by
the exporter and pins the port to Python — ratios AND shares. Change one side
and you must change the other; the fixture is what catches you.

**What-if runs never touch `outputs/predictions/latest.json`.** That file is the
ledger's record of the advisor's real verdict. `wheel_play.py` writes its own
timestamped file for the same reason.

**`api-dfe` gotcha**: `results/1/{n}` is the LEGACY single-round game (draws
≤ 3178) and silently drops round two. Two-round draws are `results/6/{n}`.
`results/1/latest` does alias the new game, which is why the collector can use
it for "latest" and `/6/` for backfill.

**Sales are an identity, not an estimate**, wherever `data/draw_pools.csv`
reaches: `(pool − previous pool) / 8.88%`, from the Game Procedures. Winner
counts (`N ≈ winners / P(tier)`) are the fallback and carry ±15% on a single
draw. `lotto_full_history.csv`'s `Jackpot` column is NOT the pool for this
purpose — for older draws it holds the advertised estimate, which is out by up
to 4%.

**`site.json` records the test count**, so almost every branch touching tests
conflicts there with every other one. Resolve by re-running `make site-data`,
never by editing the JSON.

## Where the reasoning lives

`audit-2026-09-05.md` is the current state of the analysis and the ranked
backlog; `plan.md` and `plan-ulepszen-2026-08.md` are its predecessors. Comments
in `lottery/ev.py` carry the provenance of every constant — read the comment
before changing a number, and update it when you do.
