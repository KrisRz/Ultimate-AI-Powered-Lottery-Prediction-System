#!/usr/bin/env python3
"""Export the public site's data snapshot: site/public/data/site.json.

The page under site/ makes quantitative claims about this model. This script is
what keeps them true: every number the site renders is computed here, by the
same functions the advisor and the alert email use. Nothing on the page is
typed by hand.

Determinism is the whole design constraint. Re-running on unchanged inputs must
produce a BYTE-IDENTICAL file, so CI can regenerate it and `git diff
--exit-code` to catch a page that has quietly drifted from the model behind it
(recalibrate `number_weight` and the published popularity heatmap goes stale in
the same commit). Two consequences:

  * No wall-clock reads. The priced draw is derived from the collected data -
    see `priced_at` - not from today's date, or the file would rewrite itself
    every time the draw date rolled over.
  * Floats are rounded before serialization. numpy's least-squares lands on
    different last bits under macOS Accelerate and Linux OpenBLAS, so an
    unrounded regression coefficient would diff between a laptop and a runner.

Usage:
  python scripts/export_site_data.py            # write the snapshot
  python scripts/export_site_data.py --check    # fail if it would change
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from math import exp
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    ARITHMETIC_MULT,
    BIRTHDAY_MULT,
    CONSECUTIVE_MULT,
    MEAN_WEIGHT,
    N_BALLS,
    N_PICK,
    POPULARITY_NORMALIZATION,
    P_JACKPOT,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    P_MATCH_5,
    P_MATCH_5_BONUS,
    ROLLOVER_CAP,
    TICKET_PRICE,
    TOTAL_COMBOS,
    DrawConditions,
    best_unpopular_reference_line,
    break_even_jackpot,
    expected_cowinner_share,
    mbw_type,
    mbw_uplift,
    number_weight,
    popularity_ratio,
    should_play,
)
from scripts.calibrate_popularity import (  # noqa: E402
    add_multiplier,
    drawn_features,
    load_joined,
    per_number_regression,
)
from scripts.dashboard import _cumavg  # noqa: E402
from scripts.ev_play import next_draw_conditions  # noqa: E402

SCHEMA_VERSION = 1

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
MERGED_FILE = Path("data/merged_lottery_data.csv")
FULL_HISTORY_FILE = Path("data/lotto_full_history.csv")
VALIDATION_DIR = Path("outputs/validation")
OUT_FILE = Path("site/public/data/site.json")

# The backtest's source lives under outputs/, which is gitignored, so CI could
# not regenerate this file's backtest section from a fresh clone. Rather than
# make the whole export unreproducible for one block, the validation runs are
# pruned to the fields the page actually uses (2.1 MB of four runs becomes
# ~20 KB) and that extract is committed. Refresh it with --refresh-backtest
# after `make backtest`.
BACKTEST_SRC = Path("site/data-src/backtest.json")

# Rounding. Model coefficients are exempt: `b` is ~1e-07, so rounding it to the
# general 4 dp would collapse it to zero and put break-even at infinity.
DP = 4
PROB_DP = 12
EXACT_KEYS = frozenset({"a", "b"})
PROB_KEYS = frozenset({"probability"})


# --- serialization ----------------------------------------------------------

def _plain(value):
    """numpy scalar -> python scalar. Leaves everything else alone."""
    return value.item() if hasattr(value, "item") else value


def _round_tree(obj, key: str | None = None):
    """Round every float in the tree, by key-specific precision."""
    obj = _plain(obj)
    if isinstance(obj, dict):
        return {k: _round_tree(v, k) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_tree(v, key) for v in obj]
    if isinstance(obj, bool) or not isinstance(obj, float):
        return obj
    if key in EXACT_KEYS:
        return obj
    return round(obj, PROB_DP if key in PROB_KEYS else DP)


def serialize(payload: dict) -> str:
    """Stable JSON: sorted keys, fixed indent, ASCII, trailing newline."""
    rounded = _round_tree(payload)
    return json.dumps(rounded, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def gbp(amount: float) -> int:
    """Whole pounds. Every GBP figure the page shows is >= four digits, so the
    pence are noise that would only make the committed file diff on rounding."""
    return int(round(amount))


# --- deterministic clock ----------------------------------------------------

def priced_at(tiers: pd.DataFrame):
    """The moment the snapshot prices conditions at, derived from the data.

    The day after the last collected draw: `upcoming_draw_date` then resolves
    it to the next Wednesday/Saturday, which is exactly the draw a ticket
    bought after that collection would enter. Using the real clock here would
    give the same answer today and a different one tomorrow, and the committed
    file has to be reproducible.
    """
    last = pd.to_datetime(tiers["draw_date"]).max().date()
    return last + timedelta(days=1)


# --- section 1: the hook ----------------------------------------------------

def build_hook() -> dict:
    tiers = [
        ("jackpot", "Match 6", P_JACKPOT),
        ("match_5_bonus", "Match 5 + bonus", P_MATCH_5_BONUS),
        ("match_5", "Match 5", P_MATCH_5),
        ("match_4", "Match 4", P_MATCH_4),
        ("match_3", "Match 3", P_MATCH_3),
        ("match_2", "Match 2", P_MATCH_2),
    ]
    return {
        "total_combinations": TOTAL_COMBOS,
        "n_balls": N_BALLS,
        "n_pick": N_PICK,
        "rounds_per_draw": 2,
        "ticket_price_gbp": TICKET_PRICE,
        "odds": [
            {"key": key, "label": label, "probability": p, "one_in": round(1.0 / p, 1)}
            for key, label, p in tiers
        ],
    }


# --- section 2: can you predict the numbers? ---------------------------------

def refresh_backtest() -> dict:
    """Prune the newest validation run per method into the committed extract.

    One entry per method plus the random baseline, each carrying its
    significance block and its per-draw match counts. Everything else in a
    validation file - the plot path, the per-method metrics the page does not
    show - is dropped.
    """
    runs = sorted(VALIDATION_DIR.glob("validation_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)

    newest: dict[str, dict] = {}
    for path in runs:
        run = json.loads(path.read_text())
        if not (run.get("significance") and run.get("series")):
            continue
        newest.setdefault(run["method"], run)

    if not newest:
        raise SystemExit(f"No usable validation runs in {VALIDATION_DIR} - run `make backtest`")

    canonical = newest[sorted(newest)[0]]
    dates = [point["date"] for point in canonical["series"]["random"]]

    methods: dict[str, dict] = {}
    for name, run in sorted(newest.items()):
        # Every run must be scored on the same draws, or the chart would put
        # four differently-shaped lines on one axis and invite a comparison
        # that is not there.
        for series_name, series in run["series"].items():
            if [p["date"] for p in series] != dates:
                raise SystemExit(
                    f"{name}'s {series_name} series is on a different draw grid; "
                    "re-run all methods over one window before refreshing")
        methods[name] = {
            "significance": run["significance"][name],
            "matches": [int(p["matches"]) for p in run["series"][name]],
        }

    # The random baseline comes from the canonical run. Each validation file
    # carries its own, and they differ - they are independent draws from the
    # same null. Picking one keeps the page's baseline a single honest series
    # rather than an average of four.
    methods["random"] = {
        "significance": canonical["significance"]["random"],
        "matches": [int(p["matches"]) for p in canonical["series"]["random"]],
    }

    extract = {
        "steps": canonical["steps"],
        "lookback": canonical["lookback"],
        "n_sim": canonical["significance"]["random"]["n_sim"],
        "expected_random_avg": canonical["significance"]["random"]["expected_random_avg"],
        "dates": dates,
        "methods": methods,
    }

    BACKTEST_SRC.parent.mkdir(parents=True, exist_ok=True)
    BACKTEST_SRC.write_text(serialize(extract))
    return extract


def build_backtest() -> dict:
    """The finding that has to land before any of the arithmetic matters.

    Four prediction methods, walk-forward over the same 930 draws, against the
    no-skill mean of 36/59. Every one of them sits inside the baseline's own
    confidence interval. The page shows the cumulative averages tangled
    together because that picture is harder to argue with than a p-value.
    """
    if not BACKTEST_SRC.exists():
        raise SystemExit(
            f"{BACKTEST_SRC} is missing - run "
            "`python scripts/export_site_data.py --refresh-backtest`")

    src = json.loads(BACKTEST_SRC.read_text())
    expected = src["expected_random_avg"]

    methods = []
    for name, entry in sorted(src["methods"].items()):
        sig = entry["significance"]
        low, high = sig["observed_avg_ci95"]
        methods.append({
            "name": name,
            "is_baseline": name == "random",
            "observed_avg": sig["observed_avg"],
            "ci95": [low, high],
            "p_value_avg": sig["p_value_avg"],
            "rate_3plus": sig["observed_3plus_rate"],
            "p_value_3plus": sig["p_value_3plus"],
            # The honest reading of a p-value this size: the method's score is
            # what the null model produces anyway.
            "beats_random": sig["p_value_avg"] < 0.05,
        })

    return {
        "steps": src["steps"],
        "lookback": src["lookback"],
        "n_sim": src["n_sim"],
        "expected_random_avg": expected,
        "date_from": src["dates"][0],
        "date_to": src["dates"][-1],
        "methods": methods,
        "series": {
            "dates": src["dates"],
            "cumulative_avg": {
                name: _cumavg(entry["matches"])
                for name, entry in sorted(src["methods"].items())
            },
        },
    }


# --- section 3: expected value ----------------------------------------------

def affine(line, cond: DrawConditions) -> tuple:
    """(a, b) such that EV(J) = a + b*J under `cond`.

    `line_ev` is exactly affine in the jackpot: the pool enters only through
    the co-winner share - which depends on tickets sold, not on J - and through
    the roll-down term J/N. Shipping two coefficients instead of a sampled
    curve makes the site's slider exact rather than interpolated, and removes
    any way for the chart to disagree with the model.

    These are the same two quantities `break_even_jackpot` inverts, so
    -a/b == break_even_jackpot(cond) by construction (asserted in the tests).
    """
    a = cond.rounds * cond.prizes.ev_per_round() - cond.ticket_price
    b = cond.rounds * P_JACKPOT * expected_cowinner_share(
        line, cond.tickets_sold * cond.rounds)
    if cond.roll_down:
        b += exp(-cond.tickets_sold * cond.rounds * P_JACKPOT) / max(cond.tickets_sold, 1)
    return a, b


def _regime(key: str, label: str, cond: DrawConditions, line) -> dict:
    a, b = affine(line, cond)
    return {
        "key": key,
        "label": label,
        "tickets_sold": int(cond.tickets_sold),
        "roll_down": bool(cond.roll_down),
        "rounds": cond.rounds,
        "a": a,
        "b": b,
        "break_even_jackpot": gbp(break_even_jackpot(cond, line)),
    }


def build_ev(live: DrawConditions, ordinary: DrawConditions,
             mbw: DrawConditions) -> dict:
    line = best_unpopular_reference_line()
    verdict = should_play(live, threshold=0.0)

    # The sales band is the honest part of a roll-down verdict: EV is dominated
    # by J/N, so the whole answer turns on one estimated number. Ship a regime
    # per quartile so the reader can move it themselves.
    up_mid, up_p25, up_p75 = mbw_uplift(mbw.draw_date)
    baseline = mbw.tickets_sold / up_mid
    band = {}
    for name, uplift in (("p25", up_p25), ("p75", up_p75)):
        cond = DrawConditions(
            jackpot=mbw.jackpot, tickets_sold=max(int(baseline * uplift), 1),
            roll_down=True, rounds=mbw.rounds, ticket_price=mbw.ticket_price,
            prizes=mbw.prizes, rollover_count=mbw.rollover_count,
            draw_date=mbw.draw_date)
        a, b = affine(line, cond)
        band[name] = {
            "uplift": uplift,
            "tickets_sold": cond.tickets_sold,
            "a": a,
            "b": b,
            "break_even_jackpot": gbp(break_even_jackpot(cond, line)),
        }

    return {
        "reference_line": list(line),
        "reference_popularity": popularity_ratio(line),
        "fixed_prizes": {
            "match_5_bonus": live.prizes.match_5_bonus,
            "match_5": live.prizes.match_5,
            "match_4": live.prizes.match_4,
            "match_3": live.prizes.match_3,
            "match_2": live.prizes.match_2,
            "source": live.prizes.source,
            "ev_per_round": live.prizes.ev_per_round(),
        },
        "slider": {"min_gbp": 2_000_000, "max_gbp": 50_000_000, "step_gbp": 100_000},
        "regimes": [
            _regime("ordinary", "Ordinary draw", ordinary, line),
            _regime("mbw", "Must-Be-Won roll-down", mbw, line),
        ],
        "mbw_sales_band": band,
        "live": {
            "draw_date": live.draw_date.isoformat(),
            "jackpot_gbp": gbp(live.jackpot),
            "tickets_sold": int(live.tickets_sold),
            "roll_down": bool(live.roll_down),
            "rollover_count": int(live.rollover_count),
            "rollover_cap": ROLLOVER_CAP,
            "mbw_type": mbw_type(live.roll_down, live.rollover_count),
            "ev_best_line": verdict["ev_best_line"],
            "break_even_jackpot": gbp(verdict["break_even_jackpot"]),
            "verdict": "PLAY" if verdict["play"] else "SKIP",
            "robust": bool((verdict["sales_sensitivity"] or {}).get("robust", False)),
        },
    }


# --- section 5: who else is holding your line -------------------------------

def build_popularity() -> dict:
    """The popularity model, as installed and as recovered from the data.

    Both are shipped on purpose. The installed model is a three-band step
    function; the per-number regression it was distilled from is noisier and
    shows a "lucky 7/9/11" bump that the calibration concluded sits within
    noise. Publishing the raw fit next to the simplification is the honest way
    to show a modelling decision instead of asserting one.
    """
    df = drawn_features(add_multiplier(load_joined()))
    recovered = per_number_regression(df)

    ranked = sorted(enumerate(recovered, start=1), key=lambda kv: kv[1], reverse=True)

    # The single clearest evidence that the effect is real: draws made entirely
    # of birthday-range numbers produce ~2.5x the Match-3 winners per ticket
    # that all-high draws do, and the trend is monotone across every bucket.
    by_low31 = df.groupby("n_low31")["multiplier"].agg(["size", "mean"]).reset_index()

    examples = [
        ([1, 2, 3, 4, 5, 6], "consecutive run and an arithmetic line"),
        ([5, 10, 15, 20, 25, 30], "arithmetic, all birthday-range"),
        ([7, 14, 21, 28, 35, 42], "arithmetic, the 7 times table"),
        ([3, 7, 11, 19, 23, 31], "all birthday-range, no pattern"),
        ([1, 7, 13, 25, 31, 42], "mixed"),
        (best_unpopular_reference_line(), "the reference line this model plays"),
    ]

    return {
        "recovered": [float(w) for w in recovered],
        "recovered_range": [float(recovered.min()), float(recovered.max())],
        "n_observations": int(len(df)),
        "installed_step": [
            {"from": 1, "to": 12, "weight": number_weight(1)},
            {"from": 13, "to": 31, "weight": number_weight(13)},
            {"from": 32, "to": N_BALLS, "weight": number_weight(N_BALLS)},
        ],
        "model": {
            "mean_weight": MEAN_WEIGHT,
            "normalization": POPULARITY_NORMALIZATION,
            "arithmetic_mult": ARITHMETIC_MULT,
            "consecutive_mult": CONSECUTIVE_MULT,
            "birthday_mult": BIRTHDAY_MULT,
            "consecutive_run": 3,
            "birthday_max": 31,
        },
        "most_played": [{"number": n, "weight": float(w)} for n, w in ranked[:6]],
        "least_played": [{"number": n, "weight": float(w)} for n, w in ranked[-6:][::-1]],
        "examples": [
            {"line": list(line), "ratio": popularity_ratio(line), "note": note}
            for line, note in examples
        ],
        "match3_multiplier_by_low31": [
            {
                "n_low31": int(r["n_low31"]),
                "draws": int(r["size"]),
                "mean_multiplier": float(r["mean"]),
            }
            for _, r in by_low31.iterrows()
        ],
    }


# --- assembly ---------------------------------------------------------------

def build_payload() -> dict:
    if not PRIZE_TIERS_FILE.exists():
        raise SystemExit(f"{PRIZE_TIERS_FILE} is missing - run `make play` first")

    tiers = pd.read_csv(PRIZE_TIERS_FILE)
    now = priced_at(tiers)

    live = next_draw_conditions(now=now)
    ordinary = next_draw_conditions(force_ordinary=True, now=now)
    mbw = next_draw_conditions(force_roll_down=True, now=now)

    merged = pd.read_csv(MERGED_FILE)
    full = pd.read_csv(FULL_HISTORY_FILE)

    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot": {
            "as_of_draw_date": live.draw_date.isoformat(),
            "data_through": str(pd.to_datetime(merged["Draw Date"]).max().date()),
            "draws_all_time": int(full["DrawNumber"].nunique()),
            "draws_59_ball_era": int(len(merged)),
        },
        "hook": build_hook(),
        "backtest": build_backtest(),
        "ev": build_ev(live, ordinary, mbw),
        "popularity": build_popularity(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Exit 1 if the snapshot on disk is out of date")
    parser.add_argument("--refresh-backtest", action="store_true",
                        help=f"Rebuild {BACKTEST_SRC} from outputs/validation "
                             "(needs a local `make backtest`), then export")
    args = parser.parse_args()

    if args.refresh_backtest:
        extract = refresh_backtest()
        print(f"Wrote {BACKTEST_SRC} "
              f"({len(extract['methods'])} methods x {extract['steps']} draws)")

    text = serialize(build_payload())

    if args.check:
        current = OUT_FILE.read_text() if OUT_FILE.exists() else ""
        if current != text:
            print(f"{OUT_FILE} is stale - run `make site-data` and commit the result",
                  file=sys.stderr)
            return 1
        print(f"{OUT_FILE} is up to date")
        return 0

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(text)
    print(f"Wrote {OUT_FILE} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
