#!/usr/bin/env python3
"""ROI ledger - the only metric that tells the truth about winning money.

Records every line actually played, settles it against real draw results
(both rounds since the June 2026 two-round format), and reports cumulative
spend vs winnings.

Usage:
  python scripts/roi_ledger.py add --draw-date next --lines "1 2 3 4 5 6; 7 8 9 10 11 12"
  python scripts/roi_ledger.py add --draw-date next --from-latest
  python scripts/roi_ledger.py settle
  python scripts/roi_ledger.py report
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
LEDGER_FILE = DATA_DIR / "ledger.csv"
FULL_HISTORY_FILE = DATA_DIR / "lotto_full_history.csv"
PRIZE_TIERS_FILE = DATA_DIR / "prize_tiers.csv"
LATEST_PREDICTIONS = Path("outputs/predictions/latest.json")

COST_PER_LINE = 2.0
TWO_ROUND_START = date(2026, 6, 7)

# Fallback per-winner prize estimates when prize_tiers.csv has no actuals.
# Keyed by (matches, bonus_hit). Jackpot/5+bonus are rough placeholders -
# real amounts are pari-mutuel and drawn from prize_tiers.csv when possible.
PRIZES_NEW_RULES = {(6, False): 2_000_000, (5, True): 250_000, (5, False): 1_000,
                    (4, False): 50, (3, False): 24, (2, False): 5}
PRIZES_OLD_RULES = {(6, False): 3_000_000, (5, True): 1_000_000, (5, False): 1_750,
                    (4, False): 140, (3, False): 30, (2, False): 0}

LEDGER_COLUMNS = [
    "added_at", "draw_date", "line", "cost", "settled",
    "matches_r1", "bonus_r1", "matches_r2", "bonus_r2", "prize", "prize_source",
]


def _load_ledger() -> pd.DataFrame:
    if LEDGER_FILE.exists():
        return pd.read_csv(LEDGER_FILE)
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _next_draw_date(today: date | None = None) -> date:
    """Lotto draws are on Wednesday (2) and Saturday (5)."""
    d = (today or date.today()) + timedelta(days=1)
    while d.weekday() not in (2, 5):
        d += timedelta(days=1)
    return d


def _parse_lines(lines_arg: str) -> list[list[int]]:
    lines = []
    for chunk in lines_arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        nums = sorted(int(t) for t in chunk.replace(",", " ").split())
        if len(set(nums)) != 6 or not all(1 <= n <= 59 for n in nums):
            raise ValueError(f"Invalid line: {chunk!r} (need 6 unique numbers 1-59)")
        lines.append(nums)
    if not lines:
        raise ValueError("No valid lines given")
    return lines


def _lines_from_latest() -> list[list[int]]:
    if not LATEST_PREDICTIONS.exists():
        raise FileNotFoundError(f"{LATEST_PREDICTIONS} not found - run a prediction first")
    payload = json.load(open(LATEST_PREDICTIONS))
    preds = payload.get("predictions", payload if isinstance(payload, list) else [])
    lines = [sorted(int(n) for n in p) for p in preds]
    if not lines:
        raise ValueError(f"No predictions found in {LATEST_PREDICTIONS}")
    return lines


def cmd_add(args) -> None:
    draw_date = (
        _next_draw_date() if args.draw_date == "next"
        else datetime.strptime(args.draw_date, "%Y-%m-%d").date()
    )
    lines = _lines_from_latest() if args.from_latest else _parse_lines(args.lines)

    ledger = _load_ledger()
    new_rows = pd.DataFrame([{
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "draw_date": draw_date.isoformat(),
        "line": " ".join(map(str, line)),
        "cost": args.cost_per_line,
        "settled": False,
        "matches_r1": None, "bonus_r1": None,
        "matches_r2": None, "bonus_r2": None,
        "prize": None, "prize_source": None,
    } for line in lines])
    ledger = pd.concat([ledger, new_rows], ignore_index=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    ledger.to_csv(LEDGER_FILE, index=False)
    print(f"Recorded {len(lines)} line(s) for draw {draw_date} "
          f"(cost £{len(lines) * args.cost_per_line:.2f}) in {LEDGER_FILE}")


def _prize_for(matches: int, bonus_hit: bool, draw_date: date,
               draw_number: int | None, round_no: int) -> tuple[float, str]:
    if matches < 2:
        return 0.0, "none"
    # Tier mapping: 1=jackpot, 2=5+bonus, 3=match5, 4=match4, 5=match3, 6=match2
    tier = {6: 1, 5: 2 if bonus_hit else 3, 4: 4, 3: 5, 2: 6}[matches]

    if draw_number is not None and PRIZE_TIERS_FILE.exists():
        tiers = pd.read_csv(PRIZE_TIERS_FILE)
        row = tiers[(tiers["draw_number"] == draw_number)
                    & (tiers["round"] == round_no) & (tiers["tier"] == tier)]
        if len(row) == 1 and row.iloc[0]["winners"] > 0:
            return float(row.iloc[0]["prize_total"] / row.iloc[0]["winners"]), "actual"

    table = PRIZES_NEW_RULES if draw_date >= TWO_ROUND_START else PRIZES_OLD_RULES
    key = (matches, bonus_hit if matches == 5 else False)
    return float(table.get(key, 0.0)), "estimate"


def cmd_settle(_args) -> None:
    ledger = _load_ledger()
    if ledger.empty:
        print("Ledger is empty - nothing to settle")
        return
    if not FULL_HISTORY_FILE.exists():
        raise FileNotFoundError("Run scripts/backfill_history.py first")
    history = pd.read_csv(FULL_HISTORY_FILE)

    settled_count = 0
    for idx, row in ledger[ledger["settled"] != True].iterrows():  # noqa: E712
        draws = history[history["Draw Date"] == row["draw_date"]]
        if draws.empty:
            continue
        line = set(int(t) for t in str(row["line"]).split())
        d_date = datetime.strptime(row["draw_date"], "%Y-%m-%d").date()
        total_prize, sources = 0.0, []
        for _, draw in draws.iterrows():
            drawn = {int(draw[f"Number_{i}"]) for i in range(1, 7)}
            matches = len(line & drawn)
            bonus_hit = int(draw["Bonus"]) in line
            round_no = int(draw.get("Round", 1))
            prize, source = _prize_for(
                matches, bonus_hit, d_date, int(draw["DrawNumber"]), round_no)
            total_prize += prize
            sources.append(source)
            ledger.loc[idx, f"matches_r{round_no}"] = matches
            ledger.loc[idx, f"bonus_r{round_no}"] = bonus_hit
        ledger.loc[idx, "prize"] = total_prize
        ledger.loc[idx, "prize_source"] = "+".join(sources)
        ledger.loc[idx, "settled"] = True
        settled_count += 1

    ledger.to_csv(LEDGER_FILE, index=False)
    print(f"Settled {settled_count} line(s). Unsettled remaining: "
          f"{int((ledger['settled'] != True).sum())}")  # noqa: E712


def cmd_report(_args) -> None:
    ledger = _load_ledger()
    if ledger.empty:
        print("Ledger is empty - record played lines with: roi_ledger.py add")
        return
    settled = ledger[ledger["settled"] == True]  # noqa: E712
    spent = float(ledger["cost"].sum())
    won = float(settled["prize"].fillna(0).sum())
    print("=" * 50)
    print("ROI LEDGER")
    print("=" * 50)
    print(f"Lines played:   {len(ledger)}  (settled: {len(settled)})")
    print(f"Total spent:    £{spent:.2f}")
    print(f"Total won:      £{won:.2f}")
    print(f"Net result:     £{won - spent:+.2f}")
    if spent:
        print(f"ROI:            {100.0 * (won - spent) / spent:+.1f}%")
    if not settled.empty:
        best = settled.loc[settled["prize"].fillna(0).idxmax()]
        for r in (1, 2):
            col = settled[f"matches_r{r}"].dropna()
            if not col.empty:
                dist = col.astype(int).value_counts().sort_index().to_dict()
                print(f"Round {r} match distribution: {dist}")
        print(f"Best line so far: {best['line']} on {best['draw_date']} -> £{best['prize']:.2f}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Record played lines")
    p_add.add_argument("--draw-date", default="next", help="YYYY-MM-DD or 'next'")
    p_add.add_argument("--lines", help='e.g. "1 2 3 4 5 6; 7 8 9 10 11 12"')
    p_add.add_argument("--from-latest", action="store_true",
                       help="Use lines from outputs/predictions/latest.json")
    p_add.add_argument("--cost-per-line", type=float, default=COST_PER_LINE)
    p_add.set_defaults(func=cmd_add)

    p_settle = sub.add_parser("settle", help="Settle lines against real results")
    p_settle.set_defaults(func=cmd_settle)

    p_report = sub.add_parser("report", help="Cumulative ROI report")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if args.command == "add" and not args.from_latest and not args.lines:
        parser.error("add requires --lines or --from-latest")
    args.func(args)


if __name__ == "__main__":
    main()
