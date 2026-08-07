#!/usr/bin/env python3
"""Per-draw ticket sales from Merseyworld -> data/sales_history.csv.

Why this exists: N (lines sold) is the most sensitive input in the whole EV
model - on a Must-Be-Won draw the verdict is essentially J/N (audit
2026-08-06) - and until now N was only ever *inferred* from winner counts and
a flat 1.38x uplift. lottery.merseyworld.com publishes per-draw sales for
every Lotto draw since 1994-11-19, which makes it the only per-draw sales
source in existence and a second, independent estimator of N.

Source quirks, verified against the live site 2026-08-07:

* `sales=1` is the MAIN Lotto game and carries BOTH draw days. (`sales=2` is
  scratchcards - an earlier reading of the site's index as "1=Sat, 2=Wed"
  was wrong.)
* `year=0` returns the entire history in one request; per-year pages
  (`year=2026`) duplicate every row, so parsing dedupes by date either way.
* Figures are GBP of sales, not lines. Lines = sales / ticket price, and the
  price changed 1 -> 2 GBP with the 2013-10-05 relaunch draw. A 2-round-era
  line (since 2026) still costs 2 GBP for both rounds, so no further split.
* Since 2003-07-12 the operator publishes no per-draw sales; Merseyworld
  derives them from the published prize-fund percentages. They are estimates
  - which is exactly why `--validate` exists: it cross-checks them against
  the winner-count estimator on every draw where we hold tier data. Round
  figures like 11,000,000 are that estimation showing through, not errors.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import P_MATCH_2, P_MATCH_3, P_MATCH_4  # noqa: E402

DATA_DIR = Path("data")
SALES_FILE = DATA_DIR / "sales_history.csv"
FULL_HISTORY_FILE = DATA_DIR / "lotto_full_history.csv"
TIERS_HISTORY_FILE = DATA_DIR / "prize_tiers_history.csv"
TIERS_LIVE_FILE = DATA_DIR / "prize_tiers.csv"

SALES_URL = (
    "https://lottery.merseyworld.com/cgi-bin/lottery"
    "?sales=1&year=0&display=CSV"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

# First draw sold at 2 GBP a line (the "new Lotto" relaunch). Every draw
# before it sold at 1 GBP.
PRICE_2_START = date(2013, 10, 5)

# Winner-count probabilities for the per-draw N estimator: the same
# high-count tiers estimate_tickets_sold() uses, keyed by the tier codes of
# data/prize_tiers_history.csv / prize_tiers.csv.
TIER_PROBS = {4: P_MATCH_4, 5: P_MATCH_3, 6: P_MATCH_2}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ticket_price(d: date) -> float:
    return 2.0 if d >= PRICE_2_START else 1.0


def fetch_sales_page(url: str = SALES_URL) -> str:
    logger.info("Downloading per-draw sales history from Merseyworld...")
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_sales(html: str) -> pd.DataFrame:
    """Parse the CSV table inside Merseyworld's <PRE> block.

    Returns one row per draw date: draw_date, sales_gbp, pct_chg, lines_sold.
    """
    m = re.search(r"<PRE>(.*?)</PRE>", html, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No <PRE> block in sales response - format changed?")

    rows = []
    for ln in m.group(1).splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 6 or parts[0] not in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"):
            continue  # headers, blanks, footer
        try:
            when = datetime.strptime(f"{parts[1]} {parts[2]} {parts[3]}", "%d %b %Y").date()
            sales = int(parts[4])
        except ValueError as exc:
            raise ValueError(f"Unparseable sales row {ln!r}") from exc
        pct = None
        try:
            pct = float(parts[5])
        except ValueError:
            pass  # %Chg is cosmetic; never fail on it
        rows.append({"draw_date": when, "sales_gbp": sales, "pct_chg": pct})

    if not rows:
        raise ValueError("Sales table parsed to zero rows - format changed?")

    df = pd.DataFrame(rows).drop_duplicates(subset="draw_date").sort_values("draw_date")
    df["lines_sold"] = (
        df.apply(lambda r: r["sales_gbp"] / ticket_price(r["draw_date"]), axis=1)
        .round()
        .astype(int)
    )
    return df.reset_index(drop=True)


def attach_draw_numbers(sales: pd.DataFrame,
                        history_file: Path = FULL_HISTORY_FILE) -> pd.DataFrame:
    """Left-join draw_number from the backfilled draw history, by date."""
    if not history_file.exists():
        logger.warning("%s missing - run `make backfill`; draw_number left empty",
                       history_file)
        sales["draw_number"] = pd.NA
        return sales
    hist = pd.read_csv(history_file, parse_dates=["Draw Date"])
    lookup = (
        hist.drop_duplicates(subset="Draw Date")      # 2-round era: 1 row per round
        .assign(draw_date=lambda d: d["Draw Date"].dt.date)
        .set_index("draw_date")["DrawNumber"]
    )
    sales["draw_number"] = sales["draw_date"].map(lookup).astype("Int64")
    unmatched = int(sales["draw_number"].isna().sum())
    if unmatched:
        # Tonight's draw appears in the sales table before the results do;
        # more than a couple missing means the histories have drifted apart.
        logger.warning("%d sales rows have no draw in %s", unmatched, history_file)
    return sales


def winner_count_estimates(tiers_files=None) -> pd.DataFrame:
    """Independent per-draw N estimate: median of winners / P(tier) over the
    high-count tiers (Match 4/3/2), every round of every draw we hold.

    Same estimator estimate_tickets_sold() applies to a recent window, but
    resolved per draw so each historical draw can be compared 1:1 with the
    Merseyworld figure.
    """
    if tiers_files is None:     # resolved at call time so tests can repoint them
        tiers_files = (TIERS_HISTORY_FILE, TIERS_LIVE_FILE)
    frames = []
    for f in tiers_files:
        if Path(f).exists():
            df = pd.read_csv(f, usecols=["draw_number", "tier", "winners"])
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["draw_number", "est_lines"])
    tiers = pd.concat(frames).drop_duplicates()
    tiers = tiers[tiers["tier"].isin(TIER_PROBS) & (tiers["winners"] > 0)]
    tiers["est"] = tiers.apply(lambda r: r["winners"] / TIER_PROBS[int(r["tier"])], axis=1)
    out = tiers.groupby("draw_number")["est"].median().round().astype(int)
    return out.rename("est_lines").reset_index()


def validate(sales: pd.DataFrame) -> pd.DataFrame:
    """Merseyworld lines vs winner-count estimate, draw by draw.

    Returns the merged frame; prints the agreement summary. The ratio should
    sit near 1.0 - a stable offset means one source is biased (and the
    winner-count side has no price-era assumptions, so a jump at 2013-10-05
    would point at the price constant instead).
    """
    est = winner_count_estimates()
    if est.empty:
        logger.warning("No tier data for validation - run `make backfill` first")
        return sales
    merged = sales.merge(est, on="draw_number", how="inner")
    merged["ratio"] = merged["lines_sold"] / merged["est_lines"]

    q = merged["ratio"].quantile([0.25, 0.5, 0.75])
    print(f"\nValidation: {len(merged)} draws with both sources")
    print(f"  lines(Merseyworld) / lines(winner-count estimate):")
    print(f"  median {q[0.5]:.3f}   IQR {q[0.25]:.3f} - {q[0.75]:.3f}")
    for since, label in ((date(2015, 10, 10), "59-ball era"),
                         (date(2026, 6, 7), "2-round era")):
        part = merged[merged["draw_date"] >= since]
        if len(part):
            print(f"  {label}: median {part['ratio'].median():.3f}  (n={len(part)})")
    worst = merged.loc[(merged["ratio"] - 1).abs().nlargest(5).index]
    print("  largest disagreements:")
    for _, r in worst.iterrows():
        print(f"    draw {r['draw_number']} {r['draw_date']}: "
              f"MW {r['lines_sold']:,} vs est {r['est_lines']:,} (x{r['ratio']:.2f})")
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-file", type=Path,
                    help="parse a saved HTML page instead of fetching (tests/offline)")
    ap.add_argument("--validate", action="store_true",
                    help="cross-check against the winner-count N estimator")
    ap.add_argument("--out", type=Path, default=SALES_FILE)
    args = ap.parse_args()

    html = (args.from_file.read_text()
            if args.from_file else fetch_sales_page())
    sales = attach_draw_numbers(parse_sales(html))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["draw_date", "draw_number", "sales_gbp", "lines_sold", "pct_chg"]
    sales[cols].to_csv(args.out, index=False)
    logger.info("Wrote %d draws (%s - %s) to %s", len(sales),
                sales["draw_date"].min(), sales["draw_date"].max(), args.out)

    if args.validate:
        validate(sales)
    return 0


if __name__ == "__main__":
    sys.exit(main())
