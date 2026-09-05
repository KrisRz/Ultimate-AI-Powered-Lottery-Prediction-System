"""The uplift calibration on exact pools.

The constant this produces multiplies the sales baseline on every Must-Be-Won
verdict, and lowering it RAISES expected value - so what matters here is not
only that the arithmetic is right but that the script declines to measure what
it cannot measure, instead of quietly measuring it a different way.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from lottery.ev import (
    JACKPOT_SHARE_OF_SALES,
    TICKET_PRICE,
    exact_sales_baseline,
)
from scripts.calibrate_mbw_uplift import exact_era_uplifts

SAT_LINES, WED_LINES = 8_000_000, 5_000_000
MBW_LINES = 9_600_000                      # a Saturday Must-Be-Won, uplift 1.2
BASE = 2_000_000.0


def _cycles(count: int, first: date = date(2026, 1, 3)) -> list:
    """Base, five rollovers, a Must-Be-Won, repeat - as the cap actually runs."""
    rows, when, draw = [], first, 3300
    for _ in range(count):
        pool = BASE
        for step in range(1, 7):
            ordinary = SAT_LINES if when.weekday() == 5 else WED_LINES
            lines = MBW_LINES if step == 6 else ordinary
            if step > 1:
                pool += lines * TICKET_PRICE * JACKPOT_SHARE_OF_SALES
            rows.append({"draw_number": draw, "draw_date": when.isoformat(),
                         "pool_gbp": round(pool, 2),
                         "rollover_count": step if step < 6 else None})
            draw += 1
            when += timedelta(days=4 if when.weekday() == 5 else 3)
    return rows


class TestExactEraUplifts:
    def test_measures_a_must_be_won_against_its_own_weekday(self):
        pools = pd.DataFrame(_cycles(4))
        priced = [r for r in exact_era_uplifts(pools) if r["uplift"] is not None]
        assert priced, "expected at least one priceable Must-Be-Won draw"
        for row in priced:
            expected = exact_sales_baseline(
                pools[pools["draw_number"] < row["draw"]], row["date"])
            assert row["baseline"] == expected
            assert row["uplift"] == pytest.approx(row["lines"] / expected)

    def test_declines_the_draws_it_cannot_price(self):
        """The first Must-Be-Won on file has too few same-weekday draws before
        it. Draw 3184 is the live case: two prior exact Saturdays, not three."""
        early = pd.DataFrame(_cycles(1))
        rows = exact_era_uplifts(early)
        assert rows and all(r["uplift"] is None for r in rows)
        assert all("too few" in r["why"] for r in rows)

    def test_a_reset_draw_is_named_as_unpriceable_not_skipped(self):
        """Silence would read as "no Must-Be-Won draws", which is a different
        statement from "one happened and its pool cannot price it"."""
        pools = pd.DataFrame(_cycles(2))
        pools.loc[pools["draw_number"] == 3305, "pool_gbp"] = BASE
        rows = exact_era_uplifts(pools)
        reset = [r for r in rows if r["draw"] == 3305]
        assert reset and reset[0]["why"] == "pool reset - not priceable"

    def test_no_must_be_won_draws_is_an_empty_answer_not_a_crash(self):
        ordinary = pd.DataFrame(_cycles(2))
        ordinary["rollover_count"] = 1          # nothing ever reaches the cap
        assert exact_era_uplifts(ordinary) == []
