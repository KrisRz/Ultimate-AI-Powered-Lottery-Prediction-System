"""Pricing the Must-Be-Won draw before it arrives.

The advisor could always say WHEN the cap forces a payout and never what it
would be worth when it got there, so the verdict on the one kind of draw worth
planning for existed only once the draw before it had been collected.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from lottery.ev import (
    JACKPOT_SHARE_OF_SALES,
    TICKET_PRICE,
    DrawConditions,
    exact_sales_baseline,
    forecast_must_be_won,
    mbw_uplift,
    must_be_won_outlook,
)

SAT_LINES, WED_LINES = 8_000_000, 5_000_000
BASE = 2_000_000.0


def _synthetic_pools(cycles: int = 4, first: date = date(2026, 1, 3)) -> pd.DataFrame:
    """Clean two-round-era cycles: base, five rollovers, Must-Be-Won, repeat.

    Real pools would do for most of this, but a fixture that states its own
    sales makes the arithmetic checkable rather than merely reproducible.
    """
    rows, when = [], first                      # first is a Saturday
    draw = 3300
    for _ in range(cycles):
        pool = BASE
        for step in range(1, 7):
            lines = SAT_LINES if when.weekday() == 5 else WED_LINES
            if step > 1:
                pool += lines * TICKET_PRICE * JACKPOT_SHARE_OF_SALES
            rows.append({"draw_number": draw, "draw_date": when.isoformat(),
                         "pool_gbp": round(pool, 2),
                         "rollover_count": step if step < 6 else None})
            draw += 1
            when += timedelta(days=4 if when.weekday() == 5 else 3)
    return pd.DataFrame(rows)


POOLS = _synthetic_pools()


class TestTheProjection:
    def test_each_further_draw_adds_its_own_weekday_sales(self):
        """The advertised estimate already holds the upcoming draw's sales, so
        only the draws AFTER it contribute."""
        saturday = date(2026, 9, 5)
        f = forecast_must_be_won(4, saturday, jackpot=7_000_000, pools_df=POOLS)
        assert f["expected_date"] == date(2026, 9, 9)      # the following Wednesday
        added = (WED_LINES * mbw_uplift(f["expected_date"])[0]
                 * TICKET_PRICE * JACKPOT_SHARE_OF_SALES)
        assert f["projected_pool"] == pytest.approx(7_000_000 + added)

    def test_the_uplift_lands_on_the_must_be_won_draw_only(self):
        """Its own sales fund its own pool, and they are the uplifted ones."""
        wednesday = date(2026, 9, 2)
        f = forecast_must_be_won(3, wednesday, jackpot=5_000_000, pools_df=POOLS)
        assert f["draws_away"] == 3                        # Wed -> Sat -> Wed
        ordinary = SAT_LINES * TICKET_PRICE * JACKPOT_SHARE_OF_SALES
        boosted = (WED_LINES * mbw_uplift(date(2026, 9, 9))[0]
                   * TICKET_PRICE * JACKPOT_SHARE_OF_SALES)
        assert f["projected_pool"] == pytest.approx(5_000_000 + ordinary + boosted)

    def test_no_pools_no_projection(self):
        f = forecast_must_be_won(4, date(2026, 9, 5), jackpot=7_000_000)
        assert f["projected_pool"] is None
        assert f["expected_date"] == date(2026, 9, 9)      # the rest still answers

    def test_a_thin_window_declines_rather_than_guesses(self):
        thin = POOLS.tail(3)
        assert forecast_must_be_won(4, date(2026, 9, 5), jackpot=7e6,
                                    pools_df=thin)["projected_pool"] is None


class TestTheOutlook:
    def _cond(self, jackpot: float = 7_000_000, **kw) -> DrawConditions:
        return DrawConditions(jackpot=jackpot, rollover_count=4,
                              draw_date=date(2026, 9, 5), **kw)

    def test_silent_when_the_draw_being_priced_is_the_must_be_won_one(self):
        assert must_be_won_outlook(self._cond(roll_down=True), POOLS) is None

    def test_prices_the_future_draw_on_its_own_weekday(self):
        """A Wednesday Must-Be-Won shares its pool between ~1.6x fewer lines
        than a Saturday one, so the threshold it must clear is much lower."""
        out = must_be_won_outlook(self._cond(), POOLS, date(2026, 9, 5))
        assert out["expected_date"] == date(2026, 9, 9)
        assert out["tickets_sold"] == int(
            exact_sales_baseline(POOLS, date(2026, 9, 9)) * mbw_uplift(date(2026, 9, 9))[0])
        assert out["break_even_jackpot"] < 12_000_000

    def test_the_verdict_follows_the_projected_pool(self):
        """Same draw, a pool that clears the threshold, and it says PLAY."""
        lean = must_be_won_outlook(self._cond(), POOLS, date(2026, 9, 5))
        rich = must_be_won_outlook(self._cond(jackpot=lean["break_even_jackpot"] * 1.5),
                                   POOLS, date(2026, 9, 5))
        assert lean["play"] is False
        assert rich["play"] is True
        assert rich["ev_best_line"] > lean["ev_best_line"]
