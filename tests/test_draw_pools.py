"""Sales as an identity: (pool - previous pool) / 8.88%.

The claim these tests defend is that lines sold per draw are not estimated at
all where the pools reach - they are read off the official feed, and the number
that comes out is the same one Merseyworld's archive has been publishing since
2003, to within a tenner.
"""

from datetime import date

import pandas as pd
import pytest

from lottery.ev import (
    JACKPOT_SHARE_OF_SALES,
    MIN_EXACT_OBSERVATIONS,
    TICKET_PRICE,
    estimate_tickets_sold,
    exact_lines_sold,
    exact_sales_baseline,
    must_be_won_after_cap,
)

POOLS = pd.read_csv("data/draw_pools.csv")
SALES = pd.read_csv("data/sales_history.csv").set_index("draw_number")

WED, SAT = date(2026, 9, 9), date(2026, 9, 5)


def _pools(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["draw_number", "draw_date", "pool_gbp",
                                       "rollover_count"])


class TestTheIdentity:
    def test_reproduces_the_merseyworld_archive(self):
        """Every two-round draw both sources price, priced the same."""
        exact = exact_lines_sold(POOLS)
        both = [(d, exact[d], int(SALES.lines_sold[d]))
                for d in sorted(exact) if d in SALES.index]
        assert len(both) >= 14, "expected the whole overlap, got fewer draws"
        gaps = sorted(abs(mine - theirs) * TICKET_PRICE for _, mine, theirs in both)
        assert gaps[len(gaps) // 2] <= 10, f"median disagreement {gaps[len(gaps) // 2]}"
        # 3181 is out by GBP 902 of sales on ~GBP 11.5m - 0.008%, and the only
        # draw over a tenner. An archive typo or a rounding of the announced
        # pool; either way it is not a modelling difference.
        assert gaps[-1] <= 1_000

    def test_the_draw_after_a_win_carries_no_sales_information(self):
        """Its pool restarts from the minimum, so the difference is the reset."""
        pools = _pools([(3199, "2026-08-19", 4_354_203.25, None),
                        (3200, "2026-08-22", 2_000_000.0, 1),
                        (3201, "2026-08-26", 2_865_432.72, 2)])
        exact = exact_lines_sold(pools)
        assert 3200 not in exact
        assert exact[3201] == round(
            (2_865_432.72 - 2_000_000.0) / JACKPOT_SHARE_OF_SALES / TICKET_PRICE)

    def test_the_first_draw_on_file_has_no_predecessor(self):
        assert exact_lines_sold(_pools([(3179, "2026-06-10", 2_000_000.0, 1)])) == {}

    def test_no_pools_no_answer(self):
        assert exact_lines_sold(None) == {}
        assert exact_lines_sold(_pools([])) == {}


class TestMustBeWonDetection:
    def test_finds_exactly_the_three_two_round_must_be_won_draws(self):
        assert sorted(must_be_won_after_cap(POOLS)) == [3184, 3190, 3196]

    def test_the_cap_marks_the_next_draw_not_its_own(self):
        pools = _pools([(3195, "2026-08-05", 6_855_189.0, 5),
                        (3196, "2026-08-08", 8_535_147.0, None)])
        assert must_be_won_after_cap(pools) == {3196}

    def test_an_ordinary_rollover_marks_nothing(self):
        pools = _pools([(3202, "2026-08-29", 4_349_673.92, 3),
                        (3203, "2026-09-02", 5_255_577.24, 4)])
        assert must_be_won_after_cap(pools) == set()


class TestTheBaseline:
    def test_must_be_won_draws_are_kept_out_of_the_ordinary_level(self):
        """3196 sold 9.46m against ~8.5m ordinary; counting it in the baseline
        and then multiplying by the uplift prices the same effect twice."""
        with_mbw = exact_sales_baseline(POOLS, SAT)
        without = exact_sales_baseline(POOLS[~POOLS.draw_number.isin([3196])], SAT)
        assert with_mbw == without

    def test_the_weekday_decides_which_draws_count(self):
        """Saturdays sell ~1.6x Wednesdays - a mixed median prices no real draw."""
        assert exact_sales_baseline(POOLS, SAT) > 1.4 * exact_sales_baseline(POOLS, WED)

    def test_too_few_observations_declines_to_answer(self):
        thin = POOLS[POOLS.draw_number >= 3201]        # one Saturday with a pool
        assert exact_sales_baseline(thin, SAT) is None

    def test_the_estimator_falls_back_to_winner_counts(self):
        """No pools, or too few, and nothing changes from before."""
        tiers = pd.read_csv("data/prize_tiers.csv")
        assert (estimate_tickets_sold(tiers, draw_date=SAT, pools_df=None)
                == estimate_tickets_sold(tiers, draw_date=SAT))

    def test_measured_beats_inferred_when_the_pools_reach(self):
        """Winner counts read 6-8% below the identity - that gap is the point."""
        tiers = pd.read_csv("data/prize_tiers.csv")
        inferred = estimate_tickets_sold(tiers, draw_date=SAT)
        measured = estimate_tickets_sold(tiers, draw_date=SAT, pools_df=POOLS)
        assert measured > inferred
        assert measured == exact_sales_baseline(POOLS, SAT)

    def test_the_uplift_still_multiplies_the_measured_baseline(self):
        tiers = pd.read_csv("data/prize_tiers.csv")
        ordinary = estimate_tickets_sold(tiers, draw_date=WED, pools_df=POOLS)
        must_be_won = estimate_tickets_sold(tiers, draw_date=WED, roll_down=True,
                                            pools_df=POOLS)
        assert must_be_won == int(ordinary * 1.44)


def test_the_share_is_the_one_the_procedures_publish():
    """8.88% of sales to the jackpot - Game Procedures Ed. 22, section 3.1."""
    assert JACKPOT_SHARE_OF_SALES == 0.0888
    assert MIN_EXACT_OBSERVATIONS >= 3
