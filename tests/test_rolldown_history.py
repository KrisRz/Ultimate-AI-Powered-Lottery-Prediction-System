"""Replaying the roll-down archive.

These figures reach the public page, and until this module existed they lived
only as prose in plan.md. The tests below are what makes them a claim rather
than a memo.
"""

import pandas as pd
import pytest

from lottery.ev import ROLLOVER_CAP
from scripts.rolldown_history import (
    CAVEAT,
    replay_rolldowns,
    rollover_streaks,
    summarise,
)


@pytest.fixture(scope="module")
def rows():
    return replay_rolldowns()


@pytest.fixture(scope="module")
def stats(rows):
    return summarise(rows)


class TestRolloverStreaks:
    def test_a_rolldown_ends_the_sequence(self):
        """The regression this module was born with.

        A roll-down leaves Match 6 with no winners, so counting only outright
        wins never resets the streak and it runs away - the first version
        produced a streak of 29 against a cap of 5. Both endings must reset it.
        """
        full = pd.DataFrame({
            "DrawNumber": [1, 2, 3, 4, 5, 6],
            "JackpotWins": [0, 0, 0, 0, 0, 0],
        })
        # Draw 4 rolled down; without that knowledge the streak just climbs.
        streaks = rollover_streaks(full, boosted={4})
        assert streaks[4] == 3
        assert streaks[5] == 0, "the roll-down should have reset the count"
        assert streaks[6] == 1

    def test_an_outright_win_also_ends_it(self):
        full = pd.DataFrame({
            "DrawNumber": [1, 2, 3, 4],
            "JackpotWins": [0, 0, 1, 0],
        })
        streaks = rollover_streaks(full, boosted=set())
        assert streaks[3] == 2
        assert streaks[4] == 0

    def test_no_rolldown_in_the_capped_era_exceeds_the_cap(self, rows):
        """Every cap-driven roll-down should sit exactly at the cap, not past
        it. A streak above ROLLOVER_CAP means the reconstruction is wrong,
        because the rule forces a payout at the cap."""
        tiers = pd.read_csv("data/prize_tiers_history.csv", parse_dates=["draw_date"])
        full = pd.read_csv("data/lotto_full_history.csv")
        from scripts.calibrate_mbw_uplift import rolldown_draws

        boosted = rolldown_draws(tiers)
        streaks = rollover_streaks(full, boosted)
        at_rolldowns = [streaks.get(d, 0) for d in boosted]
        assert max(at_rolldowns) <= ROLLOVER_CAP


class TestReplay:
    def test_every_row_is_priceable(self, rows):
        assert rows, "no roll-downs found - the detector or the archive changed"
        for row in rows:
            assert row["pool_gbp"] > 0, row
            assert row["tickets_sold"] > 0, row
            assert row["date"], row
            assert isinstance(row["cap_driven"], bool)

    def test_rows_are_sorted_and_unique(self, rows):
        numbers = [r["draw_number"] for r in rows]
        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers))

    def test_cap_driven_count_matches_the_archive(self, stats):
        """53 of the roll-downs followed a jackpot reaching the cap. That
        figure is independently recorded in the project's own notes, so it
        doubles as a check on the streak reconstruction."""
        assert stats["cap_driven"] == 53
        assert stats["detected"] == stats["cap_driven"] + stats["special_event"]

    def test_most_rolldowns_do_not_pay(self, stats):
        """The headline the page quotes: fewer than half clear break-even,
        which is why the advisor still says SKIP on most of them."""
        assert 0.3 < stats["positive_ev_share"] < 0.6
        assert stats["median_ev"] < 0

    def test_quartiles_bracket_the_median(self, stats):
        low, high = stats["ev_quartiles"]
        assert low <= stats["median_ev"] <= high

    def test_the_counterfactual_is_labelled(self, stats):
        assert stats["caveat"] == CAVEAT
        assert "counterfactual" in CAVEAT.lower()
