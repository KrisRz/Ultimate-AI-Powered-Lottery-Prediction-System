"""The archive must extend the frozen backfill without changing what it says.

Draws 3190-3195 are in both files - the backfill and the collector's - which
makes them the test: rebuild those six draws from the collector's rows and
they must match the backfill wherever the backfill is self-consistent.
"""

import pandas as pd
import pytest

from scripts.archive import load_sales_archive, load_tier_archive

KEY = ["draw_number", "round", "tier"]
# Pinned: the collector appends every draw and would move a live-tail test.
THROUGH = 3209


@pytest.fixture(scope="module")
def truncated(tmp_path_factory):
    d = tmp_path_factory.mktemp("archive")
    tiers = pd.read_csv("data/prize_tiers_history.csv")
    tiers[tiers["draw_number"] <= 3189].to_csv(d / "tiers.csv", index=False)
    sales = pd.read_csv("data/sales_history.csv")
    sales[sales["draw_number"] <= 3179].to_csv(d / "sales.csv", index=False)
    return d


def test_collected_rows_reproduce_the_backfill_on_the_overlap(truncated):
    rebuilt = load_tier_archive(truncated / "tiers.csv")
    backfill = pd.read_csv("data/prize_tiers_history.csv")
    pick = lambda df: (df[df["draw_number"].between(3190, 3195)]
                       .sort_values(KEY).reset_index(drop=True))
    got, want = pick(rebuilt), pick(backfill)
    assert len(got) == len(want) == 72
    assert (got["winners"] == want["winners"]).all()
    assert (got["category"] == want["category"]).all()
    comparable = (got["winners"] > 0) | (got["tier"] == 1)
    off = comparable & ((got["prize_per_winner"] - want["prize_per_winner"]).abs() > 1)
    # The only disagreements are the backfill's own: a Match 5+Bonus winner
    # recorded at £0. The tier pays a fixed £1,000,000; the collector has it.
    assert sorted(zip(got.loc[off, "draw_number"], got.loc[off, "round"])) == [
        (3191, 2), (3195, 2)]
    assert (got.loc[off, "tier"] == 2).all()
    assert (want.loc[off, "prize_per_winner"] == 0).all()
    assert (got.loc[off, "prize_per_winner"] == 1_000_000).all()


def test_the_archive_reaches_past_the_backfill():
    archive = load_tier_archive()
    archive = archive[archive["draw_number"] <= THROUGH]
    assert archive["draw_number"].min() == 2066
    assert archive["draw_number"].max() == THROUGH
    assert not archive.duplicated(KEY).any()


def test_a_shared_jackpot_keeps_its_zero_winner_round_at_zero():
    """3196: two winners in round 2, none in round 1. Filling round 1 with the
    pool would double the pool anyone reads back off this table."""
    t = load_tier_archive()
    jackpot = t[(t["draw_number"] == 3196) & (t["tier"] == 1)].set_index("round")
    assert jackpot.loc[2, "winners"] == 2
    assert jackpot.loc[1, "prize_per_winner"] == 0


def test_sales_extend_from_the_pool_identity(truncated):
    rebuilt = load_sales_archive(truncated / "sales.csv").set_index("draw_number")
    backfill = pd.read_csv("data/sales_history.csv").set_index("draw_number")
    diff = (rebuilt.loc[3180:3195, "lines_sold"]
            - backfill.loc[3180:3195, "lines_sold"]).abs()
    assert diff.max() < 0.0001 * backfill.loc[3180:3195, "lines_sold"].min()
    live = load_sales_archive().set_index("draw_number")
    assert live.loc[3205, "lines_sold"] == 5_918_273


def test_pools_read_off_the_archive_count_a_shared_jackpot_once():
    """3196: two winners shared £8.54M in round 2. Max prize x total winners
    read £17M once the full pool sat against round 1 as well."""
    from scripts.calibrate_mbw_uplift import draw_pools_from_tiers
    pools = draw_pools_from_tiers(load_tier_archive())
    assert pools[3196] == pytest.approx(8_535_146, abs=2)
    assert pools[3184] == pytest.approx(9_269_710, abs=1)   # won in round 2
    assert pools[3190] == pytest.approx(9_559_451, abs=1)   # rolled down
