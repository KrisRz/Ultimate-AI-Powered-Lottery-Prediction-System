"""Post-Must-Be-Won scorecard: detection, measurement, accumulation."""

import pandas as pd
import pytest

from lottery.ev import P_MATCH_2, P_MATCH_3, P_MATCH_4
from scripts.monitoring.post_mbw_validation import (
    append_scorecard,
    format_report,
    latest_draw_was_rolldown,
    measured_lines,
    redistributed_sum,
    validate,
)


def _draw_rows(draw_no, draw_date, n_lines, m3_prize=10.0, m2_prize=1.0,
               next_roll_down=False, next_jackpot=5_000_000.0):
    """Tier rows for one draw (both rounds) implying exactly n_lines sold."""
    rows = []
    for rnd in (1, 2):
        for tier, prob, prize in ((4, P_MATCH_4, 50.0), (5, P_MATCH_3, m3_prize),
                                  (6, P_MATCH_2, m2_prize)):
            winners = int(n_lines * prob)
            rows.append({
                "draw_number": draw_no, "draw_date": draw_date, "round": rnd,
                "tier": tier, "winners": winners,
                "prize_total": winners * prize,
                "prize_per_winner": prize,
                "rollover": True, "rollover_count": 3,
                "next_jackpot_estimate": next_jackpot,
                "next_jackpot_roll_down": next_roll_down,
            })
    return rows


def _history(mbw_lines=9_000_000):
    """Ten ordinary Saturdays then a Must-Be-Won Saturday at mbw_lines."""
    rows = []
    for i in range(10):
        rows += _draw_rows(3180 + i, f"2026-05-{2 + 7 * (i % 4):02d}", 7_000_000,
                           next_roll_down=(i == 9), next_jackpot=8_000_000.0)
    rows += _draw_rows(3190, "2026-07-18", mbw_lines,
                       m3_prize=24.0, m2_prize=5.0)
    return pd.DataFrame(rows)


class TestDetection:
    def test_flagged_by_previous_draws_forward_flag(self):
        assert latest_draw_was_rolldown(_history())

    def test_quiet_on_an_ordinary_draw(self):
        rows = []
        for i in range(3):
            rows += _draw_rows(3180 + i, "2026-05-02", 7_000_000)
        df = pd.DataFrame(rows).drop(columns=["prize_per_winner"])
        assert not latest_draw_was_rolldown(df)

    def test_tier_marker_works_without_forward_flag(self):
        df = _history()
        df.loc[df["draw_number"] == 3189, "next_jackpot_roll_down"] = False
        df["tier_roll_down"] = (df["draw_number"] == 3190) & df["tier"].isin([5, 6])
        assert latest_draw_was_rolldown(df)


class TestScorecard:
    def test_measures_the_draws_own_sales(self):
        df = _history()
        rows = df[df["draw_number"] == 3190]
        assert measured_lines(rows) == pytest.approx(9_000_000, rel=0.01)

    def test_redistribution_sums_the_boosts(self):
        df = _history()
        rows = df[df["draw_number"] == 3190]
        prizes = pytest.importorskip("lottery.ev").FixedPrizes()
        got = redistributed_sum(rows, prizes)
        w3 = int(9_000_000 * P_MATCH_3)
        w2 = int(9_000_000 * P_MATCH_2)
        assert got == pytest.approx(2 * (14.0 * w3 + 4.0 * w2), rel=1e-6)

    def test_validate_full_scorecard(self):
        r = validate(_history())
        assert r["draw_number"] == 3190
        assert r["measured_lines"] == pytest.approx(9_000_000, rel=0.01)
        # forecast = ordinary baseline (7M, all Saturdays) x Sat uplift 1.27
        assert r["predicted_lines"] == pytest.approx(7_000_000 * 1.27, rel=0.02)
        assert r["uplift_measured"] == pytest.approx(9 / 7, rel=0.02)
        assert r["advertised_pool"] == 8_000_000.0
        assert r["pool_ratio"] == pytest.approx(r["redistributed"] / 8_000_000.0)
        assert "scorecard" in format_report(r)

    def test_validate_returns_none_for_ordinary(self):
        rows = []
        for i in range(3):
            rows += _draw_rows(3180 + i, "2026-05-02", 7_000_000)
        assert validate(pd.DataFrame(rows)) is None


class TestAccumulation:
    def test_appends_and_dedupes(self, tmp_path):
        path = tmp_path / "scorecard.csv"
        r = validate(_history())
        append_scorecard(r, path)
        append_scorecard({**r, "measured_lines": 1}, path)   # rerun corrects
        out = pd.read_csv(path)
        assert len(out) == 1
        assert out.iloc[0]["measured_lines"] == 1
