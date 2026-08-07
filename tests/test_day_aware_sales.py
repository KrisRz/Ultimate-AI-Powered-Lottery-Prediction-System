"""Day-aware sales estimation: weekday baseline + day-specific MBW uplift."""

from datetime import date

import pandas as pd
import pytest

from lottery.ev import (
    DrawConditions,
    MBW_SALES_UPLIFT,
    MBW_SALES_UPLIFT_P25,
    MBW_SALES_UPLIFT_P75,
    MBW_UPLIFT_BY_WEEKDAY,
    P_MATCH_3,
    estimate_tickets_sold,
    mbw_uplift,
    sales_sensitivity,
)

SAT = date(2026, 8, 8)      # a Saturday
WED = date(2026, 8, 5)      # a Wednesday


def test_mbw_uplift_selects_by_weekday():
    assert mbw_uplift(SAT) == MBW_UPLIFT_BY_WEEKDAY[5]
    assert mbw_uplift(WED) == MBW_UPLIFT_BY_WEEKDAY[2]
    assert mbw_uplift(None) == (
        MBW_SALES_UPLIFT, MBW_SALES_UPLIFT_P25, MBW_SALES_UPLIFT_P75)
    # a non-draw weekday means the date is wrong; mixed constants, not a crash
    assert mbw_uplift(date(2026, 8, 6)) == mbw_uplift(None)


def test_wednesday_uplift_exceeds_saturday():
    """Sat sells 1.59x Wed at baseline, so the RELATIVE Must-Be-Won jump is
    larger on a Wednesday. If a recalibration ever flips this, the constants
    were installed against the wrong baseline definition."""
    assert MBW_UPLIFT_BY_WEEKDAY[2][0] > MBW_UPLIFT_BY_WEEKDAY[5][0]


def _tiers(rows):
    return pd.DataFrame(rows, columns=["draw_number", "draw_date", "tier", "winners"])


def _tier_rows(draw, day, n_lines):
    """One Match-3 observation implying exactly n_lines sold."""
    return [(draw, day.isoformat(), 5, int(n_lines * P_MATCH_3))]


def test_estimate_uses_same_weekday_baseline():
    rows = []
    # alternating draws: Wednesdays sell 6M, Saturdays 10M
    for i in range(10):
        rows += _tier_rows(3000 + 2 * i, date(2026, 1, 7), 6_000_000)      # Wed
        rows += _tier_rows(3001 + 2 * i, date(2026, 1, 10), 10_000_000)    # Sat
    tiers = _tiers(rows)
    est_sat = estimate_tickets_sold(tiers, draw_date=SAT)
    est_wed = estimate_tickets_sold(tiers, draw_date=WED)
    assert est_sat == pytest.approx(10_000_000, rel=0.01)
    assert est_wed == pytest.approx(6_000_000, rel=0.01)
    # without a date the median mixes both days - for a Wednesday target that
    # baseline is contaminated upward, which is the error this change removes
    # (with 10+10 equal observations the upper-median lands ON the Sat level)
    est_mixed = estimate_tickets_sold(tiers)
    assert est_wed < est_mixed <= est_sat


def test_estimate_applies_day_specific_uplift():
    rows = []
    for i in range(10):
        rows += _tier_rows(3000 + 2 * i, date(2026, 1, 7), 6_000_000)
        rows += _tier_rows(3001 + 2 * i, date(2026, 1, 10), 10_000_000)
    tiers = _tiers(rows)
    est = estimate_tickets_sold(tiers, roll_down=True, draw_date=WED)
    assert est == pytest.approx(6_000_000 * MBW_UPLIFT_BY_WEEKDAY[2][0], rel=0.01)


def test_wednesday_mbw_no_longer_priced_at_mixed_saturday_level():
    """The regression this PR exists for: a Wednesday Must-Be-Won priced with
    the flat mixed model got a Saturday-contaminated baseline times 1.38;
    day-aware it gets the Wednesday level times 1.44 - materially fewer
    lines, hence more EV."""
    rows = []
    for i in range(10):
        rows += _tier_rows(3000 + 2 * i, date(2026, 1, 7), 6_000_000)
        rows += _tier_rows(3001 + 2 * i, date(2026, 1, 10), 10_000_000)
    tiers = _tiers(rows)
    flat = estimate_tickets_sold(tiers, roll_down=True)                    # old path
    day_aware = estimate_tickets_sold(tiers, roll_down=True, draw_date=WED)
    assert day_aware < flat * 0.85


def test_estimate_falls_back_when_weekday_sample_too_small():
    rows = []
    for i in range(8):
        rows += _tier_rows(3000 + i, date(2026, 1, 10), 10_000_000)        # all Sat
    tiers = _tiers(rows)
    # Wednesday target, but no Wednesday draws collected yet: mixed fallback
    assert estimate_tickets_sold(tiers, draw_date=WED) == pytest.approx(
        10_000_000, rel=0.01)


def test_sales_sensitivity_uses_day_quartiles():
    cond = DrawConditions(jackpot=12_000_000, tickets_sold=9_000_000,
                          roll_down=True, draw_date=SAT)
    sens = sales_sensitivity(cond)
    up = MBW_UPLIFT_BY_WEEKDAY[5]
    baseline = cond.tickets_sold / up[0]
    assert sens["uplift"] == up[0]
    assert sens["tickets_low"] == int(baseline * up[1])
    assert sens["tickets_high"] == int(baseline * up[2])
    # undated conditions keep the wide mixed quartiles
    sens_mixed = sales_sensitivity(DrawConditions(
        jackpot=12_000_000, tickets_sold=9_000_000, roll_down=True))
    assert sens_mixed["uplift"] == MBW_SALES_UPLIFT
    spread = sens["tickets_high"] - sens["tickets_low"]
    spread_mixed = sens_mixed["tickets_high"] - sens_mixed["tickets_low"]
    assert spread < spread_mixed
