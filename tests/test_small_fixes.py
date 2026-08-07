"""Shared portfolio seed, Must-Be-Won typing, Abrams & Garibaldi screen."""

from datetime import date

import pytest

from lottery.ev import (
    DrawConditions,
    ROLLOVER_CAP,
    abrams_garibaldi_screen,
    default_portfolio_seed,
    mbw_type,
)
from lottery.portfolio import build_portfolio

SAT = date(2026, 8, 8)


class TestSharedPortfolioSeed:
    def test_seed_is_the_draw_date(self):
        assert default_portfolio_seed(SAT) == 20260808
        assert default_portfolio_seed(None) is None

    def test_advisor_and_email_build_identical_lines(self):
        """The regression this exists for: ev_play seeded from nothing while
        ev_alert seeded from the draw date, so on a PLAY the email's lines
        and latest.json's lines (what `roi_ledger add --from-latest` records)
        would have been different tickets."""
        cond = DrawConditions(jackpot=16_000_000, tickets_sold=8_000_000,
                              roll_down=True, draw_date=SAT)
        seed = default_portfolio_seed(cond.draw_date)
        advisor = [p["line"] for p in build_portfolio(5, cond, seed=seed)]
        email = [p["line"] for p in build_portfolio(5, cond, seed=seed)]
        assert advisor == email

    def test_explicit_seed_still_wins(self):
        cond = DrawConditions(jackpot=16_000_000, draw_date=SAT)
        a = [p["line"] for p in build_portfolio(3, cond, seed=42)]
        b = [p["line"] for p in build_portfolio(3, cond,
                                                seed=default_portfolio_seed(SAT))]
        assert a != b


class TestMbwType:
    def test_none_when_not_a_rolldown(self):
        assert mbw_type(False, 5) is None

    def test_cap_driven_at_the_cap(self):
        assert mbw_type(True, ROLLOVER_CAP) == "cap-driven"

    def test_special_event_below_the_cap(self):
        """Allwyn schedules ~GBP 15M holiday Must-Be-Won draws without five
        rollovers - draw 3131 (2025-12-24) is one, and its archive pool is
        plainly wrong, so these deserve their own label."""
        assert mbw_type(True, 0) == "special-event"
        assert mbw_type(True, ROLLOVER_CAP - 1) == "special-event"


class TestAbramsGaribaldiScreen:
    def test_skipped_for_rolldowns(self):
        """Their theorems price a pari-mutuel jackpot, not a roll-down - so
        the screen must decline rather than misapply itself."""
        assert abrams_garibaldi_screen(
            DrawConditions(jackpot=12_000_000, roll_down=True)) is None

    def test_ordinary_draw_fails_both_conditions(self):
        ag = abrams_garibaldi_screen(
            DrawConditions(jackpot=5_000_000, tickets_sold=7_500_000))
        assert not ag["robust_good_bet"]
        assert not ag["sales_ok"]      # entries dwarf the jackpot
        assert not ag["jackpot_ok"]

    def test_huge_jackpot_passes(self):
        """A jackpot large enough that no plausible sales level can dilute
        the edge away: their sufficient condition, stricter than our exact
        break-even. It takes ~GBP 200M - both conditions bind, and the
        sales one binds later (GBP 120M clears the cutoff but still fails
        N < J/5 at typical sales)."""
        ag = abrams_garibaldi_screen(
            DrawConditions(jackpot=200_000_000, tickets_sold=7_500_000))
        assert ag["sales_ok"] and ag["jackpot_ok"] and ag["robust_good_bet"]
        mid = abrams_garibaldi_screen(
            DrawConditions(jackpot=120_000_000, tickets_sold=7_500_000))
        assert mid["jackpot_ok"] and not mid["sales_ok"]

    def test_no_real_uk_draw_has_ever_passed(self):
        """The screen's actual verdict on this game: the era-record pool is
        GBP 52.9M against a ~GBP 200M requirement, so no ordinary UK Lotto
        draw has ever been a robustly good bet. That is the honest use of
        this second opinion - it says the +EV lives only in roll-downs,
        which is exactly where our own model puts it."""
        record_pool = 52_900_000
        ag = abrams_garibaldi_screen(
            DrawConditions(jackpot=record_pool, tickets_sold=7_500_000))
        assert not ag["robust_good_bet"]

    def test_screen_is_stricter_than_our_break_even(self):
        """Sanity on the calibration: a jackpot that clears our exact
        break-even must NOT automatically clear A&G, or the screen adds
        nothing as a second opinion."""
        from lottery.ev import break_even_jackpot
        cond = DrawConditions(tickets_sold=7_500_000)
        ours = break_even_jackpot(cond)
        ag = abrams_garibaldi_screen(
            DrawConditions(jackpot=ours * 1.01, tickets_sold=7_500_000))
        assert not ag["robust_good_bet"]
        assert ag["jackpot_cutoff"] > ours
