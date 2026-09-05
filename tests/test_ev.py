"""Tests for the EV engine - probabilities, popularity model, EV, portfolio."""

from datetime import date, datetime, timezone

import pytest

from lottery.ev import (
    DrawConditions,
    FixedPrizes,
    POPULARITY_NORMALIZATION,
    ROLLOVER_CAP,
    UK_TZ,
    _weight_sums,
    PRIZE_MATCH_2,
    PRIZE_MATCH_3,
    P_JACKPOT,
    P_MATCH_2,
    P_MATCH_3,
    P_MATCH_4,
    P_MATCH_5,
    P_MATCH_5_BONUS,
    TICKET_PRICE,
    TOTAL_COMBOS,
    best_unpopular_reference_line,
    break_even_jackpot,
    calibrate_fixed_prizes,
    expected_cowinner_share,
    forecast_must_be_won,
    line_ev,
    match_probability,
    next_draw_dates,
    popularity_ratio,
    should_play,
    upcoming_draw_date,
)
from lottery.ev import (
    MBW_SALES_UPLIFT,
    MBW_SALES_UPLIFT_P25,
    MBW_SALES_UPLIFT_P75,
    estimate_tickets_sold,
    rolldown_draw_numbers,
    sales_sensitivity,
)
from lottery.portfolio import MAX_PAIRWISE_OVERLAP, build_portfolio

BIRTHDAY_LINE = [3, 7, 11, 14, 21, 27]
UNPOPULAR_LINE = [34, 38, 41, 46, 53, 58]


class TestProbabilities:
    def test_total_combos(self):
        assert TOTAL_COMBOS == 45_057_474

    def test_official_odds(self):
        # Published UK Lotto (6/59) odds
        assert 1 / P_JACKPOT == pytest.approx(45_057_474)
        assert 1 / P_MATCH_5_BONUS == pytest.approx(7_509_579)
        assert 1 / P_MATCH_5 == pytest.approx(144_415, rel=1e-4)
        assert 1 / P_MATCH_4 == pytest.approx(2_180, rel=1e-3)
        assert 1 / P_MATCH_3 == pytest.approx(96.2, rel=1e-3)
        assert 1 / P_MATCH_2 == pytest.approx(10.26, rel=1e-3)

    def test_match_probabilities_sum_to_one(self):
        assert sum(match_probability(k) for k in range(7)) == pytest.approx(1.0)


class TestPopularity:
    def test_birthday_line_is_popular(self):
        assert popularity_ratio(BIRTHDAY_LINE) > 1.5

    def test_high_number_line_is_unpopular(self):
        assert popularity_ratio(UNPOPULAR_LINE) < 0.5

    def test_arithmetic_sequence_is_heavily_played(self):
        assert popularity_ratio([1, 2, 3, 4, 5, 6]) > popularity_ratio(BIRTHDAY_LINE)

    def test_cowinner_share_favors_unpopular(self):
        n = 15_000_000
        assert expected_cowinner_share(UNPOPULAR_LINE, n) > expected_cowinner_share(
            BIRTHDAY_LINE, n
        )
        assert 0.0 < expected_cowinner_share(BIRTHDAY_LINE, n) <= 1.0

    def test_other_rounds_share_at_average_popularity(self):
        """Round two's winners hold round two's numbers, not mine.

        Pricing every round at the line's own popularity - which this did until
        2026-09-05 - overstated what an unpopular line buys on the jackpot.
        Half the exposure to sharing does not depend on the slip.
        """
        n = 8_000_000
        one = expected_cowinner_share(UNPOPULAR_LINE, n, rounds=1)
        two = expected_cowinner_share(UNPOPULAR_LINE, n, rounds=2)
        assert two < one
        # An average-popularity line is the one case where the old model was
        # right, because 1.0 is what the other round is priced at anyway.
        assert expected_cowinner_share([1, 13, 26, 38, 47, 52], n, rounds=2) \
            == pytest.approx(expected_cowinner_share([1, 13, 26, 38, 47, 52], n * 2,
                                                     rounds=1), rel=0.05)

    def test_the_unpopular_advantage_halves_in_a_two_round_game(self):
        n = 8_000_000
        average = 0.842      # a popularity-1.0 line at N=8m, both rounds
        mine = expected_cowinner_share(UNPOPULAR_LINE, n, rounds=2)
        assert mine / average - 1 == pytest.approx(0.059, abs=0.005)


class TestLineEV:
    def test_unpopular_line_has_higher_ev(self):
        cond = DrawConditions(jackpot=20_000_000)
        assert line_ev(UNPOPULAR_LINE, cond) > line_ev(BIRTHDAY_LINE, cond)

    def test_ev_increases_with_jackpot(self):
        small = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000))
        big = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=50_000_000))
        assert big > small

    def test_rolldown_increases_ev(self):
        base = DrawConditions(jackpot=12_000_000, roll_down=False)
        mbw = DrawConditions(jackpot=12_000_000, roll_down=True)
        assert line_ev(UNPOPULAR_LINE, mbw) > line_ev(UNPOPULAR_LINE, base)

    def test_typical_draw_is_negative_ev(self):
        # A normal £2M draw must show a clear expected loss - anything else
        # would mean the model is lying
        ev = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000))
        assert -2.0 < ev < 0.0

    def test_two_rounds_beat_one_round(self):
        two = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000, rounds=2))
        one = line_ev(UNPOPULAR_LINE, DrawConditions(jackpot=2_000_000, rounds=1))
        assert two > one

    def test_rolldown_pool_is_per_event_not_per_round(self):
        # The MBW jackpot is ONE pool shared across rounds (Allwyn 2026), so
        # playing two rounds must not double the roll-down uplift
        def uplift(rounds):
            base = DrawConditions(jackpot=12_000_000, roll_down=False, rounds=rounds)
            mbw = DrawConditions(jackpot=12_000_000, roll_down=True, rounds=rounds)
            return line_ev(UNPOPULAR_LINE, mbw) - line_ev(UNPOPULAR_LINE, base)

        assert uplift(2) <= uplift(1)


class TestShouldPlay:
    def test_skips_ordinary_draw_at_zero_threshold(self):
        verdict = should_play(DrawConditions(jackpot=2_000_000), threshold=0.0)
        assert verdict["play"] is False

    def test_reference_line_is_valid(self):
        line = best_unpopular_reference_line()
        assert len(set(line)) == 6
        assert all(32 <= n <= 59 for n in line)  # unpopular = high numbers

    def test_reference_line_is_actually_unpopular(self):
        # Regression: the old builder produced 32,34,36,... - an arithmetic
        # sequence whose x8 pattern penalty erased the high-number advantage
        line = best_unpopular_reference_line()
        assert len({b - a for a, b in zip(line, line[1:])}) > 1
        # Calibrated weights are less extreme than the old heuristic: a pure
        # high-number line lands near 0.83**6 ~= 0.33, still firmly unpopular.
        assert popularity_ratio(line) < 0.4

    def test_lenient_threshold_allows_play(self):
        verdict = should_play(DrawConditions(jackpot=2_000_000), threshold=-2.0)
        assert verdict["play"] is True


class TestPopularityNormalization:
    """popularity_ratio is a pick rate relative to uniform, so its mean over
    every possible line must be exactly 1.0 - otherwise
    `tickets_sold * ratio / TOTAL_COMBOS` describes more tickets than were sold.
    """

    def test_mean_over_random_lines_is_one(self):
        import random
        rng = random.Random(20260728)
        n = 40_000
        mean = sum(popularity_ratio(rng.sample(range(1, 60), 6))
                   for _ in range(n)) / n
        assert mean == pytest.approx(1.0, abs=0.02)

    def test_raw_score_is_the_one_that_drifts(self):
        # Documents WHY the constant exists: the pattern multipliers add mass
        # that nothing takes back out.
        assert POPULARITY_NORMALIZATION > 1.0

    def test_weight_sums_match_brute_force(self):
        # The DP replaces a 45M-line enumeration; check it against the real
        # thing on a lottery small enough to enumerate (4 from 12).
        import itertools
        from lottery.ev import MEAN_WEIGHT, _has_consecutive_run, number_weight

        total = run3 = 0.0
        for line in itertools.combinations(range(1, 13), 4):
            w = 1.0
            for n in line:
                w *= number_weight(n) / MEAN_WEIGHT
            total += w
            if _has_consecutive_run(line, 3):
                run3 += w

        dp_total, dp_run3 = _weight_sums(12, 4)
        assert dp_total == pytest.approx(total, rel=1e-12)
        assert dp_run3 == pytest.approx(run3, rel=1e-12)

    def test_arithmetic_enumeration_is_complete(self):
        import itertools
        from lottery.ev import _arithmetic_lines, _is_arithmetic

        enumerated = {tuple(line) for line in _arithmetic_lines(20, 4)}
        brute = {c for c in itertools.combinations(range(1, 21), 4)
                 if _is_arithmetic(c)}
        assert enumerated == brute

    def test_relative_ordering_survives_normalization(self):
        assert popularity_ratio([1, 2, 3, 4, 5, 6]) > popularity_ratio(BIRTHDAY_LINE)
        assert popularity_ratio(BIRTHDAY_LINE) > popularity_ratio(UNPOPULAR_LINE)


class TestMustBeWonForecast:
    """The only reliably +EV draws are Must-Be-Won ones, so knowing when the
    next is due is what turns 'wait for the alert' into a budget plan."""

    def test_draw_dates_are_wednesdays_and_saturdays(self):
        dates = next_draw_dates(date(2026, 7, 28), 5)
        assert [d.isoformat() for d in dates] == [
            "2026-07-29", "2026-08-01", "2026-08-05", "2026-08-08", "2026-08-12"]

    def test_draw_on_a_draw_day_returns_the_following_one(self):
        assert next_draw_dates(date(2026, 7, 29), 1)[0] == date(2026, 8, 1)

    def test_counts_down_to_the_cap(self):
        # Live state after draw 3192: two rollovers banked, so the Must-Be-Won
        # draw is the 4th from now unless somebody wins first.
        f = forecast_must_be_won(2, now=date(2026, 7, 28))
        assert f["draws_away"] == 4
        assert f["expected_date"] == date(2026, 8, 8)
        assert f["is_next_draw"] is False

    def test_at_the_cap_the_next_draw_must_be_won(self):
        f = forecast_must_be_won(ROLLOVER_CAP, now=date(2026, 7, 28))
        assert f["is_next_draw"] is True
        assert f["expected_date"] == date(2026, 7, 29)

    def test_never_forecasts_into_the_past(self):
        # A feed glitch reporting more rollovers than the cap must not produce
        # a negative countdown.
        assert forecast_must_be_won(99, now=date(2026, 7, 28))["draws_away"] == 1

    def test_countdown_starts_at_tonights_draw_on_a_draw_day(self):
        # Regression: the countdown counts the draw you can still enter as #1,
        # so on a draw day the date must not skip past tonight. Live state after
        # draw 3194 (rollover 4) asked on Wednesday 2026-08-05 is Saturday
        # 2026-08-08 - counting 08-05 and 08-08. Starting from `next_draw_dates`
        # counted 08-08 and 08-12 and named the wrong draw as Must-Be-Won.
        f = forecast_must_be_won(4, now=date(2026, 8, 5))       # Wed, sales open
        assert f["draws_away"] == 2
        assert f["expected_date"] == date(2026, 8, 8)

    def test_same_forecast_holds_the_day_before(self):
        # The forecast names a draw, so it must not move just because the clock
        # rolled over to the draw day.
        tue = forecast_must_be_won(4, now=date(2026, 8, 4))
        wed = forecast_must_be_won(4, now=date(2026, 8, 5))
        assert tue["expected_date"] == wed["expected_date"] == date(2026, 8, 8)

    def test_after_sales_close_the_countdown_moves_on(self):
        # Same Wednesday, 21:00: tonight's draw has happened, so the two draws
        # left are 08-08 and 08-12. (The feed's rollover count catches up when
        # the draw is collected an hour later.)
        f = forecast_must_be_won(4, now=datetime(2026, 8, 5, 21, 0, tzinfo=UK_TZ))
        assert f["expected_date"] == date(2026, 8, 12)


class TestUpcomingDrawDate:
    """Which draw a ticket bought right now enters - the ledger files lines
    under this date and settles them against that draw's results."""

    def test_draw_day_before_sales_close(self):
        assert upcoming_draw_date(datetime(2026, 8, 5, 14, 0, tzinfo=UK_TZ)) == date(2026, 8, 5)
        assert upcoming_draw_date(datetime(2026, 8, 8, 19, 29, tzinfo=UK_TZ)) == date(2026, 8, 8)

    def test_draw_day_after_sales_close(self):
        assert upcoming_draw_date(datetime(2026, 8, 5, 19, 30, tzinfo=UK_TZ)) == date(2026, 8, 8)
        assert upcoming_draw_date(datetime(2026, 8, 8, 23, 59, tzinfo=UK_TZ)) == date(2026, 8, 12)

    def test_non_draw_day(self):
        assert upcoming_draw_date(datetime(2026, 8, 6, 9, 0, tzinfo=UK_TZ)) == date(2026, 8, 8)

    def test_utc_input_is_read_in_uk_time(self):
        # The cloud collector runs on UTC. During BST 19:00 UTC is 20:00 in
        # London - sales already closed - and reading it as local time would
        # hand out a draw the ticket cannot enter.
        assert upcoming_draw_date(datetime(2026, 8, 5, 19, 0, tzinfo=timezone.utc)) == date(2026, 8, 8)

    def test_evening_run_and_morning_retry_agree(self):
        # ev_alert.py relies on this: the Wed-night run and the Thu-morning
        # retry must name the same draw, or the two emails propose different
        # portfolios for the same money.
        evening = upcoming_draw_date(datetime(2026, 8, 5, 21, 45, tzinfo=timezone.utc))
        retry = upcoming_draw_date(datetime(2026, 8, 6, 6, 0, tzinfo=timezone.utc))
        assert evening == retry == date(2026, 8, 8)


class TestFixedPrizes:
    """Regression cover for the roll-down-prizes-as-base-prizes bug.

    Draw 3190 (2026-07-18) was a roll-down: it paid Match 3 £24 / Match 2 £5
    over a base of £10 / £1. Reading those off that one draw inflated every
    line's EV by ~£1.07 and dropped the break-even jackpot from £30M to £4.8M -
    close enough to a real rollover to trigger a false PLAY alert.
    """

    @staticmethod
    def _tiers(match_3_by_draw):
        """prize_tiers.csv-shaped frame; {draw_number: match-3 prize} in."""
        import pandas as pd
        rows = []
        for draw, m3 in match_3_by_draw.items():
            for rnd in (1, 2):
                for tier, winners, per_winner in (
                    (3, 40, 1_000.0), (4, 3_000, 50.0),
                    (5, 80_000, m3), (6, 800_000, 1.0 if m3 == 10 else 5.0),
                ):
                    rows.append({"draw_number": draw, "round": rnd, "tier": tier,
                                 "winners": winners,
                                 "prize_total": winners * per_winner})
        return pd.DataFrame(rows)

    def test_defaults_are_base_prizes(self):
        assert PRIZE_MATCH_3 == 10.0
        assert PRIZE_MATCH_2 == 1.0

    def test_fixed_tiers_return_a_plausible_share_of_the_stake(self):
        # UK Lotto returns ~50% of stakes; the fixed tiers are only part of
        # that (the jackpot pool is the rest). At the old 24/5 this was 90%,
        # which is the tell that no lottery could pay it.
        share = DrawConditions().rounds * FixedPrizes().ev_per_round() / TICKET_PRICE
        assert 0.30 < share < 0.45

    def test_recovers_prizes_from_observed_data(self):
        prizes = calibrate_fixed_prizes(self._tiers({3191: 10, 3192: 10, 3193: 10}))
        assert prizes.match_3 == 10.0
        assert prizes.match_2 == 1.0
        assert prizes.match_5 == 1_000.0
        assert prizes.match_4 == 50.0
        assert prizes.source.startswith("observed")

    def test_median_ignores_a_minority_rolldown_draw(self):
        prizes = calibrate_fixed_prizes(
            self._tiers({3190: 24, 3191: 10, 3192: 10, 3193: 10}))
        assert prizes.match_3 == 10.0
        assert prizes.match_2 == 1.0

    def test_thin_data_keeps_the_defaults(self):
        import pandas as pd
        one_row = pd.DataFrame([{"draw_number": 3190, "round": 1, "tier": 5,
                                 "winners": 80_000, "prize_total": 80_000 * 24}])
        assert calibrate_fixed_prizes(one_row).match_3 == PRIZE_MATCH_3

    def test_no_data_keeps_the_defaults(self):
        import pandas as pd
        assert calibrate_fixed_prizes(None) == FixedPrizes()
        assert calibrate_fixed_prizes(pd.DataFrame(
            columns=["draw_number", "round", "tier", "winners",
                     "prize_total"])) == FixedPrizes()

    def test_zero_winner_tiers_are_skipped(self):
        # Unclaimed tiers carry prize_total 0 - averaging them in would drag
        # every prize toward zero.
        import pandas as pd
        rows = [{"draw_number": d, "round": r, "tier": 5, "winners": w,
                 "prize_total": w * 10.0}
                for d, r, w in ((3191, 1, 0), (3191, 2, 80_000),
                                (3192, 1, 80_000), (3192, 2, 80_000))]
        assert calibrate_fixed_prizes(pd.DataFrame(rows)).match_3 == 10.0

    def test_rolldown_prizes_are_not_double_counted(self):
        # The roll-down uplift lives in line_ev's rolldown term. If the base
        # prizes also carried it, a roll-down draw would be priced twice.
        cond = DrawConditions(jackpot=9_560_000, roll_down=True,
                              tickets_sold=7_457_262)
        boosted = DrawConditions(jackpot=9_560_000, roll_down=True,
                                 tickets_sold=7_457_262,
                                 prizes=FixedPrizes(match_3=24.0, match_2=5.0))
        assert line_ev(UNPOPULAR_LINE, boosted) - line_ev(UNPOPULAR_LINE, cond) \
            == pytest.approx(2 * (P_MATCH_3 * 14 + P_MATCH_2 * 4))


class TestBreakEvenJackpot:
    def test_break_even_jackpot_makes_ev_zero(self):
        cond = DrawConditions(tickets_sold=7_457_262)
        cond.jackpot = break_even_jackpot(cond)
        assert line_ev(best_unpopular_reference_line(), cond) == pytest.approx(0.0, abs=1e-6)

    def test_ordinary_draw_needs_an_implausible_jackpot(self):
        # With base prizes the bar is ~£30M - UK Lotto caps out well below
        # that, so "SKIP" is the answer for every non-Must-Be-Won draw.
        assert break_even_jackpot(DrawConditions(tickets_sold=7_457_262)) > 25_000_000

    def test_rolldown_break_even_is_far_lower(self):
        plain = break_even_jackpot(DrawConditions(tickets_sold=7_457_262))
        mbw = break_even_jackpot(DrawConditions(tickets_sold=7_457_262, roll_down=True))
        assert mbw < plain / 2


def _tiers_rows(draw: int, n_true: int, match_3_prize: float = 10.0):
    """One draw's high-count tiers, consistent with `n_true` lines sold."""
    rows = []
    for tier, p in ((4, P_MATCH_4), (5, P_MATCH_3), (6, P_MATCH_2)):
        winners = int(n_true * p)
        prize = {4: 50.0, 5: match_3_prize, 6: 1.0}[tier]
        rows.append({"draw_number": draw, "round": 1, "tier": tier,
                     "winners": winners, "prize_total": winners * prize})
    return rows


class TestEstimateTicketsSold:
    def _tiers_df(self, n_true: int):
        import pandas as pd
        return pd.DataFrame(_tiers_rows(3190, n_true))

    def test_recovers_true_ticket_count(self):
        est = estimate_tickets_sold(self._tiers_df(8_000_000))
        assert est == pytest.approx(8_000_000, rel=0.02)

    def test_none_without_data(self):
        import pandas as pd
        assert estimate_tickets_sold(None) is None
        assert estimate_tickets_sold(pd.DataFrame(
            columns=["draw_number", "round", "tier", "winners"])) is None

    def test_must_be_won_draw_is_scaled_up(self):
        # The 2026-08-06 bug: a Must-Be-Won draw was priced at ordinary-draw
        # sales, and the roll-down term is J/N, so its EV came out too high.
        ordinary = estimate_tickets_sold(self._tiers_df(6_000_000))
        must_be_won = estimate_tickets_sold(self._tiers_df(6_000_000), roll_down=True)
        assert must_be_won == pytest.approx(ordinary * MBW_SALES_UPLIFT, rel=0.01)
        assert must_be_won > ordinary

    def test_rolldown_draws_do_not_inflate_the_baseline(self):
        import pandas as pd
        # A roll-down in the window sells more; leaving it in the sample raises
        # the "ordinary" level it is supposed to be measured against. With the
        # 6 collected draws that single draw moved the estimate 6.57M vs 5.91M.
        ordinary_only = pd.DataFrame(
            [r for d in range(3191, 3196) for r in _tiers_rows(d, 6_000_000)])
        with_boosted = pd.concat([
            ordinary_only,
            pd.DataFrame(_tiers_rows(3190, 9_000_000, match_3_prize=24.0)),
        ], ignore_index=True)
        assert estimate_tickets_sold(with_boosted) == \
            estimate_tickets_sold(ordinary_only)

    def test_all_rolldown_window_falls_back_instead_of_returning_none(self):
        import pandas as pd
        only_boosted = pd.DataFrame(
            [r for d in (3190, 3191, 3192)
             for r in _tiers_rows(d, 9_000_000, match_3_prize=24.0)])
        assert estimate_tickets_sold(only_boosted) == pytest.approx(9_000_000, rel=0.02)


class TestRolldownDetection:
    def test_flags_the_boosted_draw_only(self):
        import pandas as pd
        df = pd.DataFrame(
            [r for d in range(3191, 3196) for r in _tiers_rows(d, 6_000_000)]
            + _tiers_rows(3190, 9_000_000, match_3_prize=24.0))
        assert rolldown_draw_numbers(df) == {3190}

    def test_no_boost_means_no_rolldowns(self):
        import pandas as pd
        df = pd.DataFrame(
            [r for d in range(3191, 3196) for r in _tiers_rows(d, 6_000_000)])
        assert rolldown_draw_numbers(df) == set()

    def test_thin_data_flags_nothing(self):
        import pandas as pd
        assert rolldown_draw_numbers(pd.DataFrame(_tiers_rows(3190, 6_000_000))) == set()
        assert rolldown_draw_numbers(None) == set()

    def test_missing_prize_column_flags_nothing_instead_of_raising(self):
        import pandas as pd
        df = pd.DataFrame([{"draw_number": d, "round": 1, "tier": 5, "winners": 100}
                           for d in range(3190, 3196)])
        assert rolldown_draw_numbers(df) == set()
        assert estimate_tickets_sold(df) is not None


class TestSalesSensitivity:
    def test_absent_for_ordinary_draws(self):
        cond = DrawConditions(tickets_sold=7_000_000)
        assert sales_sensitivity(cond) is None
        assert should_play(cond)["sales_sensitivity"] is None

    def test_busier_sales_pay_less(self):
        cond = DrawConditions(jackpot=10_000_000, tickets_sold=9_000_000, roll_down=True)
        sens = sales_sensitivity(cond)
        assert sens["ev_low"] < sens["ev_high"]
        assert sens["tickets_low"] < cond.tickets_sold < sens["tickets_high"]

    def test_quartile_ticket_counts_come_off_the_pre_uplift_baseline(self):
        cond = DrawConditions(jackpot=10_000_000, tickets_sold=9_000_000, roll_down=True)
        sens = sales_sensitivity(cond)
        baseline = cond.tickets_sold / MBW_SALES_UPLIFT
        assert sens["tickets_low"] == int(baseline * MBW_SALES_UPLIFT_P25)
        assert sens["tickets_high"] == int(baseline * MBW_SALES_UPLIFT_P75)

    def test_thin_edge_is_flagged_not_robust(self):
        # A draw that only clears break-even on the central sales estimate must
        # not read as a clean PLAY - that is what the 2026-08-08 email did.
        cond = DrawConditions(jackpot=10_000_000, tickets_sold=9_000_000, roll_down=True)
        cond.jackpot = break_even_jackpot(cond) * 1.01
        verdict = should_play(cond)
        assert verdict["play"] is True
        assert verdict["sales_sensitivity"]["robust"] is False

    def test_a_fat_pool_is_robust(self):
        cond = DrawConditions(jackpot=25_000_000, tickets_sold=9_000_000, roll_down=True)
        verdict = should_play(cond)
        assert verdict["play"] is True
        assert verdict["sales_sensitivity"]["robust"] is True


class TestAugust2026Regression:
    """The draw that exposed the bug: 2026-08-08, pool £8,391,429.

    Ordinary-draw sales of 6,574,552 made it read +£0.038. Priced as the
    Must-Be-Won draw it actually is, it is a clear SKIP - and the one roll-down
    with full data (3190) returned £1.586 per £2 line.
    """
    POOL = 8_391_429.0
    ORDINARY_BASELINE = 5_913_930      # collected draws, roll-down 3190 excluded

    def test_was_a_false_play_at_ordinary_sales(self):
        cond = DrawConditions(jackpot=self.POOL, tickets_sold=6_574_552, roll_down=True)
        assert should_play(cond)["play"] is True

    def test_is_a_skip_once_the_uplift_is_applied(self):
        cond = DrawConditions(jackpot=self.POOL, roll_down=True,
                              tickets_sold=int(self.ORDINARY_BASELINE * MBW_SALES_UPLIFT))
        verdict = should_play(cond)
        assert verdict["play"] is False
        assert verdict["ev_best_line"] < 0
        assert verdict["break_even_jackpot"] > self.POOL


class TestPortfolio:
    def test_builds_requested_size_with_constraints(self):
        cond = DrawConditions()
        portfolio = build_portfolio(5, cond, seed=42)
        assert len(portfolio) == 5
        lines = [p["line"] for p in portfolio]
        for line in lines:
            assert len(set(line)) == 6
            assert sum(1 for n in line if n > 31) >= 2
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                assert len(set(lines[i]) & set(lines[j])) <= MAX_PAIRWISE_OVERLAP

    def test_reproducible_with_seed(self):
        cond = DrawConditions()
        a = build_portfolio(3, cond, seed=7)
        b = build_portfolio(3, cond, seed=7)
        assert [p["line"] for p in a] == [p["line"] for p in b]

    def test_portfolio_prefers_unpopular(self):
        cond = DrawConditions()
        portfolio = build_portfolio(5, cond, seed=42)
        assert all(p["popularity_ratio"] < 1.0 for p in portfolio)
