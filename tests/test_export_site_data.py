"""The public site's data snapshot: determinism, and claims that match the model.

Everything the site renders comes from site/public/data/site.json. Two classes
of bug matter here and nothing else catches either one:

  * The file drifts from the model. Someone recalibrates `number_weight`, the
    committed snapshot still holds the old heatmap, and the public page quietly
    publishes a stale claim. `test_committed_snapshot_is_current` is the guard,
    and CI runs the same check.
  * The file stops being reproducible. A wall-clock read or an unrounded
    regression coefficient makes it diff on every run, the drift check starts
    crying wolf, and it gets switched off.
"""

import json
from dataclasses import replace
from datetime import date
from math import comb

import pandas as pd
import pytest

from lottery.ev import (
    N_BALLS,
    N_PICK,
    TOTAL_COMBOS,
    DrawConditions,
    break_even_jackpot,
    line_ev,
    popularity_ratio,
)
from scripts.export_site_data import (
    DP,
    EXACT_KEYS,
    OUT_FILE,
    PROB_DP,
    affine,
    build_payload,
    priced_at,
    serialize,
)


@pytest.fixture(scope="module")
def payload():
    """Built once - the popularity regression reads the whole tier history."""
    return build_payload()


@pytest.fixture(scope="module")
def text(payload):
    return serialize(payload)


@pytest.fixture(scope="module")
def written(text):
    """What actually lands in the file - rounding happens during serialization,
    so assertions about precision have to read this, not the raw payload."""
    return json.loads(text)


def walk(obj, path=""):
    """Every (path, scalar) leaf in the tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, obj


class TestDeterminism:
    def test_two_builds_serialize_identically(self, text):
        """The property the whole CI drift check rests on."""
        assert serialize(build_payload()) == text

    def test_no_wall_clock_field(self, payload):
        """dashboard.gather() stamps datetime.now(); copying that here would
        rewrite the committed file on every run."""
        stamped = [p for p, _ in walk(payload)
                   if any(w in p.lower() for w in ("generated", "timestamp", "exported_at"))]
        assert stamped == []

    def test_priced_at_is_derived_from_the_data(self):
        """The draw being priced follows the last collected draw, not today."""
        tiers = pd.DataFrame({"draw_date": ["2026-07-29", "2026-08-05"]})
        assert priced_at(tiers) == date(2026, 8, 6)

    def test_serialization_is_sorted_and_newline_terminated(self, text):
        assert text.endswith("\n")
        lines = text.splitlines()
        assert lines[0] == "{" and lines[-1] == "}"
        top = [ln.strip().split('"')[1] for ln in lines if ln.startswith('  "')]
        assert top == sorted(top)

    def test_committed_snapshot_is_current(self, text):
        """Fails when the model moved and `make site-data` was not re-run."""
        assert OUT_FILE.exists(), f"{OUT_FILE} has never been generated"
        assert OUT_FILE.read_text() == text, (
            f"{OUT_FILE} is stale - run `make site-data` and commit the result")


class TestRounding:
    def test_no_float_carries_more_precision_than_declared(self, written):
        for path, value in walk(written):
            if not isinstance(value, float) or isinstance(value, bool):
                continue
            key = path.rsplit(".", 1)[-1].split("[")[0]
            if key in EXACT_KEYS:
                continue
            limit = PROB_DP if key == "probability" else DP
            assert value == round(value, limit), f"{path} = {value!r}"

    def test_model_coefficients_keep_full_precision(self, written):
        """b is ~1e-07. Rounded to the general 4 dp it would be exactly zero,
        and break-even would read as infinity."""
        for regime in written["ev"]["regimes"]:
            assert 0 < regime["b"] < 1e-6
            assert regime["b"] != round(regime["b"], DP)


class TestBacktest:
    def test_every_method_shares_one_draw_grid(self, payload):
        """Four lines on one axis invite a comparison, so they had better be
        scored on the same draws."""
        series = payload["backtest"]["series"]
        n = len(series["dates"])
        assert n == payload["backtest"]["steps"]
        for name, values in series["cumulative_avg"].items():
            assert len(values) == n, name

    def test_the_baseline_is_present_and_labelled(self, payload):
        methods = payload["backtest"]["methods"]
        baselines = [m for m in methods if m["is_baseline"]]
        assert [m["name"] for m in baselines] == ["random"]
        assert len(methods) >= 4

    def test_no_skill_mean_is_36_over_59(self, payload):
        assert payload["backtest"]["expected_random_avg"] == pytest.approx(36 / 59, abs=1e-4)

    def test_p_values_are_probabilities(self, payload):
        for method in payload["backtest"]["methods"]:
            assert 0.0 <= method["p_value_avg"] <= 1.0, method["name"]
            assert 0.0 <= method["p_value_3plus"] <= 1.0, method["name"]

    def test_nothing_beats_random(self, payload):
        """The page's headline claim, asserted rather than asserted-in-prose.
        If a method ever does clear this, the copy in panel B is wrong and this
        test is the thing that says so."""
        winners = [m["name"] for m in payload["backtest"]["methods"] if m["beats_random"]]
        assert winners == []

    def test_every_interval_contains_the_no_skill_mean(self, payload):
        """The stronger form of the same finding, and the one the chart draws:
        each method's CI95 straddles 36/59."""
        expected = payload["backtest"]["expected_random_avg"]
        for method in payload["backtest"]["methods"]:
            low, high = method["ci95"]
            assert low <= expected <= high, f"{method['name']} excludes the no-skill mean"

    def test_cumulative_average_settles_near_the_no_skill_mean(self, payload):
        expected = payload["backtest"]["expected_random_avg"]
        for name, values in payload["backtest"]["series"]["cumulative_avg"].items():
            assert values[-1] == pytest.approx(expected, abs=0.05), name


class TestEvIsAffine:
    """EV(J) = a + b*J is what lets the site's slider be exact rather than a
    49-point interpolation. If line_ev ever stops being linear in the jackpot,
    the page would keep drawing a straight line through a curve."""

    def _conds(self):
        base = dict(tickets_sold=9_710_793, rounds=2, draw_date=date(2026, 8, 8))
        return [DrawConditions(roll_down=False, **base),
                DrawConditions(roll_down=True, **base)]

    def test_matches_line_ev_across_the_slider_range(self):
        line = [32, 34, 37, 39, 41, 43]
        for cond in self._conds():
            a, b = affine(line, cond)
            for jackpot in range(2_000_000, 50_000_001, 2_000_000):
                exact = line_ev(line, replace(cond, jackpot=float(jackpot)))
                assert a + b * jackpot == pytest.approx(exact, abs=1e-9)

    def test_break_even_is_the_root(self):
        line = [32, 34, 37, 39, 41, 43]
        for cond in self._conds():
            a, b = affine(line, cond)
            assert -a / b == pytest.approx(break_even_jackpot(cond, line), rel=1e-9)

    def test_exported_regimes_agree_with_their_break_even(self, payload):
        for regime in payload["ev"]["regimes"]:
            root = -regime["a"] / regime["b"]
            assert root == pytest.approx(regime["break_even_jackpot"], rel=1e-6)


class TestHook:
    def test_total_combinations(self, payload):
        assert payload["hook"]["total_combinations"] == comb(N_BALLS, N_PICK) == TOTAL_COMBOS

    def test_odds_are_probabilities_and_their_reciprocals(self, payload):
        odds = payload["hook"]["odds"]
        assert [o["key"] for o in odds][0] == "jackpot"
        for o in odds:
            assert 0 < o["probability"] < 1
            # one_in is rounded to 1 dp for display, so it can sit half a tenth
            # off its own reciprocal - Match 2 is 10.3 against a true 10.258.
            assert o["one_in"] == pytest.approx(1 / o["probability"], abs=0.05)

    def test_jackpot_odds_match_the_published_figure(self, payload):
        jackpot = payload["hook"]["odds"][0]
        assert round(jackpot["one_in"]) == 45_057_474


class TestPopularity:
    def test_recovered_weights_cover_every_ball_and_average_one(self, payload):
        recovered = payload["popularity"]["recovered"]
        assert len(recovered) == N_BALLS
        assert sum(recovered) / N_BALLS == pytest.approx(1.0, abs=5e-4)

    def test_examples_match_the_live_model(self, payload):
        """The site prints these ratios next to real lines; they have to be
        what popularity_ratio actually returns, not a transcription."""
        for example in payload["popularity"]["examples"]:
            assert example["ratio"] == pytest.approx(
                popularity_ratio(example["line"]), abs=10 ** -DP)

    def test_birthday_draws_yield_more_low_tier_winners(self, payload):
        """The evidence the whole popularity model rests on: the trend across
        buckets is monotone, not a single suggestive endpoint."""
        buckets = payload["popularity"]["match3_multiplier_by_low31"]
        means = [b["mean_multiplier"] for b in buckets]
        assert means == sorted(means)
        assert means[-1] > 2 * means[0]

    def test_installed_step_matches_ev_module(self, payload):
        from lottery.ev import number_weight
        for band in payload["popularity"]["installed_step"]:
            assert band["weight"] == number_weight(band["from"]) == number_weight(band["to"])

    def test_high_numbers_are_the_under_played_ones(self, payload):
        least = [x["number"] for x in payload["popularity"]["least_played"]]
        assert all(n > 31 for n in least)


class TestLiveVerdict:
    def test_verdict_is_one_of_two_words(self, payload):
        assert payload["ev"]["live"]["verdict"] in ("PLAY", "SKIP")

    def test_snapshot_prices_the_draw_after_the_data(self, payload):
        as_of = date.fromisoformat(payload["snapshot"]["as_of_draw_date"])
        through = date.fromisoformat(payload["snapshot"]["data_through"])
        assert as_of > through
        assert as_of.weekday() in (2, 5)

    def test_sales_band_brackets_the_central_estimate(self, payload):
        """A roll-down verdict turns on one estimated number, so the page shows
        the quartiles. p25 sells less, so it needs a smaller jackpot to pay."""
        band = payload["ev"]["mbw_sales_band"]
        mbw = next(r for r in payload["ev"]["regimes"] if r["key"] == "mbw")
        assert band["p25"]["tickets_sold"] < mbw["tickets_sold"] < band["p75"]["tickets_sold"]
        assert (band["p25"]["break_even_jackpot"]
                < mbw["break_even_jackpot"]
                < band["p75"]["break_even_jackpot"])
