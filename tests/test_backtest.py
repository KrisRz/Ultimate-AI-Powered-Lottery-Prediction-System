"""Tests for backtest metrics and the significance machinery."""

import numpy as np

from scripts.validations.backtest import (
    RANDOM_EXPECTED_AVG,
    calc_match_counts,
    significance_vs_random,
)


class TestCalcMatchCounts:
    def test_full_match(self):
        assert calc_match_counts([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) == 6

    def test_no_match(self):
        assert calc_match_counts([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]) == 0

    def test_partial_match(self):
        assert calc_match_counts([1, 2, 3, 40, 50, 59], [1, 2, 3, 41, 51, 58]) == 3


class TestSignificance:
    def test_empty_series(self):
        assert significance_vs_random([]) == {}

    def test_random_expectation_constant(self):
        assert abs(RANDOM_EXPECTED_AVG - 36.0 / 59.0) < 1e-12

    def test_random_level_series_is_not_significant(self):
        # A series drawn from the actual null must not look like an edge
        rng = np.random.default_rng(7)
        series = rng.hypergeometric(6, 53, 6, size=500).tolist()
        sig = significance_vs_random(series, n_sim=2000, seed=7)
        assert sig["p_value_avg"] > 0.05

    def test_strong_series_is_significant(self):
        # Matching 3 every draw would be a massive edge
        sig = significance_vs_random([3] * 100, n_sim=2000, seed=7)
        assert sig["p_value_avg"] < 0.001
        assert sig["p_value_3plus"] < 0.001

    def test_reproducible_with_seed(self):
        series = [0, 1, 1, 0, 2, 0, 1, 0, 0, 1] * 10
        a = significance_vs_random(series, n_sim=1000, seed=42)
        b = significance_vs_random(series, n_sim=1000, seed=42)
        assert a == b

    def test_ci_contains_observed_mean(self):
        series = [0, 1, 1, 0, 2, 0, 1, 0, 0, 1] * 10
        sig = significance_vs_random(series, n_sim=2000, seed=1)
        lo, hi = sig["observed_avg_ci95"]
        assert lo <= sig["observed_avg"] <= hi
