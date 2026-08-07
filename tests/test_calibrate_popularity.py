"""Tests for the popularity-calibration math.

These use synthetic data with a KNOWN popularity structure, so the calibration
must recover it: this pins the un-damping factor and the population
normalization, and guards the sales-trend division.
"""

import math

import numpy as np
import pandas as pd
import pytest

from lottery.ev import P_MATCH_3
from scripts.calibrate_popularity import (
    NUMBER_COLS,
    add_multiplier,
    fit_bucket_weights,
)

# Known bucket pick-rates, already normalized to a population mean of 1.0:
# 12*W_LOW + 19*W_MID + 28*W_HIGH == 59
W_LOW, W_MID, W_HIGH = 1.30, 1.10, 0.8035714285714286


def _weight(n: int) -> float:
    return W_LOW if n <= 12 else (W_MID if n <= 31 else W_HIGH)


def _make_draw(n_low: int, n_high: int) -> list[int]:
    """Six distinct numbers with n_low in 1-12, n_high in 32-59, rest 13-31."""
    lows = list(range(1, 1 + n_low))
    highs = list(range(32, 32 + n_high))
    mids = list(range(13, 13 + (6 - n_low - n_high)))
    return sorted(lows + mids + highs)


def _synthetic_draws() -> pd.DataFrame:
    """One draw per (n_low, n_high) combination, multiplier from the forward
    model log(mult) = 0.5 * sum_n log(weight_n) (the Match-3 first-order term)."""
    rows = []
    dn = 2066
    for n_low in range(0, 7):
        for n_high in range(0, 7 - n_low):
            nums = _make_draw(n_low, n_high)
            mult = math.exp(0.5 * sum(math.log(_weight(n)) for n in nums))
            row = {c: nums[i] for i, c in enumerate(NUMBER_COLS)}
            row["multiplier"] = mult
            row["draw_number"], row["round"] = dn, 1
            rows.append(row)
            dn += 1
    return pd.DataFrame(rows)


class TestBucketRecovery:
    def test_recovers_known_weights(self):
        buckets = fit_bucket_weights(_synthetic_draws())
        assert buckets["low12"] == pytest.approx(W_LOW, rel=1e-3)
        assert buckets["mid"] == pytest.approx(W_MID, rel=1e-3)
        assert buckets["high"] == pytest.approx(W_HIGH, rel=1e-3)

    def test_recovered_weights_average_to_one(self):
        b = fit_bucket_weights(_synthetic_draws())
        assert (12 * b["low12"] + 19 * b["mid"] + 28 * b["high"]) / 59 == pytest.approx(1.0)

    def test_ordering(self):
        b = fit_bucket_weights(_synthetic_draws())
        assert b["low12"] > b["mid"] > b["high"]


class TestSalesTrend:
    def test_multiplier_divides_out_slow_sales_trend(self):
        # Constant popularity, but sales N drift 8M -> 5M over 200 draws.
        n_draws = 200
        sales = np.linspace(8_000_000, 5_000_000, n_draws)
        df = pd.DataFrame({
            "draw_number": range(2066, 2066 + n_draws),
            "round": 1,
            "winners": np.round(sales * P_MATCH_3).astype(int),
        })
        out = add_multiplier(df, window=51)
        # With popularity held flat, the recovered multiplier must sit at ~1.
        assert out["multiplier"].median() == pytest.approx(1.0, rel=0.02)
        assert out["multiplier"].std() < 0.05


class TestMatchDegreeUndamping:
    """The un-damp factor is 6/k: a tier matching k of the drawn 6 sees only
    k/6 of the drawn set's log-weight sum. Synthetic multipliers built with a
    known k must recover the same weights for every tier."""

    def _synthetic(self, k: int, n: int = 4000, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            line = sorted(rng.choice(np.arange(1, 60), size=6, replace=False))
            logsum = sum(math.log(_weight(int(x))) for x in line)
            mult = math.exp((k / 6) * logsum)
            rows.append({**{c: v for c, v in zip(NUMBER_COLS, line)},
                         "multiplier": mult})
        return pd.DataFrame(rows)

    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_recovers_weights_for_each_match_degree(self, k):
        from scripts.calibrate_popularity import fit_bucket_weights
        b = fit_bucket_weights(self._synthetic(k), match_degree=k)
        assert b["low12"] == pytest.approx(W_LOW, rel=0.02)
        assert b["mid"] == pytest.approx(W_MID, rel=0.02)
        assert b["high"] == pytest.approx(W_HIGH, rel=0.02)

    def test_wrong_degree_biases_the_spread(self):
        """Un-damping a Match-2 signal with the Match-3 factor must understate
        the weight spread - the failure mode this parameter exists to avoid."""
        from scripts.calibrate_popularity import fit_bucket_weights
        b = fit_bucket_weights(self._synthetic(2), match_degree=3)
        assert b["low12"] < W_LOW * 0.99
        assert b["high"] > W_HIGH * 1.01
