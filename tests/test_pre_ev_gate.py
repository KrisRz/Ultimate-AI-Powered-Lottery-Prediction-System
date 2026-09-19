"""Tests for scripts/monitoring/pre_ev_gate.py.

The gate's whole justification is that it blocks a DIFFERENT class of
problem from the checks behind the alert. So the tests come in pairs: a
broken input must block, and a stale-but-sound archive must not - because
suppressing a PLAY email over stale history is the failure PR #40 fixed,
and this must not reintroduce it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.monitoring.pre_ev_gate as gate


@pytest.fixture
def pools(tmp_path, monkeypatch):
    """A sound pool file, pointed at by the gate."""
    f = tmp_path / "draw_pools.csv"
    pd.DataFrame([
        {"draw_number": 3206, "draw_date": "2026-09-12", "pool_gbp": 12_000_000.0},
        {"draw_number": 3207, "draw_date": "2026-09-16", "pool_gbp": 2_000_000.0},
    ]).to_csv(f, index=False)
    monkeypatch.setattr(gate, "POOLS_FILE", f)
    return f


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    f = tmp_path / "prize_tiers.csv"
    pd.DataFrame([{"draw_number": 3207, "tier": 5, "winners": 1000}]).to_csv(
        f, index=False)
    monkeypatch.setattr(gate, "TIERS_FILE", f)
    return f


def test_sound_inputs_pass(pools, tiers):
    assert gate.run() == []


# --- things that make tonight's number wrong ------------------------------

def test_missing_pool_file_blocks(tmp_path, monkeypatch, tiers):
    monkeypatch.setattr(gate, "POOLS_FILE", tmp_path / "absent.csv")
    assert any("missing" in p for p in gate.run())


def test_non_finite_pool_blocks(pools, tiers):
    df = pd.read_csv(pools)
    df.loc[df.index[-1], "pool_gbp"] = float("nan")
    df.to_csv(pools, index=False)
    assert any("finite" in p for p in gate.run())


def test_negative_pool_blocks(pools, tiers):
    df = pd.read_csv(pools)
    df.loc[df.index[-1], "pool_gbp"] = -5.0
    df.to_csv(pools, index=False)
    assert any("positive" in p for p in gate.run())


def test_implausible_pool_blocks(pools, tiers):
    """A units change in the feed reads as a jackpot no lottery has paid."""
    df = pd.read_csv(pools)
    df.loc[df.index[-1], "pool_gbp"] = 900_000_000.0
    df.to_csv(pools, index=False)
    assert any("units" in p for p in gate.run())


def test_pool_far_below_the_guaranteed_minimum_blocks(pools, tiers):
    df = pd.read_csv(pools)
    df.loc[df.index[-1], "pool_gbp"] = 50_000.0
    df.to_csv(pools, index=False)
    assert any("minimum" in p for p in gate.run())


def test_a_pool_dated_after_the_draw_being_priced_blocks(pools, tiers):
    """The advisor prices the NEXT draw; a later collected date means the
    sources disagree about which draw that is."""
    df = pd.read_csv(pools)
    df.loc[df.index[-1], "draw_date"] = "2099-01-01"
    df.to_csv(pools, index=False)
    assert any("already happened" in p for p in gate.run())


def test_missing_tier_file_blocks(pools, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "TIERS_FILE", tmp_path / "absent.csv")
    assert any("fixed prizes" in p for p in gate.run())


def test_empty_tier_file_blocks(pools, tiers):
    pd.DataFrame(columns=["draw_number", "tier", "winners"]).to_csv(
        tiers, index=False)
    assert any("empty" in p for p in gate.run())


# --- things that must NOT block -------------------------------------------

def test_a_stale_but_sound_archive_passes(pools, tiers):
    """The failure PR #40 fixed, guarded from the other direction.

    Old history is a monitoring problem: the advisor prices the next draw
    from the CURRENT pool, so staleness cannot make tonight's figure
    wrong, and must never suppress the alert.
    """
    df = pd.read_csv(pools)
    df["draw_date"] = ["2019-01-02", "2019-01-05"]
    df["draw_number"] = [2500, 2501]
    df.to_csv(pools, index=False)
    assert gate.run() == []


def test_an_unknown_machine_does_not_reach_this_gate(pools, tiers):
    """Category drift belongs to the data contract, after the alert."""
    assert gate.run() == []


def test_the_live_repo_passes():
    assert gate.run() == [], "the committed data should price a draw"
