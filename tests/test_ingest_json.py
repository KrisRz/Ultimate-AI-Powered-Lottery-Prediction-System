"""Tests for the api-dfe JSON ingestion (primary collection path)."""

import copy

import pandas as pd
import pytest

import scripts.fetch_data as fetch_data


def _level(draw_round, label, winners, per_winner, fund, roll_down=False):
    return {
        "drawRound": draw_round,
        "matchLabel": label,
        "allWinnersCount": winners,
        "prize": {"prizeCents": per_winner, "nonCashPrize": None},
        "prizeFundCents": fund,
        "prizeCap": False,
        "prizeRollDown": roll_down,
    }


SAMPLE_JSON = {
    "drawResult": {
        "gameId": 6,
        "drawNo": 3195,
        "drawDate": "2026-08-05T19:00:00.000Z",
        "topPrize": {"prizeCents": 685518900, "nonCashPrize": None},
        "drawnNumbers": {
            "drawnNumbers": {
                "primaryNumbers": [24, 26, 27, 39, 47, 50],
                "secondaryNumbers": [8],
            },
            "drawnNumbersAdditional": {
                "primaryNumbers": [12, 14, 16, 22, 29, 59],
                "secondaryNumbers": [58],
            },
        },
    },
    "prizeBreakdown": {
        "drawMachines": [
            {"drawNo": 3195, "machineName": "Lotto5", "ballSet": "L1"},
            {"drawNo": 3195, "machineName": "Lotto6", "ballSet": "L2"},
        ],
        "prizeLevels": [
            _level("ONE", "Match 6", 0, 0, 0),
            _level("ONE", "Match 3", 61893, 1000, 61893000),
            _level("TWO", "Match 6", 0, 0, 0),
            _level("TWO", "Match 2", 557659, 100, 55765900),
        ],
        "isJackpotRollover": True,
        "jackpotRolloverCount": 5,
    },
}


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "DOWNLOADED_FILE", tmp_path / "lotto-draw-history.csv")
    monkeypatch.setattr(fetch_data, "FULL_HISTORY_FILE", tmp_path / "lotto_full_history.csv")
    monkeypatch.setattr(fetch_data, "MERGED_FILE", tmp_path / "merged_lottery_data.csv")
    monkeypatch.setattr(fetch_data, "LATEST_JSON_FILE", tmp_path / "lotto-latest.json")
    monkeypatch.setattr(fetch_data, "PRIZE_TIERS_FILE", tmp_path / "prize_tiers.csv")
    return tmp_path


class TestIngestOfficialJson:
    def test_round1_lands_in_merged_file(self, isolated_data_dir):
        fetch_data._ingest_official_json(SAMPLE_JSON)
        merged = pd.read_csv(isolated_data_dir / "merged_lottery_data.csv")
        assert len(merged) == 1
        row = merged.iloc[0]
        assert row["Draw Date"] == "2026-08-05"
        nums = sorted(int(row[f"Number_{i}"]) for i in range(1, 7))
        assert nums == [24, 26, 27, 39, 47, 50]
        assert int(row["Bonus"]) == 8

    def test_tiers_carry_api_extras(self, isolated_data_dir):
        fetch_data._ingest_official_json(SAMPLE_JSON)
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert len(tiers) == 4
        m3 = tiers[(tiers["round"] == 1) & (tiers["tier"] == 5)].iloc[0]
        assert m3["winners"] == 61893
        assert m3["prize_per_winner"] == 10.0
        assert m3["prize_total"] == 618930.0
        assert not m3["tier_roll_down"]
        m2 = tiers[(tiers["round"] == 2) & (tiers["tier"] == 6)].iloc[0]
        assert m2["prize_per_winner"] == 1.0
        assert bool(tiers["rollover"].all())
        assert (tiers["rollover_count"] == 5).all()

    def test_forward_fields_stamped_from_xml(self, isolated_data_dir):
        fetch_data._ingest_official_json(SAMPLE_JSON, forward={
            "next_jackpot_estimate": 8391429.0,
            "next_jackpot_roll_down": True,
        })
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert (tiers["next_jackpot_estimate"] == 8391429.0).all()
        assert tiers["next_jackpot_roll_down"].all()

    def test_mbw_flag_derived_from_cap_when_xml_dead(self, isolated_data_dir):
        # rollover_count = 5 = the cap: even with no XML, the next draw MUST
        # be a Must-Be-Won - the alert path survives a dead redirect
        fetch_data._ingest_official_json(SAMPLE_JSON, forward={})
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert tiers["next_jackpot_roll_down"].all()

        below_cap = copy.deepcopy(SAMPLE_JSON)
        below_cap["prizeBreakdown"]["jackpotRolloverCount"] = 3
        below_cap["drawResult"]["drawNo"] = 3196
        fetch_data._ingest_official_json(below_cap, forward={})
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert not tiers[tiers["draw_number"] == 3196]["next_jackpot_roll_down"].any()

    def test_reingestion_does_not_duplicate_tiers(self, isolated_data_dir):
        fetch_data._ingest_official_json(SAMPLE_JSON)
        fetch_data._ingest_official_json(SAMPLE_JSON)
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert len(tiers) == 4

    def test_appends_both_rounds_to_full_history(self, isolated_data_dir):
        (isolated_data_dir / "lotto_full_history.csv").write_text(
            "Draw Date,Number_1,Number_2,Number_3,Number_4,Number_5,Number_6,"
            "Bonus,Jackpot,JackpotWins,Machine,Ball Set,DrawNumber,Round\n"
            "2026-08-01,1,2,3,4,5,6,7,5000000.0,0,Lotto5,1,3194,1\n")
        fetch_data._ingest_official_json(SAMPLE_JSON)
        full = pd.read_csv(isolated_data_dir / "lotto_full_history.csv")
        new = full[full["DrawNumber"] == 3195]
        assert sorted(new["Round"]) == [1, 2]
        # this draw's jackpot comes straight from topPrize, not a back-read
        assert (new["Jackpot"] == 6855189.0).all()
        r2 = new[new["Round"] == 2].iloc[0]
        assert int(r2["Bonus"]) == 58
        assert r2["Machine"] == "Lotto6"

    def test_invalid_numbers_raise(self, isolated_data_dir):
        bad = copy.deepcopy(SAMPLE_JSON)
        bad["drawResult"]["drawnNumbers"]["drawnNumbers"]["primaryNumbers"] = [
            24, 26, 27, 39, 47, 60]
        with pytest.raises(ValueError):
            fetch_data._ingest_official_json(bad)

    def test_unknown_tier_label_skipped_not_fatal(self, isolated_data_dir):
        odd = copy.deepcopy(SAMPLE_JSON)
        odd["prizeBreakdown"]["prizeLevels"].append(
            _level("ONE", "Raffle", 1, 100000, 100000))
        fetch_data._ingest_official_json(odd)
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert len(tiers) == 4
