"""Tests for data parsing and official XML ingestion."""

import pandas as pd
import pytest

import scripts.fetch_data as fetch_data
from scripts.fetch_data import parse_balls


class TestParseBalls:
    def test_valid_string(self):
        main, bonus = parse_balls("1 12 23 34 45 56 BONUS 7")
        assert main == [1, 12, 23, 34, 45, 56]
        assert bonus == 7

    def test_bare_numeric_is_rejected_not_fabricated(self):
        with pytest.raises(ValueError, match="bare numeric"):
            parse_balls("12345")

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            parse_balls("1 12 23 34 45 60 BONUS 7")

    def test_duplicate_rejected(self):
        with pytest.raises(ValueError):
            parse_balls("1 12 23 34 45 45 BONUS 7")


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<draw-results><game type="lotto"><draw><draw-number>3190</draw-number>
<draw-date>2026-07-18</draw-date><draw-machine>Lotto4</draw-machine>
<draw-machine>Lotto5</draw-machine></draw>
<balls><set>L3</set><ball number="1">22</ball><ball number="2">32</ball>
<ball number="3">34</ball><ball number="4">47</ball><ball number="5">52</ball>
<ball number="6">55</ball><bonus-ball type="bonusball" number="1">10</bonus-ball></balls>
<balls><set>L4</set><ball number="1">20</ball><ball number="2">29</ball>
<ball number="3">30</ball><ball number="4">42</ball><ball number="5">46</ball>
<ball number="6">55</ball><bonus-ball type="bonusball" number="1">38</bonus-ball></balls>
<winners><confirmed>Y</confirmed><prize-tiers>
<prize-tier level="1"><number-of-winners>0</number-of-winners><win-value>0.00</win-value></prize-tier>
<prize-tier level="5"><number-of-winners>82350</number-of-winners><win-value>1976400.00</win-value></prize-tier>
<prize-tier level="7"><number-of-winners>0</number-of-winners><win-value>0.00</win-value></prize-tier>
<prize-tier level="11"><number-of-winners>87088</number-of-winners><win-value>2090112.00</win-value></prize-tier>
</prize-tiers></winners>
<rollover>N</rollover><rollover-count>0</rollover-count>
<next-estimated-jackpot>2,000,000</next-estimated-jackpot>
<next-estimated-jackpot-roll-down>N</next-estimated-jackpot-roll-down>
</game></draw-results>"""


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fetch_data, "DOWNLOADED_FILE", tmp_path / "lotto-draw-history.csv")
    monkeypatch.setattr(fetch_data, "FULL_HISTORY_FILE", tmp_path / "lotto_full_history.csv")
    monkeypatch.setattr(fetch_data, "MERGED_FILE", tmp_path / "merged_lottery_data.csv")
    monkeypatch.setattr(fetch_data, "LATEST_XML_FILE", tmp_path / "lotto-latest.xml")
    monkeypatch.setattr(fetch_data, "PRIZE_TIERS_FILE", tmp_path / "prize_tiers.csv")
    return tmp_path


class TestIngestOfficialXml:
    def test_round1_lands_in_merged_file(self, isolated_data_dir):
        fetch_data._ingest_official_xml(SAMPLE_XML)
        merged = pd.read_csv(isolated_data_dir / "merged_lottery_data.csv")
        assert len(merged) == 1
        row = merged.iloc[0]
        assert row["Draw Date"] == "2026-07-18"
        nums = sorted(int(row[f"Number_{i}"]) for i in range(1, 7))
        assert nums == [22, 32, 34, 47, 52, 55]  # Round 1 set only
        assert int(row["Bonus"]) == 10

    def test_prize_tiers_are_recorded_per_round(self, isolated_data_dir):
        fetch_data._ingest_official_xml(SAMPLE_XML)
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert len(tiers) == 4
        # Level 11 maps to round 2, tier 5
        r2 = tiers[(tiers["round"] == 2) & (tiers["tier"] == 5)]
        assert len(r2) == 1
        assert int(r2.iloc[0]["winners"]) == 87088
        assert float(tiers.iloc[0]["next_jackpot_estimate"]) == 2_000_000.0

    def test_reingestion_does_not_duplicate_tiers(self, isolated_data_dir):
        fetch_data._ingest_official_xml(SAMPLE_XML)
        fetch_data._ingest_official_xml(SAMPLE_XML)
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert len(tiers) == 4

    def test_appends_both_rounds_to_full_history(self, isolated_data_dir):
        pd.DataFrame([{
            "Draw Date": "2026-07-15", "Number_1": 1, "Number_2": 2, "Number_3": 3,
            "Number_4": 4, "Number_5": 5, "Number_6": 6, "Bonus": 7, "Jackpot": 0,
            "JackpotWins": 0, "Machine": "Lotto 4", "Ball Set": "3",
            "DrawNumber": 3189, "Round": 1,
        }]).to_csv(isolated_data_dir / "lotto_full_history.csv", index=False)
        fetch_data._ingest_official_xml(SAMPLE_XML)
        full = pd.read_csv(isolated_data_dir / "lotto_full_history.csv")
        assert len(full) == 3
        new = full[full["DrawNumber"] == 3190]
        assert sorted(new["Round"]) == [1, 2]

    def test_invalid_numbers_raise(self, isolated_data_dir):
        bad = SAMPLE_XML.replace(b'<ball number="6">55</ball>', b'<ball number="6">99</ball>', 1)
        with pytest.raises(ValueError, match="invalid numbers"):
            fetch_data._ingest_official_xml(bad)
