"""Tests for the api-dfe JSON ingestion (primary collection path)."""

import copy
from datetime import date

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
    # Every collector-owned file the ingest writes belongs here. draw_pools.csv
    # was missing from this list when it was added, so the only thing standing
    # between the suite and the repo's real pools file was that the ingest
    # never reached the write at all.
    monkeypatch.setattr(fetch_data, "DRAW_POOLS_FILE", tmp_path / "draw_pools.csv")
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

    def test_records_the_pool_the_draw_actually_carried(self, isolated_data_dir):
        """The number the whole sales identity divides.

        This lived in `_ingest_official_xml` for its first day, where `draw` is
        an XML Element and `Element.get("topPrize")` is the ATTRIBUTE accessor -
        so it returned None, skipped the write, raised nothing, and the
        collector silently never recorded a pool.
        """
        fetch_data._ingest_official_json(SAMPLE_JSON)
        pools = pd.read_csv(isolated_data_dir / "draw_pools.csv")
        assert list(pools["draw_number"]) == [3195]
        assert pools["pool_gbp"].iloc[0] == 6_855_189.0
        assert pools["rollover_count"].iloc[0] == 5

    def test_an_outright_win_records_no_rollover_count_rather_than_zero(
            self, isolated_data_dir):
        """The feed sends null when the jackpot was won, and "won" is not
        "rolled zero times" - must_be_won_after_cap reads this column."""
        won = copy.deepcopy(SAMPLE_JSON)
        won["prizeBreakdown"]["jackpotRolloverCount"] = None
        fetch_data._ingest_official_json(won)
        pools = pd.read_csv(isolated_data_dir / "draw_pools.csv")
        assert pd.isna(pools["rollover_count"].iloc[0])

    def test_pools_accumulate_without_duplicating(self, isolated_data_dir):
        fetch_data._ingest_official_json(SAMPLE_JSON)
        later = copy.deepcopy(SAMPLE_JSON)
        later["drawResult"]["drawNo"] = 3196
        later["drawResult"]["topPrize"]["prizeCents"] = 853514700
        fetch_data._ingest_official_json(later)
        fetch_data._ingest_official_json(SAMPLE_JSON)          # idempotent retry
        pools = pd.read_csv(isolated_data_dir / "draw_pools.csv")
        assert list(pools["draw_number"]) == [3195, 3196]
        assert list(pools["pool_gbp"]) == [6_855_189.0, 8_535_147.0]

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


class TestDrawsSince:
    """Counting from the calendar, the one source that cannot be mid-update."""

    def _at(self, monkeypatch, upcoming):
        monkeypatch.setattr(fetch_data, "upcoming_draw_date", lambda: upcoming)

    def test_the_latest_draw_has_nothing_after_it(self, monkeypatch):
        # Ingesting Saturday's draw on Saturday night: the next draw you can
        # buy into is Wednesday, and nothing has happened in between.
        self._at(monkeypatch, date(2026, 9, 9))
        assert fetch_data._draws_since("2026-09-05") == 0

    def test_a_drawn_but_unpublished_draw_is_counted(self, monkeypatch):
        # The live case, 2026-09-05 20:16 BST: the feed still served 3203
        # (2026-09-02) as latest, with Saturday's draw already run.
        self._at(monkeypatch, date(2026, 9, 9))
        assert fetch_data._draws_since("2026-09-02") == 1

    def test_a_long_gap_counts_every_draw_in_it(self, monkeypatch):
        self._at(monkeypatch, date(2026, 9, 9))
        # 08-26, 08-29, 09-02, 09-05 - the 09-09 draw is the upcoming one,
        # which has not happened and is not counted.
        assert fetch_data._draws_since("2026-08-22") == 4

    def test_an_unparseable_date_claims_nothing(self, monkeypatch):
        self._at(monkeypatch, date(2026, 9, 9))
        assert fetch_data._draws_since("not a date") == 0


class TestForwardFieldsDuringTheUpdateWindow:
    """The minutes after a draw, when the XML answers but has nothing to say."""

    XML = (b'<?xml version="1.0"?><lotto><game>'
           b'<next-estimated-jackpot>%s</next-estimated-jackpot>'
           b'<next-estimated-jackpot-roll-down>%s</next-estimated-jackpot-roll-down>'
           b'</game></lotto>')

    class _Resp:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    class _Session:
        def __init__(self, content):
            self.content = content

        def get(self, url, **kwargs):
            return TestForwardFieldsDuringTheUpdateWindow._Resp(self.content)

    def _read(self, estimate: bytes, roll_down: bytes = b"N", tmp_path=None):
        return fetch_data._forward_fields_from_xml(
            self._Session(self.XML % (estimate, roll_down)), {})

    def test_a_published_estimate_is_taken(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_data, "LATEST_XML_FILE", tmp_path / "x.xml")
        assert self._read(b"7,706,666", b"Y") == {
            "draw_number": None,          # this fixture carries no <draw> block
            "next_jackpot_estimate": 7706666.0, "next_jackpot_roll_down": True}

    def test_an_impossible_estimate_is_an_absence_not_a_number(
            self, tmp_path, monkeypatch):
        """Observed live 2026-09-05 20:11 BST, eleven minutes after the draw:
        this endpoint served 0 and then GBP 7,706,666 three minutes later. The
        licence floor is GBP 2m, so anything under it is not yet published -
        and the roll-down flag beside it is worth no more than the number."""
        monkeypatch.setattr(fetch_data, "LATEST_XML_FILE", tmp_path / "x.xml")
        assert self._read(b"0", b"Y") == {}
        assert self._read(b"1,999,999") == {}

    def test_a_blank_estimate_still_yields_a_flag_free_block(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(fetch_data, "LATEST_XML_FILE", tmp_path / "x.xml")
        assert self._read(b"")["next_jackpot_estimate"] is None

    def test_a_retry_never_turns_off_a_promotional_must_be_won(
            self, isolated_data_dir):
        """The cap derivation cannot see a special draw Allwyn designates
        Must-Be-Won without five rollovers, so a retry must not unset it."""
        below_cap = copy.deepcopy(SAMPLE_JSON)
        below_cap["prizeBreakdown"]["jackpotRolloverCount"] = 2
        fetch_data._ingest_official_json(below_cap, forward={
            "next_jackpot_estimate": 15_000_000.0, "next_jackpot_roll_down": True})
        fetch_data._ingest_official_json(below_cap, forward={})
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert tiers["next_jackpot_roll_down"].all()
        assert (tiers["next_jackpot_estimate"] == 15_000_000.0).all()

    def test_a_retry_never_blanks_an_estimate_already_on_file(
            self, isolated_data_dir):
        """The retry run exists to fill gaps; dedupe keeps the NEWEST row, and
        newest must not mean least informed."""
        fetch_data._ingest_official_json(SAMPLE_JSON, forward={
            "next_jackpot_estimate": 8391429.0, "next_jackpot_roll_down": True})
        fetch_data._ingest_official_json(SAMPLE_JSON, forward={})
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert (tiers["next_jackpot_estimate"] == 8391429.0).all()


class TestRecoverMissingDraws:
    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class _FakeSession:
        def __init__(self, payloads):
            self.payloads = payloads
            self.requested = []

        def get(self, url, **kwargs):
            draw = int(url.rstrip("/").rsplit("/", 1)[-1])
            self.requested.append(draw)
            return TestRecoverMissingDraws._FakeResp(self.payloads[draw])

    def _payload_for(self, draw_no):
        p = copy.deepcopy(SAMPLE_JSON)
        p["drawResult"]["drawNo"] = draw_no
        p["prizeBreakdown"]["jackpotRolloverCount"] = 0
        return p

    def test_backfills_gap_between_collected_and_latest(self, isolated_data_dir):
        fetch_data._ingest_official_json(self._payload_for(3195))
        session = self._FakeSession({3196: self._payload_for(3196),
                                     3197: self._payload_for(3197)})
        n = fetch_data.recover_missing_draws(session, {}, latest_drawno=3198)
        assert n == 2
        assert session.requested == [3196, 3197]
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert set(tiers["draw_number"]) == {3195, 3196, 3197}

    def test_no_gap_requests_nothing(self, isolated_data_dir):
        fetch_data._ingest_official_json(self._payload_for(3195))
        session = self._FakeSession({})
        assert fetch_data.recover_missing_draws(session, {}, latest_drawno=3196) == 0
        assert session.requested == []

    def test_one_dead_draw_does_not_stop_the_rest(self, isolated_data_dir):
        fetch_data._ingest_official_json(self._payload_for(3195))
        session = self._FakeSession({3197: self._payload_for(3197)})  # 3196 -> KeyError
        n = fetch_data.recover_missing_draws(session, {}, latest_drawno=3198)
        assert n == 1
        tiers = pd.read_csv(isolated_data_dir / "prize_tiers.csv")
        assert 3197 in set(tiers["draw_number"])
