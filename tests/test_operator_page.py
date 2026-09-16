"""The operator's page as a second opinion on a PLAY email.

Shapes below are the live ones: the page read on 2026-09-16, and the two
values github.com/eagerterrier/pwa-badge recorded before draws 3205 and 3206.
"""

from datetime import datetime, timezone

from scripts.monitoring.operator_page import (
    compare,
    find_assigned_game,
    parse_assigned_game,
)


def _page(phase="INITIAL", pence=200000000, close="2026-09-16T18:30:00Z"):
    return {"headerArea": [{"dbgJackpotCard": {
        "@name": "Lotto-ECHO-predefined",
        "disclaimer": {"symbol": "*"},
        "assignedGame": {"sellEndDate": close, "gamePrice": 200,
                         "jackpot": pence, "phase": phase},
    }}]}


class TestReadingThePage:
    def test_finds_the_block_wherever_it_sits(self):
        game = find_assigned_game({"other": [1, {"x": None}], **_page()})
        assert game["phase"] == "INITIAL"

    def test_no_block_no_answer(self):
        assert find_assigned_game({"headerArea": []}) is None
        assert parse_assigned_game(None) == {}

    def test_pence_become_pounds_and_the_close_a_time(self):
        page = parse_assigned_game(find_assigned_game(_page()))
        assert page["jackpot"] == 2_000_000
        assert page["must_be_won"] is False
        assert page["sales_close"] == datetime(2026, 9, 16, 18, 30, tzinfo=timezone.utc)


class TestComparing:
    def test_the_3205_evening_agrees(self):
        """XML: GBP 7,706,666, roll-down Y. Page: 770666609 pence, MUST_BE_WON."""
        page = parse_assigned_game(find_assigned_game(_page("MUST_BE_WON", 770666609)))
        agrees, line = compare(True, 7_706_666, page)
        assert agrees is True
        assert "agrees" in line and "sales close" in line

    def test_a_missing_flag_is_a_disagreement(self):
        page = parse_assigned_game(find_assigned_game(_page("MUST_BE_WON", 1200000000)))
        agrees, line = compare(False, 12_000_000, page)
        assert agrees is False
        assert "phase MUST_BE_WON" in line

    def test_a_wrong_jackpot_is_a_disagreement(self):
        """The half-filled XML block of 2026-09-05 read a jackpot of 0."""
        page = parse_assigned_game(find_assigned_game(_page("MUST_BE_WON", 1500000000)))
        agrees, line = compare(True, 2_000_000, page)
        assert agrees is False
        assert "£15,000,000" in line and "before buying" in line

    def test_an_unreadable_page_is_unknown_not_wrong(self):
        agrees, line = compare(True, 15_000_000, {})
        assert agrees is None
        assert "unreachable" in line
