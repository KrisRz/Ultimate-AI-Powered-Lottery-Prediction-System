"""Tests for the lottery.co.uk prize-tier backfill parser.

Fixtures mirror the real page markup for both layouts: single-round (pre-2026)
and two-round (since 2026-06). The winner numbers below are the actual values
for draws 2528 and 3191 - the latter cross-checked against the official feed.
"""

import pytest

from scripts.backfill_prize_tiers import parse_breakdown

# --- single-round era (draw 2528, 2020-03-14) -------------------------------
SINGLE_ROUND_HTML = """
<a id="prizeBreakdown"></a>
<h2>Prize Breakdown</h2>
<table class="table lotto mobFormat">
<thead><tr>
  <th>Category</th><th>Winners</th><th>Prize Per Winner</th><th>Prize Fund Amount</th>
</tr></thead>
<tbody>
<tr><td class="colour"><strong>Match 6</strong></td>
  <td data-title="Winners">1</td>
  <td data-title="Prize Per Winner">&pound;4,037,783</td>
  <td data-title="Prize Fund Amount">&pound;4,037,783</td></tr>
<tr class="alt"><td class="colour"><strong>Match 5 plus Bonus</strong></td>
  <td data-title="Winners">6</td>
  <td data-title="Prize Per Winner">&pound;1,000,000</td>
  <td data-title="Prize Fund Amount">&pound;6,000,000</td></tr>
<tr><td class="colour"><strong>Match 5</strong></td>
  <td data-title="Winners">91</td>
  <td data-title="Prize Per Winner">&pound;1,750</td>
  <td data-title="Prize Fund Amount">&pound;159,250</td></tr>
<tr><td class="colour"><strong>Match 4</strong></td>
  <td data-title="Winners">6,218</td>
  <td data-title="Prize Per Winner">&pound;140</td>
  <td data-title="Prize Fund Amount">&pound;870,520</td></tr>
<tr><td class="colour"><strong>Match 3</strong></td>
  <td data-title="Winners">129,966</td>
  <td data-title="Prize Per Winner">&pound;30</td>
  <td data-title="Prize Fund Amount">&pound;3,898,980</td></tr>
<tr><td class="colour"><strong>Match 2</strong></td>
  <td data-title="Winners">1,135,143</td>
  <td data-title="Prize Per Winner">Free Lucky Dip</td>
  <td data-title="Prize Fund Amount">&pound;0</td></tr>
</tbody></table>
"""

# --- two-round era (draw 3191, 2026-07-22) ----------------------------------
TWO_ROUND_HTML = """
<a id="prizeBreakdown"></a>
<h2>Prize Breakdown</h2>
<table class="table lotto mobFormat">
<thead><tr>
  <th>Category</th><th>Prize</th><th>Round 1 Winners</th>
  <th>Round 2 Winners</th><th>Total Winners</th><th>Prize Fund</th>
</tr></thead>
<tbody>
<tr><td class="colour"><strong>Match 6</strong></td>
  <td data-title="Prize">&pound;2,000,000</td>
  <td data-title="Round 1 Winners">0</td>
  <td data-title="Round 2 Winners">0</td>
  <td data-title="Total Winners"><span style="color:#F00">Rollover</span> 0</td>
  <td data-title="Prize Fund Amount">&pound;0</td></tr>
<tr class="alt"><td class="colour"><strong>Match 5 plus Bonus</strong></td>
  <td data-title="Prize">&pound;0</td>
  <td data-title="Round 1 Winners">0</td>
  <td data-title="Round 2 Winners">1</td>
  <td data-title="Total Winners">1</td>
  <td data-title="Prize Fund Amount">&pound;1,000,000</td></tr>
<tr><td class="colour"><strong>Match 5</strong></td>
  <td data-title="Prize">&pound;1,000</td>
  <td data-title="Round 1 Winners">44</td>
  <td data-title="Round 2 Winners">22</td>
  <td data-title="Total Winners">66</td>
  <td data-title="Prize Fund Amount">&pound;66,000</td></tr>
<tr><td class="colour"><strong>Match 4</strong></td>
  <td data-title="Prize">&pound;50</td>
  <td data-title="Round 1 Winners">3,506</td>
  <td data-title="Round 2 Winners">1,709</td>
  <td data-title="Total Winners">5,215</td>
  <td data-title="Prize Fund Amount">&pound;260,750</td></tr>
<tr><td class="colour"><strong>Match 3</strong></td>
  <td data-title="Prize">&pound;10</td>
  <td data-title="Round 1 Winners">73,375</td>
  <td data-title="Round 2 Winners">43,540</td>
  <td data-title="Total Winners">116,915</td>
  <td data-title="Prize Fund Amount">&pound;1,169,150</td></tr>
<tr><td class="colour"><strong>Match 2</strong></td>
  <td data-title="Prize">&pound;1</td>
  <td data-title="Round 1 Winners">593,248</td>
  <td data-title="Round 2 Winners">456,532</td>
  <td data-title="Total Winners">1,049,780</td>
  <td data-title="Prize Fund Amount">&pound;1,049,780</td></tr>
</tbody></table>
"""


def _by_round_tier(rows):
    return {(r["round"], r["tier"]): r for r in rows}


class TestSingleRound:
    def test_row_count(self):
        rows = parse_breakdown(SINGLE_ROUND_HTML, 2528, "2020-03-14")
        assert len(rows) == 6  # 6 tiers, one round
        assert all(r["round"] == 1 for r in rows)

    def test_winner_counts(self):
        rows = _by_round_tier(parse_breakdown(SINGLE_ROUND_HTML, 2528, "2020-03-14"))
        assert rows[(1, 1)]["winners"] == 1          # Match 6
        assert rows[(1, 2)]["winners"] == 6          # Match 5 + Bonus
        assert rows[(1, 3)]["winners"] == 91         # Match 5
        assert rows[(1, 4)]["winners"] == 6_218      # Match 4
        assert rows[(1, 5)]["winners"] == 129_966    # Match 3
        assert rows[(1, 6)]["winners"] == 1_135_143  # Match 2

    def test_prize_amounts(self):
        rows = _by_round_tier(parse_breakdown(SINGLE_ROUND_HTML, 2528, "2020-03-14"))
        assert rows[(1, 2)]["prize_per_winner"] == 1_000_000.0
        assert rows[(1, 6)]["prize_per_winner"] == 0.0  # "Free Lucky Dip" -> 0


class TestTwoRound:
    def test_row_count(self):
        rows = parse_breakdown(TWO_ROUND_HTML, 3191, "2026-07-22")
        assert len(rows) == 12  # 6 tiers x 2 rounds
        assert {r["round"] for r in rows} == {1, 2}

    def test_winner_counts_match_official_feed(self):
        rows = _by_round_tier(parse_breakdown(TWO_ROUND_HTML, 3191, "2026-07-22"))
        # Round 1
        assert rows[(1, 3)]["winners"] == 44
        assert rows[(1, 4)]["winners"] == 3_506
        assert rows[(1, 5)]["winners"] == 73_375
        assert rows[(1, 6)]["winners"] == 593_248
        # Round 2 - note the lone Match 5+Bonus winner
        assert rows[(2, 2)]["winners"] == 1
        assert rows[(2, 3)]["winners"] == 22
        assert rows[(2, 6)]["winners"] == 456_532

    def test_rollover_text_parses_to_zero(self):
        rows = _by_round_tier(parse_breakdown(TWO_ROUND_HTML, 3191, "2026-07-22"))
        assert rows[(1, 1)]["winners"] == 0  # "Rollover 0" in Total, 0 per round


class TestGuards:
    def test_missing_table_raises(self):
        with pytest.raises(ValueError):
            parse_breakdown("<html><body>no results here</body></html>", 1, "2020-01-01")

    def test_implausible_parse_raises(self):
        tiny = """
        <a id="prizeBreakdown"></a><table class="lotto"><tbody>
        <tr><td><strong>Match 3</strong></td><td data-title="Winners">5</td></tr>
        </tbody></table>
        """
        with pytest.raises(ValueError):
            parse_breakdown(tiny, 1, "2020-01-01")
