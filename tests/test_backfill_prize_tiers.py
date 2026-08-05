"""Tests for the lottery.co.uk prize-tier backfill parser.

Fixtures mirror the real page markup for both layouts: single-round (pre-2026)
and two-round (since 2026-06). The winner numbers below are the actual values
for draws 2528 and 3191 - the latter cross-checked against the official feed.
"""

import pytest

from scripts.backfill_prize_tiers import _to_money, parse_breakdown

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


# --- roll-down draw (3190, 2026-07-18) --------------------------------------
# The one Must-Be-Won draw the project holds live data for. Winner counts and
# paid amounts below are cross-checked against data/prize_tiers.csv, collected
# independently from the official XML feed: Match 3 paid 1,976,400 / 82,350 =
# £24 and Match 2 paid 4,151,685 / 830,337 = £5. The page renders the base and
# the boosted amount in ONE cell, which is what the old digits-only parser
# glued into 1024 and 15.
ROLLDOWN_HTML = """
<a id="prizeBreakdown"></a>
<h2>Prize Breakdown</h2>
<table class="table lotto mobFormat">
<thead><tr>
  <th>Category</th><th>Prize</th><th>Round 1 Winners</th>
  <th>Round 2 Winners</th><th>Total Winners</th><th>Prize Fund</th>
</tr></thead>
<tbody>
<tr><td class="colour"><strong>Match 6</strong></td>
  <td data-title="Prize">&pound;9,559,451</td>
  <td data-title="Round 1 Winners">0</td>
  <td data-title="Round 2 Winners">0</td>
  <td data-title="Total Winners">0</td>
  <td data-title="Prize Fund Amount">&pound;0</td></tr>
<tr class="alt"><td class="colour"><strong>Match 5 plus Bonus</strong></td>
  <td data-title="Prize">&pound;1,000,000<br /><span class="rolldown">Rolldown Prize: &pound;0</span></td>
  <td data-title="Round 1 Winners">0</td>
  <td data-title="Round 2 Winners">0</td>
  <td data-title="Total Winners">0</td>
  <td data-title="Prize Fund Amount">&pound;0</td></tr>
<tr><td class="colour"><strong>Match 5</strong></td>
  <td data-title="Prize">&pound;1,000</td>
  <td data-title="Round 1 Winners">48</td>
  <td data-title="Round 2 Winners">39</td>
  <td data-title="Total Winners">87</td>
  <td data-title="Prize Fund Amount">&pound;87,000</td></tr>
<tr><td class="colour"><strong>Match 4</strong></td>
  <td data-title="Prize">&pound;50</td>
  <td data-title="Round 1 Winners">3,421</td>
  <td data-title="Round 2 Winners">3,511</td>
  <td data-title="Total Winners">6,932</td>
  <td data-title="Prize Fund Amount">&pound;346,600</td></tr>
<tr><td class="colour"><strong>Match 3</strong></td>
  <td data-title="Prize">&pound;10<br /><span class="rolldown">Rolldown Prize: &pound;24</span></td>
  <td data-title="Round 1 Winners">82,350</td>
  <td data-title="Round 2 Winners">87,088</td>
  <td data-title="Total Winners">169,438</td>
  <td data-title="Prize Fund Amount">&pound;4,066,512</td></tr>
<tr><td class="colour"><strong>Match 2</strong></td>
  <td data-title="Prize">&pound;1<br /><span class="rolldown">Rolldown Prize: &pound;5</span></td>
  <td data-title="Round 1 Winners">830,337</td>
  <td data-title="Round 2 Winners">926,053</td>
  <td data-title="Total Winners">1,756,390</td>
  <td data-title="Prize Fund Amount">&pound;8,781,950</td></tr>
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


class TestRollDown:
    """Regression cover for the glued-amounts bug.

    A roll-down cell carries two amounts. Reading digits only turned
    "£10 Rolldown Prize: £24" into 1024 and "£1 ... £5" into 15, corrupting
    prize_per_winner for all 93 roll-down draws in the archive - i.e. exactly
    the draws this project exists to play.
    """

    def test_pays_the_boosted_amount_not_the_glued_digits(self):
        rows = _by_round_tier(parse_breakdown(ROLLDOWN_HTML, 3190, "2026-07-18"))
        for rnd in (1, 2):
            assert rows[(rnd, 5)]["prize_per_winner"] == 24.0    # was 1024.0
            assert rows[(rnd, 6)]["prize_per_winner"] == 5.0     # was 15.0

    def test_matches_the_official_feed(self):
        # data/prize_tiers.csv, collected independently from the XML feed:
        # Match 3 round 1 paid 1,976,400 / 82,350 = £24.00 exactly.
        rows = _by_round_tier(parse_breakdown(ROLLDOWN_HTML, 3190, "2026-07-18"))
        assert rows[(1, 5)]["winners"] == 82_350
        assert rows[(1, 5)]["prize_per_winner"] * rows[(1, 5)]["winners"] == 1_976_400
        assert rows[(1, 6)]["prize_per_winner"] * rows[(1, 6)]["winners"] == 4_151_685

    def test_untouched_tiers_keep_their_base_prize(self):
        # A roll-down lifts Match 3 and Match 2 only; the others render
        # "Rolldown Prize: £0" and must fall back to the base, not to zero.
        rows = _by_round_tier(parse_breakdown(ROLLDOWN_HTML, 3190, "2026-07-18"))
        assert rows[(1, 2)]["prize_per_winner"] == 1_000_000.0
        assert rows[(1, 3)]["prize_per_winner"] == 1_000.0
        assert rows[(1, 4)]["prize_per_winner"] == 50.0

    def test_must_be_won_pool_survives(self):
        # Tier 1 with 0 winners carries the pool that rolled down - the number
        # the EV model needs to judge whether a Must-Be-Won draw is worth playing.
        rows = _by_round_tier(parse_breakdown(ROLLDOWN_HTML, 3190, "2026-07-18"))
        assert rows[(1, 1)]["winners"] == 0
        assert rows[(1, 1)]["prize_per_winner"] == 9_559_451.0


class TestToMoney:
    def test_single_amount(self):
        assert _to_money("&pound;1,750") == 1750.0
        assert _to_money("&pound;4,037,783") == 4037783.0

    def test_two_amounts_takes_the_larger(self):
        assert _to_money("&pound;10 Rolldown Prize: &pound;24") == 24.0
        assert _to_money("&pound;1,000,000 Rolldown Prize: &pound;0") == 1_000_000.0

    def test_no_amount(self):
        assert _to_money("Free Lucky Dip") == 0.0
        assert _to_money("") == 0.0
        assert _to_money(None) == 0.0

    def test_numeric_entity_does_not_donate_digits(self):
        # &#163; is the numeric entity for £; its digits must not become an amount.
        assert _to_money("&#163;30") == 30.0


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
