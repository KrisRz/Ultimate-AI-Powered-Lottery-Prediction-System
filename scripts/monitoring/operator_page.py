"""The operator's own Lotto page, read as a second opinion before a PLAY email.

Everything the advisor knows about the NEXT draw - its jackpot and whether it
must be won - comes from one legacy XML document, and that document has
already been caught serving a half-filled block in the minutes after a draw
(see `_forward_fields_from_xml`). A PLAY email is the one output that spends
money, so it should not rest on a single feed.

The CMS behind national-lottery.co.uk carries the same facts independently:

    headerArea[0].dbgJackpotCard.assignedGame = {
        "phase": "MUST_BE_WON" | "INITIAL" | ...,
        "jackpot": <pence>,
        "sellEndDate": "2026-09-16T18:30:00Z",
        "gamePrice": 200,
    }

`phase` read MUST_BE_WON before both 3205 (a capped roll) and 3206 (a
guaranteed GBP 12m), and a promotional one appears about two and a half days
ahead - checked in the history github.com/eagerterrier/pwa-badge commits of
this endpoint. It also gives the one thing the XML never did: when sales
close, which is what a phone reader needs next.

This module only reads and compares. It never blocks an email: a disagreement
goes into the subject, and an unreachable page is said to be unreachable.
"""

from __future__ import annotations

from datetime import datetime

CMS_LOTTO_URL = ("https://api-dfe.national-lottery.co.uk/cms-proxy/pages/v1/"
                 "games/lotto?mobileclient=false")
MUST_BE_WON_PHASE = "MUST_BE_WON"

# The CMS rounds a Must-Be-Won jackpot down to the pound, and its rolling
# estimate is published separately from the XML's; half a percent is far
# below any difference that could move a verdict and far above rounding.
JACKPOT_TOLERANCE = 0.005


def find_assigned_game(page) -> dict | None:
    """The first `assignedGame` block that carries a phase, wherever the page
    layout happens to put it this week."""
    stack = [page]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            game = node.get("assignedGame")
            if isinstance(game, dict) and "phase" in game:
                return game
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return None


def parse_assigned_game(game: dict | None) -> dict:
    """{phase, must_be_won, jackpot (GBP), sales_close (datetime)} or {}."""
    if not game or not game.get("phase"):
        return {}
    jackpot = game.get("jackpot")
    close = game.get("sellEndDate")
    try:
        sales_close = (datetime.fromisoformat(close.replace("Z", "+00:00"))
                       if close else None)
    except ValueError:
        sales_close = None
    return {
        "phase": str(game["phase"]),
        "must_be_won": str(game["phase"]) == MUST_BE_WON_PHASE,
        "jackpot": float(jackpot) / 100 if jackpot is not None else None,
        "sales_close": sales_close,
    }


def fetch_operator_page(timeout: float = 15.0) -> dict:
    """Live read of the operator's page; {} on any failure."""
    try:
        import requests
        resp = requests.get(CMS_LOTTO_URL, timeout=timeout,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return parse_assigned_game(find_assigned_game(resp.json()))
    except Exception as exc:          # a second opinion must never raise
        print(f"[operator-page] unavailable: {exc}")
        return {}


def compare(roll_down: bool, jackpot: float, page: dict) -> tuple:
    """(agrees, line for the email). `agrees` is None when the page could
    not be read - unknown, not a disagreement."""
    if not page:
        return None, ("Operator's page:      unreachable - verdict rests on the "
                      "XML feed alone")
    close = page.get("sales_close")
    closes = f"sales close {close:%a %d %b %H:%M} UTC" if close else ""
    shown = page.get("jackpot")
    amount = f"£{shown:,.0f}" if shown is not None else "no jackpot shown"
    problems = []
    if page["must_be_won"] != bool(roll_down):
        problems.append(f"phase {page['phase']} but the feed says "
                        f"{'Must-Be-Won' if roll_down else 'an ordinary draw'}")
    if shown is None or abs(shown - jackpot) > JACKPOT_TOLERANCE * max(jackpot, 1):
        problems.append(f"jackpot {amount} but the feed says £{jackpot:,.0f}")
    if problems:
        return False, ("Operator's page:      DISAGREES - " + "; ".join(problems)
                       + (f". The page says {closes}" if closes else "")
                       + ". Check national-lottery.co.uk before buying.")
    return True, ("Operator's page:      agrees (" + ", ".join(
        x for x in (page["phase"], amount, closes) if x) + ")")


if __name__ == "__main__":
    print(fetch_operator_page())
