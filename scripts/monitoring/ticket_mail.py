#!/usr/bin/env python3
"""Lines on demand, by email: the cloud version of `./lotto ticket`.

Run by .github/workflows/ticket.yml when someone presses "Run workflow" -
typically from the GitHub app, away from the Mac, deciding to play for fun.
It builds the same lines `./lotto ticket` / `./lotto wheel` would (same
advise(), same wheel_portfolio()) on the data the collector has committed,
and mails them with the verdict for that draw, stated plainly.

What the history is used for, and what it is not: every 6-of-59 line has the
same chance to be drawn, and the project's own fairness tests and backtest
found nothing to predict. The archive decides WHICH lines instead - the ones
other players pick least, so that a win is shared with fewer of them.

Writes nothing: outputs/ is not committed, and the ledger lives on the Mac,
so the mail ends with the command that records the lines there.

    TICKET_KIND=portfolio|wheel  TICKET_LINES=5  TICKET_VARIANT=0
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lottery.ev import (  # noqa: E402
    MARGINAL,
    PLAY,
    default_portfolio_seed,
    expected_cowinner_share,
    line_ev,
    popularity_ratio,
)
from scripts.archive import load_tier_archive  # noqa: E402
from scripts.ev_play import advise  # noqa: E402
from scripts.monitoring import pre_ev_gate  # noqa: E402
from scripts.monitoring.notify import maybe_send_email  # noqa: E402
from scripts.wheel_play import wheel_portfolio  # noqa: E402

# The line the site and `./lotto check` use for "what most people play":
# dates, all under 32, evenly spread.
TYPICAL_LINE = [3, 7, 12, 19, 24, 31]
ODDS = 45_057_474


def build_ticket_mail(kind: str = "portfolio", lines: int = 5, variant: int = 0,
                      gate: list | None = None, archive: tuple | None = None) -> tuple:
    """(subject, body, tickets). `variant` shifts the seed: 0 is the same set
    `./lotto ticket` shows for this draw; 1, 2, ... are other, equally good
    sets, each reproducible."""
    seed = None
    a = advise(lines=lines, force=True)
    cond, verdict = a.cond, a.verdict
    if kind == "wheel":
        tickets = wheel_portfolio(12, None, cond=cond).tickets
    else:
        if variant:
            seed = (default_portfolio_seed(cond.draw_date) or 0) + variant
            a = advise(lines=lines, force=True, seed=seed)
        tickets = [p["line"] for p in a.portfolio]

    when = cond.draw_date.strftime("%A %d %b %Y") if cond.draw_date else "the next draw"
    ev = verdict["ev_best_line"]
    label = (verdict.get("model_stability") or {}).get("label", a.advice)
    cost = len(tickets) * cond.ticket_price
    subject = (f"LOTTO ticket: {len(tickets)} lines for {when} - model says "
               f"{a.advice} (EV £{ev:+.2f} a line)")

    typical_keep = expected_cowinner_share(TYPICAL_LINE, cond.tickets_sold, cond.rounds)
    out = [f"UK Lotto - {len(tickets)} lines for {when}  "
           f"({len(tickets)} x £{cond.ticket_price:.0f} = £{cost:.2f})", ""]
    for i, t in enumerate(tickets, 1):
        keep = expected_cowinner_share(t, cond.tickets_sold, cond.rounds)
        out.append(f"  {i}.  {' '.join(f'{n:2d}' for n in t)}    "
                   f"popularity x{popularity_ratio(t):.2f}   keeps {100 * keep:.0f}% "
                   f"of a jackpot   EV £{line_ev(t, cond):+.3f}")
    out.append("")

    out.append(f"Model verdict for this draw: {a.advice} ({label}) - "
               f"EV £{ev:+.2f} a £2 line")
    if a.advice == PLAY:
        out.append("  This draw clears break-even: on average a line returns more "
                   "than it costs.")
    elif a.advice == MARGINAL:
        out.append("  Only at the sales measured since June does this draw clear "
                   "break-even - a thin edge, not a verdict.")
    else:
        out.append(f"  Entertainment, not investment: on average £2 comes back as "
                   f"£{2 + ev:.2f}.")
    out.append("")

    out += [
        "Why these numbers",
        f"  Every line has the same chance: 1 in {ODDS:,} per round, two rounds",
        "  per ticket. Nothing in the draw history predicts the balls - the",
        "  project's fairness tests and backtest found no pattern to use.",
        "  What history does tell us is which numbers other players pick (dates,",
        "  1-31, patterns). These lines avoid them, so a win is shared with fewer",
        f"  people: a typical birthday line ({' '.join(map(str, TYPICAL_LINE))}) keeps "
        f"{100 * typical_keep:.0f}%",
        "  of a jackpot on average.",
    ]
    if kind == "wheel":
        out.append("  Wheel: 6 tickets over the 12 least-played numbers - if 4 of the "
                   "6 drawn")
        out.append("  land in those 12, at least one ticket is guaranteed Match 3.")
    out.append("")

    draws = f"{archive[0]:,} draws ({archive[1]} .. {archive[2]})" if archive else "the archive"
    out.append(f"Built from: {draws}, the popularity model calibrated on their prize")
    out.append("  tiers, and ticket sales read exactly off the jackpot pools.")
    if a.stale:
        out.append(f"  WARNING: the {a.stale} draw is not collected yet - figures are one draw old.")
    if gate:
        out.append("  WARNING: the data gate reported: " + "; ".join(gate))
    if seed is not None:
        out.append(f"  Set #{variant} (seed {seed}) - run again with the same number for the same lines.")
    out.append("")

    record = "; ".join(" ".join(map(str, t)) for t in tickets)
    draw = cond.draw_date.isoformat() if cond.draw_date else "next"
    out += ["After buying, record them on the Mac:",
            f'  ./lotto ledger add --draw-date {draw} --lines "{record}"', ""]
    return subject, "\n".join(out), tickets


def _archive_span() -> tuple | None:
    try:
        t = load_tier_archive()
        return (t["draw_number"].nunique(), str(t["draw_date"].min()),
                str(t["draw_date"].max()))
    except Exception:
        return None


def main() -> int:
    kind = os.environ.get("TICKET_KIND", "portfolio")
    lines = int(os.environ.get("TICKET_LINES") or 5)
    variant = int(os.environ.get("TICKET_VARIANT") or 0)
    if kind not in ("portfolio", "wheel") or not 1 <= lines <= 10 or variant < 0:
        print(f"[ticket] bad input: kind={kind!r} lines={lines} variant={variant}")
        return 1
    subject, body, _ = build_ticket_mail(kind, lines, variant,
                                         gate=pre_ev_gate.run(), archive=_archive_span())
    print(subject)
    print(body)
    maybe_send_email(subject, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
