#!/usr/bin/env python3
"""EV-first play advisor: should you play the next draw, and with which lines?

Reads next-draw conditions (estimated jackpot, roll-down flag) from
data/prize_tiers.csv (accumulated from the official XML feed), decides
whether the draw clears the EV threshold, and builds a diversified
portfolio of unpopular lines.

Usage:
  python scripts/ev_play.py                     # advise for next draw
  python scripts/ev_play.py --lines 5 --seed 42
  python scripts/ev_play.py --jackpot 10000000 --roll-down   # what-if
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from lottery.ev import (  # noqa: E402
    DrawConditions,
    DEFAULT_TICKETS_SOLD,
    abrams_garibaldi_screen,
    calibrate_fixed_prizes,
    default_portfolio_seed,
    estimate_tickets_sold,
    forecast_must_be_won,
    guaranteed_pool,
    must_be_won_outlook,
    kelly_stake,
    last_closed_draw_date,
    mbw_type,
    mbw_uplift,
    should_play,
    upcoming_draw_date,
)
from lottery.portfolio import build_portfolio  # noqa: E402

PRIZE_TIERS_FILE = Path("data/prize_tiers.csv")
DRAW_POOLS_FILE = Path("data/draw_pools.csv")
OUT_DIR = Path("outputs/predictions")


def uncollected_draw(now: datetime | date | None = None,
                     tiers_file: Path = PRIZE_TIERS_FILE) -> date | None:
    """The draw that has closed but is not in the data yet, or None.

    Everything `next_draw_conditions` reads - the jackpot estimate, the
    roll-down flag, the rollover counter - describes the draw AFTER the last
    one collected. Once sales close on a draw day and before the collector
    runs, that is the draw that has just happened, not the one a ticket would
    enter, and the verdict and the Must-Be-Won forecast are one draw stale.
    """
    if not tiers_file.exists():
        return None
    tiers = pd.read_csv(tiers_file)
    if not len(tiers):
        return None
    latest = pd.to_datetime(tiers["draw_date"]).max().date()
    closed = last_closed_draw_date(now)
    return closed if latest < closed else None


def next_draw_conditions(force_roll_down: bool = False,
                         force_ordinary: bool = False,
                         now: datetime | date | None = None) -> DrawConditions:
    """Best known conditions for the upcoming draw, from collected data.

    The roll-down overrides belong here rather than in the caller because the
    sales estimate DEPENDS on the flag - a Must-Be-Won draw sells ~1.27x
    (Sat) / 1.44x (Wed) an ordinary one. Setting `cond.roll_down` after the
    fact would leave a what-if run priced at the wrong sales level, which is
    the same mistake that made the 2026-08-08 alert read +EV.

    `force_ordinary` is the mirror of `force_roll_down`: while the live feed
    flags a roll-down, every what-if inherits it, so "what would an ordinary
    draw at this jackpot be worth?" was unaskable.

    `now` pins the moment the conditions are priced at. Callers that must be
    reproducible pass one derived from the data rather than the clock - the
    site exporter does, because its output is committed and diffed in CI, and
    a wall-clock read would rewrite the file every time the draw date rolls.
    """
    cond = DrawConditions()
    # The draw being priced is the one a ticket bought NOW would enter; its
    # weekday selects the sales baseline and Must-Be-Won uplift (Saturdays
    # sell ~1.59x Wednesdays).
    cond.draw_date = upcoming_draw_date(now)
    if PRIZE_TIERS_FILE.exists():
        tiers = pd.read_csv(PRIZE_TIERS_FILE)
        if len(tiers):
            last = tiers.sort_values("draw_number").iloc[-1]
            if pd.notna(last.get("next_jackpot_estimate")):
                cond.jackpot = float(last["next_jackpot_estimate"])
            # bool(NaN) is True - a missing flag must never fake a Must-Be-Won
            flag = last.get("next_jackpot_roll_down")
            flagged = pd.notna(flag) and str(flag).strip().lower() in (
                "true", "y", "yes", "1")
            cond.roll_down = not force_ordinary and (force_roll_down or flagged)
            if pd.notna(last.get("rollover_count")):
                cond.rollover_count = int(last["rollover_count"])
            # A flagged draw the cap did not force is the operator's own - a
            # guaranteed pool with a campaign behind it, which sells like one
            # (SPECIAL_MBW_UPLIFT_BY_WEEKDAY). So is a capped roll the operator
            # topped up to a round guarantee. Only the live flag can say so:
            # a forced what-if asks about the cap.
            cond.special_event = bool(
                cond.roll_down and flagged and not force_roll_down
                and (mbw_type(True, cond.rollover_count) == "special-event"
                     or guaranteed_pool(cond.jackpot)))
            # Sales are an identity, not an estimate, wherever the pools
            # reach: (pool - previous pool) / 8.88%. Winner counts stay the
            # fallback for windows the pools do not cover.
            pools = (pd.read_csv(DRAW_POOLS_FILE)
                     if DRAW_POOLS_FILE.exists() else None)
            estimated = estimate_tickets_sold(tiers, roll_down=cond.roll_down,
                                              draw_date=cond.draw_date,
                                              pools_df=pools,
                                              special_event=cond.special_event)
            if estimated:
                cond.tickets_sold = estimated
            # Fixed-tier prizes come from the data too - a hardcoded table is
            # how the Match 3/2 roll-down prizes once leaked into every draw.
            cond.prizes = calibrate_fixed_prizes(tiers)
    return cond


@dataclass(frozen=True)
class Advice:
    """Everything the advisor decided about one draw, computed once.

    `render` turns it into the printout, `save` into latest.json, and any
    other front end (the email, the terminal) reads the same record - so none
    of them can price or classify the draw a second, slightly different way.
    """
    cond: DrawConditions
    verdict: dict
    threshold: float
    bankroll: float
    what_if: bool
    force: bool
    stale: date | None
    rollover: dict
    outlook: dict | None
    abrams_garibaldi: dict | None
    kelly: dict | None
    portfolio: list = field(default_factory=list)


def advise(*, lines: int = 5, jackpot: float | None = None,
           roll_down: bool = False, ordinary: bool = False,
           tickets: int | None = None, threshold: float = 0.0,
           bankroll: float = 1000.0, seed: int | None = None,
           force: bool = False, now: datetime | date | None = None) -> Advice:
    """Price the next draw and, when it clears the threshold or `force` is
    set, build its portfolio. Reads collected data; writes nothing."""
    cond = next_draw_conditions(force_roll_down=roll_down,
                                force_ordinary=ordinary, now=now)
    what_if = (jackpot is not None or roll_down or ordinary
               or tickets is not None)
    if jackpot is not None:
        cond.jackpot = jackpot
    if tickets is not None:
        cond.tickets_sold = tickets

    verdict = should_play(cond, threshold=threshold)
    rollover = forecast_must_be_won(cond.rollover_count, now)
    outlook = None
    if not cond.roll_down:
        pools = (pd.read_csv(DRAW_POOLS_FILE)
                 if DRAW_POOLS_FILE.exists() else None)
        outlook = must_be_won_outlook(cond, pools, now)

    portfolio = []
    if verdict["play"] or force:
        # Same default seed as the alert email: latest.json (what the ledger
        # records via --from-latest) and the mail must propose the SAME lines.
        chosen = seed if seed is not None else default_portfolio_seed(cond.draw_date)
        portfolio = build_portfolio(lines, cond, seed=chosen)

    return Advice(
        cond=cond, verdict=verdict, threshold=threshold, bankroll=bankroll,
        what_if=what_if, force=force, stale=uncollected_draw(now),
        rollover=rollover, outlook=outlook,
        abrams_garibaldi=abrams_garibaldi_screen(cond),
        kelly=kelly_stake(cond, bankroll) if verdict["play"] else None,
        portfolio=portfolio,
    )


def render(a: Advice) -> str:
    """The advisor's printout for `a`, exactly as `make play` shows it."""
    cond, verdict = a.cond, a.verdict
    out: list[str] = []
    say = out.append

    say("=" * 64)
    say("EV ADVISOR - next UK Lotto draw")
    say("=" * 64)
    if a.stale:
        say(f"WARNING: the {a.stale} draw has closed but is not collected yet.")
        say("  Every figure below - jackpot, Must-Be-Won flag, rollover count -")
        say("  still describes that draw, not the next one. Re-run after the")
        say("  collector (scripts/monitoring/sync_collector_data.sh).")
        say("-" * 64)
    say(f"Jackpot (event pool): £{cond.jackpot:,.0f}")
    say(f"Rounds per ticket:    {cond.rounds}")
    # From the conditions, not the count: a forced what-if has no count and
    # is priced as a capped roll, so it must not be labelled a special.
    kind = (None if not cond.roll_down
            else "special-event" if cond.special_event else "cap-driven")
    say(f"Must-Be-Won:          {f'YES ({kind})' if kind else 'no'}")
    mbw = a.rollover
    if not cond.roll_down:
        say(f"Rollover:             {mbw['rollover_count']} of {mbw['cap']} - "
            f"Must-Be-Won in {mbw['draws_away']} draw(s), ~{mbw['expected_date']}, "
            f"if nobody wins before")
        # What that draw would be worth. Without it the verdict on the one
        # draw worth planning for only existed the morning after the draw
        # before it - a day's notice on a fortnight's wait.
        outlook = a.outlook
        if outlook and not outlook["is_next_draw"]:
            say(f"  that draw:          projected pool ~£{outlook['projected_pool']:,.0f} "
                f"vs break-even £{outlook['break_even_jackpot']:,.0f} - "
                f"{'PLAY' if outlook['play'] else 'likely SKIP'} "
                f"(EV £{outlook['ev_best_line']:+.2f}, forecast)")
    day = cond.draw_date.strftime("%A") if cond.draw_date else "unknown day"
    uplift_label = (f" ({day} {kind} uplift "
                    f"x{mbw_uplift(cond.draw_date, cond.special_event)[0]})"
                    if cond.roll_down else "")
    say(f"Assumed lines sold:   {cond.tickets_sold:,}{uplift_label}")
    p = cond.prizes
    say(f"Fixed prizes/round:   5+B £{p.match_5_bonus:,.0f} · 5 £{p.match_5:,.0f} · "
        f"4 £{p.match_4:,.0f} · 3 £{p.match_3:,.0f} · 2 £{p.match_2:,.0f}  [{p.source}]")
    say(f"Best-line EV:         £{verdict['ev_best_line']:+.3f}  (threshold £{a.threshold:+.2f})")
    say(f"Break-even jackpot:   £{verdict['break_even_jackpot']:,.0f}"
        f"{' (roll-down)' if cond.roll_down else ''}")
    # Whether the verdict rests on the popularity model being the right
    # SHAPE, not just well fitted. The weights are pinned to 0.2% on the
    # threshold, but a flat model and a doubled one disagree on ordinary
    # draws near break-even - see scripts/validations/popularity_audit.py.
    stab = verdict.get("model_stability")
    if stab:
        note = ("verdict holds from a flat popularity model to a doubled one"
                if stab["stable"] else
                "VERDICT DEPENDS ON THE POPULARITY MODEL'S SHAPE - "
                "flat and doubled disagree")
        say(f"Model stability:      {stab['label']} - {note}")
        say(f"  across specs:       £{stab['ev_spec_min']:+.3f} ... "
            f"£{stab['ev_spec_max']:+.3f}  (flat -> doubled popularity)")
    ag = a.abrams_garibaldi
    if ag:
        # Second opinion for ordinary draws (Abrams & Garibaldi 2010). Their
        # cutoffs are sufficient conditions robust to ANY sales level, so
        # passing is much rarer than our exact break-even.
        status = ("robust +EV even if sales surge" if ag["robust_good_bet"]
                  else "any edge would rest on the sales estimate")
        say(f"A&G second opinion:   entries/jackpot {ag['n_over_j']:.2f} "
            f"(<0.2 wanted), robust cutoff £{ag['jackpot_cutoff'] / 1e6:,.0f}M "
            f"- {status}")
    # On a roll-down the verdict lives or dies on the sales estimate, so show
    # what it does across the plausible range instead of one tidy number.
    sens = verdict.get("sales_sensitivity")
    if sens:
        say(f"Across sales range:   £{sens['ev_low']:+.3f} at {sens['tickets_high']:,} lines "
            f"... £{sens['ev_high']:+.3f} at {sens['tickets_low']:,} (quartiles)")
        holds = ("YES" if sens["robust"]
                 else "NO - only the central sales estimate clears it")
        say(f"Holds across range:   {holds}")
    k = a.kelly
    if k:
        if k["lines_full"] >= 1:
            say(f"Kelly stake:          {k['lines_full']} lines full / "
                f"{k['lines_half']} half-Kelly on a £{a.bankroll:,.0f} bankroll")
        else:
            # The honest MacLean-Ziemba answer: a real edge that is 81% to
            # lose a given line justifies almost nothing growth-theoretically.
            say(f"Kelly stake:          £{k['stake_full']:.2f} on a "
                f"£{a.bankroll:,.0f} bankroll (f*={k['kelly_fraction']:.2e}) - "
                f"the edge is real, but at this bankroll every line is an "
                f"entertainment stake, not growth")
    say("-" * 64)

    if not verdict["play"] and not a.force:
        say("VERDICT: SKIP this draw - expected loss per £2 line is above")
        say("your threshold. Playing anyway is entertainment, not investment.")
        say("(Use --force to build a portfolio regardless.)")
        say("=" * 64)
    else:
        if verdict["play"]:
            say("VERDICT: conditions clear your threshold - if you play, play these:")
        else:
            say("VERDICT: below threshold (forced portfolio):")
        total_ev = sum(p["ev"] for p in a.portfolio)
        for i, p in enumerate(a.portfolio, 1):
            nums = " ".join(f"{n:2d}" for n in p["line"])
            say(f"  {i}. {nums}   EV £{p['ev']:+.3f}   popularity x{p['popularity_ratio']:.2f}")
        say("-" * 64)
        say(f"Portfolio: {len(a.portfolio)} lines, cost £{len(a.portfolio) * cond.ticket_price:.2f}, "
            f"total EV £{total_ev:+.2f}")
        say("=" * 64)
    return "\n".join(out)


def save(a: Advice, out_dir: Path = OUT_DIR) -> str:
    """Persist the real verdict to latest.json (and the portfolio, if any).

    A what-if run (--jackpot / --roll-down / --tickets) must NOT touch
    latest.json: that file is what `roi_ledger.py add --from-latest` records
    as really played and what the dashboard shows as the live verdict.
    Exploring "what if the jackpot were £12M" once left a PLAY portfolio
    sitting there for a draw that is actually a SKIP.

    Returns the line to print.
    """
    if a.what_if:
        return "(what-if run - latest.json left untouched)"

    # Always persist the real verdict - the dashboard reads latest.json, and a
    # SKIP with no file would tell the user to re-run make play forever.
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "predictions": [p["line"] for p in a.portfolio],
        "metadata": {
            "method": "ev_portfolio",
            "verdict": a.verdict,
            "per_line": [{"line": p["line"], "ev": p["ev"],
                          "popularity_ratio": p["popularity_ratio"]} for p in a.portfolio],
        },
    }
    with open(out_dir / "latest.json", "w") as f:
        json.dump(payload, f, indent=2)
    if a.portfolio:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(out_dir / f"ev_portfolio_{ts}.json", "w") as f:
            json.dump(payload, f, indent=2)
        return f"Saved to {out_dir}/ev_portfolio_{ts}.json (+ latest.json for roi_ledger)"
    return f"Verdict saved to {out_dir}/latest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=5, help="Portfolio size")
    parser.add_argument("--jackpot", type=float, default=None,
                        help="Override jackpot (event pool, shared across rounds)")
    parser.add_argument("--roll-down", action="store_true", help="Force Must-Be-Won roll-down")
    parser.add_argument("--ordinary", action="store_true",
                        help="Force an ordinary (non-roll-down) draw in what-if runs")
    parser.add_argument("--tickets", type=int, default=None,
                        help=f"Assumed lines sold per draw (default: estimated from "
                             f"prize_tiers.csv, else {DEFAULT_TICKETS_SOLD:,})")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum EV (GBP) per line to recommend playing")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Bankroll (GBP) the Kelly stake is sized against")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Build a portfolio even when the draw is below threshold")
    args = parser.parse_args()

    advice = advise(lines=args.lines, jackpot=args.jackpot, roll_down=args.roll_down,
                    ordinary=args.ordinary, tickets=args.tickets,
                    threshold=args.threshold, bankroll=args.bankroll,
                    seed=args.seed, force=args.force)
    print(render(advice))
    print(save(advice))


if __name__ == "__main__":
    main()
