#!/usr/bin/env python3
"""UK Lotto terminal: the decision and the ticket, in one place.

A thin front end. Every number comes from the functions `make play`,
`make wheel`, the email and the ledger already use - advise(), render(),
save(), wheel_portfolio(), roi_ledger - so this file prices nothing and
classifies nothing itself. If a figure here ever disagrees with `make play`,
that is a bug in this file.

    ./lotto                      interactive menu
    ./lotto status               the next draw, the data, the decision
    ./lotto play                 exactly `make play`
    ./lotto ticket [--yes]       5 unpopular lines, whatever the verdict
    ./lotto wheel [--yes]        the 6-ticket wheel
    ./lotto check 3 7 12 19 24 31   how your own numbers share a jackpot
    ./lotto whatif --jackpot 9000000 --roll-down
    ./lotto ledger [report|settle|add]
    ./lotto analysis [name]      read-only reports (uplift, rolldowns, ...)
    ./lotto sync                 pull the collector's data (main, clean tree)

A SKIP never blocks a ticket: interactively it asks "Generate anyway?", and
without a terminal it needs --yes, so a script cannot buy into a SKIP by
accident. The advice on file travels with the ticket into the ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lottery.ev import (  # noqa: E402
    MARGINAL,
    PLAY,
    SKIP,
    expected_cowinner_share,
    line_ev,
    popularity_ratio,
)
from lottery.portfolio import build_portfolio  # noqa: E402
from scripts import roi_ledger  # noqa: E402
from scripts.ev_play import (  # noqa: E402
    OUT_DIR,
    PRIZE_TIERS_FILE,
    Advice,
    advise,
    render,
    save,
)
from scripts.monitoring import pre_ev_gate  # noqa: E402
from scripts.wheel_play import render_wheel, save_wheel, wheel_portfolio  # noqa: E402

RULE = "─" * 44

# Read-only reports. Each is the command its make target runs; none of them
# can move today's verdict. Deliberately absent: the frozen popularity-v2
# evaluator (run once, never again), anything that fetches or writes the
# collector's files, and the legacy predictor.
ANALYSES = {
    "uplift": ("scripts/calibrate_mbw_uplift.py", [], "seconds",
               "Must-Be-Won sales uplift, installed vs measured since June"),
    "rolldowns": ("scripts/rolldown_history.py", [], "seconds",
                  "every roll-down since 2019, re-priced under today's rules"),
    "wheel-backtest": ("scripts/backtest_wheel.py", [], "seconds",
                       "the 6-ticket wheel played on every draw since 2015"),
    "popularity-audit": ("scripts/validations/popularity_audit.py", [], "~1 minute",
                         "does the popularity model hold up, and how much it matters"),
    "contract": ("scripts/data_contract.py", [], "seconds",
                 "data contract: KNOWN / UNKNOWN_BUT_VALID / INVALID"),
    "fairness": ("scripts/validations/fairness.py", ["--sims", "2000"], "minutes",
                 "are the machines fair? six tests"),
    "ensemble": ("scripts/validations/ensemble_score.py",
                 ["--candidates", "20000", "--step", "8", "--repeat", "10"], "minutes",
                 "does any number-picking method beat random?"),
    "ensemble-null": ("scripts/validations/ensemble_score.py",
                      ["--candidates", "20000", "--step", "8", "--null-sims", "40"],
                      "many minutes", "the same, against fair synthetic histories"),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _ask(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state() -> dict:
    """Branch, uncommitted edits, and distance from origin/main as of the
    last fetch. Reads refs only - no network, nothing written."""
    behind = _git("rev-list", "--count", "HEAD..origin/main")
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "behind": int(behind) if behind and behind.isdigit() else None,
    }


def latest_collected() -> tuple:
    """(draw number, date) of the newest draw in the collected tiers."""
    if not PRIZE_TIERS_FILE.exists():
        return None, None
    tiers = pd.read_csv(PRIZE_TIERS_FILE)
    if not len(tiers):
        return None, None
    last = tiers.sort_values("draw_number").iloc[-1]
    return int(last["draw_number"]), str(last["draw_date"])


def _gbp(x: float) -> str:
    return f"£{x:,.0f}"


# --------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------

def status_text(a: Advice, gate: list, git: dict) -> str:
    cond, v = a.cond, a.verdict
    draw_no, draw_day = latest_collected()
    lines = ["UK LOTTO TERMINAL", RULE]
    when = cond.draw_date.strftime("%A %d %b %Y") if cond.draw_date else "unknown"
    lines.append(f"Next draw:    {when}")
    mbw = "  MUST-BE-WON" if cond.roll_down else ""
    lines.append(f"Jackpot:      {_gbp(cond.jackpot)}{mbw}")
    if a.stale:
        lines.append(f"Data:         ⚠ the {a.stale} draw is not collected yet - "
                     f"run `./lotto sync`")
    else:
        lines.append(f"Data:         ✓ collected through {draw_day} (draw {draw_no})")
    if git.get("branch"):
        where = f"{git['branch']} @ {git['sha']}"
        if git["dirty"]:
            where += ", uncommitted edits"
        if git["behind"]:
            where += f", {git['behind']} commit(s) behind origin/main (last fetch)"
        lines.append(f"Code:         {where}")
    lines.append("Gate:         " + ("✓ inputs present and consistent" if not gate
                                     else "✗ " + "; ".join(gate)))
    stab = v.get("model_stability") or {}
    lines.append(f"Model:        3-bucket popularity (installed) · "
                 f"{stab.get('label', 'n/a')}")
    lines.append("")
    lines.append(f"Lines sold:   {cond.tickets_sold:,} (estimated)")
    lines.append(f"EV / line:    £{v['ev_best_line']:+.3f}   "
                 f"break-even {_gbp(v['break_even_jackpot'])}")
    if a.measured:
        m = a.measured
        lines.append(f"At measured:  £{m['verdict']['ev_best_line']:+.3f} "
                     f"(uplift x{m['uplift']:.3f}, installed x{m['installed']:.2f})")
    o = a.outlook
    if o and not a.stale and not o["is_next_draw"]:
        extra = ""
        if a.outlook_measured:
            extra = f", measured £{a.outlook_measured['verdict']['ev_best_line']:+.2f}"
        lines.append(f"Must-Be-Won:  ~{o['expected_date']} ({o['draws_away']} draws, if "
                     f"nobody wins first): pool ~{_gbp(o['projected_pool'])} vs "
                     f"{_gbp(o['break_even_jackpot'])} - EV £{o['ev_best_line']:+.2f}{extra}")
    elif a.stale and not cond.roll_down:
        lines.append("Must-Be-Won:  unknown until the last draw is collected")
    for note in a.notes:
        lines.append(f"Note:         {note}")
    lines.append("")
    decision = "BLOCKED - fix the inputs above first" if gate else a.advice
    lines.append(f"DECISION:     {decision}")
    return "\n".join(lines)


def cmd_status(args) -> int:
    a = advise()
    print(status_text(a, pre_ev_gate.run(), git_state()))
    return 0


def cmd_play(args) -> int:
    a = advise(lines=args.lines, threshold=args.threshold, bankroll=args.bankroll)
    print(render(a))
    print(save(a))
    return 0


# --------------------------------------------------------------------------
# generating: the override, then the ledger
# --------------------------------------------------------------------------

def confirm_override(a: Advice, yes: bool) -> bool:
    """True to go ahead. SKIP asks; without a terminal it needs --yes."""
    if a.advice in (PLAY, MARGINAL):
        return True
    stab = (a.verdict.get("model_stability") or {}).get("label", SKIP)
    print(f"⚠ MODEL VERDICT: {stab}")
    print(f"  EV £{a.verdict['ev_best_line']:+.3f} a £2 line - "
          f"playing this draw is entertainment, not investment.")
    if yes:
        print("  --yes given: generating anyway (recorded as a manual override).")
        return True
    if not _interactive():
        print("  Not generating: pass --yes to override a SKIP from a script.")
        return False
    return _ask("Generate anyway?")


def ensure_verdict_on_file(a: Advice) -> None:
    """The ledger records the saved verdict with each ticket, and refuses one
    saved for another draw - so make sure this draw's is the one on file."""
    latest = OUT_DIR / "latest.json"
    try:
        saved = json.loads(latest.read_text())["metadata"]["provenance"]["draw_date"]
    except Exception:
        saved = None
    if saved != (a.cond.draw_date.isoformat() if a.cond.draw_date else None):
        save(replace(a, portfolio=[]))


def offer_ledger(a: Advice, tickets: list, record: bool) -> None:
    """Record bought lines with the verdict they were bought against."""
    if not tickets:
        return
    if not record:
        if not _interactive():
            print("(not recorded - pass --record, or run ./lotto ledger add)")
            return
        if not _ask(f"Did you buy these {len(tickets)} line(s)? Record them in the ledger?"):
            print("Not recorded.")
            return
    roi_ledger.cmd_add(Namespace(
        draw_date=a.cond.draw_date.isoformat(),
        lines="; ".join(" ".join(map(str, t)) for t in tickets),
        from_latest=False, cost_per_line=roi_ledger.COST_PER_LINE))


def cmd_ticket(args) -> int:
    a = advise(lines=args.lines, force=True)
    if not confirm_override(a, args.yes):
        return 2
    print(render(a))
    print(save(a))
    offer_ledger(a, [p["line"] for p in a.portfolio], args.record)
    return 0


def cmd_wheel(args) -> int:
    a = advise()
    if not confirm_override(a, args.yes):
        return 2
    try:
        w = wheel_portfolio(args.pool_size, args.lines, cond=a.cond)
    except ValueError as exc:
        print(exc)
        return 1
    print(render_wheel(w))
    print(f"Saved to {save_wheel(w)} (latest.json keeps the verdict)")
    ensure_verdict_on_file(a)
    offer_ledger(a, w.tickets, args.record)
    return 0


# --------------------------------------------------------------------------
# your own numbers
# --------------------------------------------------------------------------

def parse_line(numbers: list) -> list[int]:
    try:
        line = sorted(int(n) for n in numbers)
    except ValueError:
        raise ValueError("numbers only, e.g. ./lotto check 3 7 12 19 24 31")
    if len(line) != 6 or len(set(line)) != 6 or not all(1 <= n <= 59 for n in line):
        raise ValueError("need six different numbers from 1 to 59")
    return line


def check_text(line: list[int], a: Advice) -> str:
    cond = a.cond
    mine = line_ev(line, cond)
    best = build_portfolio(1, cond, seed=0)[0]
    ratio = popularity_ratio(line)
    share = expected_cowinner_share(line, cond.tickets_sold, cond.rounds)
    best_share = expected_cowinner_share(best["line"], cond.tickets_sold, cond.rounds)
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    return "\n".join([
        f"Your line:    {' '.join(f'{n:2d}' for n in line)}",
        RULE,
        f"Popularity:   x{ratio:.2f} of an average line "
        f"({'more' if ratio > 1 else 'fewer'} people hold it)",
        f"If it hits:   you keep {pct(share)} of the jackpot on average "
        f"(an unpopular line keeps {pct(best_share)})",
        f"EV this draw: £{mine:+.3f} a £2 line   "
        f"(unpopular line £{best['ev']:+.3f}, e.g. {' '.join(map(str, best['line']))})",
        "Odds:         1 in 22,528,737 for the jackpot - identical for every line.",
        "              Numbers change who you share with, never whether you win.",
    ])


def cmd_check(args) -> int:
    try:
        line = parse_line(args.numbers)
    except ValueError as exc:
        print(exc)
        return 1
    print(check_text(line, advise()))
    return 0


def cmd_whatif(args) -> int:
    a = advise(lines=args.lines, jackpot=args.jackpot, roll_down=args.roll_down,
               ordinary=args.ordinary, tickets=args.tickets, force=args.force)
    print(render(a))
    print("(what-if - nothing saved)")
    return 0


# --------------------------------------------------------------------------
# ledger, analysis, sync
# --------------------------------------------------------------------------

def cmd_ledger(args) -> int:
    action = args.action or "report"
    if action == "add":
        if not args.lines and not args.from_latest:
            print('ledger add needs --lines "1 2 3 4 5 6; ..." or --from-latest')
            return 1
        roi_ledger.cmd_add(Namespace(draw_date=args.draw_date, lines=args.lines,
                                     from_latest=args.from_latest,
                                     cost_per_line=roi_ledger.COST_PER_LINE))
    elif action == "settle":
        roi_ledger.cmd_settle(None)
    roi_ledger.cmd_report(None)
    if not roi_ledger.LEDGER_FILE.exists():
        return 0
    print(f"⚠ {roi_ledger.LEDGER_FILE} lives only on this Mac - it has no backup.")
    return 0


def cmd_analysis(args) -> int:
    if not args.name:
        print("ANALYSIS - read-only; none of these can change today's verdict")
        print(RULE)
        for name, (_, _, runtime, what) in ANALYSES.items():
            print(f"  {name:17} {what}  ({runtime})")
        return 0
    if args.name not in ANALYSES:
        print(f"unknown analysis {args.name!r}; one of: {', '.join(ANALYSES)}")
        return 1
    script, extra, runtime, _ = ANALYSES[args.name]
    print(f"[{args.name}] running {script} - {runtime}")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, str(ROOT / script), *extra],
                          cwd=ROOT, env=env).returncode


def sync_refusal(git: dict) -> str | None:
    """Why `sync` must not run here, or None. The sync script discards the
    collector's files and fast-forwards onto origin/main."""
    if git.get("branch") is None:
        return "not a git checkout"
    if git["branch"] != "main":
        return (f"on branch {git['branch']!r} - sync fast-forwards main; "
                f"switch with `git checkout main` first")
    if git["dirty"]:
        return "uncommitted edits in tracked files - commit or stash them first"
    return None


def cmd_sync(args) -> int:
    why = sync_refusal(git_state())
    if why:
        print(f"Not syncing: {why}.")
        return 1
    script = ROOT / "scripts" / "monitoring" / "sync_collector_data.sh"
    code = subprocess.run(["bash", str(script)], cwd=ROOT).returncode
    if code == 0:
        n, day = latest_collected()
        print(f"Data now reaches draw {n} ({day}).")
    return code


# --------------------------------------------------------------------------
# menu
# --------------------------------------------------------------------------

MENU = [
    ("1", "Generate 5-line portfolio", lambda: cmd_ticket(Namespace(lines=5, yes=False, record=False))),
    ("2", "Generate wheel (6 tickets)", lambda: cmd_wheel(Namespace(pool_size=12, lines=None, yes=False, record=False))),
    ("3", "Show EV details", lambda: cmd_play(Namespace(lines=5, threshold=0.0, bankroll=1000.0))),
    ("4", "Check my own numbers", None),
    ("5", "Ticket / ROI history", lambda: cmd_ledger(Namespace(action="report"))),
    ("6", "Run analysis tools", None),
    ("7", "Sync latest data", lambda: cmd_sync(None)),
    ("0", "Exit", None),
]


def menu() -> int:
    while True:
        print()
        print(status_text(advise(), pre_ev_gate.run(), git_state()))
        print()
        print("What do you want to do?")
        for key, label, _ in MENU:
            print(f"  [{key}] {label}")
        try:
            choice = input("> ").strip()
        except EOFError:
            return 0
        if choice == "0":
            return 0
        if choice == "4":
            try:
                raw = input("Six numbers: ").replace(",", " ").split()
                print(check_text(parse_line(raw), advise()))
            except (ValueError, EOFError) as exc:
                print(exc)
        elif choice == "6":
            cmd_analysis(Namespace(name=None))
            try:
                name = input("Which one (Enter to go back)? ").strip()
            except EOFError:
                name = ""
            if name:
                cmd_analysis(Namespace(name=name))
        else:
            action = next((fn for key, _, fn in MENU if key == choice and fn), None)
            if action is None:
                print("Choose one of the numbers above.")
                continue
            action()
        try:
            input("\nEnter to continue...")
        except EOFError:
            return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lotto", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    sub.add_parser("status", help="next draw, data, decision").set_defaults(func=cmd_status)

    s = sub.add_parser("play", help="exactly `make play`")
    s.add_argument("--lines", type=int, default=5)
    s.add_argument("--threshold", type=float, default=0.0)
    s.add_argument("--bankroll", type=float, default=1000.0)
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("ticket", help="unpopular lines, whatever the verdict")
    s.add_argument("--lines", type=int, default=5)
    s.add_argument("--yes", action="store_true", help="override a SKIP without asking")
    s.add_argument("--record", action="store_true", help="record in the ledger without asking")
    s.set_defaults(func=cmd_ticket)

    s = sub.add_parser("wheel", help="the 6-ticket wheel on the unpopular pool")
    s.add_argument("--pool-size", type=int, default=12)
    s.add_argument("--lines", type=int, default=None)
    s.add_argument("--yes", action="store_true")
    s.add_argument("--record", action="store_true")
    s.set_defaults(func=cmd_wheel)

    s = sub.add_parser("check", help="how your own six numbers share a jackpot")
    s.add_argument("numbers", nargs="+")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("whatif", help="price a hypothetical draw; saves nothing")
    s.add_argument("--jackpot", type=float)
    s.add_argument("--tickets", type=int)
    s.add_argument("--roll-down", action="store_true")
    s.add_argument("--ordinary", action="store_true")
    s.add_argument("--lines", type=int, default=5)
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_whatif)

    s = sub.add_parser("ledger", help="tickets bought and what they returned")
    s.add_argument("action", nargs="?", choices=("report", "settle", "add"))
    s.add_argument("--lines")
    s.add_argument("--from-latest", action="store_true")
    s.add_argument("--draw-date", default="next")
    s.set_defaults(func=cmd_ledger)

    s = sub.add_parser("analysis", help="read-only reports")
    s.add_argument("name", nargs="?")
    s.set_defaults(func=cmd_analysis)

    sub.add_parser("sync", help="pull the collector's data").set_defaults(func=cmd_sync)
    return p


def main(argv: list[str] | None = None, root: Path | None = ROOT) -> int:
    # Every data path in the project is relative to the repo root.
    if root is not None:
        os.chdir(root)
    args = build_parser().parse_args(argv)
    if args.command is None:
        if not _interactive():
            print(status_text(advise(), pre_ev_gate.run(), git_state()))
            return 0
        return menu()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
