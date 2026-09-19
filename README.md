# Lotto EV Toolkit

An honest toolkit for UK Lotto (6/59, two rounds per draw since June 2026).

**It does not predict lottery numbers — nothing can.** Draws are independent
and uniformly random; this repo's own walk-forward backtest confirms that no
method here beats random picks (and says so on the dashboard). What the
toolkit *does* do is optimize the two things a player actually controls:

1. **When to play** — expected value of a £2 line, computed from exact tier
   probabilities, current jackpot, the two-round format, and Must-Be-Won
   roll-downs. Ordinary draws are negative-EV; the advisor says **SKIP**.
2. **What to play** — unpopular combinations. Lower-tier prizes are fixed,
   but the jackpot is shared: a line avoiding dates (1–31), "lucky" numbers,
   and visual patterns splits a jackpot with far fewer people.

Plus the two feedback loops that keep it honest:

3. **Backtest with significance testing** — every method is compared against
   the no-skill hypergeometric baseline (Monte-Carlo p-values, bootstrap CI).
4. **ROI ledger** — every line you actually play is recorded and settled
   against real results. The dashboard shows the truth in pounds.

## Quick start

```bash
make setup                 # create the conda env (environment.yml)
conda activate lotto-predict

make backfill              # full draw history since 1994 (Merseyworld archive)
make sales                 # per-draw ticket sales since 1994 + cross-validation
make play                  # EV verdict for the next draw + portfolio
make dashboard             # generate + open outputs/dashboard.html
```

Daily use:

```bash
make play                                        # should I play? with what?
python scripts/roi_ledger.py add --from-latest   # record lines you actually bought
make roi                                         # settle & report after the draw
make backtest                                    # method-vs-random, p-values
make fairness                                    # is the machine fair? six tests
make ensemble                                    # score+MonteCarlo vs random
make install-cron                                # auto post-draw routine (Wed/Sat 22:30)
```

## How the EV model works

- **Probabilities** (`lottery/ev.py`): exact hypergeometric — jackpot
  1:45,057,474; 5+bonus 1:7,509,579; match-5 1:144,415; match-4 1:2,180;
  match-3 1:96; match-2 1:10.3 — verified in tests.
- **Prizes**: fixed lower tiers re-derived from official 2026 data on every run
  (`calibrate_fixed_prizes`, median over collected draws — 5+bonus £1,000,000,
  match-5 £1000, match-4 £50, match-3 £10, match-2 £1); the jackpot is
  pari-mutuel and gets discounted by the expected number of co-winners. Note
  that a **roll-down draw pays more in the low tiers** (3190 paid £24/£5); that
  uplift is the jackpot being redistributed and is priced separately, so the
  median deliberately ignores it.
- **Popularity model**: number weights (dates over-played, high numbers
  under-played) + pattern multipliers (arithmetic sequences, birthday-only
  tickets), calibrated against 1,126 draws of Match-3 winner counts and
  normalized so the average line scores exactly 1.0. Confirmed a second,
  independent way — from how often the bonus ball turns a Match 5 into a
  5+B (Hanley & Cronin 2023), which needs no sales figure: 1.22 / 1.17 / 0.79
  against the installed 1.23 / 1.10 / 0.83, all inside the intervals. On
  big-jackpot draws players pick about 9% more uniformly (Polin et al. 2021,
  significant here too), which moves EV by under a penny a line, so it is
  reported rather than modelled. Calibration data accumulates in
  `data/prize_tiers.csv` with every fetch; re-check the fit on recent draws
  only with `python scripts/calibrate_popularity.py --last-draws 500`.
- **When the jackpot must be paid out**: rollovers are capped at 5, so the 6th
  draw of a roll is Must-Be-Won and rolls down. `make play` counts down to it
  and now prices it in advance ("Must-Be-Won in 2 draws, ~2026-09-09 —
  projected pool ~£8,123,650 vs break-even £9,112,484, likely SKIP"). The cap
  fires every few weeks, but **that is a schedule, not an opportunity.** A
  Must-Be-Won draw sells more than an ordinary one and the roll-down pays J/N,
  so the extra buyers eat the edge — and since the 7 June 2026 licence the
  jackpot takes **8.88% of sales** (was 9.79%) and restarts at **£2m on both
  days** (Saturday was £3.8m), so a capped roll now reaches only about
  £7.8–9.6m. Break-even is ~£9.3m on a Wednesday and ~£13.2m on a Saturday
  (Saturdays sell half again as many lines to share the pool between). Of the
  four capped Must-Be-Won draws of the two-round era, the three Saturdays fell
  well short; the Wednesday one (3205, £7.8m) turned out +£0.06 a line at the
  sales it actually saw, because the installed uplift overstates them — see
  `make uplift`. Re-priced at each historical draw's own measured sales, 22 of
  53 cap-driven draws since 2019 clear break-even — but every one of those 22
  carried a pool of **£11.3m or more**, which a cap alone no longer reaches;
  the last was March 2026, before the redesign.
- **Special draws are the other kind of Must-Be-Won** — a guaranteed £12m,
  £15m or £20m with a campaign behind it, 43 of them since late 2018, often
  announced only a couple of days ahead (3206's £12m appeared in the feed on
  the Thursday). They sell far more than a capped roll: ×1.39 an ordinary
  Saturday, ×2.2 an ordinary Wednesday (every Wednesday one on file was at
  Christmas, up to ×3.2). The advisor tells the two apart from the feed and
  prices each with its own uplift, so a special breaks even at ~£14m on
  either day and holds across the sales range only from ~£16.5m: **£20m pays,
  £15m is a coin flip on sales, £12m does not.** The catch is the calendar:
  the operator ran sixteen £20m specials in 2020–2023 and none since — 2024
  to date brought fourteen at £15m and two at £12m. So a clearly playable
  draw is currently **rarer than once a year**, and most PLAY verdicts will be
  the marginal kind the advisor labels as such.
  **Must-Be-Won is a necessary condition, not a sufficient one — wait for the
  advisor's verdict, not the calendar.**
- **Sales are the fragile input**: on a roll-down the EV is dominated by J/N,
  so `make play` reports the verdict across the sales range (the day's uplift
  quartiles, capped or special) and flags whether it survives the busy end. A draw that only
  clears on the central estimate is not a real opportunity. Re-calibrate with
  `python scripts/calibrate_mbw_uplift.py`.
- **Two rounds**: since 2026-06-07 every ticket enters two draws per night;
  EV sums over both.
- **Portfolio** (`lottery/portfolio.py`): greedy max-EV selection with
  diversity constraints (pairwise overlap ≤ 2, ≥ 2 numbers above 31, no
  triple sequences).

## Repository layout

```
lottery/            EV engine: ev.py (probabilities, popularity, EV), portfolio.py
scripts/
  backfill_history.py   full 1994→today history (both eras, both rounds)
  fetch_data.py         official XML feed → merged data + prize_tiers.csv
  ev_play.py            EV advisor CLI (make play)
  roi_ledger.py         real-money ledger (add / settle / report)
  dashboard.py          static dashboard generator (make dashboard)
  new_predict.py        legacy frequency/LSTM path (kept as a sanity-check)
  validations/backtest.py   walk-forward backtest + significance tests
  validations/fairness.py   is the machine fair? six tests (make fairness)
  validations/ensemble_score.py  the scoring ensemble, walk-forward tested
  monitoring/           nightly backtest, post-draw routine
data/               draw history, prize tiers, ledger (local, not committed)
outputs/            predictions, validation runs, dashboard (not committed)
tests/              pytest suite (fast, no network)
ops/                launchd template for the post-draw cron
```

## Email alerts (+EV draws)

The post-draw routine emails you only when the next draw clears the EV
threshold, which in the current structure is one or two draws a year rather
than the ~9 Must-Be-Won draws the calendar produces. Silence is the normal
signal. The email is self-contained: draw date,
jackpot, EV, break-even, **the lines to play**, and a ready-to-paste
`roi_ledger add` command. Lines are seeded from the draw date, so the evening
run and the next-morning retry propose the same portfolio rather than two
different ones.

Put SMTP credentials in `~/.lotto_env` (outside the repo, never committed):

```bash
export SMTP_SERVER=smtp.gmail.com      # SSL port 465; Gmail needs an App Password
export SMTP_USER=you@gmail.com
export SMTP_PASS=your-app-password
export EMAIL_TO=you@gmail.com
```

Test it: `make post-draw` (sends only on a PLAY verdict).

## Playing when the model says SKIP

SKIP is the usual verdict, and on most draws it is simply correct. But the
model answers two questions, not one, and the second has an answer even when
the first says no:

- **When to play** — is this draw's EV above zero? Usually no.
- **What to play** — which lines share a jackpot with the fewest people?
  Always answerable.

So if you are going to buy a ticket regardless — for the fun of it, for a
syndicate, because it is a rollover and you want to be in it — buy the lines
that share least:

```bash
PYTHONPATH=. python scripts/ev_play.py --force   # portfolio + the cost, stated
PYTHONPATH=. python scripts/wheel_play.py        # 6-line wheel, its own file
```

`--force` prints the same verdict and EV it always would, then builds the
portfolio anyway. It does not pretend the draw is worth playing: the saved
`latest.json` still records `"play": false`, so the ROI ledger tells the truth
about what the advisor actually recommended.

**What those lines do and do not do.** They do not improve your chance of
matching six — nothing can; it is 1 in 45,057,474 for every combination, and
`make fairness` demonstrates that on eleven years of draws. What they do is
avoid the numbers other people pick. Measured on this archive, draws made
mostly of 1–31 (birthday numbers) produce **77% more winners per ticket** than
draws dominated by numbers above 31. That is the entire edge, it is real, and
it only ever pays out in the branch where you win.

**Is the machine fair?** `make fairness` runs six tests over the 59-ball era
(1,171 draw-rounds, both rounds): chi-square uniformity, the most deviant ball
corrected for having searched all 59, gap analysis against the gambler's
fallacy, draw-to-draw dependence against the hypergeometric, and pair
co-occurrence corrected for having searched 1,711 pairs, and per drawing
machine and ball set — the one hypothesis with a physical mechanism behind
it. All six come back clean, and each reports the smallest bias it could
have seen, because a null result without its power is only half a finding. The two corrected tests are the ones that matter: on genuinely random
data the largest |z| among 59 balls sits near 2.5, which an uncorrected test
reads as "significant" — so an uncorrected search finds a hot number in clean
noise every time. `tests/test_fairness.py` plants a real bias in simulated
draws and checks each detector fires, which is what makes "no signal" on the
real archive mean something.

## Honest expectations

UK Lotto returns roughly half of ticket sales as prizes. The EV strategy
("play rarely, play unpopular, track everything") raises the *conditional*
payout and occasionally — big roll-downs — pushes EV above zero, but variance
is enormous. This project's guaranteed positive return is the engineering:
a clean data pipeline, a statistically sound backtest, and an EV model you
can audit line by line. See `plan.md` for the full analysis and roadmap.

## License

MIT
