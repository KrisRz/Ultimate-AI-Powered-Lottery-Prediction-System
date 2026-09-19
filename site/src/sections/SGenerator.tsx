'use client';

/**
 * Panel A - your five lines.
 *
 * The product, first, before any of the argument. The playslip grid finally
 * does the job it was drawn for: marked cells are the numbers you would put on
 * a real slip.
 *
 * The claim under it has to stay exactly right. These lines are not more
 * likely to come up - nothing is, and panel E says so with 930 draws behind
 * it. They are less likely to be *shared*, which is a different and much
 * smaller claim that happens to be worth a lot of money on the rare occasion
 * it pays.
 */

import { useMemo, useState, useSyncExternalStore } from 'react';

import { generatePortfolio } from '@/data/generator';
import { expectedShare, popularityRatio } from '@/data/popularity';
import { count, gbp, gbpPence, longDate } from '@/data/format';
import type { Ev, Hook, Popularity } from '@/data/types';

const LINES = 5;

/** A line an ordinary player might actually pick: a birthday spread. */
const TYPICAL_LINE = [3, 7, 12, 19, 24, 31];

/** One offset per tab, drawn on first read and stable for the visit. */
let visitorSeedValue = 0;
const visitorSeed = () => {
  if (visitorSeedValue === 0) {
    visitorSeedValue = 1 + Math.floor(Math.random() * 1_000_000);
  }
  return visitorSeedValue;
};
const serverSeed = () => 0;
/** Nothing ever changes it, so there is nothing to subscribe to. */
const subscribeToNothing = () => () => {};

export function SGenerator({
  popularity,
  hook,
  ev,
  seed,
}: {
  popularity: Popularity;
  hook: Hook;
  ev: Ev;
  seed: number;
}) {
  const [nonce, setNonce] = useState(0);

  // Every visitor used to get the same first slip - the seed is the draw date -
  // so at any traffic at all this page would have its readers sharing a jackpot
  // with each other, which is the one thing it tells them to avoid.
  //
  // useSyncExternalStore rather than setState in an effect: the offset is
  // client-only data that must not exist during the server render, which is
  // exactly the divergence this hook is for. The server snapshot is 0, so the
  // prerendered page stays deterministic and the committed snapshot and golden
  // fixtures still diff; React swaps in the visitor's own offset once hydrated.
  const visitorOffset = useSyncExternalStore(subscribeToNothing, visitorSeed, serverSeed);

  const model = popularity.model;
  const bands = popularity.installed_step;

  const lines = useMemo(
    () => generatePortfolio(LINES, seed + visitorOffset + nonce, model, bands, hook.n_balls),
    [seed, visitorOffset, nonce, model, bands, hook.n_balls],
  );

  // The comparison that makes the case, at a jackpot big enough to be worth
  // caring about. Both lines have identical odds of coming up.
  const jackpot = 10_000_000;
  const entries = ev.regimes[0]?.tickets_sold ?? ev.live.tickets_sold;
  const rounds = ev.regimes[0]?.rounds ?? 1;
  const typicalRatio = useMemo(
    () => popularityRatio(TYPICAL_LINE, model, bands),
    [model, bands],
  );

  // Whether the verdict survives a different popularity model - a different
  // question from `ev.live.robust`, which asks whether it survives the sales
  // forecast being wrong.
  const stability = ev.live.model_stability;
  const specRange = `${gbpPence(stability.ev_spec_min)} to ${gbpPence(stability.ev_spec_max)}`;

  const best = lines[0];
  const yourShare = best
    ? jackpot * expectedShare(best.ratio, entries, hook.total_combinations, rounds)
    : 0;
  const typicalShare =
    jackpot * expectedShare(typicalRatio, entries, hook.total_combinations, rounds);
  const difference = yourShare - typicalShare;

  return (
    <section id="panel-a" className="generator" aria-labelledby="panel-a-title">
      {/* Heading, then the numbers. The explanation goes underneath them: a
          reader who came for a slip should not have to get through a paragraph
          to reach one, and the paragraph makes more sense once they have. */}
      <div className="generator-head">
        <p className="eyebrow">Panel A &middot; your slip</p>
        <h2 className="h-section" id="panel-a-title">
          Five lines nobody else is playing
        </h2>
      </div>

      {/* The meaning of the percentage is stated once, here, instead of being
          repeated on all five cards - five copies of the same sentence read as
          noise and forced a two-line wrap into every card head. */}
      <p className="slips-caption small quiet">
        The bar on each line is how many people play it, against an average line
        at 100%. Shorter is better: fewer people to split a jackpot with.
      </p>

      <ol className="slips">
        {lines.map(({ line, ratio }, index) => (
          <li
            className="slip-card"
            data-featured={index === 0 ? 'true' : undefined}
            key={`${nonce}-${index}`}
          >
            <header className="slip-card-head">
              <span className="slip-index num">
                {index === 0 ? 'Your first line' : String(index + 1).padStart(2, '0')}
              </span>
              <span className="slip-share num">{Math.round(ratio * 100)}%</span>
            </header>
            {/* Decorative: the figure beside it already carries the number, so
                a screen reader hearing both would hear it twice. */}
            <div className="slip-meter" aria-hidden="true">
              <div
                className="slip-meter-fill"
                style={{ width: `${Math.min(100, Math.round(ratio * 100))}%` }}
              />
            </div>
            <ol
              className="slip-numbers"
              aria-label={`Line ${index + 1}, played by ${Math.round(ratio * 100)}% as many people as an average line`}
            >
              {line.map((n) => (
                <li className="slip-ball num" key={n}>
                  {n}
                </li>
              ))}
            </ol>
          </li>
        ))}
      </ol>

      <div className="generator-actions">
        <button type="button" className="button" onClick={() => setNonce((n) => n + 1)}>
          Generate five more
        </button>
        <p className="small quiet generator-note">
          Built in your browser from the same model the toolkit runs, calibrated on{' '}
          {count(popularity.n_observations)} draw-rounds of real winner counts. Nothing
          is sent anywhere.
        </p>
      </div>

      <p className="lede prose generator-lede">
        Six numbers from {hook.n_balls}, drawn to avoid the dates, the lucky sevens and
        the diagonal patterns most tickets carry. Exactly the same chance of coming up as
        any other line — but if one does come up, you are sharing it with far fewer
        people.
      </p>

      <aside className="payoff">
        <p className="eyebrow">What that is worth</p>
        <div className="payoff-grid">
          <div>
            <p className="payoff-label small">
              A typical line — {TYPICAL_LINE.join(', ')}
            </p>
            <p className="payoff-value num">{gbp(typicalShare)}</p>
          </div>
          <div data-highlight="true">
            <p className="payoff-label small">
              Your first line — {best?.line.join(', ')}
            </p>
            <p className="payoff-value num">{gbp(yourShare)}</p>
          </div>
        </div>
        <p className="payoff-note">
          Both lines have exactly the same chance of winning a {gbp(jackpot)} jackpot.
          The difference of <strong className="num">{gbp(difference)}</strong> is what
          you keep instead of splitting it with everyone who played their birthdays.
        </p>
      </aside>

      {/* The honest counterweight, and the only place on the page that says
          whether the ADVICE depends on the model the lines came from. The
          toolkit re-prices every verdict with a flat popularity model and
          with one twice as strong; if the answer survives all three it is
          labelled robust, and if it does not, saying so is the whole point.
          A page that hands out lines and hides a SKIP would be the exact
          failure this project calls a bug in prose. */}
      <aside
        className="stability"
        data-sensitive={stability.stable ? undefined : 'true'}
      >
        <p className="eyebrow">Does the advice depend on this model?</p>
        <p className="stability-line">
          The lines above come from the toolkit&rsquo;s popularity model — and so does
          its answer to whether this draw is worth playing at all:{' '}
          <strong className="stability-verdict">{ev.live.verdict}</strong> for{' '}
          {longDate(ev.live.draw_date)}, at{' '}
          <span className="num">{gbpPence(ev.live.ev_best_line)}</span> a line.
        </p>
        <p className="stability-note small">
          {!stability.stable ? (
            <>
              Re-priced with a flat popularity model and with one twice as strong, that
              answer <strong>changes</strong> — {specRange}. Read it as marginal, not as
              a recommendation.
            </>
          ) : (
            <>
              Re-priced with a flat popularity model and with one twice as strong, that
              answer does not change — {specRange} across every specification tested.
            </>
          )}{' '}
          <span className="stability-label num">{stability.label}</span>
        </p>
      </aside>
    </section>
  );
}
