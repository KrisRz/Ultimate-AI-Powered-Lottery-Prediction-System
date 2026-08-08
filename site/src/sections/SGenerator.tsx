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

import { useMemo, useState } from 'react';

import { generatePortfolio } from '@/data/generator';
import { expectedShare, popularityRatio } from '@/data/popularity';
import { count, gbp } from '@/data/format';
import type { Ev, Hook, Popularity } from '@/data/types';

const LINES = 5;

/** A line an ordinary player might actually pick: a birthday spread. */
const TYPICAL_LINE = [3, 7, 12, 19, 24, 31];

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

  const model = popularity.model;
  const bands = popularity.installed_step;

  const lines = useMemo(
    () => generatePortfolio(LINES, seed + nonce, model, bands, hook.n_balls),
    [seed, nonce, model, bands, hook.n_balls],
  );

  // The comparison that makes the case, at a jackpot big enough to be worth
  // caring about. Both lines have identical odds of coming up.
  const jackpot = 10_000_000;
  const entries = ev.regimes[0]?.tickets_sold ?? ev.live.tickets_sold;
  const typicalRatio = useMemo(
    () => popularityRatio(TYPICAL_LINE, model, bands),
    [model, bands],
  );

  const best = lines[0];
  const yourShare = best
    ? jackpot * expectedShare(best.ratio, entries * 2, hook.total_combinations)
    : 0;
  const typicalShare =
    jackpot * expectedShare(typicalRatio, entries * 2, hook.total_combinations);
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

      <ol className="slips">
        {lines.map(({ line, ratio }, index) => (
          <li className="slip-card" key={`${nonce}-${index}`}>
            <header className="slip-card-head">
              <span className="slip-index num">{String(index + 1).padStart(2, '0')}</span>
              <span className="slip-share small">
                played by <strong className="num">{Math.round(ratio * 100)}%</strong> as
                many people
              </span>
            </header>
            <ol className="slip-numbers" aria-label={`Line ${index + 1}`}>
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
    </section>
  );
}
