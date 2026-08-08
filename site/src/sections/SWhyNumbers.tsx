'use client';

/**
 * Panel B - the evidence under the generator.
 *
 * The grid is the signature motif doing its third job: in panel A it holds
 * your marked numbers, here it holds how often everyone else marks each one.
 * The cliff after 31 is the whole story and it is visible without a caption.
 *
 * Pick six and the readout says what those numbers cost you. Nothing here
 * changes the odds and the copy never suggests it does.
 */

import { useMemo, useState } from 'react';

import { expectedShare, popularityRatio } from '@/data/popularity';
import { count, gbp, times } from '@/data/format';
import type { Ev, Hook, Popularity } from '@/data/types';

const PICK = 6;
const DEMO_JACKPOT = 10_000_000;

export function SWhyNumbers({
  popularity,
  hook,
  ev,
}: {
  popularity: Popularity;
  hook: Hook;
  ev: Ev;
}) {
  const [picked, setPicked] = useState<number[]>([3, 7, 12, 19, 24, 31]);
  const [showInstalled, setShowInstalled] = useState(false);

  const model = popularity.model;
  const bands = popularity.installed_step;
  const [lowest, highest] = popularity.recovered_range;

  const weightOf = (n: number) => {
    if (showInstalled) {
      for (const band of bands) if (n <= band.to) return band.weight;
      return 1;
    }
    return popularity.recovered[n - 1] ?? 1;
  };

  const toggle = (n: number) =>
    setPicked((current) =>
      current.includes(n)
        ? current.filter((x) => x !== n)
        : current.length >= PICK
          ? current
          : [...current, n].sort((a, b) => a - b),
    );

  const complete = picked.length === PICK;
  const ratio = useMemo(
    () => (complete ? popularityRatio(picked, model, bands) : null),
    [picked, complete, model, bands],
  );

  const entries = (ev.regimes[0]?.tickets_sold ?? ev.live.tickets_sold) * 2;
  const share =
    ratio === null ? null : DEMO_JACKPOT * expectedShare(ratio, entries, hook.total_combinations);

  const buckets = popularity.match3_multiplier_by_low31;
  const maxBucket = Math.max(...buckets.map((b) => b.mean_multiplier));

  return (
    <section id="panel-b" className="why" aria-labelledby="panel-b-title">
      <hr className="perf" />
      <div className="why-head">
        <p className="eyebrow">Panel B &middot; why those numbers</p>
        <h2 className="h-section" id="panel-b-title">
          Everyone plays their birthdays
        </h2>
        <p className="lede prose">
          Numbers up to 31 are days of the month, and up to 12 they are months too. You
          can see it in the winner counts: draws made of low numbers produce far more
          small winners per ticket than draws of high ones. Nobody is picking 53.
        </p>
      </div>

      <div className="why-body">
        <div className="picker">
          <div className="picker-head">
            <p className="control-label">Pick six</p>
            <button
              type="button"
              className="toggle small"
              onClick={() => setShowInstalled((v) => !v)}
              aria-pressed={showInstalled}
            >
              {showInstalled ? 'showing the three-band model' : 'showing the raw fit'}
            </button>
          </div>

          <ol className="ball-grid" aria-label={`Numbers 1 to ${hook.n_balls}`}>
            {Array.from({ length: hook.n_balls }, (_, i) => i + 1).map((n) => {
              const weight = weightOf(n);
              const t = (weight - lowest) / (highest - lowest);
              return (
                <li key={n}>
                  <button
                    type="button"
                    className="ball"
                    aria-pressed={picked.includes(n)}
                    aria-label={`${n}, played ${weight.toFixed(2)} times as often as average`}
                    onClick={() => toggle(n)}
                    style={{ '--t': String(Math.min(1, Math.max(0, t))) } as React.CSSProperties}
                  >
                    {n}
                  </button>
                </li>
              );
            })}
          </ol>

          <p className="scale-key small quiet">
            <span className="scale-swatch" data-end="low" /> least played ({lowest.toFixed(2)}
            ) &nbsp;&rarr;&nbsp; most played ({highest.toFixed(2)}){' '}
            <span className="scale-swatch" data-end="high" />
          </p>
        </div>

        <div className="picker-readout" aria-live="polite">
          {complete && ratio !== null && share !== null ? (
            <>
              <p className="control-label">Your line is played</p>
              <p className="readout-big num">{times(ratio)}</p>
              <p className="small quiet">as often as an average line</p>
              <dl className="results">
                <div data-positive={ratio < 1}>
                  <dt>Share of a {gbp(DEMO_JACKPOT)} jackpot</dt>
                  <dd className="num">{gbp(share)}</dd>
                  <dd className="small quiet">
                    after splitting with everyone else holding it
                  </dd>
                </div>
              </dl>
              <button type="button" className="button" onClick={() => setPicked([])}>
                Clear
              </button>
            </>
          ) : (
            <>
              <p className="control-label">Your line is played</p>
              <p className="readout-big num quiet">&mdash;</p>
              <p className="small quiet">
                {PICK - picked.length} more to go. Odds are identical whatever you pick;
                only the split changes.
              </p>
            </>
          )}
        </div>
      </div>

      <figure className="buckets">
        <figcaption className="small quiet">
          Match-3 winners per ticket, by how many of the six drawn numbers were 31 or
          below. Across {count(popularity.n_observations)} draw-rounds.
        </figcaption>
        <ol className="bucket-rows">
          {buckets.map((bucket) => (
            <li key={bucket.n_low31}>
              <span className="bucket-label num">{bucket.n_low31}</span>
              <span className="bucket-track">
                <span
                  className="bucket-fill"
                  style={{ width: `${(bucket.mean_multiplier / maxBucket) * 100}%` }}
                />
              </span>
              <span className="bucket-value num">{bucket.mean_multiplier.toFixed(2)}&times;</span>
            </li>
          ))}
        </ol>
      </figure>
    </section>
  );
}
