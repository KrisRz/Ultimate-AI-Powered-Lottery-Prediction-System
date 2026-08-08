'use client';

/**
 * Panel C - when a ticket is worth buying.
 *
 * A workbench rather than another scrolling panel: after two sticky sections
 * the rhythm wants a stop, and the argument here is one the reader should make
 * themselves by moving the jackpot until the number turns positive.
 *
 * The maths is exact. EV is affine in the jackpot, so the readout is the model
 * evaluated, not a curve sampled at intervals and interpolated between.
 */

import { useId, useState } from 'react';

import { EvCurve } from '@/charts/EvCurve';
import { breakEven, evAt } from '@/data/ev';
import { count, gbp, gbpPence, longDate } from '@/data/format';
import type { Ev } from '@/data/types';

export function S3Ev({
  ev,
  asOf,
  ticketPrice,
}: {
  ev: Ev;
  asOf: string;
  ticketPrice: number;
}) {
  const sliderId = useId();
  const [jackpot, setJackpot] = useState(ev.live.jackpot_gbp);

  const ordinary = ev.regimes.find((r) => r.key === 'ordinary');
  const mbw = ev.regimes.find((r) => r.key === 'mbw');
  if (!ordinary || !mbw) return null;

  const evOrdinary = evAt(ordinary, jackpot);
  const evMbw = evAt(mbw, jackpot);
  const anyPositive = Math.max(evOrdinary, evMbw) >= 0;

  const readout =
    `Jackpot ${gbp(jackpot)}. An ordinary draw returns ${gbpPence(evOrdinary)} per line; ` +
    `a Must-Be-Won draw returns ${gbpPence(evMbw)}.`;

  return (
    <section id="panel-c" className="workbench" aria-labelledby="panel-c-title">
      <hr className="perf" />
      <div className="workbench-head">
        <p className="eyebrow">Panel C &middot; the first decision</p>
        <h2 className="h-section" id="panel-c-title">
          When is £2 worth spending?
        </h2>
        <p className="lede prose">
          A line pays back the fixed tiers whatever happens, plus a share of the jackpot
          if it wins. Add those up, subtract the {gbp(ticketPrice)} you paid, and you have
          what the ticket is actually worth. Move the jackpot and watch for the crossing.
        </p>
      </div>

      <div className="workbench-body">
        <EvCurve ev={ev} jackpot={jackpot} />

        <div className="controls">
          <label className="control-label" htmlFor={sliderId}>
            Jackpot
          </label>
          <output className="control-value num" htmlFor={sliderId}>
            {gbp(jackpot)}
          </output>
          <input
            id={sliderId}
            type="range"
            min={ev.slider.min_gbp}
            max={ev.slider.max_gbp}
            step={ev.slider.step_gbp}
            value={jackpot}
            onChange={(e) => setJackpot(Number(e.target.value))}
            aria-valuetext={readout}
          />

          <dl className="results" aria-live="polite">
            <div data-positive={evOrdinary >= 0}>
              <dt>Ordinary draw</dt>
              <dd className="num">{gbpPence(evOrdinary)}</dd>
              <dd className="small quiet">
                {count(ordinary.tickets_sold)} lines sold &middot; breaks even at{' '}
                {gbp(breakEven(ordinary))}
              </dd>
            </div>
            <div data-positive={evMbw >= 0}>
              <dt>Must-Be-Won roll-down</dt>
              <dd className="num">{gbpPence(evMbw)}</dd>
              <dd className="small quiet">
                {count(mbw.tickets_sold)} lines sold &middot; breaks even at{' '}
                {gbp(breakEven(mbw))}
              </dd>
            </div>
          </dl>

          <p className="verdict-line small" data-positive={anyPositive}>
            {anyPositive
              ? 'Above zero: the ticket is worth more than it costs.'
              : 'Both below zero. Every pound spent here is a donation.'}
          </p>
        </div>
      </div>

      <div className="workbench-notes prose small quiet">
        <p>
          Two things carry most of the difference. A Must-Be-Won draw must pay its jackpot
          out, so an unclaimed pool rolls down into the lower tiers and every ticket takes
          a slice — worth far more than a share of a jackpot almost nobody wins. And more
          people play those draws, which dilutes the slice: the figures here assume{' '}
          {count(mbw.tickets_sold)} lines against an ordinary draw&rsquo;s{' '}
          {count(ordinary.tickets_sold)}.
        </p>
        <p>
          Fixed prizes are {ev.fixed_prizes.source}, and both regimes are priced for the
          draw of {longDate(asOf)}.
        </p>
      </div>
    </section>
  );
}
