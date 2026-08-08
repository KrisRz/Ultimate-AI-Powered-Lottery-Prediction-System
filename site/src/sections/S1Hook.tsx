'use client';

/**
 * Panel A - the size of the problem.
 *
 * The field starts at ten marks, which you can count, and multiplies until it
 * is 45,057,474, which you cannot. Nothing here argues; it just shows the
 * denominator, because every honest claim later in the page is a fraction of
 * it.
 */

import { useEffect, useMemo, useRef } from 'react';

import { createDotfield } from '@/canvas/dotfield';
import { count, gbp, oneIn } from '@/data/format';
import type { Ev, Hook } from '@/data/types';
import { usePrefersReducedMotion, useActiveStep, useStickyProgress } from '@/scroll/hooks';

const START_AT = 10;

/** Quantised so the grain is rebuilt a bounded number of times per scroll. */
const SMOOTH_STEPS = 110;
const REDUCED_STEPS = 7;

export function S1Hook({ hook, ev }: { hook: Hook; ev: Ev }) {
  const container = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const progress = useStickyProgress(container);
  const reduced = usePrefersReducedMotion();
  const { active, stepRef } = useActiveStep();

  const total = hook.total_combinations;

  // Log-space, because the interesting part is the number of digits. Snapped
  // to steps: under reduced motion to seven discrete stops, otherwise to a
  // resolution fine enough to read as continuous without rebuilding the grain
  // on every frame.
  const shown = useMemo(() => {
    const steps = reduced ? REDUCED_STEPS : SMOOTH_STEPS;
    const q = Math.round(progress * steps) / steps;
    const span = Math.log(total) - Math.log(START_AT);
    return Math.round(Math.exp(Math.log(START_AT) + q * span));
  }, [progress, reduced, total]);

  // The canvas is React's element but its bitmap is not React's business:
  // one effect owns the field's lifetime, another pushes counts into it.
  const fieldRef = useRef<ReturnType<typeof createDotfield> | null>(null);

  useEffect(() => {
    const node = canvas.current;
    if (!node) return;
    fieldRef.current = createDotfield(node, { total, shimmer: !reduced });
    const theme = window.matchMedia('(prefers-color-scheme: dark)');
    const onTheme = () => fieldRef.current?.refreshTheme();
    theme.addEventListener('change', onTheme);
    return () => {
      theme.removeEventListener('change', onTheme);
      fieldRef.current?.destroy();
      fieldRef.current = null;
    };
  }, [total, reduced]);

  useEffect(() => {
    fieldRef.current?.render(shown);
  }, [shown]);

  const match2 = hook.odds.find((o) => o.key === 'match_2');
  const jackpotOdds = hook.odds.find((o) => o.key === 'jackpot');
  const coverEverything = total * hook.ticket_price_gbp;

  return (
    <section id="interlude" aria-labelledby="interlude-title">
      <div className="scrolly" ref={container}>
        <div className="scrolly-graphic">
          <figure className="field">
            <canvas ref={canvas} className="field-canvas" aria-hidden="true" />
            <figcaption className="field-readout">
              <span className="num field-count">{count(shown)}</span>
              <span className="field-label quiet small">
                {shown >= total ? 'ways to pick six numbers' : 'combinations so far'}
              </span>
            </figcaption>
          </figure>
        </div>

        <div className="scrolly-steps">
          <p className="eyebrow">Interlude &middot; the scale of it</p>
          <h2 className="h-section" id="interlude-title">
            Six numbers from fifty&#8209;nine
          </h2>

          <div className="step" ref={stepRef(0)} data-active={active === 0}>
            <p className="lede">
              Pick six. There are{' '}
              <strong className="num">{count(total)}</strong> ways to do it, and
              the draw picks one.
            </p>
            <p className="prose small quiet">
              That is the whole game. Everything on this page is a fraction with
              that number underneath it.
            </p>
          </div>

          <div className="step" ref={stepRef(1)} data-active={active === 1}>
            <h3 className="h-step">You cannot buy your way out</h3>
            <p className="prose">
              A line costs {gbp(hook.ticket_price_gbp)}. Covering every
              combination would cost{' '}
              <strong className="num">{gbp(coverEverything)}</strong> — far more
              than any jackpot this game has ever paid.
            </p>
          </div>

          <div className="step" ref={stepRef(2)} data-active={active === 2}>
            <h3 className="h-step">Almost every ticket is a loser</h3>
            <p className="prose">
              {jackpotOdds && (
                <>
                  The jackpot is <span className="num">{oneIn(jackpotOdds.one_in)}</span>.{' '}
                </>
              )}
              {match2 && (
                <>
                  The likeliest thing that happens to a ticket, other than
                  nothing, is matching two numbers for{' '}
                  {gbp(ev.fixed_prizes.match_2)} — and that is still only{' '}
                  <span className="num">{oneIn(match2.one_in)}</span>.
                </>
              )}
            </p>
          </div>

          <div className="step" ref={stepRef(3)} data-active={active === 3}>
            <h3 className="h-step">Which is why the sharing matters</h3>
            <p className="prose">
              Against a number this size, nothing you do moves your chance of
              winning. That is also why the two levers above are the whole game:
              they are not about winning more often, they are about what the win
              is worth and whether the ticket was priced fairly in the first
              place.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
