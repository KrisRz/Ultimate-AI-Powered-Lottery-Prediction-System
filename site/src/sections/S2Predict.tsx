'use client';

/**
 * Panel B - the premise, killed.
 *
 * This has to come before any of the arithmetic, because a page about a
 * lottery model that does not open by admitting the model cannot predict
 * anything is selling something. The graphic switches from the tangled running
 * averages to the intervals once the reader has seen the shape.
 */

import { useRef } from 'react';

import { ForestPlot } from '@/charts/ForestPlot';
import { ScoreLines } from '@/charts/ScoreLines';
import { count } from '@/data/format';
import type { Backtest } from '@/data/types';
import { useActiveStep } from '@/scroll/hooks';

export function S2Predict({ backtest }: { backtest: Backtest }) {
  const container = useRef<HTMLDivElement>(null);
  const { active, stepRef } = useActiveStep();

  const best = [...backtest.methods]
    .filter((m) => !m.is_baseline)
    .sort((a, b) => b.observed_avg - a.observed_avg)[0];
  const baseline = backtest.methods.find((m) => m.is_baseline);
  const methodCount = backtest.methods.filter((m) => !m.is_baseline).length;

  return (
    <section id="panel-b" aria-labelledby="panel-b-title">
      <hr className="perf" />
      <div className="scrolly" data-side="left" ref={container}>
        <div className="scrolly-graphic">
          {active >= 2 ? <ForestPlot backtest={backtest} /> : <ScoreLines backtest={backtest} />}
        </div>

        <div className="scrolly-steps">
          <p className="eyebrow">Panel B &middot; the premise</p>
          <h2 className="h-section" id="panel-b-title">
            No, you cannot predict them
          </h2>

          <div className="step" ref={stepRef(0)} data-active={active === 0}>
            <p className="lede">
              Every lottery system claims an edge. This one tested its own, on{' '}
              <strong className="num">{count(backtest.steps)}</strong> real draws, and
              found none.
            </p>
            <p className="prose small quiet">
              Walk-forward: each draw is predicted using only the{' '}
              {backtest.lookback} draws before it, so nothing is fitted on the answer.
            </p>
          </div>

          <div className="step" ref={stepRef(1)} data-active={active === 1}>
            <h3 className="h-step">{methodCount} methods, one picture</h3>
            <p className="prose">
              Frequency counting, weighted sampling, a probability map, and a consensus of
              all three. Their running averages converge on{' '}
              <span className="num">{backtest.expected_random_avg.toFixed(3)}</span> —
              which is just <span className="num">36/59</span>, what six random numbers
              match by arithmetic — and stay there for nine years.
            </p>
          </div>

          <div className="step" ref={stepRef(2)} data-active={active === 2}>
            <h3 className="h-step">Inside the noise, every one</h3>
            <p className="prose">
              Against {count(backtest.n_sim)} Monte-Carlo replicates of the no-skill
              model, the best method scores{' '}
              {best && <span className="num">{best.observed_avg.toFixed(4)}</span>} against
              a random baseline&rsquo;s{' '}
              {baseline && <span className="num">{baseline.observed_avg.toFixed(4)}</span>}.
              Every confidence interval straddles the no-skill mean.
            </p>
          </div>

          <div className="step" ref={stepRef(3)} data-active={active === 3}>
            <h3 className="h-step">Which leaves the arithmetic</h3>
            <p className="prose">
              Draws are independent and uniform, and no amount of history changes what
              comes out of the machine. So the rest of this page is not about predicting
              anything. It is about the two decisions that remain — and those turn out to
              be worth real money, just not often.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
