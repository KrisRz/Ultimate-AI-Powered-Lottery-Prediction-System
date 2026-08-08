/**
 * Running average matched, draw by draw, for every method at once.
 *
 * The p-values in the forest plot are the rigorous statement; this is the one
 * that convinces. Five lines start noisy, converge, and stay tangled on top of
 * the no-skill mean for nine years. There is nothing to pick between them, and
 * you can see it without reading a number.
 *
 * Geometry follows the operator dashboard's own chart
 * (scripts/dashboard.py:107) so the two read as the same instrument.
 */

import { AxisBottom, AxisLeft } from '@/charts/axis';
import { linearScale, polyline } from '@/charts/scale';
import type { Backtest } from '@/data/types';

const W = 640;
const H = 300;
const PAD = { top: 18, right: 74, bottom: 30, left: 44 };

/** Cumulative averages are smooth, so sampling one point in three costs the
 *  reader nothing and the payload a great deal. It is a sample, not an
 *  aggregate: no value is invented or averaged away. */
const STRIDE = 3;

export function ScoreLines({ backtest }: { backtest: Backtest }) {
  const { dates, cumulative_avg: series } = backtest.series;
  const names = Object.keys(series).sort();
  const expected = backtest.expected_random_avg;

  const x = linearScale([0, dates.length - 1], [PAD.left, W - PAD.right]);

  // Fixed window rather than the data's own extent: the first few draws swing
  // between 0 and 2 and would flatten everything that follows into a hairline.
  const y = linearScale([0.3, 0.95], [H - PAD.bottom, PAD.top]);

  const yearAt = (i: number) => (dates[i] ?? '').slice(0, 4);

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
        {/* The running average over the first handful of draws swings between
            0 and 2 and lands well outside the plotted window. The axis labels
            need to escape the frame, so the SVG allows overflow; the series
            must not, or those early excursions scribble over the text beside
            the chart. */}
        <defs>
          <clipPath id="scorelines-plot">
            <rect
              x={PAD.left}
              y={PAD.top}
              width={W - PAD.right - PAD.left}
              height={H - PAD.bottom - PAD.top}
            />
          </clipPath>
        </defs>

        <AxisLeft scale={y} x={PAD.left} format={(v) => v.toFixed(1)} ticks={4} gridTo={W - PAD.right} />

        <line
          className="rule-expected"
          x1={PAD.left}
          y1={y(expected)}
          x2={W - PAD.right}
          y2={y(expected)}
        />
        <text className="axis" x={W - PAD.right + 6} y={y(expected) + 4}>
          no skill
        </text>

        <g clipPath="url(#scorelines-plot)">
          {names.map((name) => {
            const values = series[name];
            if (!values) return null;
            const sampled = values.filter((_, i) => i % STRIDE === 0);
            const d = polyline(sampled, (i) => x(i * STRIDE), y);
            if (!d) return null;
            return (
              <path
                key={name}
                d={d}
                className={name === 'random' ? 'score-line score-line-random' : 'score-line'}
              />
            );
          })}
        </g>

        <AxisBottom scale={x} y={H - 8} format={(i) => yearAt(Math.round(i))} ticks={6} />
      </svg>

      <figcaption className="small quiet">
        Running average of numbers matched, {dates[0]} to {dates[dates.length - 1]}. Four
        prediction methods and a random baseline, drawn identically because that is the
        finding.
      </figcaption>
    </figure>
  );
}
