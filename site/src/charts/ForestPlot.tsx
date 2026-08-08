/**
 * Four methods and the baseline, each as an interval against the no-skill mean.
 *
 * Every series is drawn in the same grey on purpose. Colour-coding them would
 * imply they are alternatives worth choosing between; the finding is that they
 * are indistinguishable, and the picture should say so before the caption
 * does. The reader's eye lands on the dashed rule at 36/59 running straight
 * through every interval.
 */

import { linearScale } from '@/charts/scale';
import type { Backtest } from '@/data/types';

const W = 640;
const ROW = 46;
const PAD = { top: 34, right: 22, bottom: 30, left: 104 };

export function ForestPlot({ backtest }: { backtest: Backtest }) {
  const rows = backtest.methods;
  const height = PAD.top + rows.length * ROW + PAD.bottom;
  const expected = backtest.expected_random_avg;

  const lo = Math.min(...rows.map((r) => r.ci95[0]));
  const hi = Math.max(...rows.map((r) => r.ci95[1]));
  const pad = (hi - lo) * 0.12;

  const x = linearScale([lo - pad, hi + pad], [PAD.left, W - PAD.right]);

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-hidden="true">
        <line
          className="rule-expected"
          x1={x(expected)}
          y1={PAD.top - 14}
          x2={x(expected)}
          y2={height - PAD.bottom}
        />
        <text className="axis" x={x(expected)} y={PAD.top - 20} textAnchor="middle">
          no-skill mean {expected.toFixed(3)}
        </text>

        {rows.map((row, i) => {
          const y = PAD.top + i * ROW + ROW / 2;
          return (
            <g key={row.name} className={row.is_baseline ? 'forest-baseline' : undefined}>
              <text className="forest-label" x={PAD.left - 12} y={y + 4} textAnchor="end">
                {row.name}
              </text>
              <line className="forest-ci" x1={x(row.ci95[0])} y1={y} x2={x(row.ci95[1])} y2={y} />
              <line className="forest-cap" x1={x(row.ci95[0])} y1={y - 6} x2={x(row.ci95[0])} y2={y + 6} />
              <line className="forest-cap" x1={x(row.ci95[1])} y1={y - 6} x2={x(row.ci95[1])} y2={y + 6} />
              <circle className="forest-dot" cx={x(row.observed_avg)} cy={y} r={5} />
              <text className="forest-p" x={W - PAD.right} y={y - 12} textAnchor="end">
                p = {row.p_value_avg.toFixed(2)}
              </text>
            </g>
          );
        })}

        <text className="axis" x={PAD.left} y={height - 8}>
          average numbers matched per draw, with 95% interval
        </text>
      </svg>

      <table className="visually-hidden">
        <caption>
          Average numbers matched per draw over {backtest.steps} draws, with 95% confidence
          intervals and Monte-Carlo p-values against the no-skill mean of{' '}
          {expected.toFixed(4)}.
        </caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Average matched</th>
            <th scope="col">95% interval</th>
            <th scope="col">p-value</th>
            <th scope="col">Beats random</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <th scope="row">{row.name}</th>
              <td>{row.observed_avg.toFixed(4)}</td>
              <td>
                {row.ci95[0].toFixed(4)} to {row.ci95[1].toFixed(4)}
              </td>
              <td>{row.p_value_avg.toFixed(4)}</td>
              <td>{row.beats_random ? 'yes' : 'no'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
