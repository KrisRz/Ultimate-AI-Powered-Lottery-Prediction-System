/**
 * Panel D - where an unclaimed jackpot goes.
 *
 * The ribbons are sized by pounds, not by winner counts, because the
 * counter-intuitive part is the money: a fiver to each of nearly a million and
 * a half Match-2 winners eats almost half the pool before Match 3 sees any of
 * it. Winner counts would draw the opposite picture.
 *
 * Then the honest half. Roll-downs are the only draws worth entering, and
 * fewer than half of them clear break-even even so. The strip below plots
 * every one this archive holds, against zero.
 */

import { linearScale } from '@/charts/scale';
import { count, gbp, gbpPence, gbpShort, percent } from '@/data/format';
import type { Rolldown } from '@/data/types';

const W = 640;
const H = 250;
const STRIP_H = 96;

export function SRolldown({ rolldown }: { rolldown: Rolldown }) {
  const { split, history } = rolldown;
  const total = split.match_2_total_gbp + split.match_3_total_gbp;
  const m2Frac = total ? split.match_2_total_gbp / total : 0;

  // Ribbon geometry: one source, two sinks, widths in proportion to pounds.
  const srcX = 130;
  const dstX = 470;
  const barW = 26;
  const top = 26;
  const usable = H - top - 40;
  const m2H = Math.max(6, usable * m2Frac);
  const m3H = Math.max(6, usable * (1 - m2Frac));
  const gap = 22;

  const ribbon = (y0: number, h0: number, y1: number, h1: number) =>
    `M${srcX + barW},${y0} C${(srcX + dstX) / 2},${y0} ${(srcX + dstX) / 2},${y1} ${dstX},${y1}` +
    ` L${dstX},${y1 + h1} C${(srcX + dstX) / 2},${y1 + h1} ${(srcX + dstX) / 2},${y0 + h0} ${srcX + barW},${y0 + h0} Z`;

  const evs = history.draws.map((d) => d.ev);
  const lo = Math.min(...evs, -0.6);
  const hi = Math.max(...evs, 0.6);
  const x = linearScale([lo, hi], [40, W - 40]);

  return (
    <section id="panel-d" className="rolldown" aria-labelledby="panel-d-title">
      <hr className="perf" />
      <div className="rolldown-head">
        <p className="eyebrow">Panel D &middot; the one exception</p>
        <h2 className="h-section" id="panel-d-title">
          When the jackpot has to be paid out
        </h2>
        <p className="lede prose">
          After the jackpot rolls over {rolldown.rollover_cap} times it becomes Must-Be-Won:
          if nobody matches six, the whole pool cascades into the lower tiers instead of
          rolling again. Suddenly a {gbpPence(2)} line is claiming a share of eight figures
          for matching two numbers.
        </p>
      </div>

      <figure className="chart flow">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
          <rect className="flow-source" x={srcX} y={top} width={barW} height={usable} />
          <text className="flow-label" x={srcX - 10} y={top + usable / 2} textAnchor="end">
            {gbpShort(split.jackpot_gbp)}
          </text>
          <text className="axis" x={srcX - 10} y={top + usable / 2 + 15} textAnchor="end">
            unclaimed pool
          </text>

          <path className="flow-ribbon flow-ribbon-m2" d={ribbon(top, m2H, top, m2H)} />
          <path
            className="flow-ribbon flow-ribbon-m3"
            d={ribbon(top + m2H, m3H, top + m2H + gap, m3H)}
          />

          <rect className="flow-sink" x={dstX} y={top} width={barW} height={m2H} />
          <rect className="flow-sink" x={dstX} y={top + m2H + gap} width={barW} height={m3H} />

          <text className="flow-label" x={dstX + barW + 10} y={top + m2H / 2}>
            {gbpShort(split.match_2_total_gbp)}
          </text>
          <text className="axis" x={dstX + barW + 10} y={top + m2H / 2 + 15}>
            Match 2 &middot; {count(split.expected_match_2_winners)} winners
          </text>

          <text className="flow-label" x={dstX + barW + 10} y={top + m2H + gap + m3H / 2}>
            {gbpShort(split.match_3_total_gbp)}
          </text>
          <text className="axis" x={dstX + barW + 10} y={top + m2H + gap + m3H / 2 + 15}>
            Match 3 &middot; {count(split.expected_match_3_winners)} winners
          </text>
        </svg>
        <figcaption className="small quiet">
          {rolldown.rule}. Ribbons sized by pounds, at the {split.basis} of{' '}
          {gbp(split.jackpot_gbp)} — so a Match 2 pays{' '}
          {gbpPence(split.match_2_boost + 1)} instead of £1, and a Match 3 about{' '}
          {gbpPence(split.match_3_boost + 10)} instead of £10.
        </figcaption>
      </figure>

      <figure className="chart strip">
        <svg viewBox={`0 0 ${W} ${STRIP_H}`} role="img" aria-hidden="true">
          <line className="rule-zero" x1={x(0)} y1={16} x2={x(0)} y2={STRIP_H - 26} />
          <text className="axis" x={x(0)} y={12} textAnchor="middle">
            break even
          </text>
          {history.draws.map((draw) => (
            <circle
              key={draw.draw_number}
              className="strip-dot"
              data-positive={draw.ev >= 0}
              data-cap={draw.cap_driven}
              cx={x(draw.ev)}
              cy={STRIP_H - 46}
              r={4}
            />
          ))}
          <text className="axis" x={40} y={STRIP_H - 8}>
            {gbpPence(lo)} per line
          </text>
          <text className="axis" x={W - 40} y={STRIP_H - 8} textAnchor="end">
            {gbpPence(hi)}
          </text>
        </svg>
      </figure>

      <div className="rolldown-facts">
        <dl className="ledger-figures">
          <div>
            <dt>Roll-downs found</dt>
            <dd className="num">{history.detected}</dd>
          </div>
          <div>
            <dt>Driven by the cap</dt>
            <dd className="num">{history.cap_driven}</dd>
          </div>
          <div data-loss={history.positive_ev_share < 0.5}>
            <dt>Actually worth playing</dt>
            <dd className="num">{percent(history.positive_ev_share)}</dd>
          </div>
          <div data-loss={history.median_ev < 0}>
            <dt>Median line</dt>
            <dd className="num">{gbpPence(history.median_ev)}</dd>
          </div>
        </dl>
        <p className="prose small quiet">
          {history.detected} roll-downs between {history.window[0]} and {history.window[1]},
          about {history.per_year} a year. Only {history.positive_ev} of the{' '}
          {history.cap_driven} cap-driven ones clear break-even — the rest sold too many
          tickets for the pool to be worth chasing. {history.caveat}
        </p>
      </div>
    </section>
  );
}
