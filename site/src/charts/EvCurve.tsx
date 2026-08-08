/**
 * What a line is worth, against how big the jackpot is.
 *
 * Two straight lines, because the model is affine - not a simplification of a
 * curve, the actual shape. Where each one crosses zero is the only number in
 * this section that matters, so the crossings are labelled and everything else
 * is quiet.
 *
 * The accent marks the region above zero: the rare draw where buying a ticket
 * is arithmetically defensible. It is the one place on this page where the
 * colour means "go".
 */

import { linearScale } from '@/charts/scale';
import { AxisBottom, AxisLeft } from '@/charts/axis';
import { evAt } from '@/data/ev';
import { gbpShort, gbpPence } from '@/data/format';
import type { Ev } from '@/data/types';

const W = 640;
const H = 330;
const PAD = { top: 22, right: 118, bottom: 34, left: 52 };

export function EvCurve({ ev, jackpot }: { ev: Ev; jackpot: number }) {
  const { min_gbp: min, max_gbp: max } = ev.slider;
  const regimes = ev.regimes;

  const x = linearScale([min, max], [PAD.left, W - PAD.right]);

  const values = regimes.flatMap((r) => [evAt(r, min), evAt(r, max)]);
  const lo = Math.min(...values, 0);
  const hi = Math.max(...values, 0);
  const margin = (hi - lo) * 0.1;
  const y = linearScale([lo - margin, hi + margin], [H - PAD.bottom, PAD.top]);

  const zero = y(0);

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
        {/* Everything above this band is a ticket worth buying. */}
        <rect
          className="ev-profit"
          x={PAD.left}
          y={PAD.top}
          width={W - PAD.right - PAD.left}
          height={Math.max(0, zero - PAD.top)}
        />

        <AxisLeft scale={y} x={PAD.left} format={(v) => gbpPence(v)} ticks={4} gridTo={W - PAD.right} />
        <line className="rule-zero" x1={PAD.left} y1={zero} x2={W - PAD.right} y2={zero} />

        {regimes.map((regime) => {
          const y1 = y(evAt(regime, min));
          const y2 = y(evAt(regime, max));
          const crossing = -regime.a / regime.b;
          const inRange = crossing >= min && crossing <= max;
          return (
            <g key={regime.key} className={`ev-regime ev-regime-${regime.key}`}>
              <line className="ev-line" x1={x(min)} y1={y1} x2={x(max)} y2={y2} />
              <text className="ev-label" x={W - PAD.right + 8} y={y2 + 4}>
                {regime.key === 'mbw' ? 'Must-Be-Won' : 'ordinary draw'}
              </text>
              {inRange && (
                <>
                  <circle className="ev-crossing" cx={x(crossing)} cy={zero} r={4.5} />
                  <text className="ev-crossing-label" x={x(crossing)} y={zero + 20} textAnchor="middle">
                    {gbpShort(crossing)}
                  </text>
                </>
              )}
            </g>
          );
        })}

        <line className="ev-cursor" x1={x(jackpot)} y1={PAD.top} x2={x(jackpot)} y2={H - PAD.bottom} />

        <AxisBottom scale={x} y={H - 10} format={(v) => gbpShort(v)} ticks={5} />
      </svg>
    </figure>
  );
}
