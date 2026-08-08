/**
 * Axes, drawn by React.
 *
 * d3-axis mutates a DOM node it owns, which is the one thing this codebase
 * does not let D3 do. `scale.ticks()` is the useful half of it and returns
 * plain numbers, so the marks are ordinary JSX and React keeps control of the
 * tree.
 */

import type { LinearScale } from '@/charts/scale';

type Scale = LinearScale;

export function AxisBottom({
  scale,
  y,
  format,
  ticks = 5,
}: {
  scale: Scale;
  y: number;
  format: (value: number) => string;
  ticks?: number;
}) {
  return (
    <g className="axis" aria-hidden="true">
      {scale.ticks(ticks).map((tick) => (
        <text key={tick} x={scale(tick)} y={y} textAnchor="middle">
          {format(tick)}
        </text>
      ))}
    </g>
  );
}

export function AxisLeft({
  scale,
  x,
  format,
  ticks = 4,
  gridTo,
}: {
  scale: Scale;
  x: number;
  format: (value: number) => string;
  ticks?: number;
  /** When set, each tick also draws a hairline across to this x. */
  gridTo?: number;
}) {
  return (
    <g aria-hidden="true">
      {scale.ticks(ticks).map((tick) => (
        <g key={tick}>
          {gridTo !== undefined && (
            <line className="grid" x1={x} y1={scale(tick)} x2={gridTo} y2={scale(tick)} />
          )}
          <text className="axis" x={x - 8} y={scale(tick) + 4} textAnchor="end">
            {format(tick)}
          </text>
        </g>
      ))}
    </g>
  );
}
