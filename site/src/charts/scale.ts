/**
 * The two pieces of charting maths this page actually needs.
 *
 * d3-scale was here first. It is excellent, but `scaleLinear` reaches through
 * d3-interpolate, d3-format, d3-time and d3-time-format, and nine packages
 * arrived to provide a linear mapping and a list of round numbers. The charts
 * here plot numbers on straight axes; a general grammar of graphics is not
 * what they are short of.
 *
 * If a future panel needs ordinal bands, colour interpolation or time axes,
 * take d3 back — that is the point at which it earns its weight.
 */

export interface LinearScale {
  (value: number): number;
  invert(pixel: number): number;
  /** Round values inside the domain, roughly `count` of them. */
  ticks(count?: number): number[];
}

/**
 * Round tick step, following the 1 / 2 / 5 / 10 progression everyone expects
 * from an axis. Same rule d3 applies, minus the machinery for other scale
 * types.
 */
function niceStep(span: number, count: number): number {
  if (span <= 0 || count <= 0) return 1;
  const rough = span / count;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const normalised = rough / magnitude;
  const factor = normalised >= 7.5 ? 10 : normalised >= 3.5 ? 5 : normalised >= 1.5 ? 2 : 1;
  return factor * magnitude;
}

export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): LinearScale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;

  const scale = ((value: number) =>
    span === 0 ? r0 : r0 + ((value - d0) / span) * (r1 - r0)) as LinearScale;

  scale.invert = (pixel: number) =>
    r1 === r0 ? d0 : d0 + ((pixel - r0) / (r1 - r0)) * span;

  scale.ticks = (count = 5) => {
    const lo = Math.min(d0, d1);
    const hi = Math.max(d0, d1);
    const step = niceStep(hi - lo, count);
    const out: number[] = [];
    // Walk by index rather than accumulating, so floating-point drift cannot
    // push the last tick outside the domain or duplicate one.
    const first = Math.ceil(lo / step);
    const last = Math.floor(hi / step);
    for (let i = first; i <= last; i += 1) {
      out.push(Number((i * step).toPrecision(12)));
    }
    return out;
  };

  return scale;
}

/**
 * An SVG path through the points, straight segments only. Returns null for
 * fewer than two points so callers can skip rendering rather than emit a
 * degenerate `d`.
 */
export function polyline(
  values: readonly number[],
  x: (index: number) => number,
  y: (value: number) => number,
): string | null {
  if (values.length < 2) return null;
  let d = '';
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (value === undefined) continue;
    d += `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(value).toFixed(2)}`;
  }
  return d;
}
