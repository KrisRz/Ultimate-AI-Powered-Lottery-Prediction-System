/**
 * Expected value of one line, in the browser.
 *
 * `line_ev` in lottery/ev.py is exactly affine in the jackpot: the pool enters
 * only through the co-winner share, which depends on how many tickets are
 * sold rather than on how big the prize is, and through the roll-down term
 * J/N. So the whole model collapses to two numbers per regime, computed in
 * Python and shipped in the snapshot.
 *
 * That is why the slider can be exact instead of interpolating between sampled
 * points, and why there is no way for the chart to drift from the model behind
 * it. Verified to 8.9e-16 across the slider's range by
 * tests/test_export_site_data.py.
 */

import type { Regime, SalesBandPoint } from './types';

type Affine = Pick<Regime, 'a' | 'b'>;

/** EV in pounds of a single line at this jackpot. Negative almost always. */
export function evAt(regime: Affine, jackpot: number): number {
  return regime.a + regime.b * jackpot;
}

/** The jackpot at which the line stops losing money. */
export function breakEven(regime: Affine): number {
  return -regime.a / regime.b;
}

export type { Affine, SalesBandPoint };
