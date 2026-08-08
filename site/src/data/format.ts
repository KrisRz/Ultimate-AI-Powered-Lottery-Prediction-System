/**
 * Formatting. One place, because the same figure appears in a heading, a
 * screen-reader table and an aria-valuetext, and they must agree.
 *
 * Locale is pinned to en-GB rather than the visitor's: these are UK Lotto
 * figures in pounds, and a build-time-rendered string that then changes on
 * hydration is a React mismatch waiting to happen.
 */

const LOCALE = 'en-GB';

export const count = (n: number): string => n.toLocaleString(LOCALE);

/** Whole pounds: "£8,391,429". */
export const gbp = (n: number): string =>
  n.toLocaleString(LOCALE, {
    style: 'currency',
    currency: 'GBP',
    maximumFractionDigits: 0,
  });

/** Pounds and pence, signed: "−£0.36". Used for per-line EV, which is small. */
export const gbpPence = (n: number): string => {
  const sign = n < 0 ? '−' : '';
  return `${sign}£${Math.abs(n).toFixed(2)}`;
};

/** "£8.4m" / "£11.7m" for axis labels and tight headings. */
export const gbpShort = (n: number): string => {
  if (Math.abs(n) >= 1_000_000) return `£${(n / 1_000_000).toFixed(1)}m`;
  if (Math.abs(n) >= 1_000) return `£${Math.round(n / 1_000)}k`;
  return `£${n.toFixed(0)}`;
};

/** "1 in 45,057,474". */
export const oneIn = (n: number): string =>
  `1 in ${Math.round(n).toLocaleString(LOCALE)}`;

/** "×0.32" - popularity relative to an average player. */
export const times = (n: number): string => `×${n.toFixed(2)}`;

/** "8 August 2026" from an ISO date, without dragging in a date library. */
export const longDate = (iso: string): string =>
  new Date(`${iso}T00:00:00Z`).toLocaleDateString(LOCALE, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
