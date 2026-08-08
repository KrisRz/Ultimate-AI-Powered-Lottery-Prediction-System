/**
 * How many other people are holding your line.
 *
 * A port of `popularity_ratio` from lottery/ev.py, because the generator runs
 * in the visitor's browser and there is no server to ask. Every constant is
 * read from the exported snapshot rather than written here, so recalibrating
 * the model in Python moves the page without anyone remembering to. The
 * arithmetic itself is pinned to Python by golden fixtures - see
 * popularity.test.ts.
 *
 * The number this returns is not a chance of winning. Draws are uniform and
 * every line is equally likely. It is how heavily a line is *played*: a ratio
 * of 0.3 means roughly a third as many other tickets carry it, so a jackpot
 * it wins is split fewer ways.
 */

import type { Popularity } from './types';

export type Model = Popularity['model'];
export type Bands = Popularity['installed_step'];

/** Relative pick-rate of one number, from the calibrated three-band fit. */
export function numberWeight(n: number, bands: Bands): number {
  for (const band of bands) {
    if (n <= band.to) return band.weight;
  }
  return bands[bands.length - 1]?.weight ?? 1;
}

export function isArithmetic(line: readonly number[]): boolean {
  const s = [...line].sort((a, b) => a - b);
  const first = s[1]! - s[0]!;
  return s.every((v, i) => i === 0 || v - s[i - 1]! === first);
}

export function hasConsecutiveRun(line: readonly number[], run: number): boolean {
  const s = [...line].sort((a, b) => a - b);
  let streak = 1;
  for (let i = 1; i < s.length; i += 1) {
    streak = s[i]! === s[i - 1]! + 1 ? streak + 1 : 1;
    if (streak >= run) return true;
  }
  return false;
}

/**
 * Relative to an average player's line: above 1 means more people hold it.
 *
 * Multiplier order follows the Python exactly - arithmetic *or* a consecutive
 * run, never both, then the all-birthday bonus on top of either.
 */
export function popularityRatio(
  line: readonly number[],
  model: Model,
  bands: Bands,
): number {
  let ratio = 1;
  for (const n of line) {
    ratio *= numberWeight(n, bands) / model.mean_weight;
  }

  if (isArithmetic(line)) {
    ratio *= model.arithmetic_mult;
  } else if (hasConsecutiveRun(line, model.consecutive_run)) {
    ratio *= model.consecutive_mult;
  }

  if (line.every((n) => n <= model.birthday_max)) {
    ratio *= model.birthday_mult;
  }

  return ratio / model.normalization;
}

/**
 * What fraction of a jackpot this line keeps, on average, if it wins.
 *
 * E[1/(1+K)] for K other winners Poisson-distributed with mean
 * tickets x popularity / combinations. This is the whole argument for playing
 * unpopular numbers, and the only place the choice of line changes the money.
 */
export function expectedShare(
  ratio: number,
  ticketsSold: number,
  totalCombinations: number,
): number {
  const lambda = (ticketsSold * ratio) / totalCombinations;
  if (lambda < 1e-12) return 1;
  return (1 - Math.exp(-lambda)) / lambda;
}
