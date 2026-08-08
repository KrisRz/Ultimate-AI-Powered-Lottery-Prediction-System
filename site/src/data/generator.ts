/**
 * Five lines of six numbers, chosen to be shared with as few people as
 * possible.
 *
 * A port of lottery/portfolio.py, with one simplification that is exact rather
 * than approximate. Python picks the highest-EV candidate; but the fixed
 * prizes are identical for every line, so EV differs only through the jackpot
 * share, and that share falls monotonically as popularity rises. Maximising EV
 * and minimising popularity are the same instruction, so this ranks by
 * popularity and never needs the EV model at all.
 *
 * The sampler's randomness deliberately does NOT match Python's. Reproducing
 * Mersenne Twister and CPython's weighted-choice internals would be a lot of
 * code to make two different machines agree on which arbitrary line to offer,
 * and nothing depends on that agreement. What must agree is the scoring, and
 * that is pinned by golden fixtures.
 */

import { hasConsecutiveRun, numberWeight, popularityRatio, type Bands, type Model } from './popularity';

/** Diversity constraints, matching lottery/portfolio.py. */
const N_PICK = 6;
const MAX_PAIRWISE_OVERLAP = 2;
const MIN_HIGH_NUMBERS = 2;
const SUM_BAND: readonly [number, number] = [100, 260];
const CANDIDATES_PER_LINE = 400;

export interface GeneratedLine {
  line: number[];
  ratio: number;
}

/** mulberry32 - small, fast, and good enough to shuffle a lottery slip. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function passesConstraints(line: readonly number[], balls: number, model: Model): boolean {
  return (
    new Set(line).size === N_PICK &&
    line.every((n) => n >= 1 && n <= balls) &&
    line.filter((n) => n > model.birthday_max).length >= MIN_HIGH_NUMBERS &&
    line.reduce((a, b) => a + b, 0) >= SUM_BAND[0] &&
    line.reduce((a, b) => a + b, 0) <= SUM_BAND[1] &&
    !hasConsecutiveRun(line, model.consecutive_run)
  );
}

/** Draw six numbers with the odds tilted towards the under-played ones. */
function sampleLine(next: () => number, balls: number, bands: Bands): number[] {
  const numbers = Array.from({ length: balls }, (_, i) => i + 1);
  const weights = numbers.map((n) => 1 / numberWeight(n, bands));
  const line: number[] = [];

  for (let pick = 0; pick < N_PICK; pick += 1) {
    const total = weights.reduce((a, b) => a + b, 0);
    let target = next() * total;
    let index = weights.length - 1;
    for (let i = 0; i < weights.length; i += 1) {
      target -= weights[i]!;
      if (target <= 0) {
        index = i;
        break;
      }
    }
    line.push(numbers[index]!);
    numbers.splice(index, 1);
    weights.splice(index, 1);
  }

  return line.sort((a, b) => a - b);
}

function overlap(a: readonly number[], b: readonly number[]): number {
  const set = new Set(a);
  return b.filter((n) => set.has(n)).length;
}

/**
 * Build `count` lines. Each is the least-played candidate that shares at most
 * two numbers with every line already chosen - so one unlucky draw cannot
 * write off the whole slip, and one lucky one cannot be wasted on near
 * duplicates.
 */
export function generatePortfolio(
  count: number,
  seed: number,
  model: Model,
  bands: Bands,
  balls: number,
): GeneratedLine[] {
  const next = rng(seed);
  const chosen: GeneratedLine[] = [];

  while (chosen.length < count) {
    let best: GeneratedLine | null = null;

    for (let attempt = 0; attempt < CANDIDATES_PER_LINE; attempt += 1) {
      const line = sampleLine(next, balls, bands);
      if (!passesConstraints(line, balls, model)) continue;
      if (chosen.some((c) => overlap(line, c.line) > MAX_PAIRWISE_OVERLAP)) continue;

      const ratio = popularityRatio(line, model, bands);
      if (!best || ratio < best.ratio) best = { line, ratio };
    }

    if (!best) {
      // Only reachable if the constraints are tightened past what the pool can
      // satisfy. Returning what we have beats throwing inside a render.
      break;
    }
    chosen.push(best);
  }

  return chosen;
}
