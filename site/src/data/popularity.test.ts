/**
 * The browser's copy of the popularity model, checked against Python's.
 *
 * This is the only thing standing between a recalibration in lottery/ev.py and
 * a page that quietly keeps scoring lines by last month's model. The fixtures
 * are written by scripts/export_site_data.py, so the two move together or the
 * suite goes red.
 */

import { describe, expect, it } from 'vitest';

import golden from '@/__fixtures__/popularity-golden.json';
import { generatePortfolio } from '@/data/generator';
import { hasConsecutiveRun, isArithmetic, numberWeight, popularityRatio } from '@/data/popularity';
import { popularity } from '@/data/siteData';

const model = popularity.model;
const bands = popularity.installed_step;

describe('popularityRatio matches Python', () => {
  it.each(golden.cases.map((c) => [c.line.join('-'), c.line, c.ratio] as const))(
    '%s',
    (_label, line, expected) => {
      expect(popularityRatio(line, model, bands)).toBeCloseTo(expected, 9);
    },
  );

  it('uses the same per-number weights', () => {
    golden.weights.forEach((weight, index) => {
      expect(numberWeight(index + 1, bands)).toBe(weight);
    });
  });

  it('reads its constants from the snapshot, not from a copy', () => {
    expect(model.mean_weight).toBeCloseTo(golden.model.mean_weight, 12);
    expect(model.normalization).toBeCloseTo(golden.model.normalization, 12);
  });
});

describe('pattern detection', () => {
  it('spots arithmetic lines', () => {
    expect(isArithmetic([1, 2, 3, 4, 5, 6])).toBe(true);
    expect(isArithmetic([5, 10, 15, 20, 25, 30])).toBe(true);
    expect(isArithmetic([1, 2, 3, 20, 40, 55])).toBe(false);
  });

  it('spots a run of three but not a mere pair', () => {
    expect(hasConsecutiveRun([1, 2, 3, 20, 40, 55], 3)).toBe(true);
    expect(hasConsecutiveRun([1, 2, 4, 20, 40, 55], 3)).toBe(false);
  });

  it('is order independent', () => {
    expect(popularityRatio([6, 1, 4, 2, 5, 3], model, bands)).toBeCloseTo(
      popularityRatio([1, 2, 3, 4, 5, 6], model, bands),
      9,
    );
  });
});

describe('the generator', () => {
  const balls = 59;

  it('returns the asked-for number of valid lines', () => {
    const lines = generatePortfolio(5, 20260812, model, bands, balls);
    expect(lines).toHaveLength(5);
    for (const { line } of lines) {
      expect(line).toHaveLength(6);
      expect(new Set(line).size).toBe(6);
      expect(Math.min(...line)).toBeGreaterThanOrEqual(1);
      expect(Math.max(...line)).toBeLessThanOrEqual(balls);
      expect([...line].sort((a, b) => a - b)).toEqual(line);
    }
  });

  it('honours the diversity constraints', () => {
    const lines = generatePortfolio(5, 7, model, bands, balls);
    for (const { line } of lines) {
      expect(line.filter((n) => n > 31).length).toBeGreaterThanOrEqual(2);
      const sum = line.reduce((a, b) => a + b, 0);
      expect(sum).toBeGreaterThanOrEqual(100);
      expect(sum).toBeLessThanOrEqual(260);
      expect(hasConsecutiveRun(line, 3)).toBe(false);
    }
    for (let i = 0; i < lines.length; i += 1) {
      for (let j = i + 1; j < lines.length; j += 1) {
        const shared = lines[i]!.line.filter((n) => lines[j]!.line.includes(n));
        expect(shared.length).toBeLessThanOrEqual(2);
      }
    }
  });

  it('produces lines far less played than an average one', () => {
    const lines = generatePortfolio(5, 99, model, bands, balls);
    for (const { ratio } of lines) {
      expect(ratio).toBeLessThan(0.6);
    }
  });

  it('is deterministic for a seed, and varies between seeds', () => {
    const a = generatePortfolio(5, 42, model, bands, balls);
    const b = generatePortfolio(5, 42, model, bands, balls);
    const c = generatePortfolio(5, 43, model, bands, balls);
    expect(a).toEqual(b);
    expect(a).not.toEqual(c);
  });

  it('reports the ratio it actually scored the line with', () => {
    for (const { line, ratio } of generatePortfolio(5, 5, model, bands, balls)) {
      expect(popularityRatio(line, model, bands)).toBeCloseTo(ratio, 12);
    }
  });
});
