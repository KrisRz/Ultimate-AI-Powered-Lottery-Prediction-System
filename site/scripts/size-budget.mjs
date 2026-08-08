#!/usr/bin/env node
/**
 * First-load budget for the exported page.
 *
 * Two numbers, because they behave differently. The framework floor is React
 * plus the Next App Router runtime: it arrived with the stack the portfolio
 * already uses and no amount of care here moves it. Everything else is code
 * this project wrote or chose to install, and that is the number worth
 * defending - it is what stops `import * as d3` from landing in a page that
 * needs three of its modules.
 *
 * Run against out/ after a build: node scripts/size-budget.mjs
 */

import { gzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const OUT = 'out';
const TOTAL_BUDGET_KB = 200;
const APP_BUDGET_KB = 30;

// Chunks above this size are the framework; the app's own modules are small
// and many. Crude, but it is stable across builds and needs no bundler plugin.
const FRAMEWORK_CHUNK_KB = 30;

const html = readFileSync(join(OUT, 'index.html'), 'utf8');
const scripts = [...new Set([...html.matchAll(/src="(\/_next\/[^"]+\.js)"/g)].map((m) => m[1]))];

let framework = 0;
let app = 0;

for (const src of scripts) {
  const size = gzipSync(readFileSync(join(OUT, src.replace(/^\//, '')))).length / 1024;
  if (size >= FRAMEWORK_CHUNK_KB) framework += size;
  else app += size;
}

const total = framework + app;
const line = (label, value, budget) =>
  `${label.padEnd(22)} ${value.toFixed(1).padStart(7)} KB` +
  (budget ? `   budget ${budget} KB` : '');

console.log(line('framework (React/Next)', framework));
console.log(line('app code', app, APP_BUDGET_KB));
console.log(line('first-load total', total, TOTAL_BUDGET_KB));

const failures = [];
if (app > APP_BUDGET_KB) failures.push(`app code ${app.toFixed(1)} KB over ${APP_BUDGET_KB} KB`);
if (total > TOTAL_BUDGET_KB) failures.push(`total ${total.toFixed(1)} KB over ${TOTAL_BUDGET_KB} KB`);

if (failures.length) {
  console.error(`\nFirst-load budget exceeded:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
