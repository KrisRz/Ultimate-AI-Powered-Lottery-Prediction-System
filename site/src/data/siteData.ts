/**
 * The snapshot, inlined at build time.
 *
 * There is no runtime fetch on purpose: the page is objects in S3 with no
 * backend, and a build-time import means the numbers cannot fail to load, go
 * stale mid-session, or disagree between two components on the same page.
 * Refreshing them is `make site-data` plus a commit, which is also what makes
 * the CI drift check possible.
 */

import raw from '../../public/data/site.json';
import type { SiteData } from './types';

export const site = raw as unknown as SiteData;

export const {
  snapshot,
  hook,
  backtest,
  ev,
  popularity,
  last_draw: lastDraw,
  ledger,
  rolldown,
  wheel,
  built,
} = site;
