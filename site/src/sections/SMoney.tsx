/**
 * Panel E - what actually happened.
 *
 * The only section where somebody loses real money, and it stays because a
 * page that argues about expected value and then shows no results is asking to
 * be taken on trust.
 *
 * The draw beside it makes the case for unpopular lines better than any model
 * on this page: the jackpot was won by two tickets and split. One of them
 * playing numbers nobody else picks would have taken the whole thing.
 *
 * Totals only. The ledger stays local by design; the lines it holds are the
 * wheel's, and they are already shown in full.
 */

import { count, gbp, gbpPence, longDate, percent } from '@/data/format';
import type { LastDraw, Ledger } from '@/data/types';

export function SMoney({
  ledger,
  lastDraw,
  verdictWas,
}: {
  ledger: Ledger | null;
  lastDraw: LastDraw;
  verdictWas: string;
}) {
  return (
    <section id="panel-f" className="money" aria-labelledby="panel-f-title">
      <hr className="perf" />
      <div className="money-head">
        <p className="eyebrow">Panel F &middot; real money</p>
        <h2 className="h-section" id="panel-f-title">
          What it cost to find out
        </h2>
      </div>

      <div className="money-body">
        {ledger && (
          <div className="ledger">
            <p className="lede prose">
              The model said <strong>{verdictWas}</strong> for the draw of{' '}
              {longDate(ledger.last_draw_date)}. A ticket went in anyway — ten lines,
              because a claim about expected value that has never been tested with money
              is just arithmetic.
            </p>

            <dl className="ledger-figures">
              <div>
                <dt>Staked</dt>
                <dd className="num">{gbp(ledger.spent_gbp)}</dd>
              </div>
              <div>
                <dt>Returned</dt>
                <dd className="num">{gbp(ledger.won_gbp)}</dd>
              </div>
              <div data-loss={ledger.net_gbp < 0}>
                <dt>Net</dt>
                <dd className="num">{gbpPence(ledger.net_gbp)}</dd>
              </div>
              <div data-loss={(ledger.roi ?? 0) < 0}>
                <dt>Return</dt>
                <dd className="num">
                  {ledger.roi === null ? '—' : percent(ledger.roi)}
                </dd>
              </div>
            </dl>

            <p className="prose small quiet">
              Of {ledger.lines} lines,{' '}
              {Object.entries(ledger.match_histogram)
                .sort(([a], [b]) => Number(b) - Number(a))
                .map(([matched, lines]) => `${lines} matched ${matched}`)
                .join(', ')}
              . Nothing above two numbers, in either round. That is the ordinary outcome,
              and the reason the verdict is almost always {verdictWas}.
            </p>
          </div>
        )}

        <aside className="split-story">
          <p className="eyebrow">The same night</p>
          <p className="split-numbers" aria-label="Winning numbers">
            {lastDraw.numbers.map((n) => (
              <span className="slip-ball num" key={n}>
                {n}
              </span>
            ))}
            {lastDraw.bonus !== null && (
              <span className="slip-ball slip-ball-bonus num">{lastDraw.bonus}</span>
            )}
          </p>

          {lastDraw.was_shared ? (
            <>
              <p className="split-headline">
                <strong className="num">{count(lastDraw.jackpot_winners)}</strong> tickets
                matched all six.
              </p>
              <p className="prose">
                They split {gbp(lastDraw.jackpot_total_gbp)} — {' '}
                <strong className="num">{gbp(lastDraw.jackpot_per_winner_gbp)}</strong>{' '}
                each. Either one of them holding a line nobody else plays would have taken
                the whole pool, on exactly the same odds.
              </p>
              <p className="small quiet">
                This is the argument in panel A, made by the draw rather than by a model.
              </p>
            </>
          ) : (
            <>
              <p className="split-headline">
                {lastDraw.jackpot_winners === 0
                  ? 'Nobody matched all six.'
                  : 'One ticket matched all six.'}
              </p>
              <p className="prose">
                {lastDraw.jackpot_winners === 0
                  ? 'The jackpot rolls into the next draw, which is how a Must-Be-Won builds in the first place.'
                  : `They took ${gbp(lastDraw.jackpot_total_gbp)} without splitting it — which is what playing an unpopular line is for.`}
              </p>
            </>
          )}
        </aside>
      </div>
    </section>
  );
}
