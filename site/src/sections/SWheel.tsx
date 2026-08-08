/**
 * Panel F - the wheel, and why it is not an edge.
 *
 * A covering design over twelve under-played numbers, with a guarantee that
 * was measured exhaustively rather than copied from a published table. It held
 * on every real draw where it could apply.
 *
 * And it makes no money. Same return as ten random lines, because nothing
 * changes the odds. What changes is the shape: wins arrive in fewer draws but
 * several at a time. Selling that as an advantage would be the exact dishonesty
 * this project exists to avoid, so the section says it plainly.
 */

import { count, gbp, percent } from '@/data/format';
import type { Wheel } from '@/data/types';

export function SWheel({ wheel }: { wheel: Wheel }) {
  const bt = wheel.backtest;
  const guaranteeFor4 = wheel.guarantees['4'];

  return (
    <section id="panel-e" className="wheel" aria-labelledby="panel-e-title">
      <hr className="perf" />
      <div className="wheel-head">
        <p className="eyebrow">Panel E &middot; the wheel</p>
        <h2 className="h-section" id="panel-e-title">
          Ten lines that cover each other
        </h2>
        <p className="lede prose">
          Pick the {wheel.pool_size} least-played numbers and arrange ten lines across
          them so that any four coming up guarantees at least{' '}
          {guaranteeFor4 !== undefined && <strong className="num">{guaranteeFor4}</strong>}{' '}
          of your lines match three. Not a better chance of winning — a promise about
          what happens when you do.
        </p>
      </div>

      <div className="wheel-body">
        <div>
          <p className="control-label">The pool</p>
          <ol className="slip-numbers pool-row" aria-label="Pool numbers">
            {wheel.pool.map((n) => (
              <li className="slip-ball num" key={n}>
                {n}
              </li>
            ))}
          </ol>

          <table className="coverage" aria-describedby="coverage-note">
            <caption className="visually-hidden">
              Which of the pool&rsquo;s numbers each of the ten lines carries
            </caption>
            <thead>
              <tr>
                <th scope="col" className="visually-hidden">
                  Line
                </th>
                {wheel.pool.map((n) => (
                  <th scope="col" key={n} className="num">
                    {n}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {wheel.lines.map((line, i) => (
                <tr key={i}>
                  <th scope="row" className="num">
                    {String(i + 1).padStart(2, '0')}
                  </th>
                  {wheel.pool.map((n) => (
                    <td key={n} data-on={line.includes(n)}>
                      <span className="visually-hidden">
                        {line.includes(n) ? 'in this line' : 'not in this line'}
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p id="coverage-note" className="small quiet">
            Every line is played by {percent(wheel.line_popularity)} as many people as an
            average one, and every line has the same expected value as every other.
          </p>
        </div>

        <aside className="wheel-verdict">
          <p className="eyebrow">Played over every draw since 2015</p>
          <dl className="ledger-figures">
            <div>
              <dt>Draws</dt>
              <dd className="num">{count(bt.draws)}</dd>
            </div>
            <div>
              <dt>Wins</dt>
              <dd className="num">{bt.hits_total}</dd>
            </div>
            <div data-loss>
              <dt>Returned</dt>
              <dd className="num">{percent(bt.return_pct ?? 0)}</dd>
            </div>
            <div>
              <dt>Guarantee broke</dt>
              <dd className="num">{bt.guarantee_violations}</dd>
            </div>
          </dl>

          <p className="prose">
            {gbp(bt.cash_gbp)} back from {gbp(bt.cost_gbp)} staked. That is what ten random
            lines return too — <strong>the wheel is not an edge</strong>. The guarantee never
            once failed, and the wins landed in only {bt.draws_with_win} of the{' '}
            {count(bt.draws)} draws, arriving several at a time instead of one at a time.
          </p>
          <p className="small quiet">
            Same money, lumpier. Worth having if you would rather win £40 four times than
            £10 sixteen times; worth nothing at all if you were hoping for an advantage.
          </p>
        </aside>
      </div>
    </section>
  );
}
