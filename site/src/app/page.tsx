import { S1Hook } from '@/sections/S1Hook';
import { S2Predict } from '@/sections/S2Predict';
import { S3Ev } from '@/sections/S3Ev';
import { count, gbp, gbpPence, longDate } from '@/data/format';
import { backtest, ev, hook, popularity, snapshot } from '@/data/siteData';

/**
 * The masthead states the thesis by showing the product's own output: on
 * almost every draw this toolkit says SKIP. That verdict is the most
 * characteristic thing the project produces, so it opens the page rather than
 * arriving as a conclusion two thousand words later.
 */
function Masthead() {
  const live = ev.live;
  const playing = live.verdict === 'PLAY';

  return (
    <header className="masthead">
      <div className="masthead-grid">
        <div>
          <p className="eyebrow">UK Lotto &middot; 6 from 59 &middot; two rounds per draw</p>
          <h1 className="masthead-title">
            Honest arithmetic
            <br />
            about a game
            <br />
            you cannot beat
          </h1>
          <p className="lede masthead-lede">
            This is a toolkit for a lottery that cannot be predicted. It does not
            pick numbers for you. It works out the only two things a player
            actually controls — whether a ticket is worth buying at all, and which
            line to put on it — and then checks its own answers against real
            draws and real money.
          </p>
        </div>

        <figure className="verdict" data-playing={playing}>
          <figcaption className="verdict-label eyebrow">
            Verdict for {longDate(live.draw_date)}
          </figcaption>
          <p className="verdict-stamp">{live.verdict}</p>
          <dl className="verdict-facts small">
            <div>
              <dt>Jackpot</dt>
              <dd className="num">{gbp(live.jackpot_gbp)}</dd>
            </div>
            <div>
              <dt>Best line is worth</dt>
              <dd className="num">{gbpPence(live.ev_best_line)}</dd>
            </div>
            <div>
              <dt>Breaks even at</dt>
              <dd className="num">{gbp(live.break_even_jackpot)}</dd>
            </div>
          </dl>
        </figure>
      </div>
      <hr className="perf" />
    </header>
  );
}

/**
 * The whole argument in eight sentences.
 *
 * This is where the skip link lands, and it is what the page amounts to with
 * JavaScript switched off. It is not a teaser - if someone reads only this,
 * they have the finding.
 */
function Summary() {
  const ordinary = ev.regimes.find((r) => r.key === 'ordinary');
  const mbw = ev.regimes.find((r) => r.key === 'mbw');

  return (
    <section id="summary" className="summary" aria-labelledby="summary-title">
      <p className="eyebrow">In short</p>
      <h2 className="h-section" id="summary-title">
        What this found
      </h2>
      <div className="summary-cols prose">
        <p>
          There are <strong className="num">{count(hook.total_combinations)}</strong>{' '}
          ways to pick six numbers from fifty-nine. The draw is uniform and
          independent, and this project&rsquo;s own walk-forward backtest says
          plainly that none of its prediction methods beats random picking.
        </p>
        <p>
          What is left is arithmetic. A {gbp(hook.ticket_price_gbp)} line on an
          ordinary draw is worth about{' '}
          {ordinary && (
            <strong className="num">
              {gbpPence(ordinary.a + ordinary.b * ev.live.jackpot_gbp)}
            </strong>
          )}{' '}
          — you are paying two pounds for less than one. On an ordinary draw the
          jackpot would have to reach{' '}
          {ordinary && <strong className="num">{gbp(ordinary.break_even_jackpot)}</strong>}{' '}
          before the sums work.
        </p>
        <p>
          The exception is a Must-Be-Won draw, where an unclaimed jackpot rolls
          down into the lower tiers. Those happen roughly nine times a year, and
          break even nearer{' '}
          {mbw && <strong className="num">{gbp(mbw.break_even_jackpot)}</strong>} — but
          fewer than half of them actually clear it.
        </p>
        <p>
          The second lever is which numbers. Prizes below the jackpot are fixed,
          but the jackpot is shared, and people overwhelmingly play dates. A line
          avoiding 1&ndash;31 splits a jackpot with far fewer people: the
          reference line this model uses is picked by{' '}
          <strong className="num">
            {(ev.reference_popularity * 100).toFixed(0)}%
          </strong>{' '}
          as many players as an average line, measured across{' '}
          {count(popularity.n_observations)} draw-rounds of real winner counts.
        </p>
        <p className="quiet small">
          Figures on this page price the draw of {longDate(snapshot.as_of_draw_date)},
          using draw data collected through {longDate(snapshot.data_through)}.
        </p>
      </div>
      <hr className="perf" />
    </section>
  );
}

export default function Page() {
  return (
    <main className="page">
      <Masthead />
      <Summary />
      <S1Hook hook={hook} ev={ev} />
      <S2Predict backtest={backtest} />
      <S3Ev ev={ev} asOf={snapshot.as_of_draw_date} ticketPrice={hook.ticket_price_gbp} />
    </main>
  );
}
