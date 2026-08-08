import { S1Hook } from '@/sections/S1Hook';
import { S2Predict } from '@/sections/S2Predict';
import { S3Ev } from '@/sections/S3Ev';
import { SGenerator } from '@/sections/SGenerator';
import { count, gbp, gbpPence, longDate } from '@/data/format';
import { backtest, ev, hook, popularity, snapshot } from '@/data/siteData';

/**
 * The masthead leads with what the toolkit gives you, not with what it refuses
 * to promise. Both are true; only one is a reason to keep reading.
 *
 * The verdict card stays because it is the honest half of the same offer -
 * knowing which draws to skip is most of the value, and on the days it says
 * PLAY it turns accent.
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
            Same odds.
            <br />
            Fewer people
            <br />
            to share with.
          </h1>
          <p className="lede masthead-lede">
            Nothing changes your chance of winning the lottery. Two things do change how
            much you take home: <strong>which draws are worth entering</strong>, and{' '}
            <strong>which numbers you put on the slip</strong>. This toolkit works out
            both, from real draw data, and checks its own answers against real money.
          </p>
        </div>

        <figure className="verdict" data-playing={playing}>
          <figcaption className="verdict-label eyebrow">
            Next draw &middot; {longDate(live.draw_date)}
          </figcaption>
          <p className="verdict-stamp">{live.verdict}</p>
          <dl className="verdict-facts small">
            <div>
              <dt>Jackpot</dt>
              <dd className="num">{gbp(live.jackpot_gbp)}</dd>
            </div>
            <div>
              <dt>A £2 line returns</dt>
              <dd className="num">{gbpPence(live.ev_best_line)}</dd>
            </div>
            <div>
              <dt>Worth playing above</dt>
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
 * The whole argument in five sentences, and what the page amounts to with
 * JavaScript switched off. It leads with the useful half.
 */
function Summary() {
  const mbw = ev.regimes.find((r) => r.key === 'mbw');
  const ordinary = ev.regimes.find((r) => r.key === 'ordinary');

  return (
    <section id="summary" className="summary" aria-labelledby="summary-title">
      <p className="eyebrow">In short</p>
      <h2 className="h-section" id="summary-title">
        How this works
      </h2>
      <div className="summary-cols prose">
        <p>
          <strong>Pick numbers nobody else picks.</strong> Prizes below the jackpot are
          fixed, but the jackpot is split between everyone holding the winning line — and
          people overwhelmingly play dates. A line avoiding 1&ndash;31 is picked by about{' '}
          <strong className="num">
            {(ev.reference_popularity * 100).toFixed(0)}%
          </strong>{' '}
          as many players as an average one, measured across{' '}
          {count(popularity.n_observations)} draw-rounds of real winner counts. Same odds,
          a much bigger share.
        </p>
        <p>
          <strong>Then play only the draws that pay.</strong> An ordinary draw needs a
          jackpot of{' '}
          {ordinary && <strong className="num">{gbp(ordinary.break_even_jackpot)}</strong>}{' '}
          before a {gbp(hook.ticket_price_gbp)} line is worth its price. A Must-Be-Won
          draw — where an unclaimed jackpot rolls down into the lower tiers — needs only{' '}
          {mbw && <strong className="num">{gbp(mbw.break_even_jackpot)}</strong>}. Those
          come round roughly nine times a year.
        </p>
        <p>
          <strong>And do not believe anyone who claims more.</strong> This project tested
          four prediction methods over {count(backtest.steps)} real draws and none of them
          beat random picking. That finding is at the bottom of this page, with the
          numbers behind it, because a tool that hides its own negative result is not
          worth trusting with the positive ones.
        </p>
        <p className="quiet small">
          Figures price the draw of {longDate(snapshot.as_of_draw_date)}, from data
          collected through {longDate(snapshot.data_through)}.
        </p>
      </div>
      <hr className="perf" />
    </section>
  );
}

export default function Page() {
  // Deterministic first render: the same slip on the server and in the browser,
  // and the same slip for everyone looking at a given draw. The button moves
  // off it from there.
  const seed = Number(snapshot.as_of_draw_date.replaceAll('-', ''));

  return (
    <main className="page">
      <Masthead />
      <Summary />
      <SGenerator popularity={popularity} hook={hook} ev={ev} seed={seed} />
      <S3Ev ev={ev} asOf={snapshot.as_of_draw_date} ticketPrice={hook.ticket_price_gbp} />
      <S1Hook hook={hook} ev={ev} />
      <S2Predict backtest={backtest} />
    </main>
  );
}
