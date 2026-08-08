/**
 * Panel H - how the thing runs.
 *
 * Hand-drawn SVG rather than a diagram library: the structure is static, so
 * React renders the boxes and CSS animates the edges, and nothing has to be
 * downloaded to draw eleven rectangles.
 *
 * The caveats are body copy, not footnotes. A page arguing for honest
 * arithmetic that then oversold its own uptime would refute itself, so the
 * cron drift and the silent-fallback trap sit in the same type as everything
 * else.
 */

import { count } from '@/data/format';
import type { Built } from '@/data/types';

const W = 640;
const H = 300;

/** Three lanes: what runs, where the data lives, what serves it. */
const LANES = [
  { x: 24, w: 176, label: 'GitHub Actions' },
  { x: 244, w: 152, label: 'Git as the datastore' },
  { x: 440, w: 176, label: 'AWS' },
] as const;

const BOXES = [
  { lane: 0, y: 66, h: 34, text: 'collect (×2 per draw)' },
  { lane: 0, y: 108, h: 34, text: 'watchdog' },
  { lane: 0, y: 150, h: 34, text: 'CI + drift check' },
  { lane: 0, y: 192, h: 34, text: 'deploy site' },
  { lane: 1, y: 96, h: 34, text: 'draw CSVs' },
  { lane: 1, y: 138, h: 34, text: 'site.json' },
  { lane: 2, y: 96, h: 34, text: 'S3 (private)' },
  { lane: 2, y: 138, h: 34, text: 'CloudFront' },
  { lane: 2, y: 180, h: 34, text: 'Route 53' },
] as const;

const EDGES = [
  { from: [200, 83], to: [244, 113] },
  { from: [200, 209], to: [244, 155] },
  { from: [396, 113], to: [440, 113] },
  { from: [396, 155], to: [440, 113] },
  { from: [516, 130], to: [516, 138] },
  { from: [516, 172], to: [516, 180] },
] as const;

export function SBuilt({ built }: { built: Built }) {
  return (
    <section id="panel-h" className="built" aria-labelledby="panel-h-title">
      <hr className="perf" />
      <div className="built-head">
        <p className="eyebrow">Panel H &middot; how it runs</p>
        <h2 className="h-section" id="panel-h-title">
          Nobody presses anything
        </h2>
        <p className="lede prose">
          Every figure on this page was collected, checked and published without a human
          in the loop. {count(built.tests.count)} tests across {built.tests.files} files
          gate every change, and the alerts stay silent unless something is worth reading.
        </p>
      </div>

      <div className="built-body">
        <figure className="chart arch">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden="true">
            {LANES.map((lane) => (
              <g key={lane.label}>
                <rect className="lane" x={lane.x} y={44} width={lane.w} height={228} />
                <text className="lane-label" x={lane.x + lane.w / 2} y={34} textAnchor="middle">
                  {lane.label}
                </text>
              </g>
            ))}

            {EDGES.map((edge, i) => (
              <path
                key={i}
                className="arch-edge"
                d={`M${edge.from[0]},${edge.from[1]} C${(edge.from[0] + edge.to[0]) / 2},${edge.from[1]} ${(edge.from[0] + edge.to[0]) / 2},${edge.to[1]} ${edge.to[0]},${edge.to[1]}`}
                style={{ animationDelay: `${i * 0.35}s` }}
              />
            ))}

            {BOXES.map((box) => {
              const lane = LANES[box.lane]!;
              return (
                <g key={box.text}>
                  <rect
                    className="arch-box"
                    x={lane.x + 12}
                    y={box.y}
                    width={lane.w - 24}
                    height={box.h}
                  />
                  <text
                    className="arch-text"
                    x={lane.x + lane.w / 2}
                    y={box.y + box.h / 2 + 4}
                    textAnchor="middle"
                  >
                    {box.text}
                  </text>
                </g>
              );
            })}
          </svg>
          <figcaption className="small quiet">
            No compute on AWS. It stores files and serves them; the lottery toolkit runs
            on GitHub Actions and a laptop, and the data lives in the repository.
          </figcaption>
        </figure>

        <div className="built-notes">
          <dl className="workflows">
            {built.workflows.map((flow) => (
              <div key={flow.file}>
                <dt>{flow.name}</dt>
                <dd className="small">{flow.does}</dd>
                <dd className="small quiet mono-note">{flow.schedule.join(' · ')}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <div className="caveats prose">
        <h3 className="h-step">Things that are not true of it</h3>
        <p>
          <strong>It is not punctual.</strong> {built.scheduler_caveat}.
        </p>
        <p>
          <strong>Success was not always success.</strong> {built.freshness_gate}.
        </p>
        <p>
          <strong>A missed run is survivable.</strong> {built.self_healing.how} — a window
          of about {built.self_healing.window_days} days.
        </p>
        <p className="quiet">
          Deployment assumes a role over OIDC, so no long-lived AWS key exists to leak.
          The bucket is private and reachable only through the CDN, and the role that
          publishes the site cannot change the infrastructure serving it.
        </p>
      </div>
    </section>
  );
}
