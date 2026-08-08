/**
 * The combination field: 45,057,474 marks, one of which wins.
 *
 * Two drawing regimes, because no technique covers eight orders of magnitude:
 *
 *   Grid   - while a cell is at least MIN_CELL_PX across, every combination is
 *            an actual drawn square. At ten or a hundred you can count them,
 *            which is the point: the reader starts somewhere comprehensible.
 *   Grain  - past that, a per-pixel hash fills an ImageData at CSS resolution
 *            and it gets scaled up. 45 million squares is not 45 million draw
 *            calls; it is one putImageData over ~400k pixels.
 *
 * The winner sits at a fixed fraction of the field the whole way through, so
 * the eye can hold onto it while everything around it multiplies.
 */

const MIN_CELL_PX = 2;
const MAX_DPR = 2; // a full-bleed canvas at DPR 3 is ~7M pixels and drops frames
const WINNER_AT: [number, number] = [0.618, 0.383];
const SHIMMER_PIXELS = 2200;

export interface Dotfield {
  /** Draw `count` combinations. Cheap to call on every frame. */
  render(count: number): void;
  /** Re-read colours after a theme change. */
  refreshTheme(): void;
  destroy(): void;
}

interface Palette {
  ink: string;
  inkRgb: [number, number, number];
  mark: string;
  paper: string;
}

function hash2(x: number, y: number, seed: number): number {
  let h = Math.imul(x, 374761393) + Math.imul(y, 668265263) + Math.imul(seed, 2246822519);
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

function parseRgb(colour: string): [number, number, number] {
  const match = colour.match(/(\d+(?:\.\d+)?)/g);
  if (!match || match.length < 3) return [20, 22, 26];
  return [Number(match[0]), Number(match[1]), Number(match[2])];
}

export function createDotfield(
  canvas: HTMLCanvasElement,
  options: { total: number; shimmer: boolean },
): Dotfield {
  const context = canvas.getContext('2d', { alpha: true });
  if (!context) {
    return { render: () => {}, refreshTheme: () => {}, destroy: () => {} };
  }
  const ctx = context;

  let width = 0;
  let height = 0;
  let dpr = 1;
  let palette = readPalette();
  let lastCount = -1;
  // The count most recently asked for, kept separately from the one last
  // drawn: on mount the canvas has no layout yet, so the first render is a
  // no-op and the field would stay blank until the number happened to change.
  // Resizing replays this instead.
  let requested = 1;
  let shimmerFrame = 0;
  let grain: HTMLCanvasElement | null = null;
  let grainCtx: CanvasRenderingContext2D | null = null;

  function readPalette(): Palette {
    const style = getComputedStyle(canvas);
    const ink = style.getPropertyValue('--ink').trim() || '#14161a';
    return {
      ink,
      inkRgb: parseRgb(
        // getComputedStyle on a custom property hands back the authored value,
        // which is hex. Resolve it to rgb via a throwaway paint.
        (() => {
          ctx.fillStyle = ink;
          const resolved = ctx.fillStyle as string;
          if (resolved.startsWith('#')) {
            const hex = resolved.slice(1);
            return `rgb(${parseInt(hex.slice(0, 2), 16)},${parseInt(
              hex.slice(2, 4),
              16,
            )},${parseInt(hex.slice(4, 6), 16)})`;
          }
          return resolved;
        })(),
      ),
      mark: style.getPropertyValue('--mark').trim() || '#d6006e',
      paper: style.getPropertyValue('--paper').trim() || '#edeee9',
    };
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    width = Math.max(1, Math.round(rect.width));
    height = Math.max(1, Math.round(rect.height));
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    grain = document.createElement('canvas');
    grain.width = width;
    grain.height = height;
    grainCtx = grain.getContext('2d');

    lastCount = -1;
    draw(requested);
  }

  /** The one combination that wins, as a dot. Grain regime only. */
  function drawWinnerDot() {
    const x = WINNER_AT[0] * width;
    const y = WINNER_AT[1] * height;

    const glow = ctx.createRadialGradient(x, y, 0, x, y, 16);
    glow.addColorStop(0, palette.mark);
    glow.addColorStop(1, 'transparent');
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.fill();

    ctx.globalAlpha = 1;
    ctx.fillStyle = palette.mark;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }

  /**
   * Combinations as playslip cells: an unplayed one is an empty box, and the
   * only filled box is the one that wins. Once the boxes are too small to have
   * an interior they become solid marks, which is the same picture at a
   * distance.
   */
  function renderGrid(count: number, cols: number, rows: number) {
    const cellW = width / cols;
    const cellH = height / rows;
    const gap = cellW > 10 ? Math.min(4, cellW * 0.14) : cellW > 4 ? 1 : 0.5;
    const w = Math.max(0.5, cellW - gap);
    const h = Math.max(0.5, cellH - gap);
    const outlined = cellW >= 11;

    // Which cell holds the winner - kept at the same fraction of the field at
    // every zoom level, so the eye can stay on it.
    const winner =
      Math.min(rows - 1, Math.floor(WINNER_AT[1] * rows)) * cols +
      Math.min(cols - 1, Math.floor(WINNER_AT[0] * cols));
    const winnerIndex = winner < count ? winner : count - 1;

    if (outlined) {
      ctx.strokeStyle = palette.ink;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      for (let i = 0; i < count; i += 1) {
        if (i === winnerIndex) continue;
        const col = i % cols;
        const row = (i - col) / cols;
        ctx.strokeRect(
          Math.round(col * cellW) + 0.5,
          Math.round(row * cellH) + 0.5,
          Math.round(w),
          Math.round(h),
        );
      }
    } else {
      ctx.fillStyle = palette.ink;
      ctx.globalAlpha = 0.78;
      for (let i = 0; i < count; i += 1) {
        if (i === winnerIndex) continue;
        const col = i % cols;
        const row = (i - col) / cols;
        ctx.fillRect(col * cellW, row * cellH, w, h);
      }
    }

    ctx.globalAlpha = 1;
    ctx.fillStyle = palette.mark;
    const wc = winnerIndex % cols;
    const wr = (winnerIndex - wc) / cols;
    ctx.fillRect(wc * cellW, wr * cellH, Math.max(w, 2), Math.max(h, 2));
  }

  function renderGrain(count: number, total: number, switchAt: number, seed: number) {
    if (!grain || !grainCtx) return;

    // Coverage picks up where the grid left off (a 1px mark on a 2px pitch is
    // a quarter of the area) and saturates as the count passes the pixel
    // budget - past that there is genuinely more than one combination per
    // pixel, and a solid field is the honest picture.
    const span = Math.log(total) - Math.log(switchAt);
    const t = span > 0 ? (Math.log(count) - Math.log(switchAt)) / span : 1;
    const coverage = Math.min(1, 0.25 + 0.75 * Math.max(0, t));

    const image = grainCtx.createImageData(width, height);
    const data = image.data;
    const [r, g, b] = palette.inkRgb;

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const i = (y * width + x) * 4;
        const v = hash2(x, y, seed);
        if (v < coverage) {
          data[i] = r;
          data[i + 1] = g;
          data[i + 2] = b;
          // Vary the ink so the field reads as print grain rather than a fill.
          data[i + 3] = 150 + Math.floor(hash2(x, y, seed + 977) * 105);
        }
      }
    }

    grainCtx.putImageData(image, 0, 0);
    ctx.drawImage(grain, 0, 0, width, height);
  }

  function draw(count: number) {
    if (!width || !height) return;
    const n = Math.max(1, Math.min(options.total, Math.round(count)));
    if (n === lastCount) return;
    lastCount = n;

    ctx.clearRect(0, 0, width, height);

    // Square cells at any canvas aspect: pick the column count from the area,
    // then take as many rows as it takes. Trailing cells simply go undrawn.
    const aspect = width / height;
    const cols = Math.max(1, Math.ceil(Math.sqrt(n * aspect)));
    const rows = Math.max(1, Math.ceil(cols / aspect));
    const cellW = width / cols;

    if (cellW >= MIN_CELL_PX) {
      renderGrid(n, cols, rows);
    } else {
      const switchAt = Math.max(2, Math.floor((width / MIN_CELL_PX) * (height / MIN_CELL_PX)));
      renderGrain(n, options.total, switchAt, 1);
      drawWinnerDot();
    }
  }

  function shimmer() {
    shimmerFrame = requestAnimationFrame(shimmer);
    if (!grain || !grainCtx || lastCount < 0) return;
    const cellW = width / Math.max(1, Math.ceil(Math.sqrt(lastCount * (width / height))));
    if (cellW >= MIN_CELL_PX) return; // only the grain regime shimmers

    // Repaint a scatter of pixels rather than the field: the churn reads as a
    // live surface and costs a couple of thousand operations a frame.
    const [r, g, b] = palette.inkRgb;
    for (let k = 0; k < SHIMMER_PIXELS; k += 1) {
      const x = Math.floor(Math.random() * width);
      const y = Math.floor(Math.random() * height);
      grainCtx.fillStyle = `rgba(${r},${g},${b},${0.35 + Math.random() * 0.5})`;
      grainCtx.fillRect(x, y, 1, 1);
    }
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(grain, 0, 0, width, height);
    drawWinnerDot();
  }

  const observer = new ResizeObserver(() => {
    resize();
  });
  observer.observe(canvas);
  resize();

  if (options.shimmer) shimmerFrame = requestAnimationFrame(shimmer);

  return {
    render(count: number) {
      requested = count;
      draw(count);
    },
    refreshTheme() {
      palette = readPalette();
      lastCount = -1;
      draw(requested);
    },
    destroy() {
      observer.disconnect();
      if (shimmerFrame) cancelAnimationFrame(shimmerFrame);
      grain = null;
      grainCtx = null;
    },
  };
}
