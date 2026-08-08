'use client';

/**
 * Scrollytelling primitives.
 *
 * Deliberately not scrollama. The useful part is ~40 lines of
 * IntersectionObserver, and a third-party scroll library brings its own
 * teardown to argue with React's double-invoked effects in StrictMode.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

/**
 * How far the viewport has travelled through `ref`, from 0 to 1.
 *
 * For a container holding a `position: sticky` graphic: 0 when its top meets
 * the viewport top, 1 when its bottom does. Driven by a scroll listener
 * coalesced into requestAnimationFrame rather than a threshold array on an
 * IntersectionObserver - an array of thresholds fires unevenly and reads as
 * jitter on a continuous value.
 */
export function useStickyProgress(
  ref: React.RefObject<HTMLElement | null>,
): number {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let frame = 0;

    const measure = () => {
      frame = 0;
      const rect = el.getBoundingClientRect();
      const travel = rect.height - window.innerHeight;
      if (travel <= 0) {
        setProgress(rect.top <= 0 ? 1 : 0);
        return;
      }
      const p = -rect.top / travel;
      setProgress(p < 0 ? 0 : p > 1 ? 1 : p);
    };

    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };

    // Through the frame, not straight away: reading layout during commit both
    // forces a synchronous reflow and sets state inside the effect body.
    schedule();
    window.addEventListener('scroll', schedule, { passive: true });
    // Bounds move when fonts land, the viewport rotates, or a step wraps to a
    // new line - all of which change `travel` without a scroll event.
    const observer = new ResizeObserver(schedule);
    observer.observe(el);

    return () => {
      if (frame) cancelAnimationFrame(frame);
      window.removeEventListener('scroll', schedule);
      observer.disconnect();
    };
  }, [ref]);

  return progress;
}

/**
 * Index of the step currently in the middle band of the viewport.
 *
 * One observer for all steps. The -45%/-45% root margin collapses the viewport
 * to a thin horizontal band across its middle, so exactly one step is active
 * at a time and the handover happens where the reader is actually looking.
 */
export function useActiveStep(): {
  active: number;
  stepRef: (index: number) => (el: HTMLElement | null) => void;
} {
  const [active, setActive] = useState(0);
  const nodes = useRef(new Map<number, HTMLElement>());

  const stepRef = useCallback(
    (index: number) => (el: HTMLElement | null) => {
      if (el) nodes.current.set(index, el);
      else nodes.current.delete(index);
    },
    [],
  );

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          for (const [index, node] of nodes.current) {
            if (node === entry.target) {
              setActive(index);
              break;
            }
          }
        }
      },
      { rootMargin: '-45% 0px -45% 0px', threshold: 0 },
    );

    for (const node of nodes.current.values()) observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return { active, stepRef };
}

/**
 * Whether the visitor asked for less motion.
 *
 * Read at runtime rather than left to CSS because the canvas animates outside
 * the style system: the shimmer is a rAF loop, and no media query will stop it.
 */
const REDUCED_MOTION = '(prefers-reduced-motion: reduce)';

function subscribeReducedMotion(onChange: () => void): () => void {
  const query = window.matchMedia(REDUCED_MOTION);
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeReducedMotion,
    () => window.matchMedia(REDUCED_MOTION).matches,
    // Server snapshot. The prerendered HTML carries no animation anyway, and
    // guessing "reduced" here would flash a static field at everyone else.
    () => false,
  );
}
