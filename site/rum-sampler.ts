// Pure slow-frame decision logic — no DOM, no rAF, no SDK. `rum-runtime.ts` drives this with real
// `requestAnimationFrame` timestamps and reports through `DD_RUM.addDurationVital`; this module is
// unit-testable on its own (`rum-sampler.test.ts`).

/** Raw rAF-delta threshold (ms) — the web platform's long-frame boundary (Long Tasks / LoAF),
 * cross-device stable, unlike a vsync-derived (k × median interval) threshold. */
export const SLOW_FRAME_THRESHOLD_MS = 50;

/** Rolling-window size (frame count) for the median frame interval reported as context. */
const MEDIAN_WINDOW = 20;

export interface SlowFrameContext {
    /** Duration of the slow frame, ms. */
    duration: number;
    /** Time since page load the frame ended, ms — separates the known startup compile stall from
     * steady-state jank. */
    msSinceLoad: number;
    /** Median of the last `MEDIAN_WINDOW` frame intervals *before* this frame — the steady-state
     * pacing this frame deviated from. */
    rollingMedianIntervalMs: number;
    /** Frames observed so far, including this one. */
    framesObserved: number;
}

export interface SlowFrameReport {
    startTime: number;
    duration: number;
    context: SlowFrameContext;
}

export interface FrameSamplerState {
    lastTimestamp: number | null;
    /** Bounded rolling window of frame intervals, oldest first. */
    intervals: number[];
    frameCount: number;
}

export const initialFrameSamplerState: FrameSamplerState = {
    lastTimestamp: null,
    intervals: [],
    frameCount: 0,
};

export interface FrameSampleResult {
    state: FrameSamplerState;
    report: SlowFrameReport | null;
}

function median(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

/**
 * Advances the sampler by one rAF timestamp. `timestamp` is the high-resolution time (ms) the
 * frame callback fired at, on the same clock as `performance.now()` — so it doubles as
 * ms-since-load with no separate load-time input.
 *
 * @example
 * let state = initialFrameSamplerState;
 * const { state: next, report } = sampleFrame(state, performance.now());
 */
export function sampleFrame(state: FrameSamplerState, timestamp: number): FrameSampleResult {
    const frameCount = state.frameCount + 1;

    if (state.lastTimestamp === null) {
        return {
            state: { lastTimestamp: timestamp, intervals: state.intervals, frameCount },
            report: null,
        };
    }

    const delta = timestamp - state.lastTimestamp;
    const rollingMedianIntervalMs = median(state.intervals);
    const intervals = [...state.intervals, delta].slice(-MEDIAN_WINDOW);

    const report: SlowFrameReport | null =
        delta >= SLOW_FRAME_THRESHOLD_MS
            ? {
                  startTime: state.lastTimestamp,
                  duration: delta,
                  context: {
                      duration: delta,
                      msSinceLoad: timestamp,
                      rollingMedianIntervalMs,
                      framesObserved: frameCount,
                  },
              }
            : null;

    return { state: { lastTimestamp: timestamp, intervals, frameCount }, report };
}

/**
 * Clears `lastTimestamp` so the next `sampleFrame` call is treated as a first frame (never
 * reports — the existing first-frame rule). Call this on foreground return after a backgrounded
 * tab: browsers throttle or suspend `requestAnimationFrame` while a tab is hidden, so the next
 * delta after resume would otherwise span the whole backgrounded interval and read as a fake
 * multi-second slow frame. Intervals and frame count carry over unaffected — they describe
 * genuine steady-state pacing, not the gap.
 *
 * @example
 * document.addEventListener("visibilitychange", () => {
 *     if (!document.hidden) state = resetFrameSampler(state);
 * });
 */
export function resetFrameSampler(state: FrameSamplerState): FrameSamplerState {
    return { ...state, lastTimestamp: null };
}
