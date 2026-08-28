// Pure slow-frame/LoAF correlation logic — no DOM, no observer. `rum-runtime.ts` drives this with
// buffered `long-animation-frame` PerformanceObserver entries; unit-testable on its own
// (`rum-loaf.test.ts`).

/** A single `long-animation-frame` entry's script attribution, reduced to the fields the vital
 * surfaces. `sourceURL`/`sourceFunctionName`/`invoker` come straight off the PerformanceScriptTiming
 * spec shape. */
export interface LoAFScript {
    sourceURL: string;
    sourceFunctionName: string;
    invoker: string;
    duration: number;
}

/** A single `long-animation-frame` PerformanceObserver entry, rAF-relative (same clock as
 * `performance.now()`) — the clock a slow frame's `startTime`/`duration` are already reported on. */
export interface LoAFEntry {
    startTime: number;
    duration: number;
    blockingDuration: number;
    scripts: LoAFScript[];
}

export interface SlowFrameAttribution {
    /** False on an engine without LoAF (Safari) — every other field is a zero/empty placeholder. */
    loafSupported: boolean;
    /** How many LoAF entries overlap the slow frame's window. */
    loafEntryCount: number;
    /** Summed `duration` of the overlapping entries. */
    loafOverlapDurationMs: number;
    /** Summed `blockingDuration` of the overlapping entries. */
    loafOverlapBlockingMs: number;
    /** The longest-running script across every overlapping entry, or null when none overlap —
     * an idle main thread during a slow frame is a legitimate reading (GPU-present backpressure),
     * not a missing measurement. */
    topScript: LoAFScript | null;
}

const UNSUPPORTED_ATTRIBUTION: SlowFrameAttribution = {
    loafSupported: false,
    loafEntryCount: 0,
    loafOverlapDurationMs: 0,
    loafOverlapBlockingMs: 0,
    topScript: null,
};

/** The attribution context to attach when LoAF is unsupported on this engine. */
export function unsupportedLoafAttribution(): SlowFrameAttribution {
    return UNSUPPORTED_ATTRIBUTION;
}

/** An entry overlaps `[start, end)` when their spans share more than a single instant — an entry
 * ending exactly at `start`, or starting exactly at `end`, touches the window at one point and does
 * not overlap it; an entry starting at `start` or ending at `end` does. */
function overlaps(entry: LoAFEntry, start: number, end: number): boolean {
    return entry.startTime < end && entry.startTime + entry.duration > start;
}

/**
 * Correlates a reported slow frame's window (`[start, start + duration]`, rAF-relative ms) against
 * buffered `long-animation-frame` entries, returning the attribution context to attach to the
 * `slow_frame` vital.
 *
 * @example
 * const attribution = attributeSlowFrame(1000, 80, loafEntries);
 */
export function attributeSlowFrame(
    start: number,
    duration: number,
    entries: LoAFEntry[],
): SlowFrameAttribution {
    const end = start + duration;
    const overlapping = entries.filter((entry) => overlaps(entry, start, end));

    let loafOverlapDurationMs = 0;
    let loafOverlapBlockingMs = 0;
    let topScript: LoAFScript | null = null;

    for (const entry of overlapping) {
        loafOverlapDurationMs += entry.duration;
        loafOverlapBlockingMs += entry.blockingDuration;
        for (const script of entry.scripts) {
            if (!topScript || script.duration > topScript.duration) topScript = script;
        }
    }

    return {
        loafSupported: true,
        loafEntryCount: overlapping.length,
        loafOverlapDurationMs,
        loafOverlapBlockingMs,
        topScript,
    };
}
