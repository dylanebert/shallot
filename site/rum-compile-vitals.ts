// Pure decision logic for the `pipeline_compile` RUM vital — no DOM, no `PerformanceObserver`, no
// SDK. `rum-runtime.ts` drives this with real `PerformanceObserver({ type: "measure" })` entries
// and reports through `DD_RUM.addDurationVital`; this module is unit-testable on its own
// (`rum-compile-vitals.test.ts`).
//
// The wire: `packages/shallot/src/engine/runtime/gpu.ts`'s `reportCompile` emits
// `performance.measure("<prefix><forcer.label>", { start, end })` beside the existing
// `Compute.precompiled?.(...)` call — called from `compileValidated`'s serial per-forcer path and
// directly from `precompileAll`'s multi-member batch-then-bisect fast path, so it's the one emitter
// regardless of which drain path a forcer took. A page-side RUM
// script can't reach a demo bundle's `Compute` singleton (each demo bundles the engine itself), so
// User Timing is the cross-bundle wire. `<prefix>` is `PIPELINE_COMPILE_MEASURE_PREFIX`, imported
// build-side only (see the module docblock on `../scripts/build-site.ts`'s
// `buildRumRuntimeBundle`) — never imported here, since a plain `import` of anything from
// `gpu.ts` pulls TypeGPU's whole module graph into a browser bundle through `gpu.ts`'s own
// top-level `tgpu.fn(...)` call (measured: 0.56 MB / 170 modules for a probe importing nothing
// but the string constant). This module takes the prefix as a parameter instead, so it carries no
// import of its own.

export interface CompileVitalContext {
    /** The part of the measure entry's name after the prefix — the forcer's own label. */
    label: string;
}

export interface CompileVitalReport {
    /** The source measure entry's own name — the caller clears the shared `performance` timeline
     * by this name once the report has been forwarded, bounding the buffer that grows one entry
     * per `build()` call against a device (S1's review finding, owned by this stage). */
    name: string;
    /** Epoch ms (`performance.timeOrigin + startTime`) — `DD_RUM.addDurationVital`'s `startTime`
     * is a Unix epoch timestamp, not a `performance.now()`/rAF-relative one
     * (`@datadog/browser-rum-core`'s own `AddDurationVitalOptions.startTime` doc: "expects a
     * UNIX timestamp in milliseconds", the same lesson `rum-runtime.ts`'s `tick` already
     * documents for `slow_frame`). A raw relative `startTime` silently drops the vital — it never
     * reaches the intake batch, no console error, no exception; the SDK's own event-time
     * bookkeeping just reads it as ~1970. */
    startTime: number;
    duration: number;
    context: CompileVitalContext;
}

interface MeasureEntryLike {
    name: string;
    startTime: number;
    duration: number;
}

/**
 * Filters a batch of `performance` `measure` entries down to the ones carrying `prefix`,
 * converting each to an epoch-timed `pipeline_compile` vital report. A non-matching entry is
 * silently skipped — the shared `performance` timeline can carry `measure` entries from other
 * observers, so this is a filter, never an assertion that every entry matches.
 *
 * @example
 * const reports = compileVitalReports(list.getEntries(), PIPELINE_COMPILE_MEASURE_PREFIX, performance.timeOrigin);
 */
export function compileVitalReports(
    entries: readonly MeasureEntryLike[],
    prefix: string,
    timeOrigin: number,
): CompileVitalReport[] {
    const reports: CompileVitalReport[] = [];
    for (const entry of entries) {
        if (!entry.name.startsWith(prefix)) continue;
        reports.push({
            name: entry.name,
            startTime: timeOrigin + entry.startTime,
            duration: entry.duration,
            context: { label: entry.name.slice(prefix.length) },
        });
    }
    return reports;
}

/** The concurrency-ratio oracle: how much of the prefix-matching batch's wall-clock span its
 *  durations actually fill. `count` is the number of
 *  matching entries; `sum` is their total duration; `span` is `max(startTime + duration) -
 *  min(startTime)` across the matching set — the wall-clock window the whole batch occupies,
 *  never the window of any one entry; `ratio` is `sum / span`. `ratio ≈ 1.0` reads as fully
 *  serial (each entry's span is packed back-to-back with no gap and no overlap); `ratio > 1`
 *  reads as N-way overlap (durations summing to more than the window they fit in, so at least two
 *  entries ran concurrently). `ratio < 1` is possible too — a serial drain with idle gaps between
 *  compiles reads below 1, still correctly "not overlapping".
 *
 *  Degenerate case, decided and documented rather than left to divide-by-zero: an empty
 *  prefix-matching batch (`count === 0`) reads `{ count: 0, sum: 0, span: 0, ratio: 1 }` — no
 *  entries carry no evidence of either serialization or overlap, so the vacuous ratio is the
 *  fully-serial value (1), never `NaN`, so a caller comparing this ratio against a recorded
 *  baseline never has to special-case an empty run. */
export interface CompileConcurrencyRatio {
    count: number;
    sum: number;
    span: number;
    ratio: number;
}

export function compileConcurrencyRatio(
    entries: readonly MeasureEntryLike[],
    prefix: string,
): CompileConcurrencyRatio {
    const matches = entries.filter((entry) => entry.name.startsWith(prefix));
    const count = matches.length;
    if (count === 0) return { count: 0, sum: 0, span: 0, ratio: 1 };

    let sum = 0;
    let minStart = Number.POSITIVE_INFINITY;
    let maxEnd = Number.NEGATIVE_INFINITY;
    for (const entry of matches) {
        sum += entry.duration;
        if (entry.startTime < minStart) minStart = entry.startTime;
        const end = entry.startTime + entry.duration;
        if (end > maxEnd) maxEnd = end;
    }
    const span = maxEnd - minStart;
    // a batch whose matching entries all have zero duration and one identical start has span 0 —
    // sum/0 is Infinity/NaN, not a ratio. A zero-width span carries no overlap information either
    // way (there's no window to have overlapped within), so it reads as the same vacuous 1 the
    // zero-count case above uses. A single entry of nonzero duration is not this case: its span is
    // its own duration, so the ratio is a real 1.
    const ratio = span === 0 ? 1 : sum / span;
    return { count, sum, span, ratio };
}
