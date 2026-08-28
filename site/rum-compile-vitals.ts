// Pure decision logic for the `pipeline_compile` RUM vital — no DOM, no `PerformanceObserver`, no
// SDK. `rum-runtime.ts` drives this with real `PerformanceObserver({ type: "measure" })` entries
// and reports through `DD_RUM.addDurationVital`; this module is unit-testable on its own
// (`rum-compile-vitals.test.ts`).
//
// The wire: `packages/shallot/src/engine/runtime/gpu.ts`'s `compileValidated` emits
// `performance.measure("<prefix><forcer.label>", { start, end })` beside the existing
// `Compute.precompiled?.(...)` call — a page-side RUM script can't reach a demo bundle's
// `Compute` singleton (each demo bundles the engine itself), so User Timing is the cross-bundle
// wire. `<prefix>` is `PIPELINE_COMPILE_MEASURE_PREFIX`, imported build-side only (see the
// module docblock on `../scripts/build-site.ts`'s `buildRumRuntimeBundle`) — never imported
// here, since a plain `import` of anything from `gpu.ts` pulls TypeGPU's whole module graph into
// a browser bundle through `gpu.ts`'s own top-level `tgpu.fn(...)` call (measured: 0.56 MB / 170
// modules for a probe importing nothing but the string constant). This module takes the prefix
// as a parameter instead, so it carries no import of its own.

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
