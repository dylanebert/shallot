// Browser-only rAF/SDK glue — drives `rum-sampler.ts`'s and `rum-compile-vitals.ts`'s pure
// decision logic with real `requestAnimationFrame` timestamps and `PerformanceObserver` entries,
// reporting through the Datadog RUM CDN snippet's global. Bundled by `scripts/build-site.ts`
// (`Bun.build`, target "browser") and inlined into every demo page; never imported by anything
// else, so it has no test of its own — the logic it drives is tested in `rum-sampler.test.ts` and
// `rum-compile-vitals.test.ts`.

import { compileVitalReports } from "./rum-compile-vitals";
import {
    attributeSlowFrame,
    type LoAFEntry,
    type SlowFrameAttribution,
    unsupportedLoafAttribution,
} from "./rum-loaf";
import { initialFrameSamplerState, resetFrameSampler, sampleFrame } from "./rum-sampler";

declare global {
    interface Window {
        DD_RUM?: {
            onReady: (callback: () => void) => void;
            addDurationVital: (
                name: string,
                options: { startTime: number; duration: number; context: Record<string, unknown> },
            ) => void;
        };
    }
}

// Inlined at build time (`scripts/build-site.ts`'s `buildRumRuntimeBundle`, `Bun.build`'s
// `define`) — never imported. `packages/shallot/src/engine/runtime/gpu.ts`'s
// `PIPELINE_COMPILE_MEASURE_PREFIX` can't be imported directly into this browser bundle: `gpu.ts`
// has a top-level `tgpu.fn(...)` call, a real side effect no bundler can tree-shake, so any import
// from it pulls TypeGPU's whole module graph in (measured 0.56 MB / 170 modules for a probe
// importing only the constant — `rum-compile-vitals.ts`'s own docblock records the same fact).
declare const __PIPELINE_COMPILE_MEASURE_PREFIX__: string;

// `PerformanceObserver({ type: "measure", buffered: true })` — `buffered: true` is load-bearing:
// a compile forced during boot (the sear/slab forcers `precompileAll` warms before the page is
// interactive) finishes and calls `performance.measure` well before this script's own
// `DD_RUM.onReady` fires, let alone before an observer constructed here would otherwise see it.
// `buffered` replays every matching entry already on the timeline at `observe()` time, so a
// compile finished during boot is still reported.
if (typeof PerformanceObserver !== "undefined") {
    try {
        new PerformanceObserver((list) => {
            const reports = compileVitalReports(
                list.getEntries(),
                __PIPELINE_COMPILE_MEASURE_PREFIX__,
                performance.timeOrigin,
            );
            for (const report of reports) {
                window.DD_RUM?.onReady(() => {
                    window.DD_RUM?.addDurationVital("pipeline_compile", {
                        startTime: report.startTime,
                        duration: report.duration,
                        context: report.context,
                    });
                });
                // Bound the shared `performance` timeline: the drain (`gpu.ts`'s `reportCompile`)
                // writes one `measure` entry per forcer per `build()` call against a device, so a
                // page that rebuilds scenes (a demo switcher, a hot-reloaded editor) grows the timeline unboundedly
                // if nothing ever clears it (S1's review finding, owned here). Clear only what
                // this callback just forwarded — `clearMeasures(name)` removes every entry sharing
                // that name, never the whole timeline, so an unrelated `measure` entry (there are
                // none today, but the API is shared) is untouched.
                performance.clearMeasures(report.name);
            }
        }).observe({ type: "measure", buffered: true });
    } catch {
        // Telemetry never breaks the page — an older runtime or a locked-down embedder that
        // throws on `observe` degrades silently, same posture as the engine-side emitter's own
        // guard (`gpu.ts`'s `reportCompile`).
    }
}

let state = initialFrameSamplerState;

// Bound on buffered `long-animation-frame` entries — one entry per delayed frame, so this covers
// several minutes of steady-state jank without growing unbounded on a long-lived page.
const LOAF_BUFFER_LIMIT = 200;

// `.observe({ type: "long-animation-frame" })` does NOT throw on an engine that lacks the entry
// type (measured: Chromium's own `.observe()` with a made-up type string is a silent no-op, and
// the Performance Timeline spec's `observe()` algorithm aborts without throwing for both the
// `type` and `entryTypes` forms when the type is unsupported) — a `try/catch` around the observer
// alone stays `loafSupported: true` forever on Safari with `loafEntries` permanently empty,
// indistinguishable from a genuinely idle main thread on every single slow frame. The reliable
// feature check is `PerformanceObserver.supportedEntryTypes`, read before ever calling `.observe`.
let loafSupported =
    typeof PerformanceObserver !== "undefined" &&
    (PerformanceObserver.supportedEntryTypes?.includes("long-animation-frame") ?? false);
let loafEntries: LoAFEntry[] = [];

if (loafSupported) {
    try {
        // `long-animation-frame` entries (`scripts`, `blockingDuration`) aren't in lib.dom's
        // `PerformanceEntry` yet — same shape as `packages/shallot/bin/verify.ts`'s own LoAF observer.
        new PerformanceObserver((list) => {
            for (const e of list.getEntries() as any[]) {
                loafEntries.push({
                    startTime: e.startTime,
                    duration: e.duration,
                    blockingDuration: e.blockingDuration,
                    scripts: (e.scripts ?? []).map((s: any) => ({
                        sourceURL: s.sourceURL ?? "",
                        sourceFunctionName: s.sourceFunctionName ?? "",
                        invoker: s.invoker ?? "",
                        duration: s.duration,
                    })),
                });
            }
            if (loafEntries.length > LOAF_BUFFER_LIMIT) {
                loafEntries = loafEntries.slice(-LOAF_BUFFER_LIMIT);
            }
        }).observe({ type: "long-animation-frame", buffered: true });
    } catch {
        // Belt-and-suspenders — `supportedEntryTypes` said yes but construction/`observe` still
        // threw (a real engine quirk, not the expected path); degrade rather than break the page.
        loafSupported = false;
    }
}

// `long-animation-frame` entries are delivered to the PerformanceObserver asynchronously, in a
// task separate from the frame whose work they describe — measured (a standalone Playwright probe,
// 2026-08-27) at two animation frames / ~130ms after the busy frame completes, never on the same
// task. A slow frame's own LoAF entry is therefore never in `loafEntries` yet at the moment `tick`
// detects the gap: a synchronous read at that point always sees `loafEntryCount: 0` for exactly the
// frame being attributed — the worst possible wrong reading, since every genuinely scripted stall
// would read as "idle main thread". Give the observer time to deliver the entry before correlating.
const LOAF_ATTRIBUTION_DELAY_MS = 500;

function attributeContext(startTime: number, duration: number): Promise<SlowFrameAttribution> {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(
                loafSupported
                    ? attributeSlowFrame(startTime, duration, loafEntries)
                    : unsupportedLoafAttribution(),
            );
        }, LOAF_ATTRIBUTION_DELAY_MS);
    });
}

// `wasHidden` has two setters, for two different failure modes, and one consumer:
//
//   - `tick`'s own `if (document.hidden) wasHidden = true` covers a browser that still fires
//     throttled rAF callbacks in a background tab — the flag is set from inside the loop with
//     no listener involved.
//   - the `visibilitychange` listener below covers the opposite and more common case (Chrome
//     and most browsers): rAF is fully PAUSED while hidden, so no tick ever runs to set the
//     flag itself, and without this listener `wasHidden` would stay false straight through a
//     background interval. The listener's hide callback (`document.hidden` true) fires while
//     the tab is still backgrounded, strictly before any resume rAF can fire — so it always
//     wins the race, and it stays safe to run from a separate task because it *only* sets a
//     flag; it never touches `state` and never resets.
//
// The reset itself — `resetFrameSampler` — runs nowhere but inside `tick`, synchronously with
// the first visible-frame sample, regardless of which setter got there first. That is what
// keeps the ordering race (a listener resetting `state` racing a resume rAF) from coming back:
// the only writer of `state` is `tick`, on its own thread of control.
let wasHidden = false;

document.addEventListener("visibilitychange", () => {
    if (document.hidden) wasHidden = true;
});

function tick(timestamp: number): void {
    // Browsers throttle or suspend rAF while a tab is hidden — skip sampling so a backgrounded
    // interval is never read as one raw delta.
    if (document.hidden) {
        wasHidden = true;
    } else {
        if (wasHidden) {
            // First visible frame after a backgrounded interval — reset so this frame is
            // treated as a first frame (never reports) instead of reporting the gap.
            state = resetFrameSampler(state);
            wasHidden = false;
        }
        const result = sampleFrame(state, timestamp);
        state = result.state;
        if (result.report) {
            const { startTime, duration, context } = result.report;
            // `addDurationVital`'s `startTime` is a Unix epoch timestamp (ms since 1970), not a
            // performance.now()/rAF-relative one (@datadog/browser-rum-core's own
            // `AddDurationVitalOptions.startTime` doc: "expects a UNIX timestamp in milliseconds")
            // — `performance.timeOrigin` is the epoch time of navigation start, so adding it
            // converts the sampler's rAF-relative timestamp to what the SDK expects. Passing the
            // raw relative value silently drops the vital: it never reaches the intake batch, no
            // console error, no exception (found 2026-08-26, S2's wire proof — a raw relative
            // startTime reads as ~1970 to the SDK's own event-time bookkeeping).
            const epochStartTime = performance.timeOrigin + startTime;
            // `startTime`/`duration` here are still rAF-relative, the same clock LoAF entries
            // report on — the attribution reads them before the epoch conversion above, never
            // after. The read itself is deferred (`attributeContext`'s `LOAF_ATTRIBUTION_DELAY_MS`)
            // so the frame's own LoAF entry has had time to arrive; `epochStartTime`/`duration` are
            // captured synchronously here, so the deferral doesn't skew the vital's own timing.
            attributeContext(startTime, duration).then((attribution) => {
                window.DD_RUM?.onReady(() => {
                    window.DD_RUM?.addDurationVital("slow_frame", {
                        startTime: epochStartTime,
                        duration,
                        context: { ...context, ...attribution },
                    });
                });
            });
        }
    }
    requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
