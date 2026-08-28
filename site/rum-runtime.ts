// Browser-only rAF/SDK glue — drives `rum-sampler.ts`'s pure decision logic with real
// `requestAnimationFrame` timestamps and reports through the Datadog RUM CDN snippet's global.
// Bundled by `scripts/build-site.ts` (`Bun.build`, target "browser") and inlined into every demo
// page; never imported by anything else, so it has no test of its own — the logic it drives is
// tested in `rum-sampler.test.ts`.

import { attributeSlowFrame, type LoAFEntry, unsupportedLoafAttribution } from "./rum-loaf";
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

let state = initialFrameSamplerState;

// Bound on buffered `long-animation-frame` entries — one entry per delayed frame, so this covers
// several minutes of steady-state jank without growing unbounded on a long-lived page.
const LOAF_BUFFER_LIMIT = 200;

let loafSupported = true;
let loafEntries: LoAFEntry[] = [];

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
    // Engine lacks LoAF (Safari) — `attributeContext` degrades to `loafSupported: false` below.
    loafSupported = false;
}

function attributeContext(startTime: number, duration: number) {
    return loafSupported
        ? attributeSlowFrame(startTime, duration, loafEntries)
        : unsupportedLoafAttribution();
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
            // report on — the attribution runs before the epoch conversion above, never after.
            const attribution = attributeContext(startTime, duration);
            window.DD_RUM?.onReady(() => {
                window.DD_RUM?.addDurationVital("slow_frame", {
                    startTime: epochStartTime,
                    duration,
                    context: { ...context, ...attribution },
                });
            });
        }
    }
    requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
