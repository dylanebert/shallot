// Browser-only rAF/SDK glue — drives `rum-sampler.ts`'s pure decision logic with real
// `requestAnimationFrame` timestamps and reports through the Datadog RUM CDN snippet's global.
// Bundled by `scripts/build-site.ts` (`Bun.build`, target "browser") and inlined into every demo
// page; never imported by anything else, so it has no test of its own — the logic it drives is
// tested in `rum-sampler.test.ts`.

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
            window.DD_RUM?.onReady(() => {
                window.DD_RUM?.addDurationVital("slow_frame", { startTime, duration, context });
            });
        }
    }
    requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
