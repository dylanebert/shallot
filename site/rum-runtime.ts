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

// Tracked inline rather than via a `visibilitychange` listener: a listener callback and the
// first post-resume rAF are two separately-queued tasks with no ordering guarantee between
// them, so a resume rAF firing before the listener's task would still see a stale
// `lastTimestamp` and report a fake giant slow frame. Reading and clearing `wasHidden` inside
// `tick` itself keeps the hide/resume transition and the reset in the same synchronous
// callback — there is no second task to race.
let wasHidden = false;

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
