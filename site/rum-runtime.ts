// Browser-only rAF/SDK glue — drives `rum-sampler.ts`'s pure decision logic with real
// `requestAnimationFrame` timestamps and reports through the Datadog RUM CDN snippet's global.
// Bundled by `scripts/build-site.ts` (`Bun.build`, target "browser") and inlined into every demo
// page; never imported by anything else, so it has no test of its own — the logic it drives is
// tested in `rum-sampler.test.ts`.

import { initialFrameSamplerState, sampleFrame } from "./rum-sampler";

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

function tick(timestamp: number): void {
    const result = sampleFrame(state, timestamp);
    state = result.state;
    if (result.report) {
        const { startTime, duration, context } = result.report;
        window.DD_RUM?.onReady(() => {
            window.DD_RUM?.addDurationVital("slow_frame", { startTime, duration, context });
        });
    }
    requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
