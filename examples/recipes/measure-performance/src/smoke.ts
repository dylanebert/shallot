import { type Plugin, Profile, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the profiler is
// actually measuring — the `window.__benchmark` API installed and at least one GPU pass reporting nonzero
// time on `Profile.gpu` — and that the on-open observable is real: the HUD overlay budget.ts's `warm()`
// surfaces via `showProfiler()` is in the DOM. The concept measure-performance teaches. Kept out of the
// teaching plugin (budget.ts); enabled only through this recipe's manifest.

// the overlay element ProfilePlugin mounts (`data-shallot-profile` attribute in extras/profile).
const OVERLAY = "[data-shallot-profile]";

const READY_MS = 5000; // cap the wait for the profiler to attach + drain its first frame
const SETTLE_MS = 500; // let the async timestamp readback populate Profile.gpu after ready

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));

async function settle(ms: number): Promise<void> {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) await frame();
}

export const Smoke: Plugin = {
    name: "MeasureSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            const t0 = performance.now();
            while (!window.__benchmark?.ready && performance.now() - t0 < READY_MS) await frame();
            await settle(SETTLE_MS);
            const hasBench = !!window.__benchmark;
            let peakGpu = 0;
            for (const ms of Profile.gpu.values()) peakGpu = Math.max(peakGpu, ms);
            const measuring = hasBench && peakGpu > 0;
            const overlay = !!document.querySelector(OVERLAY);
            return {
                ok: measuring && overlay,
                checks: [
                    {
                        name: "profiler reports gpu time",
                        ok: measuring,
                        detail: `__benchmark ${hasBench ? "present" : "absent"}, peak gpu pass ${peakGpu.toFixed(3)} ms (need > 0)`,
                    },
                    {
                        name: "profiler overlay visible on open",
                        ok: overlay,
                        detail: `${OVERLAY} ${overlay ? "present" : "absent"} (surfaced by budget.ts warm())`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
