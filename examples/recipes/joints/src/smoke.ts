import { Body, Physics, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the joints hold their
// load aloft — the concept joints teaches (a suspension that hangs, a cantilever that stays level). A failed
// joint drops its body to the floor, so "the lowest jointed body is still well above the ground" is the
// observable. Kept out of the teaching plugin (joints.ts); enabled only through this recipe's manifest.

const SETTLE_MS = 2000; // let the suspension find its hung rest pose and the crates land on the platform

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const dynamic = (state: State): number[] =>
    [...state.query([Body])].filter((e) => Body.mass.get(e) > 0);

async function wait(ms: number): Promise<void> {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) await frame();
}

export const Smoke: Plugin = {
    name: "JointsSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            await wait(SETTLE_MS);
            let lowest = Number.POSITIVE_INFINITY;
            for (const e of dynamic(state)) {
                const b = Physics.backend?.readBody(e);
                if (b) lowest = Math.min(lowest, b.pos[1]);
            }
            const ok = lowest > 3;
            return {
                ok,
                checks: [
                    {
                        name: "joints hold their load",
                        ok,
                        detail: `lowest jointed body at y=${lowest.toFixed(2)} (need > 3 — held aloft, not fallen)`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
