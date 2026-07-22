import { Body, Physics, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the platform actually
// slides — the concept moving-platform teaches. It lives outside the teaching plugin (elevator.ts) so the
// physics reads clean, and is enabled only through this recipe's manifest. Installing `window.__harness` is
// inert during a plain `shallot dev` run — nothing drives it unless a verifier is watching.
//
// the platform is a `mass: 0` kinematic body, so measure vertical travel across EVERY body: the static
// ground never moves, the kinematic platform strokes the full range, and the crates ride it. the platform's
// travel dominates, so the max is what proves the lift moves.

const SETTLE_MS = 800; // let the crates land onto the platform before sampling
const SAMPLE_MS = 4200; // a full up-and-down cycle is 4 s — sample just over one

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const bodies = (state: State): number[] => [...state.query([Body])];

async function settle(ms: number): Promise<void> {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) await frame();
}

export const Smoke: Plugin = {
    name: "PlatformSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            await settle(SETTLE_MS);
            const eids = bodies(state);
            const lo = new Map<number, number>();
            const hi = new Map<number, number>();
            const t0 = performance.now();
            while (performance.now() - t0 < SAMPLE_MS) {
                for (const e of eids) {
                    const b = Physics.backend?.readBody(e);
                    if (!b) continue;
                    lo.set(e, Math.min(lo.get(e) ?? b.pos[1], b.pos[1]));
                    hi.set(e, Math.max(hi.get(e) ?? b.pos[1], b.pos[1]));
                }
                await frame();
            }
            let travel = 0;
            for (const e of eids) travel = Math.max(travel, (hi.get(e) ?? 0) - (lo.get(e) ?? 0));
            const ok = travel > 1;
            return {
                ok,
                checks: [
                    {
                        name: "platform slides",
                        ok,
                        detail: `${travel.toFixed(2)} m vertical travel (need > 1)`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
