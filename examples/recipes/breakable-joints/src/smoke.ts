import { Body, Physics, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert a joint actually breaks
// under the rising load — the concept breakable-joints teaches. A box hangs from its joint at a rest height
// well above the floor; when the joint cuts, the box drops to the floor. So "a box ended up on the floor"
// (below the hung rest, not still suspended) is the observable. Kept out of the teaching plugin
// (breakable.ts); enabled only through this recipe's manifest.
//
// the load ramps fast, so the weakest joints break during the boot/handshake before run() is even called —
// measuring a fall from run-start is unreliable. the END pose is not: a still-hung box sits ~4 m up, a box
// whose joint cut rests on the floor (y ≈ 1). FLOOR_Y separates the two with a wide margin.

const SETTLE_MS = 2500; // let the load finish ramping and any freed boxes reach the floor
const FLOOR_Y = 2.5; // hung rest is ~y 5, the floor ~y 1 — a box below this cut loose and fell

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const dynamic = (state: State): number[] =>
    [...state.query([Body])].filter((e) => Body.mass.get(e) > 0);

export const Smoke: Plugin = {
    name: "BreakableSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            const eids = dynamic(state);
            const t0 = performance.now();
            while (performance.now() - t0 < SETTLE_MS) await frame();
            let broke = 0;
            for (const e of eids) {
                const b = Physics.backend?.readBody(e);
                if (b && b.pos[1] < FLOOR_Y) broke++;
            }
            const ok = broke >= 1;
            return {
                ok,
                checks: [
                    {
                        name: "a joint breaks under load",
                        ok,
                        detail: `${broke}/${eids.length} boxes cut loose and fell to the floor (need >= 1)`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
