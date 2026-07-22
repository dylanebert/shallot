import { Body, Physics, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the boxes slide at
// different rates — the concept surface-friction teaches. Same-friction boxes would all travel the same
// distance; a friction ladder spreads them out, so "the spread of travelled distance across the boxes is
// large" is the observable. Kept out of the teaching plugin (ramp.ts); enabled only through this manifest.

const SAMPLE_MS = 2600; // the slippery boxes slide off while the grippy ones grip and barely move

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));
const dynamic = (state: State): number[] =>
    [...state.query([Body])].filter((e) => Body.mass.get(e) > 0);

async function wait(ms: number): Promise<void> {
    const t0 = performance.now();
    while (performance.now() - t0 < ms) await frame();
}

export const Smoke: Plugin = {
    name: "FrictionSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            const eids = dynamic(state);
            const sx = new Map<number, number>();
            const sz = new Map<number, number>();
            for (const e of eids) {
                const b = Physics.backend?.readBody(e);
                if (b) {
                    sx.set(e, b.pos[0]);
                    sz.set(e, b.pos[2]);
                }
            }
            await wait(SAMPLE_MS);
            const travelled: number[] = [];
            for (const e of eids) {
                const b = Physics.backend?.readBody(e);
                if (b)
                    travelled.push(
                        Math.hypot(b.pos[0] - (sx.get(e) ?? 0), b.pos[2] - (sz.get(e) ?? 0)),
                    );
            }
            const spread = Math.max(...travelled) - Math.min(...travelled);
            const ok = spread > 2;
            return {
                ok,
                checks: [
                    {
                        name: "friction rates differ",
                        ok,
                        detail: `${spread.toFixed(2)} m spread between slippery and grippy (need > 2)`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
