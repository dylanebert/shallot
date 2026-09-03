import { type Plugin, type State, Transform } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

// dynamics smoke for `shallot verify` (run headless by `scripts/recipes.ts`): assert the clips actually
// move their targets — the concept this recipe teaches. The scene's two animators name `lift` and
// `bounce`; both are `transform.pos.y` clips with a 1.8 m swing, so vertical travel over one cycle is the
// observable. An animator whose clip resolves to nothing warns and moves nothing, and the first version
// of this scene shipped exactly that (no `clip:` at all) with every gate green on the static frame.

const SAMPLE_MS = 1600; // both clips cycle in ≤ 1.4 s

const frame = (): Promise<void> => new Promise((r) => requestAnimationFrame(() => r()));

export const Smoke: Plugin = {
    name: "ClipsSmoke",
    warm(state: State) {
        const h = installHarness(state);
        h.run = async (): Promise<Verdict> => {
            const eids = [...state.query([Transform])];
            const lo = new Map<number, number>();
            const hi = new Map<number, number>();
            const t0 = performance.now();
            while (performance.now() - t0 < SAMPLE_MS) {
                for (const e of eids) {
                    const y = Transform.pos.y.get(e);
                    lo.set(e, Math.min(lo.get(e) ?? y, y));
                    hi.set(e, Math.max(hi.get(e) ?? y, y));
                }
                await frame();
            }
            let moving = 0;
            let travel = 0;
            for (const e of eids) {
                const t = (hi.get(e) ?? 0) - (lo.get(e) ?? 0);
                if (t > 1) moving++;
                travel = Math.max(travel, t);
            }
            const ok = moving >= 2;
            return {
                ok,
                checks: [
                    {
                        name: "both clips move their targets",
                        ok,
                        detail: `${moving} entities travelled > 1 m vertically (need 2), max ${travel.toFixed(2)} m`,
                    },
                ],
            };
        };
    },
};

export default Smoke;
