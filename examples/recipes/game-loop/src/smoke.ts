import { type Plugin, type State, Transform } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const sample = (state: State): number[] =>
    [...state.query([Transform])].flatMap((eid) => [
        Transform.pos.x.get(eid),
        Transform.pos.y.get(eid),
        Transform.pos.z.get(eid),
        Transform.rot.x.get(eid),
        Transform.rot.y.get(eid),
        Transform.rot.z.get(eid),
        Transform.rot.w.get(eid),
    ]);

/** Real-page smoke: the recipe's live system must change an authored pose. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const before = sample(state);
            globalThis.dispatchEvent?.(new KeyboardEvent("keydown", { code: "KeyW" }));
            for (let i = 0; i < 90; i++) await frame();
            globalThis.dispatchEvent?.(new KeyboardEvent("keyup", { code: "KeyW" }));
            const after = sample(state);
            const delta = before.reduce((sum, value, i) => sum + Math.abs(value - after[i]), 0);
            const ok = delta > 0.001;
            return {
                ok,
                checks: [
                    {
                        name: "game loop responds",
                        ok,
                        detail: `${delta.toFixed(3)} aggregate pose change`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
