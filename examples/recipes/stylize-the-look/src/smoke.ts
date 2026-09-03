import { Outline, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import { Pick } from "./select";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const active = (state: State): number =>
    [...state.query([Pick])].find((eid) => state.has(eid, Outline)) ?? -1;

/** Real-page smoke: the selection cursor moves the Outline component between pickable boxes. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const before = active(state);
            let after = before;
            for (let i = 0; i < 90 && after === before; i++) {
                await frame();
                after = active(state);
            }
            const ok = before >= 0 && after >= 0 && before !== after;
            return {
                ok,
                checks: [
                    {
                        name: "outline selection advances",
                        ok,
                        detail: `outlined entity ${before} -> ${after}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
