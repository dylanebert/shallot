import { type Plugin, type State, Text } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import { Counter } from "./annotate";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the elapsed-seconds annotation changes its interned text content. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const label = state.only([Counter, Text]);
            const before = Text.content.get(label);
            for (let i = 0; i < 90; i++) await frame();
            const after = Text.content.get(label);
            const ok = label >= 0 && before !== after;
            return {
                ok,
                checks: [
                    {
                        name: "world annotation advances",
                        ok,
                        detail: `content id ${before} -> ${after}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
