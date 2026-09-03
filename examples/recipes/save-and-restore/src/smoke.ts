import type { Plugin, State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const KEY = "shallot:save-and-restore";

/** Real-page smoke: pressing S serializes the authored world into the recipe's persistence slot. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            localStorage.removeItem(KEY);
            globalThis.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyS" }));
            await frame();
            globalThis.dispatchEvent(new KeyboardEvent("keyup", { code: "KeyS" }));
            await frame();
            const saved = localStorage.getItem(KEY) ?? "";
            const ok = saved.includes("<scene>") && saved.includes('id="hero"');
            return {
                ok,
                checks: [
                    {
                        name: "save writes authored world",
                        ok,
                        detail: `${saved.length} serialized characters`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
