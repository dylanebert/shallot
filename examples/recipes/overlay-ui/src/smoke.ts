import type { Plugin, State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the canvas-bounded overlay mounts and its elapsed-time HUD updates. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const before = document.querySelector("canvas")?.parentElement?.textContent ?? "";
            for (let i = 0; i < 30; i++) await frame();
            const after = document.querySelector("canvas")?.parentElement?.textContent ?? "";
            const ok = /running \d+\.\d+s/.test(after) && before !== after;
            return {
                ok,
                checks: [
                    {
                        name: "overlay hud advances",
                        ok,
                        detail: `${before.trim()} -> ${after.trim()}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
