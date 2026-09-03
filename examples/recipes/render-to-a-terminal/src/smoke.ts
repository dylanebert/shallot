import { Orbit, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the terminal recipe's arrow-key path changes orbit yaw. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const camera = state.only([Orbit]);
            const before = Orbit.yaw.get(camera);
            globalThis.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowLeft" }));
            for (let i = 0; i < 15; i++) await frame();
            globalThis.dispatchEvent(new KeyboardEvent("keyup", { code: "ArrowLeft" }));
            const after = Orbit.yaw.get(camera);
            const ok = camera >= 0 && after < before;
            return {
                ok,
                checks: [
                    {
                        name: "terminal key orbit responds",
                        ok,
                        detail: `yaw ${before.toFixed(3)} -> ${after.toFixed(3)}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
