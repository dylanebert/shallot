import { DirectionalLight, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the day-night system moves the sun direction consumed by the sky. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const sun = state.only([DirectionalLight]);
            const before = sun < 0 ? 0 : DirectionalLight.direction.x.get(sun);
            for (let i = 0; i < 30; i++) await frame();
            const after = sun < 0 ? 0 : DirectionalLight.direction.x.get(sun);
            const ok = sun >= 0 && Math.abs(after - before) > 0.001;
            return {
                ok,
                checks: [
                    {
                        name: "day night sun advances",
                        ok,
                        detail: `direction.x ${before.toFixed(4)} -> ${after.toFixed(4)}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
