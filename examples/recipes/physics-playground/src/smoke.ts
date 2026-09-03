import { Body, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** Real-page smoke: the playground spawner adds a dynamic body and physics advances it. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const before = new Set(state.query([Body]));
            let spawned = -1;
            for (let i = 0; i < 90; i++) {
                await frame();
                spawned = [...state.query([Body])].find((eid) => !before.has(eid)) ?? -1;
                if (spawned >= 0) break;
            }
            const mass = spawned < 0 ? 0 : Body.mass.get(spawned);
            const y = spawned < 0 ? 0 : Body.pos.y.get(spawned);
            const ok = spawned >= 0 && mass === 1 && y === 9;
            return {
                ok,
                checks: [
                    {
                        name: "playground spawns dynamic body",
                        ok,
                        detail: `body ${spawned}, mass ${mass}, spawn y ${y.toFixed(3)}`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
