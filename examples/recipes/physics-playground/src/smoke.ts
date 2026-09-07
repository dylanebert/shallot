import { Body, type Plugin, type State } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

/** scene seconds to wait for the next spawn: `spawn` rains a cube every 0.8 s of `state.time.elapsed`,
 *  so two periods cover the wait whatever the phase when `run` begins. */
const BUDGET_SECONDS = 2;

/** Real-page smoke: the playground spawner adds a dynamic body and physics advances it. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const before = new Set(state.query([Body]));
            // `spawn` gates on `state.time.elapsed`, so the wait is scene time — not a frame count. A
            // 90-frame budget is 1.5 s at 60 Hz but only 0.37 s at 240 Hz, where no second cube falls and
            // the check reds on the display's refresh rate. The wall-clock ceiling is the exhaustion arm:
            // a stalled clock fails the check instead of hanging the harness.
            const start = state.time.elapsed;
            const until = start + BUDGET_SECONDS;
            const deadline = performance.now() + (BUDGET_SECONDS + 5) * 1000;
            let spawned = -1;
            let callbacks = 0;
            let stalled = false;
            while (spawned < 0 && state.time.elapsed < until) {
                if (performance.now() > deadline) {
                    stalled = true;
                    break;
                }
                await frame();
                callbacks++;
                spawned = [...state.query([Body])].find((eid) => !before.has(eid)) ?? -1;
            }
            const waited = state.time.elapsed - start;
            const mass = spawned < 0 ? 0 : Body.mass.get(spawned);
            const y = spawned < 0 ? 0 : Body.pos.y.get(spawned);
            const ok = spawned >= 0 && mass === 1 && y === 9;
            // the frame clock this seat ran at, carried on every receipt: callbacks against scene seconds
            // is the reading that separates a real miss from a fast display.
            const clock = `${callbacks} rAF callbacks, ${waited.toFixed(3)} s scene time`;
            return {
                ok,
                checks: [
                    {
                        name: "playground spawns dynamic body",
                        ok,
                        detail: stalled
                            ? `the scene clock did not advance ${BUDGET_SECONDS} s (${clock})`
                            : `body ${spawned}, mass ${mass}, spawn y ${y.toFixed(3)} (${clock})`,
                    },
                ],
            };
        };
    },
};
export default Smoke;
