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
            // `cursor` steps the highlight on `floor(state.time.elapsed * 1.5)`, so one step is 2/3 s of
            // scene time. Wait in that unit, not in frames: a 90-frame budget is 1.5 s at 60 Hz but only
            // 0.37 s at 240 Hz, where the cursor never steps and the check reds on the refresh rate. The
            // wall-clock ceiling is the exhaustion arm — a stalled clock fails rather than hangs.
            const until = state.time.elapsed + 1 / 1.5;
            const deadline = performance.now() + 6000;
            while (after === before && state.time.elapsed < until) {
                if (performance.now() > deadline) break;
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
