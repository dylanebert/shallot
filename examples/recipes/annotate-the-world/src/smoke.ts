import { type Plugin, type State, Text } from "@dylanebert/shallot";
import { installHarness, type Verdict } from "@dylanebert/shallot/harness";
import { Counter } from "./annotate";

const frame = (): Promise<void> => new Promise((resolve) => requestAnimationFrame(() => resolve()));

// `tick` rewrites the label on each whole second of `state.time.elapsed`, so the wait is one second of
// scene time — not a frame count. A 90-frame budget is 1.5 s at 60 Hz but only 0.37 s at 240 Hz, where
// the label never ticks and the check reds on the display's refresh rate. The wall-clock ceiling is the
// exhaustion arm: a stalled clock fails the check instead of hanging the harness.
const advance = async (state: State, seconds: number): Promise<boolean> => {
    const until = state.time.elapsed + seconds;
    const deadline = performance.now() + (seconds + 5) * 1000;
    while (state.time.elapsed < until) {
        if (performance.now() > deadline) return false;
        await frame();
    }
    return true;
};

/** Real-page smoke: the elapsed-seconds annotation changes its interned text content. */
const Smoke: Plugin = {
    name: "RecipeSmoke",
    warm(state: State) {
        const harness = installHarness(state);
        harness.run = async (): Promise<Verdict> => {
            const label = state.only([Counter, Text]);
            const before = Text.content.get(label);
            const advanced = await advance(state, 1);
            const after = Text.content.get(label);
            const ok = label >= 0 && advanced && before !== after;
            return {
                ok,
                checks: [
                    {
                        name: "world annotation advances",
                        ok,
                        detail: advanced
                            ? `content id ${before} -> ${after}`
                            : "the scene clock did not advance one second",
                    },
                ],
            };
        };
    },
};
export default Smoke;
