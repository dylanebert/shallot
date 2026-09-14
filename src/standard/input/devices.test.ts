import {
    devices,
    InputPlugin,
    pointerButton,
    pointerMove,
    pointerWheel,
    pressKey,
    releaseKey,
    State,
    Time,
    touchPoint,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";

function inputState(): State {
    const state = new State();
    for (const system of InputPlugin.systems ?? []) state.addSystem(system, InputPlugin.name);
    return state;
}

check(
    "device keyboard edges survive one frame",
    {
        claim: "a State-scoped keyboard record reports held, pressed, released, pointer and touch facts",
    },
    () => {
        const state = inputState();
        const seen: Array<{ held: boolean; pressed: boolean; released: boolean }> = [];
        const reader = {
            group: "simulation" as const,
            update(s: State) {
                const keys = devices(s).keys;
                seen.push({
                    held: keys.held.has("KeyW"),
                    pressed: keys.pressed.has("KeyW"),
                    released: keys.released.has("KeyW"),
                });
            },
        };
        state.addSystem(reader);
        pressKey(state, "KeyW");
        pointerMove(state, 12, 24, 3, -2);
        pointerButton(state, "left", true);
        pointerWheel(state, 7);
        touchPoint(state, 1, 10, 20);
        state.step(Time.FIXED_DT);
        if (!seen[0]?.held || !seen[0].pressed || seen[0].released)
            throw new Error("press edge was not visible");
        const d = devices(state);
        if (d.mouse.x !== 12 || d.mouse.y !== 24 || d.mouse.deltaX !== 0 || d.mouse.scroll !== 0) {
            // frame latches are deliberately cleared at the draw boundary; the reader above observes them.
            throw new Error("frame device deltas were not cleared");
        }
        if (d.touch.count !== 1 || !d.mouse.left) throw new Error("pointer/touch fact was lost");
        releaseKey(state, "KeyW");
        const releaseReader = {
            group: "simulation" as const,
            update(s: State) {
                if (!devices(s).keys.released.has("KeyW") || devices(s).keys.held.has("KeyW")) {
                    throw new Error("release edge was not visible");
                }
            },
        };
        state.addSystem(releaseReader);
        state.step(0);
        state.dispose();
    },
);

check(
    "device fixed edge is consumed once across two ticks",
    { claim: "a key press before a two-tick frame reaches exactly one fixed tick" },
    () => {
        const state = inputState();
        let count = 0;
        state.addSystem({
            group: "fixed",
            update(s: State) {
                if (devices(s).keys.tickPressed.has("Space")) count++;
            },
        });
        pressKey(state, "Space");
        state.step(Time.FIXED_DT * 2);
        if (count !== 1) throw new Error(`expected one fixed edge, got ${count}`);
        state.dispose();
    },
);

check(
    "device fixed edge carries over a zero-tick frame",
    {
        claim: "a key press in a frame with zero fixed ticks reaches the next frame's first fixed tick",
    },
    () => {
        const state = inputState();
        let count = 0;
        state.addSystem({
            group: "fixed",
            update(s: State) {
                if (devices(s).keys.tickPressed.has("KeyA")) count++;
            },
        });
        pressKey(state, "KeyA");
        state.step(0);
        state.step(Time.FIXED_DT);
        if (count !== 1) throw new Error(`expected carried edge, got ${count}`);
        state.dispose();
    },
);

check(
    "device press tick is deterministic",
    { claim: "a press records the State fixed tick rather than a wall-clock timestamp" },
    () => {
        const state = inputState();
        state.step(Time.FIXED_DT);
        const tick = state.time.fixedTick;
        pressKey(state, "KeyB");
        if (devices(state).keys.pressedTick.get("KeyB") !== tick)
            throw new Error("wrong pressedTick");
        state.dispose();
    },
);

check(
    "device press and release latch between frames",
    {
        claim: "a press and release between frames reports both edges and no held key",
    },
    () => {
        const state = inputState();
        let seen = false;
        state.addSystem({
            group: "simulation",
            update(s: State) {
                const keys = devices(s).keys;
                seen =
                    keys.pressed.has("KeyQ") && keys.released.has("KeyQ") && !keys.held.has("KeyQ");
            },
        });
        pressKey(state, "KeyQ");
        releaseKey(state, "KeyQ");
        state.step(0);
        if (!seen) throw new Error("between-frame edges were not latched");
        state.dispose();
    },
);

check(
    "device edge count is cadence invariant",
    {
        claim: "the same press has one fixed edge under both batched and one-tick-per-frame cadence",
    },
    () => {
        const batched = inputState();
        const stepped = inputState();
        const count = (state: State) => {
            let value = 0;
            state.addSystem({
                group: "fixed",
                update(s: State) {
                    if (devices(s).keys.tickPressed.has("KeyE")) value++;
                },
            });
            return () => value;
        };
        const batchedCount = count(batched);
        const steppedCount = count(stepped);
        pressKey(batched, "KeyE");
        pressKey(stepped, "KeyE");
        batched.step(Time.FIXED_DT * 2);
        stepped.step(Time.FIXED_DT);
        stepped.step(Time.FIXED_DT);
        if (batchedCount() !== 1 || steppedCount() !== 1)
            throw new Error("cadence changed edge count");
        batched.dispose();
        stepped.dispose();
    },
);

check(
    "device edges are State-scoped",
    {
        claim: "two States fed the same interleaved producer sequence keep equal independent records",
    },
    () => {
        const a = inputState();
        const b = inputState();
        pressKey(a, "KeyW");
        pressKey(b, "KeyW");
        a.step(Time.FIXED_DT);
        b.step(Time.FIXED_DT);
        if (devices(a).keys.held.has("KeyW") !== devices(b).keys.held.has("KeyW"))
            throw new Error("twin held mismatch");
        releaseKey(a, "KeyW");
        releaseKey(b, "KeyW");
        if (!devices(a).keys.released.has("KeyW") || !devices(b).keys.released.has("KeyW"))
            throw new Error("twin release mismatch");
        a.dispose();
        b.dispose();
    },
);
