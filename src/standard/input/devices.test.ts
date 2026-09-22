import {
    audioContextState,
    blur,
    devices,
    focus,
    InputPlugin,
    inputSource,
    pointerButton,
    pointerLockChanged,
    pointerLockStatus,
    pointerMove,
    pointerWheel,
    pressKey,
    releaseKey,
    requestPointerLock,
    requirePointerLock,
    resizeViewport,
    State,
    setInputEnabled,
    setInputSource,
    Time,
    touchPoint,
    visibilityChanged,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { sizeView, type View } from "@dylanebert/shallot/render";
import { reportAudioContextState, reportViewport } from "./index";

function inputState(): State {
    const state = new State();
    for (const system of InputPlugin.systems ?? []) state.addSystem(system, InputPlugin.name);
    return state;
}

function replaceGlobal(name: "document" | "window", value: unknown): () => void {
    const prior = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    return () => {
        if (prior) Object.defineProperty(globalThis, name, prior);
        else Reflect.deleteProperty(globalThis, name);
    };
}

check(
    "controlled input does not inspect absent host globals",
    {
        claim: "controlled input setup reads ambient browser globals when a headless State has none",
    },
    () => {
        const restoreDocument = replaceGlobal("document", undefined);
        const restoreWindow = replaceGlobal("window", undefined);
        const state = inputState();
        try {
            setInputSource(state, "controlled");
            state.step(0);
            if (inputSource(state) !== "controlled") throw new Error("source was not selected");
            pressKey(state, "KeyW");
            state.step(Time.FIXED_DT);
            if (!devices(state).keys.held.has("KeyW")) throw new Error("controlled input was lost");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "controlled input ignores present host reports",
    {
        claim: "controlled input lets browser setup, viewport reports or audio callbacks overwrite supplied facts",
    },
    () => {
        let queried = 0;
        let listeners = 0;
        let requested = 0;
        const canvas = {
            style: { touchAction: "" },
            addEventListener: () => listeners++,
            requestPointerLock: () => {
                requested++;
            },
        } as unknown as HTMLCanvasElement;
        const restoreDocument = replaceGlobal("document", {
            querySelectorAll: () => {
                queried++;
                return [canvas];
            },
            addEventListener: () => listeners++,
        });
        const restoreWindow = replaceGlobal("window", {
            addEventListener: () => listeners++,
        });
        const state = inputState();
        try {
            setInputSource(state, "controlled");
            state.step(0);
            requestPointerLock(state);
            if (queried !== 0 || listeners !== 0 || requested !== 0)
                throw new Error("controlled setup touched the host");

            resizeViewport(state, 0, 100, 50, 2);
            audioContextState(state, "running");
            reportViewport(state, 0, 640, 360, 1);
            reportAudioContextState(state, "closed");
            const input = devices(state);
            const viewport = input.viewport.get(0);
            if (
                viewport?.cssWidth !== 100 ||
                viewport.cssHeight !== 50 ||
                viewport.dpr !== 2 ||
                input.audio.context !== "running"
            )
                throw new Error("a host report overwrote controlled facts");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "browser remains the default input producer",
    {
        claim: "the default input source stops binding browser listeners or requesting pointer lock",
    },
    () => {
        let listeners = 0;
        let requested = 0;
        const canvas = {
            style: { touchAction: "" },
            addEventListener: () => listeners++,
            requestPointerLock: () => {
                requested++;
            },
        } as unknown as HTMLCanvasElement;
        const restoreDocument = replaceGlobal("document", {
            querySelectorAll: () => [canvas],
            addEventListener: () => listeners++,
        });
        const restoreWindow = replaceGlobal("window", {
            addEventListener: () => listeners++,
        });
        const state = inputState();
        try {
            state.step(0);
            requestPointerLock(state);
            if (inputSource(state) !== "browser" || listeners === 0 || requested !== 1)
                throw new Error("browser input was not the default adapter");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "input source selection closes at setup",
    {
        claim: "changing a State's input producer after setup silently hot-switches its host boundary",
    },
    () => {
        const state = inputState();
        state.step(0);
        let refused = false;
        try {
            setInputSource(state, "controlled");
        } catch (error) {
            refused = error instanceof Error && /input setup/i.test(error.message);
        } finally {
            state.dispose();
        }
        if (!refused) throw new Error("late input source selection was accepted");
    },
);

check(
    "controlled edges survive zero and multiple fixed ticks",
    {
        claim: "controlled input edges depend on frame cadence rather than the independent fixed and draw boundaries",
    },
    () => {
        const state = inputState();
        setInputSource(state, "controlled");
        const fixedSeen: string[] = [];
        state.addSystem({
            group: "fixed",
            update(s: State) {
                if (devices(s).keys.tickPressed.has("KeyA")) fixedSeen.push("A");
                if (devices(s).keys.tickPressed.has("KeyB")) fixedSeen.push("B");
            },
        });
        pressKey(state, "KeyA");
        state.step(0);
        if (!devices(state).keys.tickPressed.has("KeyA"))
            throw new Error("zero-tick frame reset a press");
        state.step(Time.FIXED_DT * 2);
        pressKey(state, "KeyB");
        state.step(Time.FIXED_DT);
        if (fixedSeen.join("") !== "AB") throw new Error(`unexpected fixed edges: ${fixedSeen}`);
        if (devices(state).keys.pressed.has("KeyA") || devices(state).keys.tickPressed.size !== 0)
            throw new Error("draw or fixed edge reset did not run");
        state.dispose();
    },
);

check(
    "device keyboard edges survive one frame",
    {
        claim: "a key press, pointer, wheel or touch fact is lost before the frame's readers see it, or a release edge never appears",
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
    "device fixed edge carries over a zero-tick frame",
    {
        claim: "a key press in a frame with zero fixed ticks is dropped before the next frame's first fixed tick",
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
    {
        claim: "a press records a wall-clock timestamp instead of the State fixed tick, so replays diverge",
    },
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
        claim: "a press and release between frames loses an edge or leaves the key held",
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
        claim: "the same press yields a different fixed edge count under batched and one-tick-per-frame cadence",
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
    "blur releases held device inputs with edges",
    { claim: "window blur leaves keys or pointer buttons held, or releases them without an edge" },
    () => {
        const state = inputState();
        pressKey(state, "KeyW");
        pointerButton(state, "left", true);
        blur(state);
        const input = devices(state);
        if (input.keys.held.has("KeyW") || !input.keys.released.has("KeyW"))
            throw new Error("blur did not emit a key release edge");
        if (input.mouse.left) throw new Error("blur left a pointer button held");
        state.dispose();
    },
);

check(
    "hidden visibility releases held device inputs with edges",
    {
        claim: "hiding the page leaves keys or pointer buttons held, or releases them without an edge",
    },
    () => {
        const state = inputState();
        focus(state, 0);
        pressKey(state, "KeyA");
        pointerButton(state, "right", true);
        visibilityChanged(state, true);
        const input = devices(state);
        if (input.keys.held.has("KeyA") || !input.keys.released.has("KeyA"))
            throw new Error("hidden visibility did not emit a key release edge");
        if (input.mouse.right) throw new Error("hidden visibility left a pointer button held");
        state.dispose();
    },
);

check(
    "pointer-lock exit releases held device inputs with edges",
    {
        claim: "pointer-lock exit leaves keys or pointer buttons held, or loses the lock refusal reason",
    },
    () => {
        const state = inputState();
        pointerLockChanged(state, true);
        pressKey(state, "KeyD");
        pointerButton(state, "middle", true);
        pointerLockChanged(state, false);
        const input = devices(state);
        if (pointerLockStatus(state) !== "unlocked") throw new Error("lock status did not exit");
        pointerLockChanged(state, true);
        pointerLockChanged(state, false, "denied by browser");
        if (
            pointerLockStatus(state) !== "refused" ||
            devices(state).pointer.lock.refusal !== "denied by browser"
        )
            throw new Error("lock refusal was not recorded");
        if (input.keys.held.has("KeyD") || !input.keys.released.has("KeyD"))
            throw new Error("lock exit did not emit a key release edge");
        if (input.mouse.middle) throw new Error("lock exit left a pointer button held");
        state.dispose();
    },
);

check(
    "require-lock gates pointer buttons",
    { claim: "a pointer button reads down before a required pointer lock engages" },
    () => {
        const state = inputState();
        requirePointerLock(state, true);
        pointerButton(state, "left", true);
        if (devices(state).mouse.left) throw new Error("button crossed the lock gate");
        pointerLockChanged(state, true);
        pointerButton(state, "left", true);
        if (!devices(state).mouse.left) throw new Error("locked button did not engage");
        state.dispose();
    },
);

check(
    "suspension neutralizes device reads with release edges",
    {
        claim: "a suspended State still reads held keys, pointer or touch data, or accepts new presses",
    },
    () => {
        const state = inputState();
        pressKey(state, "KeyS");
        pointerButton(state, "left", true);
        setInputEnabled(state, false);
        const input = devices(state);
        if (!input.suspended || input.keys.held.has("KeyS") || !input.keys.released.has("KeyS"))
            throw new Error("suspension did not neutralize the key with an edge");
        if (input.mouse.left || input.mouse.deltaX !== 0 || input.touch.count !== 0)
            throw new Error("suspension did not neutralize pointer/touch reads");
        pressKey(state, "KeyQ");
        if (input.keys.held.has("KeyQ")) throw new Error("suspended producer changed the record");
        state.dispose();
    },
);

check(
    "suspension is State-scoped",
    { claim: "suspending one State suspends or clears a second State's device record" },
    () => {
        const first = inputState();
        const second = inputState();
        pressKey(first, "KeyW");
        pressKey(second, "KeyW");
        setInputEnabled(first, false);
        if (!devices(first).suspended || devices(first).keys.held.has("KeyW"))
            throw new Error("first State did not suspend");
        if (devices(second).suspended || !devices(second).keys.held.has("KeyW"))
            throw new Error("second State was affected by suspension");
        first.dispose();
        second.dispose();
    },
);

check(
    "device edges are State-scoped",
    {
        claim: "interleaved producer calls on two States leak edges or held keys between their records",
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

check(
    "viewport resize sizes a headless view",
    {
        claim: "a State-scoped viewport row supplies CSS size and DPR to sizeView through backingSize",
    },
    () => {
        const state = inputState();
        resizeViewport(state, 0, 100, 50, 2);
        const canvas = {
            height: 0,
            style: { imageRendering: "" },
            width: 0,
        } as unknown as HTMLCanvasElement;
        const view = {
            canvas,
            clientHeight: 0,
            clientWidth: 0,
            context: null,
            depth: null,
            framebuffer: null,
            height: 0,
            observer: null,
            present: null,
            slot: 0,
            stamp: 0,
            tag: null,
            viewportIndex: 0,
            width: 0,
        } as View;
        sizeView(state, 1, view);
        if (canvas.width !== 200 || canvas.height !== 100)
            throw new Error(`expected 200x100 backing, got ${canvas.width}x${canvas.height}`);
        state.dispose();
    },
);

check(
    "normalized pointer follows viewport resize",
    {
        claim: "the normalized pointer coordinate uses the State-scoped viewport row after resize",
    },
    () => {
        const state = inputState();
        resizeViewport(state, 0, 100, 50, 1);
        focus(state, 0);
        pointerMove(state, 50, 25);
        if (devices(state).mouse.normalizedX !== 0.5 || devices(state).mouse.normalizedY !== 0.5)
            throw new Error("initial normalized pointer coordinate was wrong");
        resizeViewport(state, 0, 200, 100, 1);
        if (devices(state).mouse.normalizedX !== 0.25 || devices(state).mouse.normalizedY !== 0.25)
            throw new Error("normalized pointer coordinate did not follow resize");
        state.dispose();
    },
);

check(
    "viewport rows are State-scoped",
    {
        claim: "two States hold independent per-canvas viewport records",
    },
    () => {
        const first = inputState();
        const second = inputState();
        resizeViewport(first, 0, 320, 180, 1);
        resizeViewport(second, 0, 640, 360, 2);
        const a = devices(first).viewport.get(0);
        const b = devices(second).viewport.get(0);
        if (a?.cssWidth !== 320 || a?.cssHeight !== 180 || a?.dpr !== 1)
            throw new Error("first viewport row was changed");
        if (b?.cssWidth !== 640 || b?.cssHeight !== 360 || b?.dpr !== 2)
            throw new Error("second viewport row was changed");
        first.dispose();
        second.dispose();
    },
);
