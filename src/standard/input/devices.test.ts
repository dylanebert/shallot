import {
    audioContextState,
    BrowserInputPlugin,
    blur,
    createBrowserInputPlugin,
    devices,
    focus,
    type InputHost,
    InputPlugin,
    pointerButton,
    pointerLockChanged,
    pointerLockStatus,
    pointerMove,
    pointerWheel,
    pressKey,
    releaseKey,
    releasePointerLock,
    requestPointerLock,
    requirePointerLock,
    resizeViewport,
    State,
    setInputEnabled,
    Time,
    touchPoint,
    visibilityChanged,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { sizeView, type View } from "@dylanebert/shallot/render";

function inputState(): State {
    const state = new State();
    for (const system of InputPlugin.systems ?? []) state.addSystem(system, InputPlugin.name);
    return state;
}

function browserInputState(): State {
    const state = inputState();
    for (const system of BrowserInputPlugin.systems ?? [])
        state.addSystem(system, BrowserInputPlugin.name);
    return state;
}

type FixtureListener = (event: any) => void;

function fixtureTarget() {
    const listeners = new Map<string, Set<FixtureListener>>();
    return {
        addEventListener(type: string, listener: FixtureListener) {
            let entries = listeners.get(type);
            if (!entries) listeners.set(type, (entries = new Set()));
            entries.add(listener);
        },
        removeEventListener(type: string, listener: FixtureListener) {
            listeners.get(type)?.delete(listener);
        },
        emit(type: string, event: Record<string, unknown> = {}) {
            for (const listener of listeners.get(type) ?? []) listener({ ...event, type });
        },
        listenerCount() {
            let count = 0;
            for (const entries of listeners.values()) count += entries.size;
            return count;
        },
    };
}

function declaredHost(options: { failOn?: string; pendingLock?: boolean } = {}) {
    const hostWindow = fixtureTarget();
    const hostDocument = fixtureTarget();
    const canvasTarget = fixtureTarget();
    let pointerLockElement: HTMLCanvasElement | null = null;
    let hidden = false;
    let captured: number | null = null;
    let requestCount = 0;
    let releaseCount = 0;
    let rejectLock: ((error: unknown) => void) | null = null;
    const canvas = {
        style: { touchAction: "auto" },
        addEventListener(type: string, listener: FixtureListener) {
            if (options.failOn === type) throw new Error("fixture listener failure");
            canvasTarget.addEventListener(type, listener);
        },
        removeEventListener(type: string, listener: FixtureListener) {
            canvasTarget.removeEventListener(type, listener);
        },
        setPointerCapture(pointerId: number) {
            captured = pointerId;
        },
        hasPointerCapture(pointerId: number) {
            return captured === pointerId;
        },
        releasePointerCapture(pointerId: number) {
            if (captured === pointerId) captured = null;
        },
        getBoundingClientRect: () => ({ left: 10, top: 20 }),
    } as unknown as HTMLCanvasElement;
    const host = {
        window: Object.assign(hostWindow, { focus() {} }) as unknown as Window,
        document: Object.defineProperties(hostDocument, {
            hidden: { configurable: true, get: () => hidden },
            pointerLockElement: { configurable: true, get: () => pointerLockElement },
        }) as unknown as Document,
        queryCanvases: () => [canvas],
        supportsPointerLock: () => true,
        requestPointerLock: () => {
            requestCount++;
            if (options.pendingLock)
                return new Promise<void>((_resolve, reject) => {
                    rejectLock = reject;
                });
            pointerLockElement = canvas;
        },
        releasePointerLock: (owned: HTMLCanvasElement) => {
            if (pointerLockElement === owned) {
                releaseCount++;
                pointerLockElement = null;
            }
        },
    } satisfies InputHost;
    return {
        host,
        canvas,
        window: hostWindow,
        document: hostDocument,
        emitLockChange() {
            hostDocument.emit("pointerlockchange");
        },
        emitCanvas(type: string, event: Record<string, unknown> = {}) {
            canvasTarget.emit(type, { ...event, target: canvas });
        },
        setHidden(value: boolean) {
            hidden = value;
        },
        rejectLock(error: unknown) {
            rejectLock?.(error);
        },
        get requestCount() {
            return requestCount;
        },
        get releaseCount() {
            return releaseCount;
        },
        listenerCount() {
            return (
                hostWindow.listenerCount() +
                hostDocument.listenerCount() +
                canvasTarget.listenerCount()
            );
        },
        captured() {
            return captured;
        },
    };
}

check(
    "declared browser host translates representative input events",
    {
        claim: "the production browser adapter bypasses shared input producers or loses focus, pointer, touch and visibility transitions",
    },
    () => {
        const fixture = declaredHost();
        const state = inputState();
        const plugin = createBrowserInputPlugin(fixture.host);
        for (const system of plugin.systems ?? []) state.addSystem(system, plugin.name);
        try {
            state.step(0);
            fixture.emitCanvas("pointerdown", {
                pointerId: 1,
                pointerType: "mouse",
                button: 0,
                buttons: 1,
                clientX: 30,
                clientY: 50,
                preventDefault() {},
            });
            fixture.window.emit("keydown", { code: "KeyW" });
            fixture.window.emit("pointermove", {
                target: fixture.canvas,
                pointerId: 1,
                buttons: 1,
                clientX: 35,
                clientY: 54,
                preventDefault() {},
            });
            fixture.emitCanvas("wheel", { target: fixture.canvas, deltaY: 7, preventDefault() {} });
            fixture.emitCanvas("pointerdown", {
                pointerId: 2,
                pointerType: "touch",
                button: 0,
                buttons: 1,
                clientX: 40,
                clientY: 60,
                preventDefault() {},
            });
            const input = devices(state);
            if (
                !input.keys.held.has("KeyW") ||
                input.focused !== 0 ||
                input.mouse.x !== 25 ||
                input.mouse.y !== 34 ||
                input.mouse.deltaX !== 5 ||
                input.mouse.deltaY !== 4 ||
                input.mouse.scroll !== 7 ||
                input.touch.count !== 1 ||
                fixture.captured() !== 1
            )
                throw new Error("declared host events did not reach shared producers");

            fixture.setHidden(true);
            fixture.document.emit("visibilitychange");
            if (input.keys.held.has("KeyW") || Number(input.touch.count) !== 0 || input.mouse.left)
                throw new Error("visibility did not release captured input");
            state.dispose();
            if (
                fixture.listenerCount() !== 0 ||
                fixture.canvas.style.touchAction !== "auto" ||
                fixture.captured() !== null
            )
                throw new Error("adapter teardown left listeners, capture or canvas state");
        } finally {
            state.dispose();
        }
    },
);

function replaceGlobal(name: "document" | "window", value: unknown): () => void {
    const prior = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    return () => {
        if (prior) Object.defineProperty(globalThis, name, prior);
        else Reflect.deleteProperty(globalThis, name);
    };
}

check(
    "adapter setup failure unwinds acquired effects",
    {
        claim: "a browser adapter setup exception leaves listeners or canvas capture effects installed",
    },
    () => {
        const fixture = declaredHost({ failOn: "wheel" });
        const state = inputState();
        const plugin = createBrowserInputPlugin(fixture.host);
        for (const system of plugin.systems ?? []) state.addSystem(system, plugin.name);
        const report = console.error;
        console.error = () => {};
        try {
            state.step(0);
        } finally {
            console.error = report;
        }
        state.dispose();
        if (fixture.listenerCount() !== 0 || fixture.canvas.style.touchAction !== "auto")
            throw new Error("partial adapter setup was not unwound");
    },
);

check(
    "late lock rejection cannot mutate a retired State",
    {
        claim: "a pointer-lock promise rejection after disposal changes the retired State or releases another canvas",
    },
    async () => {
        const fixture = declaredHost({ pendingLock: true });
        const state = inputState();
        const plugin = createBrowserInputPlugin(fixture.host);
        for (const system of plugin.systems ?? []) state.addSystem(system, plugin.name);
        state.step(0);
        requestPointerLock(state);
        state.dispose();
        fixture.rejectLock(new Error("late refusal"));
        await Promise.resolve();
        const input = devices(state);
        if (input.pointer.lock.status !== "unlocked" || input.pointer.lock.refusal !== null)
            throw new Error("late lock rejection changed retired facts");
        if (fixture.releaseCount !== 0 || fixture.listenerCount() !== 0)
            throw new Error("late lock rejection crossed the retired adapter boundary");
    },
);

check(
    "fresh State does not inherit a retired browser adapter",
    {
        claim: "disposing and recreating a State leaves old listeners delivering input to the replacement",
    },
    () => {
        const fixture = declaredHost();
        const first = inputState();
        const plugin = createBrowserInputPlugin(fixture.host);
        for (const system of plugin.systems ?? []) first.addSystem(system, plugin.name);
        first.step(0);
        first.dispose();
        const second = inputState();
        const replacement = createBrowserInputPlugin(fixture.host);
        for (const system of replacement.systems ?? []) second.addSystem(system, replacement.name);
        second.step(0);
        fixture.window.emit("keydown", { code: "KeyR" });
        if (!devices(second).keys.held.has("KeyR"))
            throw new Error("replacement adapter did not bind");
        second.dispose();
        if (fixture.listenerCount() !== 0) throw new Error("replacement adapter leaked listeners");
    },
);

check(
    "lock release stays with its adapter canvas",
    {
        claim: "releasing one State's pointer lock affects a different State's canvas",
    },
    () => {
        const firstFixture = declaredHost();
        const secondFixture = declaredHost();
        const first = inputState();
        const second = inputState();
        const firstPlugin = createBrowserInputPlugin(firstFixture.host);
        const secondPlugin = createBrowserInputPlugin(secondFixture.host);
        for (const system of firstPlugin.systems ?? []) first.addSystem(system, firstPlugin.name);
        for (const system of secondPlugin.systems ?? [])
            second.addSystem(system, secondPlugin.name);
        first.step(0);
        second.step(0);
        requestPointerLock(first);
        firstFixture.emitLockChange();
        first.dispose();
        if (firstFixture.releaseCount !== 1)
            throw new Error("adapter disposal did not release its own lock");
        requestPointerLock(second);
        secondFixture.emitLockChange();
        releasePointerLock(second);
        if (secondFixture.releaseCount !== 1 || firstFixture.releaseCount !== 1)
            throw new Error("lock release escaped its adapter");
        second.dispose();
    },
);

check(
    "input data owner omits browser producer even when a host is present",
    {
        claim: "a State with the plain input owner binds browser listeners merely because browser globals exist",
    },
    () => {
        let queried = 0;
        let listeners = 0;
        const canvas = {
            style: { touchAction: "" },
            addEventListener: () => listeners++,
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
            state.step(0);
            pressKey(state, "KeyW");
            state.step(Time.FIXED_DT);
            if (!devices(state).keys.held.has("KeyW"))
                throw new Error("application input was lost");
            if (queried !== 0 || listeners !== 0)
                throw new Error("host producer was composed implicitly");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "omitted host reports preserve supplied viewport and audio facts",
    {
        claim: "browser viewport and audio-status producers overwrite application facts when those producers are omitted",
    },
    () => {
        let listeners = 0;
        const canvas = {
            style: { touchAction: "" },
            addEventListener: () => listeners++,
        } as unknown as HTMLCanvasElement;
        const restoreDocument = replaceGlobal("document", {
            querySelectorAll: () => [canvas],
            addEventListener: () => listeners++,
        });
        const restoreWindow = replaceGlobal("window", { addEventListener: () => listeners++ });
        const state = inputState();
        try {
            resizeViewport(state, 0, 100, 50, 2);
            audioContextState(state, "running");
            state.step(0);
            const input = devices(state);
            const viewport = input.viewport.get(0);
            if (
                viewport?.cssWidth !== 100 ||
                viewport.cssHeight !== 50 ||
                viewport.dpr !== 2 ||
                input.audio.context !== "running" ||
                listeners !== 0
            )
                throw new Error("omitted host reports changed supplied facts");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "browser producer remains an explicit ordinary composition",
    {
        claim: "the browser input producer fails to bind listeners or request pointer lock when composed",
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
        const state = browserInputState();
        try {
            state.step(0);
            requestPointerLock(state);
            if (listeners === 0 || requested !== 1)
                throw new Error("browser input was not the default adapter");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "browser producer composes without host globals",
    {
        claim: "composing the browser input producer without host globals preserves the plain input owner's transitions",
    },
    () => {
        const restoreDocument = replaceGlobal("document", undefined);
        const restoreWindow = replaceGlobal("window", undefined);
        const state = browserInputState();
        let pressed = false;
        let released = false;
        state.addSystem({
            group: "simulation",
            update(s: State) {
                const keys = devices(s).keys;
                pressed ||= keys.pressed.has("KeyW");
                released ||= keys.released.has("KeyW");
            },
        });
        try {
            state.step(0);
            pressKey(state, "KeyW");
            state.step(Time.FIXED_DT);
            releaseKey(state, "KeyW");
            state.step(0);
            if (!pressed || !released || devices(state).keys.held.has("KeyW"))
                throw new Error("browser composition lost shared input transitions");
        } finally {
            state.dispose();
            restoreWindow();
            restoreDocument();
        }
    },
);

check(
    "controlled edges survive zero and multiple fixed ticks",
    {
        claim: "controlled input edges depend on frame cadence rather than the independent fixed and draw boundaries",
    },
    () => {
        const state = inputState();
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
