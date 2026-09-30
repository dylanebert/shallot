import { test } from "bun:test";
import {
    audioContextState,
    BrowserInputPlugin,
    blurCanvas,
    createBrowserInputPlugin,
    Devices,
    focusCanvas,
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
    setInputEnabled,
    Time,
    touchPoint,
    UpdatePlayerControlSystem,
    visibilityChanged,
    World,
} from "@dylanebert/shallot";
import { sizeView, type View } from "@dylanebert/shallot/rendering";

function inputState(): World {
    const world = new World();
    for (const system of InputPlugin.systems ?? []) world.addSystem(system, InputPlugin.name);
    return world;
}

function browserInputState(): World {
    const world = inputState();
    for (const system of BrowserInputPlugin.systems ?? [])
        world.addSystem(system, BrowserInputPlugin.name);
    return world;
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

function declaredHost(
    options: {
        failOn?: string;
        pendingLock?: boolean;
        pendingRequests?: number;
        pendingCanvas?: number;
        canvas?: HTMLCanvasElement;
        canvasCount?: number;
        lock?: { element: HTMLCanvasElement | null };
        documentTarget?: ReturnType<typeof fixtureTarget>;
    } = {},
) {
    const hostWindow = fixtureTarget();
    const hostDocument = options.documentTarget ?? fixtureTarget();
    const lock = options.lock ?? { element: null as HTMLCanvasElement | null };
    let hidden = false;
    let captured: number | null = null;
    let requestCount = 0;
    let releaseCount = 0;
    const canvasTargets = new Map<HTMLCanvasElement, ReturnType<typeof fixtureTarget>>();
    const pendingResolvers = new Map<
        HTMLCanvasElement,
        { resolve: () => void; reject: (error: unknown) => void }
    >();
    const makeCanvas = (target: ReturnType<typeof fixtureTarget>) =>
        ({
            style: { touchAction: "auto" },
            addEventListener(type: string, listener: FixtureListener) {
                if (options.failOn === type) throw new Error("fixture listener failure");
                target.addEventListener(type, listener);
            },
            removeEventListener(type: string, listener: FixtureListener) {
                target.removeEventListener(type, listener);
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
        }) as unknown as HTMLCanvasElement;
    const canvasTarget = fixtureTarget();
    const canvas = options.canvas ?? makeCanvas(canvasTarget);
    canvasTargets.set(canvas, canvasTarget);
    const canvases = [canvas];
    for (let i = 1; i < (options.canvasCount ?? 1); i++) {
        const target = fixtureTarget();
        const extra = makeCanvas(target);
        canvasTargets.set(extra, target);
        canvases.push(extra);
    }
    const pendingCanvas = () => pendingResolvers.keys().next().value ?? null;
    const host = {
        window: Object.assign(hostWindow, { focus() {} }) as unknown as Window,
        document: Object.defineProperties(hostDocument, {
            hidden: { configurable: true, get: () => hidden },
            pointerLockElement: { configurable: true, get: () => lock.element },
        }) as unknown as Document,
        queryCanvases: () => canvases,
        supportsPointerLock: () => true,
        requestPointerLock: (requested: HTMLCanvasElement) => {
            requestCount++;
            const index = canvases.indexOf(requested);
            if (
                (options.pendingLock && requestCount === 1) ||
                (options.pendingRequests !== undefined &&
                    requestCount <= options.pendingRequests) ||
                options.pendingCanvas === index
            )
                return new Promise<void>((resolve, reject) => {
                    pendingResolvers.set(requested, { resolve, reject });
                });
            lock.element = requested;
        },
        releasePointerLock: (owned: HTMLCanvasElement) => {
            if (lock.element === owned) {
                releaseCount++;
                lock.element = null;
            }
        },
    } satisfies InputHost;
    return {
        host,
        canvas,
        canvases,
        lock,
        window: hostWindow,
        document: hostDocument,
        emitLockChange() {
            hostDocument.emit("pointerlockchange");
        },
        exitLock() {
            lock.element = null;
            hostDocument.emit("pointerlockchange");
        },
        emitCanvas(
            type: string,
            event: Record<string, unknown> = {},
            targetCanvas: HTMLCanvasElement = canvas,
        ) {
            canvasTargets.get(targetCanvas)?.emit(type, { ...event, target: targetCanvas });
        },
        setHidden(value: boolean) {
            hidden = value;
        },
        rejectLock(error: unknown, targetCanvas = pendingCanvas()) {
            if (!targetCanvas) return;
            pendingResolvers.get(targetCanvas)?.reject(error);
        },
        resolveLock(targetCanvas = pendingCanvas()) {
            if (!targetCanvas) return;
            pendingResolvers.get(targetCanvas)?.resolve();
        },
        get requestCount() {
            return requestCount;
        },
        get releaseCount() {
            return releaseCount;
        },
        listenerCount() {
            let canvasListeners = 0;
            for (const target of canvasTargets.values()) canvasListeners += target.listenerCount();
            return hostWindow.listenerCount() + hostDocument.listenerCount() + canvasListeners;
        },
        captured() {
            return captured;
        },
    };
}

test("the production browser adapter bypasses shared input producers or loses focus, pointer, touch and visibility transitions", () => {
    const fixture = declaredHost();
    const world = inputState();
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    try {
        world.step(0);
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
        const input = world.resource(Devices);
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
        world.dispose();
        if (
            fixture.listenerCount() !== 0 ||
            fixture.canvas.style.touchAction !== "auto" ||
            fixture.captured() !== null
        )
            throw new Error("adapter teardown left listeners, capture or canvas state");
    } finally {
        world.dispose();
    }
});

function replaceGlobal(name: "document" | "window", value: unknown): () => void {
    const prior = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    return () => {
        if (prior) Object.defineProperty(globalThis, name, prior);
        else Reflect.deleteProperty(globalThis, name);
    };
}

test("a browser adapter setup exception leaves listeners or canvas capture effects installed", () => {
    const fixture = declaredHost({ failOn: "wheel" });
    const world = inputState();
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    const report = console.error;
    console.error = () => {};
    try {
        world.step(0);
    } finally {
        console.error = report;
    }
    world.dispose();
    if (fixture.listenerCount() !== 0 || fixture.canvas.style.touchAction !== "auto")
        throw new Error("partial adapter setup was not unwound");
});

test("a pointer-lock promise rejection after disposal changes the retired World or releases another canvas", async () => {
    const fixture = declaredHost({ pendingLock: true });
    const world = inputState();
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    world.step(0);
    requestPointerLock(world);
    const input = world.resource(Devices);
    world.dispose();
    fixture.rejectLock(new Error("late refusal"));
    await Promise.resolve();
    if (input.pointer.lock.status !== "unlocked" || input.pointer.lock.refusal !== null)
        throw new Error("late lock rejection changed retired facts");
    if (fixture.releaseCount !== 0 || fixture.listenerCount() !== 0)
        throw new Error("late lock rejection crossed the retired adapter boundary");
});

test("disposing and recreating a World leaves old listeners delivering input to the replacement", () => {
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
    if (!second.resource(Devices).keys.held.has("KeyR"))
        throw new Error("replacement adapter did not bind");
    second.dispose();
    if (fixture.listenerCount() !== 0) throw new Error("replacement adapter leaked listeners");
});

test("releasing one World's pointer lock affects a different World's canvas", () => {
    const firstFixture = declaredHost();
    const secondFixture = declaredHost();
    const first = inputState();
    const second = inputState();
    const firstPlugin = createBrowserInputPlugin(firstFixture.host);
    const secondPlugin = createBrowserInputPlugin(secondFixture.host);
    for (const system of firstPlugin.systems ?? []) first.addSystem(system, firstPlugin.name);
    for (const system of secondPlugin.systems ?? []) second.addSystem(system, secondPlugin.name);
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
});

test("pointer-lock exit clears the facts but leaves a live adapter's shared canvas permanently owned", () => {
    const lock = { element: null as HTMLCanvasElement | null };
    const firstFixture = declaredHost({ lock });
    const secondFixture = declaredHost({
        canvas: firstFixture.canvas,
        documentTarget: firstFixture.document,
        lock,
    });
    const first = inputState();
    const second = inputState();
    const firstPlugin = createBrowserInputPlugin(firstFixture.host);
    const secondPlugin = createBrowserInputPlugin(secondFixture.host);
    for (const system of firstPlugin.systems ?? []) first.addSystem(system, firstPlugin.name);
    for (const system of secondPlugin.systems ?? []) second.addSystem(system, secondPlugin.name);
    first.step(0);
    second.step(0);
    requestPointerLock(first);
    firstFixture.emitLockChange();
    if (pointerLockStatus(first) !== "locked") throw new Error("first lock did not engage");
    firstFixture.exitLock();
    if (pointerLockStatus(first) !== "unlocked") throw new Error("first lock did not exit");
    requestPointerLock(second);
    secondFixture.emitLockChange();
    if (secondFixture.requestCount !== 1 || pointerLockStatus(second) !== "locked")
        throw new Error("second live adapter could not acquire the released canvas");
    first.dispose();
    second.dispose();
});

test("a pending second-canvas pointer-lock request leaks the first canvas ownership or lets its late settlement affect a replacement adapter", async () => {
    const fixture = declaredHost({ canvasCount: 2, pendingCanvas: 1 });
    const first = inputState();
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) first.addSystem(system, plugin.name);
    first.step(0);

    requestPointerLock(first);
    fixture.emitLockChange();
    if (pointerLockStatus(first) !== "locked" || fixture.lock.element !== fixture.canvases[0])
        throw new Error("first canvas did not engage");

    fixture.emitCanvas(
        "pointerdown",
        { pointerId: 2, pointerType: "mouse", button: 0, buttons: 1, preventDefault() {} },
        fixture.canvases[1],
    );
    focusCanvas(first, 1);
    requestPointerLock(first);
    if (Number(fixture.requestCount) !== 2 || fixture.lock.element !== fixture.canvases[0])
        throw new Error("second canvas did not remain pending behind the first lock");

    first.dispose();
    if (fixture.releaseCount !== 1 || fixture.lock.element !== null)
        throw new Error("disposing the pending request did not release canvas A");

    const replacement = inputState();
    const replacementPlugin = createBrowserInputPlugin(fixture.host);
    for (const system of replacementPlugin.systems ?? [])
        replacement.addSystem(system, replacementPlugin.name);
    replacement.step(0);
    requestPointerLock(replacement);
    fixture.emitLockChange();
    if (pointerLockStatus(replacement) !== "locked" || fixture.lock.element !== fixture.canvases[0])
        throw new Error("replacement adapter could not reuse canvas A");

    fixture.resolveLock(fixture.canvases[1]);
    await Promise.resolve();
    if (
        pointerLockStatus(replacement) !== "locked" ||
        fixture.lock.element !== fixture.canvases[0] ||
        fixture.releaseCount !== 1
    )
        throw new Error("late canvas B settlement affected the replacement lock");
    replacement.dispose();
    if (Number(fixture.releaseCount) !== 2) throw new Error("replacement lock did not release");
});

test("a pointer-lock grant that arrives after adapter disposal leaves the retired canvas locked", async () => {
    const fixture = declaredHost({ pendingLock: true });
    const world = inputState();
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    world.step(0);
    requestPointerLock(world);
    world.dispose();

    fixture.lock.element = fixture.canvas;
    fixture.resolveLock(fixture.canvas);
    await Promise.resolve();
    if (fixture.releaseCount !== 1 || fixture.lock.element !== null)
        throw new Error("late retired grant was not released");
});

test("releasing a retired adapter's late pointer-lock grant releases a replacement adapter lock", async () => {
    const fixture = declaredHost({ pendingRequests: 1 });
    const retired = inputState();
    const retiredPlugin = createBrowserInputPlugin(fixture.host);
    for (const system of retiredPlugin.systems ?? []) retired.addSystem(system, retiredPlugin.name);
    retired.step(0);
    requestPointerLock(retired);
    retired.dispose();

    const replacement = inputState();
    const replacementPlugin = createBrowserInputPlugin(fixture.host);
    for (const system of replacementPlugin.systems ?? [])
        replacement.addSystem(system, replacementPlugin.name);
    replacement.step(0);
    requestPointerLock(replacement);
    fixture.emitLockChange();
    if (pointerLockStatus(replacement) !== "locked")
        throw new Error("replacement adapter did not acquire the canvas");

    fixture.lock.element = fixture.canvas;
    fixture.resolveLock(fixture.canvas);
    await Promise.resolve();
    if (
        pointerLockStatus(replacement) !== "locked" ||
        fixture.lock.element !== fixture.canvas ||
        fixture.releaseCount !== 0
    )
        throw new Error("late retired grant released the replacement lock");
    replacement.dispose();
    if (Number(fixture.releaseCount) !== 1) throw new Error("replacement lock did not release");
});

test("non-Player browser input requires lock, or Player lock behavior depends on direct DOM access", () => {
    const controlled = inputState();
    controlled.addSystem(UpdatePlayerControlSystem, "Player");
    controlled.step(0);
    const controlledDevices = controlled.resource(Devices);
    if (!controlledDevices.requireLock)
        throw new Error("Player did not retain its lock gate without a browser producer");
    controlled.dispose();
    if (controlledDevices.requireLock)
        throw new Error("Player disposal did not clear its lock intent");

    const fixture = declaredHost();
    const world = inputState();
    const _devices = world.resource(Devices);
    const plugin = createBrowserInputPlugin(fixture.host);
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
    world.addSystem(UpdatePlayerControlSystem, "Player");
    world.step(0);
    pointerButton(world, "left", true);
    if (_devices.mouse.left) throw new Error("Player lock gate did not hold before engagement");
    requestPointerLock(world);
    fixture.emitLockChange();
    pointerButton(world, "left", true);
    if (!_devices.mouse.left) throw new Error("Player lock gate did not open after engagement");
    world.dispose();
    if (fixture.releaseCount !== 1)
        throw new Error("Player disposal did not release through the adapter");
});

test("a World with the plain input owner binds browser listeners merely because browser globals exist", () => {
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
    const world = inputState();
    try {
        world.step(0);
        pressKey(world, "KeyW");
        world.step(Time.FIXED_DT);
        if (!world.resource(Devices).keys.held.has("KeyW"))
            throw new Error("application input was lost");
        if (queried !== 0 || listeners !== 0)
            throw new Error("host producer was composed implicitly");
    } finally {
        world.dispose();
        restoreWindow();
        restoreDocument();
    }
});

test("browser viewport and audio-status producers overwrite application facts when those producers are omitted", () => {
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
    const world = inputState();
    try {
        resizeViewport(world, 0, 100, 50, 2);
        audioContextState(world, "running");
        world.step(0);
        const input = world.resource(Devices);
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
        world.dispose();
        restoreWindow();
        restoreDocument();
    }
});

test("the browser input producer fails to bind listeners or request pointer lock when composed", () => {
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
    const world = browserInputState();
    const _devices = world.resource(Devices);
    try {
        world.step(0);
        pointerButton(world, "left", true);
        if (_devices.requireLock || !_devices.mouse.left)
            throw new Error("non-Player browser input unexpectedly required pointer lock");
        requestPointerLock(world);
        if (listeners === 0 || requested !== 1)
            throw new Error("browser input was not the default adapter");
    } finally {
        world.dispose();
        restoreWindow();
        restoreDocument();
    }
});

test("composing the browser input producer without host globals preserves the plain input owner's transitions", () => {
    const restoreDocument = replaceGlobal("document", undefined);
    const restoreWindow = replaceGlobal("window", undefined);
    const world = browserInputState();
    let pressed = false;
    let released = false;
    world.addSystem({
        group: "simulation",
        update(s: World) {
            const keys = s.resource(Devices).keys;
            pressed ||= keys.pressed.has("KeyW");
            released ||= keys.released.has("KeyW");
        },
    });
    try {
        world.step(0);
        pressKey(world, "KeyW");
        world.step(Time.FIXED_DT);
        releaseKey(world, "KeyW");
        world.step(0);
        if (!pressed || !released || world.resource(Devices).keys.held.has("KeyW"))
            throw new Error("browser composition lost shared input transitions");
    } finally {
        world.dispose();
        restoreWindow();
        restoreDocument();
    }
});

test("controlled input edges depend on frame cadence rather than the independent fixed and draw boundaries", () => {
    const world = inputState();
    const _devices = world.resource(Devices);
    const fixedSeen: string[] = [];
    world.addSystem({
        group: "fixed",
        update(s: World) {
            if (s.resource(Devices).keys.tickPressed.has("KeyA")) fixedSeen.push("A");
            if (s.resource(Devices).keys.tickPressed.has("KeyB")) fixedSeen.push("B");
        },
    });
    pressKey(world, "KeyA");
    world.step(0);
    if (!_devices.keys.tickPressed.has("KeyA")) throw new Error("zero-tick frame reset a press");
    world.step(Time.FIXED_DT * 2);
    pressKey(world, "KeyB");
    world.step(Time.FIXED_DT);
    if (fixedSeen.join("") !== "AB") throw new Error(`unexpected fixed edges: ${fixedSeen}`);
    if (_devices.keys.pressed.has("KeyA") || _devices.keys.tickPressed.size !== 0)
        throw new Error("draw or fixed edge reset did not run");
    world.dispose();
});

test("a key press, pointer, wheel or touch fact is lost before the frame's readers see it, or a release edge never appears", () => {
    const world = inputState();
    const seen: Array<{ held: boolean; pressed: boolean; released: boolean }> = [];
    const reader = {
        group: "simulation" as const,
        update(s: World) {
            const keys = s.resource(Devices).keys;
            seen.push({
                held: keys.held.has("KeyW"),
                pressed: keys.pressed.has("KeyW"),
                released: keys.released.has("KeyW"),
            });
        },
    };
    world.addSystem(reader);
    pressKey(world, "KeyW");
    pointerMove(world, 12, 24, 3, -2);
    pointerButton(world, "left", true);
    pointerWheel(world, 7);
    touchPoint(world, 1, 10, 20);
    world.step(Time.FIXED_DT);
    if (!seen[0]?.held || !seen[0].pressed || seen[0].released)
        throw new Error("press edge was not visible");
    const d = world.resource(Devices);
    if (d.mouse.x !== 12 || d.mouse.y !== 24 || d.mouse.deltaX !== 0 || d.mouse.scroll !== 0) {
        // frame latches are deliberately cleared at the draw boundary; the reader above observes them.
        throw new Error("frame device deltas were not cleared");
    }
    if (d.touch.count !== 1 || !d.mouse.left) throw new Error("pointer/touch fact was lost");
    releaseKey(world, "KeyW");
    const releaseReader = {
        group: "simulation" as const,
        update(s: World) {
            if (
                !s.resource(Devices).keys.released.has("KeyW") ||
                s.resource(Devices).keys.held.has("KeyW")
            ) {
                throw new Error("release edge was not visible");
            }
        },
    };
    world.addSystem(releaseReader);
    world.step(0);
    world.dispose();
});

test("a key press in a frame with zero fixed ticks is dropped before the next frame's first fixed tick", () => {
    const world = inputState();
    let count = 0;
    world.addSystem({
        group: "fixed",
        update(s: World) {
            if (s.resource(Devices).keys.tickPressed.has("KeyA")) count++;
        },
    });
    pressKey(world, "KeyA");
    world.step(0);
    world.step(Time.FIXED_DT);
    if (count !== 1) throw new Error(`expected carried edge, got ${count}`);
    world.dispose();
});

test("a press records a wall-clock timestamp instead of the World fixed tick, so replays diverge", () => {
    const world = inputState();
    world.step(Time.FIXED_DT);
    const tick = world.time.fixedTick;
    pressKey(world, "KeyB");
    if (world.resource(Devices).keys.pressedTick.get("KeyB") !== tick)
        throw new Error("wrong pressedTick");
    world.dispose();
});

test("a press and release between frames loses an edge or leaves the key held", () => {
    const world = inputState();
    let seen = false;
    world.addSystem({
        group: "simulation",
        update(s: World) {
            const keys = s.resource(Devices).keys;
            seen = keys.pressed.has("KeyQ") && keys.released.has("KeyQ") && !keys.held.has("KeyQ");
        },
    });
    pressKey(world, "KeyQ");
    releaseKey(world, "KeyQ");
    world.step(0);
    if (!seen) throw new Error("between-frame edges were not latched");
    world.dispose();
});

test("the same press yields a different fixed edge count under batched and one-tick-per-frame cadence", () => {
    const batched = inputState();
    const stepped = inputState();
    const count = (world: World) => {
        let value = 0;
        world.addSystem({
            group: "fixed",
            update(s: World) {
                if (s.resource(Devices).keys.tickPressed.has("KeyE")) value++;
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
    if (batchedCount() !== 1 || steppedCount() !== 1) throw new Error("cadence changed edge count");
    batched.dispose();
    stepped.dispose();
});

test("window blur leaves keys or pointer buttons held, or releases them without an edge", () => {
    const world = inputState();
    pressKey(world, "KeyW");
    pointerButton(world, "left", true);
    blurCanvas(world);
    const input = world.resource(Devices);
    if (input.keys.held.has("KeyW") || !input.keys.released.has("KeyW"))
        throw new Error("blur did not emit a key release edge");
    if (input.mouse.left) throw new Error("blur left a pointer button held");
    world.dispose();
});

test("hiding the page leaves keys or pointer buttons held, or releases them without an edge", () => {
    const world = inputState();
    focusCanvas(world, 0);
    pressKey(world, "KeyA");
    pointerButton(world, "right", true);
    visibilityChanged(world, true);
    const input = world.resource(Devices);
    if (input.keys.held.has("KeyA") || !input.keys.released.has("KeyA"))
        throw new Error("hidden visibility did not emit a key release edge");
    if (input.mouse.right) throw new Error("hidden visibility left a pointer button held");
    world.dispose();
});

test("pointer-lock exit leaves keys or pointer buttons held, or loses the lock refusal reason", () => {
    const world = inputState();
    const _devices = world.resource(Devices);
    pointerLockChanged(world, true);
    pressKey(world, "KeyD");
    pointerButton(world, "middle", true);
    pointerLockChanged(world, false);
    const input = _devices;
    if (pointerLockStatus(world) !== "unlocked") throw new Error("lock status did not exit");
    pointerLockChanged(world, true);
    pointerLockChanged(world, false, "denied by browser");
    if (
        pointerLockStatus(world) !== "refused" ||
        _devices.pointer.lock.refusal !== "denied by browser"
    )
        throw new Error("lock refusal was not recorded");
    if (input.keys.held.has("KeyD") || !input.keys.released.has("KeyD"))
        throw new Error("lock exit did not emit a key release edge");
    if (input.mouse.middle) throw new Error("lock exit left a pointer button held");
    world.dispose();
});

test("a pointer button reads down before a required pointer lock engages", () => {
    const world = inputState();
    const _devices = world.resource(Devices);
    requirePointerLock(world, true);
    pointerButton(world, "left", true);
    if (_devices.mouse.left) throw new Error("button crossed the lock gate");
    pointerLockChanged(world, true);
    pointerButton(world, "left", true);
    if (!_devices.mouse.left) throw new Error("locked button did not engage");
    world.dispose();
});

test("a suspended World still reads held keys, pointer or touch data, or accepts new presses", () => {
    const world = inputState();
    pressKey(world, "KeyS");
    pointerButton(world, "left", true);
    setInputEnabled(world, false);
    const input = world.resource(Devices);
    if (!input.suspended || input.keys.held.has("KeyS") || !input.keys.released.has("KeyS"))
        throw new Error("suspension did not neutralize the key with an edge");
    if (input.mouse.left || input.mouse.deltaX !== 0 || input.touch.count !== 0)
        throw new Error("suspension did not neutralize pointer/touch reads");
    pressKey(world, "KeyQ");
    if (input.keys.held.has("KeyQ")) throw new Error("suspended producer changed the record");
    world.dispose();
});

test("suspending one World suspends or clears a second World's device record", () => {
    const first = inputState();
    const second = inputState();
    pressKey(first, "KeyW");
    pressKey(second, "KeyW");
    setInputEnabled(first, false);
    if (!first.resource(Devices).suspended || first.resource(Devices).keys.held.has("KeyW"))
        throw new Error("first World did not suspend");
    if (second.resource(Devices).suspended || !second.resource(Devices).keys.held.has("KeyW"))
        throw new Error("second World was affected by suspension");
    first.dispose();
    second.dispose();
});

test("interleaved producer calls on two Worlds leak edges or held keys between their records", () => {
    const a = inputState();
    const b = inputState();
    pressKey(a, "KeyW");
    pressKey(b, "KeyW");
    a.step(Time.FIXED_DT);
    b.step(Time.FIXED_DT);
    if (a.resource(Devices).keys.held.has("KeyW") !== b.resource(Devices).keys.held.has("KeyW"))
        throw new Error("twin held mismatch");
    releaseKey(a, "KeyW");
    releaseKey(b, "KeyW");
    if (
        !a.resource(Devices).keys.released.has("KeyW") ||
        !b.resource(Devices).keys.released.has("KeyW")
    )
        throw new Error("twin release mismatch");
    a.dispose();
    b.dispose();
});

test("a World-scoped viewport row supplies CSS size and DPR to sizeView through backingSize", () => {
    const world = inputState();
    resizeViewport(world, 0, 100, 50, 2);
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
        pickingId: null,
        viewportIndex: 0,
        width: 0,
    } as View;
    sizeView(world, 1, view);
    if (canvas.width !== 200 || canvas.height !== 100)
        throw new Error(`expected 200x100 backing, got ${canvas.width}x${canvas.height}`);
    world.dispose();
});

test("the normalized pointer coordinate uses the World-scoped viewport row after resize", () => {
    const world = inputState();
    const _devices = world.resource(Devices);
    resizeViewport(world, 0, 100, 50, 1);
    focusCanvas(world, 0);
    pointerMove(world, 50, 25);
    if (_devices.mouse.normalizedX !== 0.5 || _devices.mouse.normalizedY !== 0.5)
        throw new Error("initial normalized pointer coordinate was wrong");
    resizeViewport(world, 0, 200, 100, 1);
    if (
        world.resource(Devices).mouse.normalizedX !== 0.25 ||
        world.resource(Devices).mouse.normalizedY !== 0.25
    )
        throw new Error("normalized pointer coordinate did not follow resize");
    world.dispose();
});

test("two Worlds hold independent per-canvas viewport records", () => {
    const first = inputState();
    const second = inputState();
    resizeViewport(first, 0, 320, 180, 1);
    resizeViewport(second, 0, 640, 360, 2);
    const a = first.resource(Devices).viewport.get(0);
    const b = second.resource(Devices).viewport.get(0);
    if (a?.cssWidth !== 320 || a?.cssHeight !== 180 || a?.dpr !== 1)
        throw new Error("first viewport row was changed");
    if (b?.cssWidth !== 640 || b?.cssHeight !== 360 || b?.dpr !== 2)
        throw new Error("second viewport row was changed");
    first.dispose();
    second.dispose();
});
