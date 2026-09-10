import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { State, type System } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { attach } from "../../testing/helpers";
import { moves } from "../character/drive";
import { InputPlugin, Inputs, setInputEnabled } from "../input";
import { Body } from "../physics";
import { Player, PlayerControlSystem, pointerLockRefusal, pointerLockStatus } from "./index";

// biome-ignore lint/complexity/noBannedTypes: test mock tracks arbitrary DOM listeners
type Fn = Function;

class ListenerTracker {
    added: [string, Fn][] = [];
    addEventListener = (type: string, fn: Fn) => {
        this.added.push([type, fn]);
    };
    removeEventListener = () => {};
}

function mockCanvas(): HTMLCanvasElement & { tracker: ListenerTracker } {
    const tracker = new ListenerTracker();
    return {
        addEventListener: tracker.addEventListener,
        removeEventListener: tracker.removeEventListener,
        requestPointerLock: () => Promise.resolve(),
        tracker,
    } as unknown as HTMLCanvasElement & { tracker: ListenerTracker };
}

// Scheduler-driven validation of the camera follow (PlayerSnapshotSystem + followPos), against the REAL
// scheduler, no GPU. A value that advances on the FIXED clock, snapshotted into prev/curr in the `fixed`
// group and read as lerp(prev, curr, fixedAlpha) in the sim group, renders at constant velocity across an
// irregular render rate. The naive alternative this replaced — reading the latest fixed value every render
// frame — steps at the fixed rate (the jitter). This is the property that makes the follow render-rate-
// independent; it would go red if the snapshot moved off the fixed group or the lerp dropped fixedAlpha.

function makeRng(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
        return s / 0x7fffffff;
    };
}

describe("camera follow interpolation (scheduler)", () => {
    test("fixed-clock snapshot + fixedAlpha renders constant velocity across an irregular render rate", () => {
        const V = 6; // m/s — the swept pose moves at constant velocity, so ideal camera velocity is constant
        const FixedDt = 1 / 60;
        const state = new State();

        let pose = 0; // the swept pose, advances one step per fixed tick
        let prev = 0;
        let curr = 0;
        let init = false;
        const lerpCam: number[] = [];
        const naiveCam: number[] = [];
        const dts: number[] = [];

        state.addSystem({
            group: "fixed",
            update: () => {
                pose += V * FixedDt;
            },
        } satisfies System);
        // the fix (PlayerSnapshotSystem): snapshot prev/curr on the fixed clock, after the step
        state.addSystem({
            group: "fixed",
            last: true,
            update: () => {
                if (!init) {
                    prev = curr = pose;
                    init = true;
                } else {
                    prev = curr;
                    curr = pose;
                }
            },
        } satisfies System);
        // the camera reads both: the interpolated follow (followPos) + the naive latest-value read it replaces
        state.addSystem({
            group: "simulation",
            update: (s) => {
                lerpCam.push(init ? prev + (curr - prev) * s.time.fixedAlpha : 0);
                naiveCam.push(pose);
            },
        } satisfies System);

        // drive at an irregular high frame rate (measured on the retired WSL seat: ~238fps, dt cv ~12%, 0.6/8.7ms outliers)
        const rng = makeRng(1);
        for (let f = 0; f < 800; f++) {
            let dt = (1 / 238) * (1 + 0.4 * (rng() * 2 - 1));
            if (rng() < 0.05) dt = 0.0006;
            if (rng() < 0.05) dt = 0.0087;
            dts.push(dt);
            state.step(dt);
        }

        const vel = (cam: number[]) => cam.slice(1).map((x, i) => (x - cam[i]) / dts[i + 1]);
        const lv = vel(lerpCam).slice(150); // steady state, past the ring fill
        const nv = vel(naiveCam).slice(150);

        // the fix: constant velocity every frame (the pose is linear, so the lerp is exact — f64 roundoff only)
        for (const v of lv) {
            expect(v).toBeGreaterThan(0); // forward only
            expect(v).toBeCloseTo(V, 4); // constant speed at any render rate
        }
        // the naive per-frame read it replaces is NOT smooth: it holds (0 velocity) whole frames then spikes
        // a full fixed step when a tick lands — the stutter the interpolation removes
        expect(Math.min(...nv)).toBe(0);
        expect(Math.max(...nv)).toBeGreaterThan(V * 2);

        state.dispose();
    });
});

// PlayerControlSystem.onDispose used to null the module-level `lock` unconditionally: a rebuild-before-
// dispose ordering (a new State's setup runs before the old State's dispose — the live-host rebuild shape
// ecs.md "Reload-safety" describes) let the OLD State's teardown clear the NEW State's live pointer-lock
// ref out from under it. The `standard/input` module guards its own module singleton the same way
// (`if (inputState === s) inputState = null`) — Player's dispose needed the identical identity check.
describe("PlayerControlSystem pointer-lock dispose identity", () => {
    test("disposing a stale State does not clear a newer State's live lock", () => {
        const canvasA = mockCanvas();
        const canvasB = mockCanvas();
        let currentCanvas: typeof canvasA = canvasA;
        const docTracker = new ListenerTracker();
        let exitCalls = 0;

        const savedDocument = globalThis.document;
        globalThis.document = {
            pointerLockElement: null,
            querySelector: () => currentCanvas,
            addEventListener: docTracker.addEventListener,
            removeEventListener: docTracker.removeEventListener,
            exitPointerLock: () => {
                exitCalls++;
            },
        } as unknown as typeof document;

        try {
            const onDocument = (type: string, nth: number): Fn =>
                docTracker.added.filter(([t]) => t === type)[nth][1];

            const stateA = new State();
            currentCanvas = canvasA;
            PlayerControlSystem.setup!(stateA);

            // the rebuild: a newer State's setup binds before the old one's dispose runs
            const stateB = new State();
            currentCanvas = canvasB;
            PlayerControlSystem.setup!(stateB);

            // B's pointer lock engages
            (globalThis.document as { pointerLockElement: unknown }).pointerLockElement = canvasB;
            onDocument("pointerlockchange", 1)();

            // A tears down — must not touch B's live, locked pointer-lock state
            stateA.dispose();

            expect(exitCalls).toBe(0); // A's own lock was never engaged, so no spurious exitPointerLock

            // B's lock is still the live one: disposing B now (its OWN teardown) is what releases it
            stateB.dispose();
            expect(exitCalls).toBe(1); // B's own teardown, correctly attributed
        } finally {
            globalThis.document = savedDocument;
        }
    });
});

// Pointer-lock capability refusal. `requestPointerLock` is not universal: touch-only browsers and some
// WebViews omit it entirely, older implementations return void rather than a promise, and a live request
// can be rejected (no user gesture, a sandboxed frame, an exit too recent). The controller used to call
// `canvas.requestPointerLock().catch(...)` unguarded, so the missing method threw a TypeError inside the
// click listener, the void return threw on `.catch`, and a rejection vanished — while `requirePointerLock`
// held every mouse button at 0 forever, leaving a game that looks playable and never aims. These legs drive
// the four API shapes through the REAL input plugin (buttons read through its production gate) crossed with
// the input-enabled gate, and read the refusal back through the exported status data.
describe("PlayerControlSystem pointer-lock capability", () => {
    type Shape = "absent" | "void" | "rejecting" | "resolving";

    const Rejection = "pointer lock refused by the browser";

    interface LockCanvas {
        canvas: HTMLCanvasElement & { tracker: ListenerTracker };
        requests: () => number;
    }

    const Rect: DOMRect = {
        x: 0,
        y: 0,
        left: 0,
        top: 0,
        width: 800,
        height: 600,
        right: 800,
        bottom: 600,
        toJSON() {},
    } as DOMRect;

    function lockCanvas(shape: Shape): LockCanvas {
        const tracker = new ListenerTracker();
        let requests = 0;
        const request = () => {
            requests++;
            if (shape === "void") return undefined;
            if (shape === "rejecting") return Promise.reject(new Error(Rejection));
            return Promise.resolve();
        };
        const canvas = {
            addEventListener: tracker.addEventListener,
            removeEventListener: tracker.removeEventListener,
            setPointerCapture() {},
            releasePointerCapture() {},
            hasPointerCapture: () => false,
            getBoundingClientRect: () => Rect,
            style: {} as CSSStyleDeclaration,
            ...(shape === "absent" ? {} : { requestPointerLock: request }),
            tracker,
        } as unknown as HTMLCanvasElement & { tracker: ListenerTracker };
        return { canvas, requests: () => requests };
    }

    interface Leg {
        state: State;
        canvas: HTMLCanvasElement & { tracker: ListenerTracker };
        requests: () => number;
        exits: () => number;
        click: () => void;
        press: (buttons: number) => void;
        key: (code: string) => void;
        setLock: (on: boolean) => void;
        player: number;
        teardown: () => void;
    }

    function start(shape: Shape): Leg {
        clear();
        const { canvas, requests } = lockCanvas(shape);
        const windowTracker = new ListenerTracker();
        (windowTracker as unknown as { focus: () => void }).focus = () => {};
        const docTracker = new ListenerTracker();
        const savedWindow = globalThis.window;
        const savedDocument = globalThis.document;
        globalThis.window = windowTracker as unknown as typeof window;
        let exits = 0;
        globalThis.document = {
            pointerLockElement: null,
            querySelector: (sel: string) => (sel === "canvas" ? canvas : null),
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
            addEventListener: docTracker.addEventListener,
            removeEventListener: docTracker.removeEventListener,
            exitPointerLock: () => {
                exits++;
            },
        } as unknown as typeof document;

        const state = new State();
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        attach(state, InputPlugin); // the real input plugin: buttons come through its production gate
        state.step();
        setInputEnabled(true);
        PlayerControlSystem.setup!(state);

        const on = (t: ListenerTracker, type: string): Fn =>
            t.added.filter(([k]) => k === type).at(-1)![1];
        const player = state.create();
        state.add(player, Player);
        state.add(player, Body);
        Player.speed.set(player, 6);
        Player.sensitivity.set(player, 1.5);

        return {
            state,
            canvas,
            requests,
            exits: () => exits,
            click: () => on(canvas.tracker, "click")(),
            press: (buttons: number) =>
                on(
                    canvas.tracker,
                    "pointerdown",
                )({
                    target: canvas,
                    pointerId: 1,
                    pointerType: "mouse",
                    button: 0,
                    buttons,
                    clientX: 10,
                    clientY: 10,
                    preventDefault() {},
                }),
            key: (code: string) => on(windowTracker, "keydown")({ code, preventDefault() {} }),
            setLock: (lockOn: boolean) => {
                (globalThis.document as { pointerLockElement: unknown }).pointerLockElement = lockOn
                    ? canvas
                    : null;
                on(docTracker, "pointerlockchange")();
            },
            player,
            teardown: () => {
                state.dispose();
                moves.clear();
                globalThis.window = savedWindow;
                globalThis.document = savedDocument;
            },
        };
    }

    // a rejected request must be caught by the controller, never escape as an unhandled rejection
    const rejections: unknown[] = [];
    const onRejection = (e: unknown) => rejections.push(e);
    beforeEach(() => {
        rejections.length = 0;
        process.on("unhandledRejection", onRejection);
    });
    afterEach(() => {
        process.off("unhandledRejection", onRejection);
        setInputEnabled(true);
    });

    const settle = async () => {
        await Promise.resolve();
        await Promise.resolve();
        await new Promise((r) => setTimeout(r, 0));
    };

    test("missing requestPointerLock never calls it, never gates buttons, and still walks", async () => {
        const leg = start("absent");
        try {
            expect(() => leg.click()).not.toThrow();
            await settle();
            expect(leg.requests()).toBe(0); // never called: there is nothing to call
            expect(pointerLockStatus()).toBe("unsupported");
            expect(pointerLockRefusal()).toBe("canvas has no requestPointerLock");
            expect(rejections).toEqual([]);

            // the desktop button gate must NOT strand a device that can never lock
            leg.press(1);
            expect(Inputs.mouse.left).toBe(true);

            // keyboard move still resolves: KeyW reaches the controller and writes a drive intent
            leg.key("KeyW");
            expect(Inputs.isKeyDown("KeyW")).toBe(true);
            PlayerControlSystem.update!(leg.state);
            const m = moves.get(leg.player)!;
            expect(Math.hypot(m[0], m[1])).toBeCloseTo(6, 6);
        } finally {
            leg.teardown();
        }
    });

    test("a void-returning implementation requests without a .catch and locks on pointerlockchange", async () => {
        const leg = start("void");
        try {
            expect(() => leg.click()).not.toThrow();
            await settle();
            expect(leg.requests()).toBe(1);
            expect(pointerLockStatus()).toBe("unlocked"); // requested, unpromised, not yet engaged
            expect(pointerLockRefusal()).toBeNull();
            expect(rejections).toEqual([]);

            leg.press(1); // gated until the lock engages
            expect(Inputs.mouse.left).toBe(false);

            leg.setLock(true);
            expect(pointerLockStatus()).toBe("locked");
            leg.press(1);
            expect(Inputs.mouse.left).toBe(true);
        } finally {
            leg.teardown();
        }
    });

    test("a rejected request becomes a visible refusal, not an unhandled rejection, and retries", async () => {
        const leg = start("rejecting");
        try {
            expect(() => leg.click()).not.toThrow();
            await settle();
            expect(leg.requests()).toBe(1);
            expect(pointerLockStatus()).toBe("refused");
            expect(pointerLockRefusal()).toBe(Rejection);
            expect(rejections).toEqual([]);

            leg.click(); // the next click may retry — the browser owns the throttle
            await settle();
            expect(leg.requests()).toBe(2);
            expect(pointerLockStatus()).toBe("refused");
            expect(rejections).toEqual([]);
        } finally {
            leg.teardown();
        }
    });

    test("a supported request keeps the gate: locked look, gated buttons until lock, clean dispose", async () => {
        const leg = start("resolving");
        try {
            leg.click();
            await settle();
            expect(leg.requests()).toBe(1);
            expect(pointerLockStatus()).toBe("unlocked");

            leg.press(1);
            expect(Inputs.mouse.left).toBe(false); // the capturing click only focuses

            leg.setLock(true);
            expect(pointerLockStatus()).toBe("locked");
            expect(pointerLockRefusal()).toBeNull();
            leg.press(1);
            expect(Inputs.mouse.left).toBe(true);
            expect(rejections).toEqual([]);

            leg.state.dispose(); // exits the engaged lock and drops the module ref
            expect(leg.exits()).toBe(1);
            expect(pointerLockStatus()).toBe("unlocked");
            expect(pointerLockRefusal()).toBeNull();
        } finally {
            leg.teardown();
        }
    });

    // the second dimension: the input-enabled gate. Suspended input never requests a lock, whatever the
    // API shape, and a refusal already recorded stays readable — the product of the two axes.
    for (const shape of ["absent", "void", "rejecting", "resolving"] as const) {
        test(`suspended input makes no ${shape} pointer-lock request`, async () => {
            const leg = start(shape);
            try {
                setInputEnabled(false);
                expect(() => leg.click()).not.toThrow();
                await settle();
                expect(leg.requests()).toBe(0);
                expect(pointerLockStatus()).toBe(shape === "absent" ? "unsupported" : "unlocked");
                expect(rejections).toEqual([]);

                // re-enabled, the same click takes its normal branch (non-vacuity for the gate above)
                setInputEnabled(true);
                leg.click();
                await settle();
                expect(leg.requests()).toBe(shape === "absent" ? 0 : 1);
            } finally {
                leg.teardown();
            }
        });
    }
});
