import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import {
    InputPlugin,
    Inputs,
    inputEnabled,
    requirePointerLock,
    State,
    setInputEnabled,
} from "../..";
import { clear, register } from "../../engine/ecs/core";

// biome-ignore lint/complexity/noBannedTypes: test mock tracks arbitrary DOM listeners
type Fn = Function;

class ListenerTracker {
    added: [string, Fn][] = [];
    removed: [string, Fn][] = [];

    addEventListener = (type: string, fn: Fn, _opts?: unknown) => {
        this.added.push([type, fn]);
    };
    removeEventListener = (type: string, fn: Fn) => {
        this.removed.push([type, fn]);
    };
}

function mockCanvas(): HTMLCanvasElement & { tracker: ListenerTracker } {
    const tracker = new ListenerTracker();
    return {
        addEventListener: tracker.addEventListener,
        removeEventListener: tracker.removeEventListener,
        setPointerCapture() {},
        releasePointerCapture() {},
        hasPointerCapture() {
            return false;
        },
        style: {} as CSSStyleDeclaration,
        tracker: tracker,
    } as unknown as HTMLCanvasElement & { tracker: ListenerTracker };
}

// Input exposes no logic to test apart from the real plugin: `Inputs` reads an internal `InputState`
// the DOM event handlers mutate. So the spec drives the *real* handlers (the ones `setup` attaches to
// the listener trackers) with synthetic events and reads `Inputs` back — the same path a browser walks.
// There is deliberately no mock of the input state machine: a mock would only test the mock.
describe("InputPlugin", () => {
    let state: State;
    let canvas: ReturnType<typeof mockCanvas>;
    let windowTracker: ListenerTracker;
    let savedWindow: typeof globalThis.window;
    let savedDocument: typeof globalThis.document;

    beforeEach(() => {
        clear();
        windowTracker = new ListenerTracker();
        (windowTracker as unknown as { focus: () => void }).focus = () => {};
        savedWindow = globalThis.window;
        savedDocument = globalThis.document;
        globalThis.window = windowTracker as unknown as typeof window;

        canvas = mockCanvas();
        // input binds to document <canvas> elements directly (no legacy viewport)
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
        } as unknown as typeof document;
        state = new State();
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        attach(state, InputPlugin);

        state.step();
    });

    afterEach(() => {
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    // the real handler `setup` bound for an event, pulled off the tracker it attached to
    const onWindow = (type: string): Fn => windowTracker.added.find(([t]) => t === type)![1];
    const onCanvas = (type: string): Fn => canvas.tracker.added.find(([t]) => t === type)![1];

    // simulate the browser engaging/releasing pointer lock on the canvas
    const setLock = (on: boolean): void => {
        (globalThis.document as { pointerLockElement: unknown }).pointerLockElement = on
            ? canvas
            : null;
    };

    test("setup populates Inputs singleton", () => {
        expect(Inputs.mouse).toBeDefined();
        expect(Inputs.focused).toBe(0);
    });

    test("attaches canvas listeners on setup", () => {
        const types = canvas.tracker.added.map(([t]) => t).sort();
        expect(types).toEqual([
            "contextmenu",
            "pointerdown",
            "pointerenter",
            "pointerleave",
            "pointermove",
            "wheel",
        ]);
    });

    test("attaches global listeners on setup", () => {
        const types = windowTracker.added.map(([t]) => t).sort();
        expect(types).toEqual([
            "blur",
            "keydown",
            "keyup",
            "pointercancel",
            "pointerdown",
            "pointermove",
            "pointerup",
        ]);
    });

    test("dispose removes all canvas listeners", () => {
        state.dispose();
        const added = canvas.tracker.added.map(([t, fn]) => [t, fn]);
        const removed = canvas.tracker.removed.map(([t, fn]) => [t, fn]);
        for (const [type, fn] of added) {
            expect(removed).toContainEqual([type, fn]);
        }
    });

    test("dispose removes all global listeners", () => {
        state.dispose();
        const added = windowTracker.added.map(([t, fn]) => [t, fn]);
        const removed = windowTracker.removed.map(([t, fn]) => [t, fn]);
        for (const [type, fn] of added) {
            expect(removed).toContainEqual([type, fn]);
        }
    });

    test("keyboard events flow through to Inputs", () => {
        onWindow("keydown")({ code: "KeyW" });
        expect(Inputs.isKeyDown("KeyW")).toBe(true);
        expect(Inputs.isKeyPressed("KeyW")).toBe(true);
    });

    test("a repeated keydown does not re-fire isKeyPressed", () => {
        // keyDown adds to keysPressed only when the key wasn't already held — the edge-trigger guard.
        // a held key autorepeats keydown events; isKeyPressed must stay a single-frame pulse
        onWindow("keydown")({ code: "KeyA" });
        expect(Inputs.isKeyPressed("KeyA")).toBe(true);

        state.step(); // InputResetSystem clears the per-frame pulse; the key stays held
        expect(Inputs.isKeyPressed("KeyA")).toBe(false);
        expect(Inputs.isKeyDown("KeyA")).toBe(true);

        onWindow("keydown")({ code: "KeyA" }); // still held → no second pulse
        expect(Inputs.isKeyPressed("KeyA")).toBe(false);
        expect(Inputs.isKeyDown("KeyA")).toBe(true);
    });

    test("a blur clears held keys and gates further keys until refocus", () => {
        onWindow("keydown")({ code: "KeyW" });
        expect(Inputs.isKeyDown("KeyW")).toBe(true);

        onWindow("blur")();
        expect(Inputs.isKeyDown("KeyW")).toBe(false); // held keys release on focus loss

        onWindow("keydown")({ code: "KeyA" });
        expect(Inputs.isKeyDown("KeyA")).toBe(false); // ignored while unfocused
    });

    test("setInputEnabled suspends every read, then restores on resume", () => {
        onWindow("keydown")({ code: "KeyW" });
        expect(Inputs.isKeyDown("KeyW")).toBe(true);
        expect(inputEnabled()).toBe(true);

        setInputEnabled(false);
        expect(inputEnabled()).toBe(false);
        expect(Inputs.isKeyDown("KeyW")).toBe(false); // a held key reads up while suspended
        onWindow("keydown")({ code: "KeyD" });
        expect(Inputs.isKeyDown("KeyD")).toBe(false); // a fresh press is ignored too

        setInputEnabled(true);
        expect(inputEnabled()).toBe(true);
        onWindow("keydown")({ code: "KeyD" });
        expect(Inputs.isKeyDown("KeyD")).toBe(true); // control resumes
    });

    test("a fresh input bind starts enabled — never inherits a suspended gate", () => {
        setInputEnabled(false);
        expect(inputEnabled()).toBe(false);

        // a new State re-runs InputSystem.setup, the per-State (re)bind
        state.dispose();
        clear();
        const s = new State();
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        attach(s, InputPlugin);
        s.step();
        expect(inputEnabled()).toBe(true);
    });

    test("requirePointerLock: the click that engages the lock only focuses; locked clicks register", () => {
        // a pointer-lock controller (Player) gates buttons on lock: the click that captures the
        // pointer must NOT latch a button, or a downstream click-command (a gun's shoot/grab) reads
        // the rising edge and misfires on the focus click. only clicks made once locked count.
        requirePointerLock(true);

        // unlocked: the focus/capture click reports no button (it still focuses + requests lock)
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        expect(Inputs.mouse.left).toBe(false);
        expect(Inputs.focused).toBe(0); // but focus is established
        onWindow("pointerup")({ pointerId: 1, button: 0, buttons: 0, preventDefault() {} });

        // the lock engages — now a press registers as a command
        setLock(true);
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        expect(Inputs.mouse.left).toBe(true);
    });

    test("pointer down captures the pointer and tracks the pressed button", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 200,
            preventDefault() {},
        });
        expect(Inputs.mouse.left).toBe(true);
        expect(Inputs.focused).toBe(0); // focuses the canvas it captured on
    });

    test("button state tracks the pointer bitmask, not the click button", () => {
        // production reads `e.buttons` (the held-button bitmask), so a right-click sets `right`
        // with no left press — inferring from the single `button` field would mis-set it
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 2,
            buttons: 2,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        expect(Inputs.mouse.right).toBe(true);
        expect(Inputs.mouse.left).toBe(false);
    });

    test("pointer capture binds one pointer: a second pointer's move is ignored", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onWindow("pointermove")({
            pointerId: 1,
            buttons: 1,
            clientX: 110,
            clientY: 100,
            preventDefault() {},
        });
        // a different pointerId is not the captured one — its motion must not leak into the delta
        onWindow("pointermove")({
            pointerId: 2,
            buttons: 1,
            clientX: 200,
            clientY: 200,
            preventDefault() {},
        });
        expect(Inputs.mouse.deltaX).toBe(10); // only the captured pointer's 100→110 move
    });

    test("under pointer lock a held-button move does not cancel pointermove — keeps the compat mousemove the look reads", () => {
        // canceling pointermove suppresses the compatibility mousemove a pointer-lock controller (Player)
        // reads its look from; under lock there's nothing to prevent, so the handler must NOT cancel — else
        // holding a mouse button silences mouse-look. The handler only reaches the cancel branch with a
        // pointer captured (activePointerId set), so press first.
        requirePointerLock(true);
        setLock(true);
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        let canceled = false;
        onWindow("pointermove")({
            pointerId: 1,
            buttons: 1,
            clientX: 5,
            clientY: 5,
            preventDefault() {
                canceled = true;
            },
        });
        expect(canceled).toBe(false);
    });

    test("a held-button move still cancels pointermove when not locked — the skip is gated on actual lock", () => {
        // requireLock on but the lock not yet engaged (the focus-click frame, or a non-locked drag): the
        // selection/scroll guard stays, so the skip needs BOTH requireLock AND a live lock.
        requirePointerLock(true);
        setLock(false);
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        let canceled = false;
        onWindow("pointermove")({
            pointerId: 1,
            buttons: 1,
            clientX: 5,
            clientY: 5,
            preventDefault() {
                canceled = true;
            },
        });
        expect(canceled).toBe(true);
    });

    test("wheel events flow through to Inputs", () => {
        onCanvas("wheel")({ target: canvas, deltaY: 120, preventDefault() {} });
        expect(Inputs.mouse.scroll).toBe(120);
    });

    test("InputResetSystem clears per-frame state but keeps held keys", () => {
        onWindow("keydown")({ code: "KeyA" });
        onCanvas("wheel")({ target: canvas, deltaY: 40, preventDefault() {} });
        expect(Inputs.isKeyPressed("KeyA")).toBe(true);
        expect(Inputs.mouse.scroll).toBe(40);

        state.step();
        expect(Inputs.isKeyPressed("KeyA")).toBe(false); // pulse cleared
        expect(Inputs.isKeyDown("KeyA")).toBe(true); // held key survives
        expect(Inputs.mouse.scroll).toBe(0);
        expect(Inputs.mouse.deltaX).toBe(0);
    });

    test("setup with no document canvas skips canvas attach", () => {
        clear();
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: () => [],
        } as unknown as typeof document;
        const s = new State();
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        attach(s, InputPlugin);
        const beforeAttach = canvas.tracker.added.length;
        s.step();
        expect(canvas.tracker.added.length).toBe(beforeAttach);
    });
});
