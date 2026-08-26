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
    added: [string, Fn, AddEventListenerOptions | undefined][] = [];

    addEventListener = (type: string, fn: Fn, opts?: AddEventListenerOptions) => {
        this.added.push([type, fn, opts]);
    };
    // input attaches with `{ signal }` and never removes by hand; kept as a no-op so a stray call can't throw
    removeEventListener = () => {};
}

// a fixed rect the mock canvas reports — the window-level pointerMove handler reads it to
// compute canvas-relative coordinates while a pointer is active, so the mock must provide one.
const MOCK_RECT: DOMRect = {
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
        getBoundingClientRect() {
            return MOCK_RECT;
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

    // teardown is signal-based: every listener is attached with `{ signal: state.signal }`, so
    // `state.dispose()` aborting that signal detaches them all with no removal code (the browser
    // contract). The mock can't fire abort, so assert the wiring — every listener carries the
    // State's signal, and dispose aborts it.
    test("every canvas listener is attached with the State's signal, aborted on dispose", () => {
        expect(canvas.tracker.added.length).toBeGreaterThan(0);
        for (const [, , opts] of canvas.tracker.added) {
            expect(opts?.signal).toBe(state.signal);
        }
        expect(state.signal.aborted).toBe(false);
        state.dispose();
        expect(state.signal.aborted).toBe(true);
    });

    test("every global listener is attached with the State's signal, aborted on dispose", () => {
        expect(windowTracker.added.length).toBeGreaterThan(0);
        for (const [, , opts] of windowTracker.added) {
            expect(opts?.signal).toBe(state.signal);
        }
        expect(state.signal.aborted).toBe(false);
        state.dispose();
        expect(state.signal.aborted).toBe(true);
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

    // RED-FIRST WITNESS (stage 9, roads-interactive.md): while a pointer is active, the window-level
    // pointermove handler must keep mouse.x/y tracking even when the pointer's target is not the canvas.
    // Against the shipped shape, mouse.x/y were written only by the canvas-scoped pointerHover handler
    // (which early-returns unless e.target is a registered canvas), so a pointermove dispatched at a
    // non-canvas target froze the coordinates while hover stayed true (pointerLeave only clears hover
    // when activePointerId === null). The failure text witnessed before the fix:
    //   "Expected: 300, Received: 100" (mouse.x stayed at the initial 100 when the window-level
    //   pointermove fired at a non-canvas target — the canvas-scoped pointerHover early-returned, so
    //   mouse.x/y were never written for the off-canvas move)
    // The fix: the window-level pointerMove handler writes mouse.x/y using the active canvas's rect.
    // multi-touch: a second finger no longer just gets ignored (the old
    // `activePointerId` early-return) — it's tracked in a per-pointerId cache
    // (MDN multi-touch pattern) that feeds `Inputs.touch`, independent of the
    // single-pointer `Mouse` capture path above.
    test("a second touch pointer registers in Inputs.touch.count without disturbing Mouse", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        expect(Inputs.touch.count).toBe(1);

        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        });
        expect(Inputs.touch.count).toBe(2);
        // the second finger must not masquerade as the captured pointer's button/delta
        expect(Inputs.mouse.deltaX).toBe(0);
    });

    test("pinch: two-finger spread produces a positive pinchDelta, matching the hand-computed distance change", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        }); // initial distance: 100

        onWindow("pointermove")({
            pointerId: 1,
            pointerType: "touch",
            buttons: 1,
            clientX: 80,
            clientY: 100,
            preventDefault() {},
        });
        onWindow("pointermove")({
            pointerId: 2,
            pointerType: "touch",
            buttons: 1,
            clientX: 220,
            clientY: 100,
            preventDefault() {},
        }); // new distance: 140, delta +40

        expect(Inputs.touch.pinchDelta).toBe(40);
    });

    test("two-finger drag produces a centroid delta", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        }); // initial centroid: (150, 100)

        onWindow("pointermove")({
            pointerId: 1,
            pointerType: "touch",
            buttons: 1,
            clientX: 110,
            clientY: 130,
            preventDefault() {},
        });
        onWindow("pointermove")({
            pointerId: 2,
            pointerType: "touch",
            buttons: 1,
            clientX: 210,
            clientY: 130,
            preventDefault() {},
        }); // new centroid: (160, 130), delta (+10, +30)

        expect(Inputs.touch.deltaX).toBe(10);
        expect(Inputs.touch.deltaY).toBe(30);
        // pinch stays flat — the two fingers moved together, distance unchanged
        expect(Inputs.touch.pinchDelta).toBe(0);
    });

    test("touch deltas reset each frame, count survives", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        });
        onWindow("pointermove")({
            pointerId: 1,
            pointerType: "touch",
            buttons: 1,
            clientX: 90,
            clientY: 100,
            preventDefault() {},
        });
        expect(Inputs.touch.pinchDelta).toBe(10);

        state.step();
        expect(Inputs.touch.pinchDelta).toBe(0);
        expect(Inputs.touch.deltaX).toBe(0);
        expect(Inputs.touch.count).toBe(2); // still two fingers down
    });

    test("lifting one finger drops the pinch baseline — a third pointerdown does not resume a stale delta", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        });
        onWindow("pointerup")({
            pointerId: 2,
            pointerType: "touch",
            buttons: 0,
            preventDefault() {},
        });
        expect(Inputs.touch.count).toBe(1);

        // a fresh second finger — no baseline exists yet, so its first move must not
        // report a delta computed against the old (now-stale) pair
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 3,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 300,
            clientY: 100,
            preventDefault() {},
        });
        expect(Inputs.touch.count).toBe(2);
        onWindow("pointermove")({
            pointerId: 3,
            pointerType: "touch",
            buttons: 1,
            clientX: 305,
            clientY: 100,
            preventDefault() {},
        });
        // distance was rebaselined at (1,3)'s pointerdown, so this move's delta is
        // just the +5 shift of pointer 3, not a jump from the stale (1,2) baseline
        expect(Inputs.touch.pinchDelta).toBe(5);
    });

    // RED-FIRST WITNESS: `pointerDown` never reassigns `activePointerId` to an already-down second
    // finger, and it was only ever cleared (never re-seeded) on the captured pointer's up/cancel. So
    // lifting the FIRST finger of a two-finger gesture left `activePointerId` null (or the departed
    // id) for the rest of the sequence — the survivor's moves hit `pointerMove`'s
    // `e.pointerId !== s.activePointerId` early return and `Mouse.deltaX/Y`/`x`/`y` froze until the
    // whole gesture ended, silently, since `Inputs.touch.count` (1) still read as a live single-finger
    // drag. The fix re-captures to a remaining touch pointer on up/cancel.
    test("lifting the capturing finger re-captures to the surviving finger — Mouse tracking doesn't freeze", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        }); // pointer 1 captures (first down)
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        }); // pointer 2 joins the touch cache but never captures

        onWindow("pointerup")({
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 0,
            preventDefault() {},
        }); // the capturing finger lifts — pointer 2 is still down
        expect(Inputs.touch.count).toBe(1);

        onWindow("pointermove")({
            pointerId: 2,
            pointerType: "touch",
            buttons: 1,
            clientX: 230,
            clientY: 100,
            preventDefault() {},
        }); // survivor moves +30 from its own (200, 100) — not from pointer 1's old (100, 100)

        // without the fix this reads 0 (frozen): pointer 2 was never the captured pointer, so its
        // move hit the early return
        expect(Inputs.mouse.deltaX).toBe(30);
        // seeded from pointer 2's own cached position, not the departed pointer 1's — so the first
        // post-transition move reports its own +30 shift, never the 100px gap between the two fingers
        expect(Inputs.mouse.deltaX).not.toBe(130);
        expect(Inputs.mouse.x).toBe(230);
    });

    test("setInputEnabled(false) neutralizes touch reads and clears the cache", () => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 100,
            preventDefault() {},
        });
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 2,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX: 200,
            clientY: 100,
            preventDefault() {},
        });
        expect(Inputs.touch.count).toBe(2);

        setInputEnabled(false);
        expect(Inputs.touch.count).toBe(0);
        expect(Inputs.touch.pinchDelta).toBe(0);

        setInputEnabled(true);
        expect(Inputs.touch.count).toBe(0); // cache was cleared, not just gated
    });

    test("a held-pointer move at a non-canvas target keeps mouse.x/y tracking", () => {
        // pointer enters the canvas — sets hover true
        onCanvas("pointerenter")({ target: canvas });
        expect(Inputs.mouse.hover).toBe(true);
        // pointer down on the canvas — establishes the active pointer and activeCanvas
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 200,
            preventDefault() {},
        });
        // initial position from the canvas-scoped pointerHover
        onCanvas("pointermove")({
            target: canvas,
            pointerId: 1,
            buttons: 1,
            clientX: 100,
            clientY: 200,
            preventDefault() {},
        });
        expect(Inputs.mouse.x).toBe(100);
        expect(Inputs.mouse.y).toBe(200);

        // now the pointer moves off the canvas — the window-level pointermove fires with a
        // non-canvas target. The canvas-scoped pointerHover does NOT fire (e.target is not the
        // canvas), so without the fix, mouse.x/y would stay at (100, 200).
        onWindow("pointermove")({
            pointerId: 1,
            buttons: 1,
            clientX: 300,
            clientY: 400,
            preventDefault() {},
        });
        // mouse.x/y must track the new position (300 - rect.left=0, 400 - rect.top=0)
        expect(Inputs.mouse.x).toBe(300);
        expect(Inputs.mouse.y).toBe(400);
        // hover stays true — the drag survives leaving the canvas
        expect(Inputs.mouse.hover).toBe(true);
    });
});

// `Inputs.focused` is document-order index of the bound canvas (the Nth `<canvas>` InputSystem found),
// never an ECS entity id — InputSystem's sole caller has no real view/eid substrate to draw one from
// (`querySelectorAll` returns bare elements). Two canvases pin that it's positional, not identity-derived.
describe("InputPlugin focus with multiple canvases", () => {
    test("a second canvas focuses at its document-order index, not an eid", () => {
        clear();
        const windowTracker = new ListenerTracker();
        (windowTracker as unknown as { focus: () => void }).focus = () => {};
        const savedWindow = globalThis.window;
        const savedDocument = globalThis.document;
        globalThis.window = windowTracker as unknown as typeof window;

        const first = mockCanvas();
        const second = mockCanvas();
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: (sel: string) => (sel === "canvas" ? [first, second] : []),
        } as unknown as typeof document;

        const state = new State();
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        attach(state, InputPlugin);
        state.step();

        expect(Inputs.focused).toBe(0); // defaults to the first canvas bound

        const onSecond = (type: string): Fn => second.tracker.added.find(([t]) => t === type)![1];
        onSecond("pointerdown")({
            target: second,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        expect(Inputs.focused).toBe(1); // the second canvas is index 1 in document order

        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });
});
