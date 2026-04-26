import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { State, InputPlugin, Inputs } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Views, type View } from "../src/standard/viewport";

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
        style: {} as CSSStyleDeclaration,
        tracker: tracker,
    } as unknown as HTMLCanvasElement & { tracker: ListenerTracker };
}

class MockInputState {
    private _keys = new Set<string>();
    private _keysPressed = new Set<string>();
    private _keysReleased = new Set<string>();

    private _mouse = {
        deltaX: 0,
        deltaY: 0,
        scroll: 0,
        left: false,
        right: false,
        middle: false,
    };

    private _activePointerId: number | null = null;
    private _activeButton: number | null = null;

    handleKeyDown(code: string): void {
        if (!this._keys.has(code)) {
            this._keysPressed.add(code);
        }
        this._keys.add(code);
    }

    handleKeyUp(code: string): void {
        this._keys.delete(code);
        this._keysReleased.add(code);
    }

    handlePointerDown(pointerId: number, button: number, _x: number, _y: number): boolean {
        // pointer capture only gates movement/focus; button state updates
        // for the active pointer (or none active) so a right-click during a
        // left-drag is observable. multi-touch (different pointerId on the
        // same canvas) is ignored entirely.
        if (this._activePointerId === null || this._activePointerId === pointerId) {
            this.setButtonState(button, true);
        }
        if (this._activePointerId === null) {
            this._activePointerId = pointerId;
            this._activeButton = button;
            return true;
        }
        return false;
    }

    handlePointerUp(pointerId: number, button?: number): void {
        if (pointerId !== this._activePointerId) return;
        const b = button ?? this._activeButton ?? 0;
        this.setButtonState(b, false);
        if (b === this._activeButton) {
            this._activePointerId = null;
            this._activeButton = null;
        }
    }

    handlePointerMove(pointerId: number, deltaX: number, deltaY: number): void {
        if (pointerId === this._activePointerId) {
            this._mouse.deltaX += deltaX;
            this._mouse.deltaY += deltaY;
        }
    }

    handleWheel(deltaY: number): void {
        this._mouse.scroll += deltaY;
    }

    resetFrame(): void {
        this._keysPressed.clear();
        this._keysReleased.clear();
        this._mouse.deltaX = 0;
        this._mouse.deltaY = 0;
        this._mouse.scroll = 0;
    }

    clearAll(): void {
        this._keys.clear();
        this._keysPressed.clear();
        this._keysReleased.clear();
        this._mouse.deltaX = 0;
        this._mouse.deltaY = 0;
        this._mouse.scroll = 0;
        this._mouse.left = false;
        this._mouse.right = false;
        this._mouse.middle = false;
        this._activePointerId = null;
        this._activeButton = null;
    }

    private setButtonState(button: number, pressed: boolean): void {
        if (button === 0) this._mouse.left = pressed;
        if (button === 1) this._mouse.middle = pressed;
        if (button === 2) this._mouse.right = pressed;
    }

    isKeyDown(code: string): boolean {
        return this._keys.has(code);
    }

    isKeyPressed(code: string): boolean {
        return this._keysPressed.has(code);
    }

    isKeyReleased(code: string): boolean {
        return this._keysReleased.has(code);
    }

    getMouse() {
        return this._mouse;
    }

    getActivePointerId(): number | null {
        return this._activePointerId;
    }
}

describe("input", () => {
    describe("keyboard state", () => {
        test("isKeyPressed true only on first frame", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            expect(input.isKeyPressed("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyPressed("KeyA")).toBe(false);
        });

        test("isKeyDown true while held", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            expect(input.isKeyDown("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyDown("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyDown("KeyA")).toBe(true);
        });

        test("isKeyReleased true only on release frame", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.resetFrame();

            input.handleKeyUp("KeyA");
            expect(input.isKeyReleased("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyReleased("KeyA")).toBe(false);
        });

        test("isKeyDown false after release", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            expect(input.isKeyDown("KeyA")).toBe(true);

            input.handleKeyUp("KeyA");
            expect(input.isKeyDown("KeyA")).toBe(false);
        });

        test("multiple keys can be down simultaneously", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.handleKeyDown("KeyB");
            input.handleKeyDown("KeyC");

            expect(input.isKeyDown("KeyA")).toBe(true);
            expect(input.isKeyDown("KeyB")).toBe(true);
            expect(input.isKeyDown("KeyC")).toBe(true);
        });

        test("repeated keydown does not set pressed again", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.resetFrame();
            input.handleKeyDown("KeyA");

            expect(input.isKeyPressed("KeyA")).toBe(false);
            expect(input.isKeyDown("KeyA")).toBe(true);
        });

        test("key pressed and released same frame", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.handleKeyUp("KeyA");

            expect(input.isKeyPressed("KeyA")).toBe(true);
            expect(input.isKeyReleased("KeyA")).toBe(true);
            expect(input.isKeyDown("KeyA")).toBe(false);
        });
    });

    describe("mouse state", () => {
        test("accumulates deltaX/deltaY", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerMove(1, 10, 5);
            input.handlePointerMove(1, 20, 15);

            expect(input.getMouse().deltaX).toBe(30);
            expect(input.getMouse().deltaY).toBe(20);
        });

        test("clears deltas on frame reset", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerMove(1, 10, 5);
            expect(input.getMouse().deltaX).toBe(10);

            input.resetFrame();
            expect(input.getMouse().deltaX).toBe(0);
            expect(input.getMouse().deltaY).toBe(0);
        });

        test("tracks left button state", () => {
            const input = new MockInputState();
            expect(input.getMouse().left).toBe(false);

            input.handlePointerDown(1, 0, 100, 100);
            expect(input.getMouse().left).toBe(true);

            input.handlePointerUp(1);
            expect(input.getMouse().left).toBe(false);
        });

        test("tracks right button state", () => {
            const input = new MockInputState();
            expect(input.getMouse().right).toBe(false);

            input.handlePointerDown(1, 2, 100, 100);
            expect(input.getMouse().right).toBe(true);

            input.handlePointerUp(1);
            expect(input.getMouse().right).toBe(false);
        });

        test("tracks right press during a held left-drag", () => {
            // pointer capture binds to the first button on the pointer for
            // movement tracking, but button state is independent so consumers
            // can observe a right-click during a left-drag (Esc-equivalent
            // cancel gesture in tools, etc.).
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            expect(input.getMouse().left).toBe(true);

            input.handlePointerDown(1, 2, 100, 100);
            expect(input.getMouse().left).toBe(true);
            expect(input.getMouse().right).toBe(true);

            input.handlePointerUp(1, 2);
            expect(input.getMouse().left).toBe(true);
            expect(input.getMouse().right).toBe(false);
            expect(input.getActivePointerId()).toBe(1);

            input.handlePointerUp(1, 0);
            expect(input.getMouse().left).toBe(false);
            expect(input.getActivePointerId()).toBeNull();
        });

        test("tracks middle button state", () => {
            const input = new MockInputState();
            expect(input.getMouse().middle).toBe(false);

            input.handlePointerDown(1, 1, 100, 100);
            expect(input.getMouse().middle).toBe(true);

            input.handlePointerUp(1);
            expect(input.getMouse().middle).toBe(false);
        });
    });

    describe("scroll wheel", () => {
        test("accumulates scroll delta", () => {
            const input = new MockInputState();
            input.handleWheel(100);
            input.handleWheel(50);

            expect(input.getMouse().scroll).toBe(150);
        });

        test("handles negative scroll (up)", () => {
            const input = new MockInputState();
            input.handleWheel(-100);

            expect(input.getMouse().scroll).toBe(-100);
        });

        test("clears scroll on frame reset", () => {
            const input = new MockInputState();
            input.handleWheel(100);
            expect(input.getMouse().scroll).toBe(100);

            input.resetFrame();
            expect(input.getMouse().scroll).toBe(0);
        });
    });

    describe("pointer capture", () => {
        test("only one pointer active at a time", () => {
            const input = new MockInputState();
            const first = input.handlePointerDown(1, 0, 100, 100);
            const second = input.handlePointerDown(2, 0, 200, 200);

            expect(first).toBe(true);
            expect(second).toBe(false);
            expect(input.getActivePointerId()).toBe(1);
        });

        test("second pointer ignored while first active", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerMove(1, 10, 10);
            input.handlePointerMove(2, 100, 100);

            expect(input.getMouse().deltaX).toBe(10);
        });

        test("new pointer can capture after first released", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerUp(1);

            const second = input.handlePointerDown(2, 0, 200, 200);
            expect(second).toBe(true);
            expect(input.getActivePointerId()).toBe(2);
        });

        test("releasing wrong pointer does not clear capture", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerUp(2);

            expect(input.getActivePointerId()).toBe(1);
            expect(input.getMouse().left).toBe(true);
        });
    });

    describe("frame reset", () => {
        test("clears pressed keys", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            expect(input.isKeyPressed("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyPressed("KeyA")).toBe(false);
        });

        test("clears released keys", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.handleKeyUp("KeyA");
            expect(input.isKeyReleased("KeyA")).toBe(true);

            input.resetFrame();
            expect(input.isKeyReleased("KeyA")).toBe(false);
        });

        test("clears mouse deltas", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerMove(1, 50, 50);
            input.handleWheel(100);

            input.resetFrame();
            expect(input.getMouse().deltaX).toBe(0);
            expect(input.getMouse().deltaY).toBe(0);
            expect(input.getMouse().scroll).toBe(0);
        });

        test("preserves held keys", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.resetFrame();

            expect(input.isKeyDown("KeyA")).toBe(true);
        });

        test("preserves button state", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.resetFrame();

            expect(input.getMouse().left).toBe(true);
        });
    });

    describe("clear all state", () => {
        test("clears all keys", () => {
            const input = new MockInputState();
            input.handleKeyDown("KeyA");
            input.handleKeyDown("KeyB");
            input.clearAll();

            expect(input.isKeyDown("KeyA")).toBe(false);
            expect(input.isKeyDown("KeyB")).toBe(false);
        });

        test("clears all button states", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.clearAll();

            expect(input.getMouse().left).toBe(false);
            expect(input.getMouse().right).toBe(false);
            expect(input.getMouse().middle).toBe(false);
        });

        test("clears pointer capture", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.clearAll();

            expect(input.getActivePointerId()).toBeNull();
        });

        test("clears deltas", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 100, 100);
            input.handlePointerMove(1, 50, 50);
            input.handleWheel(100);
            input.clearAll();

            expect(input.getMouse().deltaX).toBe(0);
            expect(input.getMouse().deltaY).toBe(0);
            expect(input.getMouse().scroll).toBe(0);
        });
    });

    describe("key codes", () => {
        test("handles letter keys", () => {
            const input = new MockInputState();
            for (const key of ["KeyA", "KeyB", "KeyZ"]) {
                input.handleKeyDown(key);
                expect(input.isKeyDown(key)).toBe(true);
            }
        });

        test("handles number keys", () => {
            const input = new MockInputState();
            for (const key of ["Digit0", "Digit1", "Digit9"]) {
                input.handleKeyDown(key);
                expect(input.isKeyDown(key)).toBe(true);
            }
        });

        test("handles modifier keys", () => {
            const input = new MockInputState();
            for (const key of ["ShiftLeft", "ControlLeft", "AltLeft", "MetaLeft"]) {
                input.handleKeyDown(key);
                expect(input.isKeyDown(key)).toBe(true);
            }
        });

        test("handles arrow keys", () => {
            const input = new MockInputState();
            for (const key of ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]) {
                input.handleKeyDown(key);
                expect(input.isKeyDown(key)).toBe(true);
            }
        });

        test("handles special keys", () => {
            const input = new MockInputState();
            for (const key of ["Space", "Enter", "Escape", "Tab", "Backspace"]) {
                input.handleKeyDown(key);
                expect(input.isKeyDown(key)).toBe(true);
            }
        });
    });

    describe("mouse button numbers", () => {
        test("button 0 is left", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 0, 0, 0);
            expect(input.getMouse().left).toBe(true);
            expect(input.getMouse().middle).toBe(false);
            expect(input.getMouse().right).toBe(false);
        });

        test("button 1 is middle", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 1, 0, 0);
            expect(input.getMouse().left).toBe(false);
            expect(input.getMouse().middle).toBe(true);
            expect(input.getMouse().right).toBe(false);
        });

        test("button 2 is right", () => {
            const input = new MockInputState();
            input.handlePointerDown(1, 2, 0, 0);
            expect(input.getMouse().left).toBe(false);
            expect(input.getMouse().middle).toBe(false);
            expect(input.getMouse().right).toBe(true);
        });
    });

    describe("system groups", () => {
        test("InputSystem is in simulation group with mode always", () => {
            const systems = InputPlugin.systems!;
            const input = systems[0];
            expect(input.group).toBe("simulation");
            expect(input.annotations?.mode).toBe("always");
        });

        test("InputResetSystem is in draw group with last flag", () => {
            const systems = InputPlugin.systems!;
            const reset = systems[1];
            expect(reset.group).toBe("draw");
            expect(reset.annotations?.mode).toBe("always");
            expect(reset.last).toBe(true);
        });
    });
});

describe("InputPlugin integration", () => {
    let state: State;
    let canvas: ReturnType<typeof mockCanvas>;
    let windowTracker: ListenerTracker;
    let savedWindow: typeof globalThis.window;
    let savedDocument: typeof globalThis.document;

    beforeEach(() => {
        clearRegistry();
        windowTracker = new ListenerTracker();
        (windowTracker as unknown as { focus: () => void }).focus = () => {};
        savedWindow = globalThis.window;
        savedDocument = globalThis.document;
        globalThis.window = windowTracker as unknown as typeof window;
        globalThis.document = { pointerLockElement: null } as unknown as typeof document;

        canvas = mockCanvas();
        state = new State();
        state.register(InputPlugin);

        const views = new Map<number, View>();
        views.set(1, { element: canvas } as unknown as View);
        state.setResource(Views, views);

        state.step();
    });

    afterEach(() => {
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    test("setup populates Inputs resource", () => {
        const inputs = state.getResource(Inputs);
        expect(inputs).toBeDefined();
        expect(inputs!.mouse).toBeDefined();
        expect(inputs!.focused).toBe(1);
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

    test("dispose deletes Inputs resource", () => {
        state.dispose();
        expect(state.getResource(Inputs)).toBeUndefined();
    });

    test("keyboard events flow through to Inputs resource", () => {
        const inputs = state.getResource(Inputs)!;
        const keyDown = windowTracker.added.find(([t]) => t === "keydown")![1];
        keyDown({ code: "KeyW" });

        expect(inputs.isKeyDown("KeyW")).toBe(true);
        expect(inputs.isKeyPressed("KeyW")).toBe(true);
    });

    test("pointer events flow through to Inputs resource", () => {
        const inputs = state.getResource(Inputs)!;
        const pointerDown = canvas.tracker.added.find(([t]) => t === "pointerdown")![1];
        pointerDown({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 100,
            clientY: 200,
            preventDefault() {},
        });

        expect(inputs.mouse.left).toBe(true);
    });

    test("wheel events flow through to Inputs resource", () => {
        const inputs = state.getResource(Inputs)!;
        const wheel = canvas.tracker.added.find(([t]) => t === "wheel")![1];
        wheel({ target: canvas, deltaY: 120, preventDefault() {} });

        expect(inputs.mouse.scroll).toBe(120);
    });

    test("InputResetSystem clears per-frame state on step", () => {
        const inputs = state.getResource(Inputs)!;
        const keyDown = windowTracker.added.find(([t]) => t === "keydown")![1];
        keyDown({ code: "KeyA" });

        expect(inputs.isKeyPressed("KeyA")).toBe(true);

        state.step();
        expect(inputs.isKeyPressed("KeyA")).toBe(false);
        expect(inputs.isKeyDown("KeyA")).toBe(true);
        expect(inputs.mouse.deltaX).toBe(0);
    });

    test("setup with empty Views does not set Inputs resource", () => {
        clearRegistry();
        const s = new State();
        s.register(InputPlugin);
        s.setResource(Views, new Map());
        s.step();

        expect(s.getResource(Inputs)).toBeUndefined();
    });
});
