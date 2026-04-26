import { resource, type State, type System, type Plugin } from "../../engine";
import { Views } from "../viewport";

export interface Mouse {
    deltaX: number;
    deltaY: number;
    scroll: number;
    left: boolean;
    right: boolean;
    middle: boolean;
    hover: boolean;
    x: number;
    y: number;
    canvasWidth: number;
    canvasHeight: number;
}

export interface Inputs {
    readonly mouse: Readonly<Mouse>;
    readonly focused: number;
    isKeyDown(code: string): boolean;
    isKeyPressed(code: string): boolean;
    isKeyReleased(code: string): boolean;
    isKeyPressedWithin(code: string, seconds: number): boolean;
}

export const Inputs = resource<Inputs>("inputs");

interface InputState {
    keys: Set<string>;
    keysPressed: Set<string>;
    keysReleased: Set<string>;
    keyPressedAt: Map<string, number>;
    mouse: Mouse;
    canvases: Map<HTMLCanvasElement, number>;
    activeCanvas: HTMLCanvasElement | null;
    focused: number;
    lastPointerX: number;
    lastPointerY: number;
    activePointerId: number | null;
    activeButton: number | null;
    pointerHover: (e: PointerEvent) => void;
    pointerEnter: (e: PointerEvent) => void;
    pointerLeave: (e: PointerEvent) => void;
    keyDown: (e: KeyboardEvent) => void;
    keyUp: (e: KeyboardEvent) => void;
    pointerDown: (e: PointerEvent) => void;
    pointerUp: (e: PointerEvent) => void;
    pointerCancel: (e: PointerEvent) => void;
    pointerMove: (e: PointerEvent) => void;
    wheel: (e: WheelEvent) => void;
    contextMenu: (e: Event) => void;
    canvasFocused: boolean;
    windowPointerDown: (e: PointerEvent) => void;
    windowBlur: () => void;
}

const InputResource = resource<InputState>("input");

// `PointerEvent.buttons` is the bitmask of currently-pressed buttons
// (left=1, right=2, middle=4). browsers update it on every pointer event
// regardless of pointer capture, so reading it directly is more robust than
// inferring state from `pointerDown`/`pointerUp`'s `button` field — some
// platforms suppress per-button events for non-primary buttons during
// capture, but `buttons` always reflects reality.
function syncButtons(mouse: Mouse, buttons: number): void {
    mouse.left = (buttons & 1) !== 0;
    mouse.right = (buttons & 2) !== 0;
    mouse.middle = (buttons & 4) !== 0;
}

function clearAllButtons(mouse: Mouse): void {
    mouse.left = false;
    mouse.middle = false;
    mouse.right = false;
}

// pointer capture tracks the FIRST button on a pointer for movement / focus;
// button state is derived from `e.buttons` and updates independently of
// capture, so a right-click during a left-drag is observable.
function releaseCapture(s: InputState): void {
    s.activePointerId = null;
    s.activeButton = null;
    s.activeCanvas = null;
    s.lastPointerX = 0;
    s.lastPointerY = 0;
}

function createHandlers(s: InputState): void {
    s.pointerHover = (e: PointerEvent) => {
        const target = e.target as HTMLCanvasElement;
        if (!s.canvases.has(target)) return;
        s.mouse.hover = true;
        const rect = target.getBoundingClientRect();
        s.mouse.x = e.clientX - rect.left;
        s.mouse.y = e.clientY - rect.top;
        s.mouse.canvasWidth = rect.width;
        s.mouse.canvasHeight = rect.height;
    };

    s.pointerEnter = () => {
        s.mouse.hover = true;
    };

    s.pointerLeave = () => {
        if (s.activePointerId === null) s.mouse.hover = false;
    };

    s.keyDown = (e: KeyboardEvent) => {
        const locked = document.pointerLockElement as HTMLCanvasElement | null;
        if (!s.canvasFocused && !(locked && s.canvases.has(locked))) return;
        if (!s.keys.has(e.code)) {
            s.keysPressed.add(e.code);
            s.keyPressedAt.set(e.code, performance.now());
        }
        s.keys.add(e.code);
    };

    s.keyUp = (e: KeyboardEvent) => {
        s.keys.delete(e.code);
        s.keysReleased.add(e.code);
    };

    s.pointerDown = (e: PointerEvent) => {
        const target = e.target as HTMLCanvasElement;
        const canvasEid = s.canvases.get(target);
        if (canvasEid === undefined) return;
        window.focus();
        // ignore multi-touch (different pointerId on the same canvas) so a
        // second pointer's buttons can't masquerade as the captured one's.
        if (s.activePointerId === null || s.activePointerId === e.pointerId) {
            syncButtons(s.mouse, e.buttons);
        }
        if (s.activePointerId === null) {
            s.activePointerId = e.pointerId;
            s.activeButton = e.button;
            s.activeCanvas = target;
            s.focused = canvasEid;
            s.canvasFocused = true;
            s.lastPointerX = e.clientX;
            s.lastPointerY = e.clientY;
            try {
                target.setPointerCapture(e.pointerId);
            } catch {}
        }
        e.preventDefault();
    };

    s.windowPointerDown = (e: PointerEvent) => {
        if (!s.canvases.has(e.target as HTMLCanvasElement)) {
            s.canvasFocused = false;
            s.keys.clear();
        }
    };

    s.windowBlur = () => {
        s.canvasFocused = false;
        s.keys.clear();
        s.keysPressed.clear();
    };

    s.pointerUp = (e: PointerEvent) => {
        if (e.pointerId !== s.activePointerId) return;
        syncButtons(s.mouse, e.buttons);
        // releasing the capturing button releases the pointer; secondary
        // buttons just toggle their state and leave the drag intact.
        if (e.button === s.activeButton) {
            s.activeCanvas?.releasePointerCapture(e.pointerId);
            releaseCapture(s);
        }
    };

    s.pointerCancel = (e: PointerEvent) => {
        if (e.pointerId !== s.activePointerId) return;
        clearAllButtons(s.mouse);
        releaseCapture(s);
    };

    s.pointerMove = (e: PointerEvent) => {
        if (e.pointerId !== s.activePointerId) return;
        // sync buttons during drag too — e.g. catches a right-press that
        // happened mid-drag if the per-button pointerdown was suppressed
        // and the user moved before releasing.
        syncButtons(s.mouse, e.buttons);
        e.preventDefault();
        s.mouse.deltaX += e.clientX - s.lastPointerX;
        s.mouse.deltaY += e.clientY - s.lastPointerY;
        s.lastPointerX = e.clientX;
        s.lastPointerY = e.clientY;
    };

    s.wheel = (e: WheelEvent) => {
        const target = e.target as HTMLCanvasElement;
        if (!s.canvases.has(target)) return;
        s.mouse.scroll += e.deltaY;
        e.preventDefault();
    };

    s.contextMenu = (e: Event) => {
        if (s.canvases.has(e.target as HTMLCanvasElement)) {
            e.preventDefault();
        }
    };
}

function attachCanvas(s: InputState, canvas: HTMLCanvasElement): void {
    canvas.addEventListener("pointerdown", s.pointerDown);
    canvas.addEventListener("pointermove", s.pointerHover);
    canvas.addEventListener("pointerenter", s.pointerEnter);
    canvas.addEventListener("pointerleave", s.pointerLeave);
    canvas.addEventListener("wheel", s.wheel, { passive: false });
    canvas.addEventListener("contextmenu", s.contextMenu);
}

function detachCanvas(s: InputState, canvas: HTMLCanvasElement): void {
    canvas.removeEventListener("pointerdown", s.pointerDown);
    canvas.removeEventListener("pointermove", s.pointerHover);
    canvas.removeEventListener("pointerenter", s.pointerEnter);
    canvas.removeEventListener("pointerleave", s.pointerLeave);
    canvas.removeEventListener("wheel", s.wheel);
    canvas.removeEventListener("contextmenu", s.contextMenu);
}

function attachGlobal(s: InputState): void {
    window.addEventListener("keydown", s.keyDown);
    window.addEventListener("keyup", s.keyUp);
    window.addEventListener("pointerdown", s.windowPointerDown);
    window.addEventListener("pointerup", s.pointerUp);
    window.addEventListener("pointercancel", s.pointerCancel);
    window.addEventListener("pointermove", s.pointerMove);
    window.addEventListener("blur", s.windowBlur);
}

function detachGlobal(s: InputState): void {
    window.removeEventListener("keydown", s.keyDown);
    window.removeEventListener("keyup", s.keyUp);
    window.removeEventListener("pointerdown", s.windowPointerDown);
    window.removeEventListener("pointerup", s.pointerUp);
    window.removeEventListener("pointercancel", s.pointerCancel);
    window.removeEventListener("pointermove", s.pointerMove);
    window.removeEventListener("blur", s.windowBlur);
}

function setup(state: State, views: Map<number, any>): void {
    const mouse: Mouse = {
        deltaX: 0,
        deltaY: 0,
        scroll: 0,
        left: false,
        right: false,
        middle: false,
        hover: false,
        x: 0,
        y: 0,
        canvasWidth: 0,
        canvasHeight: 0,
    };

    const canvases = new Map<HTMLCanvasElement, number>();
    let firstCanvasEid = -1;

    for (const [eid, view] of views) {
        if (!view.element) continue;
        canvases.set(view.element, eid);
        view.element.style.touchAction = "none";
        if (firstCanvasEid < 0) firstCanvasEid = eid;
    }

    if (canvases.size === 0) return;

    const inputState: InputState = {
        keys: new Set(),
        keysPressed: new Set(),
        keysReleased: new Set(),
        keyPressedAt: new Map(),
        mouse,
        canvases,
        activeCanvas: null,
        focused: firstCanvasEid,
        lastPointerX: 0,
        lastPointerY: 0,
        activePointerId: null,
        activeButton: null,
        pointerHover: null!,
        pointerEnter: null!,
        pointerLeave: null!,
        keyDown: null!,
        keyUp: null!,
        pointerDown: null!,
        pointerUp: null!,
        pointerCancel: null!,
        pointerMove: null!,
        wheel: null!,
        contextMenu: null!,
        canvasFocused: true,
        windowPointerDown: null!,
        windowBlur: null!,
    };

    createHandlers(inputState);
    attachGlobal(inputState);
    for (const canvas of canvases.keys()) {
        attachCanvas(inputState, canvas);
    }

    state.setResource(InputResource, inputState);
    state.setResource(Inputs, {
        mouse,
        get focused() {
            return inputState.focused;
        },
        isKeyDown: (code: string) => inputState.keys.has(code),
        isKeyPressed: (code: string) => inputState.keysPressed.has(code),
        isKeyReleased: (code: string) => inputState.keysReleased.has(code),
        isKeyPressedWithin: (code: string, seconds: number) =>
            performance.now() - (inputState.keyPressedAt.get(code) ?? -Infinity) < seconds * 1000,
    });
}

const InputSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    setup(state: State) {
        const views = Views.from(state);
        if (!views || views.size === 0) return;
        setup(state, views);
    },

    dispose(state: State) {
        const s = InputResource.from(state);
        if (s) {
            detachGlobal(s);
            for (const canvas of s.canvases.keys()) {
                detachCanvas(s, canvas);
            }
            state.deleteResource(InputResource);
        }
        state.deleteResource(Inputs);
    },
};

const InputResetSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,

    update(state: State) {
        const s = InputResource.from(state);
        if (!s) return;
        s.keysPressed.clear();
        s.keysReleased.clear();
        s.mouse.deltaX = 0;
        s.mouse.deltaY = 0;
        s.mouse.scroll = 0;
    },
};

export const InputPlugin: Plugin = {
    name: "Input",
    systems: [InputSystem, InputResetSystem],
};
