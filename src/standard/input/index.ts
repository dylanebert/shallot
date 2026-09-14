import type { Plugin, State, System } from "../../engine";

/** Keyboard facts owned by one {@link State}. `pressed`/`released` are the frame latches;
 * `tickPressed`/`tickReleased` are the independent fixed-clock latches. */
export interface Keys {
    /** keys currently held */
    readonly held: Set<string>;
    /** keys pressed since the last draw boundary */
    readonly pressed: Set<string>;
    /** keys released since the last draw boundary */
    readonly released: Set<string>;
    /** keys pressed since the last fixed-tick boundary */
    readonly tickPressed: Set<string>;
    /** keys released since the last fixed-tick boundary */
    readonly tickReleased: Set<string>;
    /** fixed tick at which each key was produced */
    readonly pressedTick: Map<string, number>;
}

/** live mouse state in one device record. Positions and sizes are CSS pixels; deltas accumulate over a
 * frame and reset at the draw boundary. */
export interface Mouse {
    /** horizontal pointer movement since the last frame, in CSS pixels */
    deltaX: number;
    /** vertical pointer movement since the last frame, in CSS pixels */
    deltaY: number;
    /** wheel movement accumulated this frame; positive scrolls down/away */
    scroll: number;
    /** left button held */
    left: boolean;
    /** right button held */
    right: boolean;
    /** middle button held */
    middle: boolean;
    /** pointer is over a bound canvas, or a drag is active */
    hover: boolean;
    /** pointer x within the focused canvas, CSS pixels from the left edge */
    x: number;
    /** pointer y within the focused canvas, CSS pixels from the top edge */
    y: number;
    /** focused canvas width in CSS pixels */
    canvasWidth: number;
    /** focused canvas height in CSS pixels */
    canvasHeight: number;
}

/** live multi-touch state in one device record. */
export interface Touch {
    /** number of touch pointers currently held on a bound canvas */
    count: number;
    /** two-finger distance change since the last frame */
    pinchDelta: number;
    /** two-finger centroid horizontal movement since the last frame */
    deltaX: number;
    /** two-finger centroid vertical movement since the last frame */
    deltaY: number;
}

/** all device-fed facts for one State. Producers below are the single mutation seam used by both the DOM
 * path and headless callers. The record is created lazily, so a State with no DOM still has devices. */
export interface Devices {
    readonly keys: Keys;
    readonly mouse: Mouse;
    readonly touch: Touch;
    /** document-order index of the canvas holding input focus, or -1 when none is focused */
    focused: number;
}

interface DeviceRecord extends Devices {
    enabled: boolean;
    requireLock: boolean;
    readonly touchPoints: Map<number, { x: number; y: number }>;
    pinchDistance: number | null;
    centroidX: number | null;
    centroidY: number | null;
    readonly canvases: Map<HTMLCanvasElement, number>;
    activeCanvas: HTMLCanvasElement | null;
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

const records = new WeakMap<State, DeviceRecord>();

const DEFAULT_MOUSE: Mouse = {
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

const DEFAULT_TOUCH: Touch = { count: 0, pinchDelta: 0, deltaX: 0, deltaY: 0 };
function emptyRecord(): DeviceRecord {
    return {
        keys: {
            held: new Set(),
            pressed: new Set(),
            released: new Set(),
            tickPressed: new Set(),
            tickReleased: new Set(),
            pressedTick: new Map(),
        },
        mouse: { ...DEFAULT_MOUSE },
        touch: { ...DEFAULT_TOUCH },
        focused: -1,
        enabled: true,
        requireLock: false,
        touchPoints: new Map(),
        pinchDistance: null,
        centroidX: null,
        centroidY: null,
        canvases: new Map(),
        activeCanvas: null,
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
}

/** Return the device record belonging to `state`. No DOM or plugin setup is required. */
export function devices(state: State): Devices {
    let record = records.get(state);
    if (!record) {
        record = emptyRecord();
        records.set(state, record);
    }
    return record;
}

function record(state: State): DeviceRecord {
    return devices(state) as DeviceRecord;
}

/** Produce a keyboard press. Repeated presses do not retrigger an edge. */
export function pressKey(state: State, code: string): void {
    const d = record(state);
    if (!d.enabled || d.keys.held.has(code)) return;
    d.keys.held.add(code);
    d.keys.pressed.add(code);
    d.keys.tickPressed.add(code);
    d.keys.pressedTick.set(code, state.time.fixedTick);
}

/** Produce a keyboard release. */
export function releaseKey(state: State, code: string): void {
    const d = record(state);
    if (!d.enabled || !d.keys.held.has(code)) return;
    d.keys.held.delete(code);
    d.keys.released.add(code);
    d.keys.tickReleased.add(code);
}

/** Produce a pointer position update. The object form is used by the DOM producer; the numeric form is
 * convenient for a replay or a headless test. */
export function pointerMove(
    state: State,
    x:
        | number
        | {
              x: number;
              y: number;
              deltaX?: number;
              deltaY?: number;
              hover?: boolean;
              canvasWidth?: number;
              canvasHeight?: number;
          },
    y?: number,
    deltaX = 0,
    deltaY = 0,
): void {
    const d = record(state);
    if (!d.enabled) return;
    const move = typeof x === "number" ? { x, y: y ?? 0, deltaX, deltaY } : x;
    d.mouse.x = move.x;
    d.mouse.y = move.y;
    d.mouse.deltaX += move.deltaX ?? 0;
    d.mouse.deltaY += move.deltaY ?? 0;
    if (move.hover !== undefined) d.mouse.hover = move.hover;
    if (move.canvasWidth !== undefined) d.mouse.canvasWidth = move.canvasWidth;
    if (move.canvasHeight !== undefined) d.mouse.canvasHeight = move.canvasHeight;
}

export type PointerButton = "left" | "right" | "middle" | 0 | 1 | 2;

/** Produce one pointer-button state. Numeric buttons use DOM `button` values (0 left, 1 middle, 2 right). */
export function pointerButton(state: State, button: PointerButton, pressed: boolean): void {
    const d = record(state);
    if (!d.enabled) return;
    const name =
        button === 0 || button === "left"
            ? "left"
            : button === 1 || button === "middle"
              ? "middle"
              : "right";
    d.mouse[name] = pressed;
}

/** Produce the DOM `buttons` bitmask. */
export function pointerButtons(state: State, buttons: number): void {
    pointerButton(state, "left", (buttons & 1) !== 0);
    pointerButton(state, "right", (buttons & 2) !== 0);
    pointerButton(state, "middle", (buttons & 4) !== 0);
}

/** Produce wheel movement. */
export function pointerWheel(state: State, delta: number): void {
    const d = record(state);
    if (d.enabled) d.mouse.scroll += delta;
}

/** Alias for callers that name the device fact `wheel`. */
export const wheel = pointerWheel;

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.hypot(b.x - a.x, b.y - a.y);
}

function updatePinchBaseline(d: DeviceRecord): void {
    if (d.touchPoints.size === 2) {
        const [a, b] = [...d.touchPoints.values()];
        d.pinchDistance = distance(a, b);
        d.centroidX = (a.x + b.x) / 2;
        d.centroidY = (a.y + b.y) / 2;
    } else {
        d.pinchDistance = null;
        d.centroidX = null;
        d.centroidY = null;
    }
}

/** Produce a touch-point update. `active=false` removes the point. Two-finger centroid and pinch deltas are
 * accumulated in the same record as DOM touch events. */
export function touchPoint(
    state: State,
    pointerId: number,
    x: number,
    y: number,
    active = true,
): void {
    const d = record(state);
    if (!d.enabled) return;
    if (!active) {
        d.touchPoints.delete(pointerId);
        d.touch.count = d.touchPoints.size;
        updatePinchBaseline(d);
        return;
    }
    const previous = d.touchPoints.get(pointerId);
    d.touchPoints.set(pointerId, { x, y });
    if (!previous) {
        d.touch.count = d.touchPoints.size;
        updatePinchBaseline(d);
        return;
    }
    if (d.touchPoints.size === 2) {
        const [a, b] = [...d.touchPoints.values()];
        const nextDistance = distance(a, b);
        const nextX = (a.x + b.x) / 2;
        const nextY = (a.y + b.y) / 2;
        if (d.pinchDistance !== null) d.touch.pinchDelta += nextDistance - d.pinchDistance;
        if (d.centroidX !== null && d.centroidY !== null) {
            d.touch.deltaX += nextX - d.centroidX;
            d.touch.deltaY += nextY - d.centroidY;
        }
        d.pinchDistance = nextDistance;
        d.centroidX = nextX;
        d.centroidY = nextY;
    }
}

function releaseKeyForLegacy(d: DeviceRecord, code: string): void {
    if (!d.keys.held.has(code)) return;
    d.keys.held.delete(code);
    d.keys.released.add(code);
    d.keys.tickReleased.add(code);
}

function pointerButtonsForRecord(d: DeviceRecord, buttons: number): void {
    d.mouse.left = (buttons & 1) !== 0;
    d.mouse.right = (buttons & 2) !== 0;
    d.mouse.middle = (buttons & 4) !== 0;
}

function locked(d: DeviceRecord): boolean {
    const element =
        typeof document === "undefined"
            ? null
            : (document.pointerLockElement as HTMLCanvasElement | null);
    return !!element && d.canvases.has(element);
}

function gatedButtons(d: DeviceRecord, buttons: number): number {
    return d.requireLock && !locked(d) ? 0 : buttons;
}

function releaseAll(state: State, d: DeviceRecord): void {
    for (const code of [...d.keys.held]) releaseKey(state, code);
    pointerButtons(state, 0);
}

function releaseCapture(d: DeviceRecord): void {
    d.activePointerId = null;
    d.activeButton = null;
    d.activeCanvas = null;
    d.lastPointerX = 0;
    d.lastPointerY = 0;
}

function recaptureTouch(d: DeviceRecord, canvas: HTMLCanvasElement): void {
    const [nextId, pos] = [...d.touchPoints.entries()][0];
    d.activePointerId = nextId;
    d.activeButton = 0;
    d.activeCanvas = canvas;
    d.lastPointerX = pos.x;
    d.lastPointerY = pos.y;
    try {
        canvas.setPointerCapture(nextId);
    } catch {}
}

function canvasPosition(
    state: State,
    target: HTMLCanvasElement,
    e: { clientX: number; clientY: number },
    hover = true,
) {
    const rect = target.getBoundingClientRect();
    pointerMove(state, {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        hover,
        canvasWidth: rect.width,
        canvasHeight: rect.height,
    });
}

// The compatibility facade below is intentionally only a bridge for pre-migration consumers. New code must
// use devices(state); the record itself is always reached through the State WeakMap.
let currentLegacy: DeviceRecord | null = null;

function createHandlers(d: DeviceRecord, state: State): void {
    d.pointerHover = (e) => {
        const target = e.target as HTMLCanvasElement;
        if (!d.canvases.has(target)) return;
        canvasPosition(state, target, e);
    };
    d.pointerEnter = d.pointerHover;
    d.pointerLeave = () => {
        if (d.activePointerId === null) d.mouse.hover = false;
    };
    d.keyDown = (e) => {
        if (!d.enabled) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!d.canvasFocused && !(lockElement && d.canvases.has(lockElement))) return;
        pressKey(state, e.code);
    };
    d.keyUp = (e) => {
        if (!d.enabled) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!d.canvasFocused && !(lockElement && d.canvases.has(lockElement))) return;
        releaseKey(state, e.code);
    };
    d.pointerDown = (e) => {
        const target = e.target as HTMLCanvasElement;
        const canvasIndex = d.canvases.get(target);
        if (canvasIndex === undefined) return;
        window.focus();
        if (e.pointerType === "touch") touchPoint(state, e.pointerId, e.clientX, e.clientY);
        if (d.activePointerId === null || d.activePointerId === e.pointerId) {
            d.pointerHover(e);
            pointerButtons(state, gatedButtons(d, e.buttons));
        }
        if (d.activePointerId === null) {
            d.activePointerId = e.pointerId;
            d.activeButton = e.button;
            d.activeCanvas = target;
            d.focused = canvasIndex;
            d.canvasFocused = true;
            d.lastPointerX = e.clientX;
            d.lastPointerY = e.clientY;
            try {
                target.setPointerCapture(e.pointerId);
            } catch {}
        }
        e.preventDefault();
    };
    d.windowPointerDown = (e) => {
        if (!d.canvases.has(e.target as HTMLCanvasElement)) {
            d.canvasFocused = false;
            releaseAll(state, d);
            d.keys.pressed.clear();
            d.keys.tickPressed.clear();
        }
    };
    d.windowBlur = () => {
        d.canvasFocused = false;
        releaseAll(state, d);
        d.touchPoints.clear();
        updatePinchBaseline(d);
        d.touch.count = 0;
        d.touch.pinchDelta = 0;
        d.touch.deltaX = 0;
        d.touch.deltaY = 0;
    };
    d.pointerUp = (e) => {
        const wasTouch = d.touchPoints.has(e.pointerId);
        if (wasTouch) touchPoint(state, e.pointerId, e.clientX, e.clientY, false);
        if (e.pointerId !== d.activePointerId) return;
        pointerButtons(state, gatedButtons(d, e.buttons));
        if (e.button === d.activeButton) {
            const canvas = d.activeCanvas;
            if (canvas?.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
            if (wasTouch && canvas && d.touchPoints.size > 0) recaptureTouch(d, canvas);
            else releaseCapture(d);
        }
    };
    d.pointerCancel = (e) => {
        const wasTouch = d.touchPoints.has(e.pointerId);
        if (wasTouch) touchPoint(state, e.pointerId, e.clientX, e.clientY, false);
        if (e.pointerId !== d.activePointerId) return;
        pointerButtons(state, 0);
        const canvas = d.activeCanvas;
        if (wasTouch && canvas && d.touchPoints.size > 0) recaptureTouch(d, canvas);
        else releaseCapture(d);
    };
    d.pointerMove = (e) => {
        if (d.touchPoints.has(e.pointerId)) touchPoint(state, e.pointerId, e.clientX, e.clientY);
        if (e.pointerId !== d.activePointerId) return;
        pointerButtons(state, gatedButtons(d, e.buttons));
        if (!(d.requireLock && locked(d))) e.preventDefault();
        pointerMove(state, {
            x: e.clientX - (d.activeCanvas?.getBoundingClientRect().left ?? 0),
            y: e.clientY - (d.activeCanvas?.getBoundingClientRect().top ?? 0),
            deltaX: e.clientX - d.lastPointerX,
            deltaY: e.clientY - d.lastPointerY,
            hover: true,
            canvasWidth: d.activeCanvas?.getBoundingClientRect().width,
            canvasHeight: d.activeCanvas?.getBoundingClientRect().height,
        });
        d.lastPointerX = e.clientX;
        d.lastPointerY = e.clientY;
    };
    d.wheel = (e) => {
        if (!d.canvases.has(e.target as HTMLCanvasElement)) return;
        pointerWheel(state, e.deltaY);
        e.preventDefault();
    };
    d.contextMenu = (e) => {
        if (d.canvases.has(e.target as HTMLCanvasElement)) e.preventDefault();
    };
}

function attachCanvas(d: DeviceRecord, canvas: HTMLCanvasElement, signal: AbortSignal): void {
    canvas.addEventListener("pointerdown", d.pointerDown, { signal });
    canvas.addEventListener("pointermove", d.pointerHover, { signal });
    canvas.addEventListener("pointerenter", d.pointerEnter, { signal });
    canvas.addEventListener("pointerleave", d.pointerLeave, { signal });
    canvas.addEventListener("wheel", d.wheel, { passive: false, signal });
    canvas.addEventListener("contextmenu", d.contextMenu, { signal });
}

function attachGlobal(d: DeviceRecord, signal: AbortSignal): void {
    window.addEventListener("keydown", d.keyDown, { signal });
    window.addEventListener("keyup", d.keyUp, { signal });
    window.addEventListener("pointerdown", d.windowPointerDown, { signal });
    window.addEventListener("pointerup", d.pointerUp, { signal });
    window.addEventListener("pointercancel", d.pointerCancel, { signal });
    window.addEventListener("pointermove", d.pointerMove, { signal });
    window.addEventListener("blur", d.windowBlur, { signal });
}

function setup(state: State, canvasElements: HTMLCanvasElement[]): void {
    const d = record(state);
    if (d.canvases.size > 0) return;
    for (let i = 0; i < canvasElements.length; i++) {
        d.canvases.set(canvasElements[i], i);
        canvasElements[i].style.touchAction = "none";
    }
    if (d.canvases.size === 0) return;
    createHandlers(d, state);
    attachGlobal(d, state.signal);
    for (const canvas of d.canvases.keys()) attachCanvas(d, canvas, state.signal);
    d.enabled = true;
    d.canvasFocused = true;
}

/** Legacy read facade. It remains only until the S4 consumer migration; State-scoped code uses
 * `devices(state)` and the producer functions above. */
export interface Inputs {
    readonly mouse: Readonly<Mouse>;
    readonly touch: Readonly<Touch>;
    readonly focused: number;
    isKeyDown(code: string): boolean;
    isKeyPressed(code: string): boolean;
    isKeyReleased(code: string): boolean;
}

export const Inputs: Inputs = {
    get mouse() {
        return currentLegacy?.mouse ?? DEFAULT_MOUSE;
    },
    get touch() {
        return currentLegacy?.touch ?? DEFAULT_TOUCH;
    },
    get focused() {
        return currentLegacy?.focused ?? -1;
    },
    isKeyDown(code) {
        return currentLegacy?.enabled ? currentLegacy.keys.held.has(code) : false;
    },
    isKeyPressed(code) {
        return currentLegacy?.enabled ? currentLegacy.keys.pressed.has(code) : false;
    },
    isKeyReleased(code) {
        return currentLegacy?.enabled ? currentLegacy.keys.released.has(code) : false;
    },
};

/** Suspend or resume the legacy current input binding. New code should keep this policy on its State until
 * the S2 migration lands. */
export function setInputEnabled(on: boolean): void {
    if (!currentLegacy) return;
    currentLegacy.enabled = on;
    if (!on) {
        // The legacy facade has no State argument; release its record directly while S2 migrates this API.
        for (const code of [...currentLegacy.keys.held]) releaseKeyForLegacy(currentLegacy, code);
        pointerButtonsForRecord(currentLegacy, 0);
        currentLegacy.keys.pressed.clear();
        currentLegacy.keys.tickPressed.clear();
        currentLegacy.mouse.deltaX = 0;
        currentLegacy.mouse.deltaY = 0;
        currentLegacy.mouse.scroll = 0;
        currentLegacy.touchPoints.clear();
        updatePinchBaseline(currentLegacy);
        currentLegacy.touch.count = 0;
        currentLegacy.touch.pinchDelta = 0;
        currentLegacy.touch.deltaX = 0;
        currentLegacy.touch.deltaY = 0;
    }
}

/** whether the legacy current input binding is live. */
export function inputEnabled(): boolean {
    return currentLegacy?.enabled ?? true;
}

/** Legacy pointer-lock button gate. S2 moves this producer to the State record. */
export function requirePointerLock(on: boolean): void {
    if (currentLegacy) currentLegacy.requireLock = on;
}

const InputSystem: System = {
    group: "simulation",
    setup(state: State) {
        // Ensure headless States have a record before their first producer call.
        record(state);
        if (typeof document === "undefined" || typeof document.querySelectorAll !== "function")
            return;
        const elements = Array.from(document.querySelectorAll("canvas"));
        if (elements.length > 0) setup(state, elements);
    },
    update(state: State) {
        // Compatibility only; this is not the source of device truth.
        currentLegacy = record(state);
    },
};

const InputTickResetSystem: System = {
    name: "tick-reset",
    group: "fixed",
    last: true,
    update(state: State) {
        const keys = record(state).keys;
        keys.tickPressed.clear();
        keys.tickReleased.clear();
    },
};

const InputResetSystem: System = {
    name: "frame-reset",
    group: "draw",
    last: true,
    update(state: State) {
        const d = record(state);
        d.keys.pressed.clear();
        d.keys.released.clear();
        d.mouse.deltaX = 0;
        d.mouse.deltaY = 0;
        d.mouse.scroll = 0;
        d.touch.pinchDelta = 0;
        d.touch.deltaX = 0;
        d.touch.deltaY = 0;
    },
};

/** Binds DOM listeners and installs the independent fixed- and frame-clock device-edge boundaries. */
export const InputPlugin: Plugin = {
    name: "Input",
    systems: [InputSystem, InputTickResetSystem, InputResetSystem],
};
