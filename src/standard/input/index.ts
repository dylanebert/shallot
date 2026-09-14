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

/** pointer-lock state reported by the browser or a headless producer. */
export type PointerLockStatus = "unsupported" | "refused" | "unlocked" | "locked";

export interface PointerLock {
    status: PointerLockStatus;
    /** the last browser refusal, or the reason the capability is unavailable */
    refusal: string | null;
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
export interface Pointer extends Mouse {
    readonly lock: PointerLock;
}

export interface Devices {
    readonly keys: Keys;
    /** pointer facts; `mouse` is the compatibility name for the same record */
    readonly pointer: Pointer;
    readonly mouse: Mouse;
    readonly touch: Touch;
    /** true when device producers are suspended and all reads are neutral */
    readonly suspended: boolean;
    /** when true, pointer buttons stay up until `pointer.lock.status` is locked */
    readonly requireLock: boolean;
    /** document-order index of the canvas holding input focus, or -1 when none is focused */
    focused: number;
}

interface DeviceRecord extends Devices {
    suspended: boolean;
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
    visibilityChange: () => void;
    pointerLockChange: () => void;
    pointerLockError: () => void;
    canvasClick: () => void;
    lockMove: (e: MouseEvent) => void;
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
const DEFAULT_POINTER_LOCK: PointerLock = { status: "unlocked", refusal: null };
function emptyRecord(): DeviceRecord {
    const pointer: Pointer = { ...DEFAULT_MOUSE, lock: { ...DEFAULT_POINTER_LOCK } };
    return {
        keys: {
            held: new Set(),
            pressed: new Set(),
            released: new Set(),
            tickPressed: new Set(),
            tickReleased: new Set(),
            pressedTick: new Map(),
        },
        pointer,
        mouse: pointer,
        touch: { ...DEFAULT_TOUCH },
        suspended: false,
        focused: -1,
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
        visibilityChange: null!,
        pointerLockChange: null!,
        pointerLockError: null!,
        canvasClick: null!,
        lockMove: null!,
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
    if (d.suspended || d.keys.held.has(code)) return;
    d.keys.held.add(code);
    d.keys.pressed.add(code);
    d.keys.tickPressed.add(code);
    d.keys.pressedTick.set(code, state.time.fixedTick);
}

/** Produce a keyboard release. */
export function releaseKey(state: State, code: string): void {
    const d = record(state);
    if (d.suspended || !d.keys.held.has(code)) return;
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
    if (d.suspended) return;
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
    if (d.suspended) return;
    const name =
        button === 0 || button === "left"
            ? "left"
            : button === 1 || button === "middle"
              ? "middle"
              : "right";
    d.mouse[name] = d.requireLock && d.pointer.lock.status !== "locked" ? false : pressed;
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
    if (!d.suspended) d.mouse.scroll += delta;
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
    if (d.suspended) return;
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

function setPointerLock(d: DeviceRecord, status: PointerLockStatus, refusal: string | null): void {
    d.pointer.lock.status = status;
    d.pointer.lock.refusal = refusal;
}

/** Release every held input as the window loses focus. */
export function blur(state: State): void {
    const d = record(state);
    d.canvasFocused = false;
    d.focused = -1;
    releaseAll(state, d);
}

/** Produce a focus transition for a bound canvas. */
export function focus(state: State, canvasIndex = 0): void {
    const d = record(state);
    if (d.suspended) return;
    d.canvasFocused = true;
    d.focused = canvasIndex;
}

/** Produce a document visibility transition. Hidden visibility has the same release-edge contract as blur. */
export function visibilityChanged(state: State, hidden: boolean): void {
    if (hidden) blur(state);
}

/** Produce a pointer-lock transition. Exiting a lock is an input boundary and releases every held input. */
export function pointerLockChanged(
    state: State,
    engaged: boolean,
    refusal: string | null = null,
): void {
    const d = record(state);
    const wasLocked = d.pointer.lock.status === "locked";
    if (engaged) setPointerLock(d, "locked", null);
    else setPointerLock(d, refusal === null ? "unlocked" : "refused", refusal);
    if (wasLocked && !engaged) releaseAll(state, d);
}

function pointerButtonsForRecord(d: DeviceRecord, buttons: number): void {
    d.mouse.left = (buttons & 1) !== 0;
    d.mouse.right = (buttons & 2) !== 0;
    d.mouse.middle = (buttons & 4) !== 0;
}

function gatedButtons(d: DeviceRecord, buttons: number): number {
    return d.requireLock && d.pointer.lock.status !== "locked" ? 0 : buttons;
}

function clearTouch(d: DeviceRecord): void {
    d.touchPoints.clear();
    updatePinchBaseline(d);
    d.touch.count = 0;
    d.touch.pinchDelta = 0;
    d.touch.deltaX = 0;
    d.touch.deltaY = 0;
}

function releaseAll(_state: State | null, d: DeviceRecord): void {
    for (const code of [...d.keys.held]) releaseKeyForLegacy(d, code);
    pointerButtonsForRecord(d, 0);
    clearTouch(d);
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
        if (d.suspended) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!d.canvasFocused && !(lockElement && d.canvases.has(lockElement))) return;
        pressKey(state, e.code);
    };
    d.keyUp = (e) => {
        if (d.suspended) return;
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
            d.focused = -1;
            releaseAll(state, d);
        }
    };
    d.windowBlur = () => blur(state);
    d.visibilityChange = () => visibilityChanged(state, document.hidden);
    d.pointerLockChange = () => {
        const element = document.pointerLockElement as HTMLCanvasElement | null;
        if (element && d.canvases.has(element)) pointerLockChanged(state, true);
        else if (d.pointer.lock.status === "locked") pointerLockChanged(state, false);
    };
    d.pointerLockError = () => {
        pointerLockChanged(state, false, "the browser rejected pointer lock");
    };
    d.canvasClick = () => {
        if (d.requireLock) requestPointerLock(state);
    };
    d.lockMove = (e) => {
        if (d.pointer.lock.status !== "locked") return;
        pointerMove(state, {
            x: d.mouse.x,
            y: d.mouse.y,
            deltaX: e.movementX,
            deltaY: e.movementY,
            hover: true,
        });
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
        if (d.pointer.lock.status === "locked") return;
        e.preventDefault();
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
    canvas.addEventListener("click", d.canvasClick, { signal });
}

function attachGlobal(d: DeviceRecord, signal: AbortSignal): void {
    window.addEventListener("keydown", d.keyDown, { signal });
    window.addEventListener("keyup", d.keyUp, { signal });
    window.addEventListener("pointerdown", d.windowPointerDown, { signal });
    window.addEventListener("pointerup", d.pointerUp, { signal });
    window.addEventListener("pointercancel", d.pointerCancel, { signal });
    window.addEventListener("pointermove", d.pointerMove, { signal });
    window.addEventListener("blur", d.windowBlur, { signal });
    document.addEventListener("visibilitychange", d.visibilityChange, { signal });
    document.addEventListener("pointerlockchange", d.pointerLockChange, { signal });
    document.addEventListener("pointerlockerror", d.pointerLockError, { signal });
    document.addEventListener("mousemove", d.lockMove, { signal });
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
    const supported = [...d.canvases.keys()].some(
        (canvas) => typeof canvas.requestPointerLock === "function",
    );
    setPointerLock(
        d,
        supported ? "unlocked" : "unsupported",
        supported ? null : "canvas has no requestPointerLock",
    );
    d.canvasFocused = true;
}

/** Request pointer lock from an engagement gesture. This is the only browser effect in the lock seam. */
export function requestPointerLock(state: State): void {
    const d = record(state);
    if (d.suspended) return;
    if (d.pointer.lock.status === "unsupported") return;
    const canvas =
        d.activeCanvas ??
        [...d.canvases.entries()].find(([, index]) => index === d.focused)?.[0] ??
        [...d.canvases.keys()][0];
    if (!canvas || typeof canvas.requestPointerLock !== "function") {
        setPointerLock(d, "unsupported", "canvas has no requestPointerLock");
        return;
    }
    try {
        const result = canvas.requestPointerLock() as unknown as Promise<void> | undefined;
        if (result && typeof result.catch === "function") {
            result.catch((error: unknown) =>
                pointerLockChanged(
                    state,
                    false,
                    error instanceof Error ? error.message : String(error),
                ),
            );
        }
    } catch (error) {
        pointerLockChanged(state, false, error instanceof Error ? error.message : String(error));
    }
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
        return (
            currentLegacy !== null && !currentLegacy.suspended && currentLegacy.keys.held.has(code)
        );
    },
    isKeyPressed(code) {
        return (
            currentLegacy !== null &&
            !currentLegacy.suspended &&
            currentLegacy.keys.pressed.has(code)
        );
    },
    isKeyReleased(code) {
        return (
            currentLegacy !== null &&
            !currentLegacy.suspended &&
            currentLegacy.keys.released.has(code)
        );
    },
};

/** Suspend or resume one State's device producers. Suspension releases held inputs with normal edges. */
export function setInputEnabled(state: State, on: boolean): void;
/** @deprecated pass the State explicitly; retained until the S4 consumer migration. */
export function setInputEnabled(on: boolean): void;
export function setInputEnabled(stateOrOn: State | boolean, maybeOn?: boolean): void {
    const d = typeof stateOrOn === "boolean" ? currentLegacy : record(stateOrOn);
    if (!d) return;
    d.suspended = typeof stateOrOn === "boolean" ? !stateOrOn : !maybeOn;
    if (d.suspended) {
        releaseAll(null, d);
        d.keys.pressed.clear();
        d.keys.tickPressed.clear();
        d.mouse.deltaX = 0;
        d.mouse.deltaY = 0;
        d.mouse.scroll = 0;
    }
}

/** whether one State's device producers are live. */
export function inputEnabled(state: State): boolean;
/** @deprecated pass the State explicitly; retained until the S4 consumer migration. */
export function inputEnabled(): boolean;
export function inputEnabled(state?: State): boolean {
    const d = state ? record(state) : currentLegacy;
    return d ? !d.suspended : true;
}

/** Set the pointer-button gate on one State. */
export function requirePointerLock(state: State, on: boolean): void;
/** @deprecated pass the State explicitly; retained until the S4 consumer migration. */
export function requirePointerLock(on: boolean): void;
export function requirePointerLock(stateOrOn: State | boolean, maybeOn?: boolean): void {
    const d = typeof stateOrOn === "boolean" ? currentLegacy : record(stateOrOn);
    if (d) d.requireLock = typeof stateOrOn === "boolean" ? stateOrOn : (maybeOn ?? false);
}

/** read the pointer-lock status from one State's device record. */
export function pointerLockStatus(state: State): PointerLockStatus;
/** @deprecated pass the State explicitly; retained until the S4 consumer migration. */
export function pointerLockStatus(): PointerLockStatus;
export function pointerLockStatus(state?: State): PointerLockStatus {
    return state
        ? record(state).pointer.lock.status
        : (currentLegacy?.pointer.lock.status ?? "unlocked");
}

/** read the browser's last pointer-lock refusal from one State's device record. */
export function pointerLockRefusal(state: State): string | null;
/** @deprecated pass the State explicitly; retained until the S4 consumer migration. */
export function pointerLockRefusal(): string | null;
export function pointerLockRefusal(state?: State): string | null {
    return state
        ? record(state).pointer.lock.refusal
        : (currentLegacy?.pointer.lock.refusal ?? null);
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
