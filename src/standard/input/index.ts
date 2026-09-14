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
    /** pointer x normalized to the viewport row, in [0, 1] */
    normalizedX: number;
    /** pointer y normalized to the viewport row, in [0, 1] */
    normalizedY: number;
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

/** the browser audio context state carried by one State's device record. */
export type AudioContextState = "none" | "suspended" | "running" | "closed";

export interface AudioDevice {
    context: AudioContextState;
}

/** all device-fed facts for one State. Producers below are the single mutation seam used by both the DOM
 * path and headless callers. The record is created lazily, so a State with no DOM still has devices. */
export interface Pointer extends Mouse {
    readonly lock: PointerLock;
}

/** CSS display size and device-pixel ratio for one bound canvas. */
export interface Viewport {
    cssWidth: number;
    cssHeight: number;
    dpr: number;
}

export interface Devices {
    readonly keys: Keys;
    readonly audio: AudioDevice;
    /** pointer facts for the same record */
    readonly pointer: Pointer;
    readonly mouse: Mouse;
    readonly touch: Touch;
    /** viewport rows keyed by the bound canvas's document/index slot */
    readonly viewport: ReadonlyMap<number, Viewport>;
    /** true when device producers are suspended and all reads are neutral */
    readonly suspended: boolean;
    /** when true, pointer buttons stay up until `pointer.lock.status` is locked */
    readonly requireLock: boolean;
    /** document-order index of the canvas holding input focus, or -1 when none is focused */
    focused: number;
}

interface DeviceRecord extends Devices {
    audio: AudioDevice;
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
    pointerCanvasIndex: number;
    readonly viewport: Map<number, Viewport>;
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
    normalizedX: 0,
    normalizedY: 0,
};

const DEFAULT_TOUCH: Touch = { count: 0, pinchDelta: 0, deltaX: 0, deltaY: 0 };
const DEFAULT_AUDIO: AudioDevice = { context: "none" };
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
        audio: { ...DEFAULT_AUDIO },
        pointer,
        mouse: pointer,
        touch: { ...DEFAULT_TOUCH },
        viewport: new Map(),
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
        pointerCanvasIndex: -1,
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

function unit(value: number, size: number): number {
    if (!Number.isFinite(value) || size <= 0) return 0;
    return Math.min(Math.max(value / size, 0), 1);
}

function updateNormalized(d: DeviceRecord, index: number): void {
    const viewport = d.viewport.get(index);
    if (!viewport) return;
    d.mouse.normalizedX = unit(d.mouse.x, viewport.cssWidth);
    d.mouse.normalizedY = unit(d.mouse.y, viewport.cssHeight);
}

/** Produce the viewport row for one bound canvas/index. The DOM resize observer and headless callers use
 * the same seam; `dpr` is read by the DOM caller at resize time rather than by the renderer per frame. */
export function resizeViewport(
    state: State,
    index: number,
    width: number,
    height: number,
    dpr: number,
): void {
    const d = record(state);
    const viewport = {
        cssWidth: Math.max(0, Number.isFinite(width) ? width : 0),
        cssHeight: Math.max(0, Number.isFinite(height) ? height : 0),
        dpr: Number.isFinite(dpr) && dpr > 0 ? dpr : 1,
    };
    d.viewport.set(index, viewport);
    if (d.pointerCanvasIndex === index) updateNormalized(d, index);
}

/** Produce the browser audio context state for one State. */
export function audioContextState(state: State, context: AudioContextState): void {
    record(state).audio.context = context;
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
              canvasIndex?: number;
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
    const index =
        (typeof x === "number" ? undefined : move.canvasIndex) ??
        (d.pointerCanvasIndex >= 0 ? d.pointerCanvasIndex : d.focused);
    if (index >= 0) {
        d.pointerCanvasIndex = index;
        updateNormalized(d, index);
    }
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
    d.pointerCanvasIndex = canvasIndex;
    updateNormalized(d, canvasIndex);
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
        canvasIndex: record(state).canvases.get(target),
    });
}

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
            canvasIndex: d.pointerCanvasIndex,
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
            canvasIndex: d.activeCanvas === null ? undefined : d.canvases.get(d.activeCanvas),
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

/** Suspend or resume one State's device producers. Suspension releases held inputs with normal edges. */
export function setInputEnabled(state: State, on: boolean): void {
    const d = record(state);
    d.suspended = !on;
    if (d.suspended) {
        releaseAll(state, d);
        d.keys.pressed.clear();
        d.keys.tickPressed.clear();
        d.mouse.deltaX = 0;
        d.mouse.deltaY = 0;
        d.mouse.scroll = 0;
    }
}

/** whether one State's device producers are live. */
export function inputEnabled(state: State): boolean {
    return !record(state).suspended;
}

/** Set the pointer-button gate on one State. */
export function requirePointerLock(state: State, on: boolean): void {
    record(state).requireLock = on;
}

/** read the pointer-lock status from one State's device record. */
export function pointerLockStatus(state: State): PointerLockStatus {
    return record(state).pointer.lock.status;
}

/** read the browser's last pointer-lock refusal from one State's device record. */
export function pointerLockRefusal(state: State): string | null {
    return record(state).pointer.lock.refusal;
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
    update() {},
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
