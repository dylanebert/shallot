import type { Plugin, State, System } from "../../engine";

/** Selects the host that produces device facts for one {@link State}. */
export type InputSource = "browser" | "controlled";

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
    /** host producer selected before input setup; selection is State-scoped and immutable after setup */
    readonly source: InputSource;
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
    source: InputSource;
    audio: AudioDevice;
    suspended: boolean;
    requireLock: boolean;
    setupStarted: boolean;
    readonly touchPoints: Map<number, { x: number; y: number }>;
    pinchDistance: number | null;
    centroidX: number | null;
    centroidY: number | null;
    pointerCanvasIndex: number;
    readonly viewport: Map<number, Viewport>;
}

/** Browser handles and callbacks belong to this adapter, never to the State's plain device facts. */
interface BrowserAdapter {
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
const adapters = new WeakMap<State, BrowserAdapter>();

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
        source: "browser",
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
        pointerCanvasIndex: -1,
        setupStarted: false,
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

function adapter(state: State): BrowserAdapter {
    const existing = adapters.get(state);
    if (existing) return existing;
    const created: BrowserAdapter = {
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
    adapters.set(state, created);
    return created;
}

/** Select the producer before the Input plugin's setup runs. */
export function setInputSource(state: State, source: InputSource): void {
    const d = record(state);
    if (d.setupStarted) throw new Error("input source cannot change after input setup");
    d.source = source;
}

/** Read the producer selected for one State. */
export function inputSource(state: State): InputSource {
    return record(state).source;
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

/** Produce the viewport row for a controlled caller. The DOM adapter uses {@link reportViewport}. */
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

/** Report a viewport row from a browser adapter; controlled facts reject host reports. */
export function reportViewport(
    state: State,
    index: number,
    width: number,
    height: number,
    dpr: number,
): void {
    if (record(state).source === "controlled") return;
    resizeViewport(state, index, width, height, dpr);
}

/** Produce the audio context state supplied by a controlled caller. */
export function audioContextState(state: State, context: AudioContextState): void {
    record(state).audio.context = context;
}

/** Report audio context state from the browser adapter; controlled facts reject host reports. */
export function reportAudioContextState(state: State, context: AudioContextState): void {
    if (record(state).source === "controlled") return;
    audioContextState(state, context);
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
    const a = adapters.get(state);
    if (a) a.canvasFocused = false;
    d.focused = -1;
    releaseAll(state, d);
}

/** Produce a focus transition for a bound canvas. */
export function focus(state: State, canvasIndex = 0): void {
    const d = record(state);
    if (d.suspended) return;
    const a = adapters.get(state);
    if (a) a.canvasFocused = true;
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

function releaseCapture(a: BrowserAdapter): void {
    a.activePointerId = null;
    a.activeButton = null;
    a.activeCanvas = null;
    a.lastPointerX = 0;
    a.lastPointerY = 0;
}

function recaptureTouch(a: BrowserAdapter, d: DeviceRecord, canvas: HTMLCanvasElement): void {
    const [nextId, pos] = [...d.touchPoints.entries()][0];
    a.activePointerId = nextId;
    a.activeButton = 0;
    a.activeCanvas = canvas;
    a.lastPointerX = pos.x;
    a.lastPointerY = pos.y;
    try {
        canvas.setPointerCapture(nextId);
    } catch {}
}

function canvasPosition(
    state: State,
    a: BrowserAdapter,
    target: HTMLCanvasElement,
    e: { clientX: number; clientY: number },
    hover = true,
) {
    const rect = target.getBoundingClientRect();
    pointerMove(state, {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        hover,
        canvasIndex: a.canvases.get(target),
    });
}

function createHandlers(a: BrowserAdapter, d: DeviceRecord, state: State): void {
    a.pointerHover = (e) => {
        const target = e.target as HTMLCanvasElement;
        if (!a.canvases.has(target)) return;
        canvasPosition(state, a, target, e);
    };
    a.pointerEnter = a.pointerHover;
    a.pointerLeave = () => {
        if (a.activePointerId === null) d.mouse.hover = false;
    };
    a.keyDown = (e) => {
        if (d.suspended) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!a.canvasFocused && !(lockElement && a.canvases.has(lockElement))) return;
        pressKey(state, e.code);
    };
    a.keyUp = (e) => {
        if (d.suspended) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!a.canvasFocused && !(lockElement && a.canvases.has(lockElement))) return;
        releaseKey(state, e.code);
    };
    a.pointerDown = (e) => {
        const target = e.target as HTMLCanvasElement;
        const canvasIndex = a.canvases.get(target);
        if (canvasIndex === undefined) return;
        window.focus();
        if (e.pointerType === "touch") touchPoint(state, e.pointerId, e.clientX, e.clientY);
        if (a.activePointerId === null || a.activePointerId === e.pointerId) {
            a.pointerHover(e);
            pointerButtons(state, gatedButtons(d, e.buttons));
        }
        if (a.activePointerId === null) {
            a.activePointerId = e.pointerId;
            a.activeButton = e.button;
            a.activeCanvas = target;
            d.focused = canvasIndex;
            a.canvasFocused = true;
            a.lastPointerX = e.clientX;
            a.lastPointerY = e.clientY;
            try {
                target.setPointerCapture(e.pointerId);
            } catch {}
        }
        e.preventDefault();
    };
    a.windowPointerDown = (e) => {
        if (!a.canvases.has(e.target as HTMLCanvasElement)) {
            a.canvasFocused = false;
            d.focused = -1;
            releaseAll(state, d);
        }
    };
    a.windowBlur = () => blur(state);
    a.visibilityChange = () => visibilityChanged(state, document.hidden);
    a.pointerLockChange = () => {
        const element = document.pointerLockElement as HTMLCanvasElement | null;
        if (element && a.canvases.has(element)) pointerLockChanged(state, true);
        else if (d.pointer.lock.status === "locked") pointerLockChanged(state, false);
    };
    a.pointerLockError = () => {
        pointerLockChanged(state, false, "the browser rejected pointer lock");
    };
    a.canvasClick = () => {
        if (d.requireLock) requestPointerLock(state);
    };
    a.lockMove = (e) => {
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
    a.pointerUp = (e) => {
        const wasTouch = d.touchPoints.has(e.pointerId);
        if (wasTouch) touchPoint(state, e.pointerId, e.clientX, e.clientY, false);
        if (e.pointerId !== a.activePointerId) return;
        pointerButtons(state, gatedButtons(d, e.buttons));
        if (e.button === a.activeButton) {
            const canvas = a.activeCanvas;
            if (canvas?.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
            if (wasTouch && canvas && d.touchPoints.size > 0) recaptureTouch(a, d, canvas);
            else releaseCapture(a);
        }
    };
    a.pointerCancel = (e) => {
        const wasTouch = d.touchPoints.has(e.pointerId);
        if (wasTouch) touchPoint(state, e.pointerId, e.clientX, e.clientY, false);
        if (e.pointerId !== a.activePointerId) return;
        pointerButtons(state, 0);
        const canvas = a.activeCanvas;
        if (wasTouch && canvas && d.touchPoints.size > 0) recaptureTouch(a, d, canvas);
        else releaseCapture(a);
    };
    a.pointerMove = (e) => {
        if (d.touchPoints.has(e.pointerId)) touchPoint(state, e.pointerId, e.clientX, e.clientY);
        if (e.pointerId !== a.activePointerId) return;
        pointerButtons(state, gatedButtons(d, e.buttons));
        if (d.pointer.lock.status === "locked") return;
        e.preventDefault();
        pointerMove(state, {
            x: e.clientX - (a.activeCanvas?.getBoundingClientRect().left ?? 0),
            y: e.clientY - (a.activeCanvas?.getBoundingClientRect().top ?? 0),
            deltaX: e.clientX - a.lastPointerX,
            deltaY: e.clientY - a.lastPointerY,
            hover: true,
            canvasIndex: a.activeCanvas === null ? undefined : a.canvases.get(a.activeCanvas),
        });
        a.lastPointerX = e.clientX;
        a.lastPointerY = e.clientY;
    };
    a.wheel = (e) => {
        if (!a.canvases.has(e.target as HTMLCanvasElement)) return;
        pointerWheel(state, e.deltaY);
        e.preventDefault();
    };
    a.contextMenu = (e) => {
        if (a.canvases.has(e.target as HTMLCanvasElement)) e.preventDefault();
    };
}

function attachCanvas(a: BrowserAdapter, canvas: HTMLCanvasElement, signal: AbortSignal): void {
    canvas.addEventListener("pointerdown", a.pointerDown, { signal });
    canvas.addEventListener("pointermove", a.pointerHover, { signal });
    canvas.addEventListener("pointerenter", a.pointerEnter, { signal });
    canvas.addEventListener("pointerleave", a.pointerLeave, { signal });
    canvas.addEventListener("wheel", a.wheel, { passive: false, signal });
    canvas.addEventListener("contextmenu", a.contextMenu, { signal });
    canvas.addEventListener("click", a.canvasClick, { signal });
}

function attachGlobal(a: BrowserAdapter, signal: AbortSignal): void {
    window.addEventListener("keydown", a.keyDown, { signal });
    window.addEventListener("keyup", a.keyUp, { signal });
    window.addEventListener("pointerdown", a.windowPointerDown, { signal });
    window.addEventListener("pointerup", a.pointerUp, { signal });
    window.addEventListener("pointercancel", a.pointerCancel, { signal });
    window.addEventListener("pointermove", a.pointerMove, { signal });
    window.addEventListener("blur", a.windowBlur, { signal });
    document.addEventListener("visibilitychange", a.visibilityChange, { signal });
    document.addEventListener("pointerlockchange", a.pointerLockChange, { signal });
    document.addEventListener("pointerlockerror", a.pointerLockError, { signal });
    document.addEventListener("mousemove", a.lockMove, { signal });
}

function setup(state: State, canvasElements: HTMLCanvasElement[]): void {
    const d = record(state);
    const a = adapter(state);
    if (a.canvases.size > 0) return;
    for (let i = 0; i < canvasElements.length; i++) {
        a.canvases.set(canvasElements[i], i);
        canvasElements[i].style.touchAction = "none";
    }
    if (a.canvases.size === 0) return;
    createHandlers(a, d, state);
    attachGlobal(a, state.signal);
    for (const canvas of a.canvases.keys()) attachCanvas(a, canvas, state.signal);
    const supported = [...a.canvases.keys()].some(
        (canvas) => typeof canvas.requestPointerLock === "function",
    );
    setPointerLock(
        d,
        supported ? "unlocked" : "unsupported",
        supported ? null : "canvas has no requestPointerLock",
    );
    a.canvasFocused = true;
}

/** Request pointer lock from an engagement gesture. This is the only browser effect in the lock seam. */
export function requestPointerLock(state: State): void {
    const d = record(state);
    if (d.suspended || d.source === "controlled") return;
    if (d.pointer.lock.status === "unsupported") return;
    const a = adapters.get(state);
    if (!a) return;
    const canvas =
        a.activeCanvas ??
        [...a.canvases.entries()].find(([, index]) => index === d.focused)?.[0] ??
        [...a.canvases.keys()][0];
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
        const d = record(state);
        d.setupStarted = true;
        // Controlled production owns its facts and never probes or touches a host, even when one exists.
        if (d.source === "controlled") return;
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
        // `Set.prototype.clear` mints a fresh table even on an empty set, so guard on size.
        if (keys.tickPressed.size !== 0) keys.tickPressed.clear();
        if (keys.tickReleased.size !== 0) keys.tickReleased.clear();
    },
};

const InputResetSystem: System = {
    name: "frame-reset",
    group: "draw",
    last: true,
    update(state: State) {
        const d = record(state);
        if (d.keys.pressed.size !== 0) d.keys.pressed.clear();
        if (d.keys.released.size !== 0) d.keys.released.clear();
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
