import { type Plugin, type System, Viewports, type World } from "../../engine";

/** Keyboard facts owned by one {@link World}. `pressed`/`released` are the frame latches. */
export interface Keys {
    readonly held: Set<string>;
    /** keys pressed since the last draw boundary */
    readonly pressed: Set<string>;
    /** keys released since the last draw boundary */
    readonly released: Set<string>;
}

/** pointer-lock state reported by the browser or a headless producer. */
export type PointerLockStatus = "unsupported" | "refused" | "unlocked" | "locked";

export interface PointerLock {
    status: PointerLockStatus;
    /** the last browser refusal, or the reason the capability is unavailable */
    refusal: string | null;
}

/** one pointer-button press or release, at the pointer position in CSS pixels within the focused canvas */
export interface PointerButtonEvent {
    readonly button: "left" | "right" | "middle";
    /** true for a press, false for a release */
    readonly pressed: boolean;
    readonly x: number;
    readonly y: number;
}

/** live pointer state in one device record. Positions and sizes are CSS pixels; deltas accumulate over a
 * frame and reset at the draw boundary. A one-finger touch drives `left` and the position; a second finger
 * cancels that press. */
export interface Pointer {
    readonly lock: PointerLock;
    /** horizontal pointer movement since the last frame, in CSS pixels */
    deltaX: number;
    /** vertical pointer movement since the last frame, in CSS pixels */
    deltaY: number;
    /** wheel movement accumulated this frame; positive scrolls down/away */
    scroll: number;
    left: boolean;
    right: boolean;
    middle: boolean;
    /** every button press and release since the last draw boundary, in order. Entries are reused after that
     * boundary. A cancelled press (a second touch, or `pointercancel`) ends with the button up and no release. */
    readonly buttonEvents: readonly PointerButtonEvent[];
    /** pointer is over a bound canvas, or a drag is active */
    hover: boolean;
    /** pointer x within the focused canvas, CSS pixels from the left edge */
    x: number;
    /** pointer y within the focused canvas, CSS pixels from the top edge */
    y: number;
    /** pointer x normalized to the viewport row, in [0, 1] */
    readonly normalizedX: number;
    /** pointer y normalized to the viewport row, in [0, 1] */
    readonly normalizedY: number;
}

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

/** the browser audio context state carried by one World's device record. */
export type AudioContextState = "none" | "suspended" | "running" | "closed";

export interface AudioDevice {
    /** written by the audio runtime; a headless driver writes it directly */
    context: AudioContextState;
}

/** all device-fed facts for one World, read directly. `keys.pressed`, `keys.released`, `pointer.buttonEvents`,
 * `pointer.scroll` and the pointer and touch deltas cover one frame; the input plugin clears
 * them at the draw boundary. A game or headless driver changes every field except `audio.context` through the
 * producers below, which keep edges and boundary releases; the browser producer writes the same fields from DOM events. `audio.context` has no
 * producer: the audio runtime or a headless driver writes it directly. The record is created lazily, so a World
 * with no DOM still has devices. */
export interface Devices {
    readonly keys: Keys;
    readonly audio: AudioDevice;
    readonly pointer: Pointer;
    readonly touch: Touch;
    /** false while {@link setInputEnabled} suspends the device producers; reads are then neutral */
    readonly enabled: boolean;
    /** when true, pointer buttons stay up until `pointer.lock.status` is locked */
    readonly requireLock: boolean;
    /** document-order index of the canvas holding input focus, or -1 when none is focused */
    focused: number;
}

interface DeviceRecord extends Devices {
    audio: AudioDevice;
    enabled: boolean;
    requireLock: boolean;
    readonly touchPoints: Map<number, { x: number; y: number }>;
    pinchDistance: number | null;
    centroidX: number | null;
    centroidY: number | null;
    pointerCanvasIndex: number;
    readonly buttonEvents: Mutable<PointerButtonEvent>[];
    /** every event record allocated so far, reused across frames */
    readonly buttonEventPool: Mutable<PointerButtonEvent>[];
    /** the touch that drives the pointer position, or null with no touch held */
    primaryTouch: number | null;
    /** the primary touch's press of `left` stands: no second touch has cancelled it */
    touchPress: boolean;
}

type ButtonName = PointerButtonEvent["button"];
type Mutable<T> = { -readonly [K in keyof T]: T[K] };

/** Host effects for the browser producer; the default host is created only when that producer is composed. */
export interface InputHost {
    readonly window: Window;
    readonly document: Document;
    queryCanvases(): readonly HTMLCanvasElement[];
    supportsPointerLock(canvas: HTMLCanvasElement): boolean;
    requestPointerLock(canvas: HTMLCanvasElement): void | PromiseLike<void>;
    releasePointerLock(canvas: HTMLCanvasElement): void;
}

interface ListenerRegistration {
    readonly target: EventTarget;
    readonly type: string;
    readonly listener: EventListener;
    readonly options?: AddEventListenerOptions | boolean;
}

/** Browser handles, effects and callbacks belong to this adapter, never to the World's plain device facts. */
interface BrowserAdapter {
    readonly host: InputHost;
    readonly canvases: Map<HTMLCanvasElement, number>;
    readonly listeners: ListenerRegistration[];
    readonly canvasStyles: Map<HTMLCanvasElement, string>;
    readonly pendingLocks: Set<HTMLCanvasElement>;
    readonly ownedCanvases: Set<HTMLCanvasElement>;
    activeCanvas: HTMLCanvasElement | null;
    lockCanvas: HTMLCanvasElement | null;
    disposed: boolean;
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

/** This world's input devices; available without DOM or InputPlugin setup. */
export const Devices: import("../../engine").Resource<Devices> = {
    create(world) {
        const devices = emptyRecord(world);
        world.onDispose(() => {
            devices.requireLock = false;
        });
        return devices;
    },
};
const adapters = new WeakMap<World, BrowserAdapter>();

const DEFAULT_POINTER: Omit<Pointer, "lock" | "normalizedX" | "normalizedY" | "buttonEvents"> = {
    deltaX: 0,
    deltaY: 0,
    scroll: 0,
    left: false,
    right: false,
    middle: false,
    hover: false,
    x: 0,
    y: 0,
};

const DEFAULT_TOUCH: Touch = { count: 0, pinchDelta: 0, deltaX: 0, deltaY: 0 };
const DEFAULT_AUDIO: AudioDevice = { context: "none" };
const DEFAULT_POINTER_LOCK: PointerLock = { status: "unlocked", refusal: null };
function emptyRecord(world: World): DeviceRecord {
    const buttonEvents: Mutable<PointerButtonEvent>[] = [];
    const pointer: Pointer = {
        ...DEFAULT_POINTER,
        lock: { ...DEFAULT_POINTER_LOCK },
        buttonEvents,
        get normalizedX() {
            const viewport = world.resource(Viewports).get(devices.pointerCanvasIndex);
            return unit(pointer.x, viewport?.cssWidth ?? 0);
        },
        get normalizedY() {
            const viewport = world.resource(Viewports).get(devices.pointerCanvasIndex);
            return unit(pointer.y, viewport?.cssHeight ?? 0);
        },
    };
    const devices: DeviceRecord = {
        keys: {
            held: new Set(),
            pressed: new Set(),
            released: new Set(),
        },
        audio: { ...DEFAULT_AUDIO },
        pointer,
        touch: { ...DEFAULT_TOUCH },
        enabled: true,
        focused: -1,
        requireLock: false,
        touchPoints: new Map(),
        pinchDistance: null,
        centroidX: null,
        centroidY: null,
        pointerCanvasIndex: -1,
        buttonEvents,
        buttonEventPool: [],
        primaryTouch: null,
        touchPress: false,
    };
    return devices;
}

function record(world: World): DeviceRecord {
    return world.resource(Devices) as DeviceRecord;
}

const lockOwners = new WeakMap<HTMLCanvasElement, BrowserAdapter>();

function ownLock(a: BrowserAdapter, canvas: HTMLCanvasElement): void {
    lockOwners.set(canvas, a);
    a.ownedCanvases.add(canvas);
}

function releaseOwnedLock(a: BrowserAdapter, canvas: HTMLCanvasElement): void {
    a.ownedCanvases.delete(canvas);
    if (lockOwners.get(canvas) === a) lockOwners.delete(canvas);
    if (a.lockCanvas === canvas) a.lockCanvas = null;
}

function adapter(world: World, host: InputHost): BrowserAdapter {
    const existing = adapters.get(world);
    if (existing && !existing.disposed) return existing;
    const created: BrowserAdapter = {
        host,
        canvases: new Map(),
        listeners: [],
        canvasStyles: new Map(),
        pendingLocks: new Set(),
        ownedCanvases: new Set(),
        activeCanvas: null,
        lockCanvas: null,
        disposed: false,
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
    adapters.set(world, created);
    return created;
}

function unit(value: number, size: number): number {
    if (!Number.isFinite(value) || size <= 0) return 0;
    return Math.min(Math.max(value / size, 0), 1);
}

/** Produce a keyboard press. Repeated presses do not retrigger an edge. */
export function pressKey(world: World, code: string): void {
    const d = record(world);
    if (!d.enabled || d.keys.held.has(code)) return;
    d.keys.held.add(code);
    d.keys.pressed.add(code);
}

export function releaseKey(world: World, code: string): void {
    const d = record(world);
    if (!d.enabled || !d.keys.held.has(code)) return;
    d.keys.held.delete(code);
    d.keys.released.add(code);
}

/** Set the position in CSS pixels and accumulate movement deltas. */
export function pointerMove(
    world: World,
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
    const d = record(world);
    if (!d.enabled) return;
    const move = typeof x === "number" ? { x, y: y ?? 0, deltaX, deltaY } : x;
    d.pointer.x = move.x;
    d.pointer.y = move.y;
    d.pointer.deltaX += move.deltaX ?? 0;
    d.pointer.deltaY += move.deltaY ?? 0;
    if (move.hover !== undefined) d.pointer.hover = move.hover;
    const index =
        (typeof x === "number" ? undefined : move.canvasIndex) ??
        (d.pointerCanvasIndex >= 0 ? d.pointerCanvasIndex : d.focused);
    if (index >= 0) {
        d.pointerCanvasIndex = index;
    }
}

export type PointerButton = "left" | "right" | "middle" | 0 | 1 | 2;

/** Set a button level, recording the edge at the current pointer position. */
function setButtonLevel(d: DeviceRecord, name: ButtonName, down: boolean): void {
    const next = down && !(d.requireLock && d.pointer.lock.status !== "locked");
    if (d.pointer[name] === next) return;
    d.pointer[name] = next;
    const index = d.buttonEvents.length;
    let event = d.buttonEventPool[index];
    if (!event) {
        event = { button: name, pressed: next, x: 0, y: 0 };
        d.buttonEventPool.push(event);
    }
    event.button = name;
    event.pressed = next;
    event.x = d.pointer.x;
    event.y = d.pointer.y;
    d.buttonEvents.push(event);
}

function setButton(world: World, button: PointerButton, pressed: boolean): void {
    const d = record(world);
    if (!d.enabled) return;
    const name =
        button === 0 || button === "left"
            ? "left"
            : button === 1 || button === "middle"
              ? "middle"
              : "right";
    setButtonLevel(d, name, pressed);
}

/** Produce a pointer-button press at the current pointer position. While `requireLock` holds and the pointer
 * is not locked, the button stays up. Numeric buttons use DOM `button` values (0 left, 1 middle, 2 right). */
export function pressButton(world: World, button: PointerButton): void {
    setButton(world, button, true);
}

export function releaseButton(world: World, button: PointerButton): void {
    setButton(world, button, false);
}

/** Produce the DOM `buttons` bitmask. */
export function pointerButtons(world: World, buttons: number): void {
    setButton(world, "left", (buttons & 1) !== 0);
    setButton(world, "right", (buttons & 2) !== 0);
    setButton(world, "middle", (buttons & 4) !== 0);
}

/** Accumulate wheel movement; positive scrolls down/away. */
export function pointerWheel(world: World, delta: number): void {
    const d = record(world);
    if (d.enabled) d.pointer.scroll += delta;
}

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

/** Remove a touch point; the next held touch, if any, drives the pointer without pressing it. */
function liftTouch(d: DeviceRecord, pointerId: number): void {
    d.touchPoints.delete(pointerId);
    d.touch.count = d.touchPoints.size;
    updatePinchBaseline(d);
    if (pointerId !== d.primaryTouch) return;
    d.touchPress = false;
    d.primaryTouch = d.touchPoints.keys().next().value ?? null;
}

/** Produce a touch point in CSS pixels within the focused canvas; `active=false` lifts it. The first touch moves
 * the pointer and presses `left`, and lifting it releases `left` there. A second touch cancels that press, so a
 * pinch or pan presents no press and release; while fingers stay held, the remaining one drives the pointer
 * position. Accumulates two-finger centroid and pinch deltas. */
export function touchPoint(
    world: World,
    pointerId: number,
    x: number,
    y: number,
    active = true,
): void {
    const d = record(world);
    if (!d.enabled) return;
    const previous = d.touchPoints.get(pointerId);
    if (!active) {
        if (!previous) return;
        if (pointerId === d.primaryTouch) {
            pointerMove(world, x, y, x - previous.x, y - previous.y);
            if (d.touchPress) setButtonLevel(d, "left", false);
        }
        liftTouch(d, pointerId);
        return;
    }
    if (previous) {
        const fromX = previous.x;
        const fromY = previous.y;
        previous.x = x;
        previous.y = y;
        if (pointerId === d.primaryTouch) pointerMove(world, x, y, x - fromX, y - fromY);
    } else {
        d.touchPoints.set(pointerId, { x, y });
        d.touch.count = d.touchPoints.size;
        updatePinchBaseline(d);
        if (d.touchPoints.size === 1) {
            d.primaryTouch = pointerId;
            pointerMove(world, x, y);
            setButtonLevel(d, "left", true);
            d.touchPress = d.pointer.left;
        } else {
            cancelTouchPress(d);
        }
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

/** Drop the frame's press events, keeping its releases in order. `buttonEvents[i]` is always
 * `buttonEventPool[i]`, so the next event reuses a record no listed event holds. */
function dropPresses(d: DeviceRecord): void {
    const pool = d.buttonEventPool;
    let kept = 0;
    for (let i = 0; i < d.buttonEvents.length; i++) {
        const event = pool[i];
        if (event.pressed) continue;
        pool[i] = pool[kept];
        pool[kept] = event;
        d.buttonEvents[kept++] = event;
    }
    d.buttonEvents.length = kept;
}

function releaseHeldKey(d: DeviceRecord, code: string): void {
    if (!d.keys.held.has(code)) return;
    d.keys.held.delete(code);
    d.keys.released.add(code);
}

function setPointerLock(d: DeviceRecord, status: PointerLockStatus, refusal: string | null): void {
    d.pointer.lock.status = status;
    d.pointer.lock.refusal = refusal;
}

/** Release every held input as the window loses focus or the page is hidden. */
export function blurCanvas(world: World): void {
    const d = record(world);
    const a = adapters.get(world);
    if (a) a.canvasFocused = false;
    d.focused = -1;
    releaseAll(world, d);
}

/** Produce a focus transition for a bound canvas. */
export function focusCanvas(world: World, canvasIndex = 0): void {
    const d = record(world);
    if (!d.enabled) return;
    const a = adapters.get(world);
    if (a) a.canvasFocused = true;
    d.focused = canvasIndex;
    d.pointerCanvasIndex = canvasIndex;
}

/** Produce a pointer-lock transition. Exiting a lock is an input boundary and releases every held input. */
export function pointerLockChanged(
    world: World,
    engaged: boolean,
    refusal: string | null = null,
): void {
    const d = record(world);
    const wasLocked = d.pointer.lock.status === "locked";
    if (engaged) setPointerLock(d, "locked", null);
    else setPointerLock(d, refusal === null ? "unlocked" : "refused", refusal);
    if (wasLocked && !engaged) releaseAll(world, d);
}

/** End every held button without a release. */
function cancelButtons(d: DeviceRecord): void {
    d.pointer.left = false;
    d.pointer.right = false;
    d.pointer.middle = false;
    d.touchPress = false;
}

/** End the primary touch's press of `left` without a release. */
function cancelTouchPress(d: DeviceRecord): void {
    if (!d.touchPress) return;
    d.touchPress = false;
    d.pointer.left = false;
}

function gatedButtons(d: DeviceRecord, buttons: number): number {
    return d.requireLock && d.pointer.lock.status !== "locked" ? 0 : buttons;
}

function clearTouch(d: DeviceRecord): void {
    d.touchPoints.clear();
    updatePinchBaseline(d);
    d.primaryTouch = null;
    d.touchPress = false;
    d.touch.count = 0;
    d.touch.pinchDelta = 0;
    d.touch.deltaX = 0;
    d.touch.deltaY = 0;
}

function releaseAll(_world: World | null, d: DeviceRecord): void {
    for (const code of [...d.keys.held]) releaseHeldKey(d, code);
    setButtonLevel(d, "left", false);
    setButtonLevel(d, "right", false);
    setButtonLevel(d, "middle", false);
    clearTouch(d);
}

function releaseCapture(a: BrowserAdapter): void {
    a.activePointerId = null;
    a.activeButton = null;
    a.activeCanvas = null;
    a.lastPointerX = 0;
    a.lastPointerY = 0;
}

/** Move capture to the touch that now drives the pointer, or release it when none is held. */
function followTouch(a: BrowserAdapter, d: DeviceRecord, pointerId: number): void {
    if (pointerId !== a.activePointerId) return;
    const canvas = a.activeCanvas;
    if (canvas?.hasPointerCapture(pointerId)) canvas.releasePointerCapture(pointerId);
    const next = d.primaryTouch;
    if (canvas === null || next === null) {
        releaseCapture(a);
        return;
    }
    a.activePointerId = next;
    a.activeButton = 0;
    try {
        canvas.setPointerCapture(next);
    } catch {}
}

/** The touch's position within the canvas it is captured to, or else the one it landed on. */
function touchPosition(
    a: BrowserAdapter,
    target: EventTarget | null,
    e: { clientX: number; clientY: number },
): { x: number; y: number } {
    const canvas = a.activeCanvas ?? (a.canvases.has(target as HTMLCanvasElement) ? target : null);
    const rect = (canvas as HTMLCanvasElement | null)?.getBoundingClientRect();
    return { x: e.clientX - (rect?.left ?? 0), y: e.clientY - (rect?.top ?? 0) };
}

function canvasPosition(
    world: World,
    a: BrowserAdapter,
    target: HTMLCanvasElement,
    e: { clientX: number; clientY: number },
    hover = true,
) {
    const rect = target.getBoundingClientRect();
    pointerMove(world, {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        hover,
        canvasIndex: a.canvases.get(target),
    });
}

function createHandlers(a: BrowserAdapter, d: DeviceRecord, world: World): void {
    const { document, window } = a.host;
    a.pointerHover = (e) => {
        const target = e.target as HTMLCanvasElement;
        if (!a.canvases.has(target)) return;
        canvasPosition(world, a, target, e);
    };
    a.pointerEnter = a.pointerHover;
    a.pointerLeave = () => {
        if (a.activePointerId === null) d.pointer.hover = false;
    };
    a.keyDown = (e) => {
        if (a.disposed || !d.enabled) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!a.canvasFocused && !(lockElement && a.canvases.has(lockElement))) return;
        pressKey(world, e.code);
    };
    a.keyUp = (e) => {
        if (a.disposed || !d.enabled) return;
        const lockElement = document.pointerLockElement as HTMLCanvasElement | null;
        if (!a.canvasFocused && !(lockElement && a.canvases.has(lockElement))) return;
        releaseKey(world, e.code);
    };
    a.pointerDown = (e) => {
        const target = e.target as HTMLCanvasElement;
        const canvasIndex = a.canvases.get(target);
        if (canvasIndex === undefined) return;
        window.focus();
        if (e.pointerType === "touch") {
            if (d.touchPoints.size === 0) {
                d.pointerCanvasIndex = canvasIndex;
                d.pointer.hover = true;
            }
            const at = touchPosition(a, target, e);
            touchPoint(world, e.pointerId, at.x, at.y);
        } else if (a.activePointerId === null || a.activePointerId === e.pointerId) {
            a.pointerHover(e);
            pointerButtons(world, gatedButtons(d, e.buttons));
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
            releaseAll(world, d);
        }
    };
    a.windowBlur = () => blurCanvas(world);
    a.visibilityChange = () => {
        if (document.hidden) blurCanvas(world);
    };
    a.pointerLockChange = () => {
        if (a.disposed) return;
        const element = document.pointerLockElement as HTMLCanvasElement | null;
        if (element && a.canvases.has(element)) {
            const owner = lockOwners.get(element);
            if (owner && owner !== a) return;
            if (!owner && !a.pendingLocks.has(element)) return;
            for (const canvas of a.ownedCanvases) {
                if (canvas !== element && !a.pendingLocks.has(canvas)) releaseOwnedLock(a, canvas);
            }
            ownLock(a, element);
            a.lockCanvas = element;
            pointerLockChanged(world, true);
        } else {
            if (d.pointer.lock.status === "locked") pointerLockChanged(world, false);
            for (const canvas of a.ownedCanvases) {
                if (!a.pendingLocks.has(canvas)) releaseOwnedLock(a, canvas);
            }
        }
    };
    a.pointerLockError = () => {
        if (a.disposed || a.ownedCanvases.size === 0) return;
        pointerLockChanged(world, false, "the browser rejected pointer lock");
        for (const canvas of a.ownedCanvases) {
            if (a.host.document.pointerLockElement !== canvas) releaseOwnedLock(a, canvas);
        }
    };
    a.canvasClick = () => {
        if (d.requireLock) requestPointerLock(world);
    };
    a.lockMove = (e) => {
        if (d.pointer.lock.status !== "locked") return;
        pointerMove(world, {
            x: d.pointer.x,
            y: d.pointer.y,
            deltaX: e.movementX,
            deltaY: e.movementY,
            hover: true,
            canvasIndex: d.pointerCanvasIndex,
        });
    };
    const moveActive = (e: PointerEvent) => {
        pointerMove(world, {
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
    a.pointerUp = (e) => {
        if (d.touchPoints.has(e.pointerId)) {
            const at = touchPosition(a, e.target, e);
            touchPoint(world, e.pointerId, at.x, at.y, false);
            followTouch(a, d, e.pointerId);
            return;
        }
        if (e.pointerId !== a.activePointerId) return;
        if (d.pointer.lock.status !== "locked") moveActive(e);
        pointerButtons(world, gatedButtons(d, e.buttons));
        if (e.button === a.activeButton) {
            const canvas = a.activeCanvas;
            if (canvas?.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
            releaseCapture(a);
        }
    };
    a.pointerCancel = (e) => {
        if (d.touchPoints.has(e.pointerId)) {
            if (e.pointerId === d.primaryTouch) cancelTouchPress(d);
            liftTouch(d, e.pointerId);
            followTouch(a, d, e.pointerId);
            return;
        }
        if (e.pointerId !== a.activePointerId) return;
        cancelButtons(d);
        releaseCapture(a);
    };
    a.pointerMove = (e) => {
        if (d.touchPoints.has(e.pointerId)) {
            const at = touchPosition(a, e.target, e);
            touchPoint(world, e.pointerId, at.x, at.y);
            if (e.pointerId === a.activePointerId) e.preventDefault();
            return;
        }
        if (e.pointerId !== a.activePointerId) return;
        if (d.pointer.lock.status !== "locked") {
            e.preventDefault();
            moveActive(e);
        }
        pointerButtons(world, gatedButtons(d, e.buttons));
    };
    a.wheel = (e) => {
        if (!a.canvases.has(e.target as HTMLCanvasElement)) return;
        pointerWheel(world, e.deltaY);
        e.preventDefault();
    };
    a.contextMenu = (e) => {
        if (a.canvases.has(e.target as HTMLCanvasElement)) e.preventDefault();
    };
}

function listen(
    a: BrowserAdapter,
    target: EventTarget,
    type: string,
    listener: EventListener,
    options?: AddEventListenerOptions | boolean,
): void {
    target.addEventListener(type, listener, options);
    a.listeners.push({ target, type, listener, options });
}

function attachCanvas(a: BrowserAdapter, canvas: HTMLCanvasElement): void {
    listen(a, canvas, "pointerdown", a.pointerDown as EventListener);
    listen(a, canvas, "pointermove", a.pointerHover as EventListener);
    listen(a, canvas, "pointerenter", a.pointerEnter as EventListener);
    listen(a, canvas, "pointerleave", a.pointerLeave as EventListener);
    listen(a, canvas, "wheel", a.wheel as EventListener, { passive: false });
    listen(a, canvas, "contextmenu", a.contextMenu as EventListener);
    listen(a, canvas, "click", a.canvasClick as EventListener);
}

function attachGlobal(a: BrowserAdapter): void {
    const { document, window } = a.host;
    listen(a, window, "keydown", a.keyDown as EventListener);
    listen(a, window, "keyup", a.keyUp as EventListener);
    listen(a, window, "pointerdown", a.windowPointerDown as EventListener);
    listen(a, window, "pointerup", a.pointerUp as EventListener);
    listen(a, window, "pointercancel", a.pointerCancel as EventListener);
    listen(a, window, "pointermove", a.pointerMove as EventListener);
    listen(a, window, "blur", a.windowBlur as EventListener);
    listen(a, document, "visibilitychange", a.visibilityChange as EventListener);
    listen(a, document, "pointerlockchange", a.pointerLockChange as EventListener);
    listen(a, document, "pointerlockerror", a.pointerLockError as EventListener);
    listen(a, document, "mousemove", a.lockMove as EventListener);
}

function disposeAdapter(a: BrowserAdapter): void {
    if (a.disposed) return;
    a.disposed = true;
    const captureCanvas = a.activeCanvas;
    if (captureCanvas !== null && a.activePointerId !== null) {
        try {
            if (captureCanvas.hasPointerCapture(a.activePointerId))
                captureCanvas.releasePointerCapture(a.activePointerId);
        } catch {}
    }
    for (const canvas of a.ownedCanvases) {
        if (a.host.document.pointerLockElement === canvas) {
            try {
                a.host.releasePointerLock(canvas);
            } catch {}
        }
        releaseOwnedLock(a, canvas);
    }
    for (let i = a.listeners.length - 1; i >= 0; i--) {
        const listener = a.listeners[i];
        try {
            listener.target.removeEventListener(listener.type, listener.listener, listener.options);
        } catch {}
    }
    a.listeners.length = 0;
    for (const [canvas, touchAction] of a.canvasStyles) canvas.style.touchAction = touchAction;
    a.canvasStyles.clear();
    a.canvases.clear();
    a.pendingLocks.clear();
    a.ownedCanvases.clear();
    releaseCapture(a);
    a.lockCanvas = null;
}

function setup(world: World, canvasElements: readonly HTMLCanvasElement[], host: InputHost): void {
    const d = record(world);
    const a = adapter(world, host);
    if (a.canvases.size > 0) return;
    world.onDispose(() => disposeAdapter(a));
    try {
        for (let i = 0; i < canvasElements.length; i++) {
            const canvas = canvasElements[i];
            a.canvases.set(canvas, i);
            a.canvasStyles.set(canvas, canvas.style.touchAction);
            canvas.style.touchAction = "none";
        }
        if (a.canvases.size === 0) return;
        createHandlers(a, d, world);
        attachGlobal(a);
        for (const canvas of a.canvases.keys()) attachCanvas(a, canvas);
        const supported = [...a.canvases.keys()].some((canvas) => host.supportsPointerLock(canvas));
        setPointerLock(
            d,
            supported ? "unlocked" : "unsupported",
            supported ? null : "canvas has no requestPointerLock",
        );
        a.canvasFocused = true;
    } catch (error) {
        disposeAdapter(a);
        throw error;
    }
}

/** Request pointer lock through the composed browser adapter from an engagement gesture. */
export function requestPointerLock(world: World): void {
    const d = record(world);
    if (!d.enabled) return;
    if (d.pointer.lock.status === "unsupported") return;
    const a = adapters.get(world);
    if (!a || a.disposed) return;
    const canvas =
        a.activeCanvas && a.host.supportsPointerLock(a.activeCanvas)
            ? a.activeCanvas
            : ([...a.canvases.entries()].find(
                  ([candidate, index]) =>
                      index === d.focused && a.host.supportsPointerLock(candidate),
              )?.[0] ??
              [...a.canvases.keys()].find((candidate) => a.host.supportsPointerLock(candidate)));
    if (!canvas) {
        setPointerLock(d, "unsupported", "canvas has no requestPointerLock");
        return;
    }
    const owner = lockOwners.get(canvas);
    if (owner && owner !== a) return;
    ownLock(a, canvas);
    a.lockCanvas = canvas;
    try {
        const result = a.host.requestPointerLock(canvas);
        if (result && typeof result.then === "function") {
            a.pendingLocks.add(canvas);
            result.then(
                () => finishPointerLockRequest(world, a, canvas),
                (error: unknown) => {
                    if (!a.disposed) {
                        pointerLockChanged(
                            world,
                            false,
                            error instanceof Error ? error.message : String(error),
                        );
                    }
                    finishPointerLockRequest(world, a, canvas, true);
                },
            );
        }
    } catch (error) {
        if (!a.disposed)
            pointerLockChanged(
                world,
                false,
                error instanceof Error ? error.message : String(error),
            );
        finishPointerLockRequest(world, a, canvas, true);
    }
}

function finishPointerLockRequest(
    world: World,
    a: BrowserAdapter,
    canvas: HTMLCanvasElement,
    rejected = false,
): void {
    a.pendingLocks.delete(canvas);
    if (rejected && !a.disposed && lockOwners.get(canvas) === a) {
        releaseOwnedLock(a, canvas);
    }
    if (a.disposed) {
        const owner = lockOwners.get(canvas);
        if (owner === a || owner === undefined) {
            if (a.host.document.pointerLockElement === canvas) {
                try {
                    a.host.releasePointerLock(canvas);
                } catch {}
            }
            if (owner === a) releaseOwnedLock(a, canvas);
        }
        return;
    }
    // A settled request never grants a lock fact by itself; pointerlockchange is the host report.
    void world;
}

/** Release this adapter's lock, if it owns the currently locked canvas. */
export function releasePointerLock(world: World): void {
    const a = adapters.get(world);
    if (!a || a.disposed) return;
    const element = a.host.document.pointerLockElement as HTMLCanvasElement | null;
    if (!element || !a.ownedCanvases.has(element) || lockOwners.get(element) !== a) return;
    a.host.releasePointerLock(element);
}

/** Suspend or resume one World's device producers. Suspension releases held inputs with normal edges. */
export function setInputEnabled(world: World, on: boolean): void {
    const d = record(world);
    d.enabled = on;
    if (!on) {
        releaseAll(world, d);
        d.keys.pressed.clear();
        dropPresses(d);
        d.pointer.deltaX = 0;
        d.pointer.deltaY = 0;
        d.pointer.scroll = 0;
    }
}

/** Set the pointer-button gate on one World. */
export function requirePointerLock(world: World, on: boolean): void {
    record(world).requireLock = on;
}

const InputSystem: System = {
    name: "input",
    group: "simulation",
    setup(world: World) {
        // Producers may be absent even when a DOM is present.
        record(world);
    },
    update() {},
};

function defaultInputHost(): InputHost | null {
    if (typeof document === "undefined" || typeof window === "undefined") return null;
    return {
        window,
        document,
        queryCanvases: () => Array.from(document.querySelectorAll("canvas")),
        supportsPointerLock: (canvas) => typeof canvas.requestPointerLock === "function",
        requestPointerLock: (canvas) => canvas.requestPointerLock(),
        releasePointerLock: (canvas) => {
            if (
                document.pointerLockElement === canvas &&
                typeof document.exitPointerLock === "function"
            )
                document.exitPointerLock();
        },
    };
}

/** Create a browser input producer with an explicit host, or browser globals resolved at setup when omitted. */
export function createBrowserInputPlugin(host?: InputHost): Plugin {
    const browserHost = host;
    const browserSystem: System = {
        name: "browser",
        group: "simulation",
        setup(world: World) {
            const currentHost = browserHost ?? defaultInputHost();
            if (!currentHost) return;
            const elements = currentHost.queryCanvases();
            if (elements.length > 0) setup(world, elements, currentHost);
        },
        update(world: World) {
            const input = record(world);
            if (!input.enabled && input.pointer.lock.status === "locked") releasePointerLock(world);
        },
    };
    return {
        name: "BrowserInput",
        dependencies: [InputPlugin],
        systems: [browserSystem],
    };
}

const ResetFrameInputSystem: System = {
    name: "frame-reset",
    group: "draw",
    last: true,
    update(world: World) {
        const d = record(world);
        if (d.keys.pressed.size !== 0) d.keys.pressed.clear();
        if (d.keys.released.size !== 0) d.keys.released.clear();
        if (d.buttonEvents.length !== 0) d.buttonEvents.length = 0;
        d.pointer.deltaX = 0;
        d.pointer.deltaY = 0;
        d.pointer.scroll = 0;
        d.touch.pinchDelta = 0;
        d.touch.deltaX = 0;
        d.touch.deltaY = 0;
    },
};

/** Owns plain device facts, transitions and the frame boundary. */
export const InputPlugin: Plugin = {
    name: "Input",
    systems: [InputSystem, ResetFrameInputSystem],
};

/** Optional browser producer, separate from {@link InputPlugin}. */
export const BrowserInputPlugin: Plugin = createBrowserInputPlugin();
