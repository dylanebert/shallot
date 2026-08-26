import type { Plugin, State, System } from "../../engine";

/**
 * live mouse state, read through {@link Inputs}.mouse. Positions and sizes are CSS pixels; the deltas
 * accumulate over the frame and reset at frame end.
 * @expand
 */
export interface Mouse {
    /** horizontal pointer movement since the last frame, in CSS pixels (drag/look delta) */
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
    /** pointer is over a bound canvas, or a drag is active (hover holds true for the whole drag).
     *  A drag released off-canvas leaves `hover` true: `releaseCapture` clears `activePointerId` without
     *  re-evaluating hover, and no further `pointerleave` fires once the pointer has already left the canvas */
    hover: boolean;
    /** pointer x within the focused canvas, CSS pixels from the left edge */
    x: number;
    /** pointer y within the focused canvas, CSS pixels from the top edge */
    y: number;
    /** focused canvas width in CSS pixels; divide `x` by it for a 0–1 coordinate */
    canvasWidth: number;
    /** focused canvas height in CSS pixels */
    canvasHeight: number;
}

/**
 * live multi-touch state, read through {@link Inputs}.touch. Independent of {@link Mouse} — a touch
 * gesture never synthesizes mouse buttons or scroll, so a tool reading `mouse.scroll`/`mouse.middle`
 * can't misfire on a pinch or two-finger drag. Deltas accumulate over the frame and reset at frame end.
 * @expand
 */
export interface Touch {
    /** number of touch pointers currently held on a bound canvas */
    count: number;
    /** two-finger distance change since the last frame, in CSS pixels — positive spreads (zoom in),
     *  negative pinches together (zoom out) */
    pinchDelta: number;
    /** two-finger centroid horizontal movement since the last frame, in CSS pixels */
    deltaX: number;
    /** two-finger centroid vertical movement since the last frame, in CSS pixels */
    deltaY: number;
}

/**
 * the input singleton: query keyboard, mouse, and touch state from any system. Key codes are
 * `KeyboardEvent.code` values (`"KeyW"`, `"Space"`, `"ShiftLeft"`). While input is suspended
 * ({@link setInputEnabled}) every read reports neutral.
 * @expand
 * @example
 * if (Inputs.isKeyDown("KeyW")) moveForward();
 * if (Inputs.isKeyPressed("Space")) jump();
 * if (Inputs.mouse.left) fire();
 * if (Inputs.touch.count === 2) zoom(Inputs.touch.pinchDelta);
 */
export interface Inputs {
    /** current mouse state: buttons, canvas-relative position, per-frame deltas */
    readonly mouse: Readonly<Mouse>;
    /** current multi-touch state: pointer count, pinch delta, two-finger centroid deltas */
    readonly touch: Readonly<Touch>;
    /** document-order index of the canvas holding input focus (the Nth `<canvas>` `InputSystem` bound),
     *  or -1 when none is focused — not an ECS entity id */
    readonly focused: number;
    /** whether a key is currently held down */
    isKeyDown(code: string): boolean;
    /** whether a key went down this frame: the press edge, true for one frame */
    isKeyPressed(code: string): boolean;
    /** whether a key came up this frame: the release edge, true for one frame */
    isKeyReleased(code: string): boolean;
    /** whether a key's last press was within the last `seconds`: an input buffer for jump/coyote timing */
    isKeyPressedWithin(code: string, seconds: number): boolean;
}

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

const DEFAULT_TOUCH: Touch = {
    count: 0,
    pinchDelta: 0,
    deltaX: 0,
    deltaY: 0,
};

// the global input gate. While off, every read reports neutral (no keys, no buttons, no deltas) and the
// keydown handler stops accumulating, so a menu or cutscene suspends every input consumer with one flag. A
// pointer-lock controller (the Player) watches it to release the lock too. Reset to true on a bind that
// attaches at least one canvas — a canvas-less rebind inherits the prior State's gate.
let enabled = true;

export const Inputs: Inputs = {
    get mouse(): Readonly<Mouse> {
        return enabled ? (inputState?.mouse ?? DEFAULT_MOUSE) : DEFAULT_MOUSE;
    },
    get touch(): Readonly<Touch> {
        return enabled ? (inputState?.touch ?? DEFAULT_TOUCH) : DEFAULT_TOUCH;
    },
    get focused(): number {
        return enabled ? (inputState?.focused ?? -1) : -1;
    },
    isKeyDown(code: string): boolean {
        return enabled ? (inputState?.keys.has(code) ?? false) : false;
    },
    isKeyPressed(code: string): boolean {
        return enabled ? (inputState?.keysPressed.has(code) ?? false) : false;
    },
    isKeyReleased(code: string): boolean {
        return enabled ? (inputState?.keysReleased.has(code) ?? false) : false;
    },
    isKeyPressedWithin(code: string, seconds: number): boolean {
        if (!enabled) return false;
        const t = inputState?.keyPressedAt.get(code) ?? -Infinity;
        return performance.now() - t < seconds * 1000;
    },
};

/**
 * suspend or resume all input. While suspended, every {@link Inputs} read reports neutral: no keys held, no
 * mouse buttons, zero deltas, so a menu or cutscene freezes every consumer (movement, look, grab) with one
 * call, and a pointer-lock controller releases its lock. Resets to enabled on a bind that attaches at least one canvas.
 *
 * @example
 * ```
 * setInputEnabled(false); // open a pause menu — the player stops moving + looking
 * setInputEnabled(true);  // close it — control resumes
 * ```
 */
export function setInputEnabled(on: boolean): void {
    enabled = on;
    if (!on && inputState) {
        inputState.keys.clear();
        inputState.keysPressed.clear();
        inputState.keysReleased.clear();
        inputState.keyPressedAt.clear();
        clearAllButtons(inputState.mouse);
        inputState.mouse.deltaX = 0;
        inputState.mouse.deltaY = 0;
        inputState.mouse.scroll = 0;
        inputState.touchPoints.clear();
        updatePinchBaseline(inputState);
        inputState.touch.count = 0;
        inputState.touch.pinchDelta = 0;
        inputState.touch.deltaX = 0;
        inputState.touch.deltaY = 0;
    }
}

/** whether input is currently live (the inverse of a {@link setInputEnabled} suspend). */
export function inputEnabled(): boolean {
    return enabled;
}

interface InputState {
    keys: Set<string>;
    keysPressed: Set<string>;
    keysReleased: Set<string>;
    keyPressedAt: Map<string, number>;
    mouse: Mouse;
    touch: Touch;
    // per-`pointerId` cache of the last known position of every touch pointer currently held on a
    // bound canvas (MDN multi-touch pattern) — independent of `activePointerId`'s single-pointer
    // mouse capture, so a second finger is tracked rather than discarded.
    touchPoints: Map<number, { x: number; y: number }>;
    // the two-finger distance/centroid baseline `pointerMove` diffs against; null whenever fewer
    // than two touch pointers are held, so a delta is never computed against a stale pair.
    pinchDistance: number | null;
    centroidX: number | null;
    centroidY: number | null;
    canvases: Map<HTMLCanvasElement, number>;
    activeCanvas: HTMLCanvasElement | null;
    focused: number;
    lastPointerX: number;
    lastPointerY: number;
    activePointerId: number | null;
    activeButton: number | null;
    requireLock: boolean;
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

let inputState: InputState | null = null;

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

// true while pointer lock is engaged on one of our canvases.
function locked(s: InputState): boolean {
    const el =
        typeof document === "undefined"
            ? null
            : (document.pointerLockElement as HTMLCanvasElement | null);
    return !!el && s.canvases.has(el);
}

// the button bitmask a press reports. with `requireLock` on (a pointer-lock controller
// like Player), buttons read 0 until the pointer is locked — so the click that engages
// the lock only focuses; subsequent clicks, made while locked, register as commands.
function gateButtons(s: InputState, buttons: number): number {
    return s.requireLock && !locked(s) ? 0 : buttons;
}

/**
 * gate mouse-button reporting on pointer lock. while on, {@link Inputs}.mouse buttons stay
 * up until the pointer is locked, so the click that engages the lock doesn't fire a command.
 * only clicks made once locked count. a pointer-lock controller (the {@link Player}) turns this
 * on in its setup; non-locked cameras (orbit) leave it off and read buttons immediately.
 */
export function requirePointerLock(on: boolean): void {
    if (inputState) inputState.requireLock = on;
}

function clearAllButtons(mouse: Mouse): void {
    mouse.left = false;
    mouse.middle = false;
    mouse.right = false;
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
    return Math.hypot(b.x - a.x, b.y - a.y);
}

// re-derives the two-finger pinch/centroid baseline from `touchPoints` whenever the held set
// changes size (a finger goes down or up) — recomputing rather than patching means a third finger
// joining or a pair reforming after a lift never diffs against a stale prior pair.
function updatePinchBaseline(s: InputState): void {
    if (s.touchPoints.size === 2) {
        const [a, b] = [...s.touchPoints.values()];
        s.pinchDistance = distance(a, b);
        s.centroidX = (a.x + b.x) / 2;
        s.centroidY = (a.y + b.y) / 2;
    } else {
        s.pinchDistance = null;
        s.centroidX = null;
        s.centroidY = null;
    }
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

// re-captures the active pointer to a remaining touch finger when the one currently captured lifts
// (or cancels) but others are still held — `pointerMove` only writes `mouse.deltaX/Y` and `mouse.x/y`
// while `e.pointerId === activePointerId` (below), and `pointerDown` never reassigns capture to an
// already-down second finger, so without this a still-held survivor's moves would early-return and
// every Mouse-reading consumer (drag rotate, a claimed cursor) would freeze for the rest of the touch
// sequence. Seeds `lastPointerX/Y` from the survivor's own cached position (not the departing
// pointer's) so its first post-transition move reports its own delta, not a jump across the two
// fingers' gap.
function recaptureTouch(s: InputState, canvas: HTMLCanvasElement): void {
    const [nextId, pos] = [...s.touchPoints.entries()][0];
    s.activePointerId = nextId;
    s.activeButton = 0;
    s.activeCanvas = canvas;
    s.lastPointerX = pos.x;
    s.lastPointerY = pos.y;
    try {
        canvas.setPointerCapture(nextId);
    } catch {}
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
        if (!enabled) return;
        const locked = document.pointerLockElement as HTMLCanvasElement | null;
        if (!s.canvasFocused && !(locked && s.canvases.has(locked))) return;
        if (!s.keys.has(e.code)) {
            s.keysPressed.add(e.code);
            s.keyPressedAt.set(e.code, performance.now());
        }
        s.keys.add(e.code);
    };

    s.keyUp = (e: KeyboardEvent) => {
        if (!enabled) return;
        const locked = document.pointerLockElement as HTMLCanvasElement | null;
        if (!s.canvasFocused && !(locked && s.canvases.has(locked))) return;
        s.keys.delete(e.code);
        s.keysReleased.add(e.code);
    };

    s.pointerDown = (e: PointerEvent) => {
        const target = e.target as HTMLCanvasElement;
        const canvasEid = s.canvases.get(target);
        if (canvasEid === undefined) return;
        window.focus();
        // a touch pointer joins the per-pointerId cache regardless of `activePointerId` — a second
        // (or further) finger is tracked for pinch/centroid, never discarded.
        if (e.pointerType === "touch") {
            s.touchPoints.set(e.pointerId, { x: e.clientX, y: e.clientY });
            s.touch.count = s.touchPoints.size;
            updatePinchBaseline(s);
        }
        // ignore multi-touch (different pointerId on the same canvas) so a
        // second pointer's buttons can't masquerade as the captured one's.
        if (s.activePointerId === null || s.activePointerId === e.pointerId) {
            syncButtons(s.mouse, gateButtons(s, e.buttons));
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
            for (const code of s.keys) s.keysReleased.add(code);
            s.keys.clear();
            s.keysPressed.clear();
        }
    };

    s.windowBlur = () => {
        s.canvasFocused = false;
        for (const code of s.keys) s.keysReleased.add(code);
        s.keys.clear();
        s.keysPressed.clear();
    };

    s.pointerUp = (e: PointerEvent) => {
        const wasTouch = s.touchPoints.delete(e.pointerId);
        if (wasTouch) {
            s.touch.count = s.touchPoints.size;
            updatePinchBaseline(s);
        }
        if (e.pointerId !== s.activePointerId) return;
        syncButtons(s.mouse, gateButtons(s, e.buttons));
        // releasing the capturing button releases the pointer; secondary
        // buttons just toggle their state and leave the drag intact.
        if (e.button === s.activeButton) {
            // release only a capture actually held — pointerDown's capture is
            // best-effort (a synthetic pointer can't be captured), and releasing
            // an unheld pointer throws NotFoundError.
            const canvas = s.activeCanvas;
            if (canvas?.hasPointerCapture(e.pointerId)) {
                canvas.releasePointerCapture(e.pointerId);
            }
            // a still-held second finger keeps the gesture alive — hand capture to it rather than
            // dropping to no active pointer, or its moves would freeze Mouse for the rest of the touch.
            if (wasTouch && canvas && s.touchPoints.size > 0) {
                recaptureTouch(s, canvas);
            } else {
                releaseCapture(s);
            }
        }
    };

    s.pointerCancel = (e: PointerEvent) => {
        const wasTouch = s.touchPoints.delete(e.pointerId);
        if (wasTouch) {
            s.touch.count = s.touchPoints.size;
            updatePinchBaseline(s);
        }
        if (e.pointerId !== s.activePointerId) return;
        clearAllButtons(s.mouse);
        const canvas = s.activeCanvas;
        if (wasTouch && canvas && s.touchPoints.size > 0) {
            recaptureTouch(s, canvas);
        } else {
            releaseCapture(s);
        }
    };

    s.pointerMove = (e: PointerEvent) => {
        // hand-computed distance/center from the per-pointerId cache (MDN multi-touch pattern) —
        // diffed against the baseline from the last time the held set changed size, so this only
        // ever measures the *frame's* movement, never a jump from a stale prior pair.
        if (s.touchPoints.has(e.pointerId)) {
            s.touchPoints.set(e.pointerId, { x: e.clientX, y: e.clientY });
            if (s.touchPoints.size === 2) {
                const [a, b] = [...s.touchPoints.values()];
                const dist = distance(a, b);
                const cx = (a.x + b.x) / 2;
                const cy = (a.y + b.y) / 2;
                if (s.pinchDistance !== null) s.touch.pinchDelta += dist - s.pinchDistance;
                if (s.centroidX !== null && s.centroidY !== null) {
                    s.touch.deltaX += cx - s.centroidX;
                    s.touch.deltaY += cy - s.centroidY;
                }
                s.pinchDistance = dist;
                s.centroidX = cx;
                s.centroidY = cy;
            }
        }
        if (e.pointerId !== s.activePointerId) return;
        // sync buttons during drag too — e.g. catches a right-press that
        // happened mid-drag if the per-button pointerdown was suppressed
        // and the user moved before releasing.
        syncButtons(s.mouse, gateButtons(s, e.buttons));
        // canceling pointermove suppresses the compatibility mousemove (Pointer Events spec) — and a
        // pointer-lock controller (Player) reads the look off document `mousemove`. With a button held
        // (activePointerId set, so we reach here), canceling would silence mouse-look. Under lock there's
        // nothing to prevent (cursor hidden), so skip it; non-locked drags still cancel (selection/scroll).
        if (!(s.requireLock && locked(s))) e.preventDefault();
        s.mouse.deltaX += e.clientX - s.lastPointerX;
        s.mouse.deltaY += e.clientY - s.lastPointerY;
        s.lastPointerX = e.clientX;
        s.lastPointerY = e.clientY;
        // While a pointer is active, keep mouse.x/y tracking even when the pointer's target is
        // not the canvas. `pointerLeave` already holds `hover` true while `activePointerId !== null`
        // (the engine's stated intent: a drag survives leaving the canvas), but `mouse.x`/`y` were
        // written only by the canvas-scoped `pointerHover` handler, which early-returns unless
        // `e.target` is a registered canvas. So a fast pointer off the canvas froze the coordinates
        // while `hover` stayed true — `cursorRay` returned a ray that no longer moved, and the handle
        // stopped under a live cursor. Using the active canvas's rect here makes the data follow the
        // intent the leave rule already declared.
        if (s.activeCanvas) {
            const rect = s.activeCanvas.getBoundingClientRect();
            s.mouse.x = e.clientX - rect.left;
            s.mouse.y = e.clientY - rect.top;
            s.mouse.canvasWidth = rect.width;
            s.mouse.canvasHeight = rect.height;
        }
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

// listeners bind to `state.signal`, so `state.dispose()` (which aborts it) detaches every one with no
// removal code — the per-build teardown rides the State's lifetime.
function attachCanvas(s: InputState, canvas: HTMLCanvasElement, signal: AbortSignal): void {
    canvas.addEventListener("pointerdown", s.pointerDown, { signal });
    canvas.addEventListener("pointermove", s.pointerHover, { signal });
    canvas.addEventListener("pointerenter", s.pointerEnter, { signal });
    canvas.addEventListener("pointerleave", s.pointerLeave, { signal });
    canvas.addEventListener("wheel", s.wheel, { passive: false, signal });
    canvas.addEventListener("contextmenu", s.contextMenu, { signal });
}

function attachGlobal(s: InputState, signal: AbortSignal): void {
    window.addEventListener("keydown", s.keyDown, { signal });
    window.addEventListener("keyup", s.keyUp, { signal });
    window.addEventListener("pointerdown", s.windowPointerDown, { signal });
    window.addEventListener("pointerup", s.pointerUp, { signal });
    window.addEventListener("pointercancel", s.pointerCancel, { signal });
    window.addEventListener("pointermove", s.pointerMove, { signal });
    window.addEventListener("blur", s.windowBlur, { signal });
}

// `canvasElements` is document order — `focused`'s index into it, never an ECS entity id (Input has no
// view/eid substrate of its own to draw a real one from; see the `Inputs.focused` JSDoc).
function setup(state: State, canvasElements: HTMLCanvasElement[]): void {
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

    const touch: Touch = {
        count: 0,
        pinchDelta: 0,
        deltaX: 0,
        deltaY: 0,
    };

    const canvases = new Map<HTMLCanvasElement, number>();
    for (let i = 0; i < canvasElements.length; i++) {
        const element = canvasElements[i];
        canvases.set(element, i);
        element.style.touchAction = "none";
    }

    if (canvases.size === 0) return;

    const s: InputState = {
        keys: new Set(),
        keysPressed: new Set(),
        keysReleased: new Set(),
        keyPressedAt: new Map(),
        mouse,
        touch,
        touchPoints: new Map(),
        pinchDistance: null,
        centroidX: null,
        centroidY: null,
        canvases,
        activeCanvas: null,
        focused: 0,
        lastPointerX: 0,
        lastPointerY: 0,
        activePointerId: null,
        activeButton: null,
        requireLock: false,
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

    createHandlers(s);
    attachGlobal(s, state.signal);
    for (const canvas of canvases.keys()) {
        attachCanvas(s, canvas, state.signal);
    }

    enabled = true; // a fresh bind starts live — never inherit a prior State's suspended gate
    inputState = s;
    // drop the module ref when this State tears down, so a disposed app reads neutral; guarded so a
    // newer build's inputState isn't clobbered. Listener detach is the signal's job (attach* above).
    state.onDispose(() => {
        if (inputState === s) inputState = null;
    });
}

const InputSystem: System = {
    group: "simulation",
    annotations: { mode: "always" },

    setup(state: State) {
        // Bind to canvas elements in the document directly.
        // Guarded for non-DOM test environments where `document.querySelectorAll` is absent
        if (typeof document === "undefined" || typeof document.querySelectorAll !== "function") {
            return;
        }
        const elements = Array.from(document.querySelectorAll("canvas"));
        if (elements.length === 0) return;
        setup(state, elements);
    },
};

const InputResetSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,

    update() {
        const s = inputState;
        if (!s) return;
        s.keysPressed.clear();
        s.keysReleased.clear();
        s.mouse.deltaX = 0;
        s.mouse.deltaY = 0;
        s.mouse.scroll = 0;
        s.touch.pinchDelta = 0;
        s.touch.deltaX = 0;
        s.touch.deltaY = 0;
    },
};

/**
 * binds keyboard and mouse listeners to the app's canvases and populates the {@link Inputs} singleton.
 * In the default plugin set. A system reads `Inputs` without enabling anything.
 */
export const InputPlugin: Plugin = {
    name: "Input",
    systems: [InputSystem, InputResetSystem],
};
