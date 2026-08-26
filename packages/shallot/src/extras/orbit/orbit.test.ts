import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import {
    InputPlugin,
    Orbit,
    OrbitOverlayPlugin,
    OrbitPick,
    OrbitPlugin,
    State,
    Transform,
    TransformsPlugin,
} from "../..";
import { clear, register } from "../../engine/ecs/core";
import { Slab } from "../../standard/slab";
import { OrbitSmooth } from "./smooth";

// Orbit's reload-safety (the lazy OrbitSmooth add/remove not doubling across a rebuild) is covered by
// the conformance roster. This spec covers the pose contract and the lazy-init path: with no input,
// OrbitSystem drives the camera Transform from the yaw/pitch/distance pose, so the camera always sits
// `distance` from its target. Fly/orbit/pan are input-driven and exercised live (`bun bench`), not here.
//
// The spherical pose is a unit vector scaled by distance, so |pos − target| == distance regardless of
// yaw/pitch — a behavior invariant, not a re-derivation of the pose formula. Tolerance is f32 storage of
// the position lanes (~1e-6 at magnitude 10), so `toBeCloseTo(d, 4)` (5e-5) is comfortably above it.
describe("OrbitSystem", () => {
    let state: State;

    beforeEach(() => {
        clear();
        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        Slab.collect(); // allocate the Transform slab's CPU storage (no device; build() does this normally)
        attach(state, OrbitPlugin);
    });

    const distanceTo = (eid: number, cx = 0, cy = 0, cz = 0): number =>
        Math.hypot(
            Transform.pos.x.get(eid) - cx,
            Transform.pos.y.get(eid) - cy,
            Transform.pos.z.get(eid) - cz,
        );

    test("drives the camera to `distance` from the world origin", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit); // default distance 10
        state.step(1 / 60);
        expect(distanceTo(cam)).toBeCloseTo(10, 4);
    });

    test("orbits a target entity at its position", () => {
        const target = state.create();
        state.add(target, Transform);
        Transform.pos.set(target, 5, 1, -3, 0);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        Orbit.distance.set(cam, 8);
        Orbit.target.set(cam, target);
        state.step(1 / 60);
        expect(distanceTo(cam, 5, 1, -3)).toBeCloseTo(8, 4);
    });

    test("lazily initializes when Orbit is added after the first frame", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.step(1 / 60); // a frame with no Orbit — nothing to pose

        state.add(cam, Orbit);
        state.step(1 / 60); // the dynamic-add path must add the smoothing state and pose this frame
        expect(distanceTo(cam)).toBeCloseTo(10, 4);
    });
});

// biome-ignore lint/complexity/noBannedTypes: test mock tracks arbitrary DOM listeners
type Fn = Function;

class ListenerTracker {
    added: [string, Fn][] = [];
    addEventListener = (type: string, fn: Fn, _opts?: unknown) => {
        this.added.push([type, fn]);
    };
    removeEventListener = (_type: string, _fn: Fn) => {};
}

function mockCanvas(): HTMLCanvasElement {
    const tracker = new ListenerTracker();
    return {
        addEventListener: tracker.addEventListener,
        removeEventListener: tracker.removeEventListener,
        setPointerCapture() {},
        releasePointerCapture() {},
        hasPointerCapture: () => false,
        // orbit math is the subject here, not DOM geometry — a zero-rect stub keeps
        // `input/index.ts`'s pointermove handler's `getBoundingClientRect()` read satisfied
        // without asserting any layout the arms don't test.
        getBoundingClientRect: () => ({ left: 0, top: 0, width: 0, height: 0 }) as DOMRect,
        style: {} as CSSStyleDeclaration,
        tracker,
    } as unknown as HTMLCanvasElement;
}

// Mode is the held button: left orbits, right flies (PlayCanvas-style). The fly button is the one
// input-driven path with a state transition worth pinning — fly-look holds in place while the button is
// held, and a fresh press of the same button is still fly, never an orbit snap. A right-drag looks in
// place; an orbiting camera would swing along its arc. Driving it needs real input, so this block drives
// the actual InputPlugin handlers with synthetic DOM events (the input.test.ts pattern), then reads pose.
describe("OrbitSystem fly mode is the held fly button", () => {
    let state: State;
    let canvas: HTMLCanvasElement;
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
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
        } as unknown as typeof document;

        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        Slab.collect();
        attach(state, InputPlugin);
        attach(state, OrbitPlugin);
        state.step(1 / 60); // InputSystem.setup binds the DOM handlers on its first run
    });

    afterEach(() => {
        state.dispose(); // null the module-level inputState so the input-free block above stays clean
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    const onWindow = (type: string): Fn => windowTracker.added.find(([t]) => t === type)![1];
    const onCanvas = (type: string): Fn =>
        (canvas as unknown as { tracker: ListenerTracker }).tracker.added.find(
            ([t]) => t === type,
        )![1];

    const Right = 2; // flyButton default

    // hold the right (fly) button and drag two frames; returns the camera position after the drag
    const flyDrag = (cam: number): [number, number, number] => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: Right,
            buttons: 2,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
        state.step(1 / 60);
        expect(OrbitSmooth.flyActive.get(cam)).toBe(1);
        for (const clientX of [40, 80]) {
            onWindow("pointermove")({
                pointerId: 1,
                buttons: 2,
                clientX,
                clientY: 0,
                preventDefault() {},
            });
            state.step(1 / 60);
            expect(OrbitSmooth.flyActive.get(cam)).toBe(1);
        }
        return [Transform.pos.x.get(cam), Transform.pos.y.get(cam), Transform.pos.z.get(cam)];
    };

    const release = (): void => {
        onWindow("pointerup")({ pointerId: 1, button: Right, buttons: 0, preventDefault() {} });
        state.step(1 / 60);
    };

    // scroll retargets to fly speed only while flying, so hold the fly button (no drag = fixed heading)
    const holdFly = (): void => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: Right,
            buttons: 2,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
    };

    const wheel = (deltaY: number): void => {
        onCanvas("wheel")({ target: canvas, deltaY, preventDefault() {} });
    };

    test("looks in place while the fly button is held, then exits on release", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60); // pose once in orbit mode

        const before = [
            Transform.pos.x.get(cam),
            Transform.pos.y.get(cam),
            Transform.pos.z.get(cam),
        ];
        const after = flyDrag(cam);
        // no movement key: fly only looks, so the drag rotates in place — an orbit drag would have moved it
        expect(after[0]).toBeCloseTo(before[0], 4);
        expect(after[1]).toBeCloseTo(before[1], 4);
        expect(after[2]).toBeCloseTo(before[2], 4);

        release();
        expect(OrbitSmooth.flyActive.get(cam)).toBe(0); // back to orbit, center reprojected
    });

    test("a fresh press of the fly button flies again — never an orbit snap", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60);

        flyDrag(cam);
        release();
        expect(OrbitSmooth.flyActive.get(cam)).toBe(0);

        // the scenario that felt wrong before: press → drag → release → press again. The second gesture
        // must be fly too, because the button is the mode — re-pressing right can't become orbit.
        const start = [
            Transform.pos.x.get(cam),
            Transform.pos.y.get(cam),
            Transform.pos.z.get(cam),
        ];
        const again = flyDrag(cam);
        expect(again[0]).toBeCloseTo(start[0], 4);
        expect(again[1]).toBeCloseTo(start[1], 4);
        expect(again[2]).toBeCloseTo(start[2], 4);
    });

    test("bare WASD/QE never engages fly — a gameplay scene keeps the movement keys", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60); // orbit pose

        const before = [
            Transform.pos.x.get(cam),
            Transform.pos.y.get(cam),
            Transform.pos.z.get(cam),
        ];
        // press a movement key with NO fly button held: hold-to-fly means this must not fly, so the camera
        // stays put and the key is free for gameplay (a car/character reads it) — not auto-fly-on-WASD.
        onWindow("keydown")({ code: "KeyW" });
        state.step(1 / 60);

        expect(OrbitSmooth.flyActive.get(cam)).toBe(0); // never entered fly
        expect(Transform.pos.x.get(cam)).toBeCloseTo(before[0], 6);
        expect(Transform.pos.y.get(cam)).toBeCloseTo(before[1], 6);
        expect(Transform.pos.z.get(cam)).toBeCloseTo(before[2], 6);
    });

    test("move is speed-normalized — a diagonal isn't faster than a single axis", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60); // orbit pose

        const speed = Orbit.flySpeed.get(cam) / 60; // flySpeed * dt
        const moved = (p: number[]): number =>
            Math.hypot(
                Transform.pos.x.get(cam) - p[0],
                Transform.pos.y.get(cam) - p[1],
                Transform.pos.z.get(cam) - p[2],
            );
        const pos = (): number[] => [
            Transform.pos.x.get(cam),
            Transform.pos.y.get(cam),
            Transform.pos.z.get(cam),
        ];

        // hold the fly button so WASD/QE move; no drag, so the heading is fixed
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: Right,
            buttons: 2,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });

        onWindow("keydown")({ code: "KeyW" }); // forward only
        let from = pos();
        state.step(1 / 60);
        expect(moved(from)).toBeCloseTo(speed, 4);

        onWindow("keydown")({ code: "KeyE" }); // + up: a diagonal, √2× faster without normalization
        from = pos();
        state.step(1 / 60);
        expect(moved(from)).toBeCloseTo(speed, 4);
    });

    // scroll up (deltaY < 0) speeds up, down slows down, multiplicatively (Unity's scene-view accelerator).
    // the sign of -scroll in the exp is load-bearing — up must mean faster.
    test("scroll while flying adjusts flySpeed multiplicatively", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60);

        holdFly();
        state.step(1 / 60);
        expect(OrbitSmooth.flyActive.get(cam)).toBe(1);

        wheel(-100); // one notch up → ×1.15 (FlyScrollRate = ln(1.15)/100)
        state.step(1 / 60);
        expect(Orbit.flySpeed.get(cam)).toBeCloseTo(5 * 1.15, 4);

        wheel(100); // one notch down → ÷1.15, back to the default
        state.step(1 / 60);
        expect(Orbit.flySpeed.get(cam)).toBeCloseTo(5, 4);
    });

    test("scroll-adjusted flySpeed clamps to flyMin/flyMax", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        Orbit.flyMax.set(cam, 6);
        state.step(1 / 60);

        holdFly();
        state.step(1 / 60);
        wheel(-1000); // far past flyMax in one event
        state.step(1 / 60);
        expect(Orbit.flySpeed.get(cam)).toBeCloseTo(6, 4);
    });

    test("scroll while not flying still zooms, leaving flySpeed untouched", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit); // default distance 10, flySpeed 5
        state.step(1 / 60);
        expect(OrbitSmooth.flyActive.get(cam)).toBe(0);

        wheel(100); // no fly button held → the zoom path runs, not the speed path
        state.step(1 / 60);
        expect(Orbit.distance.get(cam)).not.toBeCloseTo(10, 4);
        expect(Orbit.flySpeed.get(cam)).toBe(5);
    });

    // shift multiplies the per-frame move by flyBoost but never writes the stored base speed
    test("shift boosts fly speed transiently — stored flySpeed unchanged", () => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit);
        state.step(1 / 60);

        const boosted = (Orbit.flySpeed.get(cam) * Orbit.flyBoost.get(cam)) / 60;
        const pos = (): number[] => [
            Transform.pos.x.get(cam),
            Transform.pos.y.get(cam),
            Transform.pos.z.get(cam),
        ];
        const moved = (p: number[]): number =>
            Math.hypot(
                Transform.pos.x.get(cam) - p[0],
                Transform.pos.y.get(cam) - p[1],
                Transform.pos.z.get(cam) - p[2],
            );

        holdFly();
        onWindow("keydown")({ code: "ShiftLeft" });
        onWindow("keydown")({ code: "KeyW" }); // forward, with boost
        const from = pos();
        state.step(1 / 60);
        expect(moved(from)).toBeCloseTo(boosted, 4);
        expect(Orbit.flySpeed.get(cam)).toBe(5);
    });
});

// Contextual left-click (PlayCanvas-style): the orbit button (left by default) orbits over empty space but
// yields to an interaction when a registered OrbitPick.claim owns the press. The claim is consulted only at
// the button's down-edge and latches for the whole drag, so a mid-drag claim change can't flip an in-flight
// orbit. Driven through the real InputPlugin handlers with synthetic DOM events, the fly-mode block's shape.
describe("OrbitSystem contextual claim (left-click partition)", () => {
    let state: State;
    let canvas: HTMLCanvasElement;
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
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
        } as unknown as typeof document;

        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        Slab.collect();
        attach(state, InputPlugin);
        attach(state, OrbitPlugin);
        state.step(1 / 60); // InputSystem.setup binds the DOM handlers on its first run
    });

    afterEach(() => {
        OrbitPick.claim = undefined; // module-level singleton — clear so it can't leak across tests
        state.dispose();
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    const onWindow = (type: string): Fn => windowTracker.added.find(([t]) => t === type)![1];
    const onCanvas = (type: string): Fn =>
        (canvas as unknown as { tracker: ListenerTracker }).tracker.added.find(
            ([t]) => t === type,
        )![1];

    // press / drag / release the LEFT (default orbit) button; the buttons bitmask for left is 1
    const leftDown = (): void => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId: 1,
            button: 0,
            buttons: 1,
            clientX: 0,
            clientY: 0,
            preventDefault() {},
        });
    };
    const drag = (clientX: number): void => {
        onWindow("pointermove")({
            pointerId: 1,
            buttons: 1,
            clientX,
            clientY: 0,
            preventDefault() {},
        });
    };
    const leftUp = (): void => {
        onWindow("pointerup")({ pointerId: 1, button: 0, buttons: 0, preventDefault() {} });
    };

    const makeCam = (): number => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit); // default orbitButton 0 (left), yaw Math.PI/6
        state.step(1 / 60); // pose once
        return cam;
    };

    test("an unclaimed left-drag orbits — yaw changes", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);
        // no OrbitPick.claim registered: the press is unclaimed, so left orbits as usual
        leftDown();
        drag(40);
        state.step(1 / 60);
        expect(Orbit.yaw.get(cam)).not.toBeCloseTo(yaw0, 6);
    });

    test("a claimed left-drag never orbits, even if the claim flips false mid-drag", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);
        let claimed = true;
        let calls = 0;
        OrbitPick.claim = () => {
            calls++;
            return claimed;
        };

        leftDown();
        drag(40);
        state.step(1 / 60); // down-edge claims (true) → suppressed, the drag is ignored
        expect(Orbit.yaw.get(cam)).toBeCloseTo(yaw0, 6);
        expect(calls).toBe(1); // consulted exactly once, at the down-edge

        // the latch holds the whole drag: even though the claim would now pass the press through, the
        // in-flight orbit stays suppressed — no mid-drag flip.
        claimed = false;
        drag(80);
        state.step(1 / 60);
        expect(Orbit.yaw.get(cam)).toBeCloseTo(yaw0, 6);
        expect(calls).toBe(1); // not re-consulted mid-drag
    });

    test("release then re-press orbits again once the claim returns false", () => {
        const cam = makeCam();
        let claimed = true;
        OrbitPick.claim = () => claimed;

        // first press is claimed → suppressed, yaw unchanged
        const yaw0 = Orbit.yaw.get(cam);
        leftDown();
        drag(40);
        state.step(1 / 60);
        expect(Orbit.yaw.get(cam)).toBeCloseTo(yaw0, 6);

        leftUp();
        state.step(1 / 60); // release resets the latch to idle

        // a fresh press with the claim now false is unclaimed → orbits
        claimed = false;
        leftDown();
        drag(40);
        state.step(1 / 60);
        expect(Orbit.yaw.get(cam)).not.toBeCloseTo(yaw0, 6);
    });
});

// Touch gestures (three.js OrbitControls' / Babylon ArcRotateCameraPointersInput's touch map): one
// finger rotates, a two-finger drag pans, a two-finger pinch zooms. Touch overrides the mouse-button
// read while any finger is down rather than adding to it — the first finger's continued capture
// synthesizes `mouse.left` true for the whole gesture (`standard/input`'s pointer-capture path), so
// reading that bit during a two-finger pinch/pan would incidentally orbit alongside the intended
// gesture. `OrbitPick.claim` suppression must survive on the one-finger (rotate) path too, since a
// road's handle-drag claim doesn't know whether the press came from a mouse or a finger. Driven
// through the real InputPlugin handlers with synthetic touch PointerEvents (`input.test.ts`'s pattern).
describe("OrbitSystem touch gestures", () => {
    let state: State;
    let canvas: HTMLCanvasElement;
    let windowTracker: ListenerTracker;
    let savedWindow: typeof globalThis.window;
    let savedDocument: typeof globalThis.document;

    // a real (non-zero) rect: unlike the rotate-only suites above, pan and pinch project a screen-space
    // delta into world units via `canvasHeight` (`worldPerPixel`), so a zero-rect stub would divide by
    // zero here.
    function mockSizedCanvas(width: number, height: number): HTMLCanvasElement {
        const tracker = new ListenerTracker();
        return {
            addEventListener: tracker.addEventListener,
            removeEventListener: tracker.removeEventListener,
            setPointerCapture() {},
            releasePointerCapture() {},
            hasPointerCapture: () => false,
            getBoundingClientRect: () => ({ left: 0, top: 0, width, height }) as DOMRect,
            style: {} as CSSStyleDeclaration,
            tracker,
        } as unknown as HTMLCanvasElement;
    }

    beforeEach(() => {
        clear();
        windowTracker = new ListenerTracker();
        (windowTracker as unknown as { focus: () => void }).focus = () => {};
        savedWindow = globalThis.window;
        savedDocument = globalThis.document;
        globalThis.window = windowTracker as unknown as typeof window;
        canvas = mockSizedCanvas(800, 600);
        globalThis.document = {
            pointerLockElement: null,
            querySelectorAll: (sel: string) => (sel === "canvas" ? [canvas] : []),
        } as unknown as typeof document;

        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        for (const [n, c] of Object.entries(InputPlugin.components ?? {}))
            register(n, c, InputPlugin.traits?.[n]);
        Slab.collect();
        attach(state, InputPlugin);
        attach(state, OrbitPlugin);
        state.step(1 / 60); // InputSystem.setup binds the DOM handlers on its first run
    });

    afterEach(() => {
        OrbitPick.claim = undefined;
        state.dispose();
        globalThis.window = savedWindow;
        globalThis.document = savedDocument;
    });

    const onWindow = (type: string): Fn => windowTracker.added.find(([t]) => t === type)![1];
    const onCanvas = (type: string): Fn =>
        (canvas as unknown as { tracker: ListenerTracker }).tracker.added.find(
            ([t]) => t === type,
        )![1];

    const touchDown = (pointerId: number, clientX: number, clientY: number): void => {
        onCanvas("pointerdown")({
            target: canvas,
            pointerId,
            pointerType: "touch",
            button: 0,
            buttons: 1,
            clientX,
            clientY,
            preventDefault() {},
        });
    };

    const touchMove = (pointerId: number, clientX: number, clientY: number): void => {
        onWindow("pointermove")({
            pointerId,
            pointerType: "touch",
            buttons: 1,
            clientX,
            clientY,
            preventDefault() {},
        });
    };

    const touchUp = (pointerId: number): void => {
        onWindow("pointerup")({
            pointerId,
            pointerType: "touch",
            button: 0,
            buttons: 0,
            preventDefault() {},
        });
    };

    const makeCam = (): number => {
        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Orbit); // default distance 10, yaw Math.PI/6
        state.step(1 / 60); // pose once
        return cam;
    };

    test("a one-finger drag rotates — yaw changes", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);

        touchDown(1, 0, 0);
        touchMove(1, 40, 0);
        state.step(1 / 60);

        expect(Orbit.yaw.get(cam)).not.toBeCloseTo(yaw0, 6);
    });

    test("a two-finger drag pans — yaw stays put, pan changes", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);
        const pan0X = Orbit.pan.x.get(cam);

        touchDown(1, 100, 100);
        touchDown(2, 200, 100); // initial centroid (150, 100), distance 100

        touchMove(1, 130, 100);
        touchMove(2, 230, 100); // centroid → (180, 100), distance unchanged: a pure pan, no pinch

        state.step(1 / 60);

        expect(Orbit.yaw.get(cam)).toBeCloseTo(yaw0, 6); // two fingers never orbit
        // direction, not just magnitude: at the default yaw (30°) a rightward centroid drag projects
        // onto -rightX, so panX goes negative — the same sign a mouse pan produces for a rightward drag
        // at this pose (worldPerPixel * dragX * -cos(yawS)).
        expect(Orbit.pan.x.get(cam)).toBeLessThan(pan0X);
    });

    test("a two-finger pinch zooms — spreading fingers decreases distance", () => {
        const cam = makeCam();
        const dist0 = Orbit.distance.get(cam);

        touchDown(1, 100, 100);
        touchDown(2, 200, 100); // initial distance 100

        touchMove(1, 80, 100);
        touchMove(2, 220, 100); // distance → 140: a spread pinch, zoom in

        state.step(1 / 60);

        expect(Orbit.distance.get(cam)).toBeLessThan(dist0);
    });

    test("a claimed one-finger drag never orbits — the pick claim latches on touch too", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);
        OrbitPick.claim = () => true;

        touchDown(1, 0, 0);
        touchMove(1, 40, 0);
        state.step(1 / 60);

        expect(Orbit.yaw.get(cam)).toBeCloseTo(yaw0, 6);
    });

    // RED-FIRST WITNESS: `standard/input`'s `activePointerId` capture is never reassigned to an
    // already-down second finger, so lifting the FIRST (capturing) finger of a two-finger gesture left
    // the survivor's moves hitting `pointerMove`'s `e.pointerId !== activePointerId` early return —
    // rotation froze for the rest of the touch sequence even though `Inputs.touch.count` (1) reads as a
    // live single-finger drag. Fixed by `standard/input`'s `recaptureTouch`.
    test("ending a two-finger gesture by lifting the first finger keeps rotating from the survivor", () => {
        const cam = makeCam();
        const yaw0 = Orbit.yaw.get(cam);

        touchDown(1, 100, 100); // captures first
        touchDown(2, 200, 100); // joins the touch cache, never captures

        touchUp(1); // the capturing finger lifts; pointer 2 is still down
        state.step(1 / 60); // touchCount is now 1 — orbitHeld reads true, but nothing has moved yet

        const yawAfterLift = Orbit.yaw.get(cam);
        expect(yawAfterLift).toBeCloseTo(yaw0, 6); // the hand-off itself produces no rotation

        touchMove(2, 230, 100); // survivor's own +30px move from its cached (200, 100)
        state.step(1 / 60);

        expect(Orbit.yaw.get(cam)).not.toBeCloseTo(yawAfterLift, 6); // rotation did not freeze

        // magnitude, not just direction: Δyaw must be the survivor's own +30px move at the default
        // sensitivity (0.005), never a jump across the two fingers' initial 100px gap (which would
        // read Δyaw ≈ -0.5 instead of -0.15) — proves `lastPointerX/Y` was seeded from the survivor's
        // own cached position, not the departed pointer's.
        const dYaw = Orbit.yaw.get(cam) - yawAfterLift;
        expect(dYaw).toBeCloseTo(-30 * 0.005, 4);
    });
});

// The fly-speed overlay's module-scope runtime state (`_overlay`, `_lastSpeed`, `_shownUntil`) must be
// keyed to State/canvas lifetime per the ecs reload-safety rule. The collapse exemplar's shape is the
// contract: `mountOverlay(canvas, state)` for the disposing path, a module-scope cleanup cleared at
// top-of-warm for the re-warm/swap path, and State-derived time (no module-level accumulator). These arms
// cover the gate roster: rebuild-against-new-canvas, visibility window, fade-hold text, flying-camera
// selection, the negative-last-speed sentinel, and destroy(). DOM-mocked with the same `globalThis.document`
// stub shape the fly-mode block above uses, extended with `createElement` / `querySelector` for the overlay's
// DOM construction.
type MockEl = {
    style: Record<string, string>;
    children: MockEl[];
    removed: boolean;
    textContent: string;
    attributes: Record<string, string>;
    parentElement: MockEl | null;
    appendChild(child: MockEl): MockEl;
    append(...children: MockEl[]): void;
    setAttribute(name: string, value: string): void;
    remove(): void;
};

function mockEl(): MockEl {
    const el: MockEl = {
        style: {},
        children: [],
        removed: false,
        textContent: "",
        attributes: {},
        parentElement: null,
        appendChild(child: MockEl) {
            this.children.push(child);
            child.parentElement = this;
            return child;
        },
        append(...kids: MockEl[]) {
            for (const kid of kids) {
                this.children.push(kid);
                kid.parentElement = this;
            }
        },
        setAttribute(name: string, value: string) {
            this.attributes[name] = value;
        },
        remove() {
            this.removed = true;
            if (this.parentElement) {
                const idx = this.parentElement.children.indexOf(this);
                if (idx >= 0) this.parentElement.children.splice(idx, 1);
            }
            this.parentElement = null;
        },
    };
    return el;
}

describe("OrbitOverlayPlugin lifecycle", () => {
    let state: State;
    let canvas: MockEl;
    let canvasParent: MockEl;
    let savedDocument: typeof globalThis.document;

    beforeEach(() => {
        clear();
        canvasParent = mockEl();
        canvas = mockEl();
        canvas.parentElement = canvasParent;
        savedDocument = globalThis.document;
        globalThis.document = {
            querySelector: (sel: string) => (sel === "canvas" ? canvas : null),
            createElement: () => mockEl(),
        } as unknown as typeof document;

        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        register("OrbitSmooth", OrbitSmooth);
        Slab.collect();
        attach(state, OrbitOverlayPlugin);
    });

    afterEach(() => {
        OrbitOverlayPlugin.dispose?.(state);
        state.dispose();
        globalThis.document = savedDocument;
    });

    // traverse: canvasParent → overlay (mountOverlay) → root (data-orbit-overlay) → [speedEl, boostEl]
    function overlayRoot(): MockEl | null {
        for (const overlay of canvasParent.children) {
            for (const root of overlay.children) {
                if (root.attributes["data-orbit-overlay"] !== undefined) return root;
            }
        }
        return null;
    }

    function speedEl(): MockEl | null {
        const root = overlayRoot();
        return root ? ((root.children[0] as MockEl) ?? null) : null;
    }

    // the overlay div returned by mountOverlay (direct child of canvasParent)
    function overlayDiv(): MockEl | null {
        return canvasParent.children.length > 0 ? (canvasParent.children[0] as MockEl) : null;
    }

    function makeFlyingCam(speed = 5): number {
        const cam = state.create();
        state.add(cam, Orbit);
        state.add(cam, OrbitSmooth);
        OrbitSmooth.flyActive.set(cam, 1);
        Orbit.flySpeed.set(cam, speed);
        return cam;
    }

    // a non-flying camera (flyActive = 0)
    function makeOrbitingCam(): number {
        const cam = state.create();
        state.add(cam, Orbit);
        state.add(cam, OrbitSmooth);
        OrbitSmooth.flyActive.set(cam, 0);
        return cam;
    }

    // ── destroy() / state-owned teardown ──────────────────────────────────────

    // binds finding :108 — mountOverlay must receive `state` so state.onDispose registers the overlay's
    // removal. A direct state.dispose() (not App.dispose) fires onDispose but never the plugin dispose hook,
    // so without the state argument the overlay node leaks.
    test("a state-bound overlay auto-removes on state dispose", () => {
        makeFlyingCam();
        state.step(1 / 60); // update creates the overlay lazily
        expect(overlayRoot()).not.toBeNull();
        const div = overlayDiv()!;
        expect(div.removed).toBe(false);

        state.dispose();
        expect(div.removed).toBe(true); // onDispose fired → overlay.remove()
    });

    // ── rebuild against a new canvas ───────────────────────────────────────────

    // binds finding :71 — _overlay is module-cached forever; after a rebuild against a new canvas the
    // readout stays parented to the old canvas's overlay host. Keying the overlay to its canvas (clearing
    // on canvas change) makes update re-create it against the new canvas's parent.
    test("rebuild against a new canvas re-parents the overlay to the new canvas", () => {
        makeFlyingCam();
        state.step(1 / 60);
        const oldOverlayDiv = overlayDiv();
        expect(oldOverlayDiv).not.toBeNull();
        // the overlay is parented to canvas A's parent
        expect(canvasParent.children).toContain(oldOverlayDiv as MockEl);

        // simulate a rebuild: dispose the old State, swap in a new canvas + State
        state.dispose();
        const newCanvasParent = mockEl();
        const newCanvas = mockEl();
        newCanvas.parentElement = newCanvasParent;
        (
            globalThis.document as unknown as { querySelector: (s: string) => MockEl | null }
        ).querySelector = (sel: string) => (sel === "canvas" ? newCanvas : null);

        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        register("OrbitSmooth", OrbitSmooth);
        Slab.collect();
        attach(state, OrbitOverlayPlugin);
        OrbitOverlayPlugin.warm?.(state); // top-of-warm clears module-scope state (swap fallback)

        makeFlyingCam();
        state.step(1 / 60);

        // the new overlay is parented to the NEW canvas's parent, not the old one
        const newOverlayDiv = newCanvasParent.children[0] as MockEl | undefined;
        expect(newOverlayDiv).toBeDefined();
        expect(canvasParent.children).not.toContain(newOverlayDiv as MockEl);
        expect(newCanvasParent.children).toContain(newOverlayDiv as MockEl);
    });

    // ── visibility window + negative-last-speed sentinel ──────────────────────

    // binds finding :99 — _shownUntil survives a rebuild while state.time.elapsed resets to 0, so a fresh
    // State can inherit a stale visible window. Also covers the _lastSpeed < 0 (negative sentinel) branch:
    // entering fly arms the readout without showing it. After a rebuild with _shownUntil reset, the first
    // flying frame must NOT show the overlay (armed, not visible).
    test("a fresh State does not inherit a stale visibility window — entering fly arms without showing", () => {
        // State A: fly and change speed to set _shownUntil well into the future
        const cam = makeFlyingCam(5);
        state.step(1 / 60); // first flying frame: _lastSpeed goes from -1 to 5 (armed, not shown)
        expect(overlayRoot()?.style.opacity).toBe("0"); // armed, not visible

        // change speed to set _shownUntil = elapsed + HoldSeconds
        Orbit.flySpeed.set(cam, 8);
        state.step(1 / 60); // speed changed → _shownUntil set, visible
        expect(overlayRoot()?.style.opacity).toBe("1");

        // simulate a rebuild: dispose, new State, warm
        state.dispose();
        state = new State();
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Orbit", Orbit, OrbitPlugin.traits?.Orbit);
        register("OrbitSmooth", OrbitSmooth);
        Slab.collect();
        attach(state, OrbitOverlayPlugin);
        OrbitOverlayPlugin.warm?.(state);

        // fresh State: elapsed is 0, _shownUntil must be 0 (not stale from State A)
        makeFlyingCam(5);
        state.step(1 / 60); // first flying frame: _lastSpeed < 0 → arm without showing
        expect(overlayRoot()?.style.opacity).toBe("0"); // NOT visible — no stale window
    });

    // ── fade-hold text ─────────────────────────────────────────────────────────

    // binds finding :92 — the not-flying path called set(0, 0, false, false), rewriting the text to
    // "fly 0.0 u/s" mid-fade, contradicting set's own comment that the last value reads as it dims out.
    // After the fix, set only updates text while visible, so the last flying speed holds during the fade.
    test("the readout holds its last text while fading out", () => {
        const cam = makeFlyingCam(5);
        state.step(1 / 60); // armed
        Orbit.flySpeed.set(cam, 7);
        state.step(1 / 60); // speed changed → visible, text = "fly 7.0 u/s"
        expect(speedEl()?.textContent).toBe("fly 7.0 u/s");
        expect(overlayRoot()?.style.opacity).toBe("1");

        // stop flying — the overlay fades out but must keep showing "fly 7.0 u/s", not "fly 0.0 u/s"
        OrbitSmooth.flyActive.set(cam, 0);
        state.step(1 / 60);
        expect(overlayRoot()?.style.opacity).toBe("0"); // fading
        expect(speedEl()?.textContent).toBe("fly 7.0 u/s"); // last value holds, not rewritten to 0
    });

    // ── flying-camera selection ───────────────────────────────────────────────

    // covers the flying-camera selection gate: the first flying camera in query order owns the readout.
    // Also binds the state-owned teardown (finding :108): state.dispose() must remove the overlay.
    test("the first flying camera in query order owns the readout", () => {
        const idle = makeOrbitingCam(); // not flying — should NOT own the readout
        Orbit.flySpeed.set(idle, 3);
        const flying = makeFlyingCam(7); // flying — should own the readout
        state.step(1 / 60); // first flying frame: arm without showing
        Orbit.flySpeed.set(flying, 9); // change speed → visible on next frame
        state.step(1 / 60);

        expect(speedEl()?.textContent).toBe("fly 9.0 u/s"); // flying cam's speed, not the idle cam's

        // the idle cam starts flying instead — now it's first in query order, so it takes over
        OrbitSmooth.flyActive.set(idle, 1);
        OrbitSmooth.flyActive.set(flying, 0);
        state.step(1 / 60);
        expect(speedEl()?.textContent).toBe("fly 3.0 u/s"); // idle cam's speed now

        // state-owned teardown: dispose removes the overlay node
        const div = overlayDiv()!;
        state.dispose();
        expect(div.removed).toBe(true);
    });
});
