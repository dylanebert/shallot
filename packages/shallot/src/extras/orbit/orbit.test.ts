import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { attach } from "../../../tests/helpers";
import {
    InputPlugin,
    Orbit,
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
