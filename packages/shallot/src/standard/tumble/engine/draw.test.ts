// Stage 15 (render ABI) tests: the b3World_Draw walk resolves each shape/joint to typed callbacks.
// Behavioral, not bit-exact — the draw walk is a read-only view that never feeds the world-state hash.
// The invariants: one callback per shape by type, joints emit their connection segments, drawingBounds
// culls, flags gate categories, and the walk mutates nothing (a second draw repeats identically).

import { describe, expect, test } from "bun:test";
import { type DebugDraw, defaultDebugDraw } from "./draw";
import { BodyType, makeBoxHull, type Vec3, World } from "./index";

/** A debug draw that tallies each callback and records the last string, over the defaults + flags. */
function counting(flags: Partial<DebugDraw> = {}): DebugDraw & { counts: Record<string, number> } {
    const counts: Record<string, number> = {
        sphere: 0,
        capsule: 0,
        hull: 0,
        mesh: 0,
        heightField: 0,
        segment: 0,
        point: 0,
        transform: 0,
        aabb: 0,
        string: 0,
    };
    return {
        ...defaultDebugDraw(),
        drawShapes: true,
        drawSolidSphere: () => {
            counts.sphere++;
        },
        drawSolidCapsule: () => {
            counts.capsule++;
        },
        drawSolidHull: () => {
            counts.hull++;
        },
        drawSolidMesh: () => {
            counts.mesh++;
        },
        drawSolidHeightField: () => {
            counts.heightField++;
        },
        drawSegment: () => {
            counts.segment++;
        },
        drawPoint: () => {
            counts.point++;
        },
        drawTransform: () => {
            counts.transform++;
        },
        drawAabb: () => {
            counts.aabb++;
        },
        drawString: () => {
            counts.string++;
        },
        ...flags,
        counts,
    };
}

const box = (hx: number, hy: number, hz: number) => makeBoxHull(hx, hy, hz);

describe("world draw walk", () => {
    test("one solid callback per shape, by type", () => {
        const world = new World();
        const ground = world.createBody({ type: BodyType.Static });
        ground.createHull({}, box(10, 1, 10));

        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        b.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        b.createCapsule(
            {},
            { center1: { x: -1, y: 0, z: 0 }, center2: { x: 1, y: 0, z: 0 }, radius: 0.3 },
        );
        b.createHull({}, box(0.5, 0.5, 0.5));

        const d = counting();
        world.draw(d);
        // Ground hull + body hull = 2 hulls; 1 sphere; 1 capsule.
        expect(d.counts.hull).toBe(2);
        expect(d.counts.sphere).toBe(1);
        expect(d.counts.capsule).toBe(1);
        world.destroy();
    });

    test("flags gate categories; bounds one per shape", () => {
        const world = new World();
        const ground = world.createBody({ type: BodyType.Static });
        ground.createHull({}, box(10, 1, 10));
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        b.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });

        // Shapes off, bounds on: no solids, one AABB per shape (2), no mass frames.
        const d = counting({ drawShapes: false, drawBounds: true });
        world.draw(d);
        expect(d.counts.sphere).toBe(0);
        expect(d.counts.hull).toBe(0);
        expect(d.counts.aabb).toBe(2);

        // Mass on: a transform + a label for the single dynamic body only.
        const m = counting({ drawShapes: false, drawMass: true });
        world.draw(m);
        expect(m.counts.transform).toBe(1);
        expect(m.counts.string).toBe(1);
        world.destroy();
    });

    test("a joint draws its connection", () => {
        const world = new World();
        const a = world.createBody({ type: BodyType.Static, position: { x: 0, y: 5, z: 0 } });
        a.createHull({}, box(0.5, 0.5, 0.5));
        const bBody = world.createBody({ type: BodyType.Dynamic, position: { x: 1, y: 5, z: 0 } });
        bBody.createHull({}, box(0.5, 0.5, 0.5));
        world.createRevoluteJoint(a, bBody, {
            localFrameA: { p: { x: 0.5, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
            localFrameB: { p: { x: -0.5, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        });

        const off = counting({ drawJoints: false });
        world.draw(off);
        const segmentsWithoutJoint = off.counts.segment;

        const on = counting({ drawJoints: true });
        world.draw(on);
        // The generic joint draw emits three connection segments + two anchor points + two frames.
        expect(on.counts.segment).toBe(segmentsWithoutJoint + 3);
        expect(on.counts.point).toBeGreaterThanOrEqual(2);
        world.destroy();
    });

    test("drawingBounds culls far shapes", () => {
        const world = new World();
        const near = world.createBody({ type: BodyType.Static });
        near.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        const far = world.createBody({ type: BodyType.Static, position: { x: 500, y: 0, z: 0 } });
        far.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });

        const d = counting();
        // Default ±100 m bounds exclude the shape at x=500.
        world.draw(d);
        expect(d.counts.sphere).toBe(1);
        world.destroy();
    });

    test("the walk mutates nothing", () => {
        const world = new World();
        const b = world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 4, z: 0 } });
        b.createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world.step(1 / 60, 4);
        const before: Vec3 = b.getPosition();

        const first = counting();
        world.draw(first);
        const after = b.getPosition();
        expect(after.x).toBe(before.x);
        expect(after.y).toBe(before.y);
        expect(after.z).toBe(before.z);

        // A second identical draw repeats the same counts (no accumulated state).
        const second = counting();
        world.draw(second);
        expect(second.counts).toEqual(first.counts);
        world.destroy();
    });
});
