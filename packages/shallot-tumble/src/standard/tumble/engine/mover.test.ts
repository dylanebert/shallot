// Behavioral port of Box3D's test/test_mover.c. These assert the invariants the plane solver and the
// mover-collide functions must hold (converged iteration counts, valid normalized normals, correct
// push-out direction and depth) — independent of the bit-exact gold gate in mover.gold.test.ts, so they
// guard against the C reference itself being wrong, which equality against it cannot.

import { describe, expect, test } from "bun:test";
import {
    type Capsule,
    collideMoverAndCapsule,
    collideMoverAndSphere,
    type Sphere,
} from "./geometry";
import { collideMoverAndHull, makeBoxHull } from "./hull";
import { absf, type Vec3, vec3 } from "./math";
import { type CollisionPlane, solvePlanes } from "./mover";

const FLT_MAX = 3.4028234663852886e38;
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const rigidPlane = (normal: Vec3, offset: number): CollisionPlane => ({
    plane: { normal, offset },
    pushLimit: FLT_MAX,
    push: 0,
    clipVelocity: true,
});

describe("solvePlanes", () => {
    test("parallel planes converge in two iterations", () => {
        const planes = [rigidPlane(v(0, 0, 1), 0.5), rigidPlane(v(0, 0, 1), 1.0)];
        const result = solvePlanes(v(0, 0, 0), planes, 2);
        expect(result.iterationCount).toBe(2);
        expect(absf(result.delta.z - 1.0)).toBeLessThan(0.0055);
    });

    test("a deep target takes the full 20 iterations", () => {
        const planes = [
            rigidPlane(v(0, -0.23941046, 0.970918416), 0.390724182),
            rigidPlane(v(0, 0, 1), 1.49998093),
        ];
        const target = v(-2.5390625, 0, -73.6880798);
        planes[0].plane.offset -= vec3.dot(planes[0].plane.normal, target);
        planes[1].plane.offset -= vec3.dot(planes[1].plane.normal, target);
        const result = solvePlanes(v(0, 0, 0), planes, 2);
        expect(result.iterationCount).toBe(20);
    });
});

describe("collideMoverAndSphere", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };

    test("separated returns no plane", () => {
        const mover: Capsule = { center1: v(4, 3, 0), center2: v(6, 3, 0), radius: 0.2 };
        expect(collideMoverAndSphere(shape, mover)).toBeNull();
    });

    test("touching pushes straight out, depth 0.1", () => {
        const mover: Capsule = { center1: v(-1, 0.6, 0), center2: v(1, 0.6, 0), radius: 0.2 };
        const r = collideMoverAndSphere(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
    });

    test("deep overlap falls back to a valid perpendicular normal", () => {
        const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
        const r = collideMoverAndSphere(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        // The fallback axis is perpendicular to the mover axis (X).
        expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
        // Deepest possible penetration: the full combined radius.
        expect(absf(r.plane.offset - 0.7)).toBeLessThan(1e-5);
    });
});

describe("collideMoverAndCapsule", () => {
    const shape: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.3 };

    test("separated returns no plane", () => {
        const mover: Capsule = { center1: v(-1, 5, 0), center2: v(1, 5, 0), radius: 0.2 };
        expect(collideMoverAndCapsule(shape, mover)).toBeNull();
    });

    test("touching pushes straight out, depth 0.1", () => {
        const mover: Capsule = { center1: v(-1, 0.4, 0), center2: v(1, 0.4, 0), radius: 0.2 };
        const r = collideMoverAndCapsule(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
    });

    test("crossing core segments fall back to a normal perpendicular to both", () => {
        const mover: Capsule = { center1: v(0, 0, -1), center2: v(0, 0, 1), radius: 0.2 };
        const r = collideMoverAndCapsule(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
        expect(absf(r.plane.normal.z)).toBeLessThan(1e-5);
        expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
    });

    test("coincident axes fall back to a perpendicular of the mover axis", () => {
        const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
        const r = collideMoverAndCapsule(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
        expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
    });
});

describe("collideMoverAndHull", () => {
    const box = makeBoxHull(0.5, 0.5, 0.5);

    test("separated returns no plane", () => {
        const mover: Capsule = { center1: v(-0.3, 5, 0), center2: v(0.3, 5, 0), radius: 0.2 };
        expect(collideMoverAndHull(box, mover)).toBeNull();
    });

    test("touching the +Y face pushes up, depth 0.1", () => {
        const mover: Capsule = { center1: v(-0.3, 0.6, 0), center2: v(0.3, 0.6, 0), radius: 0.2 };
        const r = collideMoverAndHull(box, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-4);
    });

    test("deep overlap is dropped rather than emitting a zero normal", () => {
        const mover: Capsule = { center1: v(-0.2, 0, 0), center2: v(0.2, 0, 0), radius: 0.1 };
        expect(collideMoverAndHull(box, mover)).toBeNull();
    });
});
