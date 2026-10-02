// Behavioral port of Box3D's test/test_mover.c. These assert the invariants the plane solver and the
// mover-collide functions must hold (converged iteration counts, valid normalized normals, correct
// push-out direction and depth) — independent of the bit-exact gold gate in mover.gold.test.ts, so they
// guard against the C reference itself being wrong, which equality against it cannot.

import { expect, test } from "bun:test";
import { absf, type Vec3, vec3 } from "../common/math";
import {
    type Capsule,
    collideMoverAndCapsule,
    collideMoverAndSphere,
    type Sphere,
} from "../shapes/geometry";
import { collideMoverAndHull, makeBoxHull } from "../shapes/hull";
import { type CollisionPlane, solvePlanes } from "./mover";

const FLT_MAX = 3.4028234663852886e38;
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const rigidPlane = (normal: Vec3, offset: number): CollisionPlane => ({
    plane: { normal, offset },
    pushLimit: FLT_MAX,
    push: 0,
    clipVelocity: true,
});

test("the mover plane solver stops converging on an easy pair of parallel planes, burning iterations or landing short of the deeper plane", () => {
    const planes = [rigidPlane(v(0, 0, 1), 0.5), rigidPlane(v(0, 0, 1), 1.0)];
    const result = solvePlanes(v(0, 0, 0), planes, 2);
    expect(result.iterationCount).toBe(2);
    expect(absf(result.delta.z - 1.0)).toBeLessThan(0.0055);
});

test("mover-versus-sphere invents a contact plane for a mover nowhere near the sphere, so a character snags on empty space", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
    const mover: Capsule = { center1: v(4, 3, 0), center2: v(6, 3, 0), radius: 0.2 };
    expect(collideMoverAndSphere(shape, mover)).toBeNull();
});

test("mover-versus-sphere returns an unnormalized normal, the wrong push direction, or a depth other than the actual overlap on a shallow touch", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
    const mover: Capsule = { center1: v(-1, 0.6, 0), center2: v(1, 0.6, 0), radius: 0.2 };
    const r = collideMoverAndSphere(shape, mover);
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vec3.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
});

test("mover-versus-sphere degenerates when the mover axis runs through the sphere centre, emitting a zero or axis-aligned-with-the-mover normal instead of the perpendicular fallback at the combined radius", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
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

const capsuleShape: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.3 };

test("mover-versus-capsule invents a contact plane for a mover well above the capsule, so a character snags on empty space", () => {
    const mover: Capsule = { center1: v(-1, 5, 0), center2: v(1, 5, 0), radius: 0.2 };
    expect(collideMoverAndCapsule(capsuleShape, mover)).toBeNull();
});

test("mover-versus-capsule returns an unnormalized normal, the wrong push direction, or a depth other than the actual overlap on a shallow touch", () => {
    const mover: Capsule = { center1: v(-1, 0.4, 0), center2: v(1, 0.4, 0), radius: 0.2 };
    const r = collideMoverAndCapsule(capsuleShape, mover);
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vec3.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
});

test("mover-versus-capsule picks a normal in the plane of two crossing core segments instead of the mutual perpendicular, so a crossed character is pushed along a shape axis", () => {
    const mover: Capsule = { center1: v(0, 0, -1), center2: v(0, 0, 1), radius: 0.2 };
    const r = collideMoverAndCapsule(capsuleShape, mover);
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vec3.isNormalized(r.plane.normal)).toBe(true);
    expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
    expect(absf(r.plane.normal.z)).toBeLessThan(1e-5);
    expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
});

test("mover-versus-capsule degenerates when both core segments lie on the same axis, emitting a zero or along-axis normal instead of a perpendicular at the combined radius", () => {
    const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
    const r = collideMoverAndCapsule(capsuleShape, mover);
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vec3.isNormalized(r.plane.normal)).toBe(true);
    expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
    expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
});

const boxHull = makeBoxHull(0.5, 0.5, 0.5);

test("mover-versus-hull invents a contact plane for a mover far above the box, so a character snags on empty space", () => {
    const mover: Capsule = { center1: v(-0.3, 5, 0), center2: v(0.3, 5, 0), radius: 0.2 };
    expect(collideMoverAndHull(boxHull, mover)).toBeNull();
});

test("mover-versus-hull returns an unnormalized normal, a push that is not the touched face normal, or a depth other than the actual overlap", () => {
    const mover: Capsule = { center1: v(-0.3, 0.6, 0), center2: v(0.3, 0.6, 0), radius: 0.2 };
    const r = collideMoverAndHull(boxHull, mover);
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vec3.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-4);
});

test("mover-versus-hull emits a plane with a degenerate zero normal for a mover fully inside the box instead of dropping the contact", () => {
    const mover: Capsule = { center1: v(-0.2, 0, 0), center2: v(0.2, 0, 0), radius: 0.1 };
    expect(collideMoverAndHull(boxHull, mover)).toBeNull();
});
