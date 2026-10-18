import { ShapeType } from "../common/types";
import { kernelMover } from "./shape_query_gold";
// Behavioral port of Box3D's test/test_mover.c. These assert the invariants the plane solver and the
// mover-collide functions must hold (converged iteration counts, valid normalized normals, correct
// push-out direction and depth) — independent of the bit-exact gold gate in mover.gold.test.ts, so they
// guard against the C reference itself being wrong, which equality against it cannot.

import { expect, test } from "bun:test";
import type { Vec3 } from "../common/records";
import { vector } from "../kernel/math";

const absf = Math.abs;

import { makeBoxHull } from "../shapes/build-hull";
import type { Capsule, Sphere } from "../shapes/geometry";
import { type CollisionPlane, solvePlanes } from "./mover";

const FLT_MAX = 3.4028234663852886e38;
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const rigidPlane = (normal: Vec3, offset: number): CollisionPlane => ({
    plane: { normal, offset },
    pushLimit: FLT_MAX,
    push: 0,
    clipVelocity: true,
});

test("the mover plane solver resolves two parallel planes in two iterations, onto the deeper plane", () => {
    const planes = [rigidPlane(v(0, 0, 1), 0.5), rigidPlane(v(0, 0, 1), 1.0)];
    const result = solvePlanes(v(0, 0, 0), planes, 2);
    expect(result.iterationCount).toBe(2);
    expect(absf(result.delta.z - 1.0)).toBeLessThan(0.0055);
});

test("mover-versus-sphere returns no plane for a mover nowhere near the sphere", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
    const mover: Capsule = { center1: v(4, 3, 0), center2: v(6, 3, 0), radius: 0.2 };
    expect(kernelMover(ShapeType.Sphere, shape, mover)[0] ?? null).toBeNull();
});

test("mover-versus-sphere returns a normalized upward plane at the overlap depth on a shallow touch", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
    const mover: Capsule = { center1: v(-1, 0.6, 0), center2: v(1, 0.6, 0), radius: 0.2 };
    const r = kernelMover(ShapeType.Sphere, shape, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
});

test("mover-versus-sphere gives a mover whose axis runs through the sphere centre a normalized normal perpendicular to that axis at the combined radius", () => {
    const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
    const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
    const r = kernelMover(ShapeType.Sphere, shape, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    // The fallback axis is perpendicular to the mover axis (X).
    expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
    // Deepest possible penetration: the full combined radius.
    expect(absf(r.plane.offset - 0.7)).toBeLessThan(1e-5);
});

const capsuleShape: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.3 };

test("mover-versus-capsule returns no plane for a mover well above the capsule", () => {
    const mover: Capsule = { center1: v(-1, 5, 0), center2: v(1, 5, 0), radius: 0.2 };
    expect(kernelMover(ShapeType.Capsule, capsuleShape, mover)[0] ?? null).toBeNull();
});

test("mover-versus-capsule returns a normalized upward plane at the overlap depth on a shallow touch", () => {
    const mover: Capsule = { center1: v(-1, 0.4, 0), center2: v(1, 0.4, 0), radius: 0.2 };
    const r = kernelMover(ShapeType.Capsule, capsuleShape, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
});

test("mover-versus-capsule gives two crossing core segments their mutual perpendicular as the normal, at the combined radius", () => {
    const mover: Capsule = { center1: v(0, 0, -1), center2: v(0, 0, 1), radius: 0.2 };
    const r = kernelMover(ShapeType.Capsule, capsuleShape, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
    expect(absf(r.plane.normal.z)).toBeLessThan(1e-5);
    expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
});

test("mover-versus-capsule gives core segments on one axis a normalized normal perpendicular to it at the combined radius", () => {
    const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
    const r = kernelMover(ShapeType.Capsule, capsuleShape, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
    expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
});

const boxHull = makeBoxHull(0.5, 0.5, 0.5);

test("mover-versus-hull returns no plane for a mover far above the box", () => {
    const mover: Capsule = { center1: v(-0.3, 5, 0), center2: v(0.3, 5, 0), radius: 0.2 };
    expect(kernelMover(ShapeType.Hull, boxHull, mover)[0] ?? null).toBeNull();
});

test("mover-versus-hull returns the touched face's normalized normal at the overlap depth", () => {
    const mover: Capsule = { center1: v(-0.3, 0.6, 0), center2: v(0.3, 0.6, 0), radius: 0.2 };
    const r = kernelMover(ShapeType.Hull, boxHull, mover)[0] ?? null;
    expect(r).not.toBeNull();
    if (!r) return;
    expect(vector.isNormalized(r.plane.normal)).toBe(true);
    expect(r.plane.normal.y).toBeGreaterThan(0.99);
    expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-4);
});

test("mover-versus-hull returns no plane for a mover fully inside the box", () => {
    const mover: Capsule = { center1: v(-0.2, 0, 0), center2: v(0.2, 0, 0), radius: 0.1 };
    expect(kernelMover(ShapeType.Hull, boxHull, mover)[0] ?? null).toBeNull();
});
