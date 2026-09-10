// Behavioral port of Box3D's test/test_mover.c. These assert the invariants the plane solver and the
// mover-collide functions must hold (converged iteration counts, valid normalized normals, correct
// push-out direction and depth) — independent of the bit-exact gold gate in mover.gold.test.ts, so they
// guard against the C reference itself being wrong, which equality against it cannot.

import { expect } from "bun:test";
import { check } from "../../../harness/check";
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

check(
    "solvePlanes converges two parallel planes in two iterations",
    {
        claim: "the mover plane solver stops converging on an easy pair of parallel planes, burning iterations or landing short of the deeper plane",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const planes = [rigidPlane(v(0, 0, 1), 0.5), rigidPlane(v(0, 0, 1), 1.0)];
        const result = solvePlanes(v(0, 0, 0), planes, 2);
        expect(result.iterationCount).toBe(2);
        expect(absf(result.delta.z - 1.0)).toBeLessThan(0.0055);
    },
);

check(
    "solvePlanes spends the full twenty iterations on a deep target",
    {
        claim: "the mover plane solver's iteration ceiling stops holding, so a deeply penetrating target exits early or runs unbounded",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const planes = [
            rigidPlane(v(0, -0.23941046, 0.970918416), 0.390724182),
            rigidPlane(v(0, 0, 1), 1.49998093),
        ];
        const target = v(-2.5390625, 0, -73.6880798);
        planes[0].plane.offset -= vec3.dot(planes[0].plane.normal, target);
        planes[1].plane.offset -= vec3.dot(planes[1].plane.normal, target);
        const result = solvePlanes(v(0, 0, 0), planes, 2);
        expect(result.iterationCount).toBe(20);
    },
);

check(
    "a mover clear of a sphere reports no collision plane",
    {
        claim: "mover-versus-sphere invents a contact plane for a mover nowhere near the sphere, so a character snags on empty space",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
        const mover: Capsule = { center1: v(4, 3, 0), center2: v(6, 3, 0), radius: 0.2 };
        expect(collideMoverAndSphere(shape, mover)).toBeNull();
    },
);

check(
    "a mover touching a sphere is pushed straight out by the overlap depth",
    {
        claim: "mover-versus-sphere returns an unnormalized normal, the wrong push direction, or a depth other than the actual overlap on a shallow touch",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const shape: Sphere = { center: v(0, 0, 0), radius: 0.5 };
        const mover: Capsule = { center1: v(-1, 0.6, 0), center2: v(1, 0.6, 0), radius: 0.2 };
        const r = collideMoverAndSphere(shape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
    },
);

check(
    "a mover concentric with a sphere falls back to a perpendicular normal at full depth",
    {
        claim: "mover-versus-sphere degenerates when the mover axis runs through the sphere centre, emitting a zero or axis-aligned-with-the-mover normal instead of the perpendicular fallback at the combined radius",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
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
    },
);

const capsuleShape: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.3 };

check(
    "a mover clear of a capsule reports no collision plane",
    {
        claim: "mover-versus-capsule invents a contact plane for a mover well above the capsule, so a character snags on empty space",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-1, 5, 0), center2: v(1, 5, 0), radius: 0.2 };
        expect(collideMoverAndCapsule(capsuleShape, mover)).toBeNull();
    },
);

check(
    "a mover touching a capsule is pushed straight out by the overlap depth",
    {
        claim: "mover-versus-capsule returns an unnormalized normal, the wrong push direction, or a depth other than the actual overlap on a shallow touch",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-1, 0.4, 0), center2: v(1, 0.4, 0), radius: 0.2 };
        const r = collideMoverAndCapsule(capsuleShape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-5);
    },
);

check(
    "crossing capsule core segments fall back to a normal perpendicular to both axes",
    {
        claim: "mover-versus-capsule picks a normal in the plane of two crossing core segments instead of the mutual perpendicular, so a crossed character is pushed along a shape axis",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(0, 0, -1), center2: v(0, 0, 1), radius: 0.2 };
        const r = collideMoverAndCapsule(capsuleShape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
        expect(absf(r.plane.normal.z)).toBeLessThan(1e-5);
        expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
    },
);

check(
    "coincident capsule axes fall back to a perpendicular of the mover axis",
    {
        claim: "mover-versus-capsule degenerates when both core segments lie on the same axis, emitting a zero or along-axis normal instead of a perpendicular at the combined radius",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.2 };
        const r = collideMoverAndCapsule(capsuleShape, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(absf(r.plane.normal.x)).toBeLessThan(1e-5);
        expect(absf(r.plane.offset - 0.5)).toBeLessThan(1e-5);
    },
);

const boxHull = makeBoxHull(0.5, 0.5, 0.5);

check(
    "a mover clear of a box hull reports no collision plane",
    {
        claim: "mover-versus-hull invents a contact plane for a mover far above the box, so a character snags on empty space",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-0.3, 5, 0), center2: v(0.3, 5, 0), radius: 0.2 };
        expect(collideMoverAndHull(boxHull, mover)).toBeNull();
    },
);

check(
    "a mover touching a box hull's +Y face is pushed up by the overlap depth",
    {
        claim: "mover-versus-hull returns an unnormalized normal, a push that is not the touched face normal, or a depth other than the actual overlap",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-0.3, 0.6, 0), center2: v(0.3, 0.6, 0), radius: 0.2 };
        const r = collideMoverAndHull(boxHull, mover);
        expect(r).not.toBeNull();
        if (!r) return;
        expect(vec3.isNormalized(r.plane.normal)).toBe(true);
        expect(r.plane.normal.y).toBeGreaterThan(0.99);
        expect(absf(r.plane.offset - 0.1)).toBeLessThan(1e-4);
    },
);

check(
    "a mover buried inside a box hull is dropped rather than given a zero normal",
    {
        claim: "mover-versus-hull emits a plane with a degenerate zero normal for a mover fully inside the box instead of dropping the contact",
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
    },
    () => {
        const mover: Capsule = { center1: v(-0.2, 0, 0), center2: v(0.2, 0, 0), radius: 0.1 };
        expect(collideMoverAndHull(boxHull, mover)).toBeNull();
    },
);
