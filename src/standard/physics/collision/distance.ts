// Query records from Box3D's types.h (Erin Catto, MIT). The queries run in the kernel
// (crates/physics/src/distance.rs); these are the TypeScript shapes of their inputs and outputs.

import { type Vec3, vec3 } from "../common/math";

/** A convex shape as a point cloud wrapped with a rounding radius (b3ShapeProxy). */
export type ShapeProxy = {
    /** The point cloud. */
    points: Vec3[];
    /** The number of points. */
    count: number;
    /** The external radius of the point cloud. */
    radius: number;
};

/** Low level ray cast input (b3RayCastInput). */
export type RayCastInput = {
    /** Start point of the ray. */
    origin: Vec3;
    /** Ray displacement; end = origin + translation. */
    translation: Vec3;
    /** Maximum fraction of the translation to consider, typically 1. */
    maxFraction: number;
};

/** Low level shape-cast input: a point cloud + radius swept along a translation (b3ShapeCastInput). */
export type ShapeCastInput = {
    proxy: ShapeProxy;
    translation: Vec3;
    maxFraction: number;
    /** Allow an already-touching shape with radius to move slightly closer. */
    canEncroach: boolean;
};

/** Low level ray/shape-cast output (b3CastOutput). */
export type CastOutput = {
    normal: Vec3;
    point: Vec3;
    fraction: number;
    iterations: number;
    triangleIndex: number;
    childIndex: number;
    materialIndex: number;
    hit: boolean;
};

/** A zero-initialized cast output (C `b3CastOutput output = { 0 }`): a miss, all fields zero. */
export function emptyCastOutput(): CastOutput {
    return {
        normal: vec3.zero(),
        point: vec3.zero(),
        fraction: 0,
        iterations: 0,
        triangleIndex: 0,
        childIndex: 0,
        materialIndex: 0,
        hit: false,
    };
}
