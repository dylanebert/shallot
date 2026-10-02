import { entity, f32, GlobalTransform, type Plugin, registration, u32, vec4 } from "../../engine";

export { GlobalTransform } from "../../engine";
/** collision-shape tag for {@link Body}. Box collides as an OBB; sphere/capsule as a core + radius; hull as a convex polytope (geometry registered in `Hulls`, referenced by `halfExtents.w` = the hull id). */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

/**
 * shared rigid-body authoring data; a simulation plugin owns its motion and collisions
 * (`mass: 0` = static).
 *
 * @example
 * ```
 * const box = world.create();
 * world.add(box, Body, { shape: ShapeKind.Box, position: [0, 5, 0, 0], halfExtents: [0.5, 0.5, 0.5, 0], friction: 0.5 });
 * // sphere, radius 0.5
 * world.add(world.create(), Body, { shape: ShapeKind.Sphere, position: [0, 5, 0, 0], halfExtents: [0, 0, 0, 0.5] });
 * // capsule, half-height 0.5, radius 0.3
 * world.add(world.create(), Body, { shape: ShapeKind.Capsule, position: [0, 5, 0, 0], halfExtents: [0, 0.5, 0, 0.3] });
 * // hull id 2, AABB half 1×1×1
 * world.add(world.create(), Body, { shape: ShapeKind.Hull, position: [0, 5, 0, 0], halfExtents: [1, 1, 1, 2] });
 * ```
 */
export const Body = {
    /** the collider, a `ShapeKind`: `Box` (an OBB of `halfExtents`), `Sphere`, `Capsule` (a segment along local Y inflated by the radius), or `Hull` (a convex polytope registered in `Hulls`). */
    shape: u32,
    /** spawn position; physics owns it after spawn. */
    position: vec4,
    /** spawn orientation as a quaternion `(x, y, z, w)`, like `Transform.rotation`; physics-owned after spawn. */
    rotation: vec4,
    /** box/AABB half-extents in `xyz`; `w` doubles as the rounding radius (sphere/capsule) or the `Hull` id (a hull has radius 0, so the lane is free). */
    halfExtents: vec4,
    /** mass in kg; `0` or less marks a static body that never moves. */
    mass: f32,
    /** coulomb friction coefficient: `0` slides freely, higher grips. */
    friction: f32,
};

/**
 * a soft distance spring linking two bodies, pulling them toward a rest length; its own entity, holding both bodies' eids.
 *
 * @example
 * ```
 * const anchor = world.create();
 * world.add(anchor, Body, { mass: 0, position: [0, 10, 0, 0] });
 * const block = world.create();
 * world.add(block, Body, { mass: 1, position: [0, 6, 0, 0] });
 * world.add(world.create(), Spring, { a: anchor, b: block, rest: 4, stiffness: 100 });
 * ```
 */
export const Spring = {
    /** the first body's eid. */
    a: entity,
    /** the second body. */
    b: entity,
    /** anchor point on body `a`, in its local frame. */
    rA: vec4,
    /** anchor point on body `b`, in its local frame. */
    rB: vec4,
    /** pull strength; higher is stiffer. */
    stiffness: f32,
    /** the target distance the spring pulls the anchors toward. */
    rest: f32,
};

/**
 * a hard joint pinning two bodies together: a rigid linear pin plus an optional angular lock, holding both bodies' eids.
 *
 * the anchors must start coincident at the bodies' spawn poses (join a dynamic body to a static/kinematic anchor),
 * or construction rejects the joint.
 *
 * @example
 * ```
 * const pivot = world.create();
 * world.add(pivot, Body, { mass: 0, position: [0, 10, 0, 0] });
 * const bob = world.create();
 * world.add(bob, Body, { mass: 1, position: [0, 7.5, 0, 0] });
 * // spherical
 * world.add(world.create(), Joint, { a: pivot, b: bob, rA: [0, 0, 0, 0], rB: [0, 2.5, 0, 0] });
 * const link = world.create();
 * world.add(link, Body, { mass: 1, position: [1, 10, 0, 0] });
 * // fixed
 * world.add(world.create(), Joint, { a: pivot, b: link, rA: [0.5, 0, 0, 0], rB: [-0.5, 0, 0, 0], stiffnessAng: Infinity });
 * ```
 */
export const Joint = {
    /** the first body's eid. */
    a: entity,
    /** the second body. */
    b: entity,
    /** the pin's anchor on body `a`, in its local frame. */
    rA: vec4,
    /** the pin's anchor on body `b`, in its local frame. */
    rB: vec4,
    /** angular lock: `0` (default) leaves rotation free (spherical); `Infinity` locks orientation. */
    stiffnessAng: f32,
};

/** one body's live pose + velocity at the last fixed step; sleeping bodies read zero velocity. */
export interface BodyState {
    position: readonly [number, number, number];
    rotation: readonly [number, number, number, number];
    linearVelocity: readonly [number, number, number];
}

/** Registers shared physics authoring data without installing a simulation. */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    components: [
        registration("Body", Body, {
            defaults: () => ({
                shape: ShapeKind.Box,
                position: [0, 0, 0, 0],
                rotation: [0, 0, 0, 1],
                halfExtents: [0.5, 0.5, 0.5, 0], // .w = rounding radius (0 for a box)
                mass: 1,
                friction: 0.5,
            }),
            requires: [GlobalTransform],
        }),
        registration("Spring", Spring, {
            defaults: () => ({
                a: 0,
                b: 0,
                rA: [0, 0, 0, 0],
                rB: [0, 0, 0, 0],
                stiffness: 100,
                rest: 1,
            }),
        }),
        registration("Joint", Joint, {
            defaults: () => ({ a: 0, b: 0, rA: [0, 0, 0, 0], rB: [0, 0, 0, 0], stiffnessAng: 0 }),
        }),
    ],
};
export { type Hull, type HullFace, Hulls, UNIT_CUBE_ID } from "./hull";
export { bodyCandidates, grabHit, worldToLocal } from "./pick";
export {
    qRotate,
    type RayBody,
    type RayHit,
    rayCapsule,
    raycast,
    rayOBB,
    raySphere,
} from "./raycast";
