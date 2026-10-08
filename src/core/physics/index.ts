import { component, f32, type Plugin, u32, vec4 } from "../../engine";
import { GlobalTransform, TransformPlugin } from "../transform";
import { Hulls } from "./hull";
import {
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "./joints";

export {
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "./joints";

/** collision-shape tag for {@link Body}. Box collides as an OBB; sphere/capsule as a core + radius; hull as a convex polytope (geometry registered in `Hulls`, referenced by `halfExtents.w` = the hull id). */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

/** Box3D's static, kinematic and dynamic motion types; Body defaults to static. */
export const BodyType = { Static: 0, Kinematic: 1, Dynamic: 2 } as const;
export type BodyType = (typeof BodyType)[keyof typeof BodyType];

/**
 * shared rigid-body authoring data; a simulation plugin owns its motion and collisions.
 *
 * @example
 * ```
 * const box = world.create();
 * world.add(box, Body, { type: BodyType.Dynamic, shape: ShapeKind.Box, position: [0, 5, 0, 0], halfExtents: [0.5, 0.5, 0.5, 0], friction: 0.5 });
 * // Omitting type gives static geometry.
 * world.add(world.create(), Body, { position: [0, -0.5, 0, 0], halfExtents: [10, 0.5, 10, 0] });
 * // sphere, radius 0.5
 * world.add(world.create(), Body, { shape: ShapeKind.Sphere, position: [0, 5, 0, 0], halfExtents: [0, 0, 0, 0.5] });
 * // capsule, half-height 0.5, radius 0.3
 * world.add(world.create(), Body, { shape: ShapeKind.Capsule, position: [0, 5, 0, 0], halfExtents: [0, 0.5, 0, 0.3] });
 * // hull id 2, AABB half 1×1×1
 * world.add(world.create(), Body, { shape: ShapeKind.Hull, position: [0, 5, 0, 0], halfExtents: [1, 1, 1, 2] });
 * ```
 */
export const Body = component(
    "Body",
    {
        /** Spawn motion type; static by default. Read once when the body is synchronized. */
        type: u32,
        /** the collider, a `ShapeKind`: `Box` (an OBB of `halfExtents`), `Sphere`, `Capsule` (a segment along local Y inflated by the radius), or `Hull` (a convex polytope registered in `Hulls`). */
        shape: u32,
        /** spawn position; physics owns it after spawn. */
        position: vec4,
        /** spawn orientation as a quaternion `(x, y, z, w)`, like `Transform.rotation`; physics-owned after spawn. */
        rotation: vec4,
        /** box/AABB half-extents in `xyz`; `w` doubles as the rounding radius (sphere/capsule) or the `Hull` id (a hull has radius 0, so the lane is free). */
        halfExtents: vec4,
        /** Dynamic mass in kg; non-positive mass gives zero density. */
        mass: f32,
        /** coulomb friction coefficient: `0` slides freely, higher grips. */
        friction: f32,
    },
    {
        defaults: () => ({
            type: BodyType.Static,
            shape: ShapeKind.Box,
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            halfExtents: [0.5, 0.5, 0.5, 0], // .w = rounding radius (0 for a box)
            mass: 1,
            friction: 0.5,
        }),
        requires: [GlobalTransform],
    },
);

/** one body's live pose + velocity at the last fixed step; sleeping bodies read zero velocity. */
export interface BodyState {
    position: readonly [number, number, number];
    rotation: readonly [number, number, number, number];
    linearVelocity: readonly [number, number, number];
}

/** Registers shared physics authoring data without installing a simulation. */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    dependencies: [TransformPlugin],
    recovery(world) {
        const hulls = world.resource(Hulls);
        return {
            snapshot: () => structuredClone(hulls.snapshot()),
            restore: (state: ReturnType<typeof hulls.snapshot>) =>
                hulls.restore(structuredClone(state)),
        };
    },
    components: [
        Body,
        DistanceJoint,
        FilterJoint,
        MotorJoint,
        ParallelJoint,
        PrismaticJoint,
        RevoluteJoint,
        SphericalJoint,
        WeldJoint,
        WheelJoint,
    ],
};
export { type Hull, type HullFace, Hulls, UNIT_CUBE_ID } from "./hull";
