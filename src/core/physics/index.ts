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

/** Bit positions stored by `Body.motionLocks`, matching Box3D's `b3MotionLocks`. */
export const BodyMotionLock = {
    linearX: 1 << 0,
    linearY: 1 << 1,
    linearZ: 1 << 2,
    angularX: 1 << 3,
    angularY: 1 << 4,
    angularZ: 1 << 5,
} as const;

/**
 * Shared rigid-body authoring data. Standard physics uses Box3D's body-definition defaults;
 * definition fields with setters apply at the next fixed sync, while spawn pose and initial
 * velocities are read only when the body is created. Physics owns pose and velocity after spawn.
 */
export const Body = component(
    "Body",
    {
        /** Box3D motion type; changing it applies at the next fixed sync. */
        type: u32,
        /** Initial world position in meters; spawn-only. */
        position: vec4,
        /** Initial world rotation as `(x, y, z, w)`; spawn-only. */
        rotation: vec4,
        /** Initial linear velocity in meters per second; spawn-only. */
        linearVelocity: vec4,
        /** Initial angular velocity in radians per second; spawn-only. */
        angularVelocity: vec4,
        /** Linear damping; changing it applies at the next fixed sync. */
        linearDamping: f32,
        /** Angular damping; changing it applies at the next fixed sync. */
        angularDamping: f32,
        /** Non-dimensional gravity multiplier; changing it applies at the next fixed sync. */
        gravityScale: f32,
        /** Sleep speed threshold in meters per second; changing it applies at the next fixed sync. */
        sleepThreshold: f32,
        /** Six `BodyMotionLock` bits; changing them applies at the next fixed sync. */
        motionLocks: u32,
        /** Whether the body may sleep; changing it applies at the next fixed sync. */
        enableSleep: u32,
        /** Initial awake state; spawn-only. Box3D controls awake state after creation. */
        isAwake: u32,
        /** Whether the body uses continuous collision detection; changing it applies at the next fixed sync. */
        isBullet: u32,
        /** Whether the body participates in simulation; changing it applies at the next fixed sync. */
        isEnabled: u32,
        /** Whether the body bypasses rotational speed limits; changing it applies at the next fixed sync. */
        allowFastRotation: u32,
        /** Whether contacts on the body use contact recycling; changing it applies at the next fixed sync. */
        enableContactRecycling: u32,
        /** Temporary box/sphere/capsule/hull collider kind; moved to `Shape` in stage 5. */
        shape: u32,
        /** Temporary collider geometry; moved to `Shape` in stage 5. */
        halfExtents: vec4,
        /** Temporary dynamic mass in kilograms; moved to `Shape` density in stage 5. */
        mass: f32,
        /** Temporary Coulomb friction coefficient; moved to `Shape` in stage 5. */
        friction: f32,
    },
    {
        defaults: () => ({
            type: BodyType.Static,
            position: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            linearVelocity: [0, 0, 0, 0],
            angularVelocity: [0, 0, 0, 0],
            linearDamping: 0,
            angularDamping: 0,
            gravityScale: 1,
            sleepThreshold: 0.05,
            motionLocks: 0,
            enableSleep: 1,
            isAwake: 1,
            isBullet: 0,
            isEnabled: 1,
            allowFastRotation: 0,
            enableContactRecycling: 1,
            shape: ShapeKind.Box,
            halfExtents: [0.5, 0.5, 0.5, 0],
            mass: 1,
            friction: 0.5,
        }),
        requires: [GlobalTransform],
    },
);

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
