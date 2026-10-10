import { component, f32, type Plugin, Registry, type Resource, u32, vec4 } from "../../engine";
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
import { Shape, type ShapeGeometry, type ShapeMaterialSet } from "./shape";

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

export { Shape, type ShapeGeometry, ShapeKind, type ShapeMaterialSet } from "./shape";

/** Named mesh geometry referenced by a Shape of kind Mesh. */
export const PhysicsMeshes: Resource<Registry<ShapeGeometry>> = {
    create: () => new Registry(),
};
/** Named height-field geometry referenced by a Shape of kind HeightField. */
export const HeightFields: Resource<Registry<ShapeGeometry>> = {
    create: () => new Registry(),
};
/** Named compound geometry referenced by a Shape of kind Compound. */
export const Compounds: Resource<Registry<ShapeGeometry>> = {
    create: () => new Registry(),
};
/** Named per-triangle material lists referenced by Shape.materialSet (registry id plus one). */
export const ShapeMaterials: Resource<Registry<ShapeMaterialSet>> = {
    create: () => new Registry(),
};

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
        }),
        requires: [GlobalTransform],
    },
);

/** Registers shared physics authoring data without installing a simulation. */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    dependencies: [TransformPlugin],
    recovery(world) {
        const snapshot = () => ({
            hulls: structuredClone(world.resource(Hulls).snapshot()),
            meshes: structuredClone(world.resource(PhysicsMeshes).snapshot()),
            heightFields: structuredClone(world.resource(HeightFields).snapshot()),
            compounds: structuredClone(world.resource(Compounds).snapshot()),
            shapeMaterials: structuredClone(world.resource(ShapeMaterials).snapshot()),
        });
        return {
            snapshot,
            restore: (state: ReturnType<typeof snapshot>) => {
                world.resource(Hulls).restore(structuredClone(state.hulls));
                world.resource(PhysicsMeshes).restore(structuredClone(state.meshes));
                world.resource(HeightFields).restore(structuredClone(state.heightFields));
                world.resource(Compounds).restore(structuredClone(state.compounds));
                world.resource(ShapeMaterials).restore(structuredClone(state.shapeMaterials));
            },
        };
    },
    components: [
        Body,
        Shape,
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
