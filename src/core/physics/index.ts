import { component, f32, type Plugin, type Resource, u32, vec3 } from "../../engine";
import { Transform, TransformPlugin } from "../transform";
import { PhysicsAssets } from "./assets";
import { Compounds } from "./compound";
import { HeightFields } from "./height-field";
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
import { PhysicsMeshes } from "./mesh";
import { Shape, type ShapeMaterialSet } from "./shape";

export type { PhysicsAssets } from "./assets";
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

export {
    Shape,
    ShapeKind,
    type ShapeMaterialSet,
    type SurfaceMaterial,
} from "./shape";

/** World-owned per-triangle material lists. `add` returns a handle; `Shape.materialSet` holds
 * that handle plus one (zero means none). Names are display labels. */
export const ShapeMaterials: Resource<PhysicsAssets<ShapeMaterialSet>> = {
    create: () => new PhysicsAssets(),
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
 * definition fields with setters apply at the next fixed sync; initial velocities are spawn-only.
 * Pose lives in Transform, which physics writes back after each step. A marked translation or
 * rotation write teleports the body at the next step. Transform.scale scales every attached shape,
 * including shapes on separate entities, and applies changes at the next fixed sync.
 * Scale must be finite and nonzero; height fields require positive components, and spheres and
 * capsules (including compound children) require equal component magnitudes. Refused shapes retry
 * after a named warning and attach once their body scale is valid.
 */
export const Body = component(
    "Body",
    {
        /** Box3D motion type; changing it applies at the next fixed sync. */
        type: u32,
        /** Initial linear velocity in meters per second; spawn-only. */
        linearVelocity: vec3,
        /** Initial angular velocity in radians per second; spawn-only. */
        angularVelocity: vec3,
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
            linearVelocity: [0, 0, 0],
            angularVelocity: [0, 0, 0],
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
        requires: [Transform],
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
export { type Compound, Compounds } from "./compound";
export { type HeightField, HeightFields } from "./height-field";
export { type Hull, Hulls, UNIT_CUBE_ID } from "./hull";
export { type PhysicsMesh, PhysicsMeshes } from "./mesh";
