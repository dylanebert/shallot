// Public data types for world / body / shape creation, ported from Box3D's types.h + the
// b3Default* factories in types.c (Erin Catto, MIT). Pure data — the def structs bundle creation
// parameters and are safe to reuse. Default values are bit-exact with the C reference (length unit
// fixed at 1.0, see core.ts). Single-threaded port: the threading and debug-draw fields of the C
// defs (workerCount, enqueueTask, createDebugShape, ...) and the internalValue cookie are dropped.

import { hi32, lo32 } from "./bits";
import { DEFAULT_CATEGORY_BITS, DEFAULT_MASK_BITS, LENGTH_UNITS_PER_METER } from "./core";
import { f32, type Pos, type Quat, quat, type Vec3 } from "./math";

/** Body simulation type (b3BodyType). Numeric values are load-bearing (broadphase tree index). */
export const BodyType = {
    /** Zero mass, zero velocity, may be moved manually. */
    Static: 0,
    /** Zero mass, velocity set by the user, moved by the solver. */
    Kinematic: 1,
    /** Positive mass, velocity from forces, moved by the solver. */
    Dynamic: 2,
} as const;
export type BodyType = (typeof BodyType)[keyof typeof BodyType];

/** Shape type (b3ShapeType). Numeric values index the contact-register dispatch table — keep them. */
export const ShapeType = {
    Capsule: 0,
    Compound: 1,
    HeightField: 2,
    Hull: 3,
    Mesh: 4,
    Sphere: 5,
} as const;
export type ShapeType = (typeof ShapeType)[keyof typeof ShapeType];

/** Per-axis translation and rotation locks (b3MotionLocks). */
export type MotionLocks = {
    linearX: boolean;
    linearY: boolean;
    linearZ: boolean;
    angularX: boolean;
    angularY: boolean;
    angularZ: boolean;
};

/**
 * Collision filtering data (b3Filter). Category/mask are real u64 bit sets → bigint (matching the
 * broadphase tree convention). A non-zero group forces collision (positive) or skips it (negative),
 * overriding the mask.
 */
export type Filter = {
    categoryBits: bigint;
    maskBits: bigint;
    groupIndex: number;
};

/** @returns the default filter: all categories, all masks, no group (b3DefaultFilter). */
export function defaultFilter(): Filter {
    return { categoryBits: DEFAULT_CATEGORY_BITS, maskBits: DEFAULT_MASK_BITS, groupIndex: 0 };
}

/** A spatial query's collision filter (b3QueryFilter); the recording id/name fields are omitted. */
export type QueryFilter = {
    categoryBits: bigint;
    maskBits: bigint;
};

/** @returns the default query filter: all categories, all masks (b3DefaultQueryFilter). */
export function defaultQueryFilter(): QueryFilter {
    return { categoryBits: DEFAULT_CATEGORY_BITS, maskBits: DEFAULT_MASK_BITS };
}

/**
 * A {@link Filter} with the u64 bit sets split into u32 halves. This is what a shape stores and
 * what every filter test reads; `bigint` lives only on the public boundary.
 */
export type FilterBits = {
    categoryHi: number;
    categoryLo: number;
    maskHi: number;
    maskLo: number;
    groupIndex: number;
};

/** A {@link QueryFilter} in the split form; queries convert once at their entry point. */
export type QueryFilterBits = {
    categoryHi: number;
    categoryLo: number;
    maskHi: number;
    maskLo: number;
};

/** @returns the split form of a user-supplied shape filter. */
export function toFilterBits(filter: Filter): FilterBits {
    return {
        categoryHi: hi32(filter.categoryBits),
        categoryLo: lo32(filter.categoryBits),
        maskHi: hi32(filter.maskBits),
        maskLo: lo32(filter.maskBits),
        groupIndex: filter.groupIndex,
    };
}

/** @returns the split form of a user-supplied query filter. */
export function toQueryFilterBits(filter: QueryFilter): QueryFilterBits {
    return {
        categoryHi: hi32(filter.categoryBits),
        categoryLo: lo32(filter.categoryBits),
        maskHi: hi32(filter.maskBits),
        maskLo: lo32(filter.maskBits),
    };
}

/** Whether a shape's filter admits this query (b3ShouldQueryCollide). */
export function shouldQueryCollide(shape: FilterBits, query: QueryFilterBits): boolean {
    return (
        ((shape.categoryHi & query.maskHi) | (shape.categoryLo & query.maskLo)) !== 0 &&
        ((shape.maskHi & query.categoryHi) | (shape.maskLo & query.categoryLo)) !== 0
    );
}

/** Surface material properties (b3SurfaceMaterial). Per-triangle on meshes/height fields. */
export type SurfaceMaterial = {
    friction: number;
    restitution: number;
    rollingResistance: number;
    tangentVelocity: Vec3;
    /** Application material id; passed to queries and mixing callbacks. Real u64 → bigint. */
    userMaterialId: bigint;
    /** Debug color, low 24 bits RGB; 0 = ignored. */
    customColor: number;
};

/** @returns the default surface material: friction 0.6, everything else zero (b3DefaultSurfaceMaterial). */
export function defaultSurfaceMaterial(): SurfaceMaterial {
    return {
        friction: f32(0.6),
        restitution: 0,
        rollingResistance: 0,
        tangentVelocity: { x: 0, y: 0, z: 0 },
        userMaterialId: 0n,
        customColor: 0,
    };
}

/** A deep copy of a surface material, so stored materials never alias caller-owned defs. */
export function cloneMaterial(m: SurfaceMaterial): SurfaceMaterial {
    return {
        friction: m.friction,
        restitution: m.restitution,
        rollingResistance: m.rollingResistance,
        tangentVelocity: { x: m.tangentVelocity.x, y: m.tangentVelocity.y, z: m.tangentVelocity.z },
        userMaterialId: m.userMaterialId,
        customColor: m.customColor,
    };
}

/** Optional initial pool capacities (b3Capacity). */
export type Capacity = {
    staticShapeCount: number;
    dynamicShapeCount: number;
    staticBodyCount: number;
    dynamicBodyCount: number;
    contactCount: number;
};

/** Mixing callback signature for friction/restitution (b3FrictionCallback / b3RestitutionCallback). */
export type MixCallback = (
    a: number,
    userMaterialIdA: bigint,
    b: number,
    userMaterialIdB: bigint,
) => number;

/**
 * World creation parameters (b3WorldDef). Reusable pure data. Threading, task-system, and
 * debug-draw fields of the C def are dropped (single-threaded port); friction/restitution mixing
 * callbacks default to the built-in mix functions when left undefined.
 */
export type WorldDef = {
    gravity: Vec3;
    restitutionThreshold: number;
    hitEventThreshold: number;
    contactHertz: number;
    contactDampingRatio: number;
    contactSpeed: number;
    maximumLinearSpeed: number;
    frictionCallback?: MixCallback;
    restitutionCallback?: MixCallback;
    enableSleep: boolean;
    enableContinuous: boolean;
    userData?: unknown;
    capacity?: Capacity;
};

/** @returns the default world definition (b3DefaultWorldDef), bit-exact at length unit 1.0. */
export function defaultWorldDef(): WorldDef {
    const lu = LENGTH_UNITS_PER_METER;
    return {
        gravity: { x: 0, y: f32(-10.0), z: 0 },
        restitutionThreshold: f32(1.0 * lu),
        hitEventThreshold: f32(1.0 * lu),
        contactHertz: f32(30.0),
        contactDampingRatio: f32(10.0),
        contactSpeed: f32(3.0 * lu),
        // 400 m/s, faster than the speed of sound
        maximumLinearSpeed: f32(400.0 * lu),
        enableSleep: true,
        enableContinuous: true,
    };
}

/** Rigid body creation parameters (b3BodyDef). Shapes are added after construction. */
export type BodyDef = {
    type: BodyType;
    position: Pos;
    rotation: Quat;
    linearVelocity: Vec3;
    angularVelocity: Vec3;
    linearDamping: number;
    angularDamping: number;
    gravityScale: number;
    sleepThreshold: number;
    /** Optional debug name, truncated to BODY_NAME_LENGTH-1 characters. */
    name?: string;
    userData?: unknown;
    motionLocks: MotionLocks;
    enableSleep: boolean;
    isAwake: boolean;
    isBullet: boolean;
    isEnabled: boolean;
    allowFastRotation: boolean;
    enableContactRecycling: boolean;
};

/** @returns the default body definition (b3DefaultBodyDef): a static, awake, enabled body. */
export function defaultBodyDef(): BodyDef {
    return {
        type: BodyType.Static,
        position: { x: 0, y: 0, z: 0 },
        rotation: quat.identity(),
        linearVelocity: { x: 0, y: 0, z: 0 },
        angularVelocity: { x: 0, y: 0, z: 0 },
        linearDamping: 0,
        angularDamping: 0,
        gravityScale: f32(1.0),
        sleepThreshold: f32(0.05 * LENGTH_UNITS_PER_METER),
        motionLocks: {
            linearX: false,
            linearY: false,
            linearZ: false,
            angularX: false,
            angularY: false,
            angularZ: false,
        },
        enableSleep: true,
        isAwake: true,
        isBullet: false,
        isEnabled: true,
        allowFastRotation: false,
        enableContactRecycling: true,
    };
}

/** Shape creation parameters (b3ShapeDef). */
export type ShapeDef = {
    userData?: unknown;
    /** Per-triangle materials for mesh shapes; ignored for convex and compound shapes. */
    materials?: SurfaceMaterial[];
    /** The base surface material (ignored for compound shapes). */
    baseMaterial: SurfaceMaterial;
    density: number;
    explosionScale: number;
    filter: Filter;
    enableCustomFiltering: boolean;
    isSensor: boolean;
    enableSensorEvents: boolean;
    enableContactEvents: boolean;
    enableHitEvents: boolean;
    enablePreSolveEvents: boolean;
    invokeContactCreation: boolean;
    updateBodyMass: boolean;
};

/** @returns the default shape definition (b3DefaultShapeDef): water density, all-pass filter. */
export function defaultShapeDef(): ShapeDef {
    const lu = LENGTH_UNITS_PER_METER;
    return {
        baseMaterial: defaultSurfaceMaterial(),
        // density of water
        density: f32(1000.0 / f32(f32(lu * lu) * lu)),
        explosionScale: f32(1.0),
        filter: defaultFilter(),
        enableCustomFiltering: false,
        isSensor: false,
        enableSensorEvents: false,
        enableContactEvents: false,
        enableHitEvents: false,
        enablePreSolveEvents: false,
        invokeContactCreation: true,
        updateBodyMass: true,
    };
}
