import { jointCount } from "../kernel/jointrecords";
import { createSolverSet, solverSetCount, solverSetIndex } from "../kernel/solversetcolumns";
// The simulation world: the root that owns every entity pool and the broad-phase. Ported from
// Box3D's physics_world.c (Erin Catto, MIT). Each entity type has an id pool paired with a sparse
// array of records; the hot payload lives in solver sets. Worlds live in a fixed registry so a
// stale world id (to a destroyed, possibly recycled, world) is detected by a generation mismatch.
//
// fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).

import { type BroadPhase, createBroadPhase } from "../collision/broadphase";
import { contactCount } from "../collision/contact";
import { createManifoldStore, type ManifoldStore } from "../collision/manifoldstore";
import { CONTACT_RECYCLE_DISTANCE } from "../common/constants";
import type { EntityId } from "../common/ids";
import { f32, froundConfig, maxf, type Vec3 } from "../common/math";
import {
    type Capacity,
    defaultShapeDef,
    type MixCallback,
    type ShapeDef,
    type WorldDef,
} from "../common/types";
import { type BodyStore, createBodyStore } from "../kernel/bodycolumns";
import { hullDatabaseIndex, stageHullUpload } from "../kernel/geocolumns";
import { islandKernel } from "../kernel/islandcolumns";
import { kernel } from "../kernel/kernel";
import type { QueryColumns } from "../kernel/querycolumns";
import { createShapeStore, type ShapeStore } from "../kernel/shapecolumns";
import type { CompoundData } from "../shapes/compound";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { MeshData } from "../shapes/mesh";
import type { ShapeGeometry } from "../shapes/shape";
import { destroyShapeAllocations } from "../shapes/shape";
import type { StepContext } from "../solver/contactsolver";
import { createStepProfile, type StepProfile } from "./profile";
import type { Sensor, SensorBeginTouchEvent } from "./sensor";
import { destroySolverSet } from "./solverset";

/** Maximum concurrent worlds (B3_MAX_WORLDS). */
export const MAX_WORLDS = 128;

/** An opaque world handle (b3WorldId). */
export type WorldId = { index1: number; generation: number };

/**
 * A contact begin- or end-touch event (b3ContactBeginTouchEvent / b3ContactEndTouchEvent). The ids
 * are resolved to public {@link Shape}/{@link Contact} handles at getter time; end events read from
 * the previous double buffer, so they survive one step.
 */
export type ContactTouchEvent = {
    shapeIdA: EntityId;
    shapeIdB: EntityId;
    contactId: EntityId;
    normalImpulse: number;
};

/** A contact hit event (b3ContactHitEvent): a collision faster than the world hit threshold. */
export type ContactHitEvent = {
    shapeIdA: EntityId;
    shapeIdB: EntityId;
    contactId: EntityId;
    point: Vec3;
    normal: Vec3;
    approachSpeed: number;
    userMaterialIdA: bigint;
    userMaterialIdB: bigint;
};

/** A sensor end-touch event (b3SensorEndTouchEvent). */
export type SensorEndTouchEvent = { sensorShapeId: EntityId; visitorShapeId: EntityId };

/** Simple counters read back from the world (b3Counters). */
export type Counters = {
    bodyCount: number;
    shapeCount: number;
    contactCount: number;
    jointCount: number;
    islandCount: number;
};

/** The simulation world state (b3World), trimmed to the lifecycle subset the port implements. */
export type WorldState = {
    /** ECS owner; undefined only for the standalone solver API. */
    ecsState: import("../../../engine").World | undefined;
    broadPhase: BroadPhase;

    /** Public body records are the authoring/handle bridge; lifecycle lives in wasm. */
    bodyUserData: unknown[];
    bodyNames: string[];

    jointUserData: unknown[];

    /** Geometry remains host-owned until C2; all nongeometry fields live in the kernel. */
    shapeGeometry: ShapeGeometry[];
    shapeUserData: unknown[];
    shapeNames: string[];
    shapeDefInput: ShapeDef;

    meshDatabase: Map<MeshData, GeometryRecord>;
    heightFieldDatabase: Map<HeightFieldData, GeometryRecord>;
    compoundDatabase: Map<CompoundData, GeometryRecord>;
    // Set when geometry data enters or leaves the databases, or residency/region placement changes.
    geometryDirty: boolean;
    geometryUploadScratch?: import("../kernel/geocolumns").GeometryUploadScratch;
    /** Number of complete geometry uploads since world creation. */
    geometryUploadCount: number;
    // Persistent contact-manifold columns (warm-start state, column-resident): the allocator + wasm
    // region for the manifolds keyed by contactId. Slots are tracked on contact create/destroy.
    manifoldStore: ManifoldStore;
    // Resident body-state columns (velocity/delta/flags of awake bodies), held across steps in the
    // body region. BodyStore binds typed views over the solver sets' columns.
    bodyStore: BodyStore;
    // Resident shape column (type code + local geometry + nextShapeId, one record per shapeId), held
    // across steps so the in-kernel finalize refit walks a body's shape list without a marshal. Written
    // at shape create/destroy — no dirty set (shapecolumns.ts).
    shapeStore: ShapeStore;

    // Dense array of sensor overlap-tracking state, one per sensor shape (b3World.sensors).
    sensors: Sensor[];
    sensorCount: number;
    queryColumns: QueryColumns | null;

    // Event buffers. End events are double-buffered so the user needn't flush every step. Kernel
    // finalization owns the retained body move records; bodyMoveCount is their valid prefix length.
    bodyMoveCount: number;
    sensorBeginEvents: SensorBeginTouchEvent[];
    contactBeginEvents: ContactTouchEvent[];
    sensorEndEvents: [SensorEndTouchEvent[], SensorEndTouchEvent[]];
    contactEndEvents: [ContactTouchEvent[], ContactTouchEvent[]];
    contactHitEvents: ContactHitEvent[];
    jointEvents: number[];
    jointEventCount: number;
    jointEventUserData: unknown[];
    endEventArrayIndex: number;

    stepIndex: number;

    // The per-step solver context, created lazily on the first step and reused across steps (its scalar
    // fields rewritten + its collections cleared each step). One per world — dies with the world, never
    // aliased across worlds. See `step()`.
    stepContext: StepContext | null;

    profile: StepProfile;

    gravity: Vec3;
    hitEventThreshold: number;
    restitutionThreshold: number;
    maxLinearSpeed: number;
    contactSpeed: number;
    contactHertz: number;
    contactDampingRatio: number;
    contactRecycleDistance: number;

    frictionCallback: MixCallback;
    restitutionCallback: MixCallback;

    generation: number;
    maxCapacity: Capacity;

    invH: number;
    invDt: number;

    worldId: number;
    userData: unknown;

    enableSleep: boolean;
    locked: boolean;
    enableWarmStarting: boolean;
    enableContinuous: boolean;
    enableSpeculative: boolean;
    inUse: boolean;
};

/** Default friction mixing: geometric mean (b3DefaultFrictionCallback). */
export const defaultFrictionCallback: MixCallback = (a, _idA, b, _idB) =>
    f32(Math.sqrt(f32(a * b)));

/** Default restitution mixing: the larger of the two (b3DefaultRestitutionCallback). */
export const defaultRestitutionCallback: MixCallback = (a, _idA, b, _idB) => maxf(a, b);

export type GeometryRecord = { refCount: number; geoIndex: number };

/** Retain immutable query geometry by identity; only set membership changes invalidate residency. */
export function addGeometryToDatabase<T>(
    world: WorldState,
    database: Map<T, GeometryRecord>,
    data: T,
): void {
    const entry = database.get(data);
    if (entry) entry.refCount += 1;
    else {
        database.set(data, { refCount: 1, geoIndex: -1 });
        world.geometryDirty = true;
    }
}

export function removeGeometryFromDatabase<T>(
    world: WorldState,
    database: Map<T, GeometryRecord>,
    data: T,
): void {
    const entry = database.get(data);
    if (!entry) return;
    if (--entry.refCount === 0) {
        database.delete(data);
        world.geometryDirty = true;
    }
}

/** A compound datum retains its shared geometry once, independent of the number of shape instances. */
export function addCompoundToDatabase(world: WorldState, data: CompoundData): void {
    const existing = world.compoundDatabase.has(data);
    addGeometryToDatabase(world, world.compoundDatabase, data);
    if (existing) return;
    for (const child of data.hulls) addHullToDatabase(world, child.hull);
    for (const child of data.meshes)
        addGeometryToDatabase(world, world.meshDatabase, child.meshData);
}

export function removeCompoundFromDatabase(world: WorldState, data: CompoundData): void {
    const last = world.compoundDatabase.get(data)?.refCount === 1;
    removeGeometryFromDatabase(world, world.compoundDatabase, data);
    if (!last) return;
    for (const child of data.hulls) removeHullFromDatabase(world, child.hull);
    for (const child of data.meshes)
        removeGeometryFromDatabase(world, world.meshDatabase, child.meshData);
}

// --- hull database ---------------------------------------------------------------------------

/** Intern a hull by content, sharing a single copy across shapes (b3AddHullToDatabase). */
export function addHullToDatabase(world: WorldState, src: HullData): number {
    const bytes = stageHullUpload(world, src);
    return kernel(world.ecsState).hullDatabaseAdd(world.worldId, bytes) >>> 0;
}

/** Release a hull reference, dropping the shared copy when the last shape lets go (b3RemoveHullFromDatabase). */
export function removeHullFromDatabase(world: WorldState, data: HullData | number): void {
    const handle = typeof data === "number" ? data : hullDatabaseIndex(world, data);
    kernel(world.ecsState).hullDatabaseRemove(world.worldId, handle);
}

// --- world registry --------------------------------------------------------------------------

const worlds: (WorldState | undefined)[] = [];
function makeCapacity(c?: Capacity): Capacity {
    return {
        staticShapeCount: c?.staticShapeCount ?? 0,
        dynamicShapeCount: c?.dynamicShapeCount ?? 0,
        staticBodyCount: c?.staticBodyCount ?? 0,
        dynamicBodyCount: c?.dynamicBodyCount ?? 0,
        contactCount: c?.contactCount ?? 0,
    };
}

function makeWorldState(
    world: import("../../../engine").World | undefined,
    def: WorldDef,
    worldId: number,
    generation: number,
): WorldState {
    // Round every user float to f32 once at ingress; the C def is f32, so an f64 scalar would feed the
    // solver an extra bit and break bit-exact parity. Callbacks/capacity/bigints pass through.
    def = froundConfig(def);
    const capacity = makeCapacity(def.capacity);
    kernel(world).graphCreate(capacity.staticBodyCount + capacity.dynamicBodyCount);

    const physicsWorld: WorldState = {
        ecsState: world,
        broadPhase: createBroadPhase(world, capacity, worldId),
        bodyUserData: [],
        bodyNames: [],
        jointUserData: [],
        shapeGeometry: [],
        shapeUserData: [],
        shapeNames: [],
        shapeDefInput: defaultShapeDef(),
        meshDatabase: new Map(),
        heightFieldDatabase: new Map(),
        compoundDatabase: new Map(),
        geometryDirty: false,
        geometryUploadCount: 0,
        manifoldStore: createManifoldStore(world, worldId),
        bodyStore: createBodyStore(world, worldId),
        shapeStore: createShapeStore(world, worldId),
        sensors: [],
        sensorCount: 0,
        queryColumns: null,
        bodyMoveCount: 0,
        sensorBeginEvents: [],
        contactBeginEvents: [],
        sensorEndEvents: [[], []],
        contactEndEvents: [[], []],
        contactHitEvents: [],
        jointEvents: [],
        jointEventCount: 0,
        jointEventUserData: [],
        endEventArrayIndex: 0,
        stepIndex: 0,
        stepContext: null,
        profile: createStepProfile(),
        gravity: { ...def.gravity },
        hitEventThreshold: def.hitEventThreshold,
        restitutionThreshold: def.restitutionThreshold,
        maxLinearSpeed: def.maximumLinearSpeed,
        contactSpeed: def.contactSpeed,
        contactHertz: def.contactHertz,
        contactDampingRatio: def.contactDampingRatio,
        contactRecycleDistance: CONTACT_RECYCLE_DISTANCE,
        frictionCallback: def.frictionCallback ?? defaultFrictionCallback,
        restitutionCallback: def.restitutionCallback ?? defaultRestitutionCallback,
        generation,
        maxCapacity: capacity,
        invH: 0,
        invDt: 0,
        worldId,
        userData: def.userData,
        enableSleep: def.enableSleep,
        locked: false,
        enableWarmStarting: true,
        enableContinuous: def.enableContinuous,
        enableSpeculative: true,
        inUse: true,
    };

    // Wire the broad store's back-reference so a resident-region grow can refresh the sibling stores a
    // `memory.grow` detaches (the store is created before the world literal, so it can't be passed in).
    physicsWorld.broadPhase.store.world = physicsWorld;

    // Create the three permanent sets in order so their ids land 0 (static), 1 (disabled), 2 (awake).
    for (let i = 0; i < 3; ++i) {
        createSolverSet(physicsWorld);
    }

    return physicsWorld;
}

/** Create a simulation world (b3CreateWorld). @returns its id. */
export function createWorld(
    world: import("../../../engine").World | undefined,
    def: WorldDef,
): WorldId {
    const owner = kernel(world);
    let worldId = -1;
    for (let i = 0; i < MAX_WORLDS; ++i) {
        const w = worlds[i];
        if (w === undefined || w.inUse === false) {
            worldId = i;
            break;
        }
    }
    if (worldId === -1) {
        throw new Error(`physics: B3_MAX_WORLDS of ${MAX_WORLDS} exceeded`);
    }

    owner.bodySetActiveWorld(worldId);
    owner.residentResetWorld(worldId);

    const generation = worlds[worldId]?.generation ?? 0;
    const physicsWorld = makeWorldState(world, def, worldId, generation);
    worlds[worldId] = physicsWorld;

    return { index1: worldId + 1, generation };
}

/** @returns the live world state for an id, or undefined if the id is stale. */
export function getWorld(id: WorldId): WorldState | undefined {
    const i = id.index1 - 1;
    if (i < 0 || i >= MAX_WORLDS) {
        return undefined;
    }
    const w = worlds[i];
    if (w === undefined || w.worldId !== i || w.generation !== id.generation) {
        return undefined;
    }
    return w;
}

/** @returns whether a world id references a live world (b3World_IsValid). */
export function worldIsValid(id: WorldId): boolean {
    return getWorld(id) !== undefined;
}

/** Destroy a world and everything in it (b3DestroyWorld). */
export function destroyWorld(world: WorldState): void {
    world.locked = true;

    // Release every live shape's allocations (drops all hull references).
    for (let i = 0; i < world.shapeGeometry.length; ++i) {
        if (kernel(world.ecsState).shapeAlive(world.worldId, i)) {
            destroyShapeAllocations(world, i);
        }
    }

    // Every shape released its hull reference, so the database must be empty.
    if (kernel(world.ecsState).hullDatabaseCount(world.worldId) !== 0) {
        throw new Error("physics: hull database not empty at world destroy");
    }

    // Destroy live solver sets (GC reclaims the rest).
    for (let i = 0; i < solverSetCount(world); ++i) {
        if (solverSetIndex(world, i) !== -1) {
            destroySolverSet(world, i);
        }
    }

    // Wipe but preserve+bump generation so stale ids to this (possibly recycled) slot are detected.
    const generation = world.generation;
    kernel(world.ecsState).bodyResetWorld(world.worldId);
    kernel(world.ecsState).shapeResetWorld(world.worldId);
    kernel(world.ecsState).residentResetWorld(world.worldId);
    world.geometryUploadScratch = undefined;
    world.inUse = false;
    world.worldId = 0;
    world.generation = (generation + 1) & 0xffff;
}

/** @returns entity counts for a world (b3World_GetCounters). */
export function worldCounters(world: WorldState): Counters {
    return {
        bodyCount: kernel(world.ecsState).bodyCount(world.worldId),
        shapeCount: kernel(world.ecsState).shapeCount(world.worldId),
        contactCount: contactCount(world),
        jointCount: jointCount(world),
        islandCount: islandKernel(world).islandCount(),
    };
}

/** @returns a copy of the last step's phase timings (b3World_GetProfile). */
export function worldProfile(world: WorldState): StepProfile {
    return { ...world.profile };
}
