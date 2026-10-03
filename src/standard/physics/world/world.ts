// The simulation world: the root that owns every entity pool and the broad-phase. Ported from
// Box3D's physics_world.c (Erin Catto, MIT). Each entity type has an id pool paired with a sparse
// array of records; the hot payload lives in solver sets. Worlds live in a fixed registry so a
// stale world id (to a destroyed, possibly recycled, world) is detected by a generation mismatch.
//
// fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).

import type { BodyFilters } from "../collision/bodyfilter";
import { type BroadPhase, createBroadPhase } from "../collision/broadphase";
import { type Contact, initializeContactRegisters } from "../collision/contact";
import { createManifoldStore, type ManifoldStore } from "../collision/manifoldstore";
import { CONTACT_RECYCLE_DISTANCE } from "../common/constants";
import { allocId, createIdPool, type EntityId, type IdPool, idCount } from "../common/ids";
import { f32, froundConfig, maxf, type Vec3 } from "../common/math";
import type { Capacity, MixCallback, WorldDef } from "../common/types";
import { type BodyStore, createBodyStore } from "../kernel/bodycolumns";
import { kernel } from "../kernel/kernel";
import type { QueryColumns } from "../kernel/querycolumns";
import { createShapeStore, type ShapeStore } from "../kernel/shapecolumns";
import { guardViews } from "../kernel/views";
import type { CompoundData } from "../shapes/compound";
import type { HeightFieldData } from "../shapes/heightfield";
import type { HullData } from "../shapes/hull";
import type { MeshData } from "../shapes/mesh";
import type { Shape } from "../shapes/shape";
import { destroyShapeAllocations } from "../shapes/shape";
import type { StepContext } from "../solver/contactsolver";
import { type ConstraintGraph, createGraph } from "../solver/graph";
import type { Joint } from "../solver/joint";
import type { Body } from "./body";
import type { Island } from "./island";
import { createStepProfile, type StepProfile } from "./profile";
import type { Sensor, SensorBeginTouchEvent } from "./sensor";
import { destroySolverSet, emptySolverSet, type SolverSet } from "./solverset";

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

/** A joint event (b3JointEvent): an awake joint whose force/torque exceeded its threshold. */
export type JointEvent = { jointId: EntityId; userData: unknown };

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
    bodyFilters: BodyFilters;
    constraintGraph: ConstraintGraph;

    /** Public body records are the authoring/handle bridge; lifecycle lives in wasm. */
    bodies: Body[];

    solverSetIdPool: IdPool;
    solverSets: SolverSet[];

    jointIdPool: IdPool;
    joints: Joint[];

    contactIdPool: IdPool;
    contacts: Contact[];
    // Awake contacts collide processes each step, maintained on the
    // contact create/destroy + body wake/sleep/transfer events (contact.ts, solverset.ts) instead of
    // re-gathered per step. Order-free; state transitions are processed in contact-id order.
    awakeContacts: number[];

    islandIdPool: IdPool;
    islands: Island[];

    /** Public shape authoring records and handle bridge; slot lifecycle is kernel-owned. */
    shapes: Shape[];

    // Reference-counted store of shared hull data keyed by content hash (b3HullMap).
    hullDatabase: Map<number, { hull: HullData; refCount: number }>;
    meshDatabase: Map<MeshData, GeometryRecord>;
    heightFieldDatabase: Map<HeightFieldData, GeometryRecord>;
    compoundDatabase: Map<CompoundData, GeometryRecord>;
    // Set when geometry data enters or leaves the databases, or residency/region placement changes.
    geometryDirty: boolean;
    /** Number of complete geometry uploads since world creation. */
    geometryUploadCount: number;
    // Persistent contact-manifold columns (warm-start state, column-resident): the allocator + wasm
    // region for the manifolds keyed by contactId. Slots are tracked on contact create/destroy.
    manifoldStore: ManifoldStore;
    // Resident body-state columns (velocity/delta/flags of awake bodies), held across steps in the
    // body region. The awake set's `bodyStates` are offset-backed views over this store (bodycolumns.ts).
    bodyStore: BodyStore;
    // Resident shape column (type code + local geometry + nextShapeId, one record per shapeId), held
    // across steps so the in-kernel finalize refit walks a body's shape list without a marshal. Written
    // at shape create/destroy — no dirty set (shapecolumns.ts).
    shapeStore: ShapeStore;

    // Dense array of sensor overlap-tracking state, one per sensor shape (b3World.sensors).
    sensors: Sensor[];
    queryColumns: QueryColumns | null;

    // Event buffers. End events are double-buffered so the user needn't flush every step. Kernel
    // finalization owns the retained body move records; bodyMoveCount is their valid prefix length.
    bodyMoveCount: number;
    sensorBeginEvents: SensorBeginTouchEvent[];
    contactBeginEvents: ContactTouchEvent[];
    sensorEndEvents: [SensorEndTouchEvent[], SensorEndTouchEvent[]];
    contactEndEvents: [ContactTouchEvent[], ContactTouchEvent[]];
    contactHitEvents: ContactHitEvent[];
    jointEvents: JointEvent[];
    endEventArrayIndex: number;

    stepIndex: number;
    splitIslandId: number;

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
const defaultFrictionCallback: MixCallback = (a, _idA, b, _idB) => f32(Math.sqrt(f32(a * b)));

/** Default restitution mixing: the larger of the two (b3DefaultRestitutionCallback). */
const defaultRestitutionCallback: MixCallback = (a, _idA, b, _idB) => maxf(a, b);

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
export function addHullToDatabase(world: WorldState, src: HullData): HullData {
    const entry = world.hullDatabase.get(src.hash);
    if (entry !== undefined) {
        entry.refCount += 1;
        return entry.hull;
    }
    world.hullDatabase.set(src.hash, { hull: src, refCount: 1 });
    // The hull set changed: flag the kernel's static geometry columns for re-upload at the next step
    // (deferred so hull creation never triggers a main-thread wasm instantiate before `init()`).
    world.geometryDirty = true;
    return src;
}

/** Release a hull reference, dropping the shared copy when the last shape lets go (b3RemoveHullFromDatabase). */
export function removeHullFromDatabase(world: WorldState, data: HullData): void {
    const entry = world.hullDatabase.get(data.hash);
    if (entry === undefined) {
        return;
    }
    entry.refCount -= 1;
    if (entry.refCount === 0) {
        world.hullDatabase.delete(data.hash);
        // The hull set changed: re-upload the remaining hulls (compacting geo indices) at the next step.
        world.geometryDirty = true;
    }
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

    const physicsWorld: WorldState = {
        ecsState: world,
        broadPhase: createBroadPhase(world, capacity, worldId),
        bodyFilters: { data: new Uint32Array(0), capacity: 16 },
        constraintGraph: createGraph(capacity.staticBodyCount + capacity.dynamicBodyCount),
        bodies: [],
        solverSetIdPool: createIdPool(),
        solverSets: [],
        jointIdPool: createIdPool(),
        joints: [],
        contactIdPool: createIdPool(),
        contacts: [],
        awakeContacts: [],
        islandIdPool: createIdPool(),
        islands: [],
        shapes: [],
        hullDatabase: new Map(),
        meshDatabase: new Map(),
        heightFieldDatabase: new Map(),
        compoundDatabase: new Map(),
        geometryDirty: false,
        geometryUploadCount: 0,
        manifoldStore: createManifoldStore(world, worldId),
        bodyStore: createBodyStore(world, worldId),
        shapeStore: createShapeStore(world, worldId),
        sensors: [],
        queryColumns: null,
        bodyMoveCount: 0,
        sensorBeginEvents: [],
        contactBeginEvents: [],
        sensorEndEvents: [[], []],
        contactEndEvents: [[], []],
        contactHitEvents: [],
        jointEvents: [],
        endEventArrayIndex: 0,
        stepIndex: 0,
        splitIslandId: -1,
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
    guardViews(physicsWorld.bodyFilters, physicsWorld.broadPhase.store);

    // Create the three permanent sets in order so their ids land 0 (static), 1 (disabled), 2 (awake).
    for (let i = 0; i < 3; ++i) {
        const set = emptySolverSet();
        set.setIndex = allocId(physicsWorld.solverSetIdPool);
        physicsWorld.solverSets.push(set);
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

    initializeContactRegisters();
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
    for (let i = 0; i < world.shapes.length; ++i) {
        if (world.shapes[i].id !== -1) {
            destroyShapeAllocations(world, world.shapes[i]);
        }
    }

    // Every shape released its hull reference, so the database must be empty.
    if (world.hullDatabase.size !== 0) {
        throw new Error("physics: hull database not empty at world destroy");
    }

    // Destroy live solver sets (GC reclaims the rest).
    for (let i = 0; i < world.solverSets.length; ++i) {
        if (world.solverSets[i].setIndex !== -1) {
            destroySolverSet(world, i);
        }
    }

    // Wipe but preserve+bump generation so stale ids to this (possibly recycled) slot are detected.
    const generation = world.generation;
    kernel(world.ecsState).bodyResetWorld(world.worldId);
    kernel(world.ecsState).shapeResetWorld(world.worldId);
    kernel(world.ecsState).materialResetWorld(world.worldId);
    kernel(world.ecsState).residentResetWorld(world.worldId);
    world.inUse = false;
    world.worldId = 0;
    world.generation = (generation + 1) & 0xffff;
}

/** @returns entity counts for a world (b3World_GetCounters). */
export function worldCounters(world: WorldState): Counters {
    return {
        bodyCount: kernel(world.ecsState).bodyCount(world.worldId),
        shapeCount: kernel(world.ecsState).shapeCount(world.worldId),
        contactCount: idCount(world.contactIdPool),
        jointCount: idCount(world.jointIdPool),
        islandCount: idCount(world.islandIdPool),
    };
}

/** @returns a copy of the last step's phase timings (b3World_GetProfile). */
export function worldProfile(world: WorldState): StepProfile {
    return { ...world.profile };
}
