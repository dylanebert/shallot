// The simulation world: the root that owns every entity pool and the broad-phase. Ported from
// Box3D's physics_world.c (Erin Catto, MIT). Each entity type has an id pool paired with a sparse
// array of records; the hot payload lives in solver sets. Worlds live in a fixed registry so a
// stale world id (to a destroyed, possibly recycled, world) is detected by a generation mismatch.
//
// Stage 7 (lifecycle) ports create/destroy/validity/counters and the hull database. The step, the
// constraint graph, sensors, recording, and threading are later stages; their world state is added
// as those stages land. fround discipline per the README.

import type { Body } from "./body";
import { type BodyStore, createBodyStore, releaseResident } from "./bodycolumns";
import { type BroadPhase, createBroadPhase } from "./broadphase";
import { type Contact, initializeContactRegisters } from "./contact";
import type { StepContext } from "./contactsolver";
import { CONTACT_RECYCLE_DISTANCE } from "./core";
import { createFatAabbStore, type FatAabbStore } from "./fataabbcolumns";
import { type ConstraintGraph, createGraph } from "./graph";
import type { HullData } from "./hull";
import { allocId, createIdPool, type EntityId, type IdPool, idCount } from "./ids";
import type { Island } from "./island";
import type { Joint } from "./joint";
import { createManifoldStore, type ManifoldStore } from "./manifoldstore";
import { f32, froundConfig, maxf, type Vec3, type WorldTransform } from "./math";
import { newProfile, type Profile } from "./profile";
import type { Sensor, SensorBeginTouchEvent } from "./sensor";
import type { Shape } from "./shape";
import { destroyShapeAllocations } from "./shape";
import { createShapeStore, type ShapeStore } from "./shapecolumns";
import { destroySolverSet, emptySolverSet, type SolverSet } from "./solverset";
import type { Capacity, MixCallback, WorldDef } from "./types";

/** Maximum concurrent worlds (B3_MAX_WORLDS). */
export const MAX_WORLDS = 128;

/** An opaque world handle (b3WorldId). */
export type WorldId = { index1: number; generation: number };

/**
 * A contact begin- or end-touch event (b3ContactBeginTouchEvent / b3ContactEndTouchEvent). The ids
 * are resolved to public {@link Shape}/{@link Contact} handles at getter time; end events read from
 * the previous double buffer, so they survive one step.
 */
export type ContactTouchEvent = { shapeIdA: EntityId; shapeIdB: EntityId; contactId: EntityId };

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

/**
 * A body move event (b3BodyMoveEvent): a body that moved this step. Written into a reused pool in
 * finalize (zero steady-state allocation); `transform`/`bodyId`/`generation` are corrected in place
 * by CCD and `fellAsleep` is patched by the sleep path. Only the first `bodyMoveCount` are valid.
 */
export type BodyMoveEvent = {
    bodyId: number;
    generation: number;
    transform: WorldTransform;
    userData: unknown;
    fellAsleep: boolean;
};

/** A joint event (b3JointEvent): an awake joint whose force/torque exceeded its threshold. */
export type JointEvent = { jointId: EntityId; userData: unknown };

/** Copy a world transform into a pooled move event's transform in place (no allocation). */
export function setMoveTransform(move: BodyMoveEvent, t: WorldTransform): void {
    move.transform.p.x = t.p.x;
    move.transform.p.y = t.p.y;
    move.transform.p.z = t.p.z;
    move.transform.q.v.x = t.q.v.x;
    move.transform.q.v.y = t.q.v.y;
    move.transform.q.v.z = t.q.v.z;
    move.transform.q.s = t.q.s;
}

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
    broadPhase: BroadPhase;
    constraintGraph: ConstraintGraph;

    bodyIdPool: IdPool;
    bodies: Body[];

    solverSetIdPool: IdPool;
    solverSets: SolverSet[];

    jointIdPool: IdPool;
    joints: Joint[];

    contactIdPool: IdPool;
    contacts: Contact[];
    // Incremental partition of the awake contacts collide processes each step, maintained on the
    // contact create/destroy + body wake/sleep/transfer events (contact.ts, solverset.ts) instead of
    // re-gathered per step. `awakeRecycleContacts`: dynamic-dynamic direct-convex, both bodies awake —
    // the batched kernel recycle pass. `awakeOtherContacts`: the rest (static/mesh/sleeping-partner) —
    // the TS per-contact narrowphase walk. Order-free; a contact is in at most one (contact.collideKind).
    awakeRecycleContacts: number[];
    awakeOtherContacts: number[];

    islandIdPool: IdPool;
    islands: Island[];

    shapeIdPool: IdPool;
    shapes: Shape[];

    // Reference-counted store of shared hull data keyed by content hash (b3HullMap).
    hullDatabase: Map<number, { hull: HullData; refCount: number }>;
    // Set when the hull set changes; the next step re-uploads the kernel's static geometry columns.
    geometryDirty: boolean;
    // Persistent contact-manifold columns (warm-start state, column-resident): the allocator + wasm
    // region for the manifolds keyed by contactId. Slots are tracked on contact create/destroy.
    manifoldStore: ManifoldStore;
    // Resident body-state columns (velocity/delta/flags of awake bodies), held across steps in the
    // body region. The awake set's `bodyStates` are offset-backed views over this store (bodycolumns.ts).
    bodyStore: BodyStore;
    // Resident fat-AABB column (one enlarged broad-phase AABB per shape), held across steps so the
    // in-kernel recycle overlap test + finalize escape test read it without a per-step marshal. Every TS
    // site that writes `shape.fatAABB` mirrors it here inline — no dirty set (fataabbcolumns.ts).
    fatAabbStore: FatAabbStore;
    // Resident shape column (type code + local geometry + nextShapeId, one record per shapeId), held
    // across steps so the in-kernel finalize refit walks a body's shape list without a marshal. Written
    // at shape create/destroy — no dirty set (shapecolumns.ts).
    shapeStore: ShapeStore;

    // Dense array of sensor overlap-tracking state, one per sensor shape (b3World.sensors).
    sensors: Sensor[];

    // Event buffers. End events are double-buffered so the user needn't flush every step. The body
    // move buffer is a reused pool grown but never shrunk; bodyMoveCount is the valid prefix length.
    bodyMoveEvents: BodyMoveEvent[];
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

    // Per-step phase timings (b3World.profile), zeroed at the top of each step.
    profile: Profile;

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

function makeWorldState(def: WorldDef, worldId: number, generation: number): WorldState {
    // Round every user float to f32 once at ingress; the C def is f32, so an f64 scalar would feed the
    // solver an extra bit and break bit-exact parity. Callbacks/capacity/bigints pass through.
    def = froundConfig(def);
    const capacity = makeCapacity(def.capacity);

    const world: WorldState = {
        broadPhase: createBroadPhase(capacity),
        constraintGraph: createGraph(capacity.staticBodyCount + capacity.dynamicBodyCount),
        bodyIdPool: createIdPool(),
        bodies: [],
        solverSetIdPool: createIdPool(),
        solverSets: [],
        jointIdPool: createIdPool(),
        joints: [],
        contactIdPool: createIdPool(),
        contacts: [],
        awakeRecycleContacts: [],
        awakeOtherContacts: [],
        islandIdPool: createIdPool(),
        islands: [],
        shapeIdPool: createIdPool(),
        shapes: [],
        hullDatabase: new Map(),
        geometryDirty: false,
        manifoldStore: createManifoldStore(),
        bodyStore: createBodyStore(),
        fatAabbStore: createFatAabbStore(),
        shapeStore: createShapeStore(),
        sensors: [],
        bodyMoveEvents: [],
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
        profile: newProfile(),
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
    world.broadPhase.store.world = world;

    // Create the three permanent sets in order so their ids land 0 (static), 1 (disabled), 2 (awake).
    for (let i = 0; i < 3; ++i) {
        const set = emptySolverSet();
        set.setIndex = allocId(world.solverSetIdPool);
        world.solverSets.push(set);
    }

    return world;
}

/** Create a simulation world (b3CreateWorld). @returns its id. */
export function createWorld(def: WorldDef): WorldId {
    let worldId = -1;
    for (let i = 0; i < MAX_WORLDS; ++i) {
        const w = worlds[i];
        if (w === undefined || w.inUse === false) {
            worldId = i;
            break;
        }
    }
    if (worldId === -1) {
        throw new Error(`tumble: B3_MAX_WORLDS of ${MAX_WORLDS} exceeded`);
    }

    initializeContactRegisters();

    const generation = worlds[worldId]?.generation ?? 0;
    const world = makeWorldState(def, worldId, generation);
    worlds[worldId] = world;

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

    // Release the shared resident body region so a later world can claim it without eviction.
    releaseResident(world);

    // Release every live shape's allocations (drops all hull references).
    for (let i = 0; i < world.shapes.length; ++i) {
        if (world.shapes[i].id !== -1) {
            destroyShapeAllocations(world, world.shapes[i]);
        }
    }

    // Every shape released its hull reference, so the database must be empty.
    if (world.hullDatabase.size !== 0) {
        throw new Error("tumble: hull database not empty at world destroy");
    }

    // Destroy live solver sets (GC reclaims the rest).
    for (let i = 0; i < world.solverSets.length; ++i) {
        if (world.solverSets[i].setIndex !== -1) {
            destroySolverSet(world, i);
        }
    }

    // Wipe but preserve+bump generation so stale ids to this (possibly recycled) slot are detected.
    const generation = world.generation;
    world.inUse = false;
    world.worldId = 0;
    world.generation = (generation + 1) & 0xffff;
}

/** @returns entity counts for a world (b3World_GetCounters). */
export function worldCounters(world: WorldState): Counters {
    return {
        bodyCount: idCount(world.bodyIdPool),
        shapeCount: idCount(world.shapeIdPool),
        contactCount: idCount(world.contactIdPool),
        jointCount: idCount(world.jointIdPool),
        islandCount: idCount(world.islandIdPool),
    };
}

/** @returns a copy of the last step's phase timings (b3World_GetProfile). */
export function worldProfile(world: WorldState): Profile {
    return { ...world.profile };
}
