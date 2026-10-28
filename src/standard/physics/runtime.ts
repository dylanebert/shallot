/// <reference types="@webgpu/types" />

import { Body, BodyType, PhysicsPlugin, Shape as ShapeComponent } from "../../core/physics";
import {
    claimTransformPose,
    GlobalTransform,
    releaseTransformPose,
    resetInterpolation,
    Transform,
} from "../../core/transform";
import {
    type EntityRef,
    type Plugin,
    type System,
    sampleDiagnosticTime,
    Time,
    type TypedArray,
    type World,
} from "../../engine";
import { reportStartup } from "../../engine/runtime";
import { clamp } from "../../engine/utils";
import type { AABB, Pos, Quat, Vec3, WorldState, WorldTransform } from "./api";
import { init, type PhysicsSnapshot, shutdown } from "./api";
import {
    restore as restoreWorld,
    snapshotBindings,
    snapshot as snapshotWorld,
} from "./api/snapshot";
import { bodyBinding, jointBindings, shapeBinding } from "./authoring";
import { BodyDependencies } from "./body-dependencies";
import { syncBodyFields } from "./body-sync";
import type { ShapeProxy } from "./collision/distance";
import { DEFAULT_MASK_BITS, SetType } from "./common/constants";
import type { EntityId } from "./common/ids";
import { PhysicsCreationOrder, recordPhysicsMembership } from "./creation-order";
import { jointIsValid, getJointLinearSeparation as jointLinearSeparation } from "./solver/joint";

const f32 = Math.fround;

import { defaultWorldDef, type QueryFilter, type WorldDef } from "./common/types";
import { EidColumn } from "./eid-column";
import {
    clearPhysicsEvents,
    type PhysicsContactEvent,
    type PhysicsEventRecords,
    PhysicsEvents,
    type PhysicsHitEvent,
    type PhysicsJointEvent,
    type PhysicsSensorEvent,
} from "./events";
import {
    captureFieldCandidate,
    captureFieldCandidates,
    clearFieldCandidates,
    createFieldCandidates,
    dropFieldCandidate,
    type FieldCandidates,
    markFieldCandidate,
    requeueFieldCandidate,
} from "./field-candidates";
import {
    captureGeometryBuilds,
    type GeometryBuilds,
    restoreGeometryBuilds,
} from "./geometry-builds";
import {
    type ConstraintCache,
    type ConstraintIds,
    captureConstraints,
    createConstraintCache,
    markBodyJointCandidates,
    removeAuthoredJoint,
    resetConstraints,
    restoreConstraints,
    syncJoints,
} from "./joints";
import { readSimTransform } from "./kernel/bodycolumns";
import { BodyField, bodyField } from "./kernel/bodyrecords";
import { EventKind, eventCount } from "./kernel/eventbuffers";
import { readJointEventUserData } from "./kernel/jointcolumns";
import {
    artifact,
    kernel,
    kernelState,
    type QueryCallback,
    rethrowQueryError,
    setQueryCallback,
    threads,
    writeQueryReturn,
} from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import { readShapeAabb } from "./kernel/shapecolumns";
import { ShapeField, shapeField } from "./kernel/shaperecords";
import {
    setBodyCount,
    solverSetCount,
    solverSetIndex,
    wakeSolverSet,
} from "./kernel/solversetcolumns";
import { marshalBody } from "./marshal";
import { marshalShape, shapeBodyScale } from "./marshal-shape";
import {
    captureShapeRecreationState,
    clearShapeRecreationStates,
    createShapeRecreationStates,
    forgetShapeRecreationState,
    restoreShapeRecreationState,
    type ShapeRecreationStates,
    shapeBodyScaleChanged,
    shapeNeedsRecreation,
    shapeRecreationState,
    syncShapeFields,
} from "./shape-sync";
import { destroyShape, shapeIsValid } from "./shapes/shape";
import { step as stepWorld } from "./solver/step";
import {
    bodyApplyAngularImpulse,
    bodyApplyForce,
    bodyApplyForceToCenter,
    bodyApplyLinearImpulse,
    bodyApplyLinearImpulseToCenter,
    bodyApplyTorque,
    bodyGetLinearVelocity,
    getMassData as bodyGetMassData,
    bodyGetTransform,
    bodyGetType,
    bodyIsAwake,
    bodyIsValid,
    bodySetAngularVelocity,
    bodySetAwake,
    bodySetLinearVelocity,
    setMassData as bodySetMassData,
    bodySetTargetTransform,
    bodySetTransform,
    destroyBody,
} from "./world/body";
import { type DebugDraw, worldDraw } from "./world/draw";
import { hashWorldState } from "./world/hash";
import { PROFILE_FIELDS, readStepProfile } from "./world/profile";
import {
    bindShapeEntity,
    clearShapeEntities,
    liveShapeEntity,
    liveShapeGeneration,
    pruneShapeEntities,
    retireShapeEntity,
    type ShapeEntity,
    shapeEntityAt,
} from "./world/shape-entities";
import {
    createWorld,
    destroyWorld,
    getWorld,
    worldSetContactRecycleDistance,
    worldSetContactTuning,
    worldSetGravity,
    worldSetHitEventThreshold,
    worldSetMaximumLinearSpeed,
    worldSetRestitutionThreshold,
    worldStateIsValid,
} from "./world/world";
import {
    copyPhysicsWorldDefinition,
    PhysicsWorldDefinition,
    type PhysicsWorldDefinitionConfig,
} from "./world-definition";

const fixedDeltaTime = Time.FIXED_DT;

export interface PhysicsCounters {
    bodiesVisited: number;
    /** Shapes the last fixed sync visited, each creation or field reconciliation counting once. */
    shapesVisited: number;
    /** Hull builds since the World was built. */
    hullsBuilt: number;
    bytesUploaded: number;
}

interface PhysicsRuntime {
    initialized: boolean;
    bodyCandidates: FieldCandidates;
    poseCandidates: FieldCandidates;
    shapeCandidates: FieldCandidates;
    creationCandidates: FieldCandidates;
    warnedBodyFields: Set<number>;
    shapes: Map<number, EntityId>;
    /** Each attached shape's body eid. */
    shapeBodies: EidColumn;
    shapeDependencies: BodyDependencies;
    queryContexts: QueryContext[];
    queryDepth: number;
    queryCastCallback: QueryCallback;
    queryOverlapCallback: QueryCallback;
    /** Each attached shape's EntityRef at its first attachment. */
    shapeStamps: EidColumn;
    shapeRecreationState: ShapeRecreationStates;
    shapeRemovalMass: Map<number, boolean>;
    warnedShapes: Set<number>;
    physicsWorld: WorldState | null;
    bodies: Map<number, BoundBodyId>;
    constraints: ConstraintCache;
    stale: StaleScan;
    counters: Omit<PhysicsCounters, "hullsBuilt">;
    // eids whose Body membership, or a Body's Transform, changed since the last sync, as 32-bit words
    changed: Uint32Array;
    anyChanged: boolean;
    // only warm sets this to make the next sync walk every Body
    full: boolean;
    /** The change tick of the last Transform.scale capture. */
    scaleRun: number;
    observing: boolean;
    // each hit record's material-id words (A low, A high, B low, B high), so fillHitRecords rebuilds a
    // record's bigint only when its words change
    hitMaterialWords: Uint32Array;
}

export interface PhysicsQueryHit {
    shape: number;
    body: number;
    point: Pos;
    normal: Vec3;
    fraction: number;
    userMaterialId: bigint;
    triangleIndex: number;
    childIndex: number;
}

export type PhysicsCastCallback = (hit: PhysicsQueryHit) => number;
export type PhysicsOverlapCallback = (shape: number, body: number) => boolean;

interface QueryContext {
    world: World | null;
    shapeEid: number;
    bodyEid: number;
    originX: number;
    originY: number;
    originZ: number;
    callback: PhysicsCastCallback | null;
    overlap: PhysicsOverlapCallback | null;
    output: PhysicsQueryHit | null;
    hit: PhysicsQueryHit;
}

type BoundBodyId = EntityId & { stamp: EntityRef };

const physicsRuntimeKey = { create: newRuntime };

function newRuntime(): PhysicsRuntime {
    const runtime: PhysicsRuntime = {
        initialized: false,
        bodyCandidates: createFieldCandidates(),
        poseCandidates: createFieldCandidates(),
        shapeCandidates: createFieldCandidates(),
        creationCandidates: createFieldCandidates(),
        warnedBodyFields: new Set(),
        shapes: new Map(),
        shapeBodies: new EidColumn(),
        shapeDependencies: new BodyDependencies(),
        queryContexts: [newQueryContext()],
        queryDepth: 0,
        queryCastCallback: null!,
        queryOverlapCallback: null!,
        shapeStamps: new EidColumn(),
        shapeRecreationState: createShapeRecreationStates(),
        shapeRemovalMass: new Map(),
        warnedShapes: new Set(),
        physicsWorld: null,
        bodies: new Map(),
        constraints: createConstraintCache(),
        stale: { world: null, eids: [], count: 0 },
        counters: { bodiesVisited: 0, shapesVisited: 0, bytesUploaded: 0 },
        changed: new Uint32Array(64),
        anyChanged: false,
        full: true,
        scaleRun: 0,
        observing: false,
        hitMaterialWords: new Uint32Array(64),
    };
    runtime.queryCastCallback = castQueryCallback.bind(runtime);
    runtime.queryOverlapCallback = overlapQueryCallback.bind(runtime);
    return runtime;
}

function fillQueryHit(
    hit: PhysicsQueryHit,
    state: WorldState,
    shapeId: number,
    shapeEid: number,
    bodyEid: number,
    values: Float32Array,
    offset: number,
    originX: number,
    originY: number,
    originZ: number,
): void {
    hit.shape = shapeEid;
    hit.body = bodyEid;
    hit.point.x = originX + values[offset + 2];
    hit.point.y = originY + values[offset + 3];
    hit.point.z = originZ + values[offset + 4];
    hit.normal.x = values[offset + 5];
    hit.normal.y = values[offset + 6];
    hit.normal.z = values[offset + 7];
    hit.fraction = values[offset + 1];
    hit.triangleIndex = values[offset + 9];
    hit.childIndex = values[offset + 10];
    hit.userMaterialId = state.shapeStore.userMaterialIdAt(shapeId, values[offset + 11]);
}

function castQueryCallback(
    this: PhysicsRuntime,
    _kind: number,
    shapeId: number,
    data: number,
): number {
    const context = this.queryContexts[this.queryDepth - 1];
    const state = this.physicsWorld!;
    if (!queryShapeEntities(context.world!, this, shapeId, context)) {
        writeQueryReturn(state.ecsState, -1);
        return 1;
    }
    const hit = context.hit;
    const values = state.shapeStore.materialF;
    fillQueryHit(
        hit,
        state,
        shapeId,
        context.shapeEid,
        context.bodyEid,
        values,
        data >>> 2,
        context.originX,
        context.originY,
        context.originZ,
    );
    if (context.output) copyQueryHit(context.output, hit);
    writeQueryReturn(state.ecsState, context.output ? hit.fraction : context.callback!(hit));
    return 1;
}

function overlapQueryCallback(this: PhysicsRuntime, _kind: number, shapeId: number): number {
    const context = this.queryContexts[this.queryDepth - 1];
    if (!queryShapeEntities(context.world!, this, shapeId, context)) return 1;
    return Number(context.overlap!(context.shapeEid, context.bodyEid));
}

function queryShapeEntities(
    world: World,
    runtime: PhysicsRuntime,
    shapeId: number,
    context: QueryContext,
): boolean {
    const entities = runtime.physicsWorld!.shapeEntities;
    if (!liveShapeEntity(entities, shapeId, queryShapeEntity)) return false;
    const shapeEid = world.resolve(queryShapeEntity.shape);
    if (!shapeEid || !world.has(shapeEid, ShapeComponent)) return false;
    const bodyEid = world.resolve(queryShapeEntity.body);
    if (!bodyEid || !world.has(bodyEid, Body)) return false;
    const shape = runtime.shapes.get(shapeEid);
    if (
        !shape ||
        shape.index1 !== shapeId + 1 ||
        shape.generation !== liveShapeGeneration(entities, shapeId)
    )
        return false;
    context.shapeEid = shapeEid;
    context.bodyEid = bodyEid;
    return true;
}
const queryShapeEntity: ShapeEntity = { shape: 0 as EntityRef, body: 0 as EntityRef };

function newQueryContext(): QueryContext {
    return {
        world: null,
        shapeEid: 0,
        bodyEid: 0,
        originX: 0,
        originY: 0,
        originZ: 0,
        callback: null,
        output: null,
        overlap: null,
        hit: emptyQueryHit(),
    };
}

function copyQueryHit(out: PhysicsQueryHit, hit: PhysicsQueryHit): void {
    out.shape = hit.shape;
    out.body = hit.body;
    out.point.x = hit.point.x;
    out.point.y = hit.point.y;
    out.point.z = hit.point.z;
    out.normal.x = hit.normal.x;
    out.normal.y = hit.normal.y;
    out.normal.z = hit.normal.z;
    out.fraction = hit.fraction;
    out.userMaterialId = hit.userMaterialId;
    out.triangleIndex = hit.triangleIndex;
    out.childIndex = hit.childIndex;
}

/** A new hit record with every field zero, a miss, for `castRayClosest` and `castShapeClosest` to write into. */
export function emptyQueryHit(): PhysicsQueryHit {
    return {
        shape: 0,
        body: 0,
        point: { x: 0, y: 0, z: 0 },
        normal: { x: 0, y: 0, z: 0 },
        fraction: 0,
        userMaterialId: 0n,
        triangleIndex: 0,
        childIndex: 0,
    };
}

function clearQueryHit(out: PhysicsQueryHit): void {
    out.shape = 0;
    out.body = 0;
    out.point.x = 0;
    out.point.y = 0;
    out.point.z = 0;
    out.normal.x = 0;
    out.normal.y = 0;
    out.normal.z = 0;
    out.fraction = 0;
    out.userMaterialId = 0n;
    out.triangleIndex = 0;
    out.childIndex = 0;
}

function beginQuery(
    runtime: PhysicsRuntime,
    world: World,
    origin: Pos,
    callback: PhysicsCastCallback | null,
    overlap: PhysicsOverlapCallback | null,
    output: PhysicsQueryHit | null,
): QueryContext {
    const depth = runtime.queryDepth++;
    const context = (runtime.queryContexts[depth] ??= newQueryContext());
    context.world = world;
    context.originX = origin.x;
    context.originY = origin.y;
    context.originZ = origin.z;
    context.callback = callback;
    context.overlap = overlap;
    context.output = output;
    return context;
}

function endQuery(runtime: PhysicsRuntime): void {
    const context = runtime.queryContexts[--runtime.queryDepth];
    context.world = null;
    context.callback = null;
    context.overlap = null;
    context.output = null;
}

function rememberShapeEntity(
    world: World,
    runtime: PhysicsRuntime,
    shape: EntityId,
    shapeEid: number,
    bodyEid: number,
): void {
    bindShapeEntity(
        runtime.physicsWorld!.shapeEntities,
        shape.index1 - 1,
        shape.generation,
        world.ref(shapeEid),
        world.ref(bodyEid),
    );
}

function retireShape(runtime: PhysicsRuntime, shape: EntityId | undefined): void {
    const state = runtime.physicsWorld;
    if (shape && state) retireShapeEntity(state.shapeEntities, shape.index1 - 1, shape.generation);
}

function markChanged(runtime: PhysicsRuntime, eid: number): void {
    const word = eid >>> 5;
    if (word >= runtime.changed.length) {
        const grown = new Uint32Array(Math.max(word + 1, runtime.changed.length * 2));
        grown.set(runtime.changed);
        runtime.changed = grown;
    }
    runtime.changed[word] |= 1 << (eid & 31);
    runtime.anyChanged = true;
}

function runtimeFor(world: World): PhysicsRuntime {
    const runtime = world.resource(physicsRuntimeKey);
    if (!runtime.initialized)
        throw new Error("physics: StandardPhysicsPlugin is not initialized for this World");
    return runtime;
}

/** The solver body bound to eid's live entity; none for one added or destroyed since the last fixed sync. */
function boundBody(world: World, runtime: PhysicsRuntime, eid: number): BoundBodyId | undefined {
    const body = runtime.bodies.get(eid);
    return body && world.resolve(body.stamp) === eid ? body : undefined;
}

function writeGlobalTransform(
    world: World,
    eid: number,
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    vel: readonly [number, number, number],
): void {
    world.storage(GlobalTransform).translation.set(eid, pos[0], pos[1], pos[2]);
    world.storage(GlobalTransform).rotation.set(eid, quat[0], quat[1], quat[2], quat[3]);
    world.storage(GlobalTransform).linearVelocity.set(eid, vel[0], vel[1], vel[2]);
}

const spawnTransform: WorldTransform = {
    p: { x: 0, y: 0, z: 0 },
    q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
};
const spawnVelocity: Vec3 = { x: 0, y: 0, z: 0 };
function seedGlobalTransform(world: World, eid: number, state: WorldState, body: EntityId): void {
    const record = body.index1 - 1;
    const pose = bodyGetTransform(state, record, spawnTransform);
    bodyGetLinearVelocity(state, record, spawnVelocity);
    writeGlobalTransform(
        world,
        eid,
        [pose.p.x, pose.p.y, pose.p.z],
        [pose.q.v.x, pose.q.v.y, pose.q.v.z, pose.q.s],
        [spawnVelocity.x, spawnVelocity.y, spawnVelocity.z],
    );
}

function forget(world: World, runtime: PhysicsRuntime, eid: number): void {
    releaseTransformPose(world, eid);
    const dependencies = runtime.shapeDependencies;
    for (
        let shapeEid = dependencies.first(eid);
        shapeEid !== 0;
        shapeEid = dependencies.following(shapeEid)
    )
        if (runtime.shapeBodies.get(shapeEid) === eid)
            retireShape(runtime, runtime.shapes.get(shapeEid));
    const body = runtime.bodies.get(eid);
    const state = runtime.physicsWorld;
    if (body && state && bodyIsValid(state, body)) destroyBody(state, body.index1 - 1);
    runtime.bodies.delete(eid);
    for (
        let shapeEid = dependencies.first(eid);
        shapeEid !== 0;
        shapeEid = dependencies.following(shapeEid)
    )
        if (runtime.shapeBodies.get(shapeEid) === eid) {
            runtime.shapes.delete(shapeEid);
            runtime.shapeBodies.delete(shapeEid);
            runtime.shapeStamps.delete(shapeEid);
            forgetShapeRecreationState(runtime.shapeRecreationState, shapeEid);
            runtime.shapeRemovalMass.delete(shapeEid);
            runtime.warnedShapes.delete(shapeEid);
        }
    markBodyDependents(runtime, eid);
    for (let fieldIndex = 0; fieldIndex < 32; fieldIndex++)
        runtime.warnedBodyFields.delete(eid * 32 + fieldIndex);
}

function clearBodies(world: World, runtime: PhysicsRuntime): void {
    for (const eid of runtime.bodies.keys()) releaseTransformPose(world, eid);
    const state = runtime.physicsWorld;
    if (state)
        for (const body of runtime.bodies.values())
            if (bodyIsValid(state, body)) destroyBody(state, body.index1 - 1);
    runtime.bodies.clear();
    runtime.shapes.clear();
    runtime.shapeBodies.clear();
    if (runtime.physicsWorld) clearShapeEntities(runtime.physicsWorld.shapeEntities);
    runtime.shapeStamps.clear();
    clearShapeRecreationStates(runtime.shapeRecreationState);
    runtime.shapeRemovalMass.clear();
    clearFieldCandidates(runtime.poseCandidates);
    runtime.warnedShapes.clear();
    clearFieldCandidates(runtime.shapeCandidates);
    // The next captures latch every mark still set, as for a new World.
    runtime.poseCandidates.lastRun = 0;
    runtime.shapeCandidates.lastRun = 0;
    runtime.scaleRun = 0;
    runtime.shapeDependencies.clear();
    runtime.counters = { bodiesVisited: 0, shapesVisited: 0, bytesUploaded: 0 };
    runtime.warnedBodyFields.clear();
    runtime.full = true;
    resetConstraints(runtime.constraints);
}

/** Internal access to a world's Box3D state for the ECS bridge and solver-only tests. */

function stateIsValid(state: WorldState): boolean {
    return worldStateIsValid(state);
}

export function physicsState(world: World): WorldState | null {
    const state = runtimeFor(world).physicsWorld;
    return state && stateIsValid(state) ? state : null;
}

export function physicsBodyIndex(world: World, eid: number): number {
    const runtime = runtimeFor(world);
    const body = runtime.bodies.get(eid);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld || !stateIsValid(physicsWorld) || !world.has(eid, Body) || !body) {
        return -1;
    }
    const stamp = body.stamp;
    const generation = Math.floor(stamp / 2 ** 32);
    return stamp >>> 0 === eid &&
        world.matchesEntityGeneration(eid, generation) &&
        bodyIsValid(physicsWorld, body)
        ? body.index1 - 1
        : -1;
}

const queryZeroOrigin: Pos = { x: 0, y: 0, z: 0 };

/**
 * Cast a ray through the ECS World's Box3D shapes, reporting each hit's Shape and Body entities.
 * Return the new maximum fraction, usually `hit.fraction` to stop at the closest hit, or `-1` to skip.
 * The callback's hit record and nested vectors are reused and valid only until it returns.
 */
export function castRay(
    world: World,
    origin: Pos,
    translation: Vec3,
    callback: PhysicsCastCallback,
    filter?: QueryFilter,
): void {
    const runtime = runtimeFor(world);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) return;
    const state = physicsWorld;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.translation(translation);
    beginQuery(runtime, world, origin, callback, null, null);
    const previous = setQueryCallback(state.ecsState, runtime.queryCastCallback);
    try {
        k.worldQuery(state.worldId, 2, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
}

/**
 * Cast a ray and write its closest hit into caller-owned storage. Returns false on a miss; the
 * output is cleared on a miss. Make `out` once with `emptyQueryHit` and reuse it.
 */
export function castRayClosest(
    world: World,
    origin: Pos,
    translation: Vec3,
    out: PhysicsQueryHit,
    filter?: QueryFilter,
): boolean {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    if (!state) {
        clearQueryHit(out);
        return false;
    }
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.translation(translation);
    clearQueryHit(out);
    beginQuery(runtime, world, origin, null, null, out);
    const previous = setQueryCallback(state.ecsState, runtime.queryCastCallback);
    try {
        k.worldQuery(state.worldId, 2, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
    return out.shape !== 0;
}

/**
 * Cast a convex proxy through the ECS World's Box3D shapes, reporting each hit's Shape and Body entities.
 * Return the new maximum fraction, usually `hit.fraction` to stop at the closest hit, or `-1` to skip.
 * The callback's hit record and nested vectors are reused and valid only until it returns.
 */
export function castShape(
    world: World,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    callback: PhysicsCastCallback,
    filter?: QueryFilter,
): void {
    const runtime = runtimeFor(world);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) return;
    const state = physicsWorld;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.proxy(proxy);
    q.translation(translation);
    beginQuery(runtime, world, origin, callback, null, null);
    const previous = setQueryCallback(state.ecsState, runtime.queryCastCallback);
    try {
        k.worldQuery(state.worldId, 4, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
}

/**
 * Cast a convex proxy and write its closest hit into caller-owned storage. Returns false on a miss;
 * the output is cleared on a miss. Make `out` once with `emptyQueryHit` and reuse it.
 */
export function castShapeClosest(
    world: World,
    origin: Pos,
    proxy: ShapeProxy,
    translation: Vec3,
    out: PhysicsQueryHit,
    filter?: QueryFilter,
): boolean {
    const runtime = runtimeFor(world);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) {
        clearQueryHit(out);
        return false;
    }
    const state = physicsWorld;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.proxy(proxy);
    q.translation(translation);
    clearQueryHit(out);
    beginQuery(runtime, world, origin, null, null, out);
    const previous = setQueryCallback(state.ecsState, runtime.queryCastCallback);
    try {
        k.worldQuery(state.worldId, 4, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
    return out.shape !== 0;
}

/**
 * Report every ECS Shape and Body whose fat AABB overlaps `bounds`. Returning false stops this body-type tree;
 * the remaining trees are still visited.
 */
export function overlapAABB(
    world: World,
    bounds: AABB,
    callback: PhysicsOverlapCallback,
    filter?: QueryFilter,
): void {
    const runtime = runtimeFor(world);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) return;
    const state = physicsWorld;
    const q = queryColumns(state);
    const k = q.prepare(queryZeroOrigin, filter);
    q.bounds(bounds);
    beginQuery(runtime, world, queryZeroOrigin, null, callback, null);
    const previous = setQueryCallback(state.ecsState, runtime.queryOverlapCallback);
    try {
        k.worldQuery(state.worldId, 0, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
}

/**
 * Report every ECS Shape and Body overlapping a convex proxy. Returning false stops this body-type tree;
 * the remaining trees are still visited.
 */
export function overlapShape(
    world: World,
    origin: Pos,
    proxy: ShapeProxy,
    callback: PhysicsOverlapCallback,
    filter?: QueryFilter,
): void {
    const runtime = runtimeFor(world);
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) return;
    const state = physicsWorld;
    const q = queryColumns(state);
    const k = q.prepare(origin, filter);
    q.proxy(proxy);
    beginQuery(runtime, world, origin, null, callback, null);
    const previous = setQueryCallback(state.ecsState, runtime.queryOverlapCallback);
    try {
        k.worldQuery(state.worldId, 1, 1);
        rethrowQueryError(state.ecsState);
    } finally {
        setQueryCallback(state.ecsState, previous);
        endQuery(runtime);
    }
}

/** Walk Box3D's debug geometry for the selected shape, joint, bounds and mass options. */
export function drawPhysics(
    world: World,
    draw: DebugDraw,
    maskBits: bigint = DEFAULT_MASK_BITS,
): void {
    const physicsWorld = runtimeFor(world).physicsWorld;
    if (physicsWorld) worldDraw(physicsWorld, draw, maskBits);
}

// Body methods copy these reusable views synchronously.
const statePos: Pos = { x: 0, y: 0, z: 0 };
const stateQuat: Quat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const stateTarget: WorldTransform = { p: statePos, q: stateQuat };
const stateVector: Vec3 = { x: 0, y: 0, z: 0 };
const stateVelocity: Vec3 = { x: 0, y: 0, z: 0 };
const stateVelocityTuple: [number, number, number] = [0, 0, 0];

function stateTransform(
    position: readonly [number, number, number],
    rotation: readonly [number, number, number, number],
): WorldTransform {
    statePos.x = position[0];
    statePos.y = position[1];
    statePos.z = position[2];
    stateQuat.v.x = rotation[0];
    stateQuat.v.y = rotation[1];
    stateQuat.v.z = rotation[2];
    stateQuat.s = rotation[3];
    return stateTarget;
}
function publishBodyVelocity(world: World, eid: number, state: WorldState, body: EntityId): void {
    bodyGetLinearVelocity(state, body.index1 - 1, stateVelocity);
    world
        .storage(GlobalTransform)
        .linearVelocity.set(eid, stateVelocity.x, stateVelocity.y, stateVelocity.z);
}

/**
 * Drives a kinematic or dynamic body toward a world pose over `timeStep` seconds. Derives
 * linear velocity in meters per second and angular velocity in radians per second; `wake` wakes an
 * asleep body when the target motion is significant. The pose and linear velocity publish immediately.
 * Static bodies, and bodies added since the last fixed sync, are ignored.
 */
export function setTargetTransform(
    world: World,
    eid: number,
    position: readonly [number, number, number],
    rotation: readonly [number, number, number, number],
    timeStep = fixedDeltaTime,
    wake = true,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    const record = body.index1 - 1;
    if (bodyGetType(state, record) === BodyType.Static) return;
    bodySetTargetTransform(state, record, stateTransform(position, rotation), timeStep, wake);
    // Sub-threshold target motion leaves a body asleep without adopting its pose; do not publish it.
    if (!bodyIsAwake(state, record)) return;
    bodyGetLinearVelocity(state, record, stateVelocity);
    stateVelocityTuple[0] = stateVelocity.x;
    stateVelocityTuple[1] = stateVelocity.y;
    stateVelocityTuple[2] = stateVelocity.z;
    writeGlobalTransform(world, eid, position, rotation, stateVelocityTuple);
}

/**
 * Teleports a body in meters, preserving both velocities and discarding placement interpolation;
 * the pose publishes immediately. A body added since the last fixed sync spawns at this pose.
 */
export function setTransform(
    world: World,
    eid: number,
    position: readonly [number, number, number],
    rotation: readonly [number, number, number, number],
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (state && body) {
        const record = body.index1 - 1;
        bodySetTransform(state, record, stateTransform(position, rotation).p, stateQuat);
        bodyGetLinearVelocity(state, record, stateVelocity);
        stateVelocityTuple[0] = stateVelocity.x;
        stateVelocityTuple[1] = stateVelocity.y;
        stateVelocityTuple[2] = stateVelocity.z;
        writeGlobalTransform(world, eid, position, rotation, stateVelocityTuple);
    } else if (world.has(eid, Body)) {
        // Unbound: the body spawns at Transform's pose; publish it now, as a bound teleport does.
        const global = world.storage(GlobalTransform);
        global.translation.set(eid, position[0], position[1], position[2]);
        global.rotation.set(eid, rotation[0], rotation[1], rotation[2], rotation[3]);
    } else return;
    const pose = world.storage(Transform);
    pose.translation.set(eid, position[0], position[1], position[2]);
    pose.rotation.set(eid, rotation[0], rotation[1], rotation[2], rotation[3]);
    resetInterpolation(world, eid);
}

/** Sets center-of-mass velocity in meters per second; nonzero values wake the body. Box3D ignores static bodies; a body added since the last fixed sync ignores it and spawns with `Body.linearVelocity`. */
export function setLinearVelocity(world: World, eid: number, velocity: Vec3): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    const record = body.index1 - 1;
    stateVector.x = velocity.x;
    stateVector.y = velocity.y;
    stateVector.z = velocity.z;
    bodySetLinearVelocity(state, record, stateVector);
    if (bodyGetType(state, record) !== BodyType.Static)
        publishBodyVelocity(world, eid, state, body);
}

/** Sets angular velocity in radians per second; nonzero values wake the body and locked axes are zeroed. Box3D ignores static bodies; a body added since the last fixed sync ignores it and spawns with `Body.angularVelocity`. */
export function setAngularVelocity(world: World, eid: number, velocity: Vec3): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = velocity.x;
    stateVector.y = velocity.y;
    stateVector.z = velocity.z;
    bodySetAngularVelocity(state, body.index1 - 1, stateVector);
}

/** Accumulates wind force and torque for the shape's body for the next step, using its projected area,
 * relative shape velocity and air density of 1.225 kg/m³. Wind is world-space velocity in meters per second; drag opposes relative velocity and lift acts perpendicular to it.
 * maxSpeed caps relative speed for stability (typically 10 m/s or less).
 * Only spheres, capsules and hulls on enabled dynamic bodies respond. Sleeping bodies respond only when wake is true.
 * Shapes added since the last fixed sync ignore it. */
export function applyWind(
    world: World,
    eid: number,
    wind: Vec3,
    drag: number,
    lift: number,
    maxSpeed: number,
    wake: boolean,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const shape = runtime.shapes.get(eid);
    if (!state || !shape || !world.has(eid, ShapeComponent)) return;
    const slot = shape.index1 - 1;
    if (
        liveShapeGeneration(state.shapeEntities, slot) !== shape.generation ||
        world.resolve(state.shapeEntities.shapes[slot] as EntityRef) !== eid
    )
        return;
    kernel(state.ecsState).shapeApplyWind(
        state.worldId,
        shape.index1 - 1,
        wind.x,
        wind.y,
        wind.z,
        drag,
        lift,
        maxSpeed,
        wake ? 1 : 0,
    );
}

/** Accumulates world force in newtons at a world point for the next step; an off-center point adds torque. `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it. */
export function applyForce(world: World, eid: number, force: Vec3, point: Pos, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = force.x;
    stateVector.y = force.y;
    stateVector.z = force.z;
    statePos.x = point.x;
    statePos.y = point.y;
    statePos.z = point.z;
    bodyApplyForce(state, body.index1 - 1, stateVector, statePos, wake);
}

/** Accumulates world force in newtons at the center of mass for the next step; `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it. */
export function applyForceToCenter(world: World, eid: number, force: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = force.x;
    stateVector.y = force.y;
    stateVector.z = force.z;
    bodyApplyForceToCenter(state, body.index1 - 1, stateVector, wake);
}

/** Accumulates world torque in newton-meters for the next step; `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it. */
export function applyTorque(world: World, eid: number, torque: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = torque.x;
    stateVector.y = torque.y;
    stateVector.z = torque.z;
    bodyApplyTorque(state, body.index1 - 1, stateVector, wake);
}

/** Geometry-weighted radial impulse for dynamic spheres, capsules and hulls. Distances and position are in meters. */
export interface ExplosionDef {
    /** Filters shapes by their category bits. */
    maskBits: bigint;
    /** World-space center; coordinates must be finite in single precision. */
    position: Pos;
    /** Full-strength radius; must be finite in single precision and nonnegative. */
    radius: number;
    /** Distance beyond radius over which impulse falls to zero; must be finite in single precision and nonnegative. */
    falloff: number;
    /** Impulse per unit projected area; must be finite in single precision. Negative values implode. */
    impulsePerArea: number;
}

export function defaultExplosionDef(): ExplosionDef {
    return {
        maskBits: 0xffffffffffffffffn,
        position: { x: 0, y: 0, z: 0 },
        radius: 0,
        falloff: 0,
        impulsePerArea: 0,
    };
}

/** Applies an immediate radial impulse to dynamic spheres, capsules and hulls, waking affected bodies.
 * Bodies added since the last fixed sync are not considered. Shape.explosionScale weights each shape's projected area.
 * Throws RangeError if position, radius, falloff or impulsePerArea is not finite in single precision,
 * or if radius or falloff is negative. */
export function explode(world: World, def: ExplosionDef): void {
    if (!Number.isFinite(f32(def.radius)) || def.radius < 0)
        throw new RangeError("physics: explosion radius must be finite and nonnegative");
    if (!Number.isFinite(f32(def.falloff)) || def.falloff < 0)
        throw new RangeError("physics: explosion falloff must be finite and nonnegative");
    if (
        !Number.isFinite(f32(def.position.x)) ||
        !Number.isFinite(f32(def.position.y)) ||
        !Number.isFinite(f32(def.position.z))
    )
        throw new RangeError("physics: explosion position must be finite");
    if (!Number.isFinite(f32(def.impulsePerArea)))
        throw new RangeError("physics: explosion impulsePerArea must be finite");
    const state = runtimeFor(world).physicsWorld;
    if (!state) return;
    kernel(state.ecsState).worldExplode(
        state.worldId,
        Number((def.maskBits >> 32n) & 0xffffffffn),
        Number(def.maskBits & 0xffffffffn),
        def.position.x,
        def.position.y,
        def.position.z,
        def.radius,
        def.falloff,
        def.impulsePerArea,
    );
}

/** Applies a one-shot world impulse in newton-seconds at a world point; off-center impulses also change angular velocity. `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it and spawns with `Body.linearVelocity` and `Body.angularVelocity`. */
export function applyLinearImpulse(
    world: World,
    eid: number,
    impulse: Vec3,
    point: Pos,
    wake = true,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    statePos.x = point.x;
    statePos.y = point.y;
    statePos.z = point.z;
    bodyApplyLinearImpulse(state, body.index1 - 1, stateVector, statePos, wake);
}

/** Applies a one-shot center-of-mass impulse in newton-seconds; `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it and spawns with `Body.linearVelocity`. */
export function applyLinearImpulseToCenter(
    world: World,
    eid: number,
    impulse: Vec3,
    wake = true,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    bodyApplyLinearImpulseToCenter(state, body.index1 - 1, stateVector, wake);
}

/** Applies a one-shot angular impulse in kilogram-meter-squared per second; `wake` defaults true. Box3D ignores static and kinematic bodies; a body added since the last fixed sync ignores it and spawns with `Body.angularVelocity`. */
export function applyAngularImpulse(world: World, eid: number, impulse: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    bodyApplyAngularImpulse(state, body.index1 - 1, stateVector, wake);
}

/** Converts a world point in meters to the body's unscaled local frame.
 * Returns null for a body the solver does not hold: one added or destroyed since the last fixed sync.
 * The returned record belongs to the caller. */
export function getLocalPoint(world: World, eid: number, point: Pos): Vec3 | null {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return null;
    const k = kernel(state.ecsState);
    const ptr = k.bodyLocalPoint(state.worldId, body.index1 - 1, point.x, point.y, point.z);
    const p = new Float32Array(k.memory.buffer, ptr, 3);
    return { x: p[0], y: p[1], z: p[2] };
}

/** Current joint linear constraint error in meters (Box3D's linear separation).
 * Returns null for a joint added or destroyed since the last fixed sync. */
export function getJointLinearSeparation(world: World, eid: number): number | null {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    if (!state) return null;
    for (const binding of jointBindings(world)) {
        if (!world.has(eid, binding.component)) continue;
        const id = runtime.constraints.liveJoints.get(binding.index)?.get(eid);
        if (id && jointIsValid(state, id) && state.jointUserData[id.index1 - 1] === world.ref(eid))
            return jointLinearSeparation(state, id.index1 - 1);
    }
    return null;
}

/** Current solver mass in kg, center in local meters and inertia in kg·m² about that center.
 * Returns null for a body the solver does not hold: one added or destroyed since the last fixed sync.
 * The returned record belongs to the caller. */
export function getMassData(world: World, eid: number) {
    const runtime = runtimeFor(world);
    const body = boundBody(world, runtime, eid);
    return runtime.physicsWorld && body
        ? bodyGetMassData(runtime.physicsWorld, body.index1 - 1)
        : null;
}

/** Immediately sets mass in kg, center in local meters and inertia in kg·m² about that center.
 * Adjusts and publishes linear velocity to preserve the origin's velocity when the center moves.
 * Ignored while the solver is locked or before the body's first fixed sync. Later shape mass
 * updates replace this override. Mass must be finite and nonnegative; center and inertia must
 * be finite and inertia must have a nonnegative determinant. */
export function setMassData(
    world: World,
    eid: number,
    data: NonNullable<ReturnType<typeof getMassData>>,
): void {
    const runtime = runtimeFor(world);
    const body = boundBody(world, runtime, eid);
    if (!runtime.physicsWorld || !body || runtime.physicsWorld.locked) return;
    bodySetMassData(runtime.physicsWorld, body.index1 - 1, data);
    publishBodyVelocity(world, eid, runtime.physicsWorld, body);
}

/** Union of the body's current shape bounds in world meters, including Box3D's speculative margin.
 * A shape-less body returns its position.
 * Returns null for a body the solver does not hold: one added or destroyed since the last fixed sync.
 * The returned bounds belong to the caller. */
export function computeBodyAABB(world: World, eid: number) {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (!state || !body) return null;
    let shape = bodyField(state, body.index1 - 1, BodyField.headShapeId);
    const bounds = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
    if (shape === -1) {
        const p = bodyGetTransform(state, body.index1 - 1, spawnTransform).p;
        Object.assign(bounds.lowerBound, p);
        Object.assign(bounds.upperBound, p);
        return bounds;
    }
    readShapeAabb(state, shape, bounds);
    const next = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
    for (
        shape = shapeField(state, shape, ShapeField.nextShapeId);
        shape !== -1;
        shape = shapeField(state, shape, ShapeField.nextShapeId)
    ) {
        readShapeAabb(state, shape, next);
        for (const axis of ["x", "y", "z"] as const) {
            bounds.lowerBound[axis] = Math.min(bounds.lowerBound[axis], next.lowerBound[axis]);
            bounds.upperBound[axis] = Math.max(bounds.upperBound[axis], next.upperBound[axis]);
        }
    }
    return bounds;
}

/** Returns the solver's current awake state, or null for a body the solver does not hold: one added or destroyed since the last fixed sync. */
export function isAwake(world: World, eid: number): boolean | null {
    const runtime = runtimeFor(world);
    const body = boundBody(world, runtime, eid);
    return runtime.physicsWorld && body ? bodyIsAwake(runtime.physicsWorld, body.index1 - 1) : null;
}

/** Wakes a body and its island, or asks Box3D to put the whole island to sleep. A body added since the last fixed sync ignores it and spawns with `Body.isAwake`. */
export function setAwake(world: World, eid: number, awake: boolean): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = boundBody(world, runtime, eid);
    if (state && body) bodySetAwake(state, body.index1 - 1, awake);
}
export function physicsCounters(world: World): PhysicsCounters {
    return { ...runtimeFor(world).counters, hullsBuilt: kernelState(world).hullBuilds };
}

// The runtime's maps from entities to solver handles, as plain ids, so a restore rebinds them for its
// target World and the sync systems reconcile them against the live ECS on the next tick.
interface Bindings {
    creationOrder: number[];
    /** The authoring configuration paired with the captured solver settings. */
    worldDefinition: PhysicsWorldDefinitionConfig;
    /** eid, solver body index1, solver generation, EntityRef */
    bodies: number[];
    /** eid, solver shape index1, generation, body eid, EntityRef */
    shapes: number[];
    /** the last applied authored Shape definition, copied for reusable recovery images */
    shapeRecreationState: [number, Float64Array][];
    shapeRemovalMass: [number, boolean][];
    shapeDependencies: ReturnType<BodyDependencies["snapshot"]>;
    /** pending shape field candidates as [eid, field mask] pairs */
    shapeCandidates: [number, number][];
    constraints: ConstraintIds;
    /** pending sync words, copied so later marks cannot mutate the image */
    changed: Uint32Array | null;
    /** pending body field candidates as [eid, field mask] pairs */
    bodyCandidates: [number, number][];
    poseCandidates: [number, number][];
    full: boolean;
    /** geometry builds by registry id, re-paired with core's restored definitions */
    geometry: GeometryBuilds;
}

function warmWorld(runtime: PhysicsRuntime): WorldState {
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) throw new Error("physics: world is not warm");
    return physicsWorld;
}

function captureShapeRecreationStates(states: ShapeRecreationStates): [number, Float64Array][] {
    const captured: [number, Float64Array][] = [];
    for (let eid = 0; eid < states.captured.length; eid++) {
        const values = shapeRecreationState(states, eid);
        if (values) captured.push([eid, values]);
    }
    return captured;
}

function captureBodyCandidates(candidates: FieldCandidates): [number, number][] {
    const pending: [number, number][] = [];
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        if (eid !== 0) pending.push([eid, candidates.fieldMarks[eid]]);
    }
    return pending;
}

function captureBindings(world: World, runtime: PhysicsRuntime): Bindings {
    const bodies: number[] = [];
    for (const [eid, body] of runtime.bodies)
        bodies.push(eid, body.index1, body.generation, body.stamp);
    const shapes: number[] = [];
    for (const [eid, shape] of runtime.shapes)
        shapes.push(
            eid,
            shape.index1,
            shape.generation,
            runtime.shapeBodies.get(eid),
            runtime.shapeStamps.get(eid),
        );
    const order = world.resource(PhysicsCreationOrder);
    return {
        creationOrder: order.entries.slice(0, order.count),
        worldDefinition: copyPhysicsWorldDefinition(world.resource(PhysicsWorldDefinition)),
        bodies,
        shapes,
        shapeCandidates: captureBodyCandidates(runtime.shapeCandidates),
        shapeDependencies: runtime.shapeDependencies.snapshot(),
        shapeRemovalMass: Array.from(runtime.shapeRemovalMass),
        shapeRecreationState: captureShapeRecreationStates(runtime.shapeRecreationState),
        constraints: captureConstraints(runtime.constraints, warmWorld(runtime)),
        bodyCandidates: captureBodyCandidates(runtime.bodyCandidates),
        poseCandidates: captureBodyCandidates(runtime.poseCandidates),
        changed: runtime.anyChanged ? runtime.changed.slice() : null,
        full: runtime.full,
        geometry: captureGeometryBuilds(world),
    };
}

function restoreBindings(
    world: World,
    runtime: PhysicsRuntime,
    physicsWorld: WorldState,
    b: Bindings,
): void {
    Object.assign(
        world.resource(PhysicsWorldDefinition),
        copyPhysicsWorldDefinition(b.worldDefinition),
    );
    const order = world.resource(PhysicsCreationOrder);
    order.entries = b.creationOrder.slice();
    order.count = order.entries.length;
    const state = physicsWorld;
    for (const eid of runtime.bodies.keys()) releaseTransformPose(world, eid);
    runtime.bodies.clear();
    runtime.shapes.clear();
    runtime.shapeBodies.clear();
    if (runtime.physicsWorld) clearShapeEntities(runtime.physicsWorld.shapeEntities);
    runtime.shapeStamps.clear();
    clearShapeRecreationStates(runtime.shapeRecreationState);
    for (let i = 0; i < b.bodies.length; i += 4) {
        const eid = b.bodies[i];
        claimTransformPose(world, eid);
        runtime.bodies.set(eid, {
            index1: b.bodies[i + 1],
            world0: state.worldId,
            generation: b.bodies[i + 2],
            stamp: b.bodies[i + 3] as EntityRef,
        });
    }
    for (let i = 0; i < b.shapes.length; i += 5) {
        const eid = b.shapes[i];
        const shape: EntityId = {
            index1: b.shapes[i + 1],
            world0: state.worldId,
            generation: b.shapes[i + 2],
        };
        runtime.shapes.set(eid, shape);
        runtime.shapeBodies.set(eid, b.shapes[i + 3]);
        runtime.shapeStamps.set(eid, b.shapes[i + 4] as EntityRef);
        rememberShapeEntity(world, runtime, shape, eid, b.shapes[i + 3]);
    }
    runtime.shapeRemovalMass.clear();
    for (const [eid, updateMass] of b.shapeRemovalMass)
        runtime.shapeRemovalMass.set(eid, updateMass);
    for (const [eid, values] of b.shapeRecreationState)
        restoreShapeRecreationState(runtime.shapeRecreationState, eid, values);
    runtime.shapeDependencies.restore(b.shapeDependencies);
    restoreConstraints(runtime.constraints, b.constraints, physicsWorld);
    restoreGeometryBuilds(world, b.geometry);
    clearFieldCandidates(runtime.bodyCandidates);
    for (const [eid, fieldMask] of b.bodyCandidates) {
        markFieldCandidate(runtime.bodyCandidates, eid);
        runtime.bodyCandidates.fieldMarks[eid] = fieldMask;
    }
    clearFieldCandidates(runtime.poseCandidates);
    for (const [eid, fieldMask] of b.poseCandidates) {
        markFieldCandidate(runtime.poseCandidates, eid);
        runtime.poseCandidates.fieldMarks[eid] = fieldMask;
    }
    clearFieldCandidates(runtime.shapeCandidates);
    for (const [eid, fieldMask] of b.shapeCandidates) {
        markFieldCandidate(runtime.shapeCandidates, eid);
        runtime.shapeCandidates.fieldMarks[eid] = fieldMask;
    }
    runtime.warnedBodyFields.clear();
    if (b.changed) {
        if (b.changed.length > runtime.changed.length) {
            const grown = new Uint32Array(Math.max(b.changed.length, runtime.changed.length * 2));
            grown.set(runtime.changed);
            runtime.changed = grown;
        }
        for (let w = 0; w < b.changed.length; w++) {
            const bits = b.changed[w];
            if (bits === 0) continue;
            runtime.changed[w] |= bits;
            runtime.anyChanged = true;
        }
    }
    runtime.full = b.full;
}

function solverWorldDefinition(def: PhysicsWorldDefinitionConfig): Partial<WorldDef> {
    return {
        gravity: def.gravity,
        restitutionThreshold: def.restitutionThreshold,
        hitEventThreshold: def.hitEventThreshold,
        contactHertz: def.contactHertz,
        contactDampingRatio: def.contactDampingRatio,
        contactSpeed: def.contactSpeed,
        maximumLinearSpeed: def.maximumLinearSpeed,
        enableSleep: def.enableSleep,
        enableContinuous: def.enableContinuous,
        capacity: def.capacity,
    };
}

function syncWorldDefinition(state: WorldState, definition: PhysicsWorldDefinitionConfig): void {
    const gravity = definition.gravity;
    if (
        state.gravity.x !== f32(gravity.x) ||
        state.gravity.y !== f32(gravity.y) ||
        state.gravity.z !== f32(gravity.z)
    )
        worldSetGravity(state, gravity);

    const restitutionThreshold = clamp(f32(definition.restitutionThreshold), 0, Infinity);
    if (state.restitutionThreshold !== restitutionThreshold)
        worldSetRestitutionThreshold(state, definition.restitutionThreshold);
    const hitEventThreshold = clamp(f32(definition.hitEventThreshold), 0, Infinity);
    if (state.hitEventThreshold !== hitEventThreshold)
        worldSetHitEventThreshold(state, definition.hitEventThreshold);
    const contactHertz = clamp(f32(definition.contactHertz), 0, Infinity);
    const contactDampingRatio = clamp(f32(definition.contactDampingRatio), 0, Infinity);
    const contactSpeed = clamp(f32(definition.contactSpeed), 0, Infinity);
    if (
        state.contactHertz !== contactHertz ||
        state.contactDampingRatio !== contactDampingRatio ||
        state.contactSpeed !== contactSpeed
    )
        worldSetContactTuning(
            state,
            definition.contactHertz,
            definition.contactDampingRatio,
            definition.contactSpeed,
        );
    const recycleDistance = clamp(f32(definition.contactRecycleDistance), 0, Infinity);
    if (state.contactRecycleDistance !== recycleDistance)
        worldSetContactRecycleDistance(state, definition.contactRecycleDistance);
    const maximumLinearSpeed = f32(definition.maximumLinearSpeed);
    if (state.maxLinearSpeed !== maximumLinearSpeed)
        worldSetMaximumLinearSpeed(state, definition.maximumLinearSpeed);
    if (state.enableSleep !== definition.enableSleep) {
        state.enableSleep = definition.enableSleep;
        if (!definition.enableSleep) {
            const count = solverSetCount(state);
            for (let set = SetType.FirstSleeping; set < count; set++)
                if (solverSetIndex(state, set) >= 0 && setBodyCount(state, set) > 0)
                    wakeSolverSet(state, set);
        }
    }
    if (state.enableContinuous !== definition.enableContinuous)
        state.enableContinuous = definition.enableContinuous;
    if (state.enableWarmStarting !== definition.enableWarmStarting)
        state.enableWarmStarting = definition.enableWarmStarting;
    if (state.enableSpeculative !== definition.enableSpeculative)
        state.enableSpeculative = definition.enableSpeculative;
    const k = kernel(state.ecsState);
    if (state.worldCustomFilterCallback !== definition.customFilterCallback) {
        state.worldCustomFilterCallback = definition.customFilterCallback;
        k.worldSetCustomFilterCallback(
            state.worldId,
            definition.customFilterCallback !== null || state.customFilterCallback !== null,
        );
    }
    if (state.worldPreSolveCallback !== definition.preSolveCallback) {
        state.worldPreSolveCallback = definition.preSolveCallback;
        k.worldSetPreSolveCallback(
            state.worldId,
            definition.preSolveCallback !== null || state.preSolveCallback !== null,
        );
    }
    if (state.worldFrictionCallback !== definition.frictionCallback)
        state.worldFrictionCallback = definition.frictionCallback;
    if (state.worldRestitutionCallback !== definition.restitutionCallback)
        state.worldRestitutionCallback = definition.restitutionCallback;
}

function capturePhysics(world: World): PhysicsSnapshot {
    const runtime = runtimeFor(world);
    captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
    captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
    captureFieldCandidates(world, runtime.constraints.candidates, jointBindings(world));
    return snapshotWorld(warmWorld(runtime), captureBindings(world, runtime));
}
function recoverPhysics(world: World, saved: PhysicsSnapshot): void {
    const runtime = runtimeFor(world);
    const physicsWorld = warmWorld(runtime);
    const bindings = snapshotBindings(saved) as Bindings | undefined;
    if (bindings === undefined) throw new Error("physics: recovery image has no bindings");
    restoreWorld(physicsWorld, saved);
    restoreBindings(world, runtime, physicsWorld, bindings);
    clearPhysicsEvents(world.resource(PhysicsEvents));
    // ECS poses were restored by World. Warning latches are presentation, counters and stale
    // scans are overwritten on each step; solver history and binding caches are the participant.
}
export function hashPhysics(world: World): bigint {
    return hashWorldState(warmWorld(runtimeFor(world)));
}

/** Copies the entities of the shape an event names at `lane` into `out`; false for a shape no entity carried. */
function eventShapeEntity(
    state: WorldState,
    k: ReturnType<typeof kernel>,
    kind: number,
    eventIndex: number,
    lane: number,
    out: ShapeEntity,
): boolean {
    const id = k.eventWord(state.worldId, kind, eventIndex, lane) - 1;
    const generation = k.eventWord(state.worldId, kind, eventIndex, lane + 1) >>> 16;
    return id >= 0 && shapeEntityAt(state.shapeEntities, id, generation, out);
}
const eventShapeA: ShapeEntity = { shape: 0 as EntityRef, body: 0 as EntityRef };
const eventShapeB: ShapeEntity = { shape: 0 as EntityRef, body: 0 as EntityRef };

function sensorRecordAt(
    list: PhysicsEventRecords<PhysicsSensorEvent>,
    index: number,
): PhysicsSensorEvent {
    return (list.records[index] ??= {
        sensorShape: 0 as EntityRef,
        sensorBody: 0 as EntityRef,
        visitorShape: 0 as EntityRef,
        visitorBody: 0 as EntityRef,
    });
}

function contactRecordAt(
    list: PhysicsEventRecords<PhysicsContactEvent>,
    index: number,
): PhysicsContactEvent {
    return (list.records[index] ??= {
        shapeA: 0 as EntityRef,
        bodyA: 0 as EntityRef,
        shapeB: 0 as EntityRef,
        bodyB: 0 as EntityRef,
        normalImpulse: 0,
    });
}

function hitRecordAt(
    runtime: PhysicsRuntime,
    list: PhysicsEventRecords<PhysicsHitEvent>,
    index: number,
): PhysicsHitEvent {
    const existing = list.records[index];
    if (existing) return existing;
    if (runtime.hitMaterialWords.length < (index + 1) * 4) {
        const grown = new Uint32Array(runtime.hitMaterialWords.length * 2);
        grown.set(runtime.hitMaterialWords);
        runtime.hitMaterialWords = grown;
    }
    runtime.hitMaterialWords.fill(0, index * 4, index * 4 + 4);
    return (list.records[index] = {
        shapeA: 0 as EntityRef,
        bodyA: 0 as EntityRef,
        shapeB: 0 as EntityRef,
        bodyB: 0 as EntityRef,
        point: { x: 0, y: 0, z: 0 },
        normal: { x: 0, y: 0, z: 0 },
        approachSpeed: 0,
        userMaterialIdA: 0n,
        userMaterialIdB: 0n,
    });
}

// the 64-bit id from its u32 words, reusing `current` while the record's memoized words match
function materialId(words: Uint32Array, at: number, low: number, high: number, current: bigint) {
    if (words[at] === low && words[at + 1] === high) return current;
    words[at] = low;
    words[at + 1] = high;
    return BigInt(low) | (BigInt(high) << 32n);
}

function jointRecordAt(
    list: PhysicsEventRecords<PhysicsJointEvent>,
    index: number,
): PhysicsJointEvent {
    return (list.records[index] ??= { joint: 0 as EntityRef });
}

function fillSensorRecords(
    state: WorldState,
    list: PhysicsEventRecords<PhysicsSensorEvent>,
    kind: number,
): void {
    const k = kernel(state.ecsState);
    const count = eventCount(state, kind);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const sensor = eventShapeA;
        const visitor = eventShapeB;
        if (
            !eventShapeEntity(state, k, kind, i, 0, sensor) ||
            !eventShapeEntity(state, k, kind, i, 2, visitor)
        )
            continue;
        const record = sensorRecordAt(list, written++);
        record.sensorShape = sensor.shape;
        record.sensorBody = sensor.body;
        record.visitorShape = visitor.shape;
        record.visitorBody = visitor.body;
    }
    list.count = written;
}

function fillContactRecords(
    state: WorldState,
    list: PhysicsEventRecords<PhysicsContactEvent>,
    kind: number,
): void {
    const k = kernel(state.ecsState);
    const count = eventCount(state, kind);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const a = eventShapeA;
        const b = eventShapeB;
        if (
            !eventShapeEntity(state, k, kind, i, 0, a) ||
            !eventShapeEntity(state, k, kind, i, 2, b)
        )
            continue;
        const record = contactRecordAt(list, written++);
        record.shapeA = a.shape;
        record.bodyA = a.body;
        record.shapeB = b.shape;
        record.bodyB = b.body;
        record.normalImpulse =
            kind === EventKind.ContactBegin ? k.eventFloat(state.worldId, kind, i, 7) : 0;
    }
    list.count = written;
}

function fillHitRecords(
    runtime: PhysicsRuntime,
    state: WorldState,
    list: PhysicsEventRecords<PhysicsHitEvent>,
): void {
    const k = kernel(state.ecsState);
    const kind = EventKind.ContactHit;
    const count = eventCount(state, kind);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const a = eventShapeA;
        const b = eventShapeB;
        if (
            !eventShapeEntity(state, k, kind, i, 0, a) ||
            !eventShapeEntity(state, k, kind, i, 2, b)
        )
            continue;
        const at = written * 4;
        const record = hitRecordAt(runtime, list, written++);
        record.shapeA = a.shape;
        record.bodyA = a.body;
        record.shapeB = b.shape;
        record.bodyB = b.body;
        record.point.x = k.eventFloat(state.worldId, kind, i, 7);
        record.point.y = k.eventFloat(state.worldId, kind, i, 8);
        record.point.z = k.eventFloat(state.worldId, kind, i, 9);
        record.normal.x = k.eventFloat(state.worldId, kind, i, 10);
        record.normal.y = k.eventFloat(state.worldId, kind, i, 11);
        record.normal.z = k.eventFloat(state.worldId, kind, i, 12);
        record.approachSpeed = k.eventFloat(state.worldId, kind, i, 13);
        const words = runtime.hitMaterialWords;
        record.userMaterialIdA = materialId(
            words,
            at,
            k.eventWord(state.worldId, kind, i, 14) >>> 0,
            k.eventWord(state.worldId, kind, i, 15) >>> 0,
            record.userMaterialIdA,
        );
        record.userMaterialIdB = materialId(
            words,
            at + 2,
            k.eventWord(state.worldId, kind, i, 16) >>> 0,
            k.eventWord(state.worldId, kind, i, 17) >>> 0,
            record.userMaterialIdB,
        );
    }
    list.count = written;
}

function fillJointRecords(
    world: World,
    state: WorldState,
    list: PhysicsEventRecords<PhysicsJointEvent>,
): void {
    const count = eventCount(state, EventKind.Joint);
    if (count !== 0) readJointEventUserData(state);
    const bindings = jointBindings(world);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const reference = state.jointEventUserData[i];
        if (typeof reference !== "number" || !Number.isSafeInteger(reference) || reference < 0)
            continue;
        const eid = world.resolve(reference as EntityRef);
        if (eid === 0 && reference < 2 ** 32) continue;
        let isJoint = eid === 0;
        for (let j = 0; eid !== 0 && j < bindings.length; j++)
            if (world.has(eid, bindings[j].component)) {
                isJoint = true;
                break;
            }
        if (isJoint) jointRecordAt(list, written++).joint = reference as EntityRef;
    }
    list.count = written;
}

function rebuildPhysicsEvents(world: World, runtime: PhysicsRuntime): void {
    const state = runtime.physicsWorld!;
    const events = world.resource(PhysicsEvents);
    fillSensorRecords(state, events.sensorBegin, EventKind.SensorBegin);
    fillSensorRecords(state, events.sensorEnd, EventKind.SensorEnd);
    fillContactRecords(state, events.contactBegin, EventKind.ContactBegin);
    fillContactRecords(state, events.contactEnd, EventKind.ContactEnd);
    fillHitRecords(runtime, state, events.contactHit);
    fillJointRecords(world, state, events.joints);
    pruneShapeEntities(state.shapeEntities);
}

/** @internal Applies a marked teleport before a pose producer consumes the body's pose. */
export function applyTransformPose(world: World, eid: number): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    if (!state) return;
    const pose = world.storage(Transform);
    applyPose(
        world,
        runtime,
        state,
        world.fieldStorage(Transform, "translation").dirty,
        world.fieldStorage(Transform, "rotation").dirty,
        pose.translation.column,
        pose.rotation.column,
        eid,
        state.bodyStore.memoryF,
    );
}

function applyPose(
    world: World,
    runtime: PhysicsRuntime,
    state: WorldState,
    positions: Uint32Array,
    rotations: Uint32Array,
    p: TypedArray,
    q: TypedArray,
    eid: number,
    memory: Float32Array,
): boolean {
    if (
        ((positions[eid >>> 5] | rotations[eid >>> 5]) & (1 << (eid & 31))) === 0 &&
        !runtime.poseCandidates.fieldMarks[eid]
    )
        return false;
    const body = runtime.bodies.get(eid);
    if (!body) return false;
    const current = readSimTransform(state, -body.index1, spawnTransform, memory);
    const offset = eid * 4;
    if (
        Object.is(p[offset], current.p.x) &&
        Object.is(p[offset + 1], current.p.y) &&
        Object.is(p[offset + 2], current.p.z) &&
        Object.is(q[offset], current.q.v.x) &&
        Object.is(q[offset + 1], current.q.v.y) &&
        Object.is(q[offset + 2], current.q.v.z) &&
        Object.is(q[offset + 3], current.q.s)
    )
        return false;
    statePos.x = p[offset];
    statePos.y = p[offset + 1];
    statePos.z = p[offset + 2];
    stateQuat.v.x = q[offset];
    stateQuat.v.y = q[offset + 1];
    stateQuat.v.z = q[offset + 2];
    stateQuat.s = q[offset + 3];
    bodySetTransform(state, body.index1 - 1, statePos, stateQuat);
    const global = world.storage(GlobalTransform);
    global.translation.set(eid, statePos.x, statePos.y, statePos.z);
    global.rotation.set(eid, stateQuat.v.x, stateQuat.v.y, stateQuat.v.z, stateQuat.s);
    return true;
}

function capturePoseCandidates(world: World, runtime: PhysicsRuntime): void {
    const translation = world.fieldStorage(Transform, "translation");
    const rotation = world.fieldStorage(Transform, "rotation");
    const positions = translation.dirty;
    const rotations = rotation.dirty;
    const positionTicks = translation.changeTicks;
    const rotationTicks = rotation.changeTicks;
    const candidates = runtime.poseCandidates;
    const since = candidates.lastRun;
    candidates.lastRun = world.incrementChangeTick();
    if (translation.changeTick <= since && rotation.changeTick <= since) return;
    const words = Math.max(positions.length, rotations.length);
    for (let word = 0; word < words; word++) {
        let bits =
            ((positionTicks[word] ?? 0) > since ? positions[word] : 0) |
            ((rotationTicks[word] ?? 0) > since ? rotations[word] : 0);
        while (bits !== 0) {
            const mask = bits & -bits;
            bits ^= mask;
            const eid = (word << 5) | (31 - Math.clz32(mask));
            if (runtime.bodies.has(eid)) markFieldCandidate(candidates, eid, 0);
        }
    }
}

function applyTransformPoses(world: World, runtime: PhysicsRuntime): void {
    capturePoseCandidates(world, runtime);
    const candidates = runtime.poseCandidates;
    const state = runtime.physicsWorld;
    if (state) {
        const positions = world.fieldStorage(Transform, "translation").dirty;
        const rotations = world.fieldStorage(Transform, "rotation").dirty;
        const pose = world.storage(Transform);
        const p = pose.translation.column;
        const q = pose.rotation.column;
        // A teleport can allocate, so it ends the held memory view.
        let memory = state.bodyStore.memoryF;
        for (let i = 0; i < candidates.count; i++)
            if (
                applyPose(
                    world,
                    runtime,
                    state,
                    positions,
                    rotations,
                    p,
                    q,
                    candidates.eids[i],
                    memory,
                )
            )
                memory = state.bodyStore.memoryF;
    }
    clearFieldCandidates(candidates);
}

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepPhysicsSystem: System = {
    name: "step",
    group: "fixed",
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        applyTransformPoses(world, runtime);
        stepWorld(
            physicsWorld,
            fixedDeltaTime,
            world.resource(PhysicsWorldDefinition).subStepCount,
        );
        rebuildPhysicsEvents(world, runtime);
        const record = world.recordSink;
        if (record) {
            readStepProfile(physicsWorld);
            const profile = physicsWorld.profile;
            const measurement = world.resource(phaseMeasurement);
            sampleDiagnosticTime(measurement);
            for (let i = 0; i < phaseFields.length; i++) {
                const ms = profile[phaseFields[i]];
                if (ms !== 0) {
                    measurement[1] = ms;
                    record(phaseNames[i], measurement);
                }
            }
        }
        runtime.counters.bytesUploaded = 0;
        const global = world.storage(GlobalTransform);
        const rows = physicsWorld.bodyStore.movedRows();
        global.translation.writeEncoded(rows.eids, rows.pos);
        global.rotation.writeEncoded(rows.eids, rows.quat);
        global.linearVelocity.writeEncoded(rows.eids, rows.vel);
        const pose = world.storage(Transform);
        pose.translation.writeEncoded(rows.eids, rows.pos);
        pose.rotation.writeEncoded(rows.eids, rows.quat);
    },
};

/** Uploads authored constraints on change, before this tick's physics step. */
export const SyncPhysicsConstraintsSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepPhysicsSystem],
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        const bindings = jointBindings(world);
        captureFieldCandidates(world, runtime.constraints.candidates, bindings);
        syncJoints(runtime.constraints, physicsWorld, runtime.bodies, world, bindings);
    },
};

/** Latches every field mark physics reads into its candidate queues before a clear point. */
function collectFieldCandidates(world: World): void {
    const runtime = runtimeFor(world);
    capturePoseCandidates(world, runtime);
    captureScaleCandidates(world, runtime);
    captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
    captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
    captureFieldCandidates(world, runtime.constraints.candidates, jointBindings(world));
}

/** Captures fixed writes before CPU-only World.tick clears field marks on return. */
export const CollectFixedPhysicsFieldCandidatesSystem: System = {
    name: "fixed-physics-field-candidates",
    group: "fixed",
    boundary: "after",
    update: collectFieldCandidates,
};

/** Captures authored field marks before the frame's clearChanges point. */
export const CollectPhysicsFieldCandidatesSystem: System = {
    name: "physics-field-candidates",
    group: "simulation",
    boundary: "after",
    update: collectFieldCandidates,
};

/** Captures draw writes before CPU-only World.step clears frame marks on return. */
export const CollectDrawPhysicsFieldCandidatesSystem: System = {
    name: "draw-physics-field-candidates",
    group: "draw",
    boundary: "after",
    update: collectFieldCandidates,
};

// Retain query terms to avoid steady-sync allocations.
const BODY_TERMS = [Body];

// Only the first count eids are live; retain the array's high-water capacity between scans.
interface StaleScan {
    world: World | null;
    eids: number[];
    count: number;
}

// the sync's map walk, given its scan as `this`, so a steady sync mints no iterator.
function collectStale(this: StaleScan, _body: unknown, eid: number): void {
    if (!this.world!.has(eid, Body)) this.eids[this.count++] = eid;
}

// New memberships reach this in authoring order, before the field reconciliation.
function visitBody(
    world: World,
    runtime: PhysicsRuntime,
    physicsWorld: WorldState,
    eid: number,
): void {
    const stamp = world.ref(eid);
    const bound = runtime.bodies.get(eid);
    if (bound) {
        if (world.resolve(bound.stamp)) {
            claimTransformPose(world, eid);
            return;
        }
        forget(world, runtime, eid); // recycled to a new Body in one update
    }
    const tb = marshalBody(world, physicsWorld, eid);
    kernel(world).bodySetEntity(tb.world0, tb.index1 - 1, eid);
    runtime.bodies.set(eid, {
        index1: tb.index1,
        world0: tb.world0,
        generation: tb.generation,
        stamp,
    });
    claimTransformPose(world, eid);
    dropFieldCandidate(runtime.bodyCandidates, eid);
    seedGlobalTransform(world, eid, physicsWorld, tb);
    resetInterpolation(world, eid);
    markBodyDependents(runtime, eid);
}

function removeShape(
    runtime: PhysicsRuntime,
    eid: number,
    updateBodyMass = runtime.shapeRemovalMass.get(eid) ?? true,
): void {
    runtime.shapeRemovalMass.delete(eid);
    const shape = runtime.shapes.get(eid);
    const state = runtime.physicsWorld;
    retireShape(runtime, shape);
    if (shape && state && shapeIsValid(state, shape))
        destroyShape(state, shape.index1 - 1, updateBodyMass);
    runtime.shapes.delete(eid);
    runtime.shapeBodies.delete(eid);
    runtime.shapeStamps.delete(eid);
    forgetShapeRecreationState(runtime.shapeRecreationState, eid);
    runtime.warnedShapes.delete(eid);
}

function markBodyDependents(runtime: PhysicsRuntime, body: number): void {
    const dependencies = runtime.shapeDependencies;
    for (let eid = dependencies.first(body); eid !== 0; eid = dependencies.following(eid))
        requeueFieldCandidate(runtime.shapeCandidates, eid);
    markBodyJointCandidates(runtime.constraints, body);
}

function warnShape(runtime: PhysicsRuntime, eid: number, error: unknown): void {
    if (runtime.warnedShapes.has(eid)) return;
    runtime.warnedShapes.add(eid);
    console.warn(`[physics] Shape ${eid} was not attached: ${String(error)}`);
}

/** Queues each attached shape whose body's marked Transform.scale differs from its built scale. */
function captureScaleCandidates(world: World, runtime: PhysicsRuntime): void {
    const candidates = runtime.shapeCandidates;
    const scale = world.fieldStorage(Transform, "scale");
    const scaleDirty = scale.dirty;
    const ticks = scale.changeTicks;
    const since = runtime.scaleRun;
    runtime.scaleRun = world.incrementChangeTick();
    if (scale.changeTick <= since) return;
    const dependencies = runtime.shapeDependencies;
    for (let word = 0; word < scaleDirty.length; word++) {
        if (ticks[word] <= since) continue;
        let bits = scaleDirty[word];
        while (bits !== 0) {
            const low = bits & -bits;
            bits ^= low;
            const body = (word << 5) | (31 - Math.clz32(low));
            for (let eid = dependencies.first(body); eid !== 0; eid = dependencies.following(eid)) {
                if (
                    runtime.shapeBodies.get(eid) === body &&
                    shapeBodyScaleChanged(world, body, runtime.shapeRecreationState, eid)
                )
                    markFieldCandidate(candidates, eid);
            }
        }
    }
}

function syncShapes(world: World, runtime: PhysicsRuntime, checkScale = true): void {
    const state = runtime.physicsWorld;
    if (!state) return;
    const candidates = runtime.shapeCandidates;
    const binding = shapeBinding(world);
    if (checkScale) captureScaleCandidates(world, runtime);
    const count = candidates.count;
    let pending = 0;
    for (let i = 0; i < count; i++) {
        const eid = candidates.eids[i];
        if (eid === 0) continue;
        runtime.counters.shapesVisited++;
        const mask = candidates.fieldMarks[eid];
        if (!world.has(eid, ShapeComponent)) {
            runtime.shapeDependencies.set(eid, 0);
            removeShape(runtime, eid);
            candidates.positions[eid] = 0;
            candidates.fieldMarks[eid] = 0;
            continue;
        }
        if (runtime.shapeRemovalMass.has(eid)) removeShape(runtime, eid);
        const updateBodyMass = binding.storage.updateBodyMass.column[eid] !== 0;
        const previousStamp = runtime.shapeStamps.get(eid) as EntityRef;
        if (previousStamp !== 0 && !world.resolve(previousStamp)) removeShape(runtime, eid);

        const rawBody = (binding.storage.body.column as Float64Array)[eid] as EntityRef;
        const bodyEid = rawBody === 0 ? eid : world.resolve(rawBody);
        runtime.shapeDependencies.set(eid, bodyEid);
        const boundBody = runtime.shapeBodies.get(eid);
        if (boundBody !== 0 && boundBody !== bodyEid) removeShape(runtime, eid, updateBodyMass);
        if (!bodyEid) {
            removeShape(runtime, eid, updateBodyMass);
            warnShape(runtime, eid, new Error("body reference is no longer valid"));
            candidates.positions[eid] = 0;
            candidates.fieldMarks[eid] = 0;
            continue;
        }
        if (!world.has(bodyEid, Body)) {
            candidates.eids[pending] = eid;
            candidates.positions[eid] = ++pending;
            continue;
        }
        const body = runtime.bodies.get(bodyEid);
        if (!body || !world.resolve(body.stamp)) {
            candidates.eids[pending] = eid;
            candidates.positions[eid] = ++pending;
            continue;
        }

        let shape = runtime.shapes.get(eid);
        if (shape && !shapeIsValid(state, shape)) {
            removeShape(runtime, eid, updateBodyMass);
            shape = undefined;
        }
        if (shape) {
            try {
                shapeBodyScale(world, eid);
            } catch (error) {
                removeShape(runtime, eid, updateBodyMass);
                warnShape(runtime, eid, error);
                candidates.eids[pending] = eid;
                candidates.positions[eid] = ++pending;
                continue;
            }
            const scaleChanged = shapeBodyScaleChanged(
                world,
                bodyEid,
                runtime.shapeRecreationState,
                eid,
            );
            try {
                if (
                    shapeNeedsRecreation(
                        world,
                        binding,
                        eid,
                        mask,
                        runtime.shapeRecreationState,
                        shape,
                        state,
                        scaleChanged,
                    )
                ) {
                    removeShape(runtime, eid, updateBodyMass);
                    shape = undefined;
                } else {
                    syncShapeFields(world, state, binding, eid, mask, shape, scaleChanged);
                }
            } catch (error) {
                warnShape(runtime, eid, error);
                candidates.eids[pending] = eid;
                candidates.positions[eid] = ++pending;
                continue;
            }
        }
        if (!shape) {
            try {
                shape = marshalShape(world, state, body, eid);
                runtime.shapes.set(eid, shape);
                runtime.shapeBodies.set(eid, bodyEid);
                runtime.shapeStamps.set(eid, runtime.shapeStamps.get(eid) || world.ref(eid));
                runtime.warnedShapes.delete(eid);
            } catch (error) {
                warnShape(runtime, eid, error);
                candidates.eids[pending] = eid;
                candidates.positions[eid] = ++pending;
                continue;
            }
        }
        rememberShapeEntity(world, runtime, shape, eid, bodyEid);
        captureShapeRecreationState(world, binding, eid, runtime.shapeRecreationState);
        candidates.positions[eid] = 0;
        candidates.fieldMarks[eid] = 0;
    }
    candidates.count = pending;
    candidates.tombstones = 0;
}

function realizeMemberships(world: World, runtime: PhysicsRuntime): void {
    const journal = world.resource(PhysicsCreationOrder);
    const state = runtime.physicsWorld!;
    const bindings = jointBindings(world);
    const scratch = runtime.creationCandidates;
    for (let at = 0; at < journal.count; at += 4) {
        const component = journal.components[journal.entries[at]];
        let jointKind = -1;
        if (component !== Body && component !== ShapeComponent) {
            for (let i = 0; i < bindings.length; i++) {
                if (bindings[i].component === component) {
                    jointKind = i;
                    break;
                }
            }
        }
        const eid = journal.entries[at + 1];
        const reference = journal.entries[at + 2] as EntityRef;
        const present = journal.entries[at + 3] !== 0;
        if (!present) {
            if (component === Body) {
                if (runtime.bodies.has(eid)) {
                    forget(world, runtime, eid);
                }
            } else if (component === ShapeComponent) {
                runtime.shapeDependencies.set(eid, 0);
                removeShape(runtime, eid);
            } else removeAuthoredJoint(runtime.constraints, state, jointKind, eid);
            continue;
        }
        if (world.resolve(reference) !== eid) continue;
        if (component === Body) {
            if (world.has(eid, Body)) visitBody(world, runtime, state, eid);
        } else if (component === ShapeComponent) {
            if (!world.has(eid, ShapeComponent)) continue;
            const candidates = runtime.shapeCandidates;
            markFieldCandidate(candidates, eid);
            runtime.shapeCandidates = scratch;
            markFieldCandidate(scratch, eid);
            syncShapes(world, runtime, false);
            // A pending shape stays queued to retry; an attached one was created from every current field.
            const pending = scratch.count !== 0;
            clearFieldCandidates(scratch);
            runtime.shapeCandidates = candidates;
            if (!pending) dropFieldCandidate(candidates, eid);
        } else {
            const index = jointKind;
            if (!world.has(eid, bindings[index].component)) continue;
            const candidates = runtime.constraints.candidates[index];
            markFieldCandidate(candidates, eid);
            runtime.constraints.candidates[index] = scratch;
            markFieldCandidate(scratch, eid);
            syncJoints(runtime.constraints, state, runtime.bodies, world, bindings, index);
            runtime.constraints.candidates[index] = candidates;
        }
    }
    journal.count = 0;
}

const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem, StepPhysicsSystem],
    update(world: World) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        runtime.counters.shapesVisited = 0;
        captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
        captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
        // Latch poses before creation, so a body created from its Transform is not compared at the step.
        capturePoseCandidates(world, runtime);
        realizeMemberships(world, runtime);
        let ended = false;
        if (runtime.full) {
            runtime.full = false;
            runtime.changed.fill(0);
            runtime.anyChanged = false;
            let bound = 0;
            for (const eid of world.query(BODY_TERMS)) {
                visitBody(world, runtime, physicsWorld, eid);
                if (runtime.bodies.has(eid)) bound++;
            }
            // every bound body was visited live, so none is stale
            ended = bound !== runtime.bodies.size;
        } else {
            if (runtime.anyChanged) {
                runtime.anyChanged = false;
                const changed = runtime.changed;
                for (let w = 0; w < changed.length; w++) {
                    let bits = changed[w];
                    if (bits === 0) continue;
                    changed[w] = 0;
                    while (bits !== 0) {
                        const low = bits & -bits;
                        bits ^= low;
                        const eid = (w << 5) | (31 - Math.clz32(low));
                        if (world.has(eid, Body)) {
                            visitBody(world, runtime, physicsWorld, eid);
                        } else if (runtime.bodies.has(eid)) ended = true;
                    }
                }
            }
        }
        if (ended) {
            const stale = runtime.stale;
            stale.world = world;
            stale.count = 0;
            runtime.bodies.forEach(collectStale, stale);
            stale.world = null;
            for (let i = 0; i < stale.count; i++) {
                const eid = stale.eids[i];
                forget(world, runtime, eid);
            }
        }
        syncBodyFields(
            physicsWorld,
            runtime.bodyCandidates,
            bodyBinding(world),
            runtime.bodies,
            runtime.warnedBodyFields,
        );
        syncShapes(world, runtime);
        clearFieldCandidates(runtime.bodyCandidates);
        runtime.counters.bodiesVisited = runtime.bodies.size;
    },
};

/** Applies retained world-definition candidates before this tick's solver step. */
export const SyncPhysicsWorldDefinitionSystem: System = {
    name: "world-definition",
    group: "fixed",
    before: [SyncSystem],
    update(world) {
        const physicsWorld = runtimeFor(world).physicsWorld;
        if (physicsWorld) syncWorldDefinition(physicsWorld, world.resource(PhysicsWorldDefinition));
    },
};

/**
 * Opt-in rigid-body simulation; not in the default plugins. Steps the Rust/WASM kernel without
 * a GPU device, synchronizing core physics authoring and publishing placement and events.
 */
export const StandardPhysicsPlugin: Plugin = {
    name: "StandardPhysics",
    recovery(world) {
        const order = world.resource(PhysicsCreationOrder);
        order.components = [...PhysicsPlugin.components!];
        for (let kind = 0; kind < order.components.length; kind++) {
            world.onDispose(
                world.observeMembership(order.components[kind], (eid, present) =>
                    recordPhysicsMembership(world, kind, eid, present),
                ),
            );
        }
        return {
            snapshot: () => capturePhysics(world),
            restore: (image: PhysicsSnapshot) => recoverPhysics(world, image),
        };
    },
    dependencies: [PhysicsPlugin],
    systems: [
        SyncSystem,
        SyncPhysicsWorldDefinitionSystem,
        SyncPhysicsConstraintsSystem,
        StepPhysicsSystem,
        CollectFixedPhysicsFieldCandidatesSystem,
        CollectPhysicsFieldCandidatesSystem,
        CollectDrawPhysicsFieldCandidatesSystem,
    ],

    initialize(world) {
        const runtime = world.resource(physicsRuntimeKey);
        world.resource(PhysicsWorldDefinition);
        world.resource(PhysicsEvents);
        runtime.initialized = true;
        if (!runtime.observing) {
            runtime.observing = true;
            world.onDispose(
                world.observeMembership(Body, (eid, present) => {
                    if (!present) releaseTransformPose(world, eid);
                    markChanged(runtime, eid);
                }),
            );
            world.onDispose(
                world.observeMembership(ShapeComponent, (eid, present) => {
                    if (!present && runtime.shapes.has(eid))
                        runtime.shapeRemovalMass.set(
                            eid,
                            world.storage(ShapeComponent).updateBodyMass.get(eid) !== 0,
                        );
                    markFieldCandidate(runtime.shapeCandidates, eid);
                }),
            );
            const bindings = jointBindings(world);
            for (const binding of bindings) {
                world.onDispose(
                    world.observeMembership(binding.component, (eid) =>
                        markFieldCandidate(runtime.constraints.candidates[binding.index], eid),
                    ),
                );
            }
        }
    },

    async warm(world) {
        const runtime = runtimeFor(world);
        const definition = world.resource(PhysicsWorldDefinition);
        await init(world, {
            threads: definition.workerCount,
            maximumMemory: definition.maximumMemory,
        });
        if (runtime.physicsWorld) destroyWorld(runtime.physicsWorld);
        const id = createWorld(world, {
            ...defaultWorldDef(),
            ...solverWorldDefinition(definition),
        });
        runtime.physicsWorld = getWorld(id)!;
        syncWorldDefinition(runtime.physicsWorld, definition);
        clearBodies(world, runtime);
        clearPhysicsEvents(world.resource(PhysicsEvents));
        reportStartup({ kind: "physics", threads: threads(world), artifact: artifact(world) });
    },

    dispose(world) {
        const runtime = runtimeFor(world);
        clearBodies(world, runtime);
        clearPhysicsEvents(world.resource(PhysicsEvents));
        if (runtime.physicsWorld) destroyWorld(runtime.physicsWorld);
        runtime.physicsWorld = null;
        void shutdown(world);
    },
};

const phaseMeasurement = { create: () => new Float64Array(2) };
const phaseFields = PROFILE_FIELDS.filter((field) => field !== "step");
const phaseNames = phaseFields.map(
    (field) => `${StandardPhysicsPlugin.name}/${StepPhysicsSystem.name}/${field}`,
);
