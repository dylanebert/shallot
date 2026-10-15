/// <reference types="@webgpu/types" />

import { Body, BodyType, PhysicsPlugin, Shape as ShapeComponent } from "../../core/physics";
import { GlobalTransform, resetInterpolation, Transform } from "../../core/transform";
import { type EntityRef, type Plugin, type System, Time, type World } from "../../engine";
import { clamp } from "../../engine/utils";
import type { AABB, Pos, Quat, Vec3, WorldState, WorldTransform } from "./api";
import { init, type PhysicsSnapshot, shutdown } from "./api";
import {
    restore as restoreWorld,
    snapshotBindings,
    snapshot as snapshotWorld,
} from "./api/snapshot";
import { bodyBinding, jointBindings, shapeBinding } from "./authoring";
import { syncBodyFields } from "./body-sync";
import type { ShapeProxy } from "./collision/distance";
import { DEFAULT_MASK_BITS, SetType } from "./common/constants";
import type { EntityId } from "./common/ids";

const f32 = Math.fround;

import { defaultWorldDef, type QueryFilter, type WorldDef } from "./common/types";
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
    clearFieldCandidate,
    clearFieldCandidates,
    createFieldCandidates,
    type FieldCandidates,
    markFieldCandidate,
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
    markAllJointCandidates,
    resetConstraints,
    restoreConstraints,
    syncJoints,
} from "./joints";
import { EventKind, eventCount } from "./kernel/eventbuffers";
import { readJointEventUserData } from "./kernel/jointcolumns";
import {
    kernel,
    type QueryCallback,
    rethrowQueryError,
    setQueryCallback,
    writeQueryReturn,
} from "./kernel/kernel";
import { queryColumns } from "./kernel/querycolumns";
import {
    setBodyCount,
    solverSetCount,
    solverSetIndex,
    wakeSolverSet,
} from "./kernel/solversetcolumns";
import { marshalBody } from "./marshal";
import { marshalShape } from "./marshal-shape";
import { captureShapeRecreationState, shapeNeedsRecreation, syncShapeFields } from "./shape-sync";
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
    bodyGetTransform,
    bodyGetType,
    bodyIsAwake,
    bodyIsValid,
    bodySetAngularVelocity,
    bodySetAwake,
    bodySetLinearVelocity,
    bodySetTargetTransform,
    bodySetTransform,
    destroyBody,
} from "./world/body";
import { type DebugDraw, worldDraw } from "./world/draw";
import { hashWorldState } from "./world/hash";
import { PROFILE_FIELDS, readStepProfile } from "./world/profile";
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
    bytesUploaded: number;
}

interface PhysicsRuntime {
    initialized: boolean;
    bodyCandidates: FieldCandidates;
    shapeCandidates: FieldCandidates;
    warnedBodyFields: Set<number>;
    shapes: Map<number, EntityId>;
    shapeBodies: Map<number, number>;
    shapeEntities: ShapeEntityMap;
    queryContexts: QueryContext[];
    queryDepth: number;
    queryCastCallback: QueryCallback;
    queryOverlapCallback: QueryCallback;
    shapeStamps: Map<number, EntityRef>;
    shapeRecreationState: Map<number, Float64Array>;
    warnedShapes: Set<number>;
    physicsWorld: WorldState | null;
    bodies: Map<number, BoundBodyId>;
    placementWarnings: Map<number, EntityRef>;
    constraints: ConstraintCache;
    stale: StaleScan;
    counters: PhysicsCounters;
    // eids whose Body membership, or a Body's Transform, changed since the last sync, as 32-bit words
    changed: Uint32Array;
    anyChanged: boolean;
    // only warm sets this to make the next sync walk every Body
    full: boolean;
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

interface ShapeEntityBinding {
    slot: number;
    generation: number;
    shape: EntityRef;
    body: EntityRef;
    retiring: boolean;
}

interface ShapeEntityMap {
    slots: (ShapeEntityBinding[] | undefined)[];
    retiring: ShapeEntityBinding[];
}

function newShapeEntityMap(): ShapeEntityMap {
    return { slots: new Array(64), retiring: [] };
}

const physicsRuntimeKey = { create: newRuntime };

function newRuntime(): PhysicsRuntime {
    const runtime: PhysicsRuntime = {
        initialized: false,
        bodyCandidates: createFieldCandidates(),
        shapeCandidates: createFieldCandidates(),
        warnedBodyFields: new Set(),
        shapes: new Map(),
        shapeBodies: new Map(),
        shapeEntities: newShapeEntityMap(),
        queryContexts: [newQueryContext()],
        queryDepth: 0,
        queryCastCallback: null!,
        queryOverlapCallback: null!,
        shapeStamps: new Map(),
        shapeRecreationState: new Map(),
        warnedShapes: new Set(),
        physicsWorld: null,
        bodies: new Map(),
        placementWarnings: new Map(),
        constraints: createConstraintCache(),
        stale: { world: null, eids: [], count: 0 },
        counters: { bodiesVisited: 0, bytesUploaded: 0 },
        changed: new Uint32Array(64),
        anyChanged: false,
        full: true,
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
    state.shapeStore.refreshViews();
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
    const bindings = runtime.shapeEntities.slots[shapeId];
    if (!bindings) return false;
    for (let i = bindings.length - 1; i >= 0; --i) {
        const binding = bindings[i];
        if (binding.retiring) continue;
        const shapeEid = world.resolve(binding.shape);
        if (!shapeEid || !world.has(shapeEid, ShapeComponent)) continue;
        const bodyEid = world.resolve(binding.body);
        if (!bodyEid || !world.has(bodyEid, Body)) continue;
        const shape = runtime.shapes.get(shapeEid);
        if (!shape || shape.index1 !== shapeId + 1 || shape.generation !== binding.generation)
            continue;
        context.shapeEid = shapeEid;
        context.bodyEid = bodyEid;
        return true;
    }
    return false;
}

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
        hit: {
            shape: 0,
            body: 0,
            point: { x: 0, y: 0, z: 0 },
            normal: { x: 0, y: 0, z: 0 },
            fraction: 0,
            userMaterialId: 0n,
            triangleIndex: 0,
            childIndex: 0,
        },
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
    const slot = shape.index1 - 1;
    const map = runtime.shapeEntities.slots;
    if (slot >= map.length) map.length = Math.max(slot + 1, map.length * 2);
    const bindings = (map[slot] ??= []);
    const reference = world.ref(shapeEid);
    const state = runtime.physicsWorld;
    if (state) state.shapeEntityRefs[slot] = { generation: shape.generation, reference };
    for (const binding of bindings)
        if (binding.generation !== shape.generation && !binding.retiring) {
            binding.retiring = true;
            runtime.shapeEntities.retiring.push(binding);
        }
    for (const binding of bindings)
        if (binding.generation === shape.generation) {
            binding.shape = reference;
            binding.body = world.ref(bodyEid);
            return;
        }
    bindings.push({
        slot,
        generation: shape.generation,
        shape: reference,
        body: world.ref(bodyEid),
        retiring: false,
    });
}

function retireShapeEntity(runtime: PhysicsRuntime, shape: EntityId | undefined): void {
    if (!shape) return;
    const slot = shape.index1 - 1;
    const bindings = runtime.shapeEntities.slots[slot];
    if (!bindings) return;
    for (const binding of bindings)
        if (binding.generation === shape.generation && !binding.retiring) {
            binding.retiring = true;
            runtime.shapeEntities.retiring.push(binding);
            return;
        }
}

function clearShapeEntities(runtime: PhysicsRuntime): void {
    runtime.shapeEntities.slots.fill(undefined);
    runtime.shapeEntities.retiring.length = 0;
    if (runtime.physicsWorld) runtime.physicsWorld.shapeEntityRefs.length = 0;
}

function pruneShapeEntities(runtime: PhysicsRuntime): void {
    const { slots, retiring } = runtime.shapeEntities;
    for (let i = 0; i < retiring.length; i++) {
        const binding = retiring[i];
        const bindings = slots[binding.slot];
        if (!bindings) continue;
        const index = bindings.indexOf(binding);
        if (index >= 0) {
            bindings[index] = bindings[bindings.length - 1];
            bindings.pop();
            if (bindings.length === 0) slots[binding.slot] = undefined;
        }
        binding.retiring = false;
    }
    retiring.length = 0;
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

function forget(runtime: PhysicsRuntime, eid: number): void {
    for (const [shapeEid, bodyEid] of runtime.shapeBodies)
        if (bodyEid === eid) retireShapeEntity(runtime, runtime.shapes.get(shapeEid));
    const body = runtime.bodies.get(eid);
    const state = runtime.physicsWorld;
    if (body && state && bodyIsValid(state, body)) destroyBody(state, body.index1 - 1);
    runtime.bodies.delete(eid);
    for (const [shapeEid, bodyEid] of runtime.shapeBodies)
        if (bodyEid === eid) {
            runtime.shapes.delete(shapeEid);
            runtime.shapeBodies.delete(shapeEid);
            runtime.shapeStamps.delete(shapeEid);
            runtime.shapeRecreationState.delete(shapeEid);
            runtime.warnedShapes.delete(shapeEid);
            markFieldCandidate(runtime.shapeCandidates, shapeEid);
        }
    for (let fieldIndex = 0; fieldIndex < 32; fieldIndex++)
        runtime.warnedBodyFields.delete(eid * 32 + fieldIndex);
}

function clearBodies(runtime: PhysicsRuntime): void {
    const state = runtime.physicsWorld;
    if (state)
        for (const body of runtime.bodies.values())
            if (bodyIsValid(state, body)) destroyBody(state, body.index1 - 1);
    runtime.bodies.clear();
    runtime.shapes.clear();
    runtime.shapeBodies.clear();
    clearShapeEntities(runtime);
    runtime.shapeStamps.clear();
    runtime.shapeRecreationState.clear();
    runtime.warnedShapes.clear();
    clearFieldCandidates(runtime.shapeCandidates);
    runtime.counters = { bodiesVisited: 0, bytesUploaded: 0 };
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
 * output is cleared on a miss. Supply `point` and `normal` objects on `out` before calling.
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

/** Cast a convex proxy and write its closest hit into caller-owned storage; clears `out` on a miss. */
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
 * Drives a kinematic or dynamic body toward a world pose over `timeStep` seconds. Box3D derives
 * linear velocity in meters per second and angular velocity in radians per second; `wake` wakes an
 * asleep body when the target motion is significant. The pose and linear velocity publish immediately.
 * Static or unavailable bodies are ignored, as in Box3D.
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
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    const record = body.index1 - 1;
    if (bodyGetType(state, record) === BodyType.Static) return;
    bodySetTargetTransform(state, record, stateTransform(position, rotation), timeStep, wake);
    // Box3D leaves sub-threshold target motion asleep without adopting its pose.
    if (!bodyIsAwake(state, record)) return;
    bodyGetLinearVelocity(state, record, stateVelocity);
    stateVelocityTuple[0] = stateVelocity.x;
    stateVelocityTuple[1] = stateVelocity.y;
    stateVelocityTuple[2] = stateVelocity.z;
    writeGlobalTransform(world, eid, position, rotation, stateVelocityTuple);
}

/** Teleports a body in meters, preserving both velocities and discarding placement interpolation. */
export function setTransform(
    world: World,
    eid: number,
    position: readonly [number, number, number],
    rotation: readonly [number, number, number, number],
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    const record = body.index1 - 1;
    bodySetTransform(state, record, stateTransform(position, rotation).p, stateQuat);
    bodyGetLinearVelocity(state, record, stateVelocity);
    stateVelocityTuple[0] = stateVelocity.x;
    stateVelocityTuple[1] = stateVelocity.y;
    stateVelocityTuple[2] = stateVelocity.z;
    writeGlobalTransform(world, eid, position, rotation, stateVelocityTuple);
    resetInterpolation(world, eid);
}

/** Sets center-of-mass velocity in meters per second; nonzero values wake the body. Box3D ignores static bodies. */
export function setLinearVelocity(world: World, eid: number, velocity: Vec3): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    const record = body.index1 - 1;
    stateVector.x = velocity.x;
    stateVector.y = velocity.y;
    stateVector.z = velocity.z;
    bodySetLinearVelocity(state, record, stateVector);
    if (bodyGetType(state, record) !== BodyType.Static)
        publishBodyVelocity(world, eid, state, body);
}

/** Sets angular velocity in radians per second; nonzero values wake the body and locked axes are zeroed. Box3D ignores static bodies. */
export function setAngularVelocity(world: World, eid: number, velocity: Vec3): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = velocity.x;
    stateVector.y = velocity.y;
    stateVector.z = velocity.z;
    bodySetAngularVelocity(state, body.index1 - 1, stateVector);
}

/** Accumulates world force in newtons at a world point for the next step; an off-center point adds torque. `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyForce(world: World, eid: number, force: Vec3, point: Pos, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = force.x;
    stateVector.y = force.y;
    stateVector.z = force.z;
    statePos.x = point.x;
    statePos.y = point.y;
    statePos.z = point.z;
    bodyApplyForce(state, body.index1 - 1, stateVector, statePos, wake);
}

/** Accumulates world force in newtons at the center of mass for the next step; `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyForceToCenter(world: World, eid: number, force: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = force.x;
    stateVector.y = force.y;
    stateVector.z = force.z;
    bodyApplyForceToCenter(state, body.index1 - 1, stateVector, wake);
}

/** Accumulates world torque in newton-meters for the next step; `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyTorque(world: World, eid: number, torque: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = torque.x;
    stateVector.y = torque.y;
    stateVector.z = torque.z;
    bodyApplyTorque(state, body.index1 - 1, stateVector, wake);
}

/** Applies a one-shot world impulse in newton-seconds at a world point; off-center impulses also change angular velocity. `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyLinearImpulse(
    world: World,
    eid: number,
    impulse: Vec3,
    point: Pos,
    wake = true,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    statePos.x = point.x;
    statePos.y = point.y;
    statePos.z = point.z;
    bodyApplyLinearImpulse(state, body.index1 - 1, stateVector, statePos, wake);
}

/** Applies a one-shot center-of-mass impulse in newton-seconds; `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyLinearImpulseToCenter(
    world: World,
    eid: number,
    impulse: Vec3,
    wake = true,
): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    bodyApplyLinearImpulseToCenter(state, body.index1 - 1, stateVector, wake);
}

/** Applies a one-shot angular impulse in kilogram-meter-squared per second; `wake` defaults true. Box3D ignores static and kinematic bodies. */
export function applyAngularImpulse(world: World, eid: number, impulse: Vec3, wake = true): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (!state || !body) return;
    stateVector.x = impulse.x;
    stateVector.y = impulse.y;
    stateVector.z = impulse.z;
    bodyApplyAngularImpulse(state, body.index1 - 1, stateVector, wake);
}

/** Wakes a body and its island, or asks Box3D to put the whole island to sleep. */
export function setAwake(world: World, eid: number, awake: boolean): void {
    const runtime = runtimeFor(world);
    const state = runtime.physicsWorld;
    const body = runtime.bodies.get(eid);
    if (state && body) bodySetAwake(state, body.index1 - 1, awake);
}
export function physicsCounters(world: World): PhysicsCounters {
    return { ...runtimeFor(world).counters };
}

// The runtime's maps from entities to solver handles, as plain ids, so a restore rebinds them for its
// target World and the sync systems reconcile them against the live ECS on the next tick.
interface Bindings {
    /** The authoring configuration paired with the captured solver settings. */
    worldDefinition: PhysicsWorldDefinitionConfig;
    /** eid, solver body index1, solver generation, EntityRef */
    bodies: number[];
    /** eid, solver shape index1, generation, body eid, EntityRef */
    shapes: number[];
    /** the last applied authored Shape definition, copied for reusable recovery images */
    shapeRecreationState: [number, Float64Array][];
    /** pending shape field candidates as [eid, field mask] pairs */
    shapeCandidates: [number, number][];
    constraints: ConstraintIds;
    /** pending sync words, copied so later marks cannot mutate the image */
    changed: Uint32Array | null;
    /** pending body field candidates as [eid, field mask] pairs */
    bodyCandidates: [number, number][];
    full: boolean;
    /** geometry builds by registry id, re-paired with core's restored definitions */
    geometry: GeometryBuilds;
}

function warmWorld(runtime: PhysicsRuntime): WorldState {
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) throw new Error("physics: world is not warm");
    return physicsWorld;
}

function captureBodyCandidates(candidates: FieldCandidates): [number, number][] {
    const pending: [number, number][] = [];
    for (let i = 0; i < candidates.count; i++) {
        const eid = candidates.eids[i];
        pending.push([eid, candidates.fieldMarks[eid]]);
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
            runtime.shapeBodies.get(eid)!,
            runtime.shapeStamps.get(eid)!,
        );
    return {
        worldDefinition: copyPhysicsWorldDefinition(world.resource(PhysicsWorldDefinition)),
        bodies,
        shapes,
        shapeCandidates: captureBodyCandidates(runtime.shapeCandidates),
        shapeRecreationState: Array.from(runtime.shapeRecreationState, ([eid, values]) => [
            eid,
            values.slice(),
        ]),
        constraints: captureConstraints(runtime.constraints, warmWorld(runtime)),
        bodyCandidates: captureBodyCandidates(runtime.bodyCandidates),
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
    const state = physicsWorld;
    runtime.bodies.clear();
    runtime.shapes.clear();
    runtime.shapeBodies.clear();
    clearShapeEntities(runtime);
    runtime.shapeStamps.clear();
    runtime.shapeRecreationState.clear();
    for (let i = 0; i < b.bodies.length; i += 4) {
        const eid = b.bodies[i];
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
    for (const [eid, values] of b.shapeRecreationState)
        runtime.shapeRecreationState.set(eid, values.slice());
    restoreConstraints(runtime.constraints, b.constraints, physicsWorld);
    restoreGeometryBuilds(world, b.geometry);
    clearFieldCandidates(runtime.bodyCandidates);
    for (const [eid, fieldMask] of b.bodyCandidates) {
        markFieldCandidate(runtime.bodyCandidates, eid);
        runtime.bodyCandidates.fieldMarks[eid] = fieldMask;
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

function eventShapeBinding(
    runtime: PhysicsRuntime,
    k: ReturnType<typeof kernel>,
    worldId: number,
    kind: number,
    eventIndex: number,
    lane: number,
): ShapeEntityBinding | null {
    const id = k.eventWord(worldId, kind, eventIndex, lane) - 1;
    const generation = k.eventWord(worldId, kind, eventIndex, lane + 1) >>> 16;
    if (id < 0) return null;
    const bindings = runtime.shapeEntities.slots[id];
    if (!bindings) return null;
    for (let i = bindings.length - 1; i >= 0; --i)
        if (bindings[i].generation === generation) return bindings[i];
    return null;
}

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
    runtime: PhysicsRuntime,
    state: WorldState,
    list: PhysicsEventRecords<PhysicsSensorEvent>,
    kind: number,
): void {
    const k = kernel(state.ecsState);
    const count = eventCount(state, kind);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const sensor = eventShapeBinding(runtime, k, state.worldId, kind, i, 0);
        const visitor = eventShapeBinding(runtime, k, state.worldId, kind, i, 2);
        if (!sensor || !visitor) continue;
        const record = sensorRecordAt(list, written++);
        record.sensorShape = sensor.shape;
        record.sensorBody = sensor.body;
        record.visitorShape = visitor.shape;
        record.visitorBody = visitor.body;
    }
    list.count = written;
}

function fillContactRecords(
    runtime: PhysicsRuntime,
    state: WorldState,
    list: PhysicsEventRecords<PhysicsContactEvent>,
    kind: number,
): void {
    const k = kernel(state.ecsState);
    const count = eventCount(state, kind);
    let written = 0;
    for (let i = 0; i < count; i++) {
        const a = eventShapeBinding(runtime, k, state.worldId, kind, i, 0);
        const b = eventShapeBinding(runtime, k, state.worldId, kind, i, 2);
        if (!a || !b) continue;
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
        const a = eventShapeBinding(runtime, k, state.worldId, kind, i, 0);
        const b = eventShapeBinding(runtime, k, state.worldId, kind, i, 2);
        if (!a || !b) continue;
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
    fillSensorRecords(runtime, state, events.sensorBegin, EventKind.SensorBegin);
    fillSensorRecords(runtime, state, events.sensorEnd, EventKind.SensorEnd);
    fillContactRecords(runtime, state, events.contactBegin, EventKind.ContactBegin);
    fillContactRecords(runtime, state, events.contactEnd, EventKind.ContactEnd);
    fillHitRecords(runtime, state, events.contactHit);
    fillJointRecords(world, state, events.joints);
    pruneShapeEntities(runtime);
}

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepPhysicsSystem: System = {
    name: "step",
    group: "fixed",
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
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
            for (let i = 0; i < phaseFields.length; i++) {
                const ms = profile[phaseFields[i]];
                if (ms !== 0) record(phaseNames[i], ms);
            }
        }
        runtime.counters.bytesUploaded = 0;
        const global = world.storage(GlobalTransform);
        const rows = physicsWorld.bodyStore.movedRows();
        global.translation.writeEncoded(rows.eids, rows.pos);
        global.rotation.writeEncoded(rows.eids, rows.quat);
        global.linearVelocity.writeEncoded(rows.eids, rows.vel);
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

/** Captures authored field marks before the frame's clearChanges point. */
export const CollectPhysicsFieldCandidatesSystem: System = {
    name: "physics-field-candidates",
    group: "simulation",
    boundary: "after",
    update(world) {
        const runtime = runtimeFor(world);
        captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
        captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
        captureFieldCandidates(world, runtime.constraints.candidates, jointBindings(world));
    },
};

/** Captures draw writes before CPU-only World.step clears frame marks on return. */
export const CollectDrawPhysicsFieldCandidatesSystem: System = {
    name: "draw-physics-field-candidates",
    group: "draw",
    boundary: "after",
    update(world) {
        const runtime = runtimeFor(world);
        captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
        captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
        captureFieldCandidates(world, runtime.constraints.candidates, jointBindings(world));
    },
};

// query terms, held once so a steady sync mints none.
const BODY_TERMS = [Body];
const SHAPE_TERMS = [ShapeComponent];

// the stale walk's context, one per runtime: the eids of despawned bodies in the first `count` slots of a
// list that keeps its high-water capacity (truncating it releases the backing store; the next push regrows it).
interface StaleScan {
    world: World | null;
    eids: number[];
    count: number;
}

// the sync's map walk, given its scan as `this`, so a steady sync mints no iterator.
function collectStale(this: StaleScan, _body: unknown, eid: number): void {
    if (!this.world!.has(eid, Body)) this.eids[this.count++] = eid;
}

// membership-driven create/destroy, ascending eid order (world.query's natural order — creation order is
// load-bearing for solver determinism). Runs every fixed tick before the solve so a body spawned this frame
// joins THIS tick's step.
// one Body's reconciliation; true when it bound a new solver body
function visitBody(
    world: World,
    runtime: PhysicsRuntime,
    physicsWorld: WorldState,
    eid: number,
): boolean {
    const stamp = world.ref(eid);
    if (world.has(eid, Transform)) {
        const warning = runtime.placementWarnings.get(eid);
        if (!warning || !world.resolve(warning)) {
            console.warn(
                `physics-sync: entity ${eid} carries both Body and Transform; both write GlobalTransform`,
            );
            runtime.placementWarnings.set(eid, stamp);
        }
    }
    const bound = runtime.bodies.get(eid);
    if (bound) {
        if (world.resolve(bound.stamp)) return false;
        forget(runtime, eid); // recycled to a new Body in one update
    }
    const tb = marshalBody(world, physicsWorld, eid);
    kernel(world).bodySetEntity(tb.world0, tb.index1 - 1, eid);
    runtime.bodies.set(eid, { ...tb, stamp });
    clearFieldCandidate(runtime.bodyCandidates, eid);
    seedGlobalTransform(world, eid, physicsWorld, tb);
    resetInterpolation(world, eid);
    return true;
}

function removeShape(runtime: PhysicsRuntime, eid: number, updateBodyMass = true): void {
    const shape = runtime.shapes.get(eid);
    const state = runtime.physicsWorld;
    retireShapeEntity(runtime, shape);
    if (shape && state && shapeIsValid(state, shape))
        destroyShape(state, shape.index1 - 1, updateBodyMass);
    runtime.shapes.delete(eid);
    runtime.shapeBodies.delete(eid);
    runtime.shapeStamps.delete(eid);
    runtime.shapeRecreationState.delete(eid);
    runtime.warnedShapes.delete(eid);
}

function markAllShapeCandidates(world: World, runtime: PhysicsRuntime): void {
    for (const eid of world.query(SHAPE_TERMS)) markFieldCandidate(runtime.shapeCandidates, eid);
}

function warnShape(runtime: PhysicsRuntime, eid: number, error: unknown): void {
    if (runtime.warnedShapes.has(eid)) return;
    runtime.warnedShapes.add(eid);
    console.warn(`[physics] Shape ${eid} was not attached: ${String(error)}`);
}

function syncShapes(world: World, runtime: PhysicsRuntime): void {
    const state = runtime.physicsWorld;
    if (!state) return;
    const candidates = runtime.shapeCandidates;
    const binding = shapeBinding(world);
    const count = candidates.count;
    let pending = 0;
    for (let i = 0; i < count; i++) {
        const eid = candidates.eids[i];
        const mask = candidates.fieldMarks[eid];
        if (!world.has(eid, ShapeComponent)) {
            removeShape(runtime, eid);
            candidates.seen[eid] = 0;
            candidates.fieldMarks[eid] = 0;
            continue;
        }
        const updateBodyMass = binding.storage.updateBodyMass.column[eid] !== 0;
        const previousStamp = runtime.shapeStamps.get(eid);
        if (previousStamp !== undefined && !world.resolve(previousStamp))
            removeShape(runtime, eid, updateBodyMass);

        const rawBody = (binding.storage.body.column as Float64Array)[eid] as EntityRef;
        const bodyEid = rawBody === 0 ? eid : world.resolve(rawBody);
        if (runtime.shapeBodies.has(eid) && runtime.shapeBodies.get(eid) !== bodyEid)
            removeShape(runtime, eid, updateBodyMass);
        if (!bodyEid) {
            removeShape(runtime, eid, updateBodyMass);
            warnShape(runtime, eid, new Error("body reference is no longer valid"));
            candidates.seen[eid] = 0;
            candidates.fieldMarks[eid] = 0;
            continue;
        }
        if (!world.has(bodyEid, Body)) {
            candidates.eids[pending++] = eid;
            continue;
        }
        const body = runtime.bodies.get(bodyEid);
        if (!body || !world.resolve(body.stamp)) {
            candidates.eids[pending++] = eid;
            continue;
        }

        let shape = runtime.shapes.get(eid);
        if (shape && !shapeIsValid(state, shape)) {
            removeShape(runtime, eid, updateBodyMass);
            shape = undefined;
        }
        if (shape) {
            const previous = runtime.shapeRecreationState.get(eid);
            try {
                if (shapeNeedsRecreation(world, binding, eid, mask, previous, shape, state)) {
                    removeShape(runtime, eid, updateBodyMass);
                    shape = undefined;
                } else {
                    syncShapeFields(world, state, binding, eid, mask, shape);
                }
            } catch (error) {
                warnShape(runtime, eid, error);
                candidates.eids[pending++] = eid;
                continue;
            }
        }
        if (!shape) {
            try {
                shape = marshalShape(world, state, body, eid);
                runtime.shapes.set(eid, shape);
                runtime.shapeBodies.set(eid, bodyEid);
                runtime.shapeStamps.set(eid, runtime.shapeStamps.get(eid) ?? world.ref(eid));
                runtime.warnedShapes.delete(eid);
            } catch (error) {
                warnShape(runtime, eid, error);
                candidates.eids[pending++] = eid;
                continue;
            }
        }
        rememberShapeEntity(world, runtime, shape, eid, bodyEid);
        runtime.shapeRecreationState.set(
            eid,
            captureShapeRecreationState(binding, eid, runtime.shapeRecreationState.get(eid)),
        );
        candidates.seen[eid] = 0;
        candidates.fieldMarks[eid] = 0;
    }
    candidates.count = pending;
}

const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem, StepPhysicsSystem],
    update(world: World) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        captureFieldCandidate(world, runtime.bodyCandidates, bodyBinding(world));
        captureFieldCandidate(world, runtime.shapeCandidates, shapeBinding(world));
        let bodySetChanged = false;
        let ended = false;
        if (runtime.full) {
            runtime.full = false;
            runtime.changed.fill(0);
            runtime.anyChanged = false;
            let bound = 0;
            for (const eid of world.query(BODY_TERMS)) {
                if (visitBody(world, runtime, physicsWorld, eid)) bodySetChanged = true;
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
                            if (visitBody(world, runtime, physicsWorld, eid)) bodySetChanged = true;
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
                forget(runtime, eid);
                bodySetChanged = true;
            }
        }
        syncBodyFields(
            physicsWorld,
            runtime.bodyCandidates,
            bodyBinding(world),
            runtime.bodies,
            runtime.warnedBodyFields,
        );
        if (bodySetChanged) markAllShapeCandidates(world, runtime);
        syncShapes(world, runtime);
        clearFieldCandidates(runtime.bodyCandidates);
        runtime.counters.bodiesVisited = runtime.bodies.size;
        if (bodySetChanged)
            markAllJointCandidates(world, runtime.constraints, jointBindings(world));
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
 * Rigid-body physics over shared bodies and the nine joint kinds, with a CPU solver (Rust/WASM kernel, no GPU device
 * needed to step). Opt-in — add it to a scene to run physics (it's not in the default plugins).
 * Mesh/heightfield/compound colliders, sensors, CCD and sleeping ride {@link physicsWorld}.
 *
 * @example
 * ```
 * export const config: AppConfig = { plugins: [StandardPhysicsPlugin] };
 * ```
 */
export const StandardPhysicsPlugin: Plugin = {
    name: "StandardPhysics",
    recovery: (world) => ({
        snapshot: () => capturePhysics(world),
        restore: (image: PhysicsSnapshot) => recoverPhysics(world, image),
    }),
    dependencies: [PhysicsPlugin],
    systems: [
        SyncSystem,
        SyncPhysicsWorldDefinitionSystem,
        SyncPhysicsConstraintsSystem,
        StepPhysicsSystem,
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
            world.onDispose(world.observeMembership(Body, (eid) => markChanged(runtime, eid)));
            world.onDispose(
                world.observeMembership(ShapeComponent, (eid) =>
                    markFieldCandidate(runtime.shapeCandidates, eid),
                ),
            );
            const bindings = jointBindings(world);
            for (const binding of bindings) {
                world.onDispose(
                    world.observeMembership(binding.component, (eid) =>
                        markFieldCandidate(runtime.constraints.candidates[binding.index], eid),
                    ),
                );
            }
            world.onDispose(
                world.observeMembership(Transform, (eid, present) => {
                    if (present && world.has(eid, Body)) markChanged(runtime, eid);
                }),
            );
        }
    },

    async warm(world) {
        const runtime = runtimeFor(world);
        await init(world); // async wasm compile — the browser main thread can't compile it synchronously
        if (runtime.physicsWorld) destroyWorld(runtime.physicsWorld);
        const definition = world.resource(PhysicsWorldDefinition);
        const id = createWorld(world, {
            ...defaultWorldDef(),
            ...solverWorldDefinition(definition),
        });
        runtime.physicsWorld = getWorld(id)!;
        syncWorldDefinition(runtime.physicsWorld, definition);
        clearBodies(runtime);
        clearPhysicsEvents(world.resource(PhysicsEvents));
    },

    dispose(world) {
        const runtime = runtimeFor(world);
        clearBodies(runtime);
        clearPhysicsEvents(world.resource(PhysicsEvents));
        if (runtime.physicsWorld) destroyWorld(runtime.physicsWorld);
        runtime.physicsWorld = null;
        void shutdown(world);
    },
};

const phaseFields = PROFILE_FIELDS.filter((field) => field !== "step");
const phaseNames = phaseFields.map(
    (field) => `${StandardPhysicsPlugin.name}/${StepPhysicsSystem.name}/${field}`,
);
