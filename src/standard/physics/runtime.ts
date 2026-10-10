/// <reference types="@webgpu/types" />

import { Body, type BodyState, BodyType, Hulls, PhysicsPlugin } from "../../core/physics";
import { GlobalTransform, Transform, teleport as teleportPlacement } from "../../core/transform";
import { type EntityRef, type Plugin, type System, Time, type World } from "../../engine";
import {
    hash as hashWorld,
    init,
    type PhysicsSnapshot,
    PhysicsWorld,
    restore as restoreWorld,
    Body as SolverBody,
    shutdown,
    snapshot as snapshotWorld,
} from "./api";
import { snapshotBindings } from "./api/snapshot";
import { jointBindings } from "./authoring";
import { f32, maxf } from "./common/math";
import type { WorldDef } from "./common/types";
import {
    type ConstraintCache,
    type ConstraintIds,
    captureConstraints,
    captureJointFieldCandidates,
    createConstraintCache,
    markAllJointCandidates,
    markJointCandidate,
    resetConstraints,
    restoreConstraints,
    syncJoints,
} from "./joints";
import { kernel } from "./kernel/kernel";
import { marshalBody } from "./marshal";
import { PROFILE_FIELDS, readStepProfile } from "./world/profile";
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
    physicsWorld: PhysicsWorld | null;
    bodies: Map<number, SolverBody>;
    stamps: Map<number, EntityRef>;
    placementWarnings: Map<number, EntityRef>;
    failed: Map<number, { stamp: EntityRef; hulls: number }>;
    constraints: ConstraintCache;

    // whether a body's marshal failed, so the constraint uploads defer its joints; made once per runtime
    isFailed: (eid: number) => boolean;
    stale: StaleScan;
    counters: PhysicsCounters;
    // eids whose Body membership, or a Body's Transform, changed since the last sync, as 32-bit words
    changed: Uint32Array;
    anyChanged: boolean;
    // only warm sets this to make the next sync walk every Body
    full: boolean;
    hulls: number;
    observing: boolean;
}

const physicsRuntimeKey = { create: newRuntime };

function newRuntime(): PhysicsRuntime {
    const failed: PhysicsRuntime["failed"] = new Map();
    return {
        initialized: false,
        physicsWorld: null,
        bodies: new Map(),
        stamps: new Map(),
        placementWarnings: new Map(),
        failed,
        constraints: createConstraintCache(),

        isFailed: (eid) => failed.has(eid),
        stale: { world: null, eids: [], count: 0 },
        counters: { bodiesVisited: 0, bytesUploaded: 0 },
        changed: new Uint32Array(64),
        anyChanged: false,
        full: true,
        hulls: 0,
        observing: false,
    };
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
    world.storage(GlobalTransform).translation.set(eid, pos[0], pos[1], pos[2], 0);
    world.storage(GlobalTransform).rotation.set(eid, quat[0], quat[1], quat[2], quat[3]);
    world.storage(GlobalTransform).linearVelocity.set(eid, vel[0], vel[1], vel[2], 0);
}

function seedGlobalTransform(world: World, eid: number): void {
    writeGlobalTransform(
        world,
        eid,
        [
            world.storage(Body).position.x.get(eid),
            world.storage(Body).position.y.get(eid),
            world.storage(Body).position.z.get(eid),
        ],
        [
            world.storage(Body).rotation.x.get(eid),
            world.storage(Body).rotation.y.get(eid),
            world.storage(Body).rotation.z.get(eid),
            world.storage(Body).rotation.w.get(eid),
        ],
        [0, 0, 0],
    );
}

function forget(runtime: PhysicsRuntime, eid: number): void {
    runtime.bodies.get(eid)?.destroy();
    runtime.bodies.delete(eid);
}

function clearBodies(runtime: PhysicsRuntime): void {
    for (const body of runtime.bodies.values()) body.destroy();
    runtime.bodies.clear();
    runtime.stamps.clear();
    runtime.failed.clear();
    runtime.counters = { bodiesVisited: 0, bytesUploaded: 0 };
    runtime.full = true;
    resetConstraints(runtime.constraints);
}

/**
 * World-owned physics accessors. `physicsWorld(state)` exposes imperative joints, sensors,
 * contact/hit events, mesh/heightfield/compound colliders and native queries. It is
 * `null` until {@link StandardPhysicsPlugin} warms. Its `getBody(eid)` resolves an authored
 * `Body` entity to a live solver handle for joint creation, or null before marshaling.
 * The pose functions are no-ops before warm.
 */

export function physicsWorld(world: World): PhysicsWorld | null {
    const physicsWorld = runtimeFor(world).physicsWorld;
    return physicsWorld ? physicsWorld : null;
}
/** a writable {@link BodyState} that {@link readBody} fills in place. */
export interface BodyStateOut {
    position: [number, number, number];
    rotation: [number, number, number, number];
    linearVelocity: [number, number, number];
}

/** one body's live pose + velocity, or null when `eid` has no solver body. Pass `out` to fill it instead of allocating. */
export function readBody(world: World, eid: number, out?: BodyStateOut): BodyState | null {
    const tb = runtimeFor(world).bodies.get(eid);
    if (!tb) return null;
    const global = world.storage(GlobalTransform);
    if (out === undefined)
        return readBody(world, eid, {
            position: [0, 0, 0],
            rotation: [0, 0, 0, 1],
            linearVelocity: [0, 0, 0],
        });
    const offset = eid * 4;
    const p = global.translation.column,
        q = global.rotation.column,
        v = global.linearVelocity.column;
    out.position[0] = p[offset];
    out.position[1] = p[offset + 1];
    out.position[2] = p[offset + 2];
    out.rotation[0] = q[offset];
    out.rotation[1] = q[offset + 1];
    out.rotation[2] = q[offset + 2];
    out.rotation[3] = q[offset + 3];
    out.linearVelocity[0] = v[offset];
    out.linearVelocity[1] = v[offset + 1];
    out.linearVelocity[2] = v[offset + 2];
    return out;
}

const staticMotionWarnings = { create: () => new Map<number, EntityRef>() };
// Body calls copy these reusable views synchronously.
const kinPos = { x: 0, y: 0, z: 0 };
const kinQuat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const kinTarget = { p: kinPos, q: kinQuat };
const kinVel = { x: 0, y: 0, z: 0 };

/** Drives a kinematic or dynamic body toward a pose over one fixed step; Box3D derives both velocities. A teleport preserves velocity and discards interpolation. Static bodies are ignored with one warning per entity; an unavailable body is ignored. */
export function setKinematic(
    world: World,
    eid: number,
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    teleport = false,
): void {
    const runtime = runtimeFor(world);
    const tb = runtime.bodies.get(eid);
    if (!tb) return;
    if (tb.getType() === BodyType.Static) {
        const warned = world.resource(staticMotionWarnings);
        const previous = warned.get(eid);
        if (!previous || !world.resolve(previous)) {
            warned.set(eid, world.ref(eid));
            console.warn(`[physics] setKinematic ignores static body entity ${eid}`);
        }
        return;
    }
    kinPos.x = pos[0];
    kinPos.y = pos[1];
    kinPos.z = pos[2];
    kinQuat.v.x = quat[0];
    kinQuat.v.y = quat[1];
    kinQuat.v.z = quat[2];
    kinQuat.s = quat[3];
    if (teleport) tb.setTransform(kinPos, kinQuat);
    else {
        tb.setTargetTransform(kinTarget, fixedDeltaTime, true);
        // Box3D leaves sub-threshold target motion asleep without adopting its pose.
        if (!tb.isAwake()) return;
    }
    tb.getLinearVelocity(kinVel);
    // GlobalTransformHistory can upload this row on a draw-only frame before the next solver step.
    const global = world.storage(GlobalTransform);
    const offset = eid * 4;
    const pc = global.translation.column,
        qc = global.rotation.column,
        vc = global.linearVelocity.column;
    for (let lane = 0; lane < 3; lane++) pc[offset + lane] = pos[lane];
    for (let lane = 0; lane < 4; lane++) qc[offset + lane] = quat[lane];
    vc[offset] = kinVel.x;
    vc[offset + 1] = kinVel.y;
    vc[offset + 2] = kinVel.z;
    global.translation.markChanged(eid);
    global.rotation.markChanged(eid);
    global.linearVelocity.markChanged(eid);
    if (teleport) teleportPlacement(world, eid);
}
export function setVelocity(world: World, eid: number, vx: number, vy: number, vz: number): void {
    const body = runtimeFor(world).bodies.get(eid);
    if (!body) return;
    body.setLinearVelocity({ x: vx, y: vy, z: vz });
    world.storage(GlobalTransform).linearVelocity.set(eid, vx, vy, vz, 0);
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
    /** eid, EntityRef, hull count */
    failed: number[];
    constraints: ConstraintIds;
    /** pending sync words, copied so later marks cannot mutate the image */
    changed: Uint32Array | null;
    full: boolean;
    hulls: number;
}

function warmWorld(runtime: PhysicsRuntime): PhysicsWorld {
    const physicsWorld = runtime.physicsWorld;
    if (!physicsWorld) throw new Error("physics: world is not warm");
    return physicsWorld;
}

function captureBindings(world: World, runtime: PhysicsRuntime): Bindings {
    const bodies: number[] = [];
    for (const [eid, body] of runtime.bodies)
        bodies.push(eid, body.id.index1, body.id.generation, runtime.stamps.get(eid)!);
    const failed: number[] = [];
    for (const [eid, f] of runtime.failed) failed.push(eid, f.stamp, f.hulls);
    return {
        worldDefinition: copyPhysicsWorldDefinition(world.resource(PhysicsWorldDefinition)),
        bodies,
        failed,
        constraints: captureConstraints(runtime.constraints),
        changed: runtime.anyChanged ? runtime.changed.slice() : null,
        full: runtime.full,
        hulls: runtime.hulls,
    };
}

function restoreBindings(
    world: World,
    runtime: PhysicsRuntime,
    physicsWorld: PhysicsWorld,
    b: Bindings,
): void {
    Object.assign(
        world.resource(PhysicsWorldDefinition),
        copyPhysicsWorldDefinition(b.worldDefinition),
    );
    const state = physicsWorld.state;
    runtime.bodies.clear();
    runtime.stamps.clear();
    for (let i = 0; i < b.bodies.length; i += 4) {
        const eid = b.bodies[i];
        runtime.bodies.set(
            eid,
            new SolverBody(state, {
                index1: b.bodies[i + 1],
                world0: state.worldId,
                generation: b.bodies[i + 2],
            }),
        );
        runtime.stamps.set(eid, b.bodies[i + 3] as EntityRef);
    }
    runtime.failed.clear();
    for (let i = 0; i < b.failed.length; i += 3)
        runtime.failed.set(b.failed[i], {
            stamp: b.failed[i + 1] as EntityRef,
            hulls: b.failed[i + 2],
        });
    restoreConstraints(runtime.constraints, b.constraints, physicsWorld);
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
    runtime.hulls = b.hulls;
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

function syncWorldDefinition(
    physicsWorld: PhysicsWorld,
    definition: PhysicsWorldDefinitionConfig,
): void {
    const state = physicsWorld.state;
    const gravity = definition.gravity;
    if (
        state.gravity.x !== f32(gravity.x) ||
        state.gravity.y !== f32(gravity.y) ||
        state.gravity.z !== f32(gravity.z)
    )
        physicsWorld.setGravity(gravity);

    const restitutionThreshold = maxf(0, f32(definition.restitutionThreshold));
    if (state.restitutionThreshold !== restitutionThreshold)
        physicsWorld.setRestitutionThreshold(definition.restitutionThreshold);
    const hitEventThreshold = maxf(0, f32(definition.hitEventThreshold));
    if (state.hitEventThreshold !== hitEventThreshold)
        physicsWorld.setHitEventThreshold(definition.hitEventThreshold);
    const contactHertz = maxf(0, f32(definition.contactHertz));
    const contactDampingRatio = maxf(0, f32(definition.contactDampingRatio));
    const contactSpeed = maxf(0, f32(definition.contactSpeed));
    if (
        state.contactHertz !== contactHertz ||
        state.contactDampingRatio !== contactDampingRatio ||
        state.contactSpeed !== contactSpeed
    )
        physicsWorld.setContactTuning(
            definition.contactHertz,
            definition.contactDampingRatio,
            definition.contactSpeed,
        );
    const recycleDistance = maxf(0, f32(definition.contactRecycleDistance));
    if (state.contactRecycleDistance !== recycleDistance)
        physicsWorld.setContactRecycleDistance(definition.contactRecycleDistance);
    const maximumLinearSpeed = f32(definition.maximumLinearSpeed);
    if (state.maxLinearSpeed !== maximumLinearSpeed)
        physicsWorld.setMaximumLinearSpeed(definition.maximumLinearSpeed);
    if (state.enableSleep !== definition.enableSleep)
        physicsWorld.enableSleeping(definition.enableSleep);
    if (state.enableContinuous !== definition.enableContinuous)
        physicsWorld.enableContinuous(definition.enableContinuous);
    if (state.enableWarmStarting !== definition.enableWarmStarting)
        physicsWorld.enableWarmStarting(definition.enableWarmStarting);
    if (state.enableSpeculative !== definition.enableSpeculative)
        physicsWorld.enableSpeculative(definition.enableSpeculative);
    if (state.worldCustomFilterCallback !== definition.customFilterCallback)
        physicsWorld.setWorldCustomFilterCallback(definition.customFilterCallback);
    if (state.worldPreSolveCallback !== definition.preSolveCallback)
        physicsWorld.setWorldPreSolveCallback(definition.preSolveCallback);
    if (state.worldFrictionCallback !== definition.frictionCallback)
        physicsWorld.setWorldFrictionCallback(definition.frictionCallback);
    if (state.worldRestitutionCallback !== definition.restitutionCallback)
        physicsWorld.setWorldRestitutionCallback(definition.restitutionCallback);
}

function capturePhysics(world: World): PhysicsSnapshot {
    const runtime = runtimeFor(world);
    captureJointFieldCandidates(world, runtime.constraints, jointBindings(world));
    return snapshotWorld(warmWorld(runtime), captureBindings(world, runtime));
}
function recoverPhysics(world: World, saved: PhysicsSnapshot): void {
    const runtime = runtimeFor(world);
    const physicsWorld = warmWorld(runtime);
    const bindings = snapshotBindings(saved) as Bindings | undefined;
    if (bindings === undefined) throw new Error("physics: recovery image has no bindings");
    restoreWorld(physicsWorld, saved);
    restoreBindings(world, runtime, physicsWorld, bindings);
    // ECS poses were restored by World. Warning latches are presentation, counters and stale
    // scans are overwritten on each step; solver history and binding caches are the participant.
}
export function hashPhysics(world: World): bigint {
    return hashWorld(warmWorld(runtimeFor(world)));
}

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepPhysicsSystem: System = {
    name: "step",
    group: "fixed",
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        physicsWorld.step(fixedDeltaTime, world.resource(PhysicsWorldDefinition).subStepCount);
        const record = world.recordSink;
        if (record) {
            readStepProfile(physicsWorld.state);
            const profile = physicsWorld.state.profile;
            for (let i = 0; i < phaseFields.length; i++) {
                const ms = profile[phaseFields[i]];
                if (ms !== 0) record(phaseNames[i], ms);
            }
        }
        runtime.counters.bytesUploaded = 0;
        const global = world.storage(GlobalTransform);
        const rows = physicsWorld.state.bodyStore.movedRows();
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
        captureJointFieldCandidates(world, runtime.constraints, bindings);
        syncJoints(
            runtime.constraints,
            physicsWorld,
            runtime.bodies,
            world,
            bindings,
            runtime.isFailed,
        );
    },
};

/** Captures joint field dirty words before the frame's clearChanges point. */
export const CollectJointFieldCandidatesSystem: System = {
    name: "joint-field-candidates",
    group: "simulation",
    boundary: "after",
    update(world) {
        const runtime = runtimeFor(world);
        captureJointFieldCandidates(world, runtime.constraints, jointBindings(world));
    },
};

/** Captures draw writes before CPU-only World.step clears frame marks on return. */
export const CollectDrawJointFieldCandidatesSystem: System = {
    name: "draw-joint-field-candidates",
    group: "draw",
    boundary: "after",
    update(world) {
        const runtime = runtimeFor(world);
        captureJointFieldCandidates(world, runtime.constraints, jointBindings(world));
    },
};

// query terms, held once so a steady sync mints none.
const BODY_TERMS = [Body];

// the stale walk's context, one per runtime: the eids of despawned bodies in the first `count` slots of a
// list that keeps its high-water capacity (truncating it releases the backing store; the next push regrows it).
interface StaleScan {
    world: World | null;
    eids: number[];
    count: number;
}

// the sync's map walks, given the World as `this`, so a steady sync mints no iterator.
function dropDespawnedFailure(
    this: World,
    _failure: unknown,
    eid: number,
    failed: Map<number, unknown>,
): void {
    if (!this.has(eid, Body)) failed.delete(eid);
}

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
    physicsWorld: PhysicsWorld,
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
    if (runtime.bodies.has(eid)) {
        if (world.resolve(runtime.stamps.get(eid)!)) return false;
        forget(runtime, eid); // recycled to a new Body in one update
    }
    const f = runtime.failed.get(eid);
    if (f && world.resolve(f.stamp) && f.hulls === world.resource(Hulls).size) return false;
    const tb = marshalBody(world, physicsWorld, eid);
    if (!tb) {
        runtime.failed.set(eid, { stamp, hulls: world.resource(Hulls).size });
        return false;
    }
    runtime.failed.delete(eid);
    kernel(world).bodySetEntity(tb.id.world0, tb.id.index1 - 1, eid);
    runtime.bodies.set(eid, tb);
    runtime.stamps.set(eid, stamp);
    seedGlobalTransform(world, eid);
    teleportPlacement(world, eid);
    return true;
}

function markFailed(this: PhysicsRuntime, _failure: unknown, eid: number): void {
    markChanged(this, eid);
}

const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem, StepPhysicsSystem],
    update(world: World) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        let bodySetChanged = false;
        let ended = false;
        const hulls = world.resource(Hulls).size;
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
            ended = bound !== runtime.bodies.size || runtime.failed.size !== 0;
        } else {
            if (hulls !== runtime.hulls) runtime.failed.forEach(markFailed, runtime);
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
                        } else if (runtime.bodies.has(eid) || runtime.failed.has(eid)) ended = true;
                    }
                }
            }
        }
        runtime.hulls = hulls;
        if (ended) {
            runtime.failed.forEach(dropDespawnedFailure, world);
            const stale = runtime.stale;
            stale.world = world;
            stale.count = 0;
            runtime.bodies.forEach(collectStale, stale);
            stale.world = null;
            for (let i = 0; i < stale.count; i++) {
                const eid = stale.eids[i];
                forget(runtime, eid);
                runtime.stamps.delete(eid);
                bodySetChanged = true;
            }
        }
        runtime.counters.bodiesVisited = runtime.bodies.size + runtime.failed.size;
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
        CollectJointFieldCandidatesSystem,
        CollectDrawJointFieldCandidatesSystem,
    ],

    initialize(world) {
        const runtime = world.resource(physicsRuntimeKey);
        world.resource(PhysicsWorldDefinition);
        runtime.initialized = true;
        if (!runtime.observing) {
            runtime.observing = true;
            world.onDispose(world.observeMembership(Body, (eid) => markChanged(runtime, eid)));
            const bindings = jointBindings(world);
            for (const binding of bindings) {
                world.onDispose(
                    world.observeMembership(binding.component, (eid) =>
                        markJointCandidate(runtime.constraints, binding.index, eid),
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
        runtime.physicsWorld?.destroy();
        const definition = world.resource(PhysicsWorldDefinition);
        runtime.physicsWorld = new PhysicsWorld(solverWorldDefinition(definition), world, (eid) => {
            const ref = runtime.stamps.get(eid);
            if (!world.has(eid, Body) || !ref || !world.resolve(ref)) return null;
            return runtime.bodies.get(eid) ?? null;
        });
        syncWorldDefinition(runtime.physicsWorld, definition);
        clearBodies(runtime);
    },

    dispose(world) {
        const runtime = runtimeFor(world);
        clearBodies(runtime);
        runtime.physicsWorld?.destroy();
        runtime.physicsWorld = null;
        void shutdown(world);
    },
};

const phaseFields = PROFILE_FIELDS.filter((field) => field !== "step");
const phaseNames = phaseFields.map(
    (field) => `${StandardPhysicsPlugin.name}/${StepPhysicsSystem.name}/${field}`,
);
