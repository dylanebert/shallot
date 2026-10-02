/// <reference types="@webgpu/types" />

import {
    Body,
    type BodyState,
    Hulls,
    Joint,
    PhysicsPlugin,
    ShapeKind,
    Spring,
} from "../../core/physics";
import {
    GlobalTransform,
    type Plugin,
    type System,
    Time,
    Transform,
    type World,
} from "../../engine";
import {
    hash as hashWorld,
    init,
    PhysicsWorld,
    restore as restoreWorld,
    type Body as SolverBody,
    shutdown,
    snapshot as snapshotWorld,
    type WorldSnapshot,
} from "./api";
import {
    type ConstraintCache,
    createConstraintCache,
    resetConstraints,
    resyncConstraints,
    syncJoints,
    syncSprings,
} from "./joints";
import { kernel } from "./kernel/kernel";
import { marshalBody } from "./marshal";

/** an authored spring: two body eids + local anchors + stiffness/rest, derived from the world's {@link Spring} entities by {@link springDefs}. */
export interface SpringDef {
    a: number;
    b: number;
    localAnchorA: readonly [number, number, number];
    localAnchorB: readonly [number, number, number];
    stiffness: number;
    rest: number;
}

/** an authored joint: two body eids + local anchors + the angular lock, derived from the world's {@link Joint} entities by {@link jointDefs}. Richer joints (motors, limits, the nine solver joint types) ride {@link physicsWorld}. */
export interface JointDef {
    a: number;
    b: number;
    localAnchorA: readonly [number, number, number];
    localAnchorB: readonly [number, number, number];
    stiffnessAng: number;
}

const FNV_BASIS = 2166136261;
const fold = (h: number, v: number): number => Math.imul(h ^ (v >>> 0), 16777619);
const _sigF32 = new Float32Array(1);
const _sigU32 = new Uint32Array(_sigF32.buffer);
const sigBits = (x: number): number => {
    _sigF32[0] = x;
    return _sigU32[0];
};

// the signature query terms, held once so the per-step signature walk mints no array.
const SPRING_TERMS = [Spring];
const JOINT_TERMS = [Joint];

// FNV_BASIS is the empty-set signature, so an unconstrained scene's first frame already matches → no upload.
/** re-arm the warning dedupe when a plugin world is warmed. Signatures themselves are World-owned. */
const signatureWarningsKey = {
    create: () => ({
        joints: new Set<number>(),
        springs: new Set<number>(),
    }),
};

function signatureWarningsFor(world: World) {
    return world.resource(signatureWarningsKey);
}

export function resetSignatures(world: World): void {
    (() => {
        const warnings = signatureWarningsFor(world);
        warnings.joints.clear();
        warnings.springs.clear();
        resetConstraints(world.resource(physicsRuntimeKey).constraints);
    })();
}

/** a hash of the authored {@link Spring} set, endpoint create-stamps included: an uploader re-uploads only when it changes. */
function springSignature(world: World): number {
    let h = FNV_BASIS;
    for (const eid of world.query(SPRING_TERMS)) {
        h = fold(h, eid);
        const a = world.storage(Spring).a.get(eid);
        const b = world.storage(Spring).b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps: a same-update realias of an endpoint (destroy+create
        // recycling its eid) leaves a/b unchanged, so without the stamp the re-upload is suppressed and the
        // solver joint pins the NEW occupant at the old anchors.
        h = fold(h, world.generation(a));
        h = fold(h, world.generation(b));
        h = fold(h, sigBits(world.storage(Spring).localAnchorA.x.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).localAnchorA.y.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).localAnchorA.z.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).localAnchorB.x.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).localAnchorB.y.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).localAnchorB.z.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).stiffness.get(eid)));
        h = fold(h, sigBits(world.storage(Spring).rest.get(eid)));
    }
    return h;
}

/** a hash of the authored {@link Joint} set, endpoint create-stamps included: the {@link springSignature} twin. */
function jointSignature(world: World): number {
    let h = FNV_BASIS;
    for (const eid of world.query(JOINT_TERMS)) {
        h = fold(h, eid);
        const a = world.storage(Joint).a.get(eid);
        const b = world.storage(Joint).b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps — see springSignature: a realias of an endpoint must
        // force the re-upload so the solver joint rebinds to the new occupant.
        h = fold(h, world.generation(a));
        h = fold(h, world.generation(b));
        h = fold(h, sigBits(world.storage(Joint).localAnchorA.x.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).localAnchorA.y.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).localAnchorA.z.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).localAnchorB.x.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).localAnchorB.y.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).localAnchorB.z.get(eid)));
        h = fold(h, sigBits(world.storage(Joint).stiffnessAng.get(eid)));
    }
    return h;
}

/** the authored {@link Spring} set as {@link SpringDef}s, dropping (and warning once for) a negative or NaN stiffness. */
export function springDefs(world: World): SpringDef[] {
    return (() => {
        const warnings = signatureWarningsFor(world);
        const out: SpringDef[] = [];
        for (const eid of world.query([Spring])) {
            const stiffness = world.storage(Spring).stiffness.get(eid);
            // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
            // 0 and ∞ are valid authored values (0 = non-positive → downstream skip; ∞ = rigid) — only negative
            // and NaN are rejected, at the authoring layer so every solver inherits one behavior.
            if (Number.isNaN(stiffness) || stiffness < 0) {
                if (!warnings.springs.has(eid)) {
                    console.warn(
                        `[physics] spring (a: ${world.storage(Spring).a.get(eid)}, b: ${world.storage(Spring).b.get(eid)}) has negative or NaN stiffness — skipped`,
                    );
                    warnings.springs.add(eid);
                }
                continue;
            }
            warnings.springs.delete(eid);
            out.push({
                a: world.storage(Spring).a.get(eid),
                b: world.storage(Spring).b.get(eid),
                localAnchorA: [
                    world.storage(Spring).localAnchorA.x.get(eid),
                    world.storage(Spring).localAnchorA.y.get(eid),
                    world.storage(Spring).localAnchorA.z.get(eid),
                ],
                localAnchorB: [
                    world.storage(Spring).localAnchorB.x.get(eid),
                    world.storage(Spring).localAnchorB.y.get(eid),
                    world.storage(Spring).localAnchorB.z.get(eid),
                ],
                stiffness,
                rest: world.storage(Spring).rest.get(eid),
            });
        }
        return out;
    })();
}

/** the authored {@link Joint} set as {@link JointDef}s, dropping (and warning once for) a negative or NaN angular stiffness. */
export function jointDefs(world: World): JointDef[] {
    return (() => {
        const warnings = signatureWarningsFor(world);
        const out: JointDef[] = [];
        for (const eid of world.query([Joint])) {
            const stiffnessAng = world.storage(Joint).stiffnessAng.get(eid);
            // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
            // 0 (spherical) and ∞ (fixed) are valid authored values — only negative and NaN are rejected, at the
            // authoring layer so every solver inherits one behavior.
            if (Number.isNaN(stiffnessAng) || stiffnessAng < 0) {
                if (!warnings.joints.has(eid)) {
                    console.warn(
                        `[physics] joint (a: ${world.storage(Joint).a.get(eid)}, b: ${world.storage(Joint).b.get(eid)}) has negative or NaN angular stiffness — skipped`,
                    );
                    warnings.joints.add(eid);
                }
                continue;
            }
            warnings.joints.delete(eid);
            out.push({
                a: world.storage(Joint).a.get(eid),
                b: world.storage(Joint).b.get(eid),
                localAnchorA: [
                    world.storage(Joint).localAnchorA.x.get(eid),
                    world.storage(Joint).localAnchorA.y.get(eid),
                    world.storage(Joint).localAnchorA.z.get(eid),
                ],
                localAnchorB: [
                    world.storage(Joint).localAnchorB.x.get(eid),
                    world.storage(Joint).localAnchorB.y.get(eid),
                    world.storage(Joint).localAnchorB.z.get(eid),
                ],
                stiffnessAng,
            });
        }
        return out;
    })();
}

const fixedDeltaTime = Time.FIXED_DT;
const GRAVITY = -10;
const SUBSTEPS = 4; // the solver's own recommended sub-step count (World.step's default)

export interface PhysicsCounters {
    bodiesVisited: number;
    bytesUploaded: number;
}

interface PhysicsRuntime {
    initialized: boolean;
    physicsWorld: PhysicsWorld | null;
    bodies: Map<number, SolverBody>;
    stamps: Map<number, number>;
    placementWarnings: Map<number, number>;
    kinPrev: Map<number, { pos: [number, number, number]; quat: [number, number, number, number] }>;
    failed: Map<number, { stamp: number; hulls: number }>;
    constraints: ConstraintCache;

    // whether a body's marshal failed, so the constraint uploads defer its joints; made once per runtime
    isFailed: (eid: number) => boolean;
    stale: StaleScan;
    counters: PhysicsCounters;
    springSig: number;
    jointSig: number;
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
        kinPrev: new Map(),
        failed,
        constraints: createConstraintCache(),

        isFailed: (eid) => failed.has(eid),
        stale: { world: null, eids: [], count: 0 },
        counters: { bodiesVisited: 0, bytesUploaded: 0 },
        springSig: FNV_BASIS,
        jointSig: FNV_BASIS,
    };
}

function runtimeFor(world: World): PhysicsRuntime {
    const runtime = world.resource(physicsRuntimeKey);
    if (!runtime.initialized)
        throw new Error("physics: StandardPhysicsPlugin is not initialized for this World");
    return runtime;
}

// the create-stamp each body was marshaled at. Presence in `bodies` catches a
// plain spawn/despawn; a same-update destroy+create recycling an eid keeps Body membership AND the map entry,
// so the stamp is the only signal that the slot now holds a new body, and a mismatch re-marshals it.
// The remaining fields live in PhysicsRuntime; keeping them beside the state map prevents one World from
// observing another World's handles, interpolation buffers or failed marshals.

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
    const shape = world.storage(Body).shape.get(eid);
    const radius = world.storage(Body).halfExtents.w.get(eid);
    if (shape === ShapeKind.Sphere)
        world.storage(GlobalTransform).scale.set(eid, 2 * radius, 2 * radius, 2 * radius, 0);
    else if (shape === ShapeKind.Capsule)
        world
            .storage(GlobalTransform)
            .scale.set(
                eid,
                2 * radius,
                world.storage(Body).halfExtents.y.get(eid) + radius,
                2 * radius,
                0,
            );
    else
        world
            .storage(GlobalTransform)
            .scale.set(
                eid,
                2 * world.storage(Body).halfExtents.x.get(eid),
                2 * world.storage(Body).halfExtents.y.get(eid),
                2 * world.storage(Body).halfExtents.z.get(eid),
                0,
            );
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
    runtime.kinPrev.delete(eid);
}

function clearBodies(runtime: PhysicsRuntime): void {
    for (const body of runtime.bodies.values()) body.destroy();
    runtime.bodies.clear();
    runtime.stamps.clear();
    runtime.kinPrev.clear();
    runtime.failed.clear();
    runtime.springSig = FNV_BASIS;
    runtime.jointSig = FNV_BASIS;
    runtime.counters = { bodiesVisited: 0, bytesUploaded: 0 };
    resetConstraints(runtime.constraints);
}

/**
 * World-owned physics accessors. `physicsWorld(state)` is the solver escape hatch: joint types past
 * `Joint`, sensors, contact/hit events, mesh/heightfield/compound colliders and native queries; it is
 * `null` until {@link StandardPhysicsPlugin} warms. The pose functions are no-ops before warm.
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

// registers setKinematic hands the solver body; its setters round and copy them.
const kinPos = { x: 0, y: 0, z: 0 };
const kinQuat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const kinVel = { x: 0, y: 0, z: 0 };

export function setKinematic(
    world: World,
    eid: number,
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    teleport = false,
    vel?: readonly [number, number, number],
): void {
    const runtime = runtimeFor(world);
    const tb = runtime.bodies.get(eid);
    if (!tb) return;
    let prev = runtime.kinPrev.get(eid);
    const moved =
        !prev ||
        teleport ||
        pos[0] !== prev.pos[0] ||
        pos[1] !== prev.pos[1] ||
        pos[2] !== prev.pos[2] ||
        quat[0] !== prev.quat[0] ||
        quat[1] !== prev.quat[1] ||
        quat[2] !== prev.quat[2] ||
        quat[3] !== prev.quat[3];
    kinPos.x = pos[0];
    kinPos.y = pos[1];
    kinPos.z = pos[2];
    kinQuat.v.x = quat[0];
    kinQuat.v.y = quat[1];
    kinQuat.v.z = quat[2];
    kinQuat.s = quat[3];
    if (!prev || teleport) {
        prev = {
            pos: [pos[0], pos[1], pos[2]],
            quat: [quat[0], quat[1], quat[2], quat[3]],
        };
        runtime.kinPrev.set(eid, prev);
    }
    if (vel) {
        kinVel.x = vel[0];
        kinVel.y = vel[1];
        kinVel.z = vel[2];
    } else {
        kinVel.x = (pos[0] - prev.pos[0]) / Time.FIXED_DT;
        kinVel.y = (pos[1] - prev.pos[1]) / Time.FIXED_DT;
        kinVel.z = (pos[2] - prev.pos[2]) / Time.FIXED_DT;
    }
    if (moved) {
        // The solve integrates kinematic velocity once. Upload the start of that step, not its
        // already-swept endpoint; otherwise a character's displacement is applied twice.
        kinPos.x -= kinVel.x * Time.FIXED_DT;
        kinPos.y -= kinVel.y * Time.FIXED_DT;
        kinPos.z -= kinVel.z * Time.FIXED_DT;
        tb.setTransform(kinPos, kinQuat);
    }
    tb.setLinearVelocity(kinVel);
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
    if (teleport) world.teleport(eid);
    if (moved && !tb.isAwake()) tb.setAwake(true);
    prev.pos[0] = pos[0];
    prev.pos[1] = pos[1];
    prev.pos[2] = pos[2];
    prev.quat[0] = quat[0];
    prev.quat[1] = quat[1];
    prev.quat[2] = quat[2];
    prev.quat[3] = quat[3];
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
export function snapshotPhysics(world: World): WorldSnapshot {
    return (() => {
        const physicsWorld = runtimeFor(world).physicsWorld;
        if (!physicsWorld) throw new Error("physics: world is not warm");
        return snapshotWorld(physicsWorld);
    })();
}
export function restorePhysics(world: World, saved: WorldSnapshot): void {
    (() => {
        const physicsWorld = runtimeFor(world).physicsWorld;
        if (!physicsWorld) throw new Error("physics: world is not warm");
        restoreWorld(physicsWorld, saved);
        const p = { x: 0, y: 0, z: 0 };
        const q = { v: { x: 0, y: 0, z: 0 }, s: 1 };
        const v = { x: 0, y: 0, z: 0 };
        runtimeFor(world).bodies.forEach((body, eid) => {
            body.getPosition(p);
            body.getRotation(q);
            body.getLinearVelocity(v);
            writeGlobalTransform(
                world,
                eid,
                [p.x, p.y, p.z],
                [q.v.x, q.v.y, q.v.z, q.s],
                [v.x, v.y, v.z],
            );
        });
    })();
}
export function hashPhysics(world: World): bigint {
    return (() => {
        const physicsWorld = runtimeFor(world).physicsWorld;
        if (!physicsWorld) throw new Error("physics: world is not warm");
        return hashWorld(physicsWorld);
    })();
}

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepPhysicsSystem: System = {
    name: "step",
    group: "fixed",
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        physicsWorld.step(fixedDeltaTime, SUBSTEPS);
        runtime.counters.bytesUploaded = 0;
        const global = world.storage(GlobalTransform);
        const rows = physicsWorld.state.bodyStore.movedRows();
        global.translation.writeEncoded(rows.eids, rows.pos);
        global.rotation.writeEncoded(rows.eids, rows.quat);
        global.linearVelocity.writeEncoded(rows.eids, rows.vel);
    },
};

/** uploads a scene's authored {@link Spring} / {@link Joint} entities to the solver, on change only. Fixed group, `before: [StepPhysicsSystem]` so a constraint authored or edited this frame lands in this frame's solve. */
export const SyncPhysicsConstraintsSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepPhysicsSystem],
    update(world) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        const ss = springSignature(world);
        const js = jointSignature(world);
        if (ss === runtime.springSig && js === runtime.jointSig) return;
        if (ss !== runtime.springSig) {
            runtime.springSig = ss;
            syncSprings(
                runtime.constraints,
                physicsWorld,
                runtime.bodies,
                springDefs(world),
                runtime.isFailed,
            );
        }
        if (js !== runtime.jointSig) {
            runtime.jointSig = js;
            syncJoints(
                runtime.constraints,
                physicsWorld,
                runtime.bodies,
                jointDefs(world),
                runtime.isFailed,
            );
        }
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
const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem, StepPhysicsSystem],
    update(world: World) {
        const runtime = runtimeFor(world);
        const physicsWorld = runtime.physicsWorld;
        if (!physicsWorld) return;
        runtime.counters.bodiesVisited = 0;
        // a deferred body finally marshaling (or a body going stale) is the transition a dropped constraint
        // waits on, and `SyncPhysicsConstraintsSystem` re-uploads on an authored signature change only, so the constraint
        // re-sync is pumped from here on any body-set change (a no-op walk when nothing was dropped, joints.ts).
        let bodySetChanged = false;
        for (const eid of world.query(BODY_TERMS)) {
            runtime.counters.bodiesVisited += 1;
            const stamp = world.generation(eid);
            if (world.has(eid, Transform) && runtime.placementWarnings.get(eid) !== stamp) {
                console.warn(
                    `physics-sync: entity ${eid} carries both Body and Transform; both write GlobalTransform`,
                );
                runtime.placementWarnings.set(eid, stamp);
            }
            if (runtime.bodies.has(eid)) {
                if (runtime.stamps.get(eid) === stamp) continue;
                forget(runtime, eid); // recycled to a new Body in one update
            }
            const f = runtime.failed.get(eid);
            if (f && f.stamp === stamp && f.hulls === world.resource(Hulls).size) continue;
            const tb = marshalBody(world, physicsWorld, eid);
            if (!tb) {
                runtime.failed.set(eid, { stamp, hulls: world.resource(Hulls).size });
                continue;
            }
            runtime.failed.delete(eid);
            kernel(world).bodySetEntity(tb.id.world0, tb.id.index1 - 1, eid);
            runtime.bodies.set(eid, tb);
            runtime.stamps.set(eid, stamp);
            bodySetChanged = true;
            seedGlobalTransform(world, eid);
            world.teleport(eid);
        }
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
        if (bodySetChanged)
            resyncConstraints(runtime.constraints, physicsWorld, runtime.bodies, runtime.isFailed);
    },
};

/**
 * rigid-body physics: depends on the shared `Body`/`Spring`/`Joint` data and installs a CPU solver (Rust/WASM kernel, no GPU device
 * needed to step). Opt-in — add it to a scene to run physics (it's not in the default plugins). Nine joint
 * types, mesh/heightfield/compound colliders, sensors, CCD and sleeping ride {@link physicsWorld}.
 *
 * @example
 * ```
 * export const config: AppConfig = { plugins: [StandardPhysicsPlugin] };
 * ```
 */
export const StandardPhysicsPlugin: Plugin = {
    name: "StandardPhysics",
    dependencies: [PhysicsPlugin],
    systems: [SyncSystem, SyncPhysicsConstraintsSystem, StepPhysicsSystem],

    initialize(world) {
        world.resource(physicsRuntimeKey).initialized = true;
    },

    async warm(world) {
        const runtime = runtimeFor(world);
        await init(world); // async wasm compile — the browser main thread can't compile it synchronously
        runtime.physicsWorld?.destroy();
        runtime.physicsWorld = new PhysicsWorld({ gravity: { x: 0, y: GRAVITY, z: 0 } }, world);
        clearBodies(runtime);
        resetSignatures(world); // the fresh world receives the authored constraint set on its first frame
    },

    dispose(world) {
        const runtime = runtimeFor(world);
        clearBodies(runtime);
        const warnings = signatureWarningsFor(world);
        warnings.joints.clear();
        warnings.springs.clear();
        runtime.physicsWorld?.destroy();
        runtime.physicsWorld = null;
        void shutdown(world);
    },
};
