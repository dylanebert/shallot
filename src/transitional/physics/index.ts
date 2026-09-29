// Destination: core/physics and standard/physics; owner: physics-boundary.md.
/// <reference types="@webgpu/types" />

import {
    Compute,
    entity,
    FIXED_DT,
    f32,
    field,
    type Plugin,
    type Quad,
    type Single,
    type State,
    type System,
    Time,
    u32,
    vec4,
} from "../../engine";
import { currentWorld, withCompute } from "../../engine/runtime";
import { eulerAlias } from "../../engine/utils";
import { Transform, TransformsPlugin, transformTable } from "../transforms";
import {
    type ContactEvents,
    hash as hashWorld,
    init,
    type JointEvent,
    type ParallelJointConfig,
    type RevoluteJointConfig,
    restore as restoreWorld,
    type SoftJointConfig,
    type Body as SolverBody,
    type SphericalJointConfig,
    shutdown,
    snapshot as snapshotWorld,
    type WheelJointConfig,
    World,
    type WorldSnapshot,
} from "./api";
import { nlerpShortestInto } from "./compose";
import { Hulls } from "./hull";
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

export { createPool, maxWorkers, type Pool, type WorkerReady } from "./kernel/pool";

// Physics: the authoring components (`Body`/`Spring`/`Joint`), the CPU raycast + pick layer, and the
// Rust/WASM rigid-body solver behind them. CPU writeback: move events become an interpolated pose written
// into the `transforms` firehose, movers only. Storage is an eid↔solver-body map plus a capacity-sized
// prev/curr pose double buffer; no slab or mirror, since the solver serves this tick's state directly.
// Body marshaling is `marshal.ts`, Spring/Joint marshaling `joints.ts`. An outside solver plugs in through
// the physics barrel (traits, defs, signatures, system anchors) and never through this module's state.

/** collision-shape tag for {@link Body}. Box collides as an OBB; sphere/capsule as a core + radius; hull as a convex polytope (geometry registered in `Hulls`, referenced by `halfExtents.w` = the hull id). */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

/**
 * a rigid body simulated by {@link PhysicsPlugin}: falls under gravity and collides with other bodies
 * (`mass: 0` = static).
 *
 * @example
 * ```
 * <a body="shape: 0; pos: 0 5 0; half-extents: 0.5 0.5 0.5; mass: 1; friction: 0.5" />
 * <a body="shape: 1; pos: 0 5 0; half-extents: 0 0 0 0.5; mass: 1" />            <!-- sphere, radius 0.5 -->
 * <a body="shape: 2; pos: 0 5 0; half-extents: 0 0.5 0 0.3; mass: 1" />          <!-- capsule, half-height 0.5, radius 0.3 -->
 * <a body="shape: 3; pos: 0 5 0; half-extents: 1 1 1 2; mass: 1" />              <!-- hull id 2, AABB half 1×1×1 -->
 * ```
 */
export const Body = {
    /** the collider, a `ShapeKind`: `Box` (an OBB of `halfExtents`), `Sphere`, `Capsule` (a segment along local Y inflated by the radius), or `Hull` (a convex polytope registered in `Hulls`). */
    shape: field(u32),
    /** spawn position; physics owns it after spawn. */
    pos: field(vec4),
    /** spawn orientation, authored as euler degrees like `Transform.rot`; physics-owned after spawn. */
    quat: field(vec4),
    /** box/AABB half-extents in `xyz`; `w` doubles as the rounding radius (sphere/capsule) or the `Hull` id (a hull has radius 0, so the lane is free). */
    halfExtents: field(vec4),
    /** mass in kg; `0` or less marks a static body that never moves. */
    mass: field(f32),
    /** coulomb friction coefficient: `0` slides freely, higher grips. */
    friction: field(f32),
};

/** live pose written by the physics plugin; consumers such as the harness resolve it by registered name. */
export const Pose = {
    pos: field(vec4),
    quat: field(vec4),
    vel: field(vec4),
};

/** Pose is runtime-derived and never scene-authored. */
export const poseTraits = {
    derived: true,
    defaults: () => ({ pos: [0, 0, 0, 0], quat: [0, 0, 0, 1], vel: [0, 0, 0, 0] }),
};

/**
 * a soft distance spring linking two bodies, pulling them toward a rest length; its own entity, referencing the bodies by `@name`.
 *
 * @example
 * ```
 * <a id="anchor" body="mass: 0; pos: 0 10 0" />
 * <a body="mass: 1; pos: 0 6 0" />
 * <a spring="a: @anchor; b: @block; rest: 4; stiffness: 100" />
 * ```
 */
export const Spring = {
    /** the first body (a `@name` reference). */
    a: field(entity),
    /** the second body. */
    b: field(entity),
    /** anchor point on body `a`, in its local frame. */
    rA: field(vec4),
    /** anchor point on body `b`, in its local frame. */
    rB: field(vec4),
    /** pull strength; higher is stiffer. */
    stiffness: field(f32),
    /** the target distance the spring pulls the anchors toward. */
    rest: field(f32),
};

/**
 * a hard joint pinning two bodies together: a rigid linear pin plus an optional angular lock, referencing both by `@name`.
 *
 * the anchors must start coincident at the scene pose (join a dynamic body to a static/kinematic anchor),
 * or construction rejects the joint.
 *
 * @example
 * ```
 * <a id="pivot" body="mass: 0; pos: 0 10 0" />
 * <a body="mass: 1; pos: 2 8 0" />
 * <a joint="a: @pivot; b: @bob; r-a: 0 0 0; r-b: 0 2.5 0" />                              <!-- spherical -->
 * <a joint="a: @pivot; b: @link; r-a: 0.5 0 0; r-b: -0.5 0 0; stiffness-ang: fixed" />   <!-- fixed -->
 * ```
 */
export const Joint = {
    /** the first body (a `@name` reference). */
    a: field(entity),
    /** the second body. */
    b: field(entity),
    /** the pin's anchor on body `a`, in its local frame. */
    rA: field(vec4),
    /** the pin's anchor on body `b`, in its local frame. */
    rB: field(vec4),
    /** angular lock: `0` (default) leaves rotation free (spherical); `∞` locks orientation (author `stiffness-ang: fixed`). */
    stiffnessAng: field(f32),
};

// Authoring metadata for the three components above, shared with any extension solver that registers
// them. `Body`/`Spring`/`Joint` are the same objects across plugins (idempotent registration
// keeps component ids stable), so their traits live here once.

/** {@link Body}'s traits: defaults, its exclusion of {@link Transform}, and the euler-degree `quat` alias. Shared by every plugin that registers `Body`. */
export const bodyTraits = {
    defaults: () => ({
        shape: ShapeKind.Box,
        pos: [0, 0, 0, 0],
        quat: [0, 0, 0, 1],
        halfExtents: [0.5, 0.5, 0.5, 0], // .w = rounding radius (0 for a box)
        mass: 1,
        friction: 0.5,
    }),
    excludes: [Transform],
    // physics owns the entity's world transform (composed into the firehose each frame), so a
    // Body stands in for Transform: a `Part` on the same entity renders at the body's pose
    provides: [Transform],
    // a Body's orientation is stored as a quaternion but authored as euler degrees, like Transform.rot
    aliases: { quat: eulerAlias("quat") },
};

/** {@link Spring}'s traits: field defaults. Shared by every plugin that registers `Spring`. */
export const springTraits = {
    defaults: () => ({
        a: 0,
        b: 0,
        rA: [0, 0, 0, 0],
        rB: [0, 0, 0, 0],
        stiffness: 100,
        rest: 1,
    }),
};

/** {@link Joint}'s traits: field defaults plus the `stiffness-ang: fixed` parse hook. Shared by every plugin that registers `Joint`. */
export const jointTraits = {
    defaults: () => ({
        a: 0,
        b: 0,
        rA: [0, 0, 0, 0],
        rB: [0, 0, 0, 0],
        stiffnessAng: 0, // spherical; ∞ = fixed
    }),
    // author a fixed joint's angular lock as `stiffness-ang: fixed` (∞) — a number parses normally,
    // so only the keyword needs the hook; the default 0 is the spherical (free-rotation) joint.
    parse: {
        stiffnessAng: (v: string) =>
            v === "fixed" || v === "inf" ? Number.POSITIVE_INFINITY : undefined,
    },
};

/** an authored spring: two body eids + local anchors + stiffness/rest, derived from a scene's {@link Spring} entities by {@link springDefs}. */
export interface SpringDef {
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    stiffness: number;
    rest: number;
}

/** an authored joint: two body eids + local anchors + the angular lock, derived from a scene's {@link Joint} entities by {@link jointDefs}. Richer joints (motors, limits, the nine solver joint types) ride {@link physicsWorld}. */
export interface JointDef {
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    stiffnessAng: number;
}

/** one body's live pose + velocity at the last fixed step; sleeping bodies read zero velocity. */
export interface BodyState {
    pos: readonly [number, number, number];
    quat: readonly [number, number, number, number];
    vel: readonly [number, number, number];
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
/** re-arm the warning dedupe when a plugin world is warmed. Signatures themselves are State-owned. */
const signatureWarningsKey = Symbol("shallot.physics-signature-warnings");

function signatureWarningsFor(state: State) {
    return state.resource(signatureWarningsKey, () => ({
        joints: new Set<number>(),
        springs: new Set<number>(),
    }));
}

export function resetSignatures(state: State): void {
    inState(state, () => {
        const warnings = signatureWarningsFor(state);
        warnings.joints.clear();
        warnings.springs.clear();
        resetConstraints(state.resource(physicsRuntimeKey, newRuntime).constraints);
    });
}

/** a hash of the authored {@link Spring} set, endpoint create-stamps included: an uploader re-uploads only when it changes. */
function springSignatureInState(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query(SPRING_TERMS)) {
        h = fold(h, eid);
        const a = Spring.a.get(eid);
        const b = Spring.b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps: a same-update realias of an endpoint (destroy+create
        // recycling its eid) leaves a/b unchanged, so without the stamp the re-upload is suppressed and the
        // solver joint pins the NEW occupant at the old anchors.
        h = fold(h, state.stamp(a));
        h = fold(h, state.stamp(b));
        h = fold(h, sigBits(Spring.rA.x.get(eid)));
        h = fold(h, sigBits(Spring.rA.y.get(eid)));
        h = fold(h, sigBits(Spring.rA.z.get(eid)));
        h = fold(h, sigBits(Spring.rB.x.get(eid)));
        h = fold(h, sigBits(Spring.rB.y.get(eid)));
        h = fold(h, sigBits(Spring.rB.z.get(eid)));
        h = fold(h, sigBits(Spring.stiffness.get(eid)));
        h = fold(h, sigBits(Spring.rest.get(eid)));
    }
    return h;
}

/** a hash of the authored {@link Spring} set, endpoint create-stamps included: an uploader re-uploads only when it changes. */
export function springSignature(state: State): number {
    return currentWorld<State>() === state
        ? springSignatureInState(state)
        : withCompute(state.gpu, () => springSignatureInState(state));
}

/** a hash of the authored {@link Joint} set, endpoint create-stamps included: the {@link springSignature} twin. */
function jointSignatureInState(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query(JOINT_TERMS)) {
        h = fold(h, eid);
        const a = Joint.a.get(eid);
        const b = Joint.b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps — see springSignature: a realias of an endpoint must
        // force the re-upload so the solver joint rebinds to the new occupant.
        h = fold(h, state.stamp(a));
        h = fold(h, state.stamp(b));
        h = fold(h, sigBits(Joint.rA.x.get(eid)));
        h = fold(h, sigBits(Joint.rA.y.get(eid)));
        h = fold(h, sigBits(Joint.rA.z.get(eid)));
        h = fold(h, sigBits(Joint.rB.x.get(eid)));
        h = fold(h, sigBits(Joint.rB.y.get(eid)));
        h = fold(h, sigBits(Joint.rB.z.get(eid)));
        h = fold(h, sigBits(Joint.stiffnessAng.get(eid)));
    }
    return h;
}

export function jointSignature(state: State): number {
    return currentWorld<State>() === state
        ? jointSignatureInState(state)
        : withCompute(state.gpu, () => jointSignatureInState(state));
}

/** the authored {@link Spring} set as {@link SpringDef}s, dropping (and warning once for) a negative or NaN stiffness. */
export function springDefs(state: State): SpringDef[] {
    return inState(state, () => {
        const warnings = signatureWarningsFor(state);
        const out: SpringDef[] = [];
        for (const eid of state.query([Spring])) {
            const stiffness = Spring.stiffness.get(eid);
            // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
            // 0 and ∞ are valid authored values (0 = non-positive → downstream skip; ∞ = rigid) — only negative
            // and NaN are rejected, at the authoring layer so every solver inherits one behavior.
            if (Number.isNaN(stiffness) || stiffness < 0) {
                if (!warnings.springs.has(eid)) {
                    console.warn(
                        `[physics] spring (a: ${Spring.a.get(eid)}, b: ${Spring.b.get(eid)}) has negative or NaN stiffness — skipped`,
                    );
                    warnings.springs.add(eid);
                }
                continue;
            }
            warnings.springs.delete(eid);
            out.push({
                a: Spring.a.get(eid),
                b: Spring.b.get(eid),
                rA: [Spring.rA.x.get(eid), Spring.rA.y.get(eid), Spring.rA.z.get(eid)],
                rB: [Spring.rB.x.get(eid), Spring.rB.y.get(eid), Spring.rB.z.get(eid)],
                stiffness,
                rest: Spring.rest.get(eid),
            });
        }
        return out;
    });
}

/** the authored {@link Joint} set as {@link JointDef}s, dropping (and warning once for) a negative or NaN angular stiffness. */
export function jointDefs(state: State): JointDef[] {
    return inState(state, () => {
        const warnings = signatureWarningsFor(state);
        const out: JointDef[] = [];
        for (const eid of state.query([Joint])) {
            const stiffnessAng = Joint.stiffnessAng.get(eid);
            // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
            // 0 (spherical) and ∞ (fixed) are valid authored values — only negative and NaN are rejected, at the
            // authoring layer so every solver inherits one behavior.
            if (Number.isNaN(stiffnessAng) || stiffnessAng < 0) {
                if (!warnings.joints.has(eid)) {
                    console.warn(
                        `[physics] joint (a: ${Joint.a.get(eid)}, b: ${Joint.b.get(eid)}) has negative or NaN angular stiffness — skipped`,
                    );
                    warnings.joints.add(eid);
                }
                continue;
            }
            warnings.joints.delete(eid);
            out.push({
                a: Joint.a.get(eid),
                b: Joint.b.get(eid),
                rA: [Joint.rA.x.get(eid), Joint.rA.y.get(eid), Joint.rA.z.get(eid)],
                rB: [Joint.rB.x.get(eid), Joint.rB.y.get(eid), Joint.rB.z.get(eid)],
                stiffnessAng,
            });
        }
        return out;
    });
}

const GRAVITY = -10;
const SUBSTEPS = 4; // the solver's own recommended sub-step count (World.step's default)

export interface PhysicsCounters {
    bodiesVisited: number;
    bytesUploaded: number;
}

interface PhysicsRuntime {
    initialized: boolean;
    world: World | null;
    bodies: Map<number, SolverBody>;
    stamps: Map<number, number>;
    kinPrev: Map<number, { pos: [number, number, number]; quat: [number, number, number, number] }>;
    failed: Map<number, { stamp: number; hulls: number }>;
    constraints: ConstraintCache;
    handleProxies: WeakMap<object, object>;
    handleMethods: WeakMap<object, Map<PropertyKey, (...args: unknown[]) => unknown>>;
    // whether a body's marshal failed, so the constraint uploads defer its joints; made once per runtime
    isFailed: (eid: number) => boolean;
    prevPos: Float32Array;
    prevQuat: Float32Array;
    currPos: Float32Array;
    currQuat: Float32Array;
    // the eids whose move events this tick carried, one each (a body emits at most one per step), in the
    // first `movedCount` slots of a capacity-sized column: re-clearing a filled Set, or truncating an
    // array, releases its backing store and the next tick's adds allocate it again.
    movedThisTick: Int32Array;
    movedCount: number;
    renderRecords: Float32Array;
    stale: StaleScan;
    counters: PhysicsCounters;
    springSig: number;
    jointSig: number;
}

const physicsRuntimeKey = Symbol("shallot.physics-runtime");

function newRuntime(): PhysicsRuntime {
    const failed: PhysicsRuntime["failed"] = new Map();
    return {
        initialized: false,
        world: null,
        bodies: new Map(),
        stamps: new Map(),
        kinPrev: new Map(),
        failed,
        constraints: createConstraintCache(),
        handleProxies: new WeakMap(),
        handleMethods: new WeakMap(),
        isFailed: (eid) => failed.has(eid),
        prevPos: new Float32Array(0),
        prevQuat: new Float32Array(0),
        currPos: new Float32Array(0),
        currQuat: new Float32Array(0),
        movedThisTick: new Int32Array(0),
        movedCount: 0,
        renderRecords: new Float32Array(0),
        stale: { state: null, eids: [], count: 0 },
        counters: { bodiesVisited: 0, bytesUploaded: 0 },
        springSig: FNV_BASIS,
        jointSig: FNV_BASIS,
    };
}

function inState<T>(state: State, callback: () => T): T {
    return withCompute(state.gpu, callback);
}

function scopedHandle<T extends object>(state: State, value: T): T {
    if (
        Array.isArray(value) ||
        ArrayBuffer.isView(value) ||
        value instanceof ArrayBuffer ||
        value instanceof Map ||
        value instanceof Set
    )
        return value;
    const prototype = Object.getPrototypeOf(value);
    if (prototype === Object.prototype || prototype === null) return value;
    const runtime = runtimeFor(state);
    const cached = runtime.handleProxies.get(value);
    if (cached) return cached as T;
    return createScopedHandle(state, value, runtime);
}

function createScopedHandle<T extends object>(state: State, value: T, runtime: PhysicsRuntime): T {
    const proxy = new Proxy(value, {
        get(target, key) {
            const member = Reflect.get(target, key, target) as unknown;
            if (typeof member !== "function") {
                return member !== null && typeof member === "object"
                    ? scopedHandle(state, member)
                    : member;
            }
            let methods = runtime.handleMethods.get(target);
            if (!methods) {
                methods = new Map();
                runtime.handleMethods.set(target, methods);
            }
            let wrapped = methods.get(key);
            if (!wrapped) {
                wrapped = (...args: unknown[]) => {
                    if (currentWorld<State>() === state) {
                        const result = member.apply(target, args);
                        return result !== null && typeof result === "object"
                            ? scopedHandle(state, result)
                            : result;
                    }
                    return withCompute(state.gpu, () => {
                        const result = member.apply(target, args);
                        return result !== null && typeof result === "object"
                            ? scopedHandle(state, result)
                            : result;
                    });
                };
                methods.set(key, wrapped);
            }
            return wrapped;
        },
        set(target, key, next) {
            return inState(state, () => Reflect.set(target, key, next, target));
        },
    });
    runtime.handleProxies.set(value, proxy);
    return proxy;
}

function runtimeFor(state: State): PhysicsRuntime {
    const runtime = state.resource(physicsRuntimeKey, newRuntime);
    if (!runtime.initialized)
        throw new Error("physics: PhysicsPlugin is not initialized for this State");
    return runtime;
}

function growFloat(source: Float32Array, lanes: number, capacity: number): Float32Array {
    const next = new Float32Array(capacity * lanes);
    next.set(source);
    return next;
}

function ensurePoseCapacity(runtime: PhysicsRuntime, rows: number): void {
    if (rows <= runtime.movedThisTick.length) return;
    let capacity = Math.max(16, runtime.movedThisTick.length);
    while (capacity < rows) capacity *= 2;
    runtime.prevPos = growFloat(runtime.prevPos, 3, capacity);
    runtime.prevQuat = growFloat(runtime.prevQuat, 4, capacity);
    runtime.currPos = growFloat(runtime.currPos, 3, capacity);
    runtime.currQuat = growFloat(runtime.currQuat, 4, capacity);
    const moved = new Int32Array(capacity);
    moved.set(runtime.movedThisTick);
    runtime.movedThisTick = moved;
}
// the create-stamp each body was marshaled at. Presence in `bodies` catches a
// plain spawn/despawn; a same-update destroy+create recycling an eid keeps Body membership AND the map entry,
// so the stamp is the only signal that the slot now holds a new body, and a mismatch re-marshals it.
// The remaining fields live in PhysicsRuntime; keeping them beside the state map prevents one State from
// observing another State's handles, interpolation buffers or failed marshals.

function writePose(
    eid: number,
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    vel: readonly [number, number, number],
): void {
    Pose.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Pose.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
    Pose.vel.set(eid, vel[0], vel[1], vel[2], 0);
}

function seedPose(runtime: PhysicsRuntime, eid: number): void {
    const p = eid * 3;
    const q = eid * 4;
    runtime.prevPos[p] = runtime.currPos[p] = Body.pos.x.get(eid);
    runtime.prevPos[p + 1] = runtime.currPos[p + 1] = Body.pos.y.get(eid);
    runtime.prevPos[p + 2] = runtime.currPos[p + 2] = Body.pos.z.get(eid);
    runtime.prevQuat[q] = runtime.currQuat[q] = Body.quat.x.get(eid);
    runtime.prevQuat[q + 1] = runtime.currQuat[q + 1] = Body.quat.y.get(eid);
    runtime.prevQuat[q + 2] = runtime.currQuat[q + 2] = Body.quat.z.get(eid);
    runtime.prevQuat[q + 3] = runtime.currQuat[q + 3] = Body.quat.w.get(eid);
    writePose(
        eid,
        [Body.pos.x.get(eid), Body.pos.y.get(eid), Body.pos.z.get(eid)],
        [Body.quat.x.get(eid), Body.quat.y.get(eid), Body.quat.z.get(eid), Body.quat.w.get(eid)],
        [0, 0, 0],
    );
}

function forget(runtime: PhysicsRuntime, eid: number): void {
    runtime.bodies.get(eid)?.destroy();
    runtime.bodies.delete(eid);
    runtime.kinPrev.delete(eid);
    const moved = runtime.movedThisTick;
    for (let i = 0; i < runtime.movedCount; i++) {
        if (moved[i] !== eid) continue;
        runtime.movedCount -= 1;
        moved[i] = moved[runtime.movedCount];
        break;
    }
}

function clearBodies(runtime: PhysicsRuntime): void {
    for (const body of runtime.bodies.values()) body.destroy();
    runtime.bodies.clear();
    runtime.stamps.clear();
    runtime.kinPrev.clear();
    runtime.movedCount = 0;
    runtime.failed.clear();
    runtime.springSig = FNV_BASIS;
    runtime.jointSig = FNV_BASIS;
    runtime.counters = { bodiesVisited: 0, bytesUploaded: 0 };
    resetConstraints(runtime.constraints);
}

/**
 * State-owned physics accessors. `physicsWorld(state)` is the solver escape hatch: joint types past
 * `Joint`, sensors, contact/hit events, mesh/heightfield/compound colliders and native queries; it is
 * `null` until {@link PhysicsPlugin} warms. `body(state, eid)` bridges a `Body` entity to its live solver
 * handle (`null` before its first fixed tick). The pose functions are no-ops before warm.
 */
function physicsWorldOutside(state: State): World | null {
    return withCompute(state.gpu, () => physicsWorld(state));
}

export function physicsWorld(state: State): World | null {
    if (currentWorld<State>() !== state) return physicsWorldOutside(state);
    const world = runtimeFor(state).world;
    return world ? scopedHandle(state, world) : null;
}
export function body(state: State, eid: number): SolverBody | null {
    if (currentWorld<State>() !== state) return withCompute(state.gpu, () => body(state, eid));
    const live = runtimeFor(state).bodies.get(eid);
    return live ? scopedHandle(state, live) : null;
}

function requireBody(state: State, eid: number): SolverBody {
    const live = body(state, eid);
    if (!live) throw new Error(`physics: body ${eid} is not warm`);
    return live;
}

/** Create a wheel joint between two State-owned bodies without exposing the solver World. */
export function createWheelJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    config: Partial<WheelJointConfig> = {},
) {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return scopedHandle(
            state,
            world.createWheelJoint(requireBody(state, bodyA), requireBody(state, bodyB), config),
        );
    });
}

/** Create a parallel joint between two State-owned bodies without exposing the solver World. */
export function createParallelJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    config: Partial<ParallelJointConfig> = {},
) {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return scopedHandle(
            state,
            world.createParallelJoint(requireBody(state, bodyA), requireBody(state, bodyB), config),
        );
    });
}

/** Create a hinge (revolute) joint between two State-owned bodies. */
export function createRevoluteJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    config: Partial<RevoluteJointConfig> = {},
) {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return scopedHandle(
            state,
            world.createRevoluteJoint(requireBody(state, bodyA), requireBody(state, bodyB), config),
        );
    });
}

/** Create a cone/twist-capable spherical joint between two State-owned bodies. */
export function createSphericalJoint(
    state: State,
    bodyA: number,
    bodyB: number,
    config: Partial<SphericalJointConfig> = {},
) {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return scopedHandle(
            state,
            world.createSphericalJoint(
                requireBody(state, bodyA),
                requireBody(state, bodyB),
                config,
            ),
        );
    });
}

/** Create a soft spring from a State-owned body to a world-space anchor. */
export function createSoftJoint(
    state: State,
    bodyEid: number,
    anchor: { x: number; y: number; z: number },
    config: Partial<SoftJointConfig> = {},
) {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return scopedHandle(
            state,
            world.createSoftJoint(requireBody(state, bodyEid), anchor, config),
        );
    });
}

/** Read contact-begin/end/hit events for the last State-owned fixed step. */
export function getContactEvents(state: State): ContactEvents {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return world.getContactEvents();
    });
}

/** Read joint break-threshold events for the last State-owned fixed step. */
export function getJointEvents(state: State): JointEvent[] {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return world.getJointEvents();
    });
}

/** a writable {@link BodyState} that {@link readBody} fills in place. */
export interface BodyStateOut {
    pos: [number, number, number];
    quat: [number, number, number, number];
    vel: [number, number, number];
}

// registers readBody reads the solver body through; never live across calls.
const readPos = { x: 0, y: 0, z: 0 };
const readQuat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const readVel = { x: 0, y: 0, z: 0 };

/** one body's live pose + velocity, or null when `eid` has no solver body. Pass `out` to fill it instead of allocating. */
export function readBody(state: State, eid: number, out?: BodyStateOut): BodyState | null {
    if (currentWorld<State>() !== state)
        return withCompute(state.gpu, () => readBody(state, eid, out));
    const tb = runtimeFor(state).bodies.get(eid);
    if (!tb) return null;
    const p = tb.getPosition(readPos);
    const q = tb.getRotation(readQuat);
    const v = tb.getLinearVelocity(readVel);
    if (out === undefined) {
        return { pos: [p.x, p.y, p.z], quat: [q.v.x, q.v.y, q.v.z, q.s], vel: [v.x, v.y, v.z] };
    }
    out.pos[0] = p.x;
    out.pos[1] = p.y;
    out.pos[2] = p.z;
    out.quat[0] = q.v.x;
    out.quat[1] = q.v.y;
    out.quat[2] = q.v.z;
    out.quat[3] = q.s;
    out.vel[0] = v.x;
    out.vel[1] = v.y;
    out.vel[2] = v.z;
    return out;
}

// registers setKinematic hands the solver body; its setters round and copy them.
const kinPos = { x: 0, y: 0, z: 0 };
const kinQuat = { v: { x: 0, y: 0, z: 0 }, s: 1 };
const kinVel = { x: 0, y: 0, z: 0 };

export function setKinematic(
    state: State,
    eid: number,
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    teleport = false,
    vel?: readonly [number, number, number],
): void {
    if (currentWorld<State>() !== state) {
        withCompute(state.gpu, () => setKinematic(state, eid, pos, quat, teleport, vel));
        return;
    }
    const runtime = runtimeFor(state);
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
    if (moved) tb.setTransform(kinPos, kinQuat);
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
    tb.setLinearVelocity(kinVel);
    if (moved && !tb.isAwake()) tb.setAwake(true);
    prev.pos[0] = pos[0];
    prev.pos[1] = pos[1];
    prev.pos[2] = pos[2];
    prev.quat[0] = quat[0];
    prev.quat[1] = quat[1];
    prev.quat[2] = quat[2];
    prev.quat[3] = quat[3];
}
export function setVelocity(state: State, eid: number, vx: number, vy: number, vz: number): void {
    if (currentWorld<State>() !== state) {
        withCompute(state.gpu, () => setVelocity(state, eid, vx, vy, vz));
        return;
    }
    runtimeFor(state).bodies.get(eid)?.setLinearVelocity({ x: vx, y: vy, z: vz });
}
export function physicsCounters(state: State): PhysicsCounters {
    return inState(state, () => ({ ...runtimeFor(state).counters }));
}
export function snapshot(state: State): WorldSnapshot {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return snapshotWorld(world);
    });
}
export function restore(state: State, saved: WorldSnapshot): void {
    inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        restoreWorld(world, saved);
    });
}
export function hash(state: State): bigint {
    return inState(state, () => {
        const world = runtimeFor(state).world;
        if (!world) throw new Error("physics: world is not warm");
        return hashWorld(world);
    });
}

/** Static physics configuration shared by the State-first functions. */
export const Physics = {
    gravity: GRAVITY,
    dt: FIXED_DT,
} as const;

export type PhysicsStepConfig = Readonly<{
    dt: number;
    gravity: number;
    substeps: number;
}>;

/** The fixed-step values used by this initialized State's production physics system. */
export function physicsStepConfig(state: State): PhysicsStepConfig {
    return inState(state, () => {
        runtimeFor(state);
        return { dt: Time.FIXED_DT, gravity: GRAVITY, substeps: SUBSTEPS };
    });
}

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepSystem: System = {
    name: "step",
    group: "fixed",
    update(state) {
        const runtime = runtimeFor(state);
        const world = runtime.world;
        if (!world) return;
        world.step(FIXED_DT, SUBSTEPS);
        runtime.movedCount = 0;
        runtime.counters.bytesUploaded = 0;
        const pose = state.of(Pose);
        const rows = world.state.bodyStore.movedRows();
        pose.pos.write(rows.eids, rows.pos);
        pose.quat.write(rows.eids, rows.quat);
        pose.vel.write(rows.eids, rows.vel);
        for (let i = 0; i < rows.eids.length; i++) {
            const eid = rows.eids[i];
            const offset = i * 4;
            const p = eid * 3;
            const q = eid * 4;
            runtime.prevPos[p] = runtime.currPos[p];
            runtime.prevPos[p + 1] = runtime.currPos[p + 1];
            runtime.prevPos[p + 2] = runtime.currPos[p + 2];
            runtime.prevQuat[q] = runtime.currQuat[q];
            runtime.prevQuat[q + 1] = runtime.currQuat[q + 1];
            runtime.prevQuat[q + 2] = runtime.currQuat[q + 2];
            runtime.prevQuat[q + 3] = runtime.currQuat[q + 3];
            runtime.currPos[p] = rows.pos[offset];
            runtime.currPos[p + 1] = rows.pos[offset + 1];
            runtime.currPos[p + 2] = rows.pos[offset + 2];
            runtime.currQuat[q] = rows.quat[offset];
            runtime.currQuat[q + 1] = rows.quat[offset + 1];
            runtime.currQuat[q + 2] = rows.quat[offset + 2];
            runtime.currQuat[q + 3] = rows.quat[offset + 3];
            runtime.movedThisTick[runtime.movedCount++] = eid;
        }
    },
};

/** uploads a scene's authored {@link Spring} / {@link Joint} entities to the solver, on change only. Fixed group, `before: [StepSystem]` so a constraint authored or edited this frame lands in this frame's solve. */
export const ConstraintSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepSystem],
    update(state) {
        const runtime = runtimeFor(state);
        const world = runtime.world;
        if (!world) return;
        const ss = springSignature(state);
        const js = jointSignature(state);
        if (ss === runtime.springSig && js === runtime.jointSig) return;
        if (ss !== runtime.springSig) {
            runtime.springSig = ss;
            syncSprings(
                runtime.constraints,
                world,
                runtime.bodies,
                springDefs(state),
                runtime.isFailed,
            );
        }
        if (js !== runtime.jointSig) {
            runtime.jointSig = js;
            syncJoints(
                runtime.constraints,
                world,
                runtime.bodies,
                jointDefs(state),
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
    state: State | null;
    eids: number[];
    count: number;
}

// the sync's map walks, given the State as `this`, so a steady sync mints no iterator.
function dropDespawnedFailure(
    this: State,
    _failure: unknown,
    eid: number,
    failed: Map<number, unknown>,
): void {
    if (!this.has(eid, Body)) failed.delete(eid);
}

function collectStale(this: StaleScan, _body: unknown, eid: number): void {
    if (!this.state!.has(eid, Body)) this.eids[this.count++] = eid;
}

// membership-driven create/destroy, ascending eid order (state.query's natural order — creation order is
// load-bearing for solver determinism). Runs every fixed tick before the solve so a body spawned this frame
// joins THIS tick's step.
const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [ConstraintSystem, StepSystem],
    update(state: State) {
        const runtime = runtimeFor(state);
        const world = runtime.world;
        if (!world) return;
        ensurePoseCapacity(runtime, state.entityHighWater);
        runtime.counters.bodiesVisited = 0;
        // a deferred body finally marshaling (or a body going stale) is the transition a dropped constraint
        // waits on, and `ConstraintSystem` re-uploads on an authored signature change only, so the constraint
        // re-sync is pumped from here on any body-set change (a no-op walk when nothing was dropped, joints.ts).
        let bodySetChanged = false;
        for (const eid of state.query(BODY_TERMS)) {
            runtime.counters.bodiesVisited += 1;
            const stamp = state.stamp(eid);
            if (runtime.bodies.has(eid)) {
                if (runtime.stamps.get(eid) === stamp) continue;
                forget(runtime, eid); // recycled to a new Body in one update
            }
            const f = runtime.failed.get(eid);
            if (f && f.stamp === stamp && f.hulls === Hulls.size) continue;
            const tb = marshalBody(world, eid);
            if (!tb) {
                runtime.failed.set(eid, { stamp, hulls: Hulls.size });
                continue;
            }
            runtime.failed.delete(eid);
            kernel().bodySetEntity(tb.id.world0, tb.id.index1 - 1, eid);
            runtime.bodies.set(eid, tb);
            runtime.stamps.set(eid, stamp);
            if (!state.has(eid, Pose)) state.add(eid, Pose);
            bodySetChanged = true;
            seedPose(runtime, eid);
        }
        runtime.failed.forEach(dropDespawnedFailure, state);
        const stale = runtime.stale;
        stale.state = state;
        stale.count = 0;
        runtime.bodies.forEach(collectStale, stale);
        stale.state = null;
        for (let i = 0; i < stale.count; i++) {
            const eid = stale.eids[i];
            forget(runtime, eid);
            runtime.stamps.delete(eid);
            if (state.has(eid, Pose)) state.remove(eid, Pose);
            bodySetChanged = true;
        }
        if (bodySetChanged)
            resyncConstraints(runtime.constraints, world, runtime.bodies, runtime.isFailed);
    },
};

/** Fill moved dense pose rows in bulk before the engine's single range upload. */
export function composePose(
    runtime: PhysicsRuntime,
    transforms: ReturnType<typeof transformTable>,
    alpha: number,
    body: { shape: Single; halfExtents: Quad } = Body,
): void {
    if (!Compute.device || runtime.movedCount === 0) return;
    const bytes = transforms.bytes;
    if (runtime.renderRecords.buffer !== bytes.buffer)
        runtime.renderRecords = new Float32Array(bytes.buffer);
    const records = runtime.renderRecords;
    const moved = runtime.movedThisTick;
    const shapes = body.shape.column;
    const half = body.halfExtents.column;
    const stride = transforms.rowBytes / 4;
    let first = transforms.capacity;
    let last = -1;
    for (let i = 0; i < runtime.movedCount; i++) {
        const eid = moved[i];
        const row = transforms.rowIndex(eid);
        if (row < 0) continue;
        first = Math.min(first, row);
        last = Math.max(last, row);
        const p = eid * 3;
        const q = eid * 4;
        const base = row * stride;
        records[base] = runtime.prevPos[p] * (1 - alpha) + runtime.currPos[p] * alpha;
        records[base + 1] = runtime.prevPos[p + 1] * (1 - alpha) + runtime.currPos[p + 1] * alpha;
        records[base + 2] = runtime.prevPos[p + 2] * (1 - alpha) + runtime.currPos[p + 2] * alpha;
        records[base + 3] = 0;
        nlerpShortestInto(runtime.prevQuat, q, runtime.currQuat, q, alpha, records, base + 4);
        const shape = shapes[eid];
        const radius = half[q + 3];
        if (shape === ShapeKind.Sphere) {
            records[base + 8] = records[base + 9] = records[base + 10] = 2 * radius;
        } else if (shape === ShapeKind.Capsule) {
            records[base + 8] = records[base + 10] = 2 * radius;
            records[base + 9] = half[q + 1] + radius;
        } else {
            records[base + 8] = 2 * half[q];
            records[base + 9] = 2 * half[q + 1];
            records[base + 10] = 2 * half[q + 2];
        }
        records[base + 11] = 0;
    }
    if (last < first) return;
    const count = last - first + 1;
    transforms.markRange(first, count);
    runtime.counters.bytesUploaded += count * transforms.rowBytes;
}

/** Fill physics-owned pose rows at the end of simulation, before the draw-head table upload. */
export const ComposeSystem: System = {
    name: "compose",
    group: "simulation",
    last: true,
    update(state) {
        const runtime = runtimeFor(state);
        if (!runtime.world) return;
        ensurePoseCapacity(runtime, state.entityHighWater);
        composePose(runtime, transformTable(state), state.time.fixedAlpha, state.of(Body));
    },
};

/**
 * rigid-body physics: installs `Body`/`Spring`/`Joint` and a CPU solver (Rust/WASM kernel, no GPU device
 * needed to step). Opt-in — add it to a scene to run physics (it's not in the default plugins). Nine joint
 * types, mesh/heightfield/compound colliders, sensors, CCD and sleeping ride {@link physicsWorld}.
 *
 * @example
 * ```
 * export const config: Config = { plugins: [PhysicsPlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    components: { Body, Pose, Spring, Joint },
    systems: [SyncSystem, ConstraintSystem, StepSystem, ComposeSystem],
    dependencies: [TransformsPlugin],
    traits: {
        Body: bodyTraits,
        Pose: poseTraits,
        Spring: springTraits,
        Joint: jointTraits,
    },

    initialize(state) {
        state.resource(physicsRuntimeKey, newRuntime).initialized = true;
        transformTable(state).bindMembership(Body);
    },

    async warm(state) {
        const runtime = runtimeFor(state);
        await init(); // async wasm compile — the browser main thread can't compile it synchronously
        runtime.world?.destroy();
        runtime.world = new World({ gravity: { x: 0, y: GRAVITY, z: 0 } });
        clearBodies(runtime);
        ensurePoseCapacity(runtime, state.entityHighWater);
        resetSignatures(state); // the fresh world receives the authored constraint set on its first frame
    },

    dispose(state) {
        const runtime = runtimeFor(state);
        clearBodies(runtime);
        const warnings = signatureWarningsFor(state);
        warnings.joints.clear();
        warnings.springs.clear();
        runtime.world?.destroy();
        runtime.world = null;
        void shutdown();
    },
};

export type {
    ContactEvents,
    ParallelJointConfig,
    RevoluteJointConfig,
    SoftJointConfig,
    SphericalJointConfig,
    WheelJointConfig,
    WorldSnapshot,
} from "./api";
export { BodyType, CLOCK_SLOTS, JointType, type Profile, type StepClock, zeroProfile } from "./api";
export { SoftJoint } from "./api/joints";
export { World } from "./api/world";
export { nlerpShortest } from "./compose";
export { type Hull, type HullFace, Hulls, UNIT_CUBE_ID } from "./hull";
export { bodyCandidates, cursorRay, forwardRay, grabHit, worldToLocal } from "./pick";
export {
    generateRay,
    qRotate,
    type Ray,
    type RayBody,
    type RayHit,
    rayCapsule,
    raycast,
    rayOBB,
    raySphere,
    screenToRay,
} from "./raycast";
// Physics extension surface: what an outside solver or custom tooling needs past the author happy path.
// An outside solver registers the shared components with these traits, derives the authored constraint
// set from the defs/signatures, reads hull geometry from `Hulls`, and orders its systems against these
// anchors. Tooling driving `physicsWorld(state)` (or its own `World`) past the atomic core needs the solver's
// free functions: shape builders, `BodyType`/joint configs, debug draw, `hashWorldState`.
export * as solver from "./solver";
