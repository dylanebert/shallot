import {
    Compute,
    capacity,
    entity,
    f32,
    type Plugin,
    type State,
    type System,
    sparse,
    Time,
    u32,
    vec4,
} from "../../engine";
import { eulerAlias } from "../../engine/utils";
import { BeginFrameSystem, Render } from "../render/core";
import { PrepassSystem } from "../sear/core";
import { SlabPlugin, slab } from "../slab";
import { Transform } from "../transforms";
import { nlerpShortest, renderScale } from "./compose";
import { init, type Body as SolverBody, World } from "./engine";
import { Hulls } from "./hull";
import { resetConstraints, resyncConstraints, syncJoints, syncSprings } from "./joints";
import { marshalBody } from "./marshal";

// Physics: the authoring components (`Body`/`Spring`/`Joint`), the CPU raycast + pick layer, and the
// Rust/WASM rigid-body solver behind them. CPU writeback: move events become an interpolated pose written
// into the `transforms` firehose, movers only. Storage is an eid↔solver-body map plus a capacity-sized
// prev/curr pose double buffer; no slab or mirror, since the solver serves this tick's state directly.
// Body marshaling is `marshal.ts`, Spring/Joint marshaling `joints.ts`. An outside solver plugs in through
// `physics/core` (traits, defs, signatures, system anchors) and never through this module's state.

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
    shape: slab(u32),
    /** spawn position; physics owns it after spawn. */
    pos: slab(vec4),
    /** spawn orientation, authored as euler degrees like `Transform.rot`; physics-owned after spawn. */
    quat: slab(vec4),
    /** box/AABB half-extents in `xyz`; `w` doubles as the rounding radius (sphere/capsule) or the `Hull` id (a hull has radius 0, so the lane is free). */
    halfExtents: slab(vec4),
    /** mass in kg; `0` or less marks a static body that never moves. */
    mass: slab(f32),
    /** coulomb friction coefficient: `0` slides freely, higher grips. */
    friction: slab(f32),
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
    a: sparse(entity),
    /** the second body. */
    b: sparse(entity),
    /** anchor point on body `a`, in its local frame. */
    rA: sparse(vec4),
    /** anchor point on body `b`, in its local frame. */
    rB: sparse(vec4),
    /** pull strength; higher is stiffer. */
    stiffness: sparse(f32),
    /** the target distance the spring pulls the anchors toward. */
    rest: sparse(f32),
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
    a: sparse(entity),
    /** the second body. */
    b: sparse(entity),
    /** the pin's anchor on body `a`, in its local frame. */
    rA: sparse(vec4),
    /** the pin's anchor on body `b`, in its local frame. */
    rB: sparse(vec4),
    /** angular lock: `0` (default) leaves rotation free (spherical); `∞` locks orientation (author `stiffness-ang: fixed`). */
    stiffnessAng: sparse(f32),
};

// Authoring metadata for the three components above, shared with any extension solver that registers
// them. `Body`/`Spring`/`Joint` are the same objects across plugins (idempotent registration, ecs.md
// "Stable component ids"), so their traits live here once.

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

/** an authored joint: two body eids + local anchors + the angular lock, derived from a scene's {@link Joint} entities by {@link jointDefs}. Richer joints (motors, limits, the nine solver joint types) ride {@link Physics.world}. */
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

// last-uploaded signatures — reset on warm so a fresh world re-uploads the authored set on its first frame.
// FNV_BASIS is the empty-set signature, so an unconstrained scene's first frame already matches → no upload.
let springSig = FNV_BASIS;
let jointSig = FNV_BASIS;

/** re-arm the constraint upload and its warn dedupe, as a fresh world does on warm. */
export function resetSignatures(): void {
    springSig = FNV_BASIS;
    jointSig = FNV_BASIS;
    warnedJointEids.clear();
    warnedSpringEids.clear();
}

/** a hash of the authored {@link Spring} set, endpoint create-stamps included: an uploader re-uploads only when it changes. */
export function springSignature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Spring])) {
        h = fold(h, eid);
        const a = Spring.a.get(eid);
        const b = Spring.b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps: a same-update realias of an endpoint (destroy+create
        // recycling its eid) leaves a/b unchanged, so without the stamp the re-upload is suppressed and the
        // solver joint pins the NEW occupant at the old anchors (ecs.md "An eid is a borrow").
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

/** a hash of the authored {@link Joint} set, endpoint create-stamps included: the {@link springSignature} twin. */
export function jointSignature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Joint])) {
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

// warned-eid dedupe for the stiffness guard — keyed on entity id so a re-upload triggered by an
// unrelated constraint change does not re-warn the same invalid def. Cleared on warm
// (resetSignatures) so a fresh world re-warns. A def whose stiffness is fixed (invalid → valid) is
// removed from the set so a later regression to invalid re-warns.
const warnedJointEids = new Set<number>();
const warnedSpringEids = new Set<number>();

/** the authored {@link Spring} set as {@link SpringDef}s, dropping (and warning once for) a negative or NaN stiffness. */
export function springDefs(state: State): SpringDef[] {
    const out: SpringDef[] = [];
    for (const eid of state.query([Spring])) {
        const stiffness = Spring.stiffness.get(eid);
        // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
        // 0 and ∞ are valid authored values (0 = non-positive → downstream skip; ∞ = rigid) — only negative
        // and NaN are rejected, at the authoring layer so every solver inherits one behavior.
        if (Number.isNaN(stiffness) || stiffness < 0) {
            if (!warnedSpringEids.has(eid)) {
                console.warn(
                    `[physics] spring (a: ${Spring.a.get(eid)}, b: ${Spring.b.get(eid)}) has negative or NaN stiffness — skipped`,
                );
                warnedSpringEids.add(eid);
            }
            continue;
        }
        warnedSpringEids.delete(eid);
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
}

/** the authored {@link Joint} set as {@link JointDef}s, dropping (and warning once for) a negative or NaN angular stiffness. */
export function jointDefs(state: State): JointDef[] {
    const out: JointDef[] = [];
    for (const eid of state.query([Joint])) {
        const stiffnessAng = Joint.stiffnessAng.get(eid);
        // NaN is transparent to comparison-only guards (NaN < 0 is false), so state finiteness explicitly.
        // 0 (spherical) and ∞ (fixed) are valid authored values — only negative and NaN are rejected, at the
        // authoring layer so every solver inherits one behavior.
        if (Number.isNaN(stiffnessAng) || stiffnessAng < 0) {
            if (!warnedJointEids.has(eid)) {
                console.warn(
                    `[physics] joint (a: ${Joint.a.get(eid)}, b: ${Joint.b.get(eid)}) has negative or NaN angular stiffness — skipped`,
                );
                warnedJointEids.add(eid);
            }
            continue;
        }
        warnedJointEids.delete(eid);
        out.push({
            a: Joint.a.get(eid),
            b: Joint.b.get(eid),
            rA: [Joint.rA.x.get(eid), Joint.rA.y.get(eid), Joint.rA.z.get(eid)],
            rB: [Joint.rB.x.get(eid), Joint.rB.y.get(eid), Joint.rB.z.get(eid)],
            stiffnessAng,
        });
    }
    return out;
}

const GRAVITY = -10;
const SUBSTEPS = 4; // the solver's own recommended sub-step count (World.step's default)

let world: World | null = null;
const bodies = new Map<number, SolverBody>();
// the create-stamp each body was marshaled at (ecs.md "An eid is a borrow"). Presence in `bodies` catches a
// plain spawn/despawn; a same-update destroy+create recycling an eid keeps Body membership AND the map entry,
// so the stamp is the only signal that the slot now holds a new body, and a mismatch re-marshals it.
const stamps = new Map<number, number>();
// last pose passed to setKinematic per eid — setKinematic derives a platform's velocity from its per-step
// pose delta against this; teleport resets it so the derived delta is 0.
const kinPrev = new Map<number, [number, number, number]>();
// bodies whose marshal failed (an unregistered/unbuildable hull) — keyed to the stamp + hull-registry size
// they failed at, so SyncSystem retries the marshal (and re-warns) only when the eid recycles or a new hull
// is registered, never every frame.
const failed = new Map<number, { stamp: number; hulls: number }>();
// an endpoint missing from `bodies` because its marshal failed is a pending marshal (it will retry, and so
// will its constraint) rather than a non-`Body` reference (joints.ts).
const isDeferred = (eid: number): boolean => failed.has(eid);

// render-interpolation double buffer, capacity-sized flat arrays indexed by eid (3 lanes pos, 4 lanes quat).
// Rewritten only for a body that moved this fixed tick (from `getBodyEvents`), so a sleeping/static body's
// prev==curr holds from whenever it last moved (or its spawn pose) — compose then blends a no-op.
let prevPos = new Float32Array(0);
let prevQuat = new Float32Array(0);
let currPos = new Float32Array(0);
let currQuat = new Float32Array(0);
// the eids `getBodyEvents` reported this fixed tick — compose (draw group, every render frame) rewrites
// exactly these into the transforms firehose until the next fixed tick's move events replace the set.
const movedThisTick = new Set<number>();

function seedPose(eid: number): void {
    const p = eid * 3;
    const q = eid * 4;
    prevPos[p] = currPos[p] = Body.pos.x.get(eid);
    prevPos[p + 1] = currPos[p + 1] = Body.pos.y.get(eid);
    prevPos[p + 2] = currPos[p + 2] = Body.pos.z.get(eid);
    prevQuat[q] = currQuat[q] = Body.quat.x.get(eid);
    prevQuat[q + 1] = currQuat[q + 1] = Body.quat.y.get(eid);
    prevQuat[q + 2] = currQuat[q + 2] = Body.quat.z.get(eid);
    prevQuat[q + 3] = currQuat[q + 3] = Body.quat.w.get(eid);
}

function forget(eid: number): void {
    bodies.get(eid)?.destroy();
    bodies.delete(eid);
    kinPrev.delete(eid);
    movedThisTick.delete(eid);
}

function clearBodies(): void {
    bodies.clear();
    stamps.clear();
    kinPrev.clear();
    movedThisTick.clear();
    failed.clear();
    resetConstraints();
}

/**
 * the running physics world. `world` is the solver escape hatch: joint types past `Joint`, sensors,
 * contact/hit events, mesh/heightfield/compound colliders and native queries; `null` until
 * {@link PhysicsPlugin} warms. `body(eid)` bridges a `Body` entity to its live solver handle (`null` before
 * its first fixed tick). The pose API reads and drives bodies by eid and no-ops before warm.
 */
export const Physics: {
    world: World | null;
    body(eid: number): SolverBody | null;
    readBody(eid: number): BodyState | null;
    setKinematic(
        eid: number,
        pos: readonly [number, number, number],
        quat: readonly [number, number, number, number],
        teleport?: boolean,
        vel?: readonly [number, number, number],
    ): void;
    setVelocity(eid: number, vx: number, vy: number, vz: number): void;
    readonly gravity: number;
    readonly dt: number;
} = {
    world: null,
    body: (eid) => bodies.get(eid) ?? null,
    /** the live pose + velocity of a body, by eid; `null` before its first fixed tick or for a non-`Body` eid. */
    readBody(eid) {
        const tb = bodies.get(eid);
        if (!tb) return null;
        const pos = tb.getPosition();
        const quat = tb.getRotation();
        const vel = tb.getLinearVelocity();
        return {
            pos: [pos.x, pos.y, pos.z],
            quat: [quat.v.x, quat.v.y, quat.v.z, quat.s],
            vel: [vel.x, vel.y, vel.z],
        };
    },
    /** move a `mass <= 0` body (a platform, a grab anchor, the character sweep). `teleport` skips deriving a velocity from the pose delta; `vel` overrides the derived velocity. */
    setKinematic(eid, pos, quat, teleport = false, vel) {
        const tb = bodies.get(eid);
        if (!tb) return;
        let prev = kinPrev.get(eid);
        // a slept kinematic body that only setTransform moves emits no move event, so its firehose slot would
        // keep the stale pose while readBody sees the new one. Waking it on a real move makes the solver report
        // the move; a same-pose call leaves it free to sleep (a parked platform re-asserting its pose).
        const moved =
            !prev || teleport || pos[0] !== prev[0] || pos[1] !== prev[1] || pos[2] !== prev[2];
        tb.setTransform(
            { x: pos[0], y: pos[1], z: pos[2] },
            { v: { x: quat[0], y: quat[1], z: quat[2] }, s: quat[3] },
        );
        if (!prev || teleport) {
            prev = [pos[0], pos[1], pos[2]];
            kinPrev.set(eid, prev);
        }
        const dt = Time.FIXED_DT;
        const v = vel ?? [
            (pos[0] - prev[0]) / dt,
            (pos[1] - prev[1]) / dt,
            (pos[2] - prev[2]) / dt,
        ];
        tb.setLinearVelocity({ x: v[0], y: v[1], z: v[2] });
        // setTransform never wakes and setLinearVelocity wakes only on a nonzero velocity, so a zero-velocity
        // teleport (or a move whose derived velocity rounds to zero) needs an explicit wake.
        if (moved && !tb.isAwake()) tb.setAwake(true);
        prev[0] = pos[0];
        prev[1] = pos[1];
        prev[2] = pos[2];
    },
    /** set a dynamic body's linear velocity (a launch impulse, the character push). */
    setVelocity(eid, vx, vy, vz) {
        bodies.get(eid)?.setLinearVelocity({ x: vx, y: vy, z: vz });
    },
    /** the world gravity (negative), `0` before warm. */
    get gravity() {
        return world ? world.getGravity().y : 0;
    },
    /** the fixed timestep physics steps at. */
    get dt() {
        return Time.FIXED_DT;
    },
};

/** the fixed-group solver step: the ordering anchor a producer that moves bodies before the solve (the character sweep's kinematic upload) orders `before:`. */
export const StepSystem: System = {
    name: "step",
    group: "fixed",
    update() {
        if (!world) return;
        world.step(Time.FIXED_DT, SUBSTEPS);
        movedThisTick.clear();
        const events = world.getBodyEvents();
        for (let i = 0; i < events.count; i++) {
            const ev = events.moveEvents[i];
            const eid = ev.userData as number;
            const p = eid * 3;
            const q = eid * 4;
            prevPos[p] = currPos[p];
            prevPos[p + 1] = currPos[p + 1];
            prevPos[p + 2] = currPos[p + 2];
            prevQuat[q] = currQuat[q];
            prevQuat[q + 1] = currQuat[q + 1];
            prevQuat[q + 2] = currQuat[q + 2];
            prevQuat[q + 3] = currQuat[q + 3];
            currPos[p] = ev.transform.p.x;
            currPos[p + 1] = ev.transform.p.y;
            currPos[p + 2] = ev.transform.p.z;
            currQuat[q] = ev.transform.q.v.x;
            currQuat[q + 1] = ev.transform.q.v.y;
            currQuat[q + 2] = ev.transform.q.v.z;
            currQuat[q + 3] = ev.transform.q.s;
            movedThisTick.add(eid);
        }
    },
};

/** uploads a scene's authored {@link Spring} / {@link Joint} entities to the solver, on change only. Fixed group, `before: [StepSystem]` so a constraint authored or edited this frame lands in this frame's solve. */
export const ConstraintSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepSystem],
    update(state) {
        if (!world) return;
        const ss = springSignature(state);
        if (ss !== springSig) {
            springSig = ss;
            syncSprings(world, bodies, springDefs(state), isDeferred);
        }
        const js = jointSignature(state);
        if (js !== jointSig) {
            jointSig = js;
            syncJoints(world, bodies, jointDefs(state), isDeferred);
        }
    },
};

// membership-driven create/destroy, ascending eid order (state.query's natural order — creation order is
// load-bearing for solver determinism). Runs every fixed tick before the solve so a body spawned this frame
// joins THIS tick's step.
const SyncSystem: System = {
    name: "physics-sync",
    group: "fixed",
    before: [ConstraintSystem, StepSystem],
    update(state: State) {
        if (!world) return;
        // a deferred body finally marshaling (or a body going stale) is the transition a dropped constraint
        // waits on, and `ConstraintSystem` re-uploads on an authored signature change only, so the constraint
        // re-sync is pumped from here on any body-set change (a no-op walk when nothing was dropped, joints.ts).
        let bodySetChanged = false;
        for (const eid of state.query([Body])) {
            const stamp = state.stamp(eid);
            if (bodies.has(eid)) {
                if (stamps.get(eid) === stamp) continue;
                forget(eid); // recycled to a new Body in one update
            }
            const f = failed.get(eid);
            if (f && f.stamp === stamp && f.hulls === Hulls.size) continue;
            const tb = marshalBody(world, eid);
            if (!tb) {
                failed.set(eid, { stamp, hulls: Hulls.size });
                continue;
            }
            failed.delete(eid);
            bodies.set(eid, tb);
            stamps.set(eid, stamp);
            bodySetChanged = true;
            seedPose(eid);
        }
        for (const eid of failed.keys()) {
            if (!state.has(eid, Body)) failed.delete(eid);
        }
        const stale: number[] = [];
        for (const eid of bodies.keys()) {
            if (!state.has(eid, Body)) stale.push(eid);
        }
        for (const eid of stale) {
            forget(eid);
            stamps.delete(eid);
            bodySetChanged = true;
        }
        if (bodySetChanged) resyncConstraints(world, bodies, isDeferred);
    },
};

// one reused Xform-shaped record (48 B / 12 f32: pos.xyz+pad, quat.xyzw, scale.xyz+pad — the `Xform` schema).
const _record = new Float32Array(12);

/** write the movers' interpolated pose into the `transforms` firehose at `alpha` (render.md's fixedAlpha blend). */
export function composePose(transforms: GPUBuffer, alpha: number): void {
    if (!Compute.device) return;
    for (const eid of movedThisTick) {
        const p = eid * 3;
        const q = eid * 4;
        const quat = nlerpShortest(
            [prevQuat[q], prevQuat[q + 1], prevQuat[q + 2], prevQuat[q + 3]],
            [currQuat[q], currQuat[q + 1], currQuat[q + 2], currQuat[q + 3]],
            alpha,
        );
        const scale = renderScale(
            Body.shape.get(eid),
            [Body.halfExtents.x.get(eid), Body.halfExtents.y.get(eid), Body.halfExtents.z.get(eid)],
            Body.halfExtents.w.get(eid),
        );
        _record[0] = prevPos[p] * (1 - alpha) + currPos[p] * alpha;
        _record[1] = prevPos[p + 1] * (1 - alpha) + currPos[p + 1] * alpha;
        _record[2] = prevPos[p + 2] * (1 - alpha) + currPos[p + 2] * alpha;
        _record[3] = 0;
        _record[4] = quat[0];
        _record[5] = quat[1];
        _record[6] = quat[2];
        _record[7] = quat[3];
        _record[8] = scale[0];
        _record[9] = scale[1];
        _record[10] = scale[2];
        _record[11] = 0;
        Compute.device.queue.writeBuffer(transforms, eid * 48, _record);
    }
}

/** scatters the live pose into the `transforms` firehose so a Body+Part renders. A `Body` eid's slot is physics-owned: the Transform compose is membership-gated and never touches it (`Body` excludes `Transform`, so the two writers partition the firehose by slot). `after: [BeginFrameSystem]`; `before: [PrepassSystem]` so every sear geometry pass reads the fresh pose. No-op with no renderer or no transforms firehose (physics runs headless unchanged). */
export const ComposeSystem: System = {
    name: "compose",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state) {
        if (!world || !Render.encoder) return;
        const transforms = Compute.buffers.get("transforms");
        if (!transforms) return;
        composePose(transforms, state.time.fixedAlpha);
    },
};

/**
 * rigid-body physics: installs `Body`/`Spring`/`Joint` and a CPU solver (Rust/WASM kernel, no GPU device
 * needed to step). Opt-in — add it to a scene to run physics (it's not in the default plugins). Nine joint
 * types, mesh/heightfield/compound colliders, sensors, CCD and sleeping ride {@link Physics.world}.
 *
 * @example
 * ```
 * export const config: Config = { plugins: [PhysicsPlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    components: { Body, Spring, Joint },
    systems: [SyncSystem, ConstraintSystem, StepSystem, ComposeSystem],
    dependencies: [SlabPlugin],
    traits: {
        Body: bodyTraits,
        Spring: springTraits,
        Joint: jointTraits,
    },

    initialize() {
        Physics.world = null;
    },

    async warm() {
        await init(); // async wasm compile — the browser main thread can't compile it synchronously
        world?.destroy();
        world = new World({ gravity: { x: 0, y: GRAVITY, z: 0 } });
        Physics.world = world;
        clearBodies();
        prevPos = new Float32Array(capacity * 3);
        prevQuat = new Float32Array(capacity * 4);
        currPos = new Float32Array(capacity * 3);
        currQuat = new Float32Array(capacity * 4);
        resetSignatures(); // the fresh world receives the authored constraint set on its first frame
    },

    dispose() {
        clearBodies();
        world?.destroy();
        world = null;
        Physics.world = null;
    },
};
