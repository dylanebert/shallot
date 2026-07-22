import { Compute, entity, f32, type State, type System, sparse, u32, vec4 } from "../../engine";
import { eulerAlias } from "../../engine/utils";
import { BeginFrameSystem, Render } from "../render/core";
import { PrepassSystem } from "../sear/core";
import { slab } from "../slab";
import { Transform } from "../transforms";

// The physics substrate — shared authoring components (`Body`/`Spring`/`Joint`), the CPU raycast + pick
// layer, and a thin typed backend handle (`PhysicsBackend`) a backend plugin (`standard/avbd`'s
// `AvbdPlugin`, later `standard/tumble`'s `TumblePlugin`) installs at `Physics.backend`. The substrate
// owns the schedule contract (this file's `StepSystem` / `ConstraintSystem` / `ComposeSystem`, each
// delegating to the installed handle); a backend owns its mechanism (GPU pipelines, a wasm world) plus a
// richer imperative escape hatch (`Avbd.step`, `Tumble.world`) for anything past the atomic core.
// physics.md "The substrate decision" is the design record.

/** collision-shape tag for {@link Body}. Box collides as an OBB; sphere/capsule as a core + radius; hull as a convex polytope (geometry registered in `Hulls`, referenced by `halfExtents.w` = the hull id). */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

/**
 * a rigid body simulated by the installed physics backend: falls under gravity and collides with other
 * bodies (`mass: 0` = static). Requires a backend plugin (`AvbdPlugin`, `TumblePlugin`) to simulate.
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
    /** spawn position; the backend owns it after spawn. */
    pos: slab(vec4),
    /** spawn orientation, authored as euler degrees like `Transform.rot`; backend-owned after spawn. */
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

// Authoring metadata for the three components above, shared by every backend plugin that registers
// them. `Body`/`Spring`/`Joint` are the same objects across backends (idempotent registration, ecs.md
// "Stable component ids"), so their traits live here once rather than duplicated per backend.

/** {@link Body}'s traits: defaults, its exclusion of {@link Transform}, and the euler-degree `quat` alias. Shared by every backend plugin that registers `Body`. */
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
    // the backend owns the entity's world transform (composed into the firehose each frame), so a
    // Body stands in for Transform: a `Part` on the same entity renders at the body's pose
    provides: [Transform],
    // a Body's orientation is stored as a quaternion but authored as euler degrees, like Transform.rot
    aliases: { quat: eulerAlias("quat") },
};

/** {@link Spring}'s traits: field defaults. Shared by every backend plugin that registers `Spring`. */
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

/** {@link Joint}'s traits: field defaults plus the `stiffness-ang: fixed` parse hook. Shared by every backend plugin that registers `Joint`. */
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

/**
 * a backend-neutral authored spring: two body eids + local anchors + stiffness/rest. What
 * {@link ConstraintSystem} derives from a scene's {@link Spring} entities and hands to
 * {@link PhysicsBackend.setSprings}.
 */
export interface SpringDef {
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    stiffness: number;
    rest: number;
}

/**
 * a backend-neutral authored joint: two body eids + local anchors + the angular lock. What
 * {@link ConstraintSystem} derives from a scene's {@link Joint} entities and hands to
 * {@link PhysicsBackend.setJoints}. Richer per-backend joint authoring (motors, a world anchor, a soft
 * linear stiffness) rides that backend's own escape-hatch API (e.g. `Avbd.step.setJoints`).
 */
export interface JointDef {
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    stiffnessAng: number;
}

/**
 * one body's live pose + velocity, read from whichever backend is installed. May be one or more fixed
 * ticks stale (backend-dependent: a CPU backend serves this tick's fresh state; a GPU mirror is typically
 * 1-2 ticks behind, since the pose crosses a readback ring); sleeping bodies read zero velocity.
 */
export interface BodyState {
    pos: readonly [number, number, number];
    quat: readonly [number, number, number, number];
    vel: readonly [number, number, number];
}

/**
 * the physics substrate's backend contract: a plain-object handle a backend plugin installs via
 * {@link installBackend} in its `warm()`. Covers exactly the coupling points the substrate's shared
 * systems ({@link StepSystem}, {@link ConstraintSystem}, {@link ComposeSystem}) and the pose-read
 * consumers (the character sweep, `pick.ts`) need; everything richer rides the backend's own singleton
 * escape hatch (`Avbd.step`, `Tumble.world`) used imperatively in spawn scripts.
 */
export interface PhysicsBackend {
    /** advance the simulation one fixed step. */
    step(): void;
    /** the live pose + velocity of a body, by eid; `null` before the backend has anything to report. The
     *  pose may be one or more fixed ticks stale (backend-dependent — a CPU backend is fresh, a GPU mirror
     *  is typically 1-2 ticks behind); consumers treat the staleness as contractual. */
    readBody(eid: number): BodyState | null;
    /** move a `mass <= 0` body (a platform, a grab anchor, the character sweep's kinematic upload).
     *  `teleport` skips deriving a velocity from the pose delta; `vel` overrides the derived velocity. */
    setKinematic(
        eid: number,
        pos: readonly [number, number, number],
        quat: readonly [number, number, number, number],
        teleport?: boolean,
        vel?: readonly [number, number, number],
    ): void;
    /** set a dynamic body's linear velocity (a launch impulse, the character push). */
    setVelocity(eid: number, vx: number, vy: number, vz: number): void;
    /** upload the authored spring set, replacing the prior one. */
    setSprings(springs: readonly SpringDef[]): void;
    /** upload the authored joint set, replacing the prior one. */
    setJoints(joints: readonly JointDef[]): void;
    /** the configured world gravity (negative). */
    readonly gravity: number;
    /** the fixed timestep the backend steps at. */
    readonly dt: number;
    /** write the interpolated pose into the `transforms` firehose at `alpha` (render.md's fixedAlpha blend). */
    compose(encoder: GPUCommandEncoder, transforms: GPUBuffer, alpha: number): void;
}

interface PhysicsSingleton {
    backend: PhysicsBackend | null;
}

/** the installed physics backend, or `null` if no backend plugin has warmed. */
export const Physics: PhysicsSingleton = {
    backend: null,
};

/** install a backend handle. Throws if one is already installed (a scene runs exactly one physics backend at a time). Arms the constraint re-upload so the fresh backend receives the authored set on its first frame. Call from a backend plugin's `warm()`. */
export function installBackend(handle: PhysicsBackend): void {
    if (Physics.backend) {
        throw new Error(
            "[physics] a backend is already installed — only one PhysicsBackend can be active per scene",
        );
    }
    Physics.backend = handle;
    resetSignatures();
}

/** uninstall the current backend handle. Call from a backend plugin's `dispose()`. */
export function uninstallBackend(): void {
    Physics.backend = null;
}

/** the fixed-group solver step: the ordering anchor a producer that writes the backend's pose buffer
 *  before the solve (the CPU character sweep's kinematic upload) orders `before:`. Delegates to
 *  {@link Physics.backend}. */
export const StepSystem: System = {
    name: "step",
    group: "fixed",
    update() {
        Physics.backend?.step();
    },
};

// the authored-constraint upload (backend-neutral): derive the SpringDef / JointDef lists from the
// Spring / Joint entities a scene authors and push them to the installed backend, re-uploading ONLY when
// the set changes. The empty-set signature equals the initial `sig` value, so a scene with NO Spring/
// Joint entities never uploads — a scene authoring constraints imperatively (a backend's own escape
// hatch) and one authoring them as components don't fight (use one path per constraint type, not both).
const FNV_BASIS = 2166136261;
const fold = (h: number, v: number): number => Math.imul(h ^ (v >>> 0), 16777619);
const _sigF32 = new Float32Array(1);
const _sigU32 = new Uint32Array(_sigF32.buffer);
const sigBits = (x: number): number => {
    _sigF32[0] = x;
    return _sigU32[0];
};

// last-uploaded signatures — reset by {@link installBackend} so a freshly-installed backend re-uploads the
// authored set on its first frame (the reset is folded into install so a backend can't forget the two-step).
// FNV_BASIS is the empty-set signature, so an unconstrained scene's first frame already matches → no upload.
let springSig = FNV_BASIS;
let jointSig = FNV_BASIS;

function resetSignatures(): void {
    springSig = FNV_BASIS;
    jointSig = FNV_BASIS;
}

function springSignature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Spring])) {
        h = fold(h, eid);
        const a = Spring.a.get(eid);
        const b = Spring.b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps: a same-update realias of an endpoint (destroy+create
        // recycling its eid) leaves a/b unchanged, so without the stamp the re-upload is suppressed and the
        // backend joint pins the NEW occupant at the old anchors (ecs.md "An eid is a borrow").
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

function jointSignature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Joint])) {
        h = fold(h, eid);
        const a = Joint.a.get(eid);
        const b = Joint.b.get(eid);
        h = fold(h, a);
        h = fold(h, b);
        // fold the referenced bodies' create-stamps — see springSignature: a realias of an endpoint must
        // force the re-upload so the backend joint rebinds to the new occupant.
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

function springDefs(state: State): SpringDef[] {
    const out: SpringDef[] = [];
    for (const eid of state.query([Spring])) {
        out.push({
            a: Spring.a.get(eid),
            b: Spring.b.get(eid),
            rA: [Spring.rA.x.get(eid), Spring.rA.y.get(eid), Spring.rA.z.get(eid)],
            rB: [Spring.rB.x.get(eid), Spring.rB.y.get(eid), Spring.rB.z.get(eid)],
            stiffness: Spring.stiffness.get(eid),
            rest: Spring.rest.get(eid),
        });
    }
    return out;
}

function jointDefs(state: State): JointDef[] {
    const out: JointDef[] = [];
    for (const eid of state.query([Joint])) {
        out.push({
            a: Joint.a.get(eid),
            b: Joint.b.get(eid),
            rA: [Joint.rA.x.get(eid), Joint.rA.y.get(eid), Joint.rA.z.get(eid)],
            rB: [Joint.rB.x.get(eid), Joint.rB.y.get(eid), Joint.rB.z.get(eid)],
            stiffnessAng: Joint.stiffnessAng.get(eid),
        });
    }
    return out;
}

/** uploads a scene's authored {@link Spring} / {@link Joint} entities to {@link Physics.backend}, on change only. Fixed group, `before: [StepSystem]` so a constraint authored or edited this frame lands in this frame's solve. */
export const ConstraintSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepSystem],
    update(state) {
        const backend = Physics.backend;
        if (!backend) return;
        const ss = springSignature(state);
        if (ss !== springSig) {
            springSig = ss;
            backend.setSprings(springDefs(state));
        }
        const js = jointSignature(state);
        if (js !== jointSig) {
            jointSig = js;
            backend.setJoints(jointDefs(state));
        }
    },
};

/** scatters {@link Physics.backend}'s live pose into the `transforms` firehose so a Body+Part renders. A `Body` eid's slot is physics-owned: the Transform compose is membership-gated and never touches it (`Body` excludes `Transform`, so the two writers partition the firehose by slot). `after: [BeginFrameSystem]` for the frame encoder; `before: [PrepassSystem]` so every sear geometry pass reads the fresh pose. No-op with no renderer or no transforms firehose (physics runs headless unchanged). */
export const ComposeSystem: System = {
    name: "compose",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state) {
        const backend = Physics.backend;
        if (!backend || !Render.encoder) return;
        const transforms = Compute.buffers.get("transforms");
        if (!transforms) return;
        backend.compose(Render.encoder, transforms, state.time.fixedAlpha);
    },
};
