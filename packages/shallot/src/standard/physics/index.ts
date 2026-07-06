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
import { BVH_FEATURES } from "../bvh/core";
import { type Mirror, MirrorPlugin, mirror } from "../mirror";
import { BeginFrameSystem, Render } from "../render/core";
import { PrepassSystem } from "../sear/core";
import { SlabPlugin, slab } from "../slab";
import { Transform } from "../transforms";
import { Hulls, packHulls } from "./hull";
import { type Inputs, type JointDef, PENALTY_MIN, PhysicsStep, type SpringDef } from "./step";

// AVBD physics — the rigid-body solver re-added to the lean engine, validated against the f64 oracle
// (tests/avbd). `Body` is the authoring surface; the GPU step lives in step.ts (PhysicsStep). Runs the
// full `warmstart` augmented-Lagrangian layer (λ accumulation + the conditional penalty ramp + friction
// + cross-frame persistence: the collide merges last frame's λ/k by feature key, γ decay).
//
// Storage is eid-indexed over `capacity`, persistent across frames — a body's solver state lives at its
// eid slot and survives spawn/despawn. The whole CPU side is firehose: NO per-entity iteration. Each
// frame a single GPU `pack` pass (PackSystem, draw group) scans capacity gated on the Body membership
// bit and (a) compacts the live eids into the dense→eid map (`eids[0]` = count, `eids[1+d]` = eid) the
// solver passes read, and (b) one-time-seeds any newly-spawned body's slot from its authored slabs
// (gated on a GPU `seeded` flag — existing bodies untouched). The fixed-group StepSystem solves from
// last frame's pack output (a 1-frame structural latency — a new body joins the solve next frame).
//
// Body / Transform contract (roadmap): `Body.excludes [Transform]`. `Body` carries pos/quat (spawn
// pose, then physics-owned) + mass/halfExtents/friction on slab; physics owns the GPU pose after spawn.
// BodyComposeSystem scatters the live pose into the `transforms` firehose (PhysicsStep.compose), after
// the Transform compose and before the renderer reads geometry, so a `Body`+`Part` renders at the
// physics-owned pose. A CPU consumer reads the live pose by Mirror-ing `Physics.step.bodies` (the gym does).

const GRAVITY = -10;
const ALPHA = 0.99;
// penalty ramp rate (Eq. 17) + warmstart decay (Eq. 19), the canonical AVBD set. The warmstart layer
// carries λ/k across frames, so the ramp converges from the persisted state — the canonical 1e4 holds
// a resting box to ~mg/k.
const BETA_LIN = 1e4;
// the joint angular penalty-ramp rate (Phase 6.2, joint.ts betaAng) — the canonical AVBD value; contacts
// ignore it, so it only matters once a scene authors joints (via Physics.step.setJoints).
const BETA_ANG = 100;
const GAMMA = 0.999;
// solve iterations — a perf/robustness tradeoff knob, NOT a correctness gate: every fixed count explodes
// on *some* taller stack (iters=4 churns a 16384-body 12-layer pile, iters=10 a taller one), so the choice
// is a deliberate ship tradeoff, not a bug. Production ships 6: it settles a 10-storey stack (the collapse
// showcase wall) where 4 (the paper's count) under-converges and pancakes during the settle, ~1ms@1k on
// lovelace (0.996ms measured, vs 0.575 at 4). The f64 oracle + GPU gates VALIDATE at iters=10
// (corpus.oracle.ts, the gym seeded gates), where the math is proven correct independent of this ship
// value. Raise per-scene via `Physics.step.configure` for a known-harder pile. physics.md "f32 precision"
// / "iters is a free knob".
const ITERATIONS = 6;
// the live-body bound (BVH prims, the eid map, dispatch cap) — the scene runs up to this many bodies at
// once. The full scene `capacity` (Phase 4.7): a body can live at any eid, up to `capacity` at once. The
// per-eid manifold store sizes straight off it (Phase 4.9 robustness): each body owns a fixed pair block,
// so memory scales with `capacity` (~4.6 MB at 1024, ~265 MB at the default 65536 — lower `capacity` to
// shrink it), and a body can't overflow a global pool (no silent fall-through; `checkContactStore` guards
// the device per-binding limit at construction).
const MAX_BODIES = capacity;
// dispatched-color cap (Phase 4): the primal dispatches at most this many colors per iteration, so the
// dispatch count is bounded by the cap not the body count (scratch.md "Dispatch count"). The reference +
// webphysics both cap at 8; the convergence probe found realistic piles color in ≤4, well under it.
const MAX_COLORS = 8;

/** collision-shape tag for {@link Body}. Box collides as an OBB; sphere/capsule as a core + radius; hull as a convex polytope (geometry registered in `Hulls`, referenced by `halfExtents.w` = the hull id). */
export const ShapeKind = { Box: 0, Sphere: 1, Capsule: 2, Hull: 3 } as const;

/**
 * a rigid body simulated by the solver: falls under gravity and collides with other bodies (`mass: 0` = static).
 *
 * @example
 * ```
 * <a body shape="0" pos="0 5 0" half-extents="0.5 0.5 0.5" mass="1" friction="0.5" />
 * <a body shape="1" pos="0 5 0" half-extents="0 0 0 0.5" mass="1" />            <!-- sphere, radius 0.5 -->
 * <a body shape="2" pos="0 5 0" half-extents="0 0.5 0 0.3" mass="1" />          <!-- capsule, half-height 0.5, radius 0.3 -->
 * <a body shape="3" pos="0 5 0" half-extents="1 1 1 2" mass="1" />              <!-- hull id 2, AABB half 1×1×1 -->
 * ```
 */
export const Body = {
    /** the collider, a `ShapeKind`: `Box` (an OBB of `halfExtents`), `Sphere`, `Capsule` (a segment along local Y inflated by the radius), or `Hull` (a convex polytope registered in `Hulls`). */
    shape: slab(u32),
    /** spawn position; the solver owns it after spawn. */
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
    /** angular lock: `0` (default) leaves rotation free (spherical); `∞` locks orientation — author `stiffness-ang: fixed`. */
    stiffnessAng: sparse(f32),
};

interface Physics {
    step: PhysicsStep | null;
}

/** the running physics state — the gym + tools read the GPU pose from `step.bodies` (indexed by `step.eids`) */
export const Physics: Physics = {
    step: null,
};

// the frame-stale readback of `step.colorCount` — word 0 the greedy's used-color count (the readback-bounded
// color loop, Phase 4.9 Lever 1), word 1 the clamped live body count (the color loop's direct dispatch,
// rung 0). Mirror is the sanctioned GPU→CPU readback (physics.md "readback is Mirror"); StepSystem reads the
// snapshot each fixed tick and bounds the primal's color count + dispatch size. MirrorSystem (draw, last)
// flushes it after PackSystem wrote word 1, so the snapshot StepSystem (fixed, earlier) reads is from a
// prior frame — exactly the frame-stale input both bounds want. Allocated in warm (after Mirror.reset),
// released in dispose.
let colorMirror: Mirror | null = null;

// the `Hulls` registry size at the last hull upload — the GPU `hullData` buffer is re-packed + re-uploaded
// (step.setHulls) only when it changes (hulls are static once registered). Reset in warm so a fresh step
// re-uploads the (persistent module-singleton) registry. A size check suffices: hulls aren't mutated in place.
let lastHullCount = -1;

// the membership + authored slab sources the GPU pack gathers from — all stable, fixed-capacity
// buffers: the Body slab `.gpu` (allocated at SlabPlugin.warm) and the `membership` mirror (the
// draw-group `first` MembershipSystem). A draw-group consumer runs after both, so they're always up at
// the call site; a missing one is a wiring bug (SlabPlugin not a dependency), not a frame to skip.
function inputs(): Inputs {
    const membership = Compute.buffers.get("membership");
    const pos = Body.pos.gpu;
    const quat = Body.quat.gpu;
    const half = Body.halfExtents.gpu;
    const mass = Body.mass.gpu;
    const friction = Body.friction.gpu;
    const shape = Body.shape.gpu;
    if (!membership || !pos || !quat || !half || !mass || !friction || !shape) {
        throw new Error("[physics] pack sources missing — declare SlabPlugin as a dependency");
    }
    return { membership, pos, quat, half, mass, friction, shape };
}

// step the solver on the fixed timestep, from the dense→eid map last frame's pack built. Fixed group
// (deterministic dt). The pack runs in the draw group (below), which runs after fixed, so a new body
// joins the solve the next frame — a documented 1-frame structural latency. Always records: an empty
// scene is all-early-out on the GPU (the live count `eids[0]` is GPU-resident), so no CPU count guard.
/** the fixed-group solver step — the ordering anchor a producer that writes the GPU `bodies` buffer before
 *  the solve (the CPU character sweep's kinematic upload) orders `before:` (`physics/core`, the render-anchor shape). */
export const StepSystem: System = {
    name: "step",
    group: "fixed",
    update() {
        const step = Physics.step;
        if (!step || !Compute.device) return;
        // readback-bounded color loop (Phase 4.9 Lever 1) + direct color-loop dispatch (rung 0): bound the
        // primal's dispatched color count to the frame-stale used-color count and size the color loop's
        // direct dispatch off the frame-stale live count, both riding one snapshot ([0] = usedColors from
        // colorize, [1] = liveCount from packScan). No snapshot yet (first frames) → both keep the full
        // cap (the safe cold-start).
        if (colorMirror?.snapshot) {
            const counts = new Uint32Array(colorMirror.snapshot.bytes);
            // word 0 = usedColors (the color-loop bound), word 1 = the clamped live count (the
            // broadphase/LDS regime key + the primal/commit direct dispatch size).
            step.boundColors(counts[0]);
            step.boundBodies(counts[1]);
        }
        const encoder = Compute.device.createCommandEncoder({ label: "physics-step" });
        step.record(encoder);
        Compute.device.queue.submit([encoder.finish()]);
    },
};

// the authored-constraint upload (Phase 6.6): derive the SpringDef / JointDef lists from the Spring / Joint
// entities a scene authors and push them to the step, re-uploading ONLY when the set changes. setJoints
// keeps unchanged slots' live records (warmstart λ/penalty, active flag); the signature gates the
// re-upload to skip the per-frame CPU adjacency rebuild. The empty-set signature equals the initial
// `sig` value, so a scene with NO Spring/Joint entities never uploads — a scene authoring constraints
// imperatively (`Physics.step.setJoints`) and one authoring them as components don't fight (use one
// path per constraint type, not both).
const FNV_BASIS = 2166136261;
const fold = (h: number, v: number): number => Math.imul(h ^ (v >>> 0), 16777619);
const _sigF32 = new Float32Array(1);
const _sigU32 = new Uint32Array(_sigF32.buffer);
const sigBits = (x: number): number => {
    _sigF32[0] = x;
    return _sigU32[0];
};

// last-uploaded signatures — reset in warm (module state survives a State rebuild, like lastHullCount).
// FNV_BASIS is the empty-set signature, so an unconstrained scene's first frame already matches → no upload.
let springSig = FNV_BASIS;
let jointSig = FNV_BASIS;

function springSignature(state: State): number {
    let h = FNV_BASIS;
    for (const eid of state.query([Spring])) {
        h = fold(h, eid);
        h = fold(h, Spring.a.get(eid));
        h = fold(h, Spring.b.get(eid));
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
        h = fold(h, Joint.a.get(eid));
        h = fold(h, Joint.b.get(eid));
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

// push the authored springs/joints to the step before it solves, on change only (above). Fixed group,
// `before: [StepSystem]` so a constraint authored or edited this frame lands in this frame's solve.
const ConstraintSystem: System = {
    name: "constraints",
    group: "fixed",
    before: [StepSystem],
    update(state) {
        const step = Physics.step;
        if (!step) return;
        const ss = springSignature(state);
        if (ss !== springSig) {
            springSig = ss;
            step.setSprings(springDefs(state));
        }
        const js = jointSignature(state);
        if (js !== jointSig) {
            jointSig = js;
            step.setJoints(jointDefs(state));
        }
    },
};

// the GPU firehose pack: scan capacity gated on Body membership → the dense→eid map + the one-time seed
// of any newly-spawned body. Draw group, so it runs after SlabSystem + MembershipSystem (both `first`
// in draw) have flushed the Body slabs + the membership mirror — the pack reads fresh GPU data. Submits
// its own encoder (works headless, no renderer needed); its writes are visible to the next fixed step +
// to BodyComposeSystem (later submits). No CPU per-entity iteration — the membership scan + seed are GPU.
const PackSystem: System = {
    name: "pack",
    group: "draw",
    annotations: { mode: "always" },
    update() {
        const step = Physics.step;
        if (!step || !Compute.device) return;
        // upload the convex-hull geometry the collide pass reads (ShapeKind.Hull bodies) when the registry
        // changed — before the pack seeds a hull body's slot, so its first solve frame reads valid geometry.
        if (Hulls.size !== lastHullCount) {
            lastHullCount = Hulls.size;
            step.setHulls(packHulls());
        }
        const encoder = Compute.device.createCommandEncoder({ label: "physics-pack" });
        step.pack(encoder, inputs());
        Compute.device.queue.submit([encoder.finish()]);
    },
};

// scatter the physics-owned pose into the `transforms` firehose so a Body+Part renders. A render
// producer (physics.md "Body / Transform contract"): `after: [BeginFrameSystem]` so the Transform
// compose already wrote its (stale, for a body eid) slot, `before: [PrepassSystem]` so every sear
// geometry pass reads the corrected slot (render.md "System ordering"). No-op when there's no renderer
// (Render.encoder null) or no transforms firehose — physics runs headless unchanged.
//
// `time.fixedAlpha` (the fraction past the last fixed tick) drives render interpolation: compose blends
// last frame's settled pose → this frame's so a >60Hz render doesn't repeat a fixed-step pose then jump.
const BodyComposeSystem: System = {
    name: "compose",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state) {
        const step = Physics.step;
        if (!step || !Render.encoder) return;
        const transforms = Compute.buffers.get("transforms");
        if (!transforms) return;
        step.compose(Render.encoder, transforms, state.time.fixedAlpha);
    },
};

/**
 * enables the AVBD rigid-body solver and its `Body`, `Spring`, and `Joint` components; opt-in, so add it to run physics (it's not in the default plugins).
 *
 * @example
 * ```
 * export const config: Config = { plugins: [PhysicsPlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const PhysicsPlugin: Plugin = {
    name: "Physics",
    components: { Body, Spring, Joint },
    systems: [ConstraintSystem, StepSystem, PackSystem, BodyComposeSystem],
    // MirrorPlugin: the readback-bounded color loop reads `step.colorCount` through a Mirror (above).
    dependencies: [SlabPlugin, MirrorPlugin],
    // the LBVH broadphase's bounds reduction + radix sort prefer subgroup ops, falling back to an
    // LDS arm where absent (WebKit) — preferred, not required, so a no-subgroup device still runs
    // physics (bvh/core; createBvh reads `device.features` to pick the arm).
    preferredFeatures: BVH_FEATURES,
    traits: {
        Body: {
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
        },
        Spring: {
            defaults: () => ({
                a: 0,
                b: 0,
                rA: [0, 0, 0, 0],
                rB: [0, 0, 0, 0],
                stiffness: 100,
                rest: 1,
            }),
        },
        Joint: {
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
        },
    },

    initialize() {
        Physics.step = null;
    },

    async warm(state: State) {
        if (!Compute.device) return;
        lastHullCount = -1; // force a hull re-upload into the fresh step (the registry persists across states)
        springSig = FNV_BASIS; // a fresh step has no constraints — re-upload the authored set on the first frame
        jointSig = FNV_BASIS;
        // the membership gate templates the pack's per-eid skip test. `build` fixes every component's
        // bit up front, so `bit(Body)` is valid here; `capacity` is the eid range the pack walks.
        const { gen, mask } = state.membership.bit(Body);
        Physics.step = await PhysicsStep.create(Compute.device, capacity, MAX_BODIES, {
            gen,
            mask,
        });
        // static per-step params — the live count is GPU-resident (the pack writes it), not a config field.
        Physics.step.configure({
            dt: Time.FIXED_DT,
            gravity: GRAVITY,
            alpha: ALPHA,
            penalty: PENALTY_MIN, // fresh-contact seed — ramps via betaLin, persists/decays via gamma
            betaLin: BETA_LIN,
            betaAng: BETA_ANG,
            gamma: GAMMA,
            iterations: ITERATIONS,
            maxColors: MAX_COLORS,
        });
        // mirror the used-color count for the readback-bounded color loop (allocated here, after
        // MirrorPlugin.initialize's Mirror.reset, so it survives the build).
        colorMirror = mirror(Physics.step.colorCount);
    },

    dispose() {
        colorMirror?.dispose();
        colorMirror = null;
        Physics.step?.destroy();
        Physics.step = null;
    },
};
