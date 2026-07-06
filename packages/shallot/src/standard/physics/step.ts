// The AVBD time step on the GPU — the full warmstart layer (primal skeleton + dual + cross-frame
// persistence). One step is a fixed sequence of compute passes that mirror the CPU oracle
// (tests/avbd/solver.ts step()):
//
//   aabb       — each body's oriented-box world-AABB → the broadphase prims
//   broadphase — writes each live body's per-eid FIXED block `pairList[eid·PAIRS_PER_BODY + k]` directly
//                (nearest-K + static-pin, unused slots INVALID). Two regimes keyed on the frame-stale live
//                count: ≤ smallN = one O(n²) dispatch over the prims (the gameplay regime is structure-
//                tax-bound, not work-bound — C1.0); above = LBVH build + box-overlap descent. Shared
//                candidate/emit WGSL keeps the blocks identical, so warmstart carries across a flip.
//   collide    — box-box SAT (collide.ts) per per-eid pair slot; writes/warmstarts the manifold IN PLACE
//                in the persistent `pairContacts` (the (a,b)+feature key carries λ/k — no hash, no separate store)
//   inertial   — per-body inertial target + adaptive warmstart reposition (Eq. 2 + VBD)
//   colorize   — incremental-greedy body coloring, capped at maxColors (folds past it); publishes the
//                used-color count to `colorCount` for the readback-bounded color loop (Phase 4.9 Lever 1).
//                In the small-N regime the CSR build + coloring run as ONE fused single-WG dispatch
//                (CSR_COLOR_SMALL_WGSL — C1.1); the multi-WG passes are the at-scale regime
//   primal     — colored Gauss-Seidel: `boundColors`-many color-passes/iter (min(maxColors, usedColors +
//                COLOR_MARGIN), the frame-stale readback count), each dispatched DIRECT off the
//                frame-stale live count + BODY_MARGIN (`boundBodies`, rung 0), force stamp + 6×6 LDLᵀ
//   dual       — λ ← F + conditional penalty ramp + friction stick (manifold.ts updateDual), in place in
//                pairContacts so next frame's collide warmstarts off it
//   velocity   — BDF1 velocity recovery
//
// The contact force law (Taylor C, cone-clamped F, the 6×6 Jacobian/Hessian stamp) is one shared
// `contactForce` (CONTACT_FORCE_WGSL) the primal and dual both read — the reference inlines it in
// both `updatePrimal` and `updateDual`. The collide carries λ/k across frames in the persistent
// `pairContacts` store (the GPU reconstruction of the reference's force-list manifold persistence,
// Eq. 19) — no cache pass, no separate warmCache (the webphysics persistent per-pair-slot model,
// Phase 4.9 sized per-active-pair: storage + the per-pair slot contract live in physics.md "Storage").
// The GPU runs warmstart-only; the CPU oracle (tests/avbd/solver.ts) keeps the penalty/dual phase-ladder
// layers for the build-up oracle tests.
//
// Validated against the f64 oracle in the gym `pile` scenario (the canonical real-GPU + perf
// home): free-fall = closed form, resting box → margin, and the single-step-exact gate (GPU contacts
// fed to the oracle, one step each, compared).
//
// Storage is eid-indexed over `capacity` (`bodies[col*eidCap + eid]`), persistent across frames — a
// body's solver state lives at its eid slot and survives spawn/despawn (no dense scaffold, no sim-reset).
// The GPU `pack` (an eid-SORTED multi-workgroup count→scan→scatter over the Body membership bitset,
// C1.3) writes the dense→eid map (`eids[0]` = live count, `eids[1+d]` = the d-th live eid) +
// one-time-seeds a newly-spawned body's slot from its authored `Body` slabs. The warmstart slot is the
// owner eid's FIXED per-body block (`eid·PAIRS_PER_BODY + k`), so it's stable across frames unless the
// owner's own candidate set flickers (local fragility — Phase 4.9 robustness, scratch.md). Body passes
// dispatch indirect off the live count (the primal/commit color loop direct off its frame-stale readback,
// rung 0); the per-eid-block passes (collide / dual / CSR) dispatch indirect off `pairArgs`
// (= liveCount · PAIRS_PER_BODY lanes). f32 throughout (rebuild's f32-first;
// quantization deferred per gpu.md rule 8). The body + contact buffers are SoA cols-buffers (gpu.md
// consolidation #1). Per-body CSR adjacency feeds the primal: each body reads only its own contacts.

import { Compute, checkStorageBinding } from "../../engine";
import { XFORM_WGSL } from "../../engine/utils/core";
// the shared LBVH builder (roadmap "Subgroup-first algorithms": physics is a consumer of the same
// rendering-unaware builder a native-RT path would use). standard → extras is the documented exception
// for this shared GPU primitive (exports.md `bvh/core`), not the onion default.
import { BVH_ROOT_WGSL, type Bvh, createBvh } from "../bvh/core";
import {
    BOXBOX_WGSL,
    HELPERS_WGSL,
    HULL_CORE_WGSL,
    HULL_SAT_WGSL,
    MAX_CONTACTS,
    ROUNDED_POLY_WGSL,
    ROUNDED_WGSL,
    SPECULATIVE_DISTANCE,
} from "./collide";

/** logical columns per body in the eid-indexed `bodies` SoA cols-buffer (`bodies[col*eidCap + eid]`) */
export const BODY_VEC4 = 12;
/** columns per body in the `solveOut` double-buffer scratch (`solveOut[col*eidCap + eid]`): pos, quat */
const SOLVE_VEC4 = 2;
/** logical columns per contact record in the persistent `pairContacts` SoA cols-buffer
 * (`pairContacts[col*recordCap + rec]`) — meta(type,a,b,feature) / normal / rA / rB / c0 / penalty / lambda */
export const CONTACT_VEC4 = 7;
/**
 * contact records per persistent manifold block — one block holds a body pair's whole manifold at a stable
 * per-pair slot (Phase 4.7, the webphysics model). The SAT reduces a pair to MAX_CONTACTS (4, the Jolt
 * spread set — Phase 4.8.1), so a block of that size holds every contact; the collide writes this frame's
 * contacts in place + carries λ/k off last frame's records in the same slot (no hash, no separate store).
 * Matches the oracle's `Manifold.contacts`.
 */
export const CONTACTS_PER_PAIR = MAX_CONTACTS;
/**
 * the per-body FIXED-BLOCK slot count (Phase 4.9 robustness, scratch.md "warmstart addressing") — each live
 * body owns a fixed block of `PAIRS_PER_BODY` pair slots at base `eid · PAIRS_PER_BODY` in BOTH the pair list
 * (the descent's `vec2<u32>` output) AND the persistent `pairContacts` manifold store. The base is a function
 * of the owner's eid alone, so a flicker in one body's owned-candidate set churns only THAT body's slots —
 * local warmstart fragility (webphysics `broadPhase.ts` `bodyBase = body · pairsPerBody`), not the global
 * fragility a prefix-sum compaction has (any body's count change shifts every downstream slot → total
 * warmstart collapse on a churning pile). A body descends the BVH once, keeps the NEAREST by center-dist² +
 * pins statics (the importance prune, webphysics), drops the farthest + bumps `counters[3]` (a graceful drop
 * of the least-important candidate). 8 is generous over the ~3-7 a settled pile owns.
 *
 * `pairContacts = capacity · PAIRS_PER_BODY · CONTACTS_PER_PAIR · CONTACT_VEC4 · 16 B` = 235 MB at 65536 —
 * the SAME as the prior compaction (whose win over the original 16/body block was the 8 vs 16, not the
 * compaction itself), now pair-stable. No global pool ⇒ no `counters[4]` (a body can't exceed its own block;
 * the prune bounds it).
 */
export const PAIRS_PER_BODY = 8;
/**
 * one authored constraint's per-body adjacency entry in `constraintList` (AoS, 3 vec4). Springs (Phase
 * 6.1) and joints (Phase 6.2) SHARE this list + the `constraintCsr` adjacency — the binding budget forces
 * it (the primal is at the `maxStorageBuffersPerShaderStage` floor, so a joint can't add a primal/coloring
 * binding; physics.md "phase ladder"). A constraint appears in BOTH bodies' lists, each entry from the
 * owner's frame (`rSelf` = the owner's anchor, `otherEid` = the partner), and `kind` ({@link KIND_SPRING}
 * / {@link KIND_JOINT}) discriminates the stamp:
 *   0 (rSelf.xyz, w = stiffness | unused[joint])   1 (rOther.xyz, w = rest | unused[joint])
 *   2 (bitcast(otherEid), kind, recordIndex[joint], flags[joint] = isA|rigidLin<<1|rigidAng<<2)
 * A spring is stateless (`f = stiffness·C`, no λ/ramp/warmstart) so it stamps from the entry alone; a joint
 * carries warmstartable λ/penalty/c0 in a per-joint {@link JOINT_REC_VEC4} record (`jointRecords`,
 * `recordIndex`), the entry holding only the geometry the primal + coloring read.
 */
export const CONSTRAINT_VEC4 = 3;
/** `constraintList` entry kinds (the `kind` discriminator, entry vec4[2].y) */
export const KIND_SPRING = 1;
export const KIND_JOINT = 2;
/**
 * one joint's persistent record in `jointRecords` (AoS, 12 vec4 — Phase 6.2). The hard `Force` (joint.ts,
 * a port of joint.cpp) carries per-joint mutable dual state warmstarted across frames (λ + a per-iteration
 * penalty ramp + the rigid-stabilization gap `c0`), the per-joint geometry the per-body entries can't hold
 * (init/dual are dispatched one-thread-per-joint, not per-endpoint), and the recycle-version guard:
 *   0 (bitcast a, b, versionA, versionB)   1 (rA.xyz, stiffnessLin)   2 (rB.xyz, stiffnessAng)
 *   3 (torqueArm, bitcast active, _, _)    4 (penaltyLin.xyz)  5 (penaltyAng.xyz)
 *   6 (lambdaLin.xyz)  7 (lambdaAng.xyz)   8 (c0Lin.xyz)  9 (c0Ang.xyz)
 *  10 (motorAxis.xyz, motorMaxTorque)     11 (motorSpeed, motorLambda, motorPenalty, _)
 * `torqueArm` + `active` are GPU-written (jointInit); the CPU seeds geometry + versions + zeroed state. The
 * motor (a 1-DOF force-clamped angular drive, avbd-demo2d motor.cpp; `maxTorque > 0` activates it) rides cols
 * 10/11, which jointInit does NOT rewrite — its static axis/speed/maxTorque persist from `setJoints`, and its
 * λ/penalty warmstart in 11.y/.z. `active`: 0 inactive, 1 active, 2 fresh (jointInit runs the anchor-coincidence
 * guard once). ∞ stiffness (rigid) is the `1e30` sentinel (`> 1e29` reads rigid), matching the harness JSON map.
 */
export const JOINT_REC_VEC4 = 12;
/** ∞-stiffness sentinel — `> RIGID_THRESHOLD` reads "rigid" on the GPU (f32 can't compare to true inf cleanly) */
export const RIGID_STIFFNESS = 1e30;
/** joint hard-conflict coloring-repair rounds (webphysics `BODY_COLOR_HARD_REPAIR_ROUNDS`) — a hard
 * (dynamic-dynamic joint) pair degrading to same-color Jacobi destabilizes, so the greedy's tolerated
 * fold is repaired: 2 rounds of lower-eid-recolors, validated by the observable coloring-split invariant. */
export const JOINT_REPAIR_ROUNDS = 2;
/** the coloring's hard ceiling = the 32-wide usedMask width; `maxColors` clamps to it, so the primal
 * dispatches at most this many colors and the color uniform needs exactly this many slots */
export const COLOR_CAP = 32;
/**
 * the readback-bounded color loop's safety margin (Phase 4.9 Lever 1): the primal dispatches
 * `min(maxColors, usedColors + COLOR_MARGIN)` color-passes per iteration, where `usedColors` is a
 * frame-stale readback of the greedy's actual color count (`colorCount`, written by `colorize`). The
 * margin covers the readback's 1-2 frame staleness — the incremental greedy's chromatic number shifts by
 * ≤1 per frame on a settling pile, so one insurance color keeps the loop ≥ this frame's true count; a
 * frame that densifies further under-dispatches once (a soft convergence dip the next readback catches —
 * the same self-healing the live-count margin below relies on). It's a frame-staleness margin (the
 * profiler-counter class, gpu.md), not a tuned solver tolerance.
 */
export const COLOR_MARGIN = 1;
/**
 * the direct color-loop dispatch's live-count margin in bodies (dispatch-ladder rung 0): the primal/commit
 * color loop dispatches `ceil((liveCount + BODY_MARGIN) / 64)` workgroups DIRECT off the frame-stale
 * `colorCount[1]` readback (written by `packScan`) — Dawn injects a validation pass per indirect call
 * (measured ≈ 2× the direct unit cost, physics.md "Dispatch count"), so the loop's `iters × colors × 2`
 * dispatches are the place a direct dispatch pays. Over-dispatch is correctness-safe (every body pass
 * early-outs on `d >= eids[0]`); the margin only covers under-dispatch from a spawn burst inside the
 * readback's 1-2 frame staleness — one workgroup's worth of headroom, after which a burst body misses
 * one solve step and the next readback catches it (the spawn-despawn gym gate's hazard). Full-cap on
 * cold start / `configure`, like {@link COLOR_MARGIN}'s `boundColors` pattern.
 */
export const BODY_MARGIN = 64;
/**
 * box-box contact type tag — the source-agnostic seam (joints/kinematic/voxel append other tags later).
 * 1, not 0, so a zeroed (cleared) pairContacts record reads as inactive (tag 0) and the solve skips it.
 */
export const CONSTRAINT_CONTACT = 1;
/** dual-layer penalty seed — the contact penalty starts here and ramps via `betaLin` (manifold.ts) */
export const PENALTY_MIN = 1.0;

/**
 * the per-eid manifold store (`pairContacts`) is the step's largest single storage binding —
 * `eidCap · PAIRS_PER_BODY · CONTACTS_PER_PAIR · CONTACT_VEC4 · 16 B`, dominating {@link PhysicsStep.bytes}.
 * Guard its size against the device's per-binding limit so a high `capacity` fails loud + clear here —
 * naming the buffer, the needed-vs-available, and the remedy — rather than at an opaque bind-group
 * validation error. `acquireDevice` requests the adapter's full `maxStorageBufferBindingSize`
 * (engine/runtime/gpu.ts), so on real hardware this only trips at a genuinely huge capacity past the device
 * ceiling (desktop + Deck expose ~2 GB). Pure (eidCount + limit) so a unit test exercises it with no device.
 */
export function checkContactStore(eidCount: number, maxBinding: number): void {
    const bytes = eidCount * PAIRS_PER_BODY * CONTACTS_PER_PAIR * CONTACT_VEC4 * 16;
    checkStorageBinding(
        `[physics] the contact store (${eidCount} eids)`,
        bytes,
        maxBinding,
        "Lower the entity capacity (the contact store sizes to it) or PAIRS_PER_BODY, or use a device " +
            "with a higher storage-binding limit.",
    );
}

// bodies columns (SoA, eid-indexed `bodies[col*eidCap + eid]`). pos/quat (0,1) are solver-mutated; the rest are
// per-step caches + constant mass props. mass <= 0 marks a static body (skipped in primal + velocity).
//   0 posLin   1 posAng   2 inertialLin   3 inertialAng   4 initialLin   5 initialAng
//   6 velLin   7 velAng   8 prevVelLin    9 moment.xyz/mass.w   10 halfExtents.xyz/friction.w
// contact-record columns (SoA, `pairContacts[col*recordCap + rec]`):
//   0 meta(type,a,b,feature)  1 normal  2 rA  3 rB  4 c0  5 penalty.xyz/friction.w  6 lambda

/** per-step constants — one uniform, written by the driver (the live count is GPU-resident, in `eids[0]`) */
export interface StepParams {
    dt: number;
    gravity: number;
    alpha: number;
    /** the fresh-contact penalty seed: PENALTY_MIN — it ramps via `betaLin` (manifold.ts) */
    penalty: number;
    /** penalty-ramp rate (Eq. 17) */
    betaLin: number;
    /** warmstart decay (Eq. 19): λ ← α·γ·λ_prev, k ← clamp(γ·k_prev) */
    gamma: number;
    /** the joint angular penalty-ramp rate (Phase 6.2 — joint.ts `betaAng`); contacts/springs ignore it. Default 100. */
    betaAng?: number;
    iterations: number;
    /**
     * the dispatched-color cap for the incremental-greedy coloring (`colorize`): the 32-wide usedMask
     * is separate from this fold cap (scratch.md "AVBD rebuild" — two different numbers). A body that
     * can't find a free color within the cap folds to `bid % maxColors`, degrading that pair to Jacobi
     * for the step (a soft-contact conflict the iterative primal tolerates). Default 32 (no fold).
     */
    maxColors?: number;
    /**
     * the small-N regime threshold (C1.0): at a live body count ≤ this, `record` replaces the BVH
     * build + descent with the one-dispatch O(n²) broadphase (identical pair blocks, ~29 fewer
     * dependent phases — the gameplay-regime structure tax). 0 forces the BVH path at every count
     * (the A/B lever for the crossover sweep). Default {@link SMALL_N}.
     */
    smallN?: number;
    /**
     * the LDS-resident solve threshold (C1.2): at a live body count ≤ this, `record` replaces the
     * whole iters × colors primal/commit/dual block with ONE single-workgroup dispatch holding every
     * live body's pose in workgroup memory across the loop — each color phase's dependent round trip
     * becomes an in-kernel barrier on LDS, not a storage round trip. 0 forces the looped color passes
     * at every count (the A/B lever). Clamped to {@link LDS_CAP} (the workgroup-memory capacity).
     * Default {@link LDS_N}.
     */
    ldsN?: number;
    /**
     * sub-steps per fixed step (the "small steps" form, Macklin 2019): `record` runs `substeps`
     * complete AVBD sub-steps of `h = dt/substeps`, each a full broadphase → collide → solve → velocity
     * pass against the persistent warmstart store. Smaller per-step motion keeps the penalty ramp
     * (`k += betaLin·|C|`) bounded, the convergence lever a dense chaotic pile needs (raising `iterations`
     * saturates; the f64 oracle's `substeps` clears it — roadmap "dense-pile contact convergence"). `1`
     * is byte-identical to the single-sub-step path, so every gate is unchanged. Default 1.
     */
    substeps?: number;
}

/** the default small-N regime threshold — the live count at or under which `record` runs the
 * one-dispatch O(n²) broadphase rather than the BVH build + descent (the gameplay regime is
 * structure-tax-bound, not work-bound; crossover measured by the physics bench sweep) */
export const SMALL_N = 1024;

/** the LDS-resident solve capacity — what fits the 16 KB workgroup-memory floor
 * (maxComputeWorkgroupStorageSize 16384): pos (3 split f32 columns, 12 B) + quat (vec4, 16 B) =
 * 28 B/body → 512 · 28 = 14336 B resident, headroom for the kernel's control vars. Sits below the
 * {@link SMALL_N} regime threshold by construction. The DEFAULT threshold is {@link LDS_N}, not this —
 * a `ldsN` sweep up to the capacity is the floor-device lever (the Metal cell). */
export const LDS_CAP = 512;

/** the default LDS-resident solve threshold — the measured Lovelace-neutral point:
 * the single-WG kernel is parity at ≤~64 bodies and loses ~linearly above (~+19% at 130, +39% at 502 —
 * its serialized record/CSR latency on one SM outgrows the looped path's dispatch boundaries), so the
 * default engages only where it costs nothing and the gym gates keep it green. On Metal the boundary
 * constant is 4× Lovelace (~3.5 µs/phase recoverable in-kernel), so the C1.4 Apple cell decides
 * raising it toward {@link LDS_CAP}. */
export const LDS_N = 64;

// Math + the Step uniform struct, no buffer refs — prepended to every pass before its bindings.
const SHARED_WGSL = /* wgsl */ `
struct Step {
    recordCap: u32, iterations: u32, eidCap: u32, maxColors: u32,
    dt: f32, gravity: f32, alpha: f32, penalty: f32,
    invDt2: f32, betaLin: f32, gamma: f32, betaAng: f32,
    jointCount: u32, substeps: u32, _pad2: u32, _pad3: u32,
};
const KIND_SPRING: u32 = ${KIND_SPRING}u;
const KIND_JOINT: u32 = ${KIND_JOINT}u;
const CONSTRAINT_VEC4: u32 = ${CONSTRAINT_VEC4}u;
const JOINT_REC_VEC4: u32 = ${JOINT_REC_VEC4}u;
const RIGID_THRESHOLD: f32 = 1.0e29;
// a joint endpoint of WORLD_ANCHOR is the WORLD (no body): its rA is a world-space point, its orientation is
// identity, its mass/size are 0 (static). The grab pins a box to a world cursor point with no anchor body —
// hence no anchor↔box contact (joint.cpp bodyA == null). The CPU side gives it no constraint-list entry.
const WORLD_ANCHOR: u32 = 0xffffffffu;
// PAIRS_PER_BODY = the per-eid FIXED block size: body eid owns slots [eid*PAIRS_PER_BODY, …) in pairList.
// CONTACTS_PER_PAIR contact records per pair slot: a pair at slot s (= owner eid*PAIRS_PER_BODY + k) keeps
// its manifold at recordBase = s*CONTACTS_PER_PAIR in the persistent pairContacts SoA. The slot is a function
// of the owner's eid alone, so it's stable across frames unless the owner's own candidate set flickers
// (local warmstart fragility — Phase 4.9 robustness, scratch.md), not the global churn a prefix-sum had.
const PAIRS_PER_BODY: u32 = ${PAIRS_PER_BODY}u;
const CONTACTS_PER_PAIR: u32 = ${CONTACTS_PER_PAIR}u;

const COLLISION_MARGIN = 0.01;
const PENALTY_MIN: f32 = ${PENALTY_MIN.toFixed(1)};
const PENALTY_MAX: f32 = 1.0e10;

fn qConjW(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.xyz, q.w); }
fn qMulW(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}
fn qInvW(q: vec4<f32>) -> vec4<f32> { return qConjW(q) / dot(q, q); }
fn qSubW(a: vec4<f32>, b: vec4<f32>) -> vec3<f32> { return qMulW(a, qInvW(b)).xyz * 2.0; }
fn qRotateW(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
// quat + omega vector -> integrated, renormalized quat (math.ts qadd)
fn qAddW(a: vec4<f32>, w: vec3<f32>) -> vec4<f32> {
    let d = 0.5 * qMulW(vec4<f32>(w, 0.0), a);
    return normalize(a + d);
}
`;

// Body column indices (SoA `bodies[col*step.bodyCap + i]`). The consts alone are enough to write a
// column; the readers below add the typed getters. Split so the compaction pass can template just the
// consts without pulling in the readers (which reference the per-pass `bodies` binding).
const BODY_COLS_WGSL = /* wgsl */ `
const B_POS: u32 = 0u;
const B_QUAT: u32 = 1u;
const B_INERTL: u32 = 2u;
const B_INERTQ: u32 = 3u;
const B_INITL: u32 = 4u;
const B_INITQ: u32 = 5u;
const B_VELL: u32 = 6u;
const B_VELA: u32 = 7u;
const B_PREVV: u32 = 8u;
const B_MM: u32 = 9u;
const B_HF: u32 = 10u;
const B_ROUND: u32 = 11u;
`;

// Body-state accessors — appended AFTER a pass declares the `bodies` binding + the `step` uniform
// (the binding's access mode differs per pass: `read` in collide/compose, `read_write` elsewhere;
// these readers suit both). `bCol` is the one SoA index site: a warp reading sequential `i` per
// column coalesces to one cache line (gpu.md cols-buffer pattern). Total bindings unchanged from AoS.
// Everything below bPos/bQuat is loop-constant during the solve, so it always reads storage. bPos/bQuat
// are split out so the LDS-resident solve kernel (SOLVE_LDS_WGSL) can swap in workgroup-memory readers —
// the pose is the per-color dependent chain, the one thing that must not round-trip storage there.
const BODY_REST_WGSL = /* wgsl */ `
fn bCol(col: u32, i: u32) -> vec4<f32> { return bodies[col*step.eidCap + i]; }
fn bInertL(i: u32) -> vec3<f32>   { return bCol(B_INERTL, i).xyz; }
fn bInertQ(i: u32) -> vec4<f32>   { return bCol(B_INERTQ, i); }
fn bInitL(i: u32) -> vec3<f32>    { return bCol(B_INITL, i).xyz; }
fn bInitQ(i: u32) -> vec4<f32>    { return bCol(B_INITQ, i); }
fn bVelL(i: u32) -> vec3<f32>     { return bCol(B_VELL, i).xyz; }
fn bVelA(i: u32) -> vec3<f32>     { return bCol(B_VELA, i).xyz; }
fn bPrevV(i: u32) -> vec3<f32>    { return bCol(B_PREVV, i).xyz; }
fn bMass(i: u32) -> f32           { return bCol(B_MM, i).w; }
fn bHalf(i: u32) -> vec3<f32>     { return bCol(B_HF, i).xyz; }
fn bFriction(i: u32) -> f32       { return bCol(B_HF, i).w; }
// B_ROUND packs the shape tag (x, a bitcast ShapeKind) + the rounding radius (y, sphere/capsule) + the
// hull id (z, a bitcast registry index for ShapeKind.Hull). Read together with bHalf (the core extents) by
// the narrowphase + compose. A box has kind 0 + radius 0 + id 0, so a fresh box's bRound(i) is all-zero and
// the box path stays bit-identical.
fn bShape(i: u32) -> u32          { return bitcast<u32>(bCol(B_ROUND, i).x); }
fn bRadius(i: u32) -> f32         { return bCol(B_ROUND, i).y; }
fn bHullId(i: u32) -> u32         { return bitcast<u32>(bCol(B_ROUND, i).z); }
// solverStatic is the static predicate the solver checks everywhere — a real static / kinematic body
// (mass ≤ 0), skipped in the primal + velocity passes and the dual/joint all-static gate.
fn solverStatic(i: u32) -> bool   { return bMass(i) <= 0.0; }
`;

const BODY_WGSL =
    BODY_COLS_WGSL +
    /* wgsl */ `
fn bPos(i: u32) -> vec3<f32>      { return bCol(B_POS, i).xyz; }
fn bQuat(i: u32) -> vec4<f32>     { return bCol(B_QUAT, i); }
` +
    BODY_REST_WGSL;

// Contact column indices + the SoA reader (`pairContacts[col*step.recordCap + rec]`), appended after a
// pass declares the `pairContacts` binding + the `step` uniform. `cc` reads a read or read_write binding.
// A record's C_META is (type, a, b, feature): type 0 = inactive (cleared slot), CONSTRAINT_CONTACT = live;
// a/b are the body eids, so a record is self-describing — the collide checks (a,b) for the warmstart
// pair-identity match (no hash, no probe — the slot itself is the key, Phase 4.7).
const CONTACT_WGSL = /* wgsl */ `
const C_META: u32 = 0u;
const C_NORMAL: u32 = 1u;
const C_RA: u32 = 2u;
const C_RB: u32 = 3u;
const C_C0: u32 = 4u;
const C_PEN: u32 = 5u;
const C_LAMBDA: u32 = 6u;
fn cc(rec: u32, col: u32) -> vec4<f32> { return pairContacts[col*step.recordCap + rec]; }
`;

// the oriented-box world-AABB half-extent (|R|·h, R = the body's rotation). Tighter than the sphere bound
// |h|, so a settled pile overlaps only its real neighbors (keeps the per-body pair block small). Still a
// valid broadphase superset: two boxes that touch have overlapping box-AABBs, so the narrowphase (sphere
// test + SAT) never loses a contact the oracle keeps. Shared by the aabb prim + the broadphase query.
const BOX_EXTENT_WGSL = /* wgsl */ `
// the speculative band (Phase 4.8.3, collide.ts SPECULATIVE_DISTANCE): the aabb pass pads each broadphase
// prim by this so a pair within the band is found before contact. Mirrors the f64 oracle + C++ broadphase.
const SPECULATIVE_DISTANCE: f32 = ${SPECULATIVE_DISTANCE};
// world-AABB half-extent of the body's oriented CORE box (|R|·h) inflated by the rounding radius — a
// capsule/sphere's core extents are 0 along the round axes, so the radius is what bounds them (Phase 6.3).
// A box (radius 0) is unchanged. Shared by the aabb prim + the broadphase query, so both stay a valid superset.
fn boxExtent(i: u32) -> vec3<f32> {
    let q = bQuat(i);
    let h = bHalf(i);
    let ax0 = abs(qRotateW(q, vec3<f32>(1.0, 0.0, 0.0)));
    let ax1 = abs(qRotateW(q, vec3<f32>(0.0, 1.0, 0.0)));
    let ax2 = abs(qRotateW(q, vec3<f32>(0.0, 0.0, 1.0)));
    return ax0 * h.x + ax1 * h.y + ax2 * h.z + vec3<f32>(bRadius(i));
}
`;

// ── aabb: dense body oriented-box AABB → the bvh prim buffer (the broadphase BVH input) ──
// Each prim is the body's world-AABB [pos − e, pos + e], e = the oriented-box extent (boxExtent). 2
// vec4/prim (min.xyz+pad, max.xyz+pad — bvh/core prim layout). Only [0, count) is written; the build
// sentinel-pads the tail.
const AABB_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> prims: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> step: Step;
@group(0) @binding(3) var<storage, read> eids: array<u32>;
${BODY_WGSL}
${BOX_EXTENT_WGSL}
// one thread per dense slot d in [0, count); prim index = d, body = the eid at eids[1+d]. So prim d
// is body eids[1+d]'s box-AABB, and the broadphase maps a leaf's prim index back through eids.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let i = eids[1u + d];
    let p = bPos(i);
    // pad the prim by the speculative band (static skin) + the per-axis velocity sweep |vel|·dt (Phase
    // 4.8.4, the webphysics velocity-fattened-tree form) so the broadphase finds a fast approaching pair
    // before contact. The static skin is prim-only (the query box stays tight on it → combined static slack
    // = SPECULATIVE_DISTANCE); the velocity pad is on both prim + query → combined ≈ (|vA|+|vB|)·dt ≥ |vRel|·dt.
    let e = boxExtent(i) + vec3<f32>(SPECULATIVE_DISTANCE) + abs(bVelL(i)) * step.dt;
    prims[2u*d] = vec4<f32>(p - e, 0.0);
    prims[2u*d + 1u] = vec4<f32>(p + e, 0.0);
}
`;

// ── broadphase per-candidate accumulate + block emit (shared by the descent + the small-N scan) ──
// Interpolated into both broadphase mains, so the ownership rule, the nearest-K + static-pin prune, the
// sort, and the block write are one source of truth — the small-N O(n²) scan differs ONLY in how it
// enumerates candidates, the precondition for warmstart carrying across a regime flip (identical blocks).
// Operates on the mains' locals: d (own dense slot), pi (own center), nbr/nd2/count (the accumulator);
// consumes `dj` (the candidate's dense slot, != d).
const BROADPHASE_CANDIDATE_WGSL = /* wgsl */ `
            let je = eids[1u + dj];
            let staticJ = bMass(je) <= 0.0;
            // d (dynamic) owns: every dyn-static pair, and a dyn-dyn pair only when dj has the lower
            // slot (so the higher-slot body owns it — emitted once).
            if (staticJ || dj < d) {
                let dc = pi - bPos(je);
                var d2 = dot(dc, dc);
                if (staticJ) { d2 = -1.0; }         // pin: a big static's center is far, but it must never be evicted
                if (count < PAIRS_PER_BODY) {
                    nbr[count] = dj; nd2[count] = d2; count = count + 1u;
                } else {
                    atomicAdd(&counters[3], 1u);    // loud: a candidate exceeded the cap (now a graceful nearest-K drop)
                    // farthest kept candidate (tie → higher slot, deterministic); replace only if the
                    // newcomer is nearer, so the dropped one is always the least-important (farthest).
                    var wi = 0u; var wd2 = nd2[0]; var wdj = nbr[0];
                    for (var w = 1u; w < PAIRS_PER_BODY; w = w + 1u) {
                        if (nd2[w] > wd2 || (nd2[w] == wd2 && nbr[w] > wdj)) { wi = w; wd2 = nd2[w]; wdj = nbr[w]; }
                    }
                    if (d2 < wd2 || (d2 == wd2 && dj < wdj)) {
                        if (bMass(eids[1u + wdj]) <= 0.0) { atomicAdd(&counters[7], 1u); } // evicted a static — only if > K statics (the pin keeps this 0 with one ground)
                        nbr[wi] = dj; nd2[wi] = d2;
                    } else if (staticJ) {
                        atomicAdd(&counters[7], 1u); // a static newcomer was dropped — same > K-statics edge case
                    }
                }
            }
`;
const BROADPHASE_EMIT_WGSL = /* wgsl */ `
    // insertion-sort the neighbor slots ascending (stable order ⇒ stable per-pair slot across frames)
    for (var s = 1u; s < count; s = s + 1u) {
        let key = nbr[s];
        var k = s;
        loop {
            if (k == 0u) { break; }
            if (nbr[k - 1u] <= key) { break; }
            nbr[k] = nbr[k - 1u];
            k = k - 1u;
        }
        nbr[k] = key;
    }

    // write the per-eid block: [0, count) the owned pairs (oriented a > b by eid), [count, PAIRS_PER_BODY)
    // cleared to INVALID so a stale record (a prior frame / a recycled eid) is never read as a live pair.
    for (var k = 0u; k < PAIRS_PER_BODY; k = k + 1u) {
        if (k < count) {
            let je = eids[1u + nbr[k]];
            pairList[blockBase + k] = vec2<u32>(max(i, je), min(i, je));
        } else {
            pairList[blockBase + k] = vec2<u32>(INVALID_PAIR);
        }
    }
`;

// ── broadphase: LBVH box-overlap descent → each live body's per-eid FIXED pair block ──
// One thread per dense body d (the block owner) descends the BVH built over the sphere-AABBs and writes
// d's overlapping neighbors into ITS OWN fixed block `pairList[eid·PAIRS_PER_BODY …]` (a cheap `vec2<u32>`
// pair, NOT the manifold), the unused slots cleared to INVALID. The block is insertion-SORTED by the
// neighbor's dense slot (deterministic), so each pair lands at a deterministic slot in the owner eid's block
// — stable across frames unless the owner's candidate set flickers, the precondition the in-place warmstart
// needs (no hash, no prefix-sum coupling — Phase 4.9 robustness). Each pair is owned by exactly one block: a
// STATIC body owns none (its block all INVALID — the dynamic partner owns every dyn-static pair, so the
// ground never owns a huge block), a dyn-dyn pair by the higher-dense-slot (= higher-eid) body. The block
// stores the pair oriented a > b by eid (bodyA = higher creation index, the reference orientation).
// A block past PAIRS_PER_BODY keeps the NEAREST by center-dist² + pins statics (importance prune, below)
// + bumps the loud counter (counters[3]) — a graceful drop of the farthest, not an arbitrary traversal drop.
// The SAT (a large fn) stays OUT of this descent (gpu.md "never call large functions inside dynamic loops")
// — the narrowphase SATs the per-eid slots. Stack depth is the derived LBVH bound (≤62, 64 covers it).
const BROADPHASE_PASS_WGSL =
    SHARED_WGSL +
    BVH_ROOT_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pairList: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> step: Step;
@group(0) @binding(5) var<storage, read> eids: array<u32>;
const INVALID_PAIR: u32 = 0xffffffffu; // an unused per-eid block slot (the collide/dual/CSR skip it)
${BODY_WGSL}
${BOX_EXTENT_WGSL}
fn nodeLeft(n: u32) -> u32 { return bitcast<u32>(nodes[2u*n].w); }
fn nodeRight(n: u32) -> u32 { return bitcast<u32>(nodes[2u*n + 1u].w); }
fn nodeMin(n: u32) -> vec3<f32> { return nodes[2u*n].xyz; }
fn nodeMax(n: u32) -> vec3<f32> { return nodes[2u*n + 1u].xyz; }
fn aabbOverlap(qmin: vec3<f32>, qmax: vec3<f32>, n: u32) -> bool {
    return all(qmin <= nodeMax(n)) && all(qmax >= nodeMin(n));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let i = eids[1u + d];                                // i = this body's eid
    let blockBase = i * PAIRS_PER_BODY;                  // the owner-EID fixed block — the stable warmstart slot
    // static bodies own no pair — the dynamic partner owns every dyn-static pair (so a ground touching N
    // boxes never owns a huge block). Clear the block to INVALID + return.
    if (bMass(i) <= 0.0) {
        for (var k = 0u; k < PAIRS_PER_BODY; k = k + 1u) { pairList[blockBase + k] = vec2<u32>(INVALID_PAIR); }
        return;
    }
    let pi = bPos(i);
    // the query box is the tight box-extent + the velocity sweep |vel|·dt (Phase 4.8.4); the static
    // SPECULATIVE_DISTANCE skin lives on the prim only (the aabb pass), so combined static slack stays
    // SPECULATIVE_DISTANCE while the velocity slack ≈ (|vA|+|vB|)·dt covers the relative closing.
    let ei = boxExtent(i) + abs(bVelL(i)) * step.dt;
    let qmin = pi - ei;
    let qmax = pi + ei;

    // collect d's owned neighbors as the K NEAREST by center-dist² (importance prune, webphysics
    // broadPhase.ts ~560-694): when a tight pile gives a body > PAIRS_PER_BODY band-neighbors, keep the
    // nearest (the ones that actually become contacts) and drop the farthest, NOT an arbitrary
    // BVH-traversal-order drop — that drop can evict a real support, the root of the dense-pile churn AND
    // the static-ground fall-through (4.8.6's "the floor is never dropped" was disproven by a tall dense
    // pile). Static supports are pinned (d2 = −1, never the farthest) so the ground is never evicted.
    // Sorted ascending below for a stable per-pair slot.
    var nbr: array<u32, ${PAIRS_PER_BODY}>;
    var nd2: array<f32, ${PAIRS_PER_BODY}>;       // center-dist² per kept neighbor (static = −1, pinned)
    var count = 0u;

    var stack: array<u32, 64>;
    var sp = 0u;
    var node = bvhRoot(eids[0u]);
    loop {
        if (nodeLeft(node) == 0xffffffffu) {            // leaf — overlap confirmed by the descent above
            let dj = nodeRight(node);                   // neighbor's dense slot (prim index)
            if (dj != d) {
${BROADPHASE_CANDIDATE_WGSL}
            }
            if (sp == 0u) { break; }
            sp -= 1u; node = stack[sp]; continue;
        }
        let l = nodeLeft(node);
        let r = nodeRight(node);
        let okL = aabbOverlap(qmin, qmax, l);
        let okR = aabbOverlap(qmin, qmax, r);
        if (okL && okR) {
            if (sp < 64u) { stack[sp] = r; sp += 1u; }
            node = l;
        } else if (okL) { node = l; }
        else if (okR) { node = r; }
        else { if (sp == 0u) { break; } sp -= 1u; node = stack[sp]; }
    }

${BROADPHASE_EMIT_WGSL}
}
`;

// ── broadphase (small-N regime, C1.0): one-dispatch O(n²) scan → the same per-eid FIXED pair blocks ──
// At a live count ≤ the smallN threshold, record() replaces the whole BVH build + descent (~29 dependent
// phases of near-pure structure tax at gameplay counts) with this single
// dispatch: each live body's lane scans the dense live set against the SAME aabb-pass prims the BVH leaves
// carry, so the overlap test, pads, ownership, prune, and block write are identical to the descent and the
// pair blocks come out byte-identical — warmstart carries across a regime flip. O(n²) is exact at any N
// (only slow past the threshold), so the frame-stale regime switch is correctness-safe in both directions.
const BROADPHASE_SMALL_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> prims: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pairList: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> step: Step;
@group(0) @binding(5) var<storage, read> eids: array<u32>;
const INVALID_PAIR: u32 = 0xffffffffu; // an unused per-eid block slot (the collide/dual/CSR skip it)
const SPECULATIVE_DISTANCE: f32 = ${SPECULATIVE_DISTANCE};
const TILE: u32 = 64u;
${BODY_WGSL}
// the n-body LDS tiling (Bullet 3 / GPU-gems n² pattern): each round the workgroup cooperatively stages
// TILE prims into workgroup memory, then every lane tests its own query box against the staged tile — a
// naive per-lane serial scan of global prims is latency-bound (measured 0.2–0.4 ms at 1k, worse than the
// BVH front-end it replaces), the staged form reads each prim from global memory once per workgroup.
var<workgroup> tMin: array<vec4<f32>, TILE>;
var<workgroup> tMax: array<vec4<f32>, TILE>;
var<workgroup> wgN: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // every lane must reach the tile barriers, so out-of-range / static lanes stay in the loop as
    // inactive rather than returning. workgroupUniformLoad makes the live count uniform for Tint's
    // uniformity analysis (eids[0] is workgroup-uniform in fact, but a raw storage read can't prove it).
    if (lid.x == 0u) { wgN = eids[0u]; }
    let n = workgroupUniformLoad(&wgN);
    let d = gid.x;
    let inRange = d < n;
    let i = select(0u, eids[1u + min(d, n - 1u)], inRange); // i = this body's eid (0 placeholder when idle)
    let blockBase = i * PAIRS_PER_BODY;                  // the owner-EID fixed block — the stable warmstart slot
    // static bodies own no pair — the dynamic partner owns every dyn-static pair (so a ground touching N
    // boxes never owns a huge block); their block is cleared to INVALID by the emit below (count stays 0).
    let act = inRange && bMass(i) > 0.0;
    let pi = bPos(i);
    // own query box = own prim shrunk by the static skin: the prim is pos ± (boxExtent + SPECULATIVE_DISTANCE
    // + |vel|·dt), and the skin is prim-only (the descent's query stays tight on it), so shrinking recovers
    // exactly the descent's query box — combined static slack stays SPECULATIVE_DISTANCE, velocity slack
    // ≈ (|vA|+|vB|)·dt. A candidate's prim is what the BVH leaf bounds were, so the test below is the
    // descent's leaf test verbatim.
    let dq = select(0u, d, inRange); // idle tail lanes read prim 0 (unused) — keeps the access in bounds
    let qmin = prims[2u*dq].xyz + vec3<f32>(SPECULATIVE_DISTANCE);
    let qmax = prims[2u*dq + 1u].xyz - vec3<f32>(SPECULATIVE_DISTANCE);

    var nbr: array<u32, ${PAIRS_PER_BODY}>;
    var nd2: array<f32, ${PAIRS_PER_BODY}>;       // center-dist² per kept neighbor (static = −1, pinned)
    var count = 0u;

    for (var base = 0u; base < n; base = base + TILE) {
        let src = base + lid.x;
        if (src < n) {
            tMin[lid.x] = prims[2u*src];
            tMax[lid.x] = prims[2u*src + 1u];
        }
        workgroupBarrier();
        let len = min(TILE, n - base);
        if (act) {
            for (var t = 0u; t < len; t = t + 1u) {
                let dj = base + t;
                if (dj == d) { continue; }
                if (!(all(qmin <= tMax[t].xyz) && all(qmax >= tMin[t].xyz))) { continue; }
${BROADPHASE_CANDIDATE_WGSL}
            }
        }
        workgroupBarrier();
    }
    if (!inRange) { return; }
${BROADPHASE_EMIT_WGSL}
}
`;

// ── narrowphase (collide): box-box SAT over the per-eid pair blocks; in-place warmstart ──
// One thread per pair SLOT in the live bodies' fixed blocks: the dispatch is `liveCount · PAIRS_PER_BODY`
// lanes, lane → (d = lane/K, k = lane%K), owner eid = eids[1+d], slot = eid·K + k (the fixed per-eid block
// base — webphysics `bodyBase`). The slot's manifold lives at recBase = slot*CONTACTS_PER_PAIR in
// `pairContacts`, persistent across frames; because the base is the owner's eid (not a prefix-sum offset),
// the slot is STABLE unless the owner's own candidate set flickers — local warmstart fragility, not the
// global collapse a compaction has (scratch.md "Phase 4.9 robustness"). Per slot: read pairList[slot]; an
// INVALID (unused) slot or a separating / 0-contact pair clears the block (kind 0, so the solve skips it +
// it cold-starts
// next frame, matching the oracle dropping a 0-contact manifold). Otherwise SAT, then for each fresh contact
// scan THIS SLOT's prev records (read before any write — loop 1) for a record with the SAME pair (a,b) +
// feature key, and carry its λ/k decayed (Eq. 19) + sticking arms (manifold.ts initManifold). The (a,b)
// identity gate is what replaces the hash: a slot reused by a different pair fails it → cold-start, no probe,
// no separate store. Loop 2 then overwrites the slot's records in place. The sphere test (dot(dp,dp) <= r²,
// r = radiusA + radiusB) filters the AABB-overlap superset back to the exact reference contact set.
// Built as FOUR pipelines by shape-pair class — box×box, rounded×rounded, polytope×polytope (collideHull),
// and rounded×polytope (collideRoundedPolytope) — each compiling only its own SAT chunk (HELPERS_WGSL +
// BOXBOX_WGSL | ROUNDED_WGSL | HULL_CORE+HULL_SAT | HULL_CORE+ROUNDED_POLY). DXC compile is superlinear in
// kernel size, so the monolithic kernel paid the full cost on Chrome/Windows; the hull SAT and the rounded
// segment-clip — the two halves of the old combined ~920 ms long pole — now compile apart (gpu.md "DXC
// shader compilation": dead code isn't free, optimize via pipeline splits). All four build async-parallel
// (one Promise.all), so the wall-clock compile ≈ the largest single chunk, not the sum. All four dispatch
// over the SAME live pair slots (indirect off pairArgs) and act only on their class; the BOX pipeline OWNS
// slot lifecycle — it clears every INVALID + separated slot regardless of class (the sphere filter is
// shape-aware), so the other three only fill/clear their OWN class's live slots and every slot is written
// exactly once (the gates are mutually exclusive + exhaustive). The cost is 3 extra cheap early-out
// dispatches per step (collide runs once per fixed step, not per iteration), traded for not needing a
// per-class partition pass. `ownsLifecycle` injects the dead-slot policy; `gate`/`call` inject the class
// check + the SAT call.
const collidePass = (chunk: string, gate: string, call: string, ownsLifecycle: boolean): string => {
    const dead = ownsLifecycle ? "clearBlock(recBase); return;" : "return;";
    return (
        SHARED_WGSL +
        HELPERS_WGSL +
        chunk +
        /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> pairContacts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<storage, read> pairList: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> eids: array<u32>; // [0] = live count, [1+d] = the d-th live eid
@group(0) @binding(6) var<storage, read> hullData: array<u32>; // packed convex-hull geometry (./hull packHulls)
${BODY_WGSL}
${CONTACT_WGSL}
// clear a slot's whole manifold block to inactive (kind 0) — the solve + warmstart skip a 0-meta record.
fn clearBlock(recBase: u32) {
    let rc = step.recordCap;
    for (var s = 0u; s < CONTACTS_PER_PAIR; s = s + 1u) { pairContacts[C_META*rc + recBase + s] = vec4<f32>(0.0); }
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    let d = lane / PAIRS_PER_BODY;
    if (d >= eids[0u]) { return; } // past the live body count (the tail of the last workgroup)
    let slot = eids[1u + d] * PAIRS_PER_BODY + (lane % PAIRS_PER_BODY); // the owner-eid fixed-block slot
    let recBase = slot * CONTACTS_PER_PAIR;
    let pair = pairList[slot];
    if (pair.x == 0xffffffffu) { ${dead} } // INVALID — unused block slot (box pipeline keeps it inactive)
    let ia = pair.x;
    let ib = pair.y;

    let pa = bPos(ia); let pb = bPos(ib);
    // the bounding-sphere radius is shape-aware: core span + the rounding radius (Phase 6.3). A box's
    // radius is 0, so this is length(bHalf) unchanged.
    let ra = length(bHalf(ia)) + bRadius(ia); let rb = length(bHalf(ib)) + bRadius(ib);
    let dp = pa - pb;
    // dRel = (vA−vB)·dt — the velocity sweep (Phase 4.8.4): widens the sphere filter + the SAT band so a
    // fast approaching pair reaches the SAT and generates its swept contact (mirrors the C++ + oracle).
    let dRel = (bVelL(ia) - bVelL(ib)) * step.dt;
    // the sphere filter is the reference broadphase (mirrors the C++ + oracle); the static band (Phase
    // 4.8.3) + the relative sweep |vRel|·dt (4.8.4) let a pair within reach this step pass to the SAT.
    let r = ra + rb + SPECULATIVE_DISTANCE + length(dRel);
    if (dot(dp, dp) > r * r) { ${dead} } // separated past the band → cold next frame (box pipeline clears it)

    let qa = bQuat(ia); let qb = bQuat(ib);
    let sa = bShape(ia); let sb = bShape(ib);
    let roundedA = (sa == 1u || sa == 2u);
    let roundedB = (sb == 1u || sb == 2u);
    // class gate: this pipeline handles only its shape-pair class; another pipeline owns the rest (the box
    // pipeline owns dead-slot clearing), so each live slot is written exactly once. The gate returns early
    // for a pair this pipeline doesn't own; the call then runs the matching SAT (the oracle narrowphase
    // matrix, tests/avbd/rounded.ts). A/B-oriented so passing (ia, ib) returns ia-as-A.
    ${gate}
    var sat: SatResult;
    ${call}
    if (sat.count == 0u) { clearBlock(recBase); return; }

    let friction = sqrt(bFriction(ia) * bFriction(ib));
    let rc = step.recordCap;

    // loop 1: per fresh contact, scan THIS slot's prev records (intact — no write yet) for the same pair
    // (a,b) + feature key, carry λ/k decayed + sticking arms. Carried into per-contact locals so loop 2 can
    // overwrite the slot. Pair-identity gate (a==ia && b==ib) is the hash replacement.
    var cLam: array<vec3<f32>, ${CONTACTS_PER_PAIR}>;
    var cPen: array<vec3<f32>, ${CONTACTS_PER_PAIR}>;
    var cRA: array<vec3<f32>, ${CONTACTS_PER_PAIR}>;
    var cRB: array<vec3<f32>, ${CONTACTS_PER_PAIR}>;
    var merged = 0u;
    for (var k = 0u; k < sat.count; k = k + 1u) {
        let feat = sat.feat[k];
        var lam3 = vec3<f32>(0.0);
        var pen3 = vec3<f32>(step.penalty);
        var rA = sat.rA[k];
        var rB = sat.rB[k];
        for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
            let wm = cc(recBase + ls, C_META);
            if (bitcast<u32>(wm.x) == ${CONSTRAINT_CONTACT}u
                && bitcast<u32>(wm.y) == ia && bitcast<u32>(wm.z) == ib
                && bitcast<u32>(wm.w) == feat) {
                let oldLam = cc(recBase + ls, C_LAMBDA);
                lam3 = oldLam.xyz * (step.alpha * step.gamma);
                pen3 = clamp(cc(recBase + ls, C_PEN).xyz * step.gamma, vec3<f32>(PENALTY_MIN), vec3<f32>(PENALTY_MAX));
                // a sticking contact keeps its frozen arms ONLY for box-box pairs, where the feature key
                // identifies a persistent vertex/edge. Any pair INVOLVING a rounded (sphere/capsule) shape —
                // even vs a box — has a sliding closest point under a constant feature key, so freezing its
                // arm anchors a stale point: any spin rotates the frozen arm → a tangential c0 → torque →
                // runaway spin (Phase 6.3). A rounded shape re-collides fresh arms vs a box too.
                if (oldLam.w > 0.5 && !(roundedA || roundedB)) { rA = cc(recBase + ls, C_RA).xyz; rB = cc(recBase + ls, C_RB).xyz; }
                merged = merged + 1u;
                break;
            }
        }
        cLam[k] = lam3; cPen[k] = pen3; cRA[k] = rA; cRB[k] = rB;
    }

    // loop 2: overwrite the slot's records in place — [0, sat.count) live, the rest inactive (kind 0)
    let n0 = sat.basis.r0;
    for (var k = 0u; k < CONTACTS_PER_PAIR; k = k + 1u) {
        let rec = recBase + k;
        if (k < sat.count) {
            let rA = cRA[k];
            let rB = cRB[k];
            // the arms anchor the CORE feature point; the contact surface is offset ±radius along the
            // normal (rounded narrowphase, Phase 6.3 — keeps the radius part geometric, off the spin).
            // Reconstruct the surface points for the true gap. A box has radius 0, so this is the bare arm.
            let xA = pa + qRotateW(qa, rA) - n0 * bRadius(ia);
            let xB = pb + qRotateW(qb, rB) + n0 * bRadius(ib);
            let dlt = xA - xB;
            let c0 = vec3<f32>(dot(sat.basis.r0, dlt) + COLLISION_MARGIN, dot(sat.basis.r1, dlt), dot(sat.basis.r2, dlt));
            pairContacts[C_META*rc + rec] = vec4<f32>(bitcast<f32>(${CONSTRAINT_CONTACT}u), bitcast<f32>(ia), bitcast<f32>(ib), bitcast<f32>(sat.feat[k]));
            pairContacts[C_NORMAL*rc + rec] = vec4<f32>(n0, 0.0);
            pairContacts[C_RA*rc + rec] = vec4<f32>(rA, 0.0);
            pairContacts[C_RB*rc + rec] = vec4<f32>(rB, 0.0);
            pairContacts[C_C0*rc + rec] = vec4<f32>(c0, 0.0);
            pairContacts[C_PEN*rc + rec] = vec4<f32>(cPen[k], friction);
            pairContacts[C_LAMBDA*rc + rec] = vec4<f32>(cLam[k], 0.0);
        } else {
            pairContacts[C_META*rc + rec] = vec4<f32>(0.0); // inactive
        }
    }
    atomicAdd(&counters[0], sat.count);          // total active contacts (the GPU correctness gates read this)
    if (merged > 0u) { atomicAdd(&counters[6], merged); } // warmstarted-contact count (exact-persistence gate)
}
`
    );
};

// box×box — the common case + the slot-lifecycle owner (clears INVALID + separated slots for every class).
const COLLIDE_BOX_WGSL = collidePass(
    BOXBOX_WGSL,
    "if (!(sa == 0u && sb == 0u)) { return; }",
    "sat = collideBoxBox(pa, qa, bHalf(ia) * 2.0, pb, qb, bHalf(ib) * 2.0, dRel);",
    true,
);
// rounded×rounded — sphere/capsule pairs (one segment-segment closest point).
const COLLIDE_ROUNDED_WGSL = collidePass(
    ROUNDED_WGSL,
    "if (!(roundedA && roundedB)) { return; }",
    "sat = collideRounded(pa, qa, bHalf(ia) * 2.0, bRadius(ia), pb, qb, bHalf(ib) * 2.0, bRadius(ib), dRel);",
    false,
);
// hull — box×hull, hull×hull (collideHull). The polytope×polytope SAT, its own pipeline (the 4-way split,
// gpu.md "DXC shader compilation"): collideHull's big face/edge SAT compiles apart from the rounded segment-
// clip below, so neither kernel pays the other's superlinear compile cost (the combined hull kernel was the
// standing ~920 ms long pole). Gate: both non-rounded, not both box ⇒ at least one hull.
const COLLIDE_HULL_WGSL = collidePass(
    HULL_CORE_WGSL + HULL_SAT_WGSL,
    "if (!(!roundedA && !roundedB && !(sa == 0u && sb == 0u))) { return; }",
    "sat = collideHull(polyMake(sa, pa, qa, bHalf(ia) * 2.0, bHullId(ia)), polyMake(sb, pb, qb, bHalf(ib) * 2.0, bHullId(ib)), dRel);",
    false,
);
// rounded×polytope — sphere/capsule vs box/hull (collideRoundedPolytope). The other half of the old hull
// kernel, its own pipeline. Gate: exactly one shape is rounded. Mutually exclusive with the box/rounded/hull
// gates, so every live slot is still written by exactly one pipeline.
const COLLIDE_ROUNDEDPOLY_WGSL = collidePass(
    HULL_CORE_WGSL + ROUNDED_POLY_WGSL,
    "if (roundedA == roundedB) { return; }",
    "sat = collideRoundedPolytope(pa, qa, bHalf(ia) * 2.0, bRadius(ia), sa, bHullId(ia), pb, qb, bHalf(ib) * 2.0, bRadius(ib), sb, bHullId(ib), dRel);",
    false,
);

// ── inertial: inertial target (Eq. 2) + adaptive warmstart reposition (solver.cpp step 3) ──
const INERTIAL_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> eids: array<u32>;
${BODY_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let i = eids[1u + d];
    let dt = step.dt;
    let g = step.gravity;
    let dynamic = !solverStatic(i);

    let pos = bPos(i);
    let quat = bQuat(i);
    let vel = bVelL(i);
    let velA = bVelA(i);
    let prevV = bPrevV(i);

    // a static / kinematic body (solverStatic): no gravity in the inertial target, no
    // warmstart reposition — frozen inertial = initial = current pose, so dq = 0 (oracle solver.ts).
    // inertial target: full gravity (Eq. 2). The warmstart-start pos reuses the same predicted
    // step, scaling only the gravity term by accelWeight; the angular warmstart equals inertialQ.
    let predicted = pos + vel * dt;
    let inertialQ = qAddW(quat, velA * dt);
    var inertialL = predicted;
    if (dynamic) { inertialL = inertialL + vec3<f32>(0.0, g * dt * dt, 0.0); }

    // adaptive accelWeight (VBD): scales the warmstart-start gravity term, not the inertial target
    let accel = (vel - prevV) / dt;
    var accelWeight = clamp((accel.y * sign(g)) / abs(g), 0.0, 1.0);
    if (!(accelWeight == accelWeight)) { accelWeight = 0.0; } // NaN -> 0

    let cap = step.eidCap;
    bodies[B_INERTL*cap + i] = vec4<f32>(inertialL, 0.0);
    bodies[B_INERTQ*cap + i] = inertialQ;
    bodies[B_INITL*cap + i] = vec4<f32>(pos, 0.0);  // initialLin = x⁻ (the contact constraint reads this as dq=0)
    bodies[B_INITQ*cap + i] = quat;                  // initialAng

    if (dynamic) {
        let warmPos = predicted + vec3<f32>(0.0, g * accelWeight * dt * dt, 0.0);
        bodies[B_POS*cap + i] = vec4<f32>(warmPos, 0.0);
        bodies[B_QUAT*cap + i] = inertialQ;
    }
}
`;

// Mat3 (row-major, matching the oracle math.ts) + the helpers the contact force law + the 6×6
// stamp need. Appended after a pass declares its bindings — pure, binding-free. The collide pass
// has its own Mat3 in HELPERS_WGSL; primal/dual concatenate this one instead (no narrowphase chunk).
const MAT3_WGSL = /* wgsl */ `
struct Mat3 { r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32> };
fn mZero() -> Mat3 { return Mat3(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0)); }
fn mDiag(v: vec3<f32>) -> Mat3 { return Mat3(vec3<f32>(v.x,0.0,0.0), vec3<f32>(0.0,v.y,0.0), vec3<f32>(0.0,0.0,v.z)); }
fn mAdd(a: Mat3, b: Mat3) -> Mat3 { return Mat3(a.r0 + b.r0, a.r1 + b.r1, a.r2 + b.r2); }
fn mNeg(m: Mat3) -> Mat3 { return Mat3(-m.r0, -m.r1, -m.r2); }
fn mScale(m: Mat3, s: f32) -> Mat3 { return Mat3(m.r0 * s, m.r1 * s, m.r2 * s); }
// outer product a ⊗ b (M[i][j] = a[i]·b[j]) — the single-row Hessian block a spring Jacobian stamps (maths.h outer)
fn outer3(a: vec3<f32>, b: vec3<f32>) -> Mat3 { return Mat3(b * a.x, b * a.y, b * a.z); }
fn mMulV(m: Mat3, v: vec3<f32>) -> vec3<f32> { return vec3<f32>(dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)); }
fn mT(m: Mat3) -> Mat3 { return Mat3(vec3<f32>(m.r0.x,m.r1.x,m.r2.x), vec3<f32>(m.r0.y,m.r1.y,m.r2.y), vec3<f32>(m.r0.z,m.r1.z,m.r2.z)); }
fn mMul(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3(a.r0.x*b.r0 + a.r0.y*b.r1 + a.r0.z*b.r2,
                a.r1.x*b.r0 + a.r1.y*b.r1 + a.r1.z*b.r2,
                a.r2.x*b.r0 + a.r2.y*b.r1 + a.r2.z*b.r2);
}
fn orthoBasis(n: vec3<f32>) -> Mat3 {
    var t1: vec3<f32>;
    if (abs(n.x) > abs(n.y)) { t1 = vec3<f32>(-n.z, 0.0, n.x); } else { t1 = vec3<f32>(0.0, n.z, -n.y); }
    t1 = normalize(t1);
    return Mat3(n, t1, cross(t1, n));
}
// the joint's angular Jacobian + geometric-stiffness terms (maths.h skew/diagonalize, joint.cpp
// geometricStiffnessBallSocket) — row-major, matching the oracle math.ts.
fn skew(r: vec3<f32>) -> Mat3 { return Mat3(vec3<f32>(0.0, -r.z, r.y), vec3<f32>(r.z, 0.0, -r.x), vec3<f32>(-r.y, r.x, 0.0)); }
// diag of each column's length (the joint's diagonal higher-order approximation)
fn diagonalize(m: Mat3) -> Mat3 {
    return mDiag(vec3<f32>(
        length(vec3<f32>(m.r0.x, m.r1.x, m.r2.x)),
        length(vec3<f32>(m.r0.y, m.r1.y, m.r2.y)),
        length(vec3<f32>(m.r0.z, m.r1.z, m.r2.z))));
}
// geometricStiffnessBallSocket(k, v): diag(-v[k]) with v added into column k (k literal at the call sites)
fn geomStiffness(k: u32, v: vec3<f32>) -> Mat3 {
    let d = -v[k];
    var c0 = vec3<f32>(d, 0.0, 0.0); var c1 = vec3<f32>(0.0, d, 0.0); var c2 = vec3<f32>(0.0, 0.0, d);
    if (k == 0u) { c0 = c0 + v; } else if (k == 1u) { c1 = c1 + v; } else { c2 = c2 + v; }
    return Mat3(vec3<f32>(c0.x, c1.x, c2.x), vec3<f32>(c0.y, c1.y, c2.y), vec3<f32>(c0.z, c1.z, c2.z));
}
`;

// One contact's constraint C, its four Jacobian blocks, the diagonal penalty k, and the
// cone-clamped force F — the shared core of the primal stamp and the dual update (manifold.ts
// `contactForce`, which the reference inlines in both updatePrimal and updateDual). Reads the
// contact row + the two bodies' poses/deltas; appended after MAT3_WGSL + BODY_WGSL + CONTACT_WGSL.
const CONTACT_FORCE_WGSL = /* wgsl */ `
struct CForce {
    constraint: vec3<f32>, force: vec3<f32>,
    jALin: Mat3, jBLin: Mat3, jAAng: Mat3, jBAng: Mat3, k: Mat3,
    // the *pre-clamp* friction magnitude + the cone bound — the dual ramp gate reads these (manifold.ts
    // contactForce / manifold.cpp:156,169). Gating on the post-clamp force.yz (always == bounds when
    // saturated) ramps a sliding contact's tangent penalty unboundedly, fading kinetic friction.
    frictionScale: f32, bounds: f32,
};
fn contactForce(ci: u32) -> CForce {
    let m0 = cc(ci, C_META);
    let a = bitcast<u32>(m0.y);
    let b = bitcast<u32>(m0.z);
    let basis = orthoBasis(cc(ci, C_NORMAL).xyz);
    let rA = cc(ci, C_RA).xyz;
    let rB = cc(ci, C_RB).xyz;
    let c0 = cc(ci, C_C0).xyz;
    let pen = cc(ci, C_PEN);
    let friction = pen.w;
    let lambda = cc(ci, C_LAMBDA).xyz;

    let aQuat = bQuat(a); let bQ = bQuat(b);
    let dALin = bPos(a) - bInitL(a);
    let dAAng = qSubW(aQuat, bInitQ(a));
    let dBLin = bPos(b) - bInitL(b);
    let dBAng = qSubW(bQ, bInitQ(b));

    // the arm anchors the CORE feature; apply the geometric ±radius·normal offset HERE (not in the stored
    // arm) so the radius part never rotates with the body's spin — a rounded contact's normal Jacobian
    // then stays cross(−r·n, n) = 0 (a sphere's normal force passes through its centre → no torque). A box
    // has radius 0, so rAW/rBW are the bare material arms, bit-identical to before. roadmap §6.3.
    let n = basis.r0;
    let rAW = qRotateW(aQuat, rA) - n * bRadius(a);
    let rBW = qRotateW(bQ, rB) + n * bRadius(b);
    let jALin = basis;
    let jBLin = mNeg(basis);
    let jAAng = Mat3(cross(rAW, basis.r0), cross(rAW, basis.r1), cross(rAW, basis.r2));
    let jBAng = Mat3(cross(rBW, jBLin.r0), cross(rBW, jBLin.r1), cross(rBW, jBLin.r2));
    let k = mDiag(pen.xyz);

    let constraint = c0 * (1.0 - step.alpha)
        + mMulV(jALin, dALin) + mMulV(jBLin, dBLin) + mMulV(jAAng, dAAng) + mMulV(jBAng, dBAng);
    // force = k·C + λ, clamped: normal repulsion-only, friction inside the Coulomb cone
    var force = mMulV(k, constraint) + lambda;
    force.x = min(force.x, 0.0);
    let bounds = abs(force.x) * friction;
    let fs = length(force.yz);
    if (fs > bounds && fs > 0.0) { force = vec3<f32>(force.x, force.y * bounds / fs, force.z * bounds / fs); }

    return CForce(constraint, force, jALin, jBLin, jAAng, jBAng, k, fs, bounds);
}
`;

// per-joint record accessors (JOINT_REC_VEC4 SoA-free AoS layout — see JOINT_REC_VEC4 doc). Appended after
// a pass declares its `jointRecords` binding; the init/dual passes also write the vec4s directly.
const JOINT_REC_WGSL = /* wgsl */ `
fn jrec(rec: u32, col: u32) -> vec4<f32> { return jointRecords[rec * JOINT_REC_VEC4 + col]; }
fn jActive(rec: u32) -> u32 { return bitcast<u32>(jrec(rec, 3u).y); }
fn jTorqueArm(rec: u32) -> f32 { return jrec(rec, 3u).x; }
fn jPenLin(rec: u32) -> vec3<f32> { return jrec(rec, 4u).xyz; }
fn jPenAng(rec: u32) -> vec3<f32> { return jrec(rec, 5u).xyz; }
fn jLamLin(rec: u32) -> vec3<f32> { return jrec(rec, 6u).xyz; }
fn jLamAng(rec: u32) -> vec3<f32> { return jrec(rec, 7u).xyz; }
fn jC0Lin(rec: u32) -> vec3<f32> { return jrec(rec, 8u).xyz; }
fn jC0Ang(rec: u32) -> vec3<f32> { return jrec(rec, 9u).xyz; }
fn jMotorAxis(rec: u32) -> vec3<f32> { return jrec(rec, 10u).xyz; }
fn jMotorMax(rec: u32) -> f32 { return jrec(rec, 10u).w; }   // > 0 ⇒ the motor is active
fn jMotorSpeed(rec: u32) -> f32 { return jrec(rec, 11u).x; }
fn jMotorLam(rec: u32) -> f32 { return jrec(rec, 11u).y; }
fn jMotorPen(rec: u32) -> f32 { return jrec(rec, 11u).z; }
`;

// the double-buffer scratch the colored primal solves into (paper Algorithm 1 lines 22-24, webphysics
// `bodySolveOutputPose`, oracle `primalColored`): 2 cols, eid-indexed SoA (`solveOut[col*eidCap + bid]`),
// col 0 = solved pos, col 1 = solved quat. The primal reads the *committed* `bodies` and writes its result
// here; a separate commit pass copies the current color's slots into `bodies`. Reading committed poses +
// writing scratch is what makes a same-color contact pair a clean Jacobi (both read the color-start pose),
// not an order-dependent read-write race on `bodies` — the grounding Phase 4.5 Stage B adds.
const SOLVE_OUT_WGSL = /* wgsl */ `
const SO_POS: u32 = 0u;
const SO_QUAT: u32 = 1u;
`;

// ── primal: colored Gauss-Seidel, contact-force stamp + 6×6 LDLᵀ → the double-buffer scratch ──
// The primal reads the committed `bodies` (read-only) and writes each solved body's new pose into the
// `solveOut` scratch, NOT back into `bodies`; the commit pass below applies it for the current color.
// So within one color's dispatch no body's `bodies` slot is both read (by a contact) and written, and a
// same-color pair reduces to the paper's deferred-within-color (clean Jacobi), not a racy write-in-place.
const PRIMAL_BINDINGS_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pairContacts: array<vec4<f32>>;
// CSR adjacency in one binding (Phase 4.9 hygiene): offsets in [0, eidCap), counts in [eidCap, 2·eidCap)
@group(0) @binding(2) var<storage, read> csr: array<u32>;
@group(0) @binding(3) var<storage, read> csrList: array<u32>;
@group(0) @binding(4) var<storage, read> colors: array<u32>;
@group(0) @binding(5) var<uniform> step: Step;
@group(0) @binding(6) var<uniform> color: vec4<u32>;  // dynamic-offset: .x = current color
@group(0) @binding(7) var<storage, read> eids: array<u32>;
@group(0) @binding(8) var<storage, read_write> solveOut: array<vec4<f32>>;
// authored-constraint adjacency (springs Phase 6.1 + joints Phase 6.2 — built CPU-side on setSprings /
// setJoints): per-body offsets/counts in [0,eidCap)/[eidCap,2·eidCap), the inline kind-tagged entries in
// constraintList (CONSTRAINT_VEC4 vec4 each, AoS). A constraint-less scene has all-zero counts ⇒ the loop
// no-ops, so the path is always present, zero cost. binding 11 = the per-joint records (λ/penalty/c0/active),
// read by the joint stamp; springs are stateless so they stamp from the entry alone (springContrib).
@group(0) @binding(9) var<storage, read> constraintCsr: array<u32>;
@group(0) @binding(10) var<storage, read> constraintList: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> jointRecords: array<vec4<f32>>;
${BODY_WGSL}
${CONTACT_WGSL}
${MAT3_WGSL}
${CONTACT_FORCE_WGSL}
${JOINT_REC_WGSL}
${SOLVE_OUT_WGSL}
`;

// the per-body primal step, shared verbatim by the looped color pass (PRIMAL_PASS_WGSL) and the
// LDS-resident kernel (SOLVE_LDS_WGSL): stamp the body's CSR contacts + authored constraints into one
// 6×6 block system, add the inertial term, LDLᵀ-solve, integrate. Pose reads go through bPos/bQuat, so
// the composing kernel picks the backing (storage in the looped pass, workgroup memory in the LDS one).
const SOLVE_MATH_WGSL = /* wgsl */ `
struct Contrib { lhsLin: Mat3, lhsAng: Mat3, lhsCross: Mat3, rhsLin: vec3<f32>, rhsAng: vec3<f32> };

// stamp one contact's force + Hessian into bid's system (manifold.ts updatePrimal). bid is the body
// being solved; ownerIsA selects bid's Jacobian from the shared contactForce.
fn contactContrib(bid: u32, ci: u32) -> Contrib {
    let cf = contactForce(ci);
    let ownerIsA = bitcast<u32>(cc(ci, C_META).y) == bid;
    var jLin = cf.jBLin; var jAng = cf.jBAng;
    if (ownerIsA) { jLin = cf.jALin; jAng = cf.jAAng; }
    let jLinT = mT(jLin);
    let jAngT = mT(jAng);
    let jAngTk = mMul(jAngT, cf.k);
    return Contrib(mMul(mMul(jLinT, cf.k), jLin), mMul(jAngTk, jAng), mMul(jAngTk, jLin),
                   mMulV(jLinT, cf.force), mMulV(jAngT, cf.force));
}

// stamp one spring's force + Hessian into bid's system (spring.ts stampSpring, a port of spring.cpp).
// The soft Force: f = stiffness·C, no dual. Symmetric — both endpoints stamp jLin = normalize(pSelf −
// pOther), so the entry carries bid's own anchor (rSelf) + the partner eid/anchor, no isA branch. The
// 6×6 contributions are the single-row outer products (the contact's Mat3 form specialized to one row).
fn springContrib(bid: u32, e: u32) -> Contrib {
    let base = e * CONSTRAINT_VEC4;
    let s0 = constraintList[base];          // rSelf.xyz, stiffness
    let s1 = constraintList[base + 1u];     // rOther.xyz, rest
    let other = bitcast<u32>(constraintList[base + 2u].x);
    let stiffness = s0.w;
    let rW = qRotateW(bQuat(bid), s0.xyz); // bid's anchor in world — the pSelf offset AND the torque arm
    let pSelf = bPos(bid) + rW;
    let pOther = bPos(other) + qRotateW(bQuat(other), s1.xyz);
    let d = pSelf - pOther;
    let dLen = length(d);
    if (dLen <= 1.0e-6) { return Contrib(mZero(), mZero(), mZero(), vec3<f32>(0.0), vec3<f32>(0.0)); }
    let n = d / dLen;
    let f = stiffness * (dLen - s1.w);
    let jLin = n;
    let jAng = cross(rW, n);
    return Contrib(mScale(outer3(jLin, jLin), stiffness), mScale(outer3(jAng, jAng), stiffness),
                   mScale(outer3(jAng, jLin), stiffness), jLin * f, jAng * f);
}

// stamp one joint's force + Hessian into bid's system (joint.ts stampJoint, a port of joint.cpp). The
// hard Force: a linear anchor-pin row triple + an angular relative-orientation row triple, carrying
// warmstartable λ + a per-iteration penalty ramp, the rigid form adding the stabilization C −= α·C₀.
// Geometry comes from the per-body entry (rSelf/rOther/otherEid/isA); the mutable lambda/penalty/c0 + the
// GPU-computed torqueArm come from the per-joint record (recordIndex). The constraint C uses canonical
// (a minus b) order regardless of which endpoint bid is, selected by isA.
fn jointContrib(bid: u32, e: u32) -> Contrib {
    let zero = Contrib(mZero(), mZero(), mZero(), vec3<f32>(0.0), vec3<f32>(0.0));
    let base = e * CONSTRAINT_VEC4;
    let e2 = constraintList[base + 2u];
    let other = bitcast<u32>(e2.x);
    let rec = bitcast<u32>(e2.z);
    let isA = bitcast<u32>(e2.w) != 0u;
    if (jActive(rec) != 1u) { return zero; }   // version-mismatch / construction-guard deactivated
    // read the anchors from the RECORD (jrec 1 = rA, jrec 2 = rB), not the list entry — the record is the
    // single source of truth, so a moving WORLD anchor (setJointAnchor writes rA into the record each frame) is
    // seen by the primal. isA picks self/other: body A's self is rA, body B's self is rB. (Equal to the list
    // entry's rSelf/rOther for a static joint, so non-world joints are unchanged.)
    let rA = jrec(rec, 1u).xyz;
    let rB = jrec(rec, 2u).xyz;
    let rSelf = select(rB, rA, isA);
    let rOther = select(rA, rB, isA);

    let rigidLin = jrec(rec, 1u).w > RIGID_THRESHOLD;   // stiffnessLin (∞ sentinel) ⇒ rigid stabilization
    let rigidAng = jrec(rec, 2u).w > RIGID_THRESHOLD;
    let torqueArm = jTorqueArm(rec);
    let alpha = step.alpha;

    // the WORLD anchor (other == WORLD_ANCHOR): rOther is a world-space point, orientation identity, no body
    // to read. The grab pins to it (no anchor body → no contact). bid is always a real body (the world has no
    // constraint entry), so qSelf/pSelf read normally.
    let otherWorld = other == WORLD_ANCHOR;
    let qSelf = bQuat(bid);
    let qOther = select(bQuat(other), vec4<f32>(0.0, 0.0, 0.0, 1.0), otherWorld);
    let rSelfW = qRotateW(qSelf, rSelf);
    let pSelf = bPos(bid) + rSelfW;
    let pOther = select(bPos(other) + qRotateW(qOther, rOther), rOther, otherWorld);

    var acc = zero;

    // linear anchor-pin rows
    let penLin = jPenLin(rec);
    if (dot(penLin, penLin) > 0.0) {
        let K = mDiag(penLin);
        var C = select(pOther - pSelf, pSelf - pOther, isA);   // canonical pA − pB
        if (rigidLin) { C = C - jC0Lin(rec) * alpha; }
        let F = mMulV(K, C) + jLamLin(rec);
        let jLin = mDiag(vec3<f32>(select(-1.0, 1.0, isA)));    // ±I
        let jAng = skew(select(rSelfW, -rSelfW, isA));          // isA ? skew(−rA_w) : skew(rB_w)
        let jLinT = mT(jLin); let jAngT = mT(jAng); let jAngTk = mMul(jAngT, K);
        acc.lhsLin = mAdd(acc.lhsLin, mMul(mMul(jLinT, K), jLin));
        acc.lhsAng = mAdd(acc.lhsAng, mMul(jAngTk, jAng));
        acc.lhsCross = mAdd(acc.lhsCross, mMul(jAngTk, jLin));
        let r = select(-rSelfW, rSelfW, isA);                  // geometric-stiffness arm
        let H = mAdd(mAdd(mScale(geomStiffness(0u, r), F.x), mScale(geomStiffness(1u, r), F.y)),
                     mScale(geomStiffness(2u, r), F.z));
        acc.lhsAng = mAdd(acc.lhsAng, diagonalize(H));
        acc.rhsLin = acc.rhsLin + mMulV(jLinT, F);
        acc.rhsAng = acc.rhsAng + mMulV(jAngT, F);
    }
    // angular relative-orientation rows (spherical: penaltyAng 0 ⇒ skipped, rotation free)
    let penAng = jPenAng(rec);
    if (dot(penAng, penAng) > 0.0) {
        let K = mDiag(penAng);
        let qA = select(qOther, qSelf, isA);
        let qB = select(qSelf, qOther, isA);
        var C = qSubW(qA, qB) * torqueArm;
        if (rigidAng) { C = C - jC0Ang(rec) * alpha; }
        let F = mMulV(K, C) + jLamAng(rec);
        let sgn = select(-torqueArm, torqueArm, isA);          // jAng = (±I)·torqueArm (diagonal)
        acc.lhsAng = mAdd(acc.lhsAng, mScale(K, sgn * sgn));   // jAngᵀ·K·jAng = K·sgn²
        acc.rhsAng = acc.rhsAng + F * sgn;                     // jAngᵀ·F = sgn·F
    }
    // motor — a 1-DOF force-clamped angular drive about jMotorAxis (avbd-demo2d motor.cpp; maxTorque > 0
    // activates it). The angular force competes inside each iteration, clamped to ±maxTorque, so a driven body
    // holds its target ω under a load that stalls a forced-velocity drive. deltaAngle is each body's INCREMENTAL
    // rotation about the axis SINCE STEP START (bInitQ = the BDF1 step-start pose), not an absolute qSub against
    // a fixed reference — that reads 2·sin(θ/2), nonlinear far from identity, so a continuously-spinning rotor
    // would over-rotate to null it. Jacobian J = ±axis (unit), Hessian 0 (motor.cpp computeDerivatives).
    let motorMax = jMotorMax(rec);
    if (motorMax > 0.0) {
        let axis = jMotorAxis(rec);
        let dSelf = dot(qSubW(qSelf, bInitQ(bid)), axis);
        let dOther = select(dot(qSubW(qOther, bInitQ(other)), axis), 0.0, otherWorld);
        let dB = select(dSelf, dOther, isA);                   // bid == A ⇒ b is the OTHER endpoint
        let dA = select(dOther, dSelf, isA);
        let c = (dB - dA) - jMotorSpeed(rec) * step.dt;        // deltaAngle(b − a) − speed·dt
        let f = clamp(jMotorPen(rec) * c + jMotorLam(rec), -motorMax, motorMax);
        let sgn = select(1.0, -1.0, isA);                      // ∂C/∂θ = +axis on b, −axis on a
        acc.lhsAng = mAdd(acc.lhsAng, mScale(outer3(axis, axis), jMotorPen(rec)));
        acc.rhsAng = acc.rhsAng + axis * (sgn * f);
    }
    return acc;
}

struct Sol { xLin: vec3<f32>, xAng: vec3<f32> };
fn solve6(aLin: Mat3, aAng: Mat3, aCross: Mat3, bLin: vec3<f32>, bAng: vec3<f32>) -> Sol {
    let A11 = aLin.r0.x; let A21 = aLin.r1.x; let A22 = aLin.r1.y;
    let A31 = aLin.r2.x; let A32 = aLin.r2.y; let A33 = aLin.r2.z;
    let A41 = aCross.r0.x; let A42 = aCross.r0.y; let A43 = aCross.r0.z; let A44 = aAng.r0.x;
    let A51 = aCross.r1.x; let A52 = aCross.r1.y; let A53 = aCross.r1.z; let A54 = aAng.r1.x; let A55 = aAng.r1.y;
    let A61 = aCross.r2.x; let A62 = aCross.r2.y; let A63 = aCross.r2.z; let A64 = aAng.r2.x; let A65 = aAng.r2.y; let A66 = aAng.r2.z;
    let L21 = A21 / A11; let L31 = A31 / A11; let L41 = A41 / A11; let L51 = A51 / A11; let L61 = A61 / A11; let D1 = A11;
    let D2 = A22 - L21*L21*D1;
    let L32 = (A32 - L21*L31*D1) / D2; let L42 = (A42 - L21*L41*D1) / D2; let L52 = (A52 - L21*L51*D1) / D2; let L62 = (A62 - L21*L61*D1) / D2;
    let D3 = A33 - (L31*L31*D1 + L32*L32*D2);
    let L43 = (A43 - L31*L41*D1 - L32*L42*D2) / D3; let L53 = (A53 - L31*L51*D1 - L32*L52*D2) / D3; let L63 = (A63 - L31*L61*D1 - L32*L62*D2) / D3;
    let D4 = A44 - (L41*L41*D1 + L42*L42*D2 + L43*L43*D3);
    let L54 = (A54 - L41*L51*D1 - L42*L52*D2 - L43*L53*D3) / D4; let L64 = (A64 - L41*L61*D1 - L42*L62*D2 - L43*L63*D3) / D4;
    let D5 = A55 - (L51*L51*D1 + L52*L52*D2 + L53*L53*D3 + L54*L54*D4);
    let L65 = (A65 - L51*L61*D1 - L52*L62*D2 - L53*L63*D3 - L54*L64*D4) / D5;
    let D6 = A66 - (L61*L61*D1 + L62*L62*D2 + L63*L63*D3 + L64*L64*D4 + L65*L65*D5);
    let y1 = bLin.x;
    let y2 = bLin.y - L21*y1;
    let y3 = bLin.z - L31*y1 - L32*y2;
    let y4 = bAng.x - L41*y1 - L42*y2 - L43*y3;
    let y5 = bAng.y - L51*y1 - L52*y2 - L53*y3 - L54*y4;
    let y6 = bAng.z - L61*y1 - L62*y2 - L63*y3 - L64*y4 - L65*y5;
    let z1 = y1/D1; let z2 = y2/D2; let z3 = y3/D3; let z4 = y4/D4; let z5 = y5/D5; let z6 = y6/D6;
    var xAng = vec3<f32>(0.0); var xLin = vec3<f32>(0.0);
    xAng.z = z6;
    xAng.y = z5 - L65*xAng.z;
    xAng.x = z4 - L54*xAng.y - L64*xAng.z;
    xLin.z = z3 - L43*xAng.x - L53*xAng.y - L63*xAng.z;
    xLin.y = z2 - L32*xLin.z - L42*xAng.x - L52*xAng.y - L62*xAng.z;
    xLin.x = z1 - L21*xLin.y - L31*xLin.z - L41*xAng.x - L51*xAng.y - L61*xAng.z;
    return Sol(xLin, xAng);
}

struct NewPose { pos: vec3<f32>, quat: vec4<f32> };
fn solvePose(bid: u32) -> NewPose {
    // CSR adjacency: read only this body's own contacts (csrList[off .. off+count]), not a scan of every
    // contact — the O(count·contacts) → O(valence) collapse (physics.md "Dispatch"). offsets + counts share
    // one binding: csr[bid] = offset, csr[eidCap + bid] = count.
    var acc = Contrib(mZero(), mZero(), mZero(), vec3<f32>(0.0), vec3<f32>(0.0));
    let lo = csr[bid];
    let hi = lo + csr[step.eidCap + bid];
    for (var k = lo; k < hi; k = k + 1u) {
        let c = contactContrib(bid, csrList[k]);
        acc.lhsLin = mAdd(acc.lhsLin, c.lhsLin);
        acc.lhsAng = mAdd(acc.lhsAng, c.lhsAng);
        acc.lhsCross = mAdd(acc.lhsCross, c.lhsCross);
        acc.rhsLin = acc.rhsLin + c.rhsLin;
        acc.rhsAng = acc.rhsAng + c.rhsAng;
    }
    // authored constraints (springs + joints): same per-body adjacency shape as contacts, the static
    // authored list (constraintCsr) — additive into the same 6×6, so order is irrelevant. A constraint-less
    // body has count 0 ⇒ no iterations. The kind tag (entry vec4[2].y) picks the soft (spring) or hard (joint) stamp.
    let slo = constraintCsr[bid];
    let shi = slo + constraintCsr[step.eidCap + bid];
    for (var e = slo; e < shi; e = e + 1u) {
        let kind = bitcast<u32>(constraintList[e * CONSTRAINT_VEC4 + 2u].y);
        var c: Contrib;
        if (kind == KIND_JOINT) { c = jointContrib(bid, e); } else { c = springContrib(bid, e); }
        acc.lhsLin = mAdd(acc.lhsLin, c.lhsLin);
        acc.lhsAng = mAdd(acc.lhsAng, c.lhsAng);
        acc.lhsCross = mAdd(acc.lhsCross, c.lhsCross);
        acc.rhsLin = acc.rhsLin + c.rhsLin;
        acc.rhsAng = acc.rhsAng + c.rhsAng;
    }

    let mm = bCol(B_MM, bid);
    let mLin = mDiag(vec3<f32>(mm.w) * step.invDt2);
    let mAng = mDiag(mm.xyz * step.invDt2);
    let lhsLin = mAdd(acc.lhsLin, mLin);
    let lhsAng = mAdd(acc.lhsAng, mAng);
    let rhsLin = acc.rhsLin + mMulV(mLin, bPos(bid) - bInertL(bid));
    let rhsAng = acc.rhsAng + mMulV(mAng, qSubW(bQuat(bid), bInertQ(bid)));
    let r = solve6(lhsLin, lhsAng, acc.lhsCross, -rhsLin, -rhsAng);
    return NewPose(bPos(bid) + r.xLin, qAddW(bQuat(bid), r.xAng));
}
`;

const PRIMAL_PASS_WGSL =
    SHARED_WGSL +
    PRIMAL_BINDINGS_WGSL +
    SOLVE_MATH_WGSL +
    /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let bid = eids[1u + d];
    if (solverStatic(bid)) { return; }          // static / kinematic — no primal
    if (colors[bid] != color.x) { return; }  // colored GS: only this color commits
    let np = solvePose(bid);
    // double-buffer: write the solved pose to the scratch, not back into bodies. bodies is read-only in
    // this pass, so a same-color contact pair never races on the pose; the commit applies it per color.
    let cap = step.eidCap;
    solveOut[SO_POS*cap + bid] = vec4<f32>(np.pos, 0.0);
    solveOut[SO_QUAT*cap + bid] = np.quat;
}
`;

// ── commit: apply the current color's solved poses from the scratch into `bodies` (the deferred commit) ──
// The write half of the double-buffer (paper Algorithm 1 lines 22-24, webphysics `commitBodySolveKernel`).
// One dispatch after each color's primal: copy `solveOut` → `bodies` for the bodies of `color.x` only, so
// the next color's primal reads the committed pose. Gated identically to the primal (live count, static,
// color), so every body it writes the primal just solved into `solveOut` this color — no stale read, no
// clear needed. A same-color pair is now a clean Jacobi: both solved from the color-start pose, committed
// together, matching the oracle's `primalColored`.
const COMMIT_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> solveOut: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> colors: array<u32>;
@group(0) @binding(2) var<storage, read_write> bodies: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> eids: array<u32>;
@group(0) @binding(4) var<uniform> step: Step;
@group(0) @binding(5) var<uniform> color: vec4<u32>;  // dynamic-offset: .x = current color
${BODY_WGSL}
${SOLVE_OUT_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let bid = eids[1u + d];
    if (solverStatic(bid)) { return; }        // static / kinematic — no primal, no commit
    if (colors[bid] != color.x) { return; }  // only this color commits (the deferred-within-color apply)
    let cap = step.eidCap;
    bodies[B_POS*cap + bid] = solveOut[SO_POS*cap + bid];
    bodies[B_QUAT*cap + bid] = solveOut[SO_QUAT*cap + bid];
}
`;

// ── dual: λ ← F + the conditional penalty ramp + the friction stick flag (manifold.ts updateDual) ──
// One dispatch over the contact rows, once per iteration after the primal colors.
// Reads the post-primal pose, recomputes the same cone-clamped force the primal used, stores it as
// λ, and ramps the penalty: the normal stiffness only while the contact is active (F[0] < 0), the
// tangent stiffness only inside the friction cone (sticking). The stick flag rides λ.w for the
// Phase-3 warmstart merge (unused within a frame). Each contact is independent — one thread, no race.
// one pair SLOT's dual update, shared verbatim by the standalone dual pass and the LDS-resident kernel:
// loop its CONTACTS_PER_PAIR records, dual-update each active one in place (an INVALID/unused slot is all
// kind-0 records → every record continues, a no-op).
const DUAL_MATH_WGSL = /* wgsl */ `
const STICK_THRESH: f32 = 1.0e-5;
fn dualSlot(slot: u32) {
    let rc = step.recordCap;
    let recBase = slot * CONTACTS_PER_PAIR;
    for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
        let ci = recBase + ls;
        let m0 = cc(ci, C_META);
        if (bitcast<u32>(m0.x) != ${CONSTRAINT_CONTACT}u) { continue; } // inactive record
        // dual-ramp gate (roadmap §6.4): an all-static manifold — both solverStatic (mass <= 0:
        // a kinematic character against a static wall) — is unsatisfiable by any primal, so the
        // reference's unconditional ramp would escalate its penalty unbounded (the legacy
        // kinematic-pushing blow-up). Skip it, mirroring the oracle (manifold.ts updateDual).
        if (solverStatic(bitcast<u32>(m0.y)) && solverStatic(bitcast<u32>(m0.z))) { continue; }
        let cf = contactForce(ci);
        let pen = cc(ci, C_PEN);
        let friction = pen.w;
        var k = pen.xyz;
        // pre-clamp magnitude + bound (cf), not the post-clamp force.yz — a sliding contact (fs > bounds)
        // must skip the tangent ramp so its penalty stays bounded and kinetic friction holds at μ|F_n|.
        let bounds = cf.bounds;
        let fs = cf.frictionScale;
        if (cf.force.x < 0.0) {
            k.x = min(k.x + step.betaLin * abs(cf.constraint.x), PENALTY_MAX);
        }
        var stick = 0.0;
        if (fs <= bounds) {
            k.y = min(k.y + step.betaLin * abs(cf.constraint.y), PENALTY_MAX);
            k.z = min(k.z + step.betaLin * abs(cf.constraint.z), PENALTY_MAX);
            if (length(cf.constraint.yz) < STICK_THRESH) { stick = 1.0; }
        }
        pairContacts[C_LAMBDA*rc + ci] = vec4<f32>(cf.force, stick);
        pairContacts[C_PEN*rc + ci] = vec4<f32>(k, friction);
    }
}
`;

const DUAL_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> pairContacts: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> step: Step;
@group(0) @binding(3) var<storage, read> eids: array<u32>; // [0] = live count, [1+d] = the d-th live eid
${BODY_WGSL}
${CONTACT_WGSL}
${MAT3_WGSL}
${CONTACT_FORCE_WGSL}
${DUAL_MATH_WGSL}
// one thread per pair SLOT in the live bodies' per-eid blocks (lane → d → owner eid → slot = eid·K + k)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    let d = lane / PAIRS_PER_BODY;
    if (d >= eids[0u]) { return; } // past the live body count
    dualSlot(eids[1u + d] * PAIRS_PER_BODY + (lane % PAIRS_PER_BODY));
}
`;

// ── coloring: incremental-greedy body coloring (the Phase-4 crux, validated standalone) ──
// Ported from webphysics `greedyBodyColorsShader`: one thread per body, no atomics. Each body reads a
// stable prior-frame snapshot of every body's color (`colorScratch`, seeded by a copy before this pass),
// builds a 32-wide u32 mask of the colors its *higher-id dynamic neighbors* held last frame, and picks
// the lowest color not in that mask — keeping its own prior color when still free (the incremental
// reuse that makes the coloring settle across frames). The higher-id symmetry break (avoid only
// neighbors with `other > bid`) is what removes the atomics: each undirected contact edge is resolved
// by its lower-id endpoint, against the higher-id endpoint's prior color. Static neighbors (mass ≤ 0)
// are never solved, so they impose no scheduling constraint and are skipped; a static body itself is
// left uncolored (0xffffffff — the primal early-returns on mass before reading its color).
//
// The cap (`step.maxColors`) is separate from the 32-wide mask (scratch.md "AVBD rebuild" — two
// numbers): a body that finds no free color within the cap folds to `bid % maxColors`, degrading that
// pair to Jacobi for the step (a soft-contact conflict the iterative primal tolerates — the invariant
// is a low *measured* conflict rate, not zero). The neighbor scan reads the body's CSR contact list
// (csrList[csr[bid] .. +csr[eidCap+bid]]), not every contact — the same O(valence) read the primal
// does. Deterministic integer logic; the CPU reference is tests/avbd/coloring.ts (the executable
// spec) and the GPU reproduces it in the gym `pile` scenario (coloring-conflict counter).
const COLORING_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pairContacts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> csr: array<u32>;  // [0, eidCap) offsets, [eidCap, 2·eidCap) counts
@group(0) @binding(3) var<storage, read> csrList: array<u32>;
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;
@group(0) @binding(5) var<storage, read> colorScratch: array<u32>;
@group(0) @binding(6) var<uniform> step: Step;
@group(0) @binding(7) var<storage, read> eids: array<u32>;
// colorCount[0] = the used-color count this step (max dynamic color + 1), the readback-bounded color
// loop's input (Phase 4.9 Lever 1). Cleared each step before this pass; each dynamic body atomicMaxes
// its chosen color + 1. One slot, low contention (one op/body, one dispatch/step) — the counters class.
@group(0) @binding(8) var<storage, read_write> colorCount: array<atomic<u32>>;
// authored-constraint adjacency (springs Phase 6.1 + joints Phase 6.2) — the edge enters the coloring so a
// constraint-connected dynamic pair prefers different colors (avoidance, kind-agnostic). A soft spring
// tolerates a same-color clean-Jacobi pair; a hard joint must NOT be same-color, so the joint edge gets a
// second repair pass (REPAIR_PASS_WGSL) on top of this avoidance. constraintList[e+2].x = the partner eid.
@group(0) @binding(9) var<storage, read> constraintCsr: array<u32>;
@group(0) @binding(10) var<storage, read> constraintList: array<vec4<f32>>;
${BODY_WGSL}
${CONTACT_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let bid = eids[1u + d];
    if (bMass(bid) <= 0.0) { colors[bid] = 0xffffffffu; return; } // static — uncolored, not dispatched

    let colorsN = max(1u, min(step.maxColors, 32u));
    let lo = csr[bid];
    let hi = lo + csr[step.eidCap + bid];
    var usedMask = 0u;
    for (var k = lo; k < hi; k = k + 1u) {
        let m = cc(csrList[k], C_META);
        let a = bitcast<u32>(m.y);
        let b = bitcast<u32>(m.z);
        var other = a;                          // every CSR contact touches bid; pick the neighbor
        if (a == bid) { other = b; }
        if (other <= bid) { continue; }        // higher-id symmetry break — no atomics
        if (solverStatic(other)) { continue; }  // static / kinematic neighbor: never solved, no scheduling constraint
        let pc = colorScratch[other];
        if (pc < 32u) { usedMask = usedMask | (1u << pc); }
    }
    // authored-constraint neighbors (springs + joints): same higher-id-symmetry avoidance as contacts, kind-
    // agnostic (the partner eid is constraintList[e+2].x for both). A constraint-less body has count 0.
    let slo = constraintCsr[bid];
    let shi = slo + constraintCsr[step.eidCap + bid];
    for (var e = slo; e < shi; e = e + 1u) {
        let other = bitcast<u32>(constraintList[e * CONSTRAINT_VEC4 + 2u].x);
        if (other <= bid) { continue; }
        if (solverStatic(other)) { continue; } // static / kinematic neighbor: never solved, no scheduling constraint
        let pc = colorScratch[other];
        if (pc < 32u) { usedMask = usedMask | (1u << pc); }
    }

    var chosen = colorScratch[bid];             // incremental: keep the prior color when still free
    var needsNew = chosen >= colorsN;
    if (!needsNew) { needsNew = (usedMask & (1u << chosen)) != 0u; }
    if (needsNew) {
        var found = false;
        for (var c = 0u; c < colorsN; c = c + 1u) {
            if ((usedMask & (1u << c)) == 0u) { chosen = c; found = true; break; }
        }
        if (!found) { chosen = bid % colorsN; } // fold past the cap — a tolerated same-color conflict
    }
    colors[bid] = chosen;
    // publish the used-color count (max dynamic color + 1) for the readback-bounded color loop — the
    // primal next frame dispatches min(maxColors, usedColors + COLOR_MARGIN) color-passes (Phase 4.9 Lever 1).
    atomicMax(&colorCount[0], chosen + 1u);
}
`;

// ── repair: the joint hard-conflict coloring repair (Phase 6.2, webphysics repairHardBodyColors) ──
// The greedy avoids ALL constraint neighbors but folds past the cap (a tolerated same-color Jacobi). A SOFT
// spring survives that; a HARD (dynamic-dynamic) joint pair degrading to same-color Jacobi destabilizes, so
// after the greedy this runs JOINT_REPAIR_ROUNDS rounds: each round snapshots colors → colorScratch, then
// each lower-eid endpoint of a same-color joint pair recolors to a free color (excluding all its constraint
// neighbors). Reading the stable snapshot keeps it race-free + deterministic, like the greedy. GPU==oracle
// can't validate this (the oracle runs the GPU's coloring), so the gym gates the observable invariant: a
// dynamic joint pair ends colors[a] != colors[b].
const REPAIR_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> colors: array<u32>;
@group(0) @binding(2) var<storage, read> colorScratch: array<u32>;
@group(0) @binding(3) var<storage, read> constraintCsr: array<u32>;
@group(0) @binding(4) var<storage, read> constraintList: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> step: Step;
@group(0) @binding(6) var<storage, read> eids: array<u32>;
@group(0) @binding(7) var<storage, read_write> colorCount: array<atomic<u32>>;
${BODY_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let bid = eids[1u + d];
    if (bMass(bid) <= 0.0) { return; }                  // static — uncolored, never a hard mover

    let colorsN = max(1u, min(step.maxColors, 32u));
    let myColor = colorScratch[bid];
    let slo = constraintCsr[bid];
    let shi = slo + constraintCsr[step.eidCap + bid];
    var usedMask = 0u;
    var hardConflict = false;
    for (var e = slo; e < shi; e = e + 1u) {
        let base = e * CONSTRAINT_VEC4;
        let other = bitcast<u32>(constraintList[base + 2u].x);
        let kind = bitcast<u32>(constraintList[base + 2u].y);
        if (other >= step.eidCap || bMass(other) <= 0.0) { continue; } // static neighbor: no constraint
        let oc = colorScratch[other];
        if (oc < 32u) { usedMask = usedMask | (1u << oc); }
        // the lower-eid endpoint of a same-color joint pair is the one that moves (higher-id stays fixed)
        if (kind == KIND_JOINT && other > bid && oc == myColor) { hardConflict = true; }
    }
    if (!hardConflict) { return; }                      // already conflict-free — its color stays counted
    var chosen = myColor;
    var found = false;
    for (var c = 0u; c < colorsN; c = c + 1u) {
        if ((usedMask & (1u << c)) == 0u) { chosen = c; found = true; break; }
    }
    if (!found) { chosen = bid % colorsN; }             // fold (a tolerated soft conflict — no free color left)
    colors[bid] = chosen;
    atomicMax(&colorCount[0], chosen + 1u);             // keep the readback-bounded loop's count ≥ the repaired max
}
`;

// ── joint init: warmstart the per-joint dual state + capture C(x⁻) (Phase 6.2, joint.ts initJoint) ──
// One thread per joint, before the main loop (after the contact collide, before inertial init so it reads
// the step-start pose x⁻). Runs the recycle-version guard + the one-time construction guard, computes the
// torque arm GPU-side, captures C₀, then decays λ/penalty (Eq. 19) and clamps to material stiffness.
const JOINT_INIT_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> jointRecords: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> jointVersions: array<u32>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
// per-eid seed flag (the pack sets it after seeding a body's slot from its slabs): a joint must not read a
// body's pose until it's seeded. jointInit is per-JOINT, NOT gated on the live count, so on the very first
// fixed step (which can run before the first pack at 60 Hz) the bodies slots are still zero-init — without
// this gate the fresh anchor guard would see half = 0 (reach a tiny margin) + offset anchors and WRONGLY
// reject every valid joint. Skipping until both ends are seeded defers the guard to a frame with real poses.
@group(0) @binding(5) var<storage, read> seeded: array<u32>;
${BODY_WGSL}
${JOINT_REC_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let jid = gid.x;
    if (jid >= step.jointCount) { return; }
    let r0 = jrec(jid, 0u);
    let a = bitcast<u32>(r0.x);
    let b = bitcast<u32>(r0.y);
    let recBase = jid * JOINT_REC_VEC4;
    // a == WORLD_ANCHOR is the world (no body): rA is a world point, orientation identity, mass/size/radius 0,
    // always seeded, no version. The grab pins to it (no anchor body → no contact). b is always a real body.
    let aWorld = a == WORLD_ANCHOR;

    // recycle-version guard (project_stable_identity): a despawned-then-recycled endpoint must not realias
    // the joint to a new body. A version mismatch deactivates it (its primal stamp + dual then no-op). The
    // world anchor has no version (skip its side).
    let aBad = !aWorld && jointVersions[a] != bitcast<u32>(r0.z);
    if (aBad || jointVersions[b] != bitcast<u32>(r0.w)) {
        jointRecords[recBase + 3u].y = bitcast<f32>(0u);
        return;
    }
    if ((!aWorld && seeded[a] == 0u) || seeded[b] == 0u) { return; } // not seeded yet — retry after the pack

    // both-static guard (the GPU analog of joint()'s both-static throw — graceful + loud, since the GPU can't
    // throw): a joint NO dynamic body can resolve — both endpoints mass <= 0 (static/kinematic; the world is
    // static) — is never satisfiable by the primal, so its dual ramps penalty + lambda unbounded (energy
    // injection, the joint analog of the contact all-static dual guard). Checked EVERY frame (a both-static
    // joint is permanently unsatisfiable, not just at construction), so counters[1] is a persistent gauge of
    // the condition, not a one-frame blip a lagged Mirror readback can miss. Gated behind the seed gate so the
    // zero-init mass of an unseeded body can't false-trigger it. Runs before the act==0 early-out so a
    // still-present bad joint keeps being counted (deactivation alone would silence it after frame 1).
    let aStatic = aWorld || bMass(a) <= 0.0;
    if (aStatic && bMass(b) <= 0.0) {
        jointRecords[recBase + 3u].y = bitcast<f32>(0u);
        atomicAdd(&counters[1], 1u);                    // observable: a both-endpoints-static joint, deactivated
        return;
    }
    let act = jActive(jid);
    if (act == 0u) { return; }                          // already deactivated — stays off until re-authored

    let rA = jrec(jid, 1u).xyz; let stiffLin = jrec(jid, 1u).w;
    let rB = jrec(jid, 2u).xyz; let stiffAng = jrec(jid, 2u).w;
    // torqueArm = ‖sizeA + sizeB‖² (full size = 2·halfExtents) — GPU-computed so the CPU needn't carry sizes.
    // The world anchor contributes size 0 + identity orientation + its world point rA.
    let sizeA = select(2.0 * bHalf(a), vec3<f32>(0.0), aWorld);
    let sizeB = 2.0 * bHalf(b);
    let torqueArm = dot(sizeA + sizeB, sizeA + sizeB);
    let qA = select(bQuat(a), vec4<f32>(0.0, 0.0, 0.0, 1.0), aWorld);
    let pA = select(bPos(a) + qRotateW(qA, rA), rA, aWorld);
    let pB = bPos(b) + qRotateW(bQuat(b), rB);

    if (act == 2u) {
        // anchor-coincidence guard (the other GPU analog of joint()'s throw): the anchors must START coincident;
        // a gross mismatch injects energy through BDF1 recovery (the rope explosion). Reject + count rather than
        // explode 50 frames in. Reach = length(halfExtents) + the rounded radius — a sphere/capsule carries its
        // size in bRadius with halfExtents (0,0,0), so a reach off bHalf alone rejects every sphere joint at
        // the slightest offset. Pose-dependent, so only on the fresh frame.
        let reachA = select(length(bHalf(a)) + bRadius(a), 0.0, aWorld);
        let reach = reachA + length(bHalf(b)) + bRadius(b) + COLLISION_MARGIN;
        if (length(pA - pB) > reach) {
            jointRecords[recBase + 3u].y = bitcast<f32>(0u);
            atomicAdd(&counters[2], 1u);                // observable: joints rejected by the anchor guard
            return;
        }
    }
    jointRecords[recBase + 3u] = vec4<f32>(torqueArm, bitcast<f32>(1u), 0.0, 0.0);

    // C(x⁻) at the step-start pose (this pass runs before inertial init predicts the pose)
    jointRecords[recBase + 8u] = vec4<f32>(pA - pB, 0.0);
    jointRecords[recBase + 9u] = vec4<f32>(qSubW(qA, bQuat(b)) * torqueArm, 0.0);

    // warmstart λ + penalty (Eq. 19): λ ← α·γ·λ, k ← clamp(γ·k, MIN, MAX) then clamp to material stiffness
    let ag = step.alpha * step.gamma;
    jointRecords[recBase + 6u] = vec4<f32>(jLamLin(jid) * ag, 0.0);
    jointRecords[recBase + 7u] = vec4<f32>(jLamAng(jid) * ag, 0.0);
    let penLin = min(clamp(jPenLin(jid) * step.gamma, vec3<f32>(PENALTY_MIN), vec3<f32>(PENALTY_MAX)), vec3<f32>(stiffLin));
    let penAng = min(clamp(jPenAng(jid) * step.gamma, vec3<f32>(PENALTY_MIN), vec3<f32>(PENALTY_MAX)), vec3<f32>(stiffAng));
    jointRecords[recBase + 4u] = vec4<f32>(penLin, 0.0);
    jointRecords[recBase + 5u] = vec4<f32>(penAng, 0.0);

    // motor warmstart (Eq. 19): decay λ + penalty. The static axis/speed/maxTorque (col 10, 11.x) are not
    // rewritten by this pass, so they persist from setJoints across frames.
    if (jMotorMax(jid) > 0.0) {
        let mp = clamp(jMotorPen(jid) * step.gamma, PENALTY_MIN, PENALTY_MAX);
        jointRecords[recBase + 11u] = vec4<f32>(jMotorSpeed(jid), jMotorLam(jid) * ag, mp, 0.0);
    }
}
`;

// ── joint dual: advance λ + the penalty ramp per joint each iteration (Phase 6.2, joint.ts updateJointDual) ──
// One thread per joint, after each iteration's primal (like the contact dual). The rigid (∞-stiffness)
// rows store λ ← K·C + λ; both row triples ramp the penalty by β|C|, clamped to PENALTY_MAX.
// one joint's dual update, shared verbatim by the standalone joint-dual pass and the LDS-resident kernel
const JOINT_DUAL_MATH_WGSL = /* wgsl */ `
fn jointDualOne(jid: u32) {
    if (jActive(jid) != 1u) { return; }
    let recBase = jid * JOINT_REC_VEC4;
    let a = bitcast<u32>(jrec(jid, 0u).x);
    let b = bitcast<u32>(jrec(jid, 0u).y);
    let aWorld = a == WORLD_ANCHOR;                      // the world anchor: rA a world point, orientation identity
    // all-static gate (joint.ts updateJointDual): both endpoints static (the world anchor counting
    // as static) — no dynamic body can satisfy it, so ramping its λ/penalty injects energy. Mirrors the
    // construction-time both-static REJECTION (jointInit). aWorld short-circuits so solverStatic never
    // reads the sentinel eid.
    if (solverStatic(b) && (aWorld || solverStatic(a))) { return; }
    let qA = select(bQuat(a), vec4<f32>(0.0, 0.0, 0.0, 1.0), aWorld);
    let rA = jrec(jid, 1u).xyz; let stiffLin = jrec(jid, 1u).w;
    let rB = jrec(jid, 2u).xyz; let stiffAng = jrec(jid, 2u).w;
    let torqueArm = jTorqueArm(jid);
    let alpha = step.alpha;

    var penLin = jPenLin(jid);
    if (dot(penLin, penLin) > 0.0) {
        let pA = select(bPos(a) + qRotateW(qA, rA), rA, aWorld);
        let pB = bPos(b) + qRotateW(bQuat(b), rB);
        var C = pA - pB;
        if (stiffLin > RIGID_THRESHOLD) {
            C = C - jC0Lin(jid) * alpha;
            jointRecords[recBase + 6u] = vec4<f32>(penLin * C + jLamLin(jid), 0.0); // λ ← K·C + λ
        }
        penLin = min(penLin + abs(C) * step.betaLin, vec3<f32>(min(stiffLin, PENALTY_MAX)));
        jointRecords[recBase + 4u] = vec4<f32>(penLin, 0.0);
    }
    var penAng = jPenAng(jid);
    if (dot(penAng, penAng) > 0.0) {
        var C = qSubW(qA, bQuat(b)) * torqueArm;
        if (stiffAng > RIGID_THRESHOLD) {
            C = C - jC0Ang(jid) * alpha;
            jointRecords[recBase + 7u] = vec4<f32>(penAng * C + jLamAng(jid), 0.0);
        }
        penAng = min(penAng + abs(C) * step.betaAng, vec3<f32>(min(stiffAng, PENALTY_MAX)));
        jointRecords[recBase + 5u] = vec4<f32>(penAng, 0.0);
    }
    // motor dual — λ clamped to ±maxTorque (the bounded-constraint update). The penalty ramps toward
    // PENALTY_MAX (stiffness ∞) ONLY while λ is strictly inside the force bounds (solver.cpp's lambda-inside
    // -fmin/fmax gate): a saturated motor keeps a small penalty so it stays a constant-torque drive (accel
    // maxTorque/I); ramping it there over-stiffens the Hessian and drags the spin-up below that rate. The
    // rigid rows above need no such gate — their bounds are unbounded, so λ is always inside.
    let motorMax = jMotorMax(jid);
    if (motorMax > 0.0) {
        let axis = jMotorAxis(jid);
        let dB = dot(qSubW(bQuat(b), bInitQ(b)), axis);
        let dA = select(dot(qSubW(bQuat(a), bInitQ(a)), axis), 0.0, aWorld);
        let c = (dB - dA) - jMotorSpeed(jid) * step.dt;
        let lam = clamp(jMotorLam(jid) + jMotorPen(jid) * c, -motorMax, motorMax);
        var pen = jMotorPen(jid);
        if (lam > -motorMax && lam < motorMax) {
            pen = min(pen + abs(c) * step.betaAng, PENALTY_MAX);
        }
        jointRecords[recBase + 11u] = vec4<f32>(jMotorSpeed(jid), lam, pen, 0.0);
    }
}
`;

const JOINT_DUAL_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> jointRecords: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> step: Step;
${BODY_WGSL}
${JOINT_REC_WGSL}
${JOINT_DUAL_MATH_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let jid = gid.x;
    if (jid >= step.jointCount) { return; }
    jointDualOne(jid);
}
`;

// ── LDS-resident solve (C1.2): the whole iters × colors primal/commit/dual block as ONE dispatch ──
// The small-N solve is latency-bound on a serial phase chain: each color phase's cost is its dependent
// storage round trip (CSR → records → both bodies' poses), paid the same at a dispatch boundary as at an
// in-kernel storageBarrier — which is why the plain single-WG megakernel measured nothing (physics.md
// "Dispatch count", refuted 2026-06-10). This kernel removes the round trip itself: every live body's
// pose lives in workgroup memory across the loop (Bullet 3 solveContact.cl solves from __local batches),
// so a color phase's dependent chain is a workgroupBarrier on LDS. Everything else a solve reads (CSR,
// contact records, inertial targets, mass) is loop-constant or once-per-iteration storage traffic.
//
// Capacity: pos (3 split f32 columns) + quat (vec4) = 28 B/body → LDS_CAP = 512 inside the 16 KB
// workgroup-memory floor. Contacts/joints address bodies by EID, so an eid → dense map (denseMap,
// rebound over the solveOut buffer — unused by this path, ≥ 4·eidCap bytes) routes bPos/bQuat into the
// LDS slot; a live body past LDS_CAP (a spawn burst inside the regime gate's 1-2 frame staleness — the
// BODY_MARGIN class) falls back to its storage pose and skips its solve for one frame, which the next
// readback catches. The per-color double-buffer (a folded same-color pair must be a clean Jacobi) stages
// each lane's ≤2 solved poses in registers and commits them after a barrier; the dual + joint dual
// stride their slots under a storageBarrier per iteration (records are storage-resident — far too large
// for LDS, and touched once per iteration, not per color). The color count is computed in-kernel
// (atomicMax over the live dynamics' colors), fresher than the looped path's readback bound and one
// binding cheaper — the kernel is exactly AT the 10-storage-buffer floor.
const SOLVE_LDS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> pairContacts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> csr: array<u32>;
@group(0) @binding(3) var<storage, read> csrList: array<u32>;
@group(0) @binding(4) var<storage, read> colors: array<u32>;
@group(0) @binding(5) var<uniform> step: Step;
@group(0) @binding(6) var<storage, read> eids: array<u32>;
@group(0) @binding(7) var<storage, read_write> denseMap: array<u32>;
@group(0) @binding(8) var<storage, read> constraintCsr: array<u32>;
@group(0) @binding(9) var<storage, read> constraintList: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> jointRecords: array<vec4<f32>>;

const LDSN: u32 = ${LDS_CAP}u;
const SOLVE_WG: u32 = 256u;
var<workgroup> lpx: array<f32, LDSN>;
var<workgroup> lpy: array<f32, LDSN>;
var<workgroup> lpz: array<f32, LDSN>;
var<workgroup> lq: array<vec4<f32>, LDSN>;
var<workgroup> wgCount: u32;
var<workgroup> wgColorMax: atomic<u32>;
var<workgroup> wgColors: u32;

// the LDS-backed pose readers the shared solve/dual math goes through. Every live eid's denseMap entry
// is written at kernel start, so a slot ≥ LDSN means the body overflowed residency → its storage pose
// (constant this step: an overflow body is never solved) is the consistent fallback.
fn bPos(i: u32) -> vec3<f32> {
    let s = denseMap[i];
    if (s < LDSN) { return vec3<f32>(lpx[s], lpy[s], lpz[s]); }
    return bCol(B_POS, i).xyz;
}
fn bQuat(i: u32) -> vec4<f32> {
    let s = denseMap[i];
    if (s < LDSN) { return lq[s]; }
    return bCol(B_QUAT, i);
}
` +
    BODY_COLS_WGSL +
    BODY_REST_WGSL +
    CONTACT_WGSL +
    MAT3_WGSL +
    CONTACT_FORCE_WGSL +
    JOINT_REC_WGSL +
    SOLVE_MATH_WGSL +
    DUAL_MATH_WGSL +
    JOINT_DUAL_MATH_WGSL +
    /* wgsl */ `
@compute @workgroup_size(SOLVE_WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    if (lane == 0u) { wgCount = eids[0u]; }
    let count = workgroupUniformLoad(&wgCount);
    let n = min(count, LDSN);

    // load: eid → dense map + the resident poses + the used-color count (max dynamic color + 1 — the
    // looped path's readback-bounded count, computed GPU-fresh; colors past it hold no bodies, so the
    // loop bound is a dispatch-count choice, never a math change)
    for (var d = lane; d < count; d = d + SOLVE_WG) {
        let eid = eids[1u + d];
        denseMap[eid] = d;
        if (d < LDSN) {
            let p = bCol(B_POS, eid).xyz;
            lpx[d] = p.x; lpy[d] = p.y; lpz[d] = p.z;
            lq[d] = bCol(B_QUAT, eid);
        }
        if (bMass(eid) > 0.0) { atomicMax(&wgColorMax, colors[eid] + 1u); }
    }
    storageBarrier();   // denseMap visible before any bPos/bQuat routes through it
    workgroupBarrier(); // resident poses + wgColorMax
    if (lane == 0u) { wgColors = min(atomicLoad(&wgColorMax), step.maxColors); }
    let colorsToRun = workgroupUniformLoad(&wgColors);

    for (var it = 0u; it < step.iterations; it = it + 1u) {
        for (var c = 0u; c < colorsToRun; c = c + 1u) {
            // primal: each lane solves its ≤2 bodies of this color from the committed LDS poses,
            // staging the results in registers — the double-buffer (a folded same-color pair reads the
            // color-start pose on both sides, the clean Jacobi the looped solveOut/commit pair gives)
            var np0: NewPose; var w0 = false;
            let d0 = lane;
            if (d0 < n) {
                let bid = eids[1u + d0];
                if (!solverStatic(bid) && colors[bid] == c) { np0 = solvePose(bid); w0 = true; }
            }
            var np1: NewPose; var w1 = false;
            let d1 = lane + SOLVE_WG;
            if (d1 < n) {
                let bid = eids[1u + d1];
                if (!solverStatic(bid) && colors[bid] == c) { np1 = solvePose(bid); w1 = true; }
            }
            workgroupBarrier();
            // commit: the color's staged poses land in LDS together
            if (w0) { lpx[d0] = np0.pos.x; lpy[d0] = np0.pos.y; lpz[d0] = np0.pos.z; lq[d0] = np0.quat; }
            if (w1) { lpx[d1] = np1.pos.x; lpy[d1] = np1.pos.y; lpz[d1] = np1.pos.z; lq[d1] = np1.quat; }
            workgroupBarrier();
        }
        // dual + joint dual: the standalone passes' lane mappings, strided over one workgroup. They
        // read the post-color LDS poses and write λ/penalty into the persistent storage records, which
        // the next iteration's primal reads — once-per-iteration storage traffic, not per color.
        let slots = count * PAIRS_PER_BODY;
        for (var s = lane; s < slots; s = s + SOLVE_WG) {
            dualSlot(eids[1u + s / PAIRS_PER_BODY] * PAIRS_PER_BODY + (s % PAIRS_PER_BODY));
        }
        for (var j = lane; j < step.jointCount; j = j + SOLVE_WG) { jointDualOne(j); }
        storageBarrier(); // record λ/penalty writes → the next iteration's contactForce reads
    }

    // write back the solved poses. Statics (incl. kinematic characters) are skipped — the looped path
    // never writes them either, and a character's char-pass pose must not be re-stamped with w = 0.
    for (var d = lane; d < n; d = d + SOLVE_WG) {
        let eid = eids[1u + d];
        if (solverStatic(eid)) { continue; }
        bodies[B_POS*step.eidCap + eid] = vec4<f32>(lpx[d], lpy[d], lpz[d], 0.0);
        bodies[B_QUAT*step.eidCap + eid] = lq[d];
    }
}
`;

// ── compose: scatter the interpolated body pose into the eid-indexed transform firehose ──
// The bodied-entity half of the Body/Transform contract (roadmap): a `Body` is a `Part` whose world
// matrix physics owns. `Body.excludes [Transform]`, so the Transform compose writes a stale slot for a
// body eid; this pass runs after it and overwrites `transforms[eids[d]]` with the live pose. Scale is
// 2·halfExtents — the cube mesh is unit (-0.5..0.5), so the render box matches the collision box (the
// body pose itself is scale-free; this is render-only). Writes the decomposed `Xform` (the same struct
// the Transform compose gathers); readers reconstruct the world transform via XFORM_WGSL.
//
// Render interpolation (Phase 5): the solver steps at the fixed
// rate but compose runs every render frame, so at >60Hz it would repeat a fixed-step pose then jump
// (stutter). Blend prev→curr by `interp.alpha` (= time.fixedAlpha, the fraction past the last fixed tick).
// The prev pose needs no extra column or snapshot pass: the inertial pass already saves x⁻ (the pre-warmstart
// pose = last frame's settled pose) into B_INITL/B_INITQ before warmstart mutates B_POS, so prev = bInit*,
// curr = bPos/bQuat. lerp position, nlerp quat on the shortest arc. For a static or freshly-seeded body
// B_INITL == B_POS, so it's a no-op; at alpha = 1 this is exactly the bare current pose.
const COMPOSE_PASS_WGSL =
    SHARED_WGSL +
    XFORM_WGSL +
    /* wgsl */ `
struct Interp { alpha: f32 };
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> eids: array<u32>;
@group(0) @binding(2) var<storage, read_write> transforms: array<Xform>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<uniform> interp: Interp;
${BODY_WGSL}
// nlerp toward the shortest arc: flip prev into curr's hemisphere, lerp, renormalize (legacy interpolate.wgsl)
fn nlerpShortest(prev: vec4<f32>, curr: vec4<f32>, t: f32) -> vec4<f32> {
    let flip = select(1.0, -1.0, dot(prev, curr) < 0.0);
    let q = mix(prev * flip, curr, t);
    let len = length(q);
    return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), q / len, len > 1e-12);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let i = eids[1u + d];
    let a = interp.alpha;
    let p = mix(bInitL(i), bPos(i), a);          // prev = x⁻ (pre-warmstart), curr = solved pose
    let q = nlerpShortest(bInitQ(i), bQuat(i), a);
    // render scale maps the unit mesh to the body's shape (Phase 6.3): the cube/sphere meshes are unit
    // half-extent 0.5 → scale 2·extent; the capsule mesh is y∈[-1,1] (half-extents 0.5,1,0.5), so its
    // bounding box (2r, hc+r, 2r) maps with y-scale hc+r (the caps distort under a non-proportional
    // ratio — a fixed-mesh limitation, render-only; the collider is exact).
    let shape = bShape(i);
    let radius = bRadius(i);
    var s = bHalf(i) * 2.0;
    if (shape == 1u) {
        s = vec3<f32>(2.0 * radius, 2.0 * radius, 2.0 * radius);
    } else if (shape == 2u) {
        s = vec3<f32>(2.0 * radius, bHalf(i).y + radius, 2.0 * radius);
    }
    transforms[i] = Xform(p, q, s);
}
`;

// ── velocity: BDF1 recovery (solver.cpp step 5). prevVel updated for the next adaptive warmstart ──
const VELOCITY_PASS_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> step: Step;
@group(0) @binding(2) var<storage, read> eids: array<u32>;
${BODY_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= eids[0u]) { return; }
    let i = eids[1u + d];
    let cap = step.eidCap;
    bodies[B_PREVV*cap + i] = vec4<f32>(bVelL(i), 0.0); // prevVel = vel
    if (solverStatic(i)) { return; }                    // static / kinematic — keep the frozen velocity
    let velL = (bPos(i) - bInitL(i)) / step.dt;
    let velA = qSubW(bQuat(i), bInitQ(i)) / step.dt;
    bodies[B_VELL*cap + i] = vec4<f32>(velL, 0.0);
    bodies[B_VELA*cap + i] = vec4<f32>(velA, 0.0);
}
`;

// ── CSR adjacency: per-body contact lists, so the primal + coloring read only a body's own contacts ──
// From the persistent `pairContacts` (the collide wrote this frame's manifolds in place), build a
// compressed-sparse-row index keyed by eid (the Part-pack count→scan→scatter spine), packed into ONE `csr`
// buffer: `csr[eidCap+eid]` = contacts per body, `csr[eid]` = the start of its slice in `csrList`,
// `csrList[off .. off+count]` the MANIFOLD-RECORD indices touching that body (the primal reads `cc(csrList[k])`).
// Offsets + counts share one binding (Phase 4.9) so the maxed primal/coloring passes bind one slot, not two.
// Each active record lands in BOTH its bodies' slices, so a body reads every contact it's in (the primal's
// `contactContrib` picks the body's Jacobian). This is the O(count·contacts) → O(valence) collapse. count +
// scatter run one thread per pair SLOT (looping the slot's CONTACTS_PER_PAIR records, skipping inactive), so
// they scan only the live blocks; the scan (resetting count to the scatter cursor) is the single-workgroup
// parallel prefix sum.

const CSR_COUNT_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> pairContacts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> csr: array<atomic<u32>>; // counts in [eidCap, 2·eidCap)
@group(0) @binding(2) var<uniform> step: Step;
@group(0) @binding(3) var<storage, read> eids: array<u32>; // [0] = live count, [1+d] = the d-th live eid
${CONTACT_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    let d = lane / PAIRS_PER_BODY;
    if (d >= eids[0u]) { return; } // past the live body count
    let recBase = (eids[1u + d] * PAIRS_PER_BODY + (lane % PAIRS_PER_BODY)) * CONTACTS_PER_PAIR;
    for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
        let m = cc(recBase + ls, C_META);
        if (bitcast<u32>(m.x) != ${CONSTRAINT_CONTACT}u) { continue; } // inactive record
        atomicAdd(&csr[step.eidCap + bitcast<u32>(m.y)], 1u); // body a
        atomicAdd(&csr[step.eidCap + bitcast<u32>(m.z)], 1u); // body b
    }
}
`;

// exclusive prefix sum over the live (dense) eids → each body's csrList slice start; resets the count region to 0
// so the scatter can reuse it as the per-body append cursor (the part-pack scan shape). Single-workgroup
// parallel scan (Phase 4.7, folding in 4.9's scan-parallelization): thread t owns dense slots
// [t*CSR_CHUNK, …) within the live count, sums its chunk's per-body counts, the PACK_WG chunk-sums are
// scanned on lane 0, then each thread lays its bodies' offsets from its chunk base. Replaces the prior
// `@workgroup_size(1)` serial prefix sum (the O(N) cost at scale — 33 ms @ 40 960 in the capacity probe).
function csrScanWgsl(maxBodies: number, eidCap: number): string {
    const chunk = Math.ceil(maxBodies / PACK_WG);
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read> eids: array<u32>;
@group(0) @binding(1) var<storage, read_write> csr: array<u32>; // [0,eidCap) offsets, [eidCap,2·eidCap) counts
const PACK_WG: u32 = ${PACK_WG}u;
const CSR_CHUNK: u32 = ${chunk}u;
const CSR_COUNT_BASE: u32 = ${eidCap}u; // the count region base (no step uniform bound in this pass)
var<workgroup> csrSum: array<u32, ${PACK_WG}>;
@compute @workgroup_size(${PACK_WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let n = eids[0u];
    let t = lid.x;
    let lo = t * CSR_CHUNK;
    let hi = min(lo + CSR_CHUNK, n); // only live dense slots

    // phase 1: sum this chunk's per-body contact counts (the chunk total, in dense order)
    var sum = 0u;
    for (var d = lo; d < hi; d = d + 1u) {
        sum = sum + csr[CSR_COUNT_BASE + eids[1u + d]];
    }
    csrSum[t] = sum;
    workgroupBarrier();

    // phase 2: exclusive prefix over the PACK_WG chunk-sums (serial on lane 0) → each thread's chunk base
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < PACK_WG; i = i + 1u) {
            let c = csrSum[i];
            csrSum[i] = acc;
            acc = acc + c;
        }
    }
    workgroupBarrier();

    // phase 3: lay each body's exclusive offset from the chunk base + reset its count (the scatter cursor)
    var acc = csrSum[t];
    for (var d = lo; d < hi; d = d + 1u) {
        let e = eids[1u + d];
        csr[e] = acc;
        acc = acc + csr[CSR_COUNT_BASE + e];
        csr[CSR_COUNT_BASE + e] = 0u;
    }
}
`;
}

const CSR_SCATTER_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> pairContacts: array<vec4<f32>>;
// CSR in one binding: [0,eidCap) offsets (read via atomicLoad — same binding, atomic type), [eidCap,…) append cursor
@group(0) @binding(1) var<storage, read_write> csr: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> csrList: array<u32>;
@group(0) @binding(3) var<uniform> step: Step;
@group(0) @binding(4) var<storage, read> eids: array<u32>; // [0] = live count, [1+d] = the d-th live eid
${CONTACT_WGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lane = gid.x;
    let d = lane / PAIRS_PER_BODY;
    if (d >= eids[0u]) { return; } // past the live body count
    let recBase = (eids[1u + d] * PAIRS_PER_BODY + (lane % PAIRS_PER_BODY)) * CONTACTS_PER_PAIR;
    for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
        let ci = recBase + ls;
        let m = cc(ci, C_META);
        if (bitcast<u32>(m.x) != ${CONSTRAINT_CONTACT}u) { continue; } // inactive record
        let a = bitcast<u32>(m.y);
        let b = bitcast<u32>(m.z);
        csrList[atomicLoad(&csr[a]) + atomicAdd(&csr[step.eidCap + a], 1u)] = ci;
        csrList[atomicLoad(&csr[b]) + atomicAdd(&csr[step.eidCap + b], 1u)] = ci;
    }
}
`;

// the single-workgroup scan width for the pack + CSR scan: one workgroup of PACK_WG lanes scans the whole
// eid/body range, each lane owning a contiguous chunk (the O(N) work parallel across the chunks, the
// per-chunk-sum reduction serial over PACK_WG). 256 covers capacity ≤ 65536 at chunk ≤ 256.
const PACK_WG = 256;

// ── fused small-N tail (C1.1): CSR count→scan→scatter + greedy coloring, ONE single-WG dispatch ──
// In the small-N regime the step is structure-tax-bound: the 3 CSR
// dispatches + the count-region clear + the color-snapshot copy + the colorize dispatch are ~6 dependent
// phase boundaries around ~µs of work. This kernel runs the same phases under in-kernel barriers
// (0.09 µs each on Lovelace vs ~1 µs per dispatch boundary, 0.56 vs 4.08 on Metal) in one dispatch.
// Each phase is the multi-WG pass's logic verbatim, strided over PACK_WG lanes (correct at any N —
// only slow past the threshold, so the frame-stale regime switch stays correctness-safe):
//   clear   — live eids' counts only (dead eids' stale counts are never read: counts are written for
//             live contacts' bodies and read for live bodies, so the full-region clearBuffer is excess)
//   count   — CSR_COUNT: each active record increments both bodies' counts
//   scan    — csrScan with a LIVE-count-dynamic chunk (the standalone pass chunks over maxBodies, so at
//             1k live in a 65536 pool only 4 of its 256 lanes work; chunking over n uses all of them)
//   scatter — CSR_SCATTER: append each active record into both bodies' slices
//   greedy  — COLORING_PASS with INVERTED staging: priors are read from `colors` (untouched this step)
//             and the chosen color staged in `colorScratch`, committed after a barrier. The multi-WG
//             pass needs the prior snapshot COPY because its workgroups race on `colors`; a single WG
//             orders the read phase before the commit phase with one barrier, deleting the 256 KB copy.
// `csr` binds as atomic throughout (the scan uses atomicLoad/Store — same memory, one binding type).
// 10 storage buffers — at the binding floor like the primal (physics.md "phase ladder"): a new
// constraint type must reuse the merged adjacency, never add a binding here.
const CSR_COLOR_SMALL_WGSL =
    SHARED_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> pairContacts: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> csr: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> csrList: array<u32>;
@group(0) @binding(4) var<storage, read_write> colors: array<u32>;
@group(0) @binding(5) var<storage, read_write> colorScratch: array<u32>;
@group(0) @binding(6) var<uniform> step: Step;
@group(0) @binding(7) var<storage, read> eids: array<u32>;
@group(0) @binding(8) var<storage, read_write> colorCount: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read> constraintCsr: array<u32>;
@group(0) @binding(10) var<storage, read> constraintList: array<vec4<f32>>;
const PACK_WG: u32 = ${PACK_WG}u;
${BODY_WGSL}
${CONTACT_WGSL}
var<workgroup> wgN: u32;
var<workgroup> csrSum: array<u32, ${PACK_WG}>;
var<workgroup> wgMax: atomic<u32>; // max dynamic color + 1 (workgroup memory zero-inits per dispatch)
@compute @workgroup_size(${PACK_WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    // every lane reaches every barrier (no early returns — idle lanes skip strided loops), and the live
    // count flows through workgroupUniformLoad for Tint's uniformity analysis (the small-broadphase pattern).
    if (lid.x == 0u) { wgN = eids[0u]; }
    let n = workgroupUniformLoad(&wgN);
    let t = lid.x;

    // clear: live eids' counts (the only counts the scan reads — see the header)
    for (var d = t; d < n; d = d + PACK_WG) {
        atomicStore(&csr[step.eidCap + eids[1u + d]], 0u);
    }
    storageBarrier();

    // count
    let slots = n * PAIRS_PER_BODY;
    for (var s = t; s < slots; s = s + PACK_WG) {
        let recBase = (eids[1u + s / PAIRS_PER_BODY] * PAIRS_PER_BODY + (s % PAIRS_PER_BODY)) * CONTACTS_PER_PAIR;
        for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
            let m = cc(recBase + ls, C_META);
            if (bitcast<u32>(m.x) != ${CONSTRAINT_CONTACT}u) { continue; } // inactive record
            atomicAdd(&csr[step.eidCap + bitcast<u32>(m.y)], 1u); // body a
            atomicAdd(&csr[step.eidCap + bitcast<u32>(m.z)], 1u); // body b
        }
    }
    storageBarrier();

    // scan: exclusive prefix over the live counts → offsets; counts reset to 0 (the scatter cursor)
    let chunk = (n + PACK_WG - 1u) / PACK_WG;
    let lo = t * chunk;
    let hi = min(lo + chunk, n);
    var sum = 0u;
    for (var d = lo; d < hi; d = d + 1u) {
        sum = sum + atomicLoad(&csr[step.eidCap + eids[1u + d]]);
    }
    csrSum[t] = sum;
    workgroupBarrier();
    if (t == 0u) {
        var acc0 = 0u;
        for (var i = 0u; i < PACK_WG; i = i + 1u) {
            let c = csrSum[i];
            csrSum[i] = acc0;
            acc0 = acc0 + c;
        }
    }
    workgroupBarrier();
    var acc = csrSum[t];
    for (var d = lo; d < hi; d = d + 1u) {
        let e = eids[1u + d];
        atomicStore(&csr[e], acc);
        acc = acc + atomicLoad(&csr[step.eidCap + e]);
        atomicStore(&csr[step.eidCap + e], 0u);
    }
    storageBarrier();

    // scatter
    for (var s = t; s < slots; s = s + PACK_WG) {
        let recBase = (eids[1u + s / PAIRS_PER_BODY] * PAIRS_PER_BODY + (s % PAIRS_PER_BODY)) * CONTACTS_PER_PAIR;
        for (var ls = 0u; ls < CONTACTS_PER_PAIR; ls = ls + 1u) {
            let ci = recBase + ls;
            let m = cc(ci, C_META);
            if (bitcast<u32>(m.x) != ${CONSTRAINT_CONTACT}u) { continue; } // inactive record
            let a = bitcast<u32>(m.y);
            let b = bitcast<u32>(m.z);
            csrList[atomicLoad(&csr[a]) + atomicAdd(&csr[step.eidCap + a], 1u)] = ci;
            csrList[atomicLoad(&csr[b]) + atomicAdd(&csr[step.eidCap + b], 1u)] = ci;
        }
    }
    storageBarrier();

    // greedy: the incremental coloring (COLORING_PASS_WGSL's logic), priors read from colors (untouched
    // this step), chosen staged in colorScratch so every lane's prior reads complete before any commit
    for (var d = t; d < n; d = d + PACK_WG) {
        let bid = eids[1u + d];
        if (bMass(bid) <= 0.0) { colorScratch[bid] = 0xffffffffu; continue; } // static — uncolored
        let colorsN = max(1u, min(step.maxColors, 32u));
        let clo = atomicLoad(&csr[bid]);
        let chi = clo + atomicLoad(&csr[step.eidCap + bid]);
        var usedMask = 0u;
        for (var k = clo; k < chi; k = k + 1u) {
            let m = cc(csrList[k], C_META);
            let a = bitcast<u32>(m.y);
            let b = bitcast<u32>(m.z);
            var other = a;                          // every CSR contact touches bid; pick the neighbor
            if (a == bid) { other = b; }
            if (other <= bid) { continue; }         // higher-id symmetry break — no atomics
            if (bMass(other) <= 0.0) { continue; }  // static neighbor: no scheduling constraint
            let pc = colors[other];
            if (pc < 32u) { usedMask = usedMask | (1u << pc); }
        }
        let slo = constraintCsr[bid];
        let shi = slo + constraintCsr[step.eidCap + bid];
        for (var e = slo; e < shi; e = e + 1u) {
            let other = bitcast<u32>(constraintList[e * CONSTRAINT_VEC4 + 2u].x);
            if (other <= bid) { continue; }
            if (bMass(other) <= 0.0) { continue; }
            let pc = colors[other];
            if (pc < 32u) { usedMask = usedMask | (1u << pc); }
        }
        var chosen = colors[bid];                   // incremental: keep the prior color when still free
        var needsNew = chosen >= colorsN;
        if (!needsNew) { needsNew = (usedMask & (1u << chosen)) != 0u; }
        if (needsNew) {
            var found = false;
            for (var c = 0u; c < colorsN; c = c + 1u) {
                if ((usedMask & (1u << c)) == 0u) { chosen = c; found = true; break; }
            }
            if (!found) { chosen = bid % colorsN; } // fold past the cap — a tolerated same-color conflict
        }
        colorScratch[bid] = chosen;
        atomicMax(&wgMax, chosen + 1u);
    }
    storageBarrier();
    workgroupBarrier();

    // commit the staged colors + publish the used-color count (word 0; word 1 is packScan's live count)
    for (var d = t; d < n; d = d + PACK_WG) {
        let bid = eids[1u + d];
        colors[bid] = colorScratch[bid];
    }
    if (t == 0u) { atomicStore(&colorCount[0], atomicLoad(&wgMax)); }
}
`;

// ── pack: GPU membership-scan → the dense→eid map (the Part-pack firehose) ──
// One lane per eid over scene capacity, gated on the Body membership bit (the mirror the Part pack
// reads). FULLY GPU — no CPU entity iteration, not even a marker query. Each live eid does two things:
//   • PACK — a deterministic eid-sorted compaction into `eids` (`eids[0]` = the live count, `eids[1+d]`
//     = the d-th live eid — the dense→eid map every body pass reads as `i = eids[1+gid.x]`). Sorted
//     order keeps a per-body pair block's slot stable across frames (the persistent warmstart's
//     precondition); an atomic-append (arbitrary order) would shuffle them every frame.
//   • SEED (one-time) — a GPU `seeded` flag (per eid) gates the copy of the authored Body slabs
//     (pos/quat/half/mass/friction, GPU-mirrored via slab.gpu) into `bodies[*][eid]` + the moment derive
//     + velocity zero, then sets the flag. Existing bodies' slots are untouched (a mid-sim spawn never
//     disturbs the settled pile). A non-member eid resets its flag, so a recycled eid re-seeds.
// `eidCap`/`maxBodies`/`gen`/`mask` are baked (construction constants — no step uniform binding needed).
// `packScan` bounds `eids[0]` to the pool + publishes it to the BVH + the dispatch args; overflow drops
// the map write loudly.
//
// The compaction is MULTI-WORKGROUP (C1.3): count → scan → scatter over PACK_WG-eid chunks, one lane per
// eid. The prior single-WG form had each lane serially walk eidCap/PACK_WG eids — an O(capacity) span on
// one SM (~130 µs/frame at the default 65536 capacity, regardless of live count). Three dispatches cost
// ~2 extra phase boundaries (~2 µs); the serial walk they delete cost two orders more. packCount also
// carries the one-time seed/reset (per-eid work, parallel). packScan is a single small WG over the
// per-workgroup sums (numWgs = capacity/PACK_WG elements); packScatter recomputes each workgroup's local
// membership prefix (cheaper than storing per-lane bases) and writes the sorted slots — chunks are
// disjoint and eid-ordered, so the output is bit-identical to the single-WG form.
interface PackGate {
    /** the Body component's membership word index + bit mask (`state.membership.bit(Body)`) */
    gen: number;
    mask: number;
}

function packCountWgsl(gate: PackGate, eidCap: number): string {
    return (
        BODY_COLS_WGSL +
        /* wgsl */ `
@group(0) @binding(0) var<storage, read> membership: array<u32>;
@group(0) @binding(1) var<storage, read_write> packSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> seeded: array<u32>;
@group(0) @binding(3) var<storage, read> srcPos: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> srcQuat: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> srcHalf: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> srcMass: array<f32>;
@group(0) @binding(7) var<storage, read> srcFriction: array<f32>;
@group(0) @binding(8) var<storage, read> srcShape: array<u32>;
@group(0) @binding(9) var<storage, read_write> bodies: array<vec4<f32>>;
const EID_CAP: u32 = ${eidCap}u;
const PACK_WG: u32 = ${PACK_WG}u;
fn isMember(eid: u32) -> bool {
    return (membership[${gate.gen}u * EID_CAP + eid] & ${gate.mask}u) != 0u;
}
// one-time seed: copy authored slabs → eid slot, derive the per-shape moment of inertia (rigid.ts box /
// sphere / capsule), zero vel. The rounding radius rides the authored halfExtents.w lane (Phase 6.3, the
// SoA shape-geometry grouping — core extents + radius read together); B_ROUND carries (shape, radius).
fn seedBody(eid: u32) {
    let p = srcPos[eid].xyz;
    let q = srcQuat[eid];
    let h = srcHalf[eid].xyz;      // core half-extents (box / hull AABB widths / capsule segment; 0 for sphere)
    let m = srcMass[eid];
    let fr = srcFriction[eid];
    let shape = srcShape[eid];
    // halfExtents.w doubles as the rounding radius (sphere/capsule) OR the hull id (a hull has radius 0,
    // so the lane is free) — the SoA shape-geometry grouping (Phase 6.3). Box: radius 0, id 0.
    var radius = srcHalf[eid].w;
    var hullId = 0u;
    if (shape == 3u) { hullId = u32(srcHalf[eid].w + 0.5); radius = 0.0; }
    bodies[B_POS*EID_CAP + eid] = vec4<f32>(p, 0.0);
    bodies[B_QUAT*EID_CAP + eid] = q;
    bodies[B_INERTL*EID_CAP + eid] = vec4<f32>(p, 0.0);
    bodies[B_INERTQ*EID_CAP + eid] = q;
    bodies[B_INITL*EID_CAP + eid] = vec4<f32>(p, 0.0);
    bodies[B_INITQ*EID_CAP + eid] = q;
    bodies[B_VELL*EID_CAP + eid] = vec4<f32>(0.0);
    bodies[B_VELA*EID_CAP + eid] = vec4<f32>(0.0);
    bodies[B_PREVV*EID_CAP + eid] = vec4<f32>(0.0);
    var moment: vec3<f32>;
    if (shape == 1u) {            // sphere — (2/5)·m·r²
        let i = 0.4 * m * radius * radius;
        moment = vec3<f32>(i, i, i);
    } else if (shape == 2u) {     // capsule (core along local Y): cylinder + 2 hemispheres, mass split by
                                  // volume (rigid.ts capsuleMoment; PI cancels in the ratio so it's PI-free)
        let hc = h.y;
        let L = 2.0 * hc;
        let mc = m * L / (L + (4.0 / 3.0) * radius);  // cylinder mass fraction
        let ms = m - mc;                              // two-hemisphere mass fraction
        let r2 = radius * radius;
        let iy = mc * 0.5 * r2 + ms * 0.4 * r2;
        let iPerp = mc * (L * L / 12.0 + r2 * 0.25)
                  + ms * (0.4 * r2 + L * L * 0.25 + 0.375 * L * radius);
        moment = vec3<f32>(iPerp, iy, iPerp);
    } else {                      // box — solid-box diagonal from full widths
        let s = h * 2.0;
        moment = vec3<f32>(
            ((s.y*s.y + s.z*s.z) / 12.0) * m,
            ((s.x*s.x + s.z*s.z) / 12.0) * m,
            ((s.x*s.x + s.y*s.y) / 12.0) * m);
    }
    bodies[B_MM*EID_CAP + eid] = vec4<f32>(moment, m);
    bodies[B_HF*EID_CAP + eid] = vec4<f32>(h, fr);
    bodies[B_ROUND*EID_CAP + eid] = vec4<f32>(bitcast<f32>(shape), radius, bitcast<f32>(hullId), 0.0);
}
// lane ↔ eid (workgroup wid owns eids [wid·PACK_WG, …)): count the workgroup's live members + one-time
// seed (members) + reset seeded (non-members, so a recycled eid re-seeds). The per-WG total feeds packScan.
var<workgroup> packSum: array<u32, ${PACK_WG}>;
@compute @workgroup_size(${PACK_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let eid = wid.x * PACK_WG + t;
    var cnt = 0u;
    if (eid < EID_CAP) {
        if (isMember(eid)) {
            cnt = 1u;
            if (seeded[eid] == 0u) { seeded[eid] = 1u; seedBody(eid); }
        } else {
            seeded[eid] = 0u;
        }
    }
    packSum[t] = cnt;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < PACK_WG; i = i + 1u) { acc = acc + packSum[i]; }
        packSums[wid.x] = acc;
    }
}
`
    );
}

// exclusive prefix over the per-workgroup sums (single small WG — numWgs = capacity/PACK_WG elements,
// the CSR-scan shape: lane t sums its chunk, lane 0 scans the chunk-sums, lanes write back exclusive
// bases in place). Lane 0 also CLAMPS + PUBLISHES (the prior standalone clampCount, fused here — it only
// ever ran behind this scan, so fusing deletes a phase boundary): bound the live count to the body pool
// (overflow bumps the loud counters[5], never a silent drop), copy it to the BVH prim count + the
// colorCount readback (word 1 feeds boundBodies — the direct color-loop dispatch, rung 0), and write the
// indirect dispatch args — body passes = ceil(count/64), the per-eid-block passes (collide / dual / CSR)
// = ceil(count·PAIRS_PER_BODY/64) (lane → d → owner eid → slot = eid·K + k).
function packScanWgsl(numWgs: number, maxBodies: number): string {
    const chunk = Math.ceil(numWgs / PACK_WG);
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> packSums: array<u32>;
@group(0) @binding(1) var<storage, read_write> eids: array<u32>;
@group(0) @binding(2) var<storage, read_write> bvhCount: array<u32>;
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> dispatchArgs: array<u32>;
@group(0) @binding(5) var<storage, read_write> pairArgs: array<u32>;
@group(0) @binding(6) var<storage, read_write> colorCount: array<u32>;
const NUM_WGS: u32 = ${numWgs}u;
const PACK_WG: u32 = ${PACK_WG}u;
const CHUNK: u32 = ${chunk}u;
const MAX_BODIES: u32 = ${maxBodies}u;
var<workgroup> scanSum: array<u32, ${PACK_WG}>;
@compute @workgroup_size(${PACK_WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let lo = t * CHUNK;
    let hi = min(lo + CHUNK, NUM_WGS);
    var cnt = 0u;
    for (var i = lo; i < hi; i = i + 1u) { cnt = cnt + packSums[i]; }
    scanSum[t] = cnt;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < PACK_WG; i = i + 1u) {
            let c = scanSum[i];
            scanSum[i] = acc;
            acc = acc + c;
        }
        if (acc > MAX_BODIES) { atomicAdd(&counters[5], acc - MAX_BODIES); }
        let clamped = min(acc, MAX_BODIES);
        eids[0u] = clamped;
        bvhCount[0u] = clamped;
        colorCount[1u] = clamped;
        dispatchArgs[0u] = (clamped + 63u) / 64u;
        dispatchArgs[1u] = 1u;
        dispatchArgs[2u] = 1u;
        pairArgs[0u] = (clamped * ${PAIRS_PER_BODY}u + 63u) / 64u;
        pairArgs[1u] = 1u;
        pairArgs[2u] = 1u;
    }
    workgroupBarrier();
    var acc = scanSum[t];
    for (var i = lo; i < hi; i = i + 1u) {
        let c = packSums[i];
        packSums[i] = acc;
        acc = acc + c;
    }
}
`;
}

// scatter each member into its sorted dense slot: recompute the workgroup-local membership prefix
// (cheaper than a per-lane base buffer), base = the workgroup's scanned exclusive sum. Chunks are
// disjoint + eid-ordered, so slots never overlap and the dense order is sorted by eid — a per-body pair
// block keeps its slot across frames (the stable per-pair slot the persistent warmstart needs).
function packScatterWgsl(gate: PackGate, eidCap: number, maxBodies: number): string {
    return /* wgsl */ `
@group(0) @binding(0) var<storage, read> membership: array<u32>;
@group(0) @binding(1) var<storage, read> packSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> eids: array<u32>;
const EID_CAP: u32 = ${eidCap}u;
const PACK_WG: u32 = ${PACK_WG}u;
const MAX_BODIES: u32 = ${maxBodies}u;
fn isMember(eid: u32) -> bool {
    return (membership[${gate.gen}u * EID_CAP + eid] & ${gate.mask}u) != 0u;
}
var<workgroup> packSum: array<u32, ${PACK_WG}>;
@compute @workgroup_size(${PACK_WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let eid = wid.x * PACK_WG + t;
    var m = 0u;
    if (eid < EID_CAP && isMember(eid)) { m = 1u; }
    packSum[t] = m;
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var i = 0u; i < PACK_WG; i = i + 1u) {
            let c = packSum[i];
            packSum[i] = acc;
            acc = acc + c;
        }
    }
    workgroupBarrier();
    if (m == 1u) {
        let slot = packSums[wid.x] + packSum[t];
        if (slot < MAX_BODIES) { eids[1u + slot] = eid; }
    }
}
`;
}

/** the membership + authored slab sources the GPU pack (compaction + one-time seed) gathers from */
export interface Inputs {
    membership: GPUBuffer;
    pos: GPUBuffer;
    quat: GPUBuffer;
    half: GPUBuffer; // halfExtents slab: .xyz core extents, .w the rounding radius (Phase 6.3)
    mass: GPUBuffer;
    friction: GPUBuffer;
    shape: GPUBuffer; // ShapeKind per eid (u32): 0 box, 1 sphere, 2 capsule
}

/**
 * an authored spring (Phase 6.1, the soft `Force`): a body-body distance constraint `C = ‖pA − pB‖ − rest`,
 * force `f = stiffness·C`. `a`/`b` are body eids, `rA`/`rB` the anchors in each body's local frame.
 */
export interface SpringDef {
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    stiffness: number;
    rest: number;
}

/** the {@link JointDef} `a` sentinel for a WORLD anchor (no body A): `rA` is then a world-space point and `b`
 *  dangles freely from it (no anchor body → no anchor↔b contact). The mouse-drag grab (avbd-demo3d
 *  `bodyA == null`); move the world point each frame via {@link PhysicsStep.setJointAnchor}. */
export const WORLD = -1;
const WORLD_ANCHOR_U32 = 0xffffffff; // the GPU sentinel (WGSL WORLD_ANCHOR) a < 0 maps to in the joint record

/**
 * an authored joint (Phase 6.2, the hard `Force`): two stacked constraints pinning `b`'s anchor `rB` to
 * `a`'s anchor `rA` — a linear anchor pin + an angular relative-orientation lock. `a`/`b` are body eids,
 * `rA`/`rB` the anchors in each body's local frame. **`a = {@link WORLD}` (any `a < 0`) makes `rA` a
 * world-space point** with no body A — the constraint pins `b` to a fixed world anchor with no anchor body
 * (so no anchor↔b contact), `b` dangling freely; drive the point each frame with {@link PhysicsStep.setJointAnchor}.
 * Defaults match `joint()`/`Joint::Joint`: rigid linear (`stiffnessLin = ∞`) + free rotation
 * (`stiffnessAng = 0`) = a spherical joint; pass `stiffnessAng = Infinity` for a fixed joint. Two
 * construction-time guards reject an energy-injecting authoring (deactivate, the GPU analog of `joint()`'s
 * throw): the two endpoints must NOT both be non-dynamic (`mass ≤ 0`, the world counting as static) — a joint
 * no dynamic body can resolve ramps its dual penalty + λ unbounded (a `counters[1]` bump) — and the anchors
 * MUST start coincident at the scene pose, a gross mismatch injecting energy through BDF1 recovery (the rope
 * explosion, a `counters[2]` bump). Joint one dynamic body to a static/kinematic/world anchor. The torque arm
 * `‖sizeA + sizeB‖²` is GPU-computed from the bodies' half-extents.
 */
export interface JointDef {
    /** body-A eid, OR {@link WORLD} (`< 0`) for a world anchor (`rA` is then a world-space point, no body) */
    a: number;
    b: number;
    rA: readonly [number, number, number];
    rB: readonly [number, number, number];
    /** ∞ (default) = rigid linear (adds the C −= α·C₀ stabilization); a finite value = a soft linear joint */
    stiffnessLin?: number;
    /** 0 (default) = spherical (rotation free); ∞ = fixed (orientation locked) */
    stiffnessAng?: number;
    /**
     * a 1-DOF force-clamped **angular motor** (avbd-demo2d motor.cpp): drives `b`'s orientation relative to
     * `a` at `speed` rad/s about `axis`, the angular torque clamped to ±`maxTorque`. Unlike forcing
     * {@link PhysicsStep.setAngularVelocity} (consumed once by the inertial prediction), the motor competes
     * inside every solver iteration, so it HOLDS the target ω under a load up to `maxTorque` and yields past
     * it. Independent of `stiffnessAng` (a spherical joint still motors). Drive `speed` live with
     * {@link PhysicsStep.setMotor}. Absent ⇒ no motor.
     */
    motor?: {
        /** unit world axis the drive acts about */
        axis: readonly [number, number, number];
        /** target rad/s of `b` relative to `a` about `axis` (a world anchor `a` ⇒ `b` spins at `speed`) */
        speed: number;
        /** |angular torque| clamp — holds the target ω under load up to this, yields past it */
        maxTorque: number;
    };
}

// 256-byte minimum dynamic-uniform-offset alignment (the colorUbo stride)
const UBO_ALIGN = 256;

interface Pass {
    pipeline: GPUComputePipeline;
    layout: GPUBindGroupLayout;
}

async function buildPass(
    device: GPUDevice,
    label: string,
    code: string,
    entries: GPUBindGroupLayoutEntry[],
): Promise<Pass> {
    const layout = device.createBindGroupLayout({ label, entries });
    const pipeline = await device.createComputePipelineAsync({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    return { pipeline, layout };
}

const ro: GPUBufferBindingType = "read-only-storage";
const rw: GPUBufferBindingType = "storage";
const uni = (hasDynamicOffset = false): GPUBufferBindingLayout => ({
    type: "uniform",
    hasDynamicOffset,
});
const buf = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
});

// the four narrowphase pipelines (box / rounded / hull / rounded-poly) share ONE bind-group layout + bind
// group; each compiles only its own shape-pair SAT chunk (the DXC pipeline split, see `collidePass`). The
// four `createComputePipelineAsync` calls run concurrently, so the wall-clock compile ≈ the largest single
// chunk — the hull SAT + the rounded segment-clip split apart so neither is the old combined long pole.
async function buildCollide(
    device: GPUDevice,
): Promise<{ layout: GPUBindGroupLayout; pipelines: GPUComputePipeline[] }> {
    const layout = device.createBindGroupLayout({
        label: "phys-collide",
        entries: [
            buf(0, ro),
            buf(1, rw),
            buf(2, rw),
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
            buf(4, ro),
            buf(5, ro),
            buf(6, ro),
        ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const make = (label: string, code: string): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
            label,
            layout: pipelineLayout,
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });
    // box first — it's the lifecycle owner, dispatched first each step (see record()). The class gates are
    // mutually exclusive, so the fill order of the other three (rounded / hull / rounded-poly) doesn't matter.
    const pipelines = await Promise.all([
        make("phys-collide-box", COLLIDE_BOX_WGSL),
        make("phys-collide-rounded", COLLIDE_ROUNDED_WGSL),
        make("phys-collide-hull", COLLIDE_HULL_WGSL),
        make("phys-collide-rounded-poly", COLLIDE_ROUNDEDPOLY_WGSL),
    ]);
    return { layout, pipelines };
}

/**
 * the GPU AVBD step pipeline. Solver state is eid-indexed over `eidCap` (the scene capacity), persistent
 * across frames; the live bodies are compacted into the dense `eids` map (`eids[0]` = count, `eids[1+d]`
 * = the d-th live eid) by the GPU `pack` (plugin) or `gateSetCount` (the standalone gates). `maxBodies`
 * is the live-body bound (BVH prims, the eid map, dispatch cap); the per-eid manifold store sizes by
 * `eidCap` (each body owns a fixed block at `eid · PAIRS_PER_BODY`, Phase 4.9 robustness).
 * Records one full step onto a command encoder; the plugin and the standalone oracle cross-check drive it.
 */
export class PhysicsStep {
    readonly device: GPUDevice;
    /** eid space + bodies/colors/pair-block stride (= the scene ECS capacity) */
    readonly eidCap: number;
    /** the live-body bound — BVH prims, the eid map, the body-pass dispatch cap; = `eidCap` (Phase 4.7). */
    readonly maxBodies: number;
    /** the per-eid pair-block pool size = `eidCap * PAIRS_PER_BODY` — `pairList` + the manifold store both
     * size by the eid space (body eid owns the fixed block `[eid·PAIRS_PER_BODY, …)`, Phase 4.9 robustness) */
    readonly maxPairSlots: number;
    /** persistent contact-record pool size = `maxPairSlots * CONTACTS_PER_PAIR` (records in `pairContacts`) */
    readonly recordCap: number;
    readonly bodies: GPUBuffer;
    /** the colored-primal double-buffer scratch: each color's solved poses, committed into `bodies` per
     * color by the commit pass (Phase 4.5 Stage B) — `solveOut[col*eidCap + eid]`, col 0 pos / 1 quat */
    readonly solveOut: GPUBuffer;
    /** persistent per-eid manifold store (Phase 4.9 robustness) — ONE buffer holding both this frame's
     * contacts AND cross-frame λ/k. The owner eid's fixed slot s (= eid·PAIRS_PER_BODY + k) owns records
     * [s*CONTACTS_PER_PAIR, …); the collide writes/warmstarts in place at the slot, keyed by (a,b)+feature
     * (no hash, no separate store). The slot is a function of the owner's eid alone, so it's stable across
     * frames unless that body's candidate set flickers (local warmstart fragility — webphysics's model, not
     * the global churn a prefix-sum compaction had). SoA `pairContacts[col*recordCap + rec]`. */
    readonly pairContacts: GPUBuffer;
    readonly counters: GPUBuffer;
    /** the per-eid pair blocks the broadphase writes + the narrowphase reads: slot `eid·PAIRS_PER_BODY + k`
     * holds (aEid, bEid) oriented a > b (or INVALID for an unused slot). The owner eid's fixed block — the
     * stable warmstart address (Phase 4.9 robustness), `pairContacts[slot]` persists across frames in place. */
    readonly pairList: GPUBuffer;
    /** the per-eid-block passes' indirect dispatch args `[ceil(liveCount·PAIRS_PER_BODY/64), 1, 1]` — written
     * by packScan / gateSetCount; collide / dual / CSR dispatch off it (lane → d → owner eid → slot). */
    readonly pairArgs: GPUBuffer;
    readonly colors: GPUBuffer;
    /** prior-frame color snapshot the incremental-greedy `colorize` reads (snapshotted by a copy each step) */
    readonly colorScratch: GPUBuffer;
    /** the bounded color loop's readback words: `colorCount[0]` = the used-color count this step (max
     * dynamic color + 1, written by `colorize`), `colorCount[1]` = the clamped live body count (written by
     * `packScan`). A consumer Mirrors the buffer (frame-stale) and feeds `boundColors` + `boundBodies`,
     * which cap the color-passes at `min(maxColors, usedColors + COLOR_MARGIN)` (Phase 4.9 Lever 1) and
     * size their direct dispatch off the live count + BODY_MARGIN (rung 0). */
    readonly colorCount: GPUBuffer;
    /** the dense→eid map: `eids[0]` = live count, `eids[1+d]` = the d-th live eid (the pack's output) */
    readonly eids: GPUBuffer;
    /** per-eid one-time-seed flag (GPU-resident): the pack seeds an eid's slot once, then sets this */
    readonly seeded: GPUBuffer;
    // per-workgroup live-member sums (packCount writes, packScan scans in place, packScatter reads) —
    // one u32 per PACK_WG-eid chunk of the eid space
    private readonly _packSums: GPUBuffer;
    // the pack's workgroup count: ceil(eidCap / PACK_WG)
    private readonly _packWgs: number;
    /** packed convex-hull geometry (`./hull` packHulls), read by the collide pass for ShapeKind.Hull bodies
     * indexed by `bHullId`. Re-uploaded by `setHulls` whenever the `Hulls` registry changes; a 1-u32 stub
     * until then (a box/sphere/capsule-only scene never indexes it). Grows on demand (createBuffer). */
    hullData: GPUBuffer;
    /** CSR adjacency in one buffer (Phase 4.9): per-body csrList slice start in `[0, eidCap)`, per-body
     * contact count in `[eidCap, 2·eidCap)`. Merged so the maxed primal pass (and coloring) bind one slot
     * not two — the primal/coloring read a body's contacts as `csrList[csr[bid] .. +csr[eidCap+bid]]`. */
    readonly csr: GPUBuffer;
    /** the CSR contact-index lists, all bodies' slices packed (each contact in both its bodies' slices) */
    readonly csrList: GPUBuffer;
    /** authored-constraint adjacency (springs Phase 6.1 + joints Phase 6.2 — set by `setSprings`/`setJoints`):
     * per-body offsets in [0,eidCap), counts in [eidCap, 2·eidCap), into `constraintList` (entry units).
     * Zero-init ⇒ a constraint-less scene no-ops. */
    readonly constraintCsr: GPUBuffer;
    /** the inline per-body constraint entries (`CONSTRAINT_VEC4` vec4 each, AoS, kind-tagged): each authored
     * constraint appears in both endpoints' slices, carrying that endpoint's anchor + the partner. Grows on demand. */
    constraintList: GPUBuffer;
    /** per-joint records (`JOINT_REC_VEC4` vec4 each, AoS — Phase 6.2): the hard `Force`'s persistent λ/penalty/
     * c0/active + geometry + recycle versions, indexed by a joint's `recordIndex`. Grows on demand. Zero joints ⇒
     * the primal binds a valid 1-joint buffer (step.jointCount 0 ⇒ the joint passes are skipped). */
    jointRecords: GPUBuffer;
    /** per-eid recycle version (Phase 6.2, project_stable_identity) — the opt-in `(eid, version)` side array a
     * joint validates against; a version-mismatched joint deactivates. Zero-init; bumped via `recycleVersion`. */
    readonly jointVersions: GPUBuffer;
    /** constraintList capacity in entries — grown (with a solve-bind-group rebuild) when set* exceeds it */
    private _constraintCap: number;
    /** jointRecords capacity in joints — grown (with a solve-bind-group rebuild) when setJoints exceeds it */
    private _jointCap: number;
    /** the authored springs + joints (re-merged into constraintCsr/List on either set*) + the joint count */
    private _springDefs: readonly SpringDef[] = [];
    private _jointDefs: readonly JointDef[] = [];
    private _jointCount = 0;
    /** CPU mirror of `jointVersions` — setJoints stamps each record's versions from it, recycleVersion bumps it */
    private readonly _versions: Uint32Array;
    /** indirect body-pass dispatch args `[ceil(count/64), 1, 1]` — written by packScan / gateSetCount */
    readonly dispatchArgs: GPUBuffer;
    private readonly _stepUbo: GPUBuffer;
    private readonly _colorUbo: GPUBuffer;
    /** the render-interpolation alpha (= time.fixedAlpha) the compose pass blends prev→curr by; rewritten
     * each compose call (the one per-render-frame uniform — distinct from the per-step `_stepUbo.alpha`) */
    private readonly _interpUbo: GPUBuffer;
    private readonly _interpData = new Float32Array(1);
    /** the configured fixed timestep (set by {@link configure}) — {@link setKinematic} derives a platform's
     * velocity from its per-step pose delta over this dt. NOTE: this stays the FULL fixed dt; the GPU
     * uniform's `dt` field carries the SUB-STEP `h = dt/_substeps` (configure), so a kinematic platform
     * still moves once per fixed step while the rigid passes integrate the sub-step. */
    private _dt = 1 / 60;
    /** sub-steps per fixed step (Macklin small-steps) — `record` loops the per-sub-step passes this many
     * times at `h = dt/_substeps`. 1 = the single-sub-step path (byte-identical). Set by {@link configure}. */
    private _substeps = 1;
    /** the configured world gravity (set by {@link configure}) — the per-character gravity default when a
     * `Character.gravity` of 0 means "the world default" (the CPU character sweep reads it via {@link gravity}) */
    private _gravity = -10;
    /** the configured world gravity — the CPU character sweep reads it to resolve a per-character `gravity` 0
     * (= the world default), and `dt` to integrate the sweep on the same fixed clock the solver uses. */
    get gravity(): number {
        return this._gravity;
    }
    get dt(): number {
        return this._dt;
    }
    /** last pose per kinematic eid (px,py,pz,qx,qy,qz,qw) — {@link setKinematic} differences against it for the
     * body's velocity. Persists for the step's life (freed with it), bounded by the few distinct kinematic
     * eids. A despawned-then-reused eid keeps a stale prev → one frame of spurious velocity on its next
     * setKinematic, so re-placing a kinematic body should pass `teleport` (the grab does on pickup). */
    private readonly _kinPrev = new Map<number, Float32Array>();
    private readonly _kinScratch = new Float32Array(4); // reused per-column write (B_POS / B_QUAT / B_VELL)
    private readonly _anchorScratch = new Float32Array(3); // reused per-frame world-anchor write (setJointAnchor)
    /** the shared LBVH builder over body sphere-AABBs — the broadphase acceleration structure */
    private readonly _bvh: Bvh;
    private readonly _aabb: Pass;
    private readonly _broadphase: Pass;
    private readonly _broadphaseSmall: Pass;
    // the narrowphase is four pipelines (box / rounded / hull / rounded-poly) sharing one bind-group layout + bind group,
    // each compiling only its shape-pair SAT chunk (the DXC pipeline split). Dispatched in order each step.
    private readonly _collideLayout: GPUBindGroupLayout;
    private readonly _collidePipelines: GPUComputePipeline[];
    private readonly _inertial: Pass;
    private readonly _primal: Pass;
    /** the per-color commit: applies `solveOut` → `bodies` for the current color (the double-buffer write) */
    private readonly _commit: Pass;
    private readonly _dual: Pass;
    private readonly _solveLds: Pass;
    private readonly _coloring: Pass;
    /** the joint hard-conflict coloring repair (Phase 6.2), run JOINT_REPAIR_ROUNDS times after the greedy */
    private readonly _repair: Pass;
    /** per-joint warmstart + C₀ capture (Phase 6.2), before the main loop */
    private readonly _jointInit: Pass;
    /** per-joint dual update (Phase 6.2), after each iteration's primal */
    private readonly _jointDual: Pass;
    private readonly _velocity: Pass;
    private readonly _compose: Pass;
    private readonly _csrCount: Pass;
    private readonly _csrScan: Pass;
    private readonly _csrScatter: Pass;
    /** the fused small-N CSR + coloring tail (C1.1) — replaces the 3 CSR passes + colorize in the small regime */
    private readonly _csrColorSmall: Pass;
    // the pack (membership-scan compaction + one-time seed; count → scan → scatter, C1.3) is plugin-only
    // (membership-gated / slab-sourced); null for the standalone gates, which set `eids` via
    // `gateSetCount` + seed `bodies` by `writeBuffer` directly.
    private readonly _packCount: Pass | null;
    private readonly _packScan: Pass | null;
    private readonly _packScatter: Pass | null;
    private readonly _aabbBG: GPUBindGroup;
    private readonly _broadphaseBG: GPUBindGroup;
    private readonly _broadphaseSmallBG: GPUBindGroup;
    // rebuilt by `setHulls` when the hullData buffer grows (the `!` = assigned via `_makeCollideBG`)
    private _collideBG!: GPUBindGroup;
    private readonly _inertialBG: GPUBindGroup;
    // the solve block (coloring / primal / commit) binds the dense `eids` map in its dense-map slot — the
    // iters × colors solve iterates every live body. primal + coloring bind the growable constraintList, so
    // they're (re)built by buildSolveBindGroups (the `!`); commit binds fixed buffers (constructor, `readonly`).
    private _primalBG!: GPUBindGroup;
    private _solveLdsBG!: GPUBindGroup;
    private readonly _commitBG: GPUBindGroup;
    private readonly _dualBG: GPUBindGroup;
    private _coloringBG!: GPUBindGroup;
    // the repair + joint passes bind the growable constraintList / jointRecords, so they're (re)built by
    // buildSolveBindGroups alongside primal + coloring (the `!` = assigned via that helper)
    private _repairBG!: GPUBindGroup;
    private _csrColorSmallBG!: GPUBindGroup;
    private _jointInitBG!: GPUBindGroup;
    private _jointDualBG!: GPUBindGroup;
    private readonly _velocityBG: GPUBindGroup;
    private readonly _csrCountBG: GPUBindGroup;
    private readonly _csrScanBG: GPUBindGroup;
    private readonly _csrScatterBG: GPUBindGroup;
    // built lazily on the first pack call (the slab .gpu + membership buffers aren't allocated at warm —
    // parallel warms); rebuilt if a source identity changes (stable in practice — allocated once)
    private _packCountBG: GPUBindGroup | null = null;
    private _packScanBG: GPUBindGroup | null = null;
    private _packScatterBG: GPUBindGroup | null = null;
    private _gatherInputs: Inputs | null = null;
    // built lazily on the first compose call (needs the external firehose buffer); rebuilt if its
    // identity changes (it doesn't in practice — TransformsPlugin allocates it once)
    private _composeBG: GPUBindGroup | null = null;
    private _composeDst: GPUBuffer | null = null;
    private _iterations = 10;
    // dispatched-color cap: the primal dispatches `_maxColors` color-passes per iteration (empty colors
    // no-op via the early-out), the coloring folds bodies past it (scratch.md "Dispatch count"). 32 = no cap.
    private _maxColors = COLOR_CAP;
    // the color-passes the primal actually dispatches per iteration (Phase 4.9 Lever 1, readback-bounded
    // color loop). Defaults to `_maxColors` (the static cap — what `record` dispatches with no readback, so
    // the standalone gates that drive `record` directly keep full dispatch). `boundColors`, fed a frame-stale
    // `usedColors` readback by the plugin, lowers it to `min(_maxColors, usedColors + COLOR_MARGIN)`.
    private _colorsToRun = COLOR_CAP;
    // the full body pool's workgroup count `ceil(maxBodies/64)` — the color loop's cold-start dispatch.
    // Set in the constructor (needs maxBodies).
    private _fullGroups = 0;
    // the color loop's direct-dispatch workgroup count (rung 0). Defaults to `_fullGroups` (what `record`
    // dispatches with no readback — the standalone gates' path, all-early-out past `eids[0]`). `boundBodies`,
    // fed the frame-stale `colorCount[1]` readback by the plugin, lowers it to the live count +
    // BODY_MARGIN's workgroups.
    private _bodyGroups = 0;
    // the frame-stale live count the broadphase regime keys on (`boundBodies` / `gateSetCount` set it; 0 =
    // unknown → the BVH path, the safe cold-start). NOT reset by `configure`: both regimes are exact at any
    // N (the O(n²) path is only slow past the threshold), so a stale value costs at most a few slow-path
    // frames until the next readback — never correctness. That two-sided safety is what lets a frame-stale
    // signal pick the path at all.
    private _liveBound = 0;
    // the small-N regime threshold (StepParams.smallN; 0 = always BVH, the bench's A/B lever)
    private _smallN = SMALL_N;
    private _smallRan = false;
    private _ldsRan = false;
    // the LDS-resident solve threshold (StepParams.ldsN; 0 = always the looped color passes)
    private _ldsN = LDS_N;
    // set by `cold()` (the gates, between scenes); the next `record` clears pairContacts before the collide
    // (so a new scene's slots don't read the prior scene's records). The plugin never sets it — a fresh
    // PhysicsStep's pairContacts is zero-init (all kind 0) so it cold-starts naturally.
    private _coldNext = false;

    private constructor(
        device: GPUDevice,
        eidCap: number,
        maxBodies: number,
        bvh: Bvh,
        passes: {
            aabb: Pass;
            broadphase: Pass;
            broadphaseSmall: Pass;
            collide: { layout: GPUBindGroupLayout; pipelines: GPUComputePipeline[] };
            inertial: Pass;
            primal: Pass;
            commit: Pass;
            dual: Pass;
            solveLds: Pass;
            coloring: Pass;
            repair: Pass;
            jointInit: Pass;
            jointDual: Pass;
            velocity: Pass;
            compose: Pass;
            csrCount: Pass;
            csrScan: Pass;
            csrScatter: Pass;
            csrColorSmall: Pass;
            packCount: Pass | null;
            packScan: Pass | null;
            packScatter: Pass | null;
        },
    ) {
        this.device = device;
        this.eidCap = eidCap;
        this.maxBodies = maxBodies;
        this._fullGroups = Math.ceil(maxBodies / 64);
        this._bodyGroups = this._fullGroups;
        // the manifold store is the largest single binding; fail loud + clear if the eid space × the per-body
        // block exceeds the device's per-binding limit. acquireDevice requests the adapter's full limit, so
        // this clears at a realistic 65536 capacity (235 MB).
        checkContactStore(eidCap, device.limits.maxStorageBufferBindingSize);
        // per-eid FIXED blocks (Phase 4.9 robustness): `pairList` + the manifold store both size by the eid
        // space — body eid owns the fixed block `[eid·PAIRS_PER_BODY, …)`. The base is the owner's eid alone,
        // so a flicker churns only that body's slots (local warmstart fragility, webphysics's model), not the
        // global collapse a prefix-sum compaction had. 235 MB at 65536 (the same as the prior compaction).
        this.maxPairSlots = eidCap * PAIRS_PER_BODY;
        this.recordCap = this.maxPairSlots * CONTACTS_PER_PAIR;
        this._bvh = bvh;
        const store = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const storeSrc = store | GPUBufferUsage.COPY_SRC;
        // eid-indexed, persistent across frames — a body's solver state lives at its eid slot.
        this.bodies = device.createBuffer({
            label: "phys-bodies",
            size: eidCap * BODY_VEC4 * 16,
            usage: storeSrc,
        });
        // the colored-primal scratch (Phase 4.5 Stage B): the primal writes here, the commit applies it
        // into `bodies` per color. Never read before written for a given (color, body), so no clear needed.
        this.solveOut = device.createBuffer({
            label: "phys-solve-out",
            size: eidCap * SOLVE_VEC4 * 16,
            usage: GPUBufferUsage.STORAGE,
        });
        // ONE persistent per-eid manifold store (Phase 4.9 robustness): recordCap records, SoA
        // `pairContacts[col*recordCap + rec]`. Holds both this frame's contacts AND cross-frame λ/k — the
        // collide writes/warmstarts in place at the owner eid's fixed slot (no hash, no separate warmCache,
        // no cache pass). COPY_DST so `cold()` can clearBuffer it between scenes; COPY_SRC for a merge-crux readback.
        this.pairContacts = device.createBuffer({
            label: "phys-paircontacts",
            size: this.recordCap * CONTACT_VEC4 * 16,
            usage: storeSrc,
        });
        // counter slots (u32). Two feed the GPU correctness gates: [0] active contact count, [6] warmstarted
        // contacts. Two are joint construction guards: [1] both-endpoints-static rejected (a persistent
        // per-frame gauge), [2] anchor-coincidence rejected. The rest are fail-loudly guards (0 in any real
        // scene; a non-zero count localizes a dropped support): [3] per-body descent-block overflow (the
        // graceful nearest-K prune), [5] body-pool overflow, [7] static-support pair dropped DESPITE the pin,
        // [8] character candidate overflow (> the 64 lane cap), [9] character displacement guard (travel
        // exceeded the cull band's budget). [4] is unused (kept for index stability). 16 slots = 64 B.
        this.counters = device.createBuffer({ label: "phys-counters", size: 64, usage: storeSrc });
        // the per-eid pair blocks: vec2<u32> (aEid, bEid) per slot at `eid·PAIRS_PER_BODY + k`, written by
        // the broadphase (each live body owns its block, no atomics) + read by collide/dual/CSR. Sized by the
        // eid space — the fixed per-body base is the stable warmstart address (Phase 4.9 robustness).
        this.pairList = device.createBuffer({
            label: "phys-pairlist",
            size: this.maxPairSlots * 8,
            usage: store,
        });
        // the per-eid-block passes' indirect dispatch args [ceil(liveCount·PAIRS_PER_BODY/64), 1, 1], written
        // by packScan / gateSetCount. INDIRECT for dispatchWorkgroupsIndirect; the early-out reads eids[0].
        this.pairArgs = device.createBuffer({
            label: "phys-pairargs",
            size: 12,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // COPY_SRC so `colorize` can snapshot colors → colorScratch (and the crux test reads it back)
        this.colors = device.createBuffer({
            label: "phys-colors",
            size: eidCap * 4,
            usage: storeSrc,
        });
        this.colorScratch = device.createBuffer({
            label: "phys-color-scratch",
            size: eidCap * 4,
            usage: storeSrc,
        });
        // the used-color count (`colorCount[0]`), the readback-bounded color loop's input (Phase 4.9 Lever 1).
        // 16 B (one used slot + padding) — STORAGE for the colorize atomicMax, COPY_DST for the per-step clear,
        // COPY_SRC so a consumer can Mirror it for the frame-stale readback that feeds `boundColors`.
        this.colorCount = device.createBuffer({
            label: "phys-color-count",
            size: 16,
            usage: storeSrc,
        });
        // bootstrap coloring = the cold sentinel (0xffffffff = "needs a fresh color") over the whole
        // capacity — seed once. The incremental-greedy `colorize` reads it as its first prior-frame
        // snapshot: cold ⇒ every body takes the lowest free color, so the coloring is COMPACT (colors dense
        // from 0, max = the chromatic number − 1) and a sparse scene colors to ~1-2. An eid-identity seed
        // is NOT compact — the keep-prior reuse retains each conflict-free body's scattered eid-as-color, so
        // the used-color *range* never collapses and the readback-bounded color loop (boundColors) can't save
        // (the saving needs `usedColors` = the chromatic number). Matches the CPU twin / measured spec
        // (coloring.ts colorSweep, seed 0xffffffff). Incremental reuse owns it after.
        device.queue.writeBuffer(this.colors, 0, new Uint32Array(eidCap).fill(0xffffffff));
        // the dense→eid map [count, eid0, eid1, ...] + the seed work-list, both (1 + maxBodies) u32.
        // COPY_SRC so the gym mirrors `eids`; COPY_DST so the gates set it / record clears eids[0].
        this.eids = device.createBuffer({
            label: "phys-eids",
            size: (1 + maxBodies) * 4,
            usage: storeSrc,
        });
        // per-eid one-time-seed flag, zero-init (fresh = unseeded). The pack sets it on first seed +
        // resets it when a body leaves, so a recycled eid re-seeds. Plugin-only (the gates seed directly).
        this.seeded = device.createBuffer({
            label: "phys-seeded",
            size: eidCap * 4,
            usage: store,
        });
        // per-workgroup pack sums (C1.3) — one u32 per PACK_WG-eid chunk
        this._packWgs = Math.ceil(eidCap / PACK_WG);
        this._packSums = device.createBuffer({
            label: "phys-pack-sums",
            size: this._packWgs * 4,
            usage: store,
        });
        // packed convex-hull geometry — a 1-u32 stub until `setHulls` uploads the registry (Phase 6.3).
        this.hullData = device.createBuffer({ label: "phys-hulls", size: 4, usage: store });
        // CSR adjacency in one buffer: offsets in [0, eidCap), per-body counts in [eidCap, 2·eidCap) — merged
        // to free a binding in the maxed primal pass (the list holds 2 entries per contact — both bodies).
        this.csr = device.createBuffer({
            label: "phys-csr",
            size: 2 * eidCap * 4,
            usage: store,
        });
        // each active record lands in both endpoints' slices → 2 entries per record (worst case all active)
        this.csrList = device.createBuffer({
            label: "phys-csr-list",
            size: this.recordCap * 2 * 4,
            usage: store,
        });
        // authored-constraint adjacency (springs Phase 6.1 + joints Phase 6.2). constraintCsr is eid-indexed
        // (offsets + counts), zero-init so a constraint-less scene reads count 0 and the primal/coloring loops
        // no-op. constraintList holds the inline kind-tagged entries (CONSTRAINT_VEC4 vec4 each); it starts
        // small + grows on demand (set*) — a fresh step with no constraints still binds a valid 1-block buffer
        // so the layout is uniform across scenes.
        this.constraintCsr = device.createBuffer({
            label: "phys-constraint-csr",
            size: 2 * eidCap * 4,
            usage: store,
        });
        // entries; grows (reallocate + rebuild the solve bind groups) when a set* exceeds it. Small initial
        // cap so a real scene grows once at load (a trivial realloc) and the gym scenes trip it, exercising
        // the grow + bind-group-rebuild path.
        this._constraintCap = 4;
        this.constraintList = device.createBuffer({
            label: "phys-constraint-list",
            size: this._constraintCap * CONSTRAINT_VEC4 * 16,
            usage: store,
        });
        // per-joint records (Phase 6.2) — the hard `Force`'s persistent λ/penalty/c0/active + geometry +
        // versions, indexed by recordIndex. Grows on demand (setJoints); a fresh step binds a valid 1-joint buffer.
        this._jointCap = 4;
        // COPY_SRC: the grow path copies live records into the replacement buffer (kept-slot state)
        this.jointRecords = device.createBuffer({
            label: "phys-joint-records",
            size: this._jointCap * JOINT_REC_VEC4 * 16,
            usage: store | GPUBufferUsage.COPY_SRC,
        });
        // per-eid recycle version (Phase 6.2) — the opt-in side array a joint validates against; zero-init,
        // bumped via recycleVersion. The CPU mirror lets setJoints stamp each record's versions from it.
        this.jointVersions = device.createBuffer({
            label: "phys-joint-versions",
            size: eidCap * 4,
            usage: store,
        });
        this._versions = new Uint32Array(eidCap);
        // indirect body-pass dispatch args `[wgX, 1, 1]` — written by packScan (plugin) / gateSetCount (gates)
        this.dispatchArgs = device.createBuffer({
            label: "phys-dispatch-args",
            size: 12,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._stepUbo = device.createBuffer({
            label: "phys-step",
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // one 256-byte slot per dispatched color (slot c holds u32 c, read at dynamic offset c·256 by
        // the primal). COLOR_CAP slots is exact — `maxColors` clamps to it, so c never exceeds it.
        this._colorUbo = device.createBuffer({
            label: "phys-color-ubo",
            size: COLOR_CAP * UBO_ALIGN,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._interpUbo = device.createBuffer({
            label: "phys-interp",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const colorSeed = new Uint32Array((COLOR_CAP * UBO_ALIGN) / 4);
        for (let c = 0; c < COLOR_CAP; c++) colorSeed[(c * UBO_ALIGN) / 4] = c;
        device.queue.writeBuffer(this._colorUbo, 0, colorSeed);

        this._aabb = passes.aabb;
        this._broadphase = passes.broadphase;
        this._broadphaseSmall = passes.broadphaseSmall;
        this._collideLayout = passes.collide.layout;
        this._collidePipelines = passes.collide.pipelines;
        this._inertial = passes.inertial;
        this._primal = passes.primal;
        this._commit = passes.commit;
        this._dual = passes.dual;
        this._solveLds = passes.solveLds;
        this._coloring = passes.coloring;
        this._repair = passes.repair;
        this._jointInit = passes.jointInit;
        this._jointDual = passes.jointDual;
        this._velocity = passes.velocity;
        this._compose = passes.compose;
        this._csrCount = passes.csrCount;
        this._csrScan = passes.csrScan;
        this._csrScatter = passes.csrScatter;
        this._csrColorSmall = passes.csrColorSmall;
        this._packCount = passes.packCount;
        this._packScan = passes.packScan;
        this._packScatter = passes.packScatter;

        this._aabbBG = device.createBindGroup({
            label: "phys-aabb",
            layout: this._aabb.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: bvh.prims } },
                { binding: 2, resource: { buffer: this._stepUbo } },
                { binding: 3, resource: { buffer: this.eids } },
            ],
        });
        this._broadphaseBG = device.createBindGroup({
            label: "phys-broadphase",
            layout: this._broadphase.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: bvh.nodes } },
                { binding: 2, resource: { buffer: this.pairList } },
                { binding: 3, resource: { buffer: this.counters } },
                { binding: 4, resource: { buffer: this._stepUbo } },
                { binding: 5, resource: { buffer: this.eids } },
            ],
        });
        this._broadphaseSmallBG = device.createBindGroup({
            label: "phys-broadphase-small",
            layout: this._broadphaseSmall.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: bvh.prims } },
                { binding: 2, resource: { buffer: this.pairList } },
                { binding: 3, resource: { buffer: this.counters } },
                { binding: 4, resource: { buffer: this._stepUbo } },
                { binding: 5, resource: { buffer: this.eids } },
            ],
        });
        this._collideBG = this._makeCollideBG();
        this._inertialBG = device.createBindGroup({
            label: "phys-inertial",
            layout: this._inertial.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this._stepUbo } },
                { binding: 2, resource: { buffer: this.eids } },
            ],
        });
        // the commit binds the dense `eids` map in the dense-map slot — the solve iterates every live body.
        this._commitBG = device.createBindGroup({
            label: "phys-commit",
            layout: this._commit.layout,
            entries: [
                { binding: 0, resource: { buffer: this.solveOut } },
                { binding: 1, resource: { buffer: this.colors } },
                { binding: 2, resource: { buffer: this.bodies } },
                { binding: 3, resource: { buffer: this.eids } },
                { binding: 4, resource: { buffer: this._stepUbo } },
                { binding: 5, resource: { buffer: this._colorUbo, size: 16 } },
            ],
        });
        this._dualBG = device.createBindGroup({
            label: "phys-dual",
            layout: this._dual.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this._stepUbo } },
                { binding: 3, resource: { buffer: this.eids } },
            ],
        });
        this._velocityBG = device.createBindGroup({
            label: "phys-velocity",
            layout: this._velocity.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this._stepUbo } },
                { binding: 2, resource: { buffer: this.eids } },
            ],
        });
        this._csrCountBG = device.createBindGroup({
            label: "phys-csr-count",
            layout: this._csrCount.layout,
            entries: [
                { binding: 0, resource: { buffer: this.pairContacts } },
                { binding: 1, resource: { buffer: this.csr } },
                { binding: 2, resource: { buffer: this._stepUbo } },
                { binding: 3, resource: { buffer: this.eids } },
            ],
        });
        this._csrScanBG = device.createBindGroup({
            label: "phys-csr-scan",
            layout: this._csrScan.layout,
            entries: [
                { binding: 0, resource: { buffer: this.eids } },
                { binding: 1, resource: { buffer: this.csr } },
            ],
        });
        this._csrScatterBG = device.createBindGroup({
            label: "phys-csr-scatter",
            layout: this._csrScatter.layout,
            entries: [
                { binding: 0, resource: { buffer: this.pairContacts } },
                { binding: 1, resource: { buffer: this.csr } },
                { binding: 2, resource: { buffer: this.csrList } },
                { binding: 3, resource: { buffer: this._stepUbo } },
                { binding: 4, resource: { buffer: this.eids } },
            ],
        });
        // the solve bind groups last — they bind the growable constraintList / jointRecords, so a set* that grows
        // either rebuilds them (primal + coloring + csrColorSmall + repair + jointInit + jointDual)
        this.buildSolveBindGroups();
    }

    // (re)build the bind groups that reference the growable constraintList / jointRecords (primal, coloring,
    // csrColorSmall, repair, jointInit, jointDual) — from the constructor and again whenever a set* reallocates either past
    // its cap. The primal binds jointRecords (binding 11) — 10 storage buffers, exactly the floor.
    private _makeCollideBG(): GPUBindGroup {
        return this.device.createBindGroup({
            label: "phys-collide",
            layout: this._collideLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this.counters } },
                { binding: 3, resource: { buffer: this._stepUbo } },
                { binding: 4, resource: { buffer: this.pairList } },
                { binding: 5, resource: { buffer: this.eids } },
                { binding: 6, resource: { buffer: this.hullData } },
            ],
        });
    }

    /**
     * Upload the packed convex-hull geometry (`./hull` packHulls) the collide pass reads for ShapeKind.Hull
     * bodies. The buffer grows on demand (rebuilding the collide bind group). Idempotent — a no-op when the
     * data is unchanged is the caller's job (the plugin uploads only when the `Hulls` registry changes).
     */
    setHulls(data: Uint32Array): void {
        const bytes = Math.max(4, data.byteLength);
        if (bytes > this.hullData.size) {
            this.hullData.destroy();
            this.hullData = this.device.createBuffer({
                label: "phys-hulls",
                size: bytes,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._collideBG = this._makeCollideBG();
        }
        this.device.queue.writeBuffer(this.hullData, 0, data as Uint32Array<ArrayBuffer>);
    }

    private buildSolveBindGroups(): void {
        // the primal binds the dense `eids` map in the dense-map slot (7) — the looped color solve iterates
        // every live body.
        this._primalBG = this.device.createBindGroup({
            label: "phys-primal",
            layout: this._primal.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this.csr } },
                { binding: 3, resource: { buffer: this.csrList } },
                { binding: 4, resource: { buffer: this.colors } },
                { binding: 5, resource: { buffer: this._stepUbo } },
                { binding: 6, resource: { buffer: this._colorUbo, size: 16 } },
                { binding: 7, resource: { buffer: this.eids } },
                { binding: 8, resource: { buffer: this.solveOut } },
                { binding: 9, resource: { buffer: this.constraintCsr } },
                { binding: 10, resource: { buffer: this.constraintList } },
                { binding: 11, resource: { buffer: this.jointRecords } },
            ],
        });
        this._solveLdsBG = this.device.createBindGroup({
            label: "phys-solve-lds",
            layout: this._solveLds.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this.csr } },
                { binding: 3, resource: { buffer: this.csrList } },
                { binding: 4, resource: { buffer: this.colors } },
                { binding: 5, resource: { buffer: this._stepUbo } },
                { binding: 6, resource: { buffer: this.eids } },
                // the eid → dense map rides the solveOut scratch (unused by the LDS path; 2·eidCap
                // vec4s ≥ the eidCap u32s the map needs), rebound as array<u32> — the kernel is at the
                // 10-storage-binding floor, so the map can't be a new buffer without evicting one
                { binding: 7, resource: { buffer: this.solveOut } },
                { binding: 8, resource: { buffer: this.constraintCsr } },
                { binding: 9, resource: { buffer: this.constraintList } },
                { binding: 10, resource: { buffer: this.jointRecords } },
            ],
        });
        // the coloring binds the dense `eids` map in the dense-map slot (7): it colors every live body.
        this._coloringBG = this.device.createBindGroup({
            label: "phys-coloring",
            layout: this._coloring.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this.csr } },
                { binding: 3, resource: { buffer: this.csrList } },
                { binding: 4, resource: { buffer: this.colors } },
                { binding: 5, resource: { buffer: this.colorScratch } },
                { binding: 6, resource: { buffer: this._stepUbo } },
                { binding: 7, resource: { buffer: this.eids } },
                { binding: 8, resource: { buffer: this.colorCount } },
                { binding: 9, resource: { buffer: this.constraintCsr } },
                { binding: 10, resource: { buffer: this.constraintList } },
            ],
        });
        this._csrColorSmallBG = this.device.createBindGroup({
            label: "phys-csr-color-small",
            layout: this._csrColorSmall.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.pairContacts } },
                { binding: 2, resource: { buffer: this.csr } },
                { binding: 3, resource: { buffer: this.csrList } },
                { binding: 4, resource: { buffer: this.colors } },
                { binding: 5, resource: { buffer: this.colorScratch } },
                { binding: 6, resource: { buffer: this._stepUbo } },
                { binding: 7, resource: { buffer: this.eids } },
                { binding: 8, resource: { buffer: this.colorCount } },
                { binding: 9, resource: { buffer: this.constraintCsr } },
                { binding: 10, resource: { buffer: this.constraintList } },
            ],
        });
        this._repairBG = this.device.createBindGroup({
            label: "phys-repair",
            layout: this._repair.layout,
            entries: [
                { binding: 0, resource: { buffer: this.bodies } },
                { binding: 1, resource: { buffer: this.colors } },
                { binding: 2, resource: { buffer: this.colorScratch } },
                { binding: 3, resource: { buffer: this.constraintCsr } },
                { binding: 4, resource: { buffer: this.constraintList } },
                { binding: 5, resource: { buffer: this._stepUbo } },
                { binding: 6, resource: { buffer: this.eids } },
                { binding: 7, resource: { buffer: this.colorCount } },
            ],
        });
        this._jointInitBG = this.device.createBindGroup({
            label: "phys-joint-init",
            layout: this._jointInit.layout,
            entries: [
                { binding: 0, resource: { buffer: this.jointRecords } },
                { binding: 1, resource: { buffer: this.bodies } },
                { binding: 2, resource: { buffer: this.jointVersions } },
                { binding: 3, resource: { buffer: this._stepUbo } },
                { binding: 4, resource: { buffer: this.counters } },
                { binding: 5, resource: { buffer: this.seeded } },
            ],
        });
        this._jointDualBG = this.device.createBindGroup({
            label: "phys-joint-dual",
            layout: this._jointDual.layout,
            entries: [
                { binding: 0, resource: { buffer: this.jointRecords } },
                { binding: 1, resource: { buffer: this.bodies } },
                { binding: 2, resource: { buffer: this._stepUbo } },
            ],
        });
    }

    /**
     * compile the step pipelines + allocate the buffers. `eidCap` = the eid space (bodies/colors/pair-block
     * stride, = the scene ECS capacity); `maxBodies` = the live-body bound (the eid map + dispatch cap).
     * Pass a `packGate` (the Body membership coordinates) to build the GPU pack + seed passes — the plugin
     * does; the standalone gates omit it (so pack/seed are null) and drive the dense map via `gateSetCount`
     * + seed `bodies` directly.
     */
    static async create(
        device: GPUDevice,
        eidCap: number,
        maxBodies: number,
        packGate?: PackGate,
    ): Promise<PhysicsStep> {
        // the broadphase BVH over body sphere-AABBs — sized to the body pool, one prim per live body
        const bvh = await createBvh(device, maxBodies);
        const [
            aabb,
            broadphase,
            broadphaseSmall,
            collide,
            inertial,
            primal,
            commit,
            dual,
            solveLds,
            coloring,
            repair,
            jointInit,
            jointDual,
            velocity,
            compose,
            csrCount,
            csrScan,
            csrScatter,
            csrColorSmall,
            packCount,
            packScan,
            packScatter,
        ] = await Promise.all([
            buildPass(device, "phys-aabb", AABB_PASS_WGSL, [
                buf(0, ro),
                buf(1, rw),
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(3, ro),
            ]),
            buildPass(device, "phys-broadphase", BROADPHASE_PASS_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, rw),
                buf(3, rw),
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(5, ro),
            ]),
            buildPass(device, "phys-broadphase-small", BROADPHASE_SMALL_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, rw),
                buf(3, rw),
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(5, ro),
            ]),
            buildCollide(device),
            buildPass(device, "phys-inertial", INERTIAL_PASS_WGSL, [
                buf(0, rw),
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(2, ro),
            ]),
            buildPass(device, "phys-primal", PRIMAL_PASS_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, ro),
                buf(3, ro),
                buf(4, ro),
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: uni(true) },
                buf(7, ro),
                buf(8, rw),
                buf(9, ro),
                buf(10, ro),
                buf(11, ro),
            ]),
            buildPass(device, "phys-commit", COMMIT_PASS_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, rw),
                buf(3, ro),
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: uni(true) },
            ]),
            buildPass(device, "phys-dual", DUAL_PASS_WGSL, [
                buf(0, ro),
                buf(1, rw),
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(3, ro),
            ]),
            buildPass(device, "phys-solve-lds", SOLVE_LDS_WGSL, [
                buf(0, rw),
                buf(1, rw),
                buf(2, ro),
                buf(3, ro),
                buf(4, ro),
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(6, ro),
                buf(7, rw),
                buf(8, ro),
                buf(9, ro),
                buf(10, rw),
            ]),
            buildPass(device, "phys-coloring", COLORING_PASS_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, ro),
                buf(3, ro),
                buf(4, rw),
                buf(5, ro),
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(7, ro),
                buf(8, rw),
                buf(9, ro),
                buf(10, ro),
            ]),
            buildPass(device, "phys-repair", REPAIR_PASS_WGSL, [
                buf(0, ro),
                buf(1, rw),
                buf(2, ro),
                buf(3, ro),
                buf(4, ro),
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(6, ro),
                buf(7, rw),
            ]),
            buildPass(device, "phys-joint-init", JOINT_INIT_PASS_WGSL, [
                buf(0, rw),
                buf(1, ro),
                buf(2, ro),
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(4, rw),
                buf(5, ro),
            ]),
            buildPass(device, "phys-joint-dual", JOINT_DUAL_PASS_WGSL, [
                buf(0, rw),
                buf(1, ro),
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
            ]),
            buildPass(device, "phys-velocity", VELOCITY_PASS_WGSL, [
                buf(0, rw),
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(2, ro),
            ]),
            buildPass(device, "phys-compose", COMPOSE_PASS_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, rw),
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
            ]),
            buildPass(device, "phys-csr-count", CSR_COUNT_WGSL, [
                buf(0, ro),
                buf(1, rw),
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(3, ro),
            ]),
            buildPass(device, "phys-csr-scan", csrScanWgsl(maxBodies, eidCap), [
                buf(0, ro),
                buf(1, rw),
            ]),
            buildPass(device, "phys-csr-scatter", CSR_SCATTER_WGSL, [
                buf(0, ro),
                buf(1, rw),
                buf(2, rw),
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(4, ro),
            ]),
            buildPass(device, "phys-csr-color-small", CSR_COLOR_SMALL_WGSL, [
                buf(0, ro),
                buf(1, ro),
                buf(2, rw),
                buf(3, rw),
                buf(4, rw),
                buf(5, rw),
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: uni() },
                buf(7, ro),
                buf(8, rw),
                buf(9, ro),
                buf(10, ro),
            ]),
            packGate
                ? buildPass(device, "phys-pack-count", packCountWgsl(packGate, eidCap), [
                      buf(0, ro),
                      buf(1, rw), // packSums
                      buf(2, rw),
                      buf(3, ro),
                      buf(4, ro),
                      buf(5, ro),
                      buf(6, ro),
                      buf(7, ro),
                      buf(8, ro), // srcShape (Phase 6.3)
                      buf(9, rw), // bodies
                  ])
                : Promise.resolve(null),
            packGate
                ? buildPass(
                      device,
                      "phys-pack-scan",
                      packScanWgsl(Math.ceil(eidCap / PACK_WG), maxBodies),
                      [
                          buf(0, rw),
                          buf(1, rw),
                          buf(2, rw),
                          buf(3, rw),
                          buf(4, rw),
                          buf(5, rw),
                          buf(6, rw),
                      ],
                  )
                : Promise.resolve(null),
            packGate
                ? buildPass(
                      device,
                      "phys-pack-scatter",
                      packScatterWgsl(packGate, eidCap, maxBodies),
                      [buf(0, ro), buf(1, ro), buf(2, rw)],
                  )
                : Promise.resolve(null),
        ]);
        return new PhysicsStep(device, eidCap, maxBodies, bvh, {
            aabb,
            broadphase,
            broadphaseSmall,
            collide,
            inertial,
            primal,
            commit,
            dual,
            solveLds,
            coloring,
            repair,
            jointInit,
            jointDual,
            velocity,
            compose,
            csrCount,
            csrScan,
            csrScatter,
            csrColorSmall,
            packCount,
            packScan,
            packScatter,
        });
    }

    /** set the per-step constants (the step uniform). The live body count is GPU-resident (`eids[0]`). */
    configure(p: StepParams): void {
        this._dt = p.dt;
        this._substeps = Math.max(1, Math.round(p.substeps ?? 1));
        // the sub-step timestep — every dt-bearing GPU term (inertial g·h², BDF1 v=Δx/h, the velocity-sweep
        // pad |vRel|·h) reads the uniform's `dt` field, so packing `h` here makes one record() loop iteration
        // a complete sub-step. `_dt` (the JS field) stays the FULL fixed dt for setKinematic / the char getter.
        const h = p.dt / this._substeps;
        this._gravity = p.gravity;
        this._iterations = p.iterations;
        this._maxColors = Math.min(p.maxColors ?? COLOR_CAP, COLOR_CAP);
        this._smallN = Math.max(0, p.smallN ?? SMALL_N);
        // the kernel's LDS arrays are sized LDS_CAP at compile time, so the threshold can't exceed it
        this._ldsN = Math.min(LDS_CAP, Math.max(0, p.ldsN ?? LDS_N));
        // reset to the static caps — full dispatch until a `boundColors`/`boundBodies` readback lowers them
        // (so a reconfigure never leaves a stale-low bound, and the standalone gates that skip the readback
        // dispatch every color over the full pool).
        this._colorsToRun = this._maxColors;
        this._bodyGroups = this._fullGroups;
        const ab = new ArrayBuffer(64);
        new Uint32Array(ab, 0, 4).set([this.recordCap, p.iterations, this.eidCap, this._maxColors]);
        new Float32Array(ab, 16, 4).set([h, p.gravity, p.alpha, p.penalty]);
        // betaAng (the joint angular ramp, Phase 6.2) defaults to the canonical 100 when unset (contacts/springs
        // don't read it); jointCount is preserved (setJoints writes it separately, so a reconfigure mustn't zero it).
        new Float32Array(ab, 32, 4).set([1 / (h * h), p.betaLin, p.gamma, p.betaAng ?? 100]);
        // jointCount (48) + substeps (52). setJoints rewrites only jointCount (48), preserving substeps (52).
        new Uint32Array(ab, 48, 2).set([this._jointCount, this._substeps]);
        // offsets 56/60 (_pad2/_pad3) stay 0 — the ArrayBuffer is zero-init.
        this.device.queue.writeBuffer(this._stepUbo, 0, ab);
    }

    /**
     * set the authored springs (Phase 6.1) — the soft `Force` (spring.ts). The primal stamps each body's
     * springs alongside its contacts, the coloring avoids same-color spring pairs. A spring is stateless
     * (`f = stiffness·C`). `[]` clears them. Set once at scene load — no per-frame cost. Springs + joints
     * share one constraint adjacency, so this re-merges both (joints are unaffected, kept from `setJoints`).
     */
    setSprings(springs: readonly SpringDef[]): void {
        this._springDefs = springs;
        this._rebuildConstraints(this._jointDefs);
    }

    /**
     * set the authored joints (Phase 6.2) — the hard `Force` (joint.ts). Each joint carries persistent
     * λ/penalty/c0 in a per-joint record + a recycle-version stamp; the primal stamps it, a per-joint
     * jointInit/jointDual pair warmstarts + ramps it, the coloring repairs hard same-color conflicts. A joint
     * appears in BOTH endpoints' lists. `[]` clears them. Shares the constraint adjacency with springs
     * (re-merged here). The anchors MUST start coincident (see {@link JointDef}).
     *
     * A slot whose def is UNCHANGED (same fields at the same index as the previous set) keeps its live GPU
     * record — warmstart λ/penalty, active flag, a world anchor moved by {@link setJointAnchor} — and its
     * construction guards do NOT re-run; only changed/new slots get a fresh record (act = 2). A re-author
     * under load would otherwise disconnect loaded joints (the reach guard re-judges a stretched chain's
     * separated pins) and zero every λ. Append/remove at the TAIL (the grab pattern) so the authored
     * joints keep their slots.
     */
    setJoints(joints: readonly JointDef[]): void {
        const prev = this._jointDefs;
        this._jointDefs = joints;
        this._rebuildConstraints(prev);
    }

    /**
     * move a {@link WORLD}-anchor joint's world-space point (Phase 6.2) — call each fixed frame to drag a
     * world-anchored body (the mouse grab: `rA = rayOrigin + rayDir·dist`, avbd-demo3d's `drag->rA = …`).
     * `index` is the joint's position in {@link setJoints}. Writes only `rA` (the record's anchor lane),
     * leaving the warmstart + active flag intact — NOT a re-author, so the construction guards don't re-run
     * (a mid-drag anchor moved past the body's reach would wrongly trip the coincidence guard if it did).
     */
    setJointAnchor(index: number, x: number, y: number, z: number): void {
        // jointRecords vec4[1] = (rA.xyz, stiffnessLin); write the 3 anchor floats, leave .w (stiffness) intact
        const w = this._anchorScratch;
        w[0] = x;
        w[1] = y;
        w[2] = z;
        this.device.queue.writeBuffer(this.jointRecords, (index * JOINT_REC_VEC4 + 1) * 16, w);
    }

    /**
     * drive a {@link JointDef.motor}'s target speed (Phase 6.2) — call each fixed frame to change a powered
     * joint's rad/s (a spindle ramping up, a throttle). `index` is the joint's position in {@link setJoints}.
     * Writes only the motor's `speed` (and `maxTorque` if given) lanes of the record, leaving the warmstart
     * λ/penalty + active flag intact — NOT a re-author (the construction guards don't re-run), so the
     * authored joint set is unchanged. The joint must have been authored WITH a `motor` (`maxTorque > 0`);
     * setting `speed` on a motor-less joint is inert (the GPU gate reads `maxTorque > 0`).
     */
    setMotor(index: number, speed: number, maxTorque?: number): void {
        const w = this._anchorScratch;
        w[0] = speed; // jointRecords vec4[11].x = motorSpeed
        this.device.queue.writeBuffer(
            this.jointRecords,
            (index * JOINT_REC_VEC4 + 11) * 16,
            w,
            0,
            1,
        );
        if (maxTorque !== undefined) {
            w[0] = maxTorque; // vec4[10].w = motorMaxTorque (the .w lane → +12 bytes)
            this.device.queue.writeBuffer(
                this.jointRecords,
                (index * JOINT_REC_VEC4 + 10) * 16 + 12,
                w,
                0,
                1,
            );
        }
    }

    /**
     * bump an eid's recycle version (Phase 6.2, project_stable_identity) — call when a Body eid is despawned
     * and reused, so any joint still referencing the old occupant deactivates (it stamped the prior version)
     * rather than silently realiasing to the new body. The opt-in `(eid, version)` side array, never packed
     * into the eid. (The §6.6 declarative joint element wires this into the firehose; the gym drives it directly.)
     */
    recycleVersion(eid: number): void {
        this._versions[eid] = (this._versions[eid] + 1) >>> 0;
        this.device.queue.writeBuffer(
            this.jointVersions,
            eid * 4,
            new Uint32Array([this._versions[eid]]),
        );
    }

    /**
     * move a kinematic body (Phase 6.4) — a `mass <= 0` `Body` whose pose the SCENE drives each fixed step (a
     * moving platform, a grab anchor, the CPU character sweep). Writes the GPU pose directly (the solver never
     * moves a static) and derives the body's velocity from its per-step pose delta, so the character carry
     * rides it (it reads the supporting body's `B_VELL`) and a resting dynamic is dragged by friction. Call
     * ONCE per fixed frame the body moves; the first call seeds velocity 0. `teleport` forces velocity 0 for
     * this call — a jump that isn't motion (a grab anchor snapping onto a freshly-picked body), so it doesn't
     * fling the held body. `vel` overrides the derived velocity with an explicit one — the CPU character sweep
     * passes its realized velocity (which excludes the cosmetic ground snap, unlike the raw pose delta) so the
     * carry-of-riders + broadphase pad read the swept motion, not the snap. `eid` must be a `mass <= 0` `Body`.
     * Angular velocity is not tracked yet (a spinning platform carries by its COM velocity only). (The §6.6
     * declarative layer wires this into the firehose; the gym/Player/character drive it.)
     */
    setKinematic(
        eid: number,
        pos: readonly [number, number, number],
        quat: readonly [number, number, number, number],
        teleport = false,
        vel?: readonly [number, number, number],
    ): void {
        const cap = this.eidCap;
        let prev = this._kinPrev.get(eid);
        if (!prev || teleport) {
            prev ??= new Float32Array(7);
            prev.set([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]);
            this._kinPrev.set(eid, prev);
        }
        const w = this._kinScratch;
        w[0] = pos[0];
        w[1] = pos[1];
        w[2] = pos[2];
        w[3] = 0;
        this.device.queue.writeBuffer(this.bodies, (0 * cap + eid) * 16, w); // B_POS
        w[0] = quat[0];
        w[1] = quat[1];
        w[2] = quat[2];
        w[3] = quat[3];
        this.device.queue.writeBuffer(this.bodies, (1 * cap + eid) * 16, w); // B_QUAT
        const dt = this._dt;
        // explicit `vel` (the CPU sweep's snap-excluded realized velocity) wins; else derive from the pose delta
        // — but `teleport` always zeroes (prev was reset to pos above, so the derived delta is 0 anyway).
        w[0] = vel ? vel[0] : (pos[0] - prev[0]) / dt;
        w[1] = vel ? vel[1] : (pos[1] - prev[1]) / dt;
        w[2] = vel ? vel[2] : (pos[2] - prev[2]) / dt;
        w[3] = 0;
        this.device.queue.writeBuffer(this.bodies, (6 * cap + eid) * 16, w); // B_VELL (linear carry/interp)
        prev[0] = pos[0];
        prev[1] = pos[1];
        prev[2] = pos[2];
        prev[3] = quat[0];
        prev[4] = quat[1];
        prev[5] = quat[2];
        prev[6] = quat[3];
    }

    /**
     * set a dynamic body's linear velocity — a launch impulse (the gravity-gun throw). The next fixed
     * step's inertial pass integrates it (`predicted = pos + vel·dt`); the BDF1 velocity recovery then
     * re-owns the lane, so the write is consumed exactly once. Queue-ordered: a write before this tick's
     * StepSystem submit lands in this tick's solve, after it in the next. Call on a LIVE (seeded) body —
     * a body spawned this frame is re-seeded to velocity 0 by the next pack.
     */
    setVelocity(eid: number, vx: number, vy: number, vz: number): void {
        const w = this._kinScratch;
        w[0] = vx;
        w[1] = vy;
        w[2] = vz;
        w[3] = 0;
        this.device.queue.writeBuffer(this.bodies, (6 * this.eidCap + eid) * 16, w); // B_VELL
    }

    /**
     * set a dynamic body's angular velocity (rad/s, world axes) — the twin of {@link setVelocity}. The next
     * fixed step's inertial pass predicts the rotated pose (`inertialQ = quat ⊕ ω·dt`); the BDF1 velocity
     * recovery then re-owns the lane, so the write is consumed once. Call on a LIVE (seeded) body. A spun
     * dynamic body carries a contact by real friction (the surface's `ω×r` reaches the solver), unlike a
     * spun kinematic body whose rotation is untracked.
     */
    setAngularVelocity(eid: number, wx: number, wy: number, wz: number): void {
        const w = this._kinScratch;
        w[0] = wx;
        w[1] = wy;
        w[2] = wz;
        w[3] = 0;
        this.device.queue.writeBuffer(this.bodies, (7 * this.eidCap + eid) * 16, w); // B_VELA
    }

    // (re)build the merged spring+joint adjacency (constraintCsr offsets/counts + the kind-tagged
    // constraintList entries) + the per-joint records, from `_springDefs` + `_jointDefs`, and upload. Each
    // constraint appears in both endpoints' slices; a joint's two entries point at one shared record. Grows
    // constraintList / jointRecords (rebuilding the solve bind groups) if needed. Run on either `set*`.
    private _rebuildConstraints(prevJoints: readonly JointDef[] = []): void {
        const eidCap = this.eidCap;
        const springs = this._springDefs;
        const joints = this._jointDefs;
        // counts in [eidCap, 2·eidCap), then an exclusive prefix → offsets in [0, eidCap) (entry units)
        const csr = new Uint32Array(2 * eidCap);
        for (const s of springs) {
            csr[eidCap + s.a]++;
            csr[eidCap + s.b]++;
        }
        for (const j of joints) {
            if (j.a >= 0) csr[eidCap + j.a]++; // a < 0 = a world anchor (no body → no entry for it)
            csr[eidCap + j.b]++;
        }
        let acc = 0;
        for (let e = 0; e < eidCap; e++) {
            csr[e] = acc;
            acc += csr[eidCap + e];
        }
        const entries = acc; // 2·(springs + joints)

        let rebuild = false;
        if (entries > this._constraintCap) {
            this.constraintList.destroy();
            this._constraintCap = Math.max(entries, this._constraintCap * 2);
            this.constraintList = this.device.createBuffer({
                label: "phys-constraint-list",
                size: this._constraintCap * CONSTRAINT_VEC4 * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            rebuild = true;
        }
        if (joints.length > this._jointCap) {
            // grow without losing kept slots' live state: copy the old records into the new buffer
            // before retiring it (destroy defers until the queued copy completes)
            const old = this.jointRecords;
            const oldBytes = this._jointCap * JOINT_REC_VEC4 * 16;
            this._jointCap = Math.max(joints.length, this._jointCap * 2);
            this.jointRecords = this.device.createBuffer({
                label: "phys-joint-records",
                size: this._jointCap * JOINT_REC_VEC4 * 16,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            if (oldBytes > 0) {
                const enc = this.device.createCommandEncoder({ label: "phys-joint-grow" });
                enc.copyBufferToBuffer(old, 0, this.jointRecords, 0, oldBytes);
                this.device.queue.submit([enc.finish()]);
            }
            old.destroy();
            rebuild = true;
        }
        if (rebuild) this.buildSolveBindGroups();

        // scatter each constraint into both endpoints' slices (the per-body append cursor starts at the offset)
        const cursor = csr.slice(0, eidCap);
        const list = new Float32Array(Math.max(1, entries) * CONSTRAINT_VEC4 * 4);
        const bits = new Uint32Array(list.buffer);
        const putSpring = (
            idx: number,
            rSelf: readonly [number, number, number],
            rOther: readonly [number, number, number],
            other: number,
            stiffness: number,
            rest: number,
        ): void => {
            const o = idx * CONSTRAINT_VEC4 * 4;
            list[o] = rSelf[0];
            list[o + 1] = rSelf[1];
            list[o + 2] = rSelf[2];
            list[o + 3] = stiffness;
            list[o + 4] = rOther[0];
            list[o + 5] = rOther[1];
            list[o + 6] = rOther[2];
            list[o + 7] = rest;
            bits[o + 8] = other; // [2].x = partner eid
            bits[o + 9] = KIND_SPRING; // [2].y = kind
        };
        // a joint entry is adjacency-only ([2] = partner / kind / record / isA); the anchors live in the per-joint
        // record (jointContrib reads jrec 1/2), so the entry's [0]/[1] are unused (a static rSelf/rOther copy
        // here would also drift stale when setJointAnchor moves a world anchor).
        const putJoint = (idx: number, other: number, recordIndex: number, isA: boolean): void => {
            const o = idx * CONSTRAINT_VEC4 * 4;
            bits[o + 8] = other; // [2].x = partner eid
            bits[o + 9] = KIND_JOINT; // [2].y = kind
            bits[o + 10] = recordIndex; // [2].z = the shared per-joint record
            bits[o + 11] = isA ? 1 : 0; // [2].w = isA (this endpoint is body a)
        };
        for (const s of springs) {
            putSpring(cursor[s.a]++, s.rA, s.rB, s.b, s.stiffness, s.rest);
            putSpring(cursor[s.b]++, s.rB, s.rA, s.a, s.stiffness, s.rest);
        }
        // per-joint records: geometry + recycle versions + zeroed state + active = 2 (fresh → the GPU runs
        // the construction guards once). torqueArm + active are GPU-written; ∞ stiffness → the RIGID
        // sentinel. A slot whose def matches the previous set's at the same index is NOT rewritten — its
        // live record (warmstart λ/penalty, active flag, a setJointAnchor-moved world anchor) survives the
        // upload, so re-authoring under load never re-runs its guards or zeroes its λ (see setJoints).
        const same = (p: JointDef, j: JointDef): boolean =>
            p.a === j.a &&
            p.b === j.b &&
            p.rA[0] === j.rA[0] &&
            p.rA[1] === j.rA[1] &&
            p.rA[2] === j.rA[2] &&
            p.rB[0] === j.rB[0] &&
            p.rB[1] === j.rB[1] &&
            p.rB[2] === j.rB[2] &&
            (p.stiffnessLin ?? Number.POSITIVE_INFINITY) ===
                (j.stiffnessLin ?? Number.POSITIVE_INFINITY) &&
            (p.stiffnessAng ?? 0) === (j.stiffnessAng ?? 0) &&
            // motor: a changed axis/speed/maxTorque re-authors (fresh λ); setMotor drives `speed` live on the
            // RECORD instead, so the def's motor stays constant and the slot is kept (warmstart + live speed survive)
            (p.motor?.maxTorque ?? 0) === (j.motor?.maxTorque ?? 0) &&
            (p.motor?.speed ?? 0) === (j.motor?.speed ?? 0) &&
            (p.motor?.axis[0] ?? 0) === (j.motor?.axis[0] ?? 0) &&
            (p.motor?.axis[1] ?? 0) === (j.motor?.axis[1] ?? 0) &&
            (p.motor?.axis[2] ?? 0) === (j.motor?.axis[2] ?? 0);
        const rec = new Float32Array(JOINT_REC_VEC4 * 4);
        const rbits = new Uint32Array(rec.buffer);
        joints.forEach((j, ji) => {
            // a world anchor (a < 0) has no body → no a-side entry; b's entry points at the WORLD sentinel and
            // carries rA as a WORLD-space point (jointContrib reads `other == WORLD_ANCHOR` → uses rA directly).
            const aEid = j.a >= 0 ? j.a : WORLD_ANCHOR_U32;
            if (j.a >= 0) putJoint(cursor[j.a]++, j.b, ji, true);
            putJoint(cursor[j.b]++, aEid, ji, false);
            if (ji < prevJoints.length && same(prevJoints[ji], j)) return; // kept — live record survives
            const stiffLin = j.stiffnessLin ?? Number.POSITIVE_INFINITY;
            const stiffAng = j.stiffnessAng ?? 0;
            rec.fill(0);
            rbits[0] = aEid;
            rbits[1] = j.b;
            rbits[2] = j.a >= 0 ? this._versions[j.a] : 0;
            rbits[3] = this._versions[j.b];
            rec[4] = j.rA[0];
            rec[5] = j.rA[1];
            rec[6] = j.rA[2];
            rec[7] = Number.isFinite(stiffLin) ? stiffLin : RIGID_STIFFNESS; // [1].w stiffnessLin
            rec[8] = j.rB[0];
            rec[9] = j.rB[1];
            rec[10] = j.rB[2];
            rec[11] = Number.isFinite(stiffAng) ? stiffAng : RIGID_STIFFNESS; // [2].w stiffnessAng
            rbits[13] = 2; // [3].y active = fresh (run the construction guard once); [3].x torqueArm GPU-written
            // motor (cols 10/11): axis + maxTorque static, λ/penalty 0 (rec.fill zeroed them). maxTorque 0 ⇒
            // no motor (the GPU's `jMotorMax > 0` gate). jointInit never rewrites these lanes, so they persist.
            if (j.motor) {
                rec[40] = j.motor.axis[0];
                rec[41] = j.motor.axis[1];
                rec[42] = j.motor.axis[2];
                rec[43] = j.motor.maxTorque; // [10].w motorMaxTorque
                rec[44] = j.motor.speed; // [11].x motorSpeed
            }
            this.device.queue.writeBuffer(this.jointRecords, ji * JOINT_REC_VEC4 * 16, rec);
        });

        this._jointCount = joints.length;
        this.device.queue.writeBuffer(this.constraintCsr, 0, csr);
        if (entries > 0) this.device.queue.writeBuffer(this.constraintList, 0, list);
        // publish the joint count into the step uniform (jointInit/jointDual dispatch + early-out off it)
        this.device.queue.writeBuffer(this._stepUbo, 48, new Uint32Array([this._jointCount]));
    }

    /**
     * seed the dense→eid map directly (the standalone gates, no GPU pack): `eids = [n, 0, 1, …, n-1]`
     * (identity, so dense slot d = eid d — keeping their `(col*eidCap + i)` seed layout) + the BVH prim
     * count. The plugin uses the GPU `pack` instead. `n` must be ≤ maxBodies.
     */
    gateSetCount(n: number): void {
        const map = new Uint32Array(1 + n);
        map[0] = n;
        for (let d = 0; d < n; d++) map[1 + d] = d;
        this.device.queue.writeBuffer(this.eids, 0, map);
        this.device.queue.writeBuffer(this._bvh.count, 0, new Uint32Array([n]));
        // the gate seeds `bodies` directly (writeBuffer), so its bodies ARE seeded — mark the flag the pack
        // would set, so jointInit (which retries until both endpoints are seeded) + the both-static guard run
        // in a seeded constraint gate. `seeded` is read only by jointInit + the pack, so this is inert for the
        // pile/character gates (no joints, no pack).
        this.device.queue.writeBuffer(this.seeded, 0, new Uint32Array(n).fill(1));
        // the indirect dispatch args packScan writes on the plugin path: body passes = ceil(n/64), the
        // per-eid-block passes (collide/dual/CSR) = ceil(n·PAIRS_PER_BODY/64) lanes.
        this.device.queue.writeBuffer(
            this.dispatchArgs,
            0,
            new Uint32Array([Math.ceil(n / 64), 1, 1]),
        );
        this.device.queue.writeBuffer(
            this.pairArgs,
            0,
            new Uint32Array([Math.ceil((n * PAIRS_PER_BODY) / 64), 1, 1]),
        );
        // the gate knows its count exactly, so the broadphase regime follows it — the gym single-step
        // gates exercise the small-N path at gate counts (the oracle's broadphase is itself O(n²))
        this._liveBound = n;
    }

    /** flag a cold-start: the next `record` clears `pairContacts` before the collide (the gates call this
     * between scenes; the plugin never does — a fresh step's pairContacts is zero-init = already cold). */
    cold(): void {
        this._coldNext = true;
    }

    /**
     * the readback-bounded color loop (Phase 4.9 Lever 1): set the dispatched color count from a frame-stale
     * `usedColors` (= the greedy's max dynamic color + 1, read from a Mirror of {@link colorCount}). The primal
     * dispatches `min(maxColors, usedColors + COLOR_MARGIN)` color-passes per iteration — a sparse scene runs
     * ~2-3 dispatched colors, a dense pile caps at `maxColors` (the empty color-passes above the used count
     * are the saving, the overhead-bound common case; gpu.md "Dispatch count is a first-class cost").
     * `usedColors <= 0` (no readback yet / empty scene) keeps the full cap, the safe cold-start (legacy
     * `colorsToRun = lastUsedColors > 0 ? … : MAX`). The margin covers the readback's 1-2 frame staleness; a
     * frame that densifies further under-dispatches once (a soft convergence dip the next readback catches).
     * The solve math is unchanged — this resizes the loop, never the per-color solve (the gym GPU==oracle
     * gates, which drive `record` directly without a readback, stay at full dispatch + identical).
     */
    boundColors(usedColors: number): void {
        this._colorsToRun =
            usedColors > 0 ? Math.min(this._maxColors, usedColors + COLOR_MARGIN) : this._maxColors;
    }

    /**
     * the direct color-loop dispatch (dispatch-ladder rung 0): size the primal/commit color loop's
     * dispatch from a frame-stale live body count (`colorCount[1]`, written by `packScan`, read from
     * the same Mirror as {@link boundColors}'s word). The loop dispatches
     * `ceil((liveCount + BODY_MARGIN) / 64)` workgroups DIRECT — an indirect dispatch costs ≈ 2× a direct
     * one (Dawn's injected validation pass, physics.md "Dispatch count"), and the color loop is
     * `iters × colors × 2` dispatches, the dominant block. Over-dispatch is correctness-safe (the body
     * passes early-out on `d >= eids[0]`); under-dispatch from a spawn burst past the margin skips a
     * burst body's solve for one frame, which the next readback catches. `liveCount <= 0` (no readback
     * yet / empty scene) keeps the full cap, the safe cold-start — the same contract as `boundColors`.
     */
    boundBodies(liveCount: number): void {
        // the primal/commit dispatch off the live count + BODY_MARGIN. The broadphase + LDS regimes key on
        // the same live count (they process all live).
        this._bodyGroups =
            liveCount > 0
                ? Math.min(this._fullGroups, Math.ceil((liveCount + BODY_MARGIN) / 64))
                : this._fullGroups;
        // the live count keys the broadphase regime (see _liveBound) — 0 keeps the BVH path
        this._liveBound = Math.max(0, liveCount);
    }

    /** color-passes the primal dispatches per iteration — the readback-bounded count (Phase 4.9 Lever 1),
     * `min(maxColors, usedColors + COLOR_MARGIN)`; the full cap until `boundColors` is fed a readback */
    get dispatchedColors(): number {
        return this._colorsToRun;
    }

    /** which specialized regimes the last recorded step ran (the frame-stale gates — C1.0 small
     * broadphase, C1.2 LDS solve). The witness a regime-crossing gate reads: span timings can't
     * distinguish the paths (the profiler holds a non-firing pass's last span). */
    get regimes(): { small: boolean; lds: boolean } {
        return { small: this._smallRan, lds: this._ldsRan };
    }

    /**
     * total GPU bytes the step's buffers occupy — the eid-indexed solver state (sized to `eidCap`) + the
     * per-eid pair blocks (`pairList`, sized to `eidCap · PAIRS_PER_BODY`) + the per-eid manifold store +
     * CSR (sized to `maxPairSlots`) + the broadphase BVH (sized to `maxBodies`). `pairContacts`
     * dominates: recordCap × CONTACT_VEC4 × 16 B.
     */
    get bytes(): number {
        let total = this._bvh.bytes;
        for (const b of [
            this.bodies,
            this.solveOut,
            this.pairContacts,
            this.counters,
            this.pairList,
            this.pairArgs,
            this.colors,
            this.colorScratch,
            this.colorCount,
            this.eids,
            this.seeded,
            this.csr,
            this.csrList,
            this.constraintCsr,
            this.constraintList,
            this.jointRecords,
            this.jointVersions,
            this.dispatchArgs,
            this._stepUbo,
            this._colorUbo,
        ]) {
            total += b.size;
        }
        return total;
    }

    /**
     * the GPU pack (fused compaction + one-time seed): membership-scan over capacity → the dense→eid map
     * (`eids[0]` = live count, `eids[1+d]` = the d-th live eid) + a one-time seed of any newly-spawned
     * body's slot from its authored slabs (gated on the GPU `seeded` flag). The plugin's draw-group
     * PackSystem calls it after the slab + membership flush; the next fixed-step `record` reads the map
     * (a 1-frame structural latency). No CPU per-entity iteration — the firehose endpoint. Records the
     * multi-WG count → scan → scatter (C1.3 — one lane per eid, parallel across capacity/PACK_WG
     * workgroups, never a serial per-lane capacity walk); the scan's lane 0 clamps the count + publishes
     * the dispatch args (the prior standalone clampCount, fused). Throws without a packGate.
     */
    pack(encoder: GPUCommandEncoder, inputs: Inputs): void {
        if (!this._packCount) throw new Error("PhysicsStep.pack: created without a packGate");
        this.bindPack(inputs);
        this.pass(
            encoder,
            this._packCount.pipeline,
            this._packCountBG!,
            this._packWgs,
            Compute.span?.("phys:pack"),
        );
        this.pass(
            encoder,
            this._packScan!.pipeline,
            this._packScanBG!,
            1,
            Compute.span?.("phys:pack"),
        );
        this.pass(
            encoder,
            this._packScatter!.pipeline,
            this._packScatterBG!,
            this._packWgs,
            Compute.span?.("phys:pack"),
        );
    }

    // build the pack bind groups from the stable membership + slab sources (lazy — they aren't
    // allocated at warm; rebuilt only if a source identity changes, which doesn't happen in practice).
    private bindPack(inputs: Inputs): void {
        const prev = this._gatherInputs;
        if (
            this._packCountBG &&
            prev?.membership === inputs.membership &&
            prev?.pos === inputs.pos &&
            prev?.quat === inputs.quat &&
            prev?.half === inputs.half &&
            prev?.mass === inputs.mass &&
            prev?.friction === inputs.friction &&
            prev?.shape === inputs.shape
        ) {
            return;
        }
        this._gatherInputs = inputs;
        this._packCountBG = this.device.createBindGroup({
            label: "phys-pack-count",
            layout: this._packCount!.layout,
            entries: [
                { binding: 0, resource: { buffer: inputs.membership } },
                { binding: 1, resource: { buffer: this._packSums } },
                { binding: 2, resource: { buffer: this.seeded } },
                { binding: 3, resource: { buffer: inputs.pos } },
                { binding: 4, resource: { buffer: inputs.quat } },
                { binding: 5, resource: { buffer: inputs.half } },
                { binding: 6, resource: { buffer: inputs.mass } },
                { binding: 7, resource: { buffer: inputs.friction } },
                { binding: 8, resource: { buffer: inputs.shape } },
                { binding: 9, resource: { buffer: this.bodies } },
            ],
        });
        this._packScanBG = this.device.createBindGroup({
            label: "phys-pack-scan",
            layout: this._packScan!.layout,
            entries: [
                { binding: 0, resource: { buffer: this._packSums } },
                { binding: 1, resource: { buffer: this.eids } },
                { binding: 2, resource: { buffer: this._bvh.count } },
                { binding: 3, resource: { buffer: this.counters } },
                { binding: 4, resource: { buffer: this.dispatchArgs } },
                { binding: 5, resource: { buffer: this.pairArgs } },
                { binding: 6, resource: { buffer: this.colorCount } },
            ],
        });
        this._packScatterBG = this.device.createBindGroup({
            label: "phys-pack-scatter",
            layout: this._packScatter!.layout,
            entries: [
                { binding: 0, resource: { buffer: inputs.membership } },
                { binding: 1, resource: { buffer: this._packSums } },
                { binding: 2, resource: { buffer: this.eids } },
            ],
        });
    }

    /**
     * record one full AVBD step onto `encoder`. Taps `Compute.span` if `ProfilePlugin` is installed.
     *
     * The colored primal is `iterations × maxColors` primal+commit pairs — `colorize` (ahead of the
     * primal) caps the colors at `maxColors`, so the dispatch count is bounded by the cap, not the body
     * count (physics.md "Dispatch count is the binding cost"; WebGPU's dominant cost is the per-dispatch
     * CPU encode, ~5.9µs all-in on desktop D3D12, multiples higher on the Deck — not the GPU solve span).
     * Four levers applied here: the cap bounds the dispatch count; the current color rides a dynamic
     * uniform offset (no per-color `advanceColor` dispatch); all colors of an iteration share one compute
     * pass (consecutive same-pass dispatches with a write→read hazard are ordered by the implementation's
     * barrier — the single-step gate confirms); the color loop dispatches direct off the frame-stale live
     * count (`boundBodies`, rung 0 — an indirect dispatch's injected validation costs ≈ 2× a direct one). The solve is **double-buffered** (Phase 4.5 Stage B, the
     * grounded method — paper Algorithm 1, webphysics `commitBodySolveKernel`, oracle `primalColored`):
     * the primal solves a color into the `solveOut` scratch reading the committed `bodies`, then a commit
     * dispatch applies `solveOut` → `bodies` for that color, so a same-color contact pair is a clean
     * Jacobi (both read the color-start pose), not an order-dependent write-in-place race. The commit
     * roughly doubles the primal-related dispatches — the price of a reference-grounded colored commit.
     *
     * One dual dispatch follows each iteration's primal pass (λ ← F + the penalty ramp), reading the
     * post-primal pose written by the colors above it.
     *
     * The collide reads + rewrites each pair slot's records in `pairContacts` (the persistent store) in
     * place, carrying last frame's λ/k; the dual writes the final λ/k back to the same records. No cache
     * pass — the store IS this frame's contacts. A separated pair's slot is cleared by the collide (cold
     * next frame); `cold()` (the gates, between scenes) clears the whole store before the collide.
     *
     * No CPU count: the body passes dispatch over the body pool and early-out past `eids[0]` (the
     * GPU-resident live count the pack wrote); an empty scene is all-early-out, harmless.
     */
    record(encoder: GPUCommandEncoder): void {
        encoder.clearBuffer(this.counters, 0, 64);

        // cold-start the persistent store between scenes (the gates' `cold()`): clear pairContacts so this
        // run's first collide doesn't read the prior scene's records (all kind 0 ⇒ no warmstart). The plugin
        // never sets the flag — a fresh PhysicsStep's pairContacts is zero-init, so it cold-starts naturally.
        if (this._coldNext) {
            encoder.clearBuffer(this.pairContacts);
            this._coldNext = false;
        }

        // sub-step loop (Macklin small-steps): `_substeps` complete AVBD sub-steps of h = dt/_substeps (the
        // uniform's `dt` field carries h — configure), each a full broadphase → collide → solve → BDF1 velocity
        // against the PERSISTENT warmstart store (pairContacts), so each sub-step warmstarts off the previous
        // exactly as a frame warmstarts off the prior frame. `_substeps` = 1 is one iteration = the prior path,
        // byte-identical. The profiler sums same-named spans, so phys:* report the full per-fixed-step time.
        for (let sub = 0; sub < this._substeps; sub++) {
            // body passes dispatch INDIRECT off the live count (dispatchArgs = [ceil(count/64),1,1], written
            // by the pack's scan / gateSetCount) — exactly the live count's workgroups, no over-dispatch.
            // Each thread maps `d = gid.x → eid = eids[1+d]` and early-outs past `eids[0]`.
            // aabb: each body's padded world-AABB → the prim buffer, read by BOTH broadphase regimes below
            // (the BVH builds over it; the small-N scan tiles it). Shares this encoder, so the regime that
            // runs sees this step's prims; either writes each live body's per-eid fixed block
            // `pairList[eid·PAIRS_PER_BODY + k]` directly (nearest-K + static-pin, unused slots INVALID).
            this.passIndirect(
                encoder,
                this._aabb.pipeline,
                this._aabbBG,
                Compute.span?.("phys:aabb"),
            );
            // broadphase regime (C1.0): at a frame-stale live count ≤ smallN the one-dispatch O(n²) scan
            // covers the whole front-end (the BVH build's ~28 dependent phases are structure tax at gameplay
            // counts); past it, the BVH build + descent. Both write identical
            // per-eid pair blocks (shared candidate/emit WGSL), so warmstart carries across a regime flip,
            // and both are exact at any N, so the stale switch is correctness-safe. Same span name — the
            // profiler's phys:broadphase is the front-end pair search whichever regime ran.
            const small = this._liveBound > 0 && this._liveBound <= this._smallN;
            this._smallRan = small;
            if (small) {
                this.passIndirect(
                    encoder,
                    this._broadphaseSmall.pipeline,
                    this._broadphaseSmallBG,
                    Compute.span?.("phys:broadphase"),
                );
            } else {
                this._bvh.build(encoder);
                this.passIndirect(
                    encoder,
                    this._broadphase.pipeline,
                    this._broadphaseBG,
                    Compute.span?.("phys:broadphase"),
                );
            }
            // narrowphase (collide): SAT + in-place warmstart over the live bodies' per-eid pair blocks. FOUR
            // pipelines (box / rounded / hull / rounded-poly) by shape-pair class, the DXC pipeline split (see collidePass).
            // All dispatch indirect off pairArgs (= ceil(liveCount·PAIRS_PER_BODY/64) workgroups, written by
            // packScan) over the SAME slots; each lane → d → owner eid → slot = eid·K + k, early-out past the
            // live body count + the class gate. One compute pass with the shared bind group: the consecutive
            // dispatches' write→read hazards on pairContacts are barrier-ordered (the box pipeline clears dead
            // slots first), as with the primal/commit pair below. Same span name → the profiler sums all four.
            {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:collide"),
                });
                pass.setBindGroup(0, this._collideBG);
                for (const pipeline of this._collidePipelines) {
                    pass.setPipeline(pipeline);
                    pass.dispatchWorkgroupsIndirect(this.pairArgs, 0);
                }
                pass.end();
            }

            // CSR + coloring tail: the small regime runs ONE single-WG fused dispatch (C1.1 — the multi-WG
            // passes' boundaries are near-pure structure tax at gameplay counts; see CSR_COLOR_SMALL_WGSL).
            // The fused coloring runs before jointInit/inertial while the BVH regime's colorize runs after —
            // safe, the coloring reads only mass + this step's contact/constraint adjacency, none of which
            // those passes write. The joint hard-conflict repair keeps its own snapshot+pass rounds in both
            // regimes. The BVH regime keeps the multi-WG passes (work-bound at scale, where a single WG
            // would serialize).
            if (small) {
                this.pass(
                    encoder,
                    this._csrColorSmall.pipeline,
                    this._csrColorSmallBG,
                    1,
                    Compute.span?.("phys:csr"),
                );
                if (this._jointCount > 0) this.repairColors(encoder);
            } else {
                this.buildCsr(encoder);
            }

            // joint warmstart + C₀ capture (Phase 6.2) — before inertial init, so it reads the step-start pose
            // x⁻ (the contact warmstart in collide above reads it the same way). One thread per joint, direct
            // dispatch (the count is CPU-authored, not GPU-resident); skipped entirely when there are no joints.
            if (this._jointCount > 0) {
                this.pass(
                    encoder,
                    this._jointInit.pipeline,
                    this._jointInitBG,
                    Math.ceil(this._jointCount / 64),
                    Compute.span?.("phys:joint"),
                );
            }

            this.passIndirect(
                encoder,
                this._inertial.pipeline,
                this._inertialBG,
                Compute.span?.("phys:inertial"),
            );

            // real incremental-greedy coloring ahead of the primal — the dispatch collapse: the primal
            // loops `maxColors` colors, not one per body (physics.md "Dispatch count"). The coloring reads
            // this step's CSR adjacency + last step's colors snapshot, folds bodies past the cap. In the
            // small regime the fused tail above already colored.
            if (!small) this.colorize(encoder);

            // LDS-resident solve regime (C1.2): at a frame-stale live count ≤ ldsN the whole iters × colors
            // primal/commit/dual block below runs as ONE single-workgroup dispatch with every live body's
            // pose in workgroup memory (SOLVE_LDS_WGSL — the per-color dependent round trip the looped path
            // pays in storage becomes an in-kernel barrier on LDS). Same solve math (solvePose / dualSlot /
            // jointDualOne are shared chunks), so GPU == oracle holds on either path; the full block reports
            // under phys:primal (phys:dual / phys:joint read 0 in this regime, the phys:csr precedent).
            const lds = this._liveBound > 0 && this._liveBound <= this._ldsN;
            this._ldsRan = lds;
            if (lds) {
                this.pass(
                    encoder,
                    this._solveLds.pipeline,
                    this._solveLdsBG,
                    1,
                    Compute.span?.("phys:primal"),
                );
            } else {
                for (let it = 0; it < this._iterations; it++) {
                    // timestamp every iteration — the profiler sums same-named spans, so this reports the FULL
                    // primal GPU time (all iterations × color dispatches), not just it 0.
                    const pass = encoder.beginComputePass({
                        timestampWrites: Compute.span?.("phys:primal"),
                    });
                    // `_colorsToRun` colors/iteration (the readback-bounded count, Phase 4.9 Lever 1 — full cap until
                    // boundColors is fed a usedColors readback), one compute pass (the color rides a dynamic uniform
                    // offset, no advanceColor dispatch). Each color is primal-then-commit: the primal solves that
                    // color's bodies into `solveOut` reading the committed `bodies`, then the commit applies `solveOut`
                    // → `bodies` for that color so the next color's primal sees it (the double-buffer, Phase 4.5
                    // Stage B). Consecutive same-pass dispatches with a write→read hazard (primal→commit on
                    // solveOut, commit→next-primal on bodies) are ordered by the implementation's barrier — the
                    // single-step gate confirms. Each dispatches DIRECT off `_bodyGroups` (the frame-stale live count
                    // + BODY_MARGIN, rung 0 — an indirect dispatch costs ≈ 2× and this loop is the dominant block);
                    // over-dispatched workgroups and colors past the live set early-out on `eids[0]` and no-op.
                    for (let c = 0; c < this._colorsToRun; c++) {
                        pass.setPipeline(this._primal.pipeline);
                        pass.setBindGroup(0, this._primalBG, [c * UBO_ALIGN]);
                        pass.dispatchWorkgroups(this._bodyGroups);
                        pass.setPipeline(this._commit.pipeline);
                        pass.setBindGroup(0, this._commitBG, [c * UBO_ALIGN]);
                        pass.dispatchWorkgroups(this._bodyGroups);
                    }
                    pass.end();

                    // dual update: λ ← F + the penalty ramp over the live contacts, reading the pose the primal
                    // colors just wrote. One thread per per-eid pair slot (indirect off pairArgs, looping its records,
                    // inactive records early-out). Updates λ/k in place in pairContacts — which IS the persistent
                    // store, so next frame's collide warmstarts off it.
                    this.passIndirect(
                        encoder,
                        this._dual.pipeline,
                        this._dualBG,
                        Compute.span?.("phys:dual"),
                        this.pairArgs,
                    );

                    // joint dual: advance λ + the penalty ramp per joint, reading the pose this iteration's primal
                    // wrote (like the contact dual). One thread per joint, in place in the persistent jointRecords.
                    if (this._jointCount > 0) {
                        this.pass(
                            encoder,
                            this._jointDual.pipeline,
                            this._jointDualBG,
                            Math.ceil(this._jointCount / 64),
                            Compute.span?.("phys:joint"),
                        );
                    }
                }
            }

            this.passIndirect(
                encoder,
                this._velocity.pipeline,
                this._velocityBG,
                Compute.span?.("phys:velocity"),
            );
            // No cache pass: warmstart is in place — the dual wrote this sub-step's final λ/k into pairContacts,
            // the persistent store, so the next sub-step's (or frame's) collide reads it at the same slot (Phase 4.7).
        }
    }

    /**
     * scatter the live pose into `transforms` (the eid-indexed mat4 firehose), so a `Body`+`Part`
     * entity renders at the pose physics owns. Dispatches over the body pool (early-out past the live
     * count). Call after the Transform compose (which writes a stale slot for a body eid) and before
     * the renderer reads geometry — physics.md "Body / Transform contract".
     *
     * `alpha` (= time.fixedAlpha, default 1) blends the previous settled pose → the current one for render
     * interpolation (Phase 5): at >60Hz this stops a fixed-step pose repeating then jumping. 1 = the bare
     * current pose. The standalone gates read raw `bodies`, not the composed transform, so leave it default.
     */
    compose(encoder: GPUCommandEncoder, transforms: GPUBuffer, alpha = 1): void {
        if (this._composeDst !== transforms) {
            this._composeDst = transforms;
            this._composeBG = this.device.createBindGroup({
                label: "phys-compose",
                layout: this._compose.layout,
                entries: [
                    { binding: 0, resource: { buffer: this.bodies } },
                    { binding: 1, resource: { buffer: this.eids } },
                    { binding: 2, resource: { buffer: transforms } },
                    { binding: 3, resource: { buffer: this._stepUbo } },
                    { binding: 4, resource: { buffer: this._interpUbo } },
                ],
            });
        }
        this._interpData[0] = alpha;
        this.device.queue.writeBuffer(this._interpUbo, 0, this._interpData);
        this.passIndirect(
            encoder,
            this._compose.pipeline,
            this._composeBG!,
            Compute.span?.("phys:compose"),
        );
    }

    /**
     * recompute the incremental-greedy body coloring (run by `record` ahead of the primal in the BVH
     * regime — the small regime's fused tail colors in-kernel; also the
     * standalone entry the coloring crux test drives). Snapshots colors → colorScratch (the whole eid
     * range — colors are eid-indexed, the neighbors sparse over it) so the greedy reads a stable
     * prior-frame coloring (no atomics, no in-pass read-after-write), then one sweep over the body pool
     * reading the contact graph the collide produced. Reads the dense→eid map (`eids`), so set it first.
     */
    colorize(encoder: GPUCommandEncoder): void {
        encoder.copyBufferToBuffer(this.colors, 0, this.colorScratch, 0, this.eidCap * 4);
        // reset the used-color count so this pass's atomicMax measures only this step's coloring (the
        // readback-bounded color loop's input, Phase 4.9 Lever 1). Word 0 only — word 1 is the live
        // count packScan owns (the direct color-loop dispatch's input).
        encoder.clearBuffer(this.colorCount, 0, 4);
        // color every live body — indirect off the live count (dispatchArgs).
        this.passIndirect(
            encoder,
            this._coloring.pipeline,
            this._coloringBG,
            Compute.span?.("phys:coloring"),
            this.dispatchArgs,
        );
        // joint hard-conflict repair (Phase 6.2): skipped without joints (no hard edges)
        if (this._jointCount > 0) this.repairColors(encoder);
    }

    // the joint hard-conflict coloring repair (Phase 6.2): the greedy avoids but tolerates a folded
    // same-color pair — fine for a soft spring, destabilizing for a hard joint. Each round re-snapshots
    // colors → colorScratch then recolors the lower-eid endpoint of any same-color joint pair. Runs after
    // the greedy in BOTH broadphase regimes (the fused small-N tail colors but never repairs — joint
    // scenes are authored-sparse, so the rounds stay their own passes).
    private repairColors(encoder: GPUCommandEncoder): void {
        for (let r = 0; r < JOINT_REPAIR_ROUNDS; r++) {
            encoder.copyBufferToBuffer(this.colors, 0, this.colorScratch, 0, this.eidCap * 4);
            this.passIndirect(
                encoder,
                this._repair.pipeline,
                this._repairBG,
                Compute.span?.("phys:coloring"),
            );
        }
    }

    /**
     * build the per-body CSR contact adjacency from this step's contacts (count → scan → scatter), so
     * the primal + coloring read only a body's own contacts. Run by `record` after the collide in the
     * BVH regime (the small regime fuses CSR + coloring into one dispatch — CSR_COLOR_SMALL_WGSL); also the
     * standalone entry the coloring crux test calls (it seeds the contact graph, then builds the CSR the
     * coloring reads). The count + scatter dispatch indirect off pairArgs (one thread per per-eid pair slot,
     * looping its records, skipping inactive records); the scan is the single-workgroup parallel prefix.
     */
    buildCsr(encoder: GPUCommandEncoder): void {
        // zero only the count region [eidCap, 2·eidCap); the offset region is fully rewritten by the scan
        encoder.clearBuffer(this.csr, this.eidCap * 4, this.eidCap * 4);
        this.passIndirect(
            encoder,
            this._csrCount.pipeline,
            this._csrCountBG,
            Compute.span?.("phys:csr"),
            this.pairArgs,
        );
        this.pass(encoder, this._csrScan.pipeline, this._csrScanBG, 1, Compute.span?.("phys:csr"));
        this.passIndirect(
            encoder,
            this._csrScatter.pipeline,
            this._csrScatterBG,
            Compute.span?.("phys:csr"),
            this.pairArgs,
        );
    }

    private pass(
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        bg: GPUBindGroup,
        groups: number,
        timestampWrites?: GPUComputePassTimestampWrites,
    ): void {
        const pass = encoder.beginComputePass({ timestampWrites });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(groups);
        pass.end();
    }

    // a pass dispatched indirect off an args buffer ([wgX,1,1] at offset 0). Default `dispatchArgs` =
    // the live body count (packScan / gateSetCount write it); the per-eid-block passes pass `pairArgs`
    // (= ceil(liveCount·PAIRS_PER_BODY/64) lanes) — exactly the live bodies' blocks, never the whole pool.
    private passIndirect(
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        bg: GPUBindGroup,
        timestampWrites?: GPUComputePassTimestampWrites,
        args: GPUBuffer = this.dispatchArgs,
    ): void {
        const pass = encoder.beginComputePass({ timestampWrites });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroupsIndirect(args, 0);
        pass.end();
    }

    destroy(): void {
        this._bvh.destroy();
        this.pairList.destroy();
        this.pairArgs.destroy();
        this.bodies.destroy();
        this.solveOut.destroy();
        this.pairContacts.destroy();
        this.counters.destroy();
        this.colors.destroy();
        this.colorScratch.destroy();
        this.colorCount.destroy();
        this.eids.destroy();
        this.seeded.destroy();
        this._packSums.destroy();
        this.hullData.destroy();
        this.csr.destroy();
        this.csrList.destroy();
        this.constraintCsr.destroy();
        this.constraintList.destroy();
        this.jointRecords.destroy();
        this.jointVersions.destroy();
        this.dispatchArgs.destroy();
        this._stepUbo.destroy();
        this._colorUbo.destroy();
        this._interpUbo.destroy();
    }
}
