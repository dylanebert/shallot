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
//                (csrColorSmallWgsl — C1.1); the multi-WG passes are the at-scale regime
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
// owner's own candidate set flickers (local fragility — Phase 4.9 robustness, avbd.md). Body passes
// dispatch indirect off the live count (the primal/commit color loop direct off its frame-stale readback,
// rung 0); the per-eid-block passes (collide / dual / CSR) dispatch indirect off `pairArgs`
// (= liveCount · PAIRS_PER_BODY lanes). f32 throughout (rebuild's f32-first;
// quantization deferred per gpu.md rule 8). The body + contact buffers are SoA cols-buffers (gpu.md
// consolidation #1). Per-body CSR adjacency feeds the primal: each body reads only its own contacts.

import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute, checkStorageBinding } from "../../engine";
import { precompile, precompileScope } from "../../engine/runtime";
import { bitcastF32toU32, chunk, idiv, uniformLoad, Xform } from "../../engine/utils/core";
// the shared LBVH builder (roadmap "Subgroup-first algorithms": physics is a consumer of the same
// rendering-unaware builder a native-RT path would use). standard → extras is the documented exception
// for this shared GPU primitive (exports.md `bvh/core`), not the onion default.
import { BVH_INVALID, type Bvh, bvhRoot, createBvh } from "../bvh/core";
import {
    collideBoxBox,
    collideHull,
    collideRounded,
    collideRoundedPolytope,
    MAX_CONTACTS,
    polyMake,
    SatResult,
    SPECULATIVE_DISTANCE,
} from "./collide";

/** logical columns per body in the eid-indexed `bodies` SoA cols-buffer (`bodies[col*eidCap + eid]`) */
export const BODY_VEC4 = 12;
/** columns per body in the `solveOut` double-buffer scratch (`solveOut[col*eidCap + eid]`): pos, quat */
const SOLVE_VEC4 = 2;
/** logical columns per contact record in the persistent `pairContacts` SoA cols-buffer
 * (`pairContacts[col*recordCap + rec]`): meta(type,a,b,feature) / normal / rA / rB / c0 / penalty / lambda */
export const CONTACT_VEC4 = 7;
/**
 * contact records per persistent manifold block: one block holds a body pair's whole manifold at a stable
 * per-pair slot (Phase 4.7, the webphysics model). The SAT reduces a pair to MAX_CONTACTS (4, the Jolt
 * spread set, Phase 4.8.1), so a block of that size holds every contact; the collide writes this frame's
 * contacts in place + carries λ/k off last frame's records in the same slot (no hash, no separate store).
 * Matches the oracle's `Manifold.contacts`.
 */
export const CONTACTS_PER_PAIR = MAX_CONTACTS;
/**
 * the per-body FIXED-BLOCK slot count (Phase 4.9 robustness, avbd.md "Storage"): each live
 * body owns a fixed block of `PAIRS_PER_BODY` pair slots at base `eid · PAIRS_PER_BODY` in BOTH the pair list
 * (the descent's `vec2<u32>` output) AND the persistent `pairContacts` manifold store. The base is a function
 * of the owner's eid alone, so a flicker in one body's owned-candidate set churns only THAT body's slots,
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
 * 6.1) and joints (Phase 6.2) SHARE this list + the `constraintCsr` adjacency: the binding budget forces
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
 * one joint's persistent record in `jointRecords` (AoS, 12 vec4, Phase 6.2). The hard `Force` (joint.ts,
 * a port of joint.cpp) carries per-joint mutable dual state warmstarted across frames (λ + a per-iteration
 * penalty ramp + the rigid-stabilization gap `c0`), the per-joint geometry the per-body entries can't hold
 * (init/dual are dispatched one-thread-per-joint, not per-endpoint), and the recycle-version guard:
 *   0 (bitcast a, b, versionA, versionB)   1 (rA.xyz, stiffnessLin)   2 (rB.xyz, stiffnessAng)
 *   3 (torqueArm, bitcast active, _, _)    4 (penaltyLin.xyz)  5 (penaltyAng.xyz)
 *   6 (lambdaLin.xyz)  7 (lambdaAng.xyz)   8 (c0Lin.xyz)  9 (c0Ang.xyz)
 *  10 (motorAxis.xyz, motorMaxTorque)     11 (motorSpeed, motorLambda, motorPenalty, _)
 * `torqueArm` + `active` are GPU-written (jointInit); the CPU seeds geometry + versions + zeroed state. The
 * motor (a 1-DOF force-clamped angular drive, avbd-demo2d motor.cpp; `maxTorque > 0` activates it) rides cols
 * 10/11, which jointInit does NOT rewrite: its static axis/speed/maxTorque persist from `setJoints`, and its
 * λ/penalty warmstart in 11.y/.z. `active`: 0 inactive, 1 active, 2 fresh (jointInit runs the anchor-coincidence
 * guard once). ∞ stiffness (rigid) is the `1e30` sentinel (`> 1e29` reads rigid), matching the harness JSON map.
 */
export const JOINT_REC_VEC4 = 12;
/** ∞-stiffness sentinel: `> RIGID_THRESHOLD` reads "rigid" on the GPU (f32 can't compare to true inf cleanly) */
export const RIGID_STIFFNESS = 1e30;
/** joint hard-conflict coloring-repair rounds (webphysics `BODY_COLOR_HARD_REPAIR_ROUNDS`): a hard
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
 * margin covers the readback's 1-2 frame staleness: the incremental greedy's chromatic number shifts by
 * ≤1 per frame on a settling pile, so one insurance color keeps the loop ≥ this frame's true count; a
 * frame that densifies further under-dispatches once (a soft convergence dip the next readback catches,
 * the same self-healing the live-count margin below relies on). It's a frame-staleness margin (the
 * profiler-counter class, gpu.md), not a tuned solver tolerance.
 */
export const COLOR_MARGIN = 1;
/**
 * the direct color-loop dispatch's live-count margin in bodies (dispatch-ladder rung 0): the primal/commit
 * color loop dispatches `ceil((liveCount + BODY_MARGIN) / 64)` workgroups DIRECT off the frame-stale
 * `colorCount[1]` readback (written by `packScan`): Dawn injects a validation pass per indirect call
 * (measured ≈ 2× the direct unit cost, physics.md "Dispatch count"), so the loop's `iters × colors × 2`
 * dispatches are the place a direct dispatch pays. Over-dispatch is correctness-safe (every body pass
 * early-outs on `d >= eids[0]`); the margin only covers under-dispatch from a spawn burst inside the
 * readback's 1-2 frame staleness: one workgroup's worth of headroom, after which a burst body misses
 * one solve step and the next readback catches it (the spawn-despawn gym gate's hazard). Full-cap on
 * cold start / `configure`, like {@link COLOR_MARGIN}'s `boundColors` pattern.
 */
export const BODY_MARGIN = 64;
/**
 * box-box contact type tag: the source-agnostic seam (joints/kinematic/voxel append other tags later).
 * 1, not 0, so a zeroed (cleared) pairContacts record reads as inactive (tag 0) and the solve skips it.
 */
export const CONSTRAINT_CONTACT = 1;
/** dual-layer penalty seed: the contact penalty starts here and ramps via `betaLin` (manifold.ts) */
export const PENALTY_MIN = 1.0;

/**
 * the per-eid manifold store (`pairContacts`) is the step's largest single storage binding:
 * `eidCap · PAIRS_PER_BODY · CONTACTS_PER_PAIR · CONTACT_VEC4 · 16 B`, dominating {@link PhysicsStep.bytes}.
 * Guard its size against the device's per-binding limit so a high `capacity` fails loud + clear here
 * (naming the buffer, the needed-vs-available, and the remedy) rather than at an opaque bind-group
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

/** per-step constants: one uniform, written by the driver (the live count is GPU-resident, in `eids[0]`) */
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
     * is separate from this fold cap (avbd.md — two different numbers). A body that
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

/** the default small-N regime threshold: the live count at or under which `record` runs the
 * one-dispatch O(n²) broadphase rather than the BVH build + descent (the gameplay regime is
 * structure-tax-bound, not work-bound; crossover measured by the physics bench sweep) */
export const SMALL_N = 1024;

/** the LDS-resident solve capacity: what fits the 16 KB workgroup-memory floor
 * (maxComputeWorkgroupStorageSize 16384): pos (3 split f32 columns, 12 B) + quat (vec4, 16 B) =
 * 28 B/body → 512 · 28 = 14336 B resident, headroom for the kernel's control vars. Sits below the
 * {@link SMALL_N} regime threshold by construction. The DEFAULT threshold is {@link LDS_N}, not this:
 * a `ldsN` sweep up to the capacity is the floor-device lever (the Metal cell). */
export const LDS_CAP = 512;

/** the default LDS-resident solve threshold: the measured Lovelace-neutral point:
 * the single-WG kernel is parity at ≤~64 bodies and loses ~linearly above (~+19% at 130, +39% at 502,
 * its serialized record/CSR latency on one SM outgrows the looped path's dispatch boundaries), so the
 * default engages only where it costs nothing and the gym gates keep it green. On Metal the boundary
 * constant is 4× Lovelace (~3.5 µs/phase recoverable in-kernel), so the C1.4 Apple cell decides
 * raising it toward {@link LDS_CAP}. */
export const LDS_N = 64;

/**
 * the per-step uniform (`params` in every kernel). The schema is the one source of the layout on BOTH
 * sides: the WGSL struct resolves from it, and `configure`'s staging write sizes + offsets derive from
 * `d.sizeOf` / `d.memoryLayoutOf` (`step.test.ts` pins them), so reordering a field can't leave the CPU
 * writer stamping the old offsets.
 * @internal
 */
export const Step = d
    .struct({
        recordCap: d.u32,
        iterations: d.u32,
        eidCap: d.u32,
        maxColors: d.u32,
        dt: d.f32,
        gravity: d.f32,
        alpha: d.f32,
        penalty: d.f32,
        invDt2: d.f32,
        betaLin: d.f32,
        gamma: d.f32,
        betaAng: d.f32,
        jointCount: d.u32,
        substeps: d.u32,
        pad2: d.u32,
        pad3: d.u32,
    })
    .$name("Step");

/** the step uniform's byte size + the `jointCount` byte offset, from the schema — `configure` /
 * `setJoints` stage their write against these rather than hand-counted numbers */
const STEP_BYTES = d.sizeOf(Step);
const STEP_JOINT_COUNT = d.memoryLayoutOf(Step, (s) => s.jointCount).offset;

/** logical columns in the eid-indexed `bodies` SoA cols-buffer (`bodies[col*eidCap + eid]`). A ported
 * chunk folds every constant into the expression that uses it, so a kernel writing a column by
 * hand interpolates these instead of naming a spliced const. */
const B_POS = 0;
const B_QUAT = 1;
const B_INERTL = 2;
const B_INERTQ = 3;
const B_INITL = 4;
const B_INITQ = 5;
const B_VELL = 6;
const B_VELA = 7;
const B_PREVV = 8;
const B_MM = 9;
const B_HF = 10;
const B_ROUND = 11;

/** contact-record columns (SoA, `pairContacts[col*recordCap + rec]`): a record's `C_META` is
 * (type, a, b, feature) — type 0 = inactive (cleared slot), {@link CONSTRAINT_CONTACT} = live; a/b are the
 * body eids, so a record is self-describing (the collide's warmstart pair-identity gate). */
const C_META = 0;
const C_NORMAL = 1;
const C_RA = 2;
const C_RB = 3;
const C_C0 = 4;
const C_PEN = 5;
const C_LAMBDA = 6;

/** `solveOut` columns (`solveOut[col*eidCap + bid]`): the primal's double-buffer scratch */
const SO_POS = 0;
const SO_QUAT = 1;

const RIGID_THRESHOLD = 1.0e29;
/** a joint endpoint of `WORLD_ANCHOR` is the WORLD (no body): its rA is a world-space point, its
 * orientation is identity, its mass/size are 0 (static). The grab pins a box to a world cursor point with
 * no anchor body — hence no anchor↔box contact (joint.cpp bodyA == null). The CPU side gives it no
 * constraint-list entry. */
const WORLD_ANCHOR = 0xffffffff;
const COLLISION_MARGIN = 0.01;
const PENALTY_MAX = 1.0e10;
/** an unused per-eid block slot in `pairList` (the collide/dual/CSR skip it) */
const INVALID_PAIR = 0xffffffff;

// The `bodies` cols-buffer column indices `packCountWgsl`'s still-raw `seedBody` names. A ported chunk
// emits no WGSL consts — every constant folds into the expression that uses it — so the pack kernel, the
// last raw consumer, reaches them by interpolating this TS-side block (the ported-chunk-emits-no-consts
// law). Solve/dual/joint-record constants live only in the TGSL call sites they fold into.
const PACK_B_CONSTS_WGSL = /* wgsl */ `
const B_POS: u32 = ${B_POS}u;
const B_QUAT: u32 = ${B_QUAT}u;
const B_INERTL: u32 = ${B_INERTL}u;
const B_INERTQ: u32 = ${B_INERTQ}u;
const B_INITL: u32 = ${B_INITL}u;
const B_INITQ: u32 = ${B_INITQ}u;
const B_VELL: u32 = ${B_VELL}u;
const B_VELA: u32 = ${B_VELA}u;
const B_PREVV: u32 = ${B_PREVV}u;
const B_MM: u32 = ${B_MM}u;
const B_HF: u32 = ${B_HF}u;
const B_ROUND: u32 = ${B_ROUND}u;
`;

// ── the shared solver bindings, one layout per storage access-mode combination ─────────────────────────
// `params` + `bodies` + `pairContacts` are what nearly every kernel binds, so they live in ONE shared
// layout at group 1 and every kernel's own I/O keeps group 0 (the same bind-by-layout-object shape
// `nodeLayout` uses elsewhere). WGSL access mode is part of a binding's type, so a read-only reader and a read-write writer
// need distinct layouts — hence one variant per (bodies, pairContacts) access pair, and the accessors
// below are FACTORIES over the variant (one authored accessor re-emits per
// layout). A variant's chunks carry their own `Namespace`, so every variant emits the shared math + the
// accessors under the authored names and a raw splice site calls `bPos` / `cc` by those names.
//
// Three variants cover the step: `roRo` (aabb / broadphase / primal / coloring / compose / CSR),
// `roRw` (collide / dual — read poses, write manifolds), `rwRw` (inertial / commit / velocity / joint /
// solve-lds — write poses). A kernel that never touches one of the three bindings simply doesn't
// reference it, so nothing is emitted for it; the bind group still binds the buffer (a layout may be a
// superset of what the shader declares).
type Access = "readonly" | "mutable";

/** the shared solver bind group's group index; every kernel's own I/O is group 0. @internal */
export const SOLVER_GROUP = 1;
/** the shared layout's storage-buffer count (`bodies` + `pairContacts`) — they count toward
 * `maxStorageBuffersPerShaderStage` in every pass that binds the group, declared or not. @internal */
export const SHARED_STORAGE = 2;

const solverLayout = (bodies: Access, pairContacts: Access) =>
    tgpu
        .bindGroupLayout({
            params: { uniform: Step },
            bodies: { storage: d.arrayOf(d.vec4f), access: bodies },
            pairContacts: { storage: d.arrayOf(d.vec4f), access: pairContacts },
        })
        .$idx(SOLVER_GROUP);

type SolverLayout = ReturnType<typeof solverLayout>;

/**
 * the shared quaternion math (oracle math.ts) + the body/contact accessors over one shared-layout
 * variant. Everything here is pure TGSL, so each function is also the CPU truth for its arithmetic.
 * Exported for the chunk-forcing test, which needs a variant no other test has resolved.
 * @internal
 */
export function accessors(bodiesAccess: Access, contactsAccess: Access) {
    const L: SolverLayout = solverLayout(bodiesAccess, contactsAccess);
    const ns = tgpu["~unstable"].namespace({ names: "strict" });

    const qConjW = tgpu
        .fn(
            [d.vec4f],
            d.vec4f,
        )((q) => {
            "use gpu";
            return d.vec4f(std.neg(q.xyz), q.w);
        })
        .$name("qConjW");

    const qMulW = tgpu
        .fn(
            [d.vec4f, d.vec4f],
            d.vec4f,
        )((a, b) => {
            "use gpu";
            return d.vec4f(
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            );
        })
        .$name("qMulW");

    const qInvW = tgpu
        .fn(
            [d.vec4f],
            d.vec4f,
        )((q) => {
            "use gpu";
            return std.div(qConjW(q), std.dot(q, q));
        })
        .$name("qInvW");

    /** the reference's overloaded quat `operator-`, RIGHT-multiplied (avbd.md "Velocity recovery") */
    const qSubW = tgpu
        .fn(
            [d.vec4f, d.vec4f],
            d.vec3f,
        )((a, b) => {
            "use gpu";
            return std.mul(qMulW(a, qInvW(b)).xyz, d.f32(2));
        })
        .$name("qSubW");

    const qRotateW = tgpu
        .fn(
            [d.vec4f, d.vec3f],
            d.vec3f,
        )((q, v) => {
            "use gpu";
            const t = std.mul(d.f32(2), std.cross(q.xyz, v));
            return std.add(std.add(v, std.mul(q.w, t)), std.cross(q.xyz, t));
        })
        .$name("qRotateW");

    /** quat + omega vector → integrated, renormalized quat (math.ts qadd) */
    const qAddW = tgpu
        .fn(
            [d.vec4f, d.vec3f],
            d.vec4f,
        )((a, w) => {
            "use gpu";
            const dq = std.mul(d.f32(0.5), qMulW(d.vec4f(w, 0), a));
            return std.normalize(std.add(a, dq));
        })
        .$name("qAddW");

    // Body-state accessors over the shared `bodies` cols-buffer. `bCol` is the one SoA index site: a warp
    // reading sequential `i` per column coalesces to one cache line (gpu.md cols-buffer pattern).
    // Everything below bPos/bQuat is loop-constant during the solve, so it always reads storage; bPos/bQuat
    // are their own chunk so the LDS-resident solve kernel can declare workgroup-memory readers of the same
    // names instead — the pose is the per-color dependent chain, the one thing that must not round-trip
    // storage there.
    const bCol = tgpu
        .fn(
            [d.u32, d.u32],
            d.vec4f,
        )((col, i) => {
            "use gpu";
            return L.$.bodies[col * L.$.params.eidCap + i];
        })
        .$name("bCol");

    const bInertL = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_INERTL, i).xyz;
        })
        .$name("bInertL");
    const bInertQ = tgpu
        .fn(
            [d.u32],
            d.vec4f,
        )((i) => {
            "use gpu";
            return bCol(B_INERTQ, i);
        })
        .$name("bInertQ");
    const bInitL = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_INITL, i).xyz;
        })
        .$name("bInitL");
    const bInitQ = tgpu
        .fn(
            [d.u32],
            d.vec4f,
        )((i) => {
            "use gpu";
            return bCol(B_INITQ, i);
        })
        .$name("bInitQ");
    const bVelL = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_VELL, i).xyz;
        })
        .$name("bVelL");
    const bVelA = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_VELA, i).xyz;
        })
        .$name("bVelA");
    const bPrevV = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_PREVV, i).xyz;
        })
        .$name("bPrevV");
    const bMass = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((i) => {
            "use gpu";
            return bCol(B_MM, i).w;
        })
        .$name("bMass");
    const bHalf = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_HF, i).xyz;
        })
        .$name("bHalf");
    const bFriction = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((i) => {
            "use gpu";
            return bCol(B_HF, i).w;
        })
        .$name("bFriction");
    // B_ROUND packs the shape tag (x, a bitcast ShapeKind) + the rounding radius (y, sphere/capsule) + the
    // hull id (z, a bitcast registry index for ShapeKind.Hull). Read together with bHalf (the core extents)
    // by the narrowphase + compose. A box has kind 0 + radius 0 + id 0, so a fresh box's bRound(i) is
    // all-zero and the box path stays bit-identical.
    const bShape = tgpu
        .fn(
            [d.u32],
            d.u32,
        )((i) => {
            "use gpu";
            return bitcastF32toU32(bCol(B_ROUND, i).x);
        })
        .$name("bShape");
    const bRadius = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((i) => {
            "use gpu";
            return bCol(B_ROUND, i).y;
        })
        .$name("bRadius");
    const bHullId = tgpu
        .fn(
            [d.u32],
            d.u32,
        )((i) => {
            "use gpu";
            return bitcastF32toU32(bCol(B_ROUND, i).z);
        })
        .$name("bHullId");
    /** the static predicate the solver checks everywhere — a real static / kinematic body (mass ≤ 0),
     *  skipped in the primal + velocity passes and the dual/joint all-static gate */
    const solverStatic = tgpu
        .fn(
            [d.u32],
            d.bool,
        )((i) => {
            "use gpu";
            return bMass(i) <= 0;
        })
        .$name("solverStatic");

    const bPos = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            return bCol(B_POS, i).xyz;
        })
        .$name("bPos");
    const bQuat = tgpu
        .fn(
            [d.u32],
            d.vec4f,
        )((i) => {
            "use gpu";
            return bCol(B_QUAT, i);
        })
        .$name("bQuat");

    /** the contact-record SoA reader (`pairContacts[col*recordCap + rec]`) */
    const cc = tgpu
        .fn(
            [d.u32, d.u32],
            d.vec4f,
        )((rec, col) => {
            "use gpu";
            return L.$.pairContacts[col * L.$.params.recordCap + rec];
        })
        .$name("cc");

    // the oriented-box world-AABB half-extent (|R|·h, R = the body's rotation) inflated by the rounding
    // radius — a capsule/sphere's core extents are 0 along the round axes, so the radius is what bounds
    // them (Phase 6.3); a box (radius 0) is unchanged. Tighter than the sphere bound |h|, so a settled pile
    // overlaps only its real neighbors (keeps the per-body pair block small). Still a valid broadphase
    // superset: two boxes that touch have overlapping box-AABBs, so the narrowphase (sphere test + SAT)
    // never loses a contact the oracle keeps. Shared by the aabb prim + the broadphase query.
    const boxExtent = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((i) => {
            "use gpu";
            const q = bQuat(i);
            const h = bHalf(i);
            const ax0 = std.abs(qRotateW(q, d.vec3f(1, 0, 0)));
            const ax1 = std.abs(qRotateW(q, d.vec3f(0, 1, 0)));
            const ax2 = std.abs(qRotateW(q, d.vec3f(0, 0, 1)));
            return std.add(
                std.add(std.add(std.mul(ax0, h.x), std.mul(ax1, h.y)), std.mul(ax2, h.z)),
                d.vec3f(bRadius(i)),
            );
        })
        .$name("boxExtent");

    // one chunk per definition tier, each forcing its base first so a shared definition lands in the
    // lowest chunk that needs it (the collide.ts chunk-forcing law) — a kernel splices only the tiers it
    // calls, so the aabb pass never compiles the contact reader.
    const mathChunk = chunk("avbd-math", [qConjW, qMulW, qInvW, qSubW, qRotateW, qAddW], ns);
    const bodyRestChunk = chunk(
        "avbd-body-rest",
        [
            bCol,
            bInertL,
            bInertQ,
            bInitL,
            bInitQ,
            bVelL,
            bVelA,
            bPrevV,
            bMass,
            bHalf,
            bFriction,
            bShape,
            bRadius,
            bHullId,
            solverStatic,
        ],
        ns,
    );
    const bodyPoseChunk = chunk("avbd-body-pose", [bPos, bQuat], ns);
    const contactChunk = chunk("avbd-contact", [cc], ns);
    const boxExtentChunk = chunk("avbd-box-extent", [boxExtent], ns);

    /** the shared quat math (`qConjW` … `qAddW`) — every kernel splices it first */
    const mathWgsl = (): string => mathChunk();
    /** the body accessors EXCEPT `bPos`/`bQuat` — the LDS-resident solve declares its own pose readers */
    const bodyRestWgsl = (): string => {
        mathWgsl();
        return bodyRestChunk();
    };
    /** the body accessors including the storage `bPos`/`bQuat` */
    const bodyWgsl = (): string => `${bodyRestWgsl()}\n${bodyPoseChunk()}`;
    /** the contact-record reader `cc`. Forces the body accessors, which then own the shared `params`
     *  declaration whatever order a process resolves in — so a contact consumer splices
     *  {@link bodyRestWgsl} (or {@link bodyWgsl}) ahead of this, never this alone. */
    const contactWgsl = (): string => {
        bodyRestWgsl();
        return contactChunk();
    };
    /** `boxExtent` — the broadphase prim + query half-extent */
    const boxExtentWgsl = (): string => {
        bodyWgsl();
        return boxExtentChunk();
    };

    return {
        layout: L,
        qConjW,
        qMulW,
        qInvW,
        qSubW,
        qRotateW,
        qAddW,
        bCol,
        bInertL,
        bInertQ,
        bInitL,
        bInitQ,
        bVelL,
        bVelA,
        bPrevV,
        bMass,
        bHalf,
        bFriction,
        bShape,
        bRadius,
        bHullId,
        solverStatic,
        bPos,
        bQuat,
        cc,
        boxExtent,
        mathWgsl,
        bodyRestWgsl,
        bodyWgsl,
        contactWgsl,
        boxExtentWgsl,
    };
}

/** bodies read-only, manifolds read-only: aabb / broadphase / primal / coloring / compose / CSR */
const roRo = accessors("readonly", "readonly");
/** bodies read-only, manifolds read-write: collide / dual */
const roRw = accessors("readonly", "mutable");
/** bodies read-write, manifolds read-write: inertial / commit / velocity / joint / solve-lds */
const rwRw = accessors("mutable", "mutable");

// ── the shared solver factory ──────────────────────────────────────────────────────────────────────
// MAT3 / CONTACT_FORCE / JOINT_REC ported to TGSL fns, parameterized by a pose-reader set (the
// factory-closure law: one authored kernel re-emits per reader set — storage readers, and the LDS
// set solve-lds shadows bPos/bQuat with). Unlike the typed kernels below, these are plain `tgpu.fn`
// definitions with no manual chunk()/ns bookkeeping: nothing here produces standalone spliced WGSL text
// the way the raw *PassWgsl() functions do, so `tgpu.resolve([kernel], {names:"strict"})` — one clean
// call per consuming kernel — walks the whole call graph itself with no forcing needed.

/**
 * the persistent per-joint record group's index. `jointRecords` needs its own bind group — solve-lds
 * sits exactly at the 10-storage floor within group 0 + the shared solver group, so a third binding has
 * nowhere else to go. joint-init / joint-dual (this stage) and primal / solve-lds all bind it here.
 * @internal
 */
export const JOINT_GROUP = 2;

const jointLayout = (access: Access) =>
    tgpu
        .bindGroupLayout({ jointRecords: { storage: d.arrayOf(d.vec4f), access } })
        .$idx(JOINT_GROUP);

type JointLayout = ReturnType<typeof jointLayout>;

/**
 * the per-joint record accessors (`JOINT_REC_VEC4` AoS layout) over one shared-layout variant — the
 * `jointRecords` analogue of {@link accessors}. Exported for the chunk-forcing test.
 * @internal
 */
export function jointAccessors(access: Access) {
    const L: JointLayout = jointLayout(access);

    const jrec = tgpu
        .fn(
            [d.u32, d.u32],
            d.vec4f,
        )((rec, col) => {
            "use gpu";
            return L.$.jointRecords[rec * d.u32(JOINT_REC_VEC4) + col];
        })
        .$name("jrec");
    const jActive = tgpu
        .fn(
            [d.u32],
            d.u32,
        )((rec) => {
            "use gpu";
            return bitcastF32toU32(jrec(rec, 3).y);
        })
        .$name("jActive");
    const jTorqueArm = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((rec) => {
            "use gpu";
            return jrec(rec, 3).x;
        })
        .$name("jTorqueArm");
    const jPenLin = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 4).xyz;
        })
        .$name("jPenLin");
    const jPenAng = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 5).xyz;
        })
        .$name("jPenAng");
    const jLamLin = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 6).xyz;
        })
        .$name("jLamLin");
    const jLamAng = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 7).xyz;
        })
        .$name("jLamAng");
    const jC0Lin = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 8).xyz;
        })
        .$name("jC0Lin");
    const jC0Ang = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 9).xyz;
        })
        .$name("jC0Ang");
    const jMotorAxis = tgpu
        .fn(
            [d.u32],
            d.vec3f,
        )((rec) => {
            "use gpu";
            return jrec(rec, 10).xyz;
        })
        .$name("jMotorAxis");
    const jMotorMax = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((rec) => {
            "use gpu";
            return jrec(rec, 10).w;
        })
        .$name("jMotorMax");
    const jMotorSpeed = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((rec) => {
            "use gpu";
            return jrec(rec, 11).x;
        })
        .$name("jMotorSpeed");
    const jMotorLam = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((rec) => {
            "use gpu";
            return jrec(rec, 11).y;
        })
        .$name("jMotorLam");
    const jMotorPen = tgpu
        .fn(
            [d.u32],
            d.f32,
        )((rec) => {
            "use gpu";
            return jrec(rec, 11).z;
        })
        .$name("jMotorPen");

    return {
        layout: L,
        jrec,
        jActive,
        jTorqueArm,
        jPenLin,
        jPenAng,
        jLamLin,
        jLamAng,
        jC0Lin,
        jC0Ang,
        jMotorAxis,
        jMotorMax,
        jMotorSpeed,
        jMotorLam,
        jMotorPen,
    };
}

/** joint records read-only — used by primal / solve-lds. */
export const jointRo = jointAccessors("readonly");
/** joint records read-write — this stage's joint-init / joint-dual. */
export const jointRw = jointAccessors("mutable");

/**
 * the primal / solve-lds own-group bindings ({@link solvePose}'s CSR adjacency + authored constraints) —
 * not consumed by the raw pipeline (primal / solve-lds stay raw at this point), but `solvePose` /
 * `jointContrib` need concrete bindings to resolve + CPU-differential now. The typed primal / solve-lds
 * reuse this layout unchanged.
 * @internal
 */
const primalOwnLayout = tgpu
    .bindGroupLayout({
        csr: { storage: d.arrayOf(d.u32), access: "readonly" },
        csrList: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintCsr: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintList: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    })
    .$idx(0);

/** row-major 3×3, matching the oracle's math.ts — the TGSL twin of the trio's matrix algebra.
 *  @internal */
const SolverMat3 = d.struct({ r0: d.vec3f, r1: d.vec3f, r2: d.vec3f }).$name("Mat3");

const mZero = tgpu
    .fn(
        [],
        SolverMat3,
    )(() => {
        "use gpu";
        return SolverMat3({ r0: d.vec3f(), r1: d.vec3f(), r2: d.vec3f() });
    })
    .$name("mZero");
const mDiag = tgpu
    .fn(
        [d.vec3f],
        SolverMat3,
    )((v) => {
        "use gpu";
        return SolverMat3({
            r0: d.vec3f(v.x, 0, 0),
            r1: d.vec3f(0, v.y, 0),
            r2: d.vec3f(0, 0, v.z),
        });
    })
    .$name("mDiag");
const mAdd = tgpu
    .fn(
        [SolverMat3, SolverMat3],
        SolverMat3,
    )((a, b) => {
        "use gpu";
        return SolverMat3({
            r0: std.add(a.r0, b.r0),
            r1: std.add(a.r1, b.r1),
            r2: std.add(a.r2, b.r2),
        });
    })
    .$name("mAdd");
const mNeg = tgpu
    .fn(
        [SolverMat3],
        SolverMat3,
    )((m) => {
        "use gpu";
        return SolverMat3({ r0: std.neg(m.r0), r1: std.neg(m.r1), r2: std.neg(m.r2) });
    })
    .$name("mNeg");
const mScale = tgpu
    .fn(
        [SolverMat3, d.f32],
        SolverMat3,
    )((m, s) => {
        "use gpu";
        return SolverMat3({ r0: std.mul(m.r0, s), r1: std.mul(m.r1, s), r2: std.mul(m.r2, s) });
    })
    .$name("mScale");
/** outer product a ⊗ b (M[i][j] = a[i]·b[j]) — the single-row Hessian block a spring Jacobian stamps
 *  (maths.h `outer`). */
const outer3 = tgpu
    .fn(
        [d.vec3f, d.vec3f],
        SolverMat3,
    )((a, b) => {
        "use gpu";
        return SolverMat3({ r0: std.mul(b, a.x), r1: std.mul(b, a.y), r2: std.mul(b, a.z) });
    })
    .$name("outer3");
const mMulV = tgpu
    .fn(
        [SolverMat3, d.vec3f],
        d.vec3f,
    )((m, v) => {
        "use gpu";
        return d.vec3f(std.dot(m.r0, v), std.dot(m.r1, v), std.dot(m.r2, v));
    })
    .$name("mMulV");
const mT = tgpu
    .fn(
        [SolverMat3],
        SolverMat3,
    )((m) => {
        "use gpu";
        return SolverMat3({
            r0: d.vec3f(m.r0.x, m.r1.x, m.r2.x),
            r1: d.vec3f(m.r0.y, m.r1.y, m.r2.y),
            r2: d.vec3f(m.r0.z, m.r1.z, m.r2.z),
        });
    })
    .$name("mT");
const mMul = tgpu
    .fn(
        [SolverMat3, SolverMat3],
        SolverMat3,
    )((a, b) => {
        "use gpu";
        return SolverMat3({
            r0: std.add(
                std.add(std.mul(b.r0, a.r0.x), std.mul(b.r1, a.r0.y)),
                std.mul(b.r2, a.r0.z),
            ),
            r1: std.add(
                std.add(std.mul(b.r0, a.r1.x), std.mul(b.r1, a.r1.y)),
                std.mul(b.r2, a.r1.z),
            ),
            r2: std.add(
                std.add(std.mul(b.r0, a.r2.x), std.mul(b.r1, a.r2.y)),
                std.mul(b.r2, a.r2.z),
            ),
        });
    })
    .$name("mMul");
const orthoBasis = tgpu
    .fn(
        [d.vec3f],
        SolverMat3,
    )((n) => {
        "use gpu";
        let t1 = d.vec3f();
        if (std.abs(n.x) > std.abs(n.y)) t1 = d.vec3f(std.neg(n.z), 0, n.x);
        else t1 = d.vec3f(0, n.z, std.neg(n.y));
        t1 = std.normalize(t1);
        return SolverMat3({ r0: n, r1: t1, r2: std.cross(t1, n) });
    })
    .$name("orthoBasis");
/** the joint's angular Jacobian + geometric-stiffness terms (maths.h skew/diagonalize, joint.cpp
 *  geometricStiffnessBallSocket) — row-major, matching the oracle math.ts. */
const skew = tgpu
    .fn(
        [d.vec3f],
        SolverMat3,
    )((r) => {
        "use gpu";
        return SolverMat3({
            r0: d.vec3f(0, std.neg(r.z), r.y),
            r1: d.vec3f(r.z, 0, std.neg(r.x)),
            r2: d.vec3f(std.neg(r.y), r.x, 0),
        });
    })
    .$name("skew");
/** diag of each column's length (the joint's diagonal higher-order approximation) */
const diagonalize = tgpu
    .fn(
        [SolverMat3],
        SolverMat3,
    )((m) => {
        "use gpu";
        return mDiag(
            d.vec3f(
                std.length(d.vec3f(m.r0.x, m.r1.x, m.r2.x)),
                std.length(d.vec3f(m.r0.y, m.r1.y, m.r2.y)),
                std.length(d.vec3f(m.r0.z, m.r1.z, m.r2.z)),
            ),
        );
    })
    .$name("diagonalize");
/** geometricStiffnessBallSocket(k, v): diag(-v[k]) with v added into column k. Dynamic vector indexing
 *  (`v[k]`, `k` a runtime u32) — TGSL can't express it (the smallest-3 codec precedent), so this stays a
 *  raw-WGSL-bodied leaf; every caller (jointContrib → solvePose) loses CPU-callability transitively,
 *  precedented and accepted (the f64 oracle stays the gate). */
const geomStiffness = tgpu
    .fn(
        [d.u32, d.vec3f],
        SolverMat3,
    )(/* wgsl */ `(k: u32, v: vec3f) -> Mat3 {
    let d = -v[k];
    var c0 = vec3f(d, 0.0, 0.0); var c1 = vec3f(0.0, d, 0.0); var c2 = vec3f(0.0, 0.0, d);
    if (k == 0u) { c0 = c0 + v; } else if (k == 1u) { c1 = c1 + v; } else { c2 = c2 + v; }
    return Mat3(vec3f(c0.x, c1.x, c2.x), vec3f(c0.y, c1.y, c2.y), vec3f(c0.z, c1.z, c2.z));
}`)
    .$name("geomStiffness");

/** one contact's constraint C, its four Jacobian blocks, the diagonal penalty k, and the cone-clamped
 *  force F — the shared core of the primal stamp and the dual update (manifold.ts `contactForce`, which
 *  the reference inlines in both `updatePrimal` and `updateDual`). */
const CForce = d
    .struct({
        constraint: d.vec3f,
        force: d.vec3f,
        jALin: SolverMat3,
        jBLin: SolverMat3,
        jAAng: SolverMat3,
        jBAng: SolverMat3,
        k: SolverMat3,
        // the *pre-clamp* friction magnitude + the cone bound — the dual ramp gate reads these
        // (manifold.ts contactForce / manifold.cpp:156,169), never the post-clamp force.yz.
        frictionScale: d.f32,
        bounds: d.f32,
    })
    .$name("CForce");

const Contrib = d
    .struct({
        lhsLin: SolverMat3,
        lhsAng: SolverMat3,
        lhsCross: SolverMat3,
        rhsLin: d.vec3f,
        rhsAng: d.vec3f,
    })
    .$name("Contrib");

const Sol = d.struct({ xLin: d.vec3f, xAng: d.vec3f }).$name("Sol");
const NewPose = d.struct({ pos: d.vec3f, quat: d.vec4f }).$name("NewPose");

const solve6 = tgpu
    .fn(
        [SolverMat3, SolverMat3, SolverMat3, d.vec3f, d.vec3f],
        Sol,
    )((aLin, aAng, aCross, bLin, bAng) => {
        "use gpu";
        const A11 = aLin.r0.x;
        const A21 = aLin.r1.x;
        const A22 = aLin.r1.y;
        const A31 = aLin.r2.x;
        const A32 = aLin.r2.y;
        const A33 = aLin.r2.z;
        const A41 = aCross.r0.x;
        const A42 = aCross.r0.y;
        const A43 = aCross.r0.z;
        const A44 = aAng.r0.x;
        const A51 = aCross.r1.x;
        const A52 = aCross.r1.y;
        const A53 = aCross.r1.z;
        const A54 = aAng.r1.x;
        const A55 = aAng.r1.y;
        const A61 = aCross.r2.x;
        const A62 = aCross.r2.y;
        const A63 = aCross.r2.z;
        const A64 = aAng.r2.x;
        const A65 = aAng.r2.y;
        const A66 = aAng.r2.z;
        const L21 = A21 / A11;
        const L31 = A31 / A11;
        const L41 = A41 / A11;
        const L51 = A51 / A11;
        const L61 = A61 / A11;
        const D1 = A11;
        const D2 = A22 - L21 * L21 * D1;
        const L32 = (A32 - L21 * L31 * D1) / D2;
        const L42 = (A42 - L21 * L41 * D1) / D2;
        const L52 = (A52 - L21 * L51 * D1) / D2;
        const L62 = (A62 - L21 * L61 * D1) / D2;
        const D3 = A33 - (L31 * L31 * D1 + L32 * L32 * D2);
        const L43 = (A43 - L31 * L41 * D1 - L32 * L42 * D2) / D3;
        const L53 = (A53 - L31 * L51 * D1 - L32 * L52 * D2) / D3;
        const L63 = (A63 - L31 * L61 * D1 - L32 * L62 * D2) / D3;
        const D4 = A44 - (L41 * L41 * D1 + L42 * L42 * D2 + L43 * L43 * D3);
        const L54 = (A54 - L41 * L51 * D1 - L42 * L52 * D2 - L43 * L53 * D3) / D4;
        const L64 = (A64 - L41 * L61 * D1 - L42 * L62 * D2 - L43 * L63 * D3) / D4;
        const D5 = A55 - (L51 * L51 * D1 + L52 * L52 * D2 + L53 * L53 * D3 + L54 * L54 * D4);
        const L65 = (A65 - L51 * L61 * D1 - L52 * L62 * D2 - L53 * L63 * D3 - L54 * L64 * D4) / D5;
        const D6 =
            A66 -
            (L61 * L61 * D1 + L62 * L62 * D2 + L63 * L63 * D3 + L64 * L64 * D4 + L65 * L65 * D5);
        const y1 = bLin.x;
        const y2 = bLin.y - L21 * y1;
        const y3 = bLin.z - L31 * y1 - L32 * y2;
        const y4 = bAng.x - L41 * y1 - L42 * y2 - L43 * y3;
        const y5 = bAng.y - L51 * y1 - L52 * y2 - L53 * y3 - L54 * y4;
        const y6 = bAng.z - L61 * y1 - L62 * y2 - L63 * y3 - L64 * y4 - L65 * y5;
        const z1 = y1 / D1;
        const z2 = y2 / D2;
        const z3 = y3 / D3;
        const z4 = y4 / D4;
        const z5 = y5 / D5;
        const z6 = y6 / D6;
        const xAng = d.vec3f(0, 0, 0);
        const xLin = d.vec3f(0, 0, 0);
        xAng.z = z6;
        xAng.y = z5 - L65 * xAng.z;
        xAng.x = z4 - L54 * xAng.y - L64 * xAng.z;
        xLin.z = z3 - L43 * xAng.x - L53 * xAng.y - L63 * xAng.z;
        xLin.y = z2 - L32 * xLin.z - L42 * xAng.x - L52 * xAng.y - L62 * xAng.z;
        xLin.x = z1 - L21 * xLin.y - L31 * xLin.z - L41 * xAng.x - L51 * xAng.y - L61 * xAng.z;
        return Sol({ xLin, xAng });
    })
    .$name("solve6");

/**
 * the shared contact stamp over one pose-reader set (storage readers this stage; the LDS set repeats
 * this factory call with solve-lds's workgroup-memory readers instead). @internal
 */
export function contactMath(A: ReturnType<typeof accessors>) {
    const contactForce = tgpu
        .fn(
            [d.u32],
            CForce,
        )((ci) => {
            "use gpu";
            const m0 = A.cc(ci, C_META);
            const a = bitcastF32toU32(m0.y);
            const b = bitcastF32toU32(m0.z);
            const basis = orthoBasis(A.cc(ci, C_NORMAL).xyz);
            const rA = A.cc(ci, C_RA).xyz;
            const rB = A.cc(ci, C_RB).xyz;
            const c0 = A.cc(ci, C_C0).xyz;
            const pen = A.cc(ci, C_PEN);
            const friction = pen.w;
            const lambda = A.cc(ci, C_LAMBDA).xyz;

            const aQuat = A.bQuat(a);
            const bQ = A.bQuat(b);
            const dALin = std.sub(A.bPos(a), A.bInitL(a));
            const dAAng = A.qSubW(aQuat, A.bInitQ(a));
            const dBLin = std.sub(A.bPos(b), A.bInitL(b));
            const dBAng = A.qSubW(bQ, A.bInitQ(b));

            // the arm anchors the CORE feature; apply the geometric ±radius·normal offset HERE so the
            // radius part never rotates with the body's spin (avbd.md "the core arm rule")
            const n = basis.r0;
            const rAW = std.sub(A.qRotateW(aQuat, rA), std.mul(n, A.bRadius(a)));
            const rBW = std.add(A.qRotateW(bQ, rB), std.mul(n, A.bRadius(b)));
            const jALin = basis;
            const jBLin = mNeg(basis);
            const jAAng = SolverMat3({
                r0: std.cross(rAW, basis.r0),
                r1: std.cross(rAW, basis.r1),
                r2: std.cross(rAW, basis.r2),
            });
            const jBAng = SolverMat3({
                r0: std.cross(rBW, jBLin.r0),
                r1: std.cross(rBW, jBLin.r1),
                r2: std.cross(rBW, jBLin.r2),
            });
            const k = mDiag(pen.xyz);

            const t1 = std.mul(c0, std.sub(1, A.layout.$.params.alpha));
            const t2 = mMulV(jALin, dALin);
            const t3 = mMulV(jBLin, dBLin);
            const t4 = mMulV(jAAng, dAAng);
            const t5 = mMulV(jBAng, dBAng);
            const constraint = std.add(std.add(std.add(std.add(t1, t2), t3), t4), t5);
            // force = k·C + λ, clamped: normal repulsion-only, friction inside the Coulomb cone
            let force = std.add(mMulV(k, constraint), lambda);
            force.x = std.min(force.x, 0);
            const bounds = std.mul(std.abs(force.x), friction);
            const fs = std.length(force.yz);
            if (fs > bounds && fs > 0) {
                force = d.vec3f(force.x, (force.y * bounds) / fs, (force.z * bounds) / fs);
            }

            return CForce({
                constraint,
                force,
                jALin,
                jBLin,
                jAAng,
                jBAng,
                k,
                frictionScale: fs,
                bounds,
            });
        })
        .$name("contactForce");

    return { contactForce };
}

/**
 * the primal stamp's three per-constraint contributions (contact / spring / joint), plus `solvePose`
 * itself — over one (bodies, joints) pose-reader pair.
 * @internal
 */
export function contribMath(
    A: ReturnType<typeof accessors>,
    J: ReturnType<typeof jointAccessors>,
    contactForce: ReturnType<typeof contactMath>["contactForce"],
) {
    /** stamp one contact's force + Hessian into `bid`'s system (manifold.ts updatePrimal). `bid` is the
     *  body being solved; `ownerIsA` selects `bid`'s Jacobian from the shared `contactForce`. */
    const contactContrib = tgpu
        .fn(
            [d.u32, d.u32],
            Contrib,
        )((bid, ci) => {
            "use gpu";
            const cf = contactForce(ci);
            const ownerIsA = bitcastF32toU32(A.cc(ci, C_META).y) === bid;
            let jLin = SolverMat3(cf.jBLin);
            let jAng = SolverMat3(cf.jBAng);
            if (ownerIsA) {
                jLin = SolverMat3(cf.jALin);
                jAng = SolverMat3(cf.jAAng);
            }
            const jLinT = mT(jLin);
            const jAngT = mT(jAng);
            const jAngTk = mMul(jAngT, cf.k);
            return Contrib({
                lhsLin: mMul(mMul(jLinT, cf.k), jLin),
                lhsAng: mMul(jAngTk, jAng),
                lhsCross: mMul(jAngTk, jLin),
                rhsLin: mMulV(jLinT, cf.force),
                rhsAng: mMulV(jAngT, cf.force),
            });
        })
        .$name("contactContrib");

    /** stamp one spring's force + Hessian into `bid`'s system (spring.ts stampSpring). The soft Force:
     *  f = stiffness·C, no dual. Symmetric — both endpoints stamp jLin = normalize(pSelf − pOther). */
    const springContrib = tgpu
        .fn(
            [d.u32, d.u32],
            Contrib,
        )((bid, e) => {
            "use gpu";
            const base = e * d.u32(CONSTRAINT_VEC4);
            const s0 = primalOwnLayout.$.constraintList[base]; // rSelf.xyz, stiffness
            const s1 = primalOwnLayout.$.constraintList[base + 1]; // rOther.xyz, rest
            const other = bitcastF32toU32(primalOwnLayout.$.constraintList[base + 2].x);
            const stiffness = s0.w;
            const rW = A.qRotateW(A.bQuat(bid), s0.xyz); // bid's anchor in world — pSelf offset AND torque arm
            const pSelf = std.add(A.bPos(bid), rW);
            const pOther = std.add(A.bPos(other), A.qRotateW(A.bQuat(other), s1.xyz));
            const diff = std.sub(pSelf, pOther);
            const dLen = std.length(diff);
            if (dLen <= 1e-6) {
                return Contrib({
                    lhsLin: mZero(),
                    lhsAng: mZero(),
                    lhsCross: mZero(),
                    rhsLin: d.vec3f(0, 0, 0),
                    rhsAng: d.vec3f(0, 0, 0),
                });
            }
            const n = std.div(diff, dLen);
            const f = stiffness * (dLen - s1.w);
            const jLin = n;
            const jAng = std.cross(rW, n);
            return Contrib({
                lhsLin: mScale(outer3(jLin, jLin), stiffness),
                lhsAng: mScale(outer3(jAng, jAng), stiffness),
                lhsCross: mScale(outer3(jAng, jLin), stiffness),
                rhsLin: std.mul(jLin, f),
                rhsAng: std.mul(jAng, f),
            });
        })
        .$name("springContrib");

    /** stamp one joint's force + Hessian into `bid`'s system (joint.ts stampJoint). The hard Force: a
     *  linear anchor-pin row triple + an angular relative-orientation row triple + an optional 1-DOF
     *  motor, carrying warmstartable λ + a per-iteration penalty ramp read off the per-joint RECORD
     *  (the single source of truth — a moving WORLD anchor is seen here, not the list entry). */
    const jointContrib = tgpu
        .fn(
            [d.u32, d.u32],
            Contrib,
        )((bid, e) => {
            "use gpu";
            const zero = Contrib({
                lhsLin: mZero(),
                lhsAng: mZero(),
                lhsCross: mZero(),
                rhsLin: d.vec3f(0, 0, 0),
                rhsAng: d.vec3f(0, 0, 0),
            });
            const base = e * d.u32(CONSTRAINT_VEC4);
            const e2 = primalOwnLayout.$.constraintList[base + 2];
            const other = bitcastF32toU32(e2.x);
            const rec = bitcastF32toU32(e2.z);
            const isA = bitcastF32toU32(e2.w) !== 0;
            if (J.jActive(rec) !== 1) return zero;
            const rA = J.jrec(rec, 1).xyz;
            const rB = J.jrec(rec, 2).xyz;
            const rSelf = std.select(rB, rA, isA);
            const rOther = std.select(rA, rB, isA);

            const rigidLin = J.jrec(rec, 1).w > RIGID_THRESHOLD;
            const rigidAng = J.jrec(rec, 2).w > RIGID_THRESHOLD;
            const torqueArm = J.jTorqueArm(rec);
            const alpha = A.layout.$.params.alpha;

            const otherWorld = other === WORLD_ANCHOR;
            const qSelf = A.bQuat(bid);
            const qOther = std.select(A.bQuat(other), d.vec4f(0, 0, 0, 1), otherWorld);
            const rSelfW = A.qRotateW(qSelf, rSelf);
            const pSelf = std.add(A.bPos(bid), rSelfW);
            const pOther = std.select(
                std.add(A.bPos(other), A.qRotateW(qOther, rOther)),
                rOther,
                otherWorld,
            );

            const acc = Contrib(zero);

            const penLin = J.jPenLin(rec);
            if (std.dot(penLin, penLin) > 0) {
                const K = mDiag(penLin);
                let C = std.select(std.sub(pOther, pSelf), std.sub(pSelf, pOther), isA);
                if (rigidLin) C = std.sub(C, std.mul(J.jC0Lin(rec), alpha));
                const F = std.add(mMulV(K, C), J.jLamLin(rec));
                const jLin = mDiag(d.vec3f(std.select(d.f32(-1), d.f32(1), isA)));
                const jAng = skew(std.select(rSelfW, std.neg(rSelfW), isA));
                const jLinT = mT(jLin);
                const jAngT = mT(jAng);
                const jAngTk = mMul(jAngT, K);
                acc.lhsLin = mAdd(acc.lhsLin, mMul(mMul(jLinT, K), jLin));
                acc.lhsAng = mAdd(acc.lhsAng, mMul(jAngTk, jAng));
                acc.lhsCross = mAdd(acc.lhsCross, mMul(jAngTk, jLin));
                const r = std.select(std.neg(rSelfW), rSelfW, isA);
                const h1 = mScale(geomStiffness(0, r), F.x);
                const h2 = mScale(geomStiffness(1, r), F.y);
                const h3 = mScale(geomStiffness(2, r), F.z);
                const H = mAdd(mAdd(h1, h2), h3);
                acc.lhsAng = mAdd(acc.lhsAng, diagonalize(H));
                acc.rhsLin = std.add(acc.rhsLin, mMulV(jLinT, F));
                acc.rhsAng = std.add(acc.rhsAng, mMulV(jAngT, F));
            }
            const penAng = J.jPenAng(rec);
            if (std.dot(penAng, penAng) > 0) {
                const K = mDiag(penAng);
                const qA = std.select(qOther, qSelf, isA);
                const qB = std.select(qSelf, qOther, isA);
                let C = std.mul(A.qSubW(qA, qB), torqueArm);
                if (rigidAng) C = std.sub(C, std.mul(J.jC0Ang(rec), alpha));
                const F = std.add(mMulV(K, C), J.jLamAng(rec));
                const sgn = std.select(std.neg(torqueArm), torqueArm, isA);
                acc.lhsAng = mAdd(acc.lhsAng, mScale(K, sgn * sgn));
                acc.rhsAng = std.add(acc.rhsAng, std.mul(F, sgn));
            }
            const motorMax = J.jMotorMax(rec);
            if (motorMax > 0) {
                const axis = J.jMotorAxis(rec);
                const dSelf = std.dot(A.qSubW(qSelf, A.bInitQ(bid)), axis);
                const dOther = std.select(
                    std.dot(A.qSubW(qOther, A.bInitQ(other)), axis),
                    d.f32(0),
                    otherWorld,
                );
                const dB = std.select(dSelf, dOther, isA);
                const dA = std.select(dOther, dSelf, isA);
                const c = dB - dA - J.jMotorSpeed(rec) * A.layout.$.params.dt;
                const f = std.clamp(
                    J.jMotorPen(rec) * c + J.jMotorLam(rec),
                    std.neg(motorMax),
                    motorMax,
                );
                const sgn = std.select(d.f32(1), d.f32(-1), isA);
                acc.lhsAng = mAdd(acc.lhsAng, mScale(outer3(axis, axis), J.jMotorPen(rec)));
                acc.rhsAng = std.add(acc.rhsAng, std.mul(axis, sgn * f));
            }
            return acc;
        })
        .$name("jointContrib");

    /** the per-body primal step: stamp `bid`'s CSR contacts + authored constraints into one 6×6 block
     *  system, add the inertial term, LDLᵀ-solve, integrate. Pose reads go through `A.bPos`/`A.bQuat`, so
     *  the composing kernel picks the backing — storage for the typed primal, workgroup memory for
     *  solve-lds's LDS reader set. Differential-tested on CPU against the f64 oracle math. */
    const solvePose = tgpu
        .fn(
            [d.u32],
            NewPose,
        )((bid) => {
            "use gpu";
            const acc = Contrib({
                lhsLin: mZero(),
                lhsAng: mZero(),
                lhsCross: mZero(),
                rhsLin: d.vec3f(0, 0, 0),
                rhsAng: d.vec3f(0, 0, 0),
            });
            const lo = primalOwnLayout.$.csr[bid];
            const hi = lo + primalOwnLayout.$.csr[A.layout.$.params.eidCap + bid];
            for (let k = lo; k < hi; k++) {
                const c = contactContrib(bid, primalOwnLayout.$.csrList[k]);
                acc.lhsLin = mAdd(acc.lhsLin, c.lhsLin);
                acc.lhsAng = mAdd(acc.lhsAng, c.lhsAng);
                acc.lhsCross = mAdd(acc.lhsCross, c.lhsCross);
                acc.rhsLin = std.add(acc.rhsLin, c.rhsLin);
                acc.rhsAng = std.add(acc.rhsAng, c.rhsAng);
            }
            const slo = primalOwnLayout.$.constraintCsr[bid];
            const shi = slo + primalOwnLayout.$.constraintCsr[A.layout.$.params.eidCap + bid];
            for (let e = slo; e < shi; e++) {
                const kind = bitcastF32toU32(
                    primalOwnLayout.$.constraintList[e * d.u32(CONSTRAINT_VEC4) + 2].y,
                );
                let c = Contrib(springContrib(bid, e));
                if (kind === KIND_JOINT) c = Contrib(jointContrib(bid, e));
                acc.lhsLin = mAdd(acc.lhsLin, c.lhsLin);
                acc.lhsAng = mAdd(acc.lhsAng, c.lhsAng);
                acc.lhsCross = mAdd(acc.lhsCross, c.lhsCross);
                acc.rhsLin = std.add(acc.rhsLin, c.rhsLin);
                acc.rhsAng = std.add(acc.rhsAng, c.rhsAng);
            }

            const mm = A.bCol(B_MM, bid);
            const mLin = mDiag(std.mul(d.vec3f(mm.w), A.layout.$.params.invDt2));
            const mAng = mDiag(std.mul(mm.xyz, A.layout.$.params.invDt2));
            const lhsLin = mAdd(acc.lhsLin, mLin);
            const lhsAng = mAdd(acc.lhsAng, mAng);
            const rhsLin = std.add(acc.rhsLin, mMulV(mLin, std.sub(A.bPos(bid), A.bInertL(bid))));
            const rhsAng = std.add(acc.rhsAng, mMulV(mAng, A.qSubW(A.bQuat(bid), A.bInertQ(bid))));
            const r = solve6(lhsLin, lhsAng, acc.lhsCross, std.neg(rhsLin), std.neg(rhsAng));
            return NewPose({
                pos: std.add(A.bPos(bid), r.xLin),
                quat: A.qAddW(A.bQuat(bid), r.xAng),
            });
        })
        .$name("solvePose");

    return { contactContrib, springContrib, jointContrib, solvePose };
}

const STICK_THRESH = 1.0e-5;

/**
 * one pair slot's dual update (manifold.ts updateDual), shared by the standalone dual pass and the
 * LDS-resident kernel: loop its `CONTACTS_PER_PAIR` records, dual-update each active one in place.
 * @internal
 */
export function dualMath(
    A: ReturnType<typeof accessors>,
    contactForce: ReturnType<typeof contactMath>["contactForce"],
) {
    const dualSlot = tgpu
        .fn([d.u32])((slot) => {
            "use gpu";
            const rc = A.layout.$.params.recordCap;
            const recBase = slot * d.u32(CONTACTS_PER_PAIR);
            for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
                const ci = recBase + ls;
                const m0 = A.cc(ci, C_META);
                if (bitcastF32toU32(m0.x) !== CONSTRAINT_CONTACT) continue; // inactive record
                // dual-ramp gate (avbd.md): an all-static manifold is never satisfiable by the primal
                if (A.solverStatic(bitcastF32toU32(m0.y)) && A.solverStatic(bitcastF32toU32(m0.z)))
                    continue;
                const cf = contactForce(ci);
                const pen = A.cc(ci, C_PEN);
                const friction = pen.w;
                const k = pen.xyz;
                // pre-clamp magnitude + bound (cf), not the post-clamp force.yz — avbd.md "friction ramp"
                const bounds = cf.bounds;
                const fs = cf.frictionScale;
                if (cf.force.x < 0) {
                    k.x = std.min(
                        k.x + A.layout.$.params.betaLin * std.abs(cf.constraint.x),
                        PENALTY_MAX,
                    );
                }
                let stick = d.f32(0);
                if (fs <= bounds) {
                    k.y = std.min(
                        k.y + A.layout.$.params.betaLin * std.abs(cf.constraint.y),
                        PENALTY_MAX,
                    );
                    k.z = std.min(
                        k.z + A.layout.$.params.betaLin * std.abs(cf.constraint.z),
                        PENALTY_MAX,
                    );
                    if (std.length(cf.constraint.yz) < STICK_THRESH) stick = d.f32(1);
                }
                A.layout.$.pairContacts[C_LAMBDA * rc + ci] = d.vec4f(cf.force, stick);
                A.layout.$.pairContacts[C_PEN * rc + ci] = d.vec4f(k, friction);
            }
        })
        .$name("dualSlot");

    return { dualSlot };
}

/**
 * one joint's dual update (joint.ts updateJointDual), shared by the standalone joint-dual pass and the
 * LDS-resident kernel. The rigid (∞-stiffness) rows store λ ← K·C + λ; both row triples ramp the penalty
 * by β|C|, clamped to `PENALTY_MAX`.
 * @internal
 */
export function jointDualMath(
    A: ReturnType<typeof accessors>,
    J: ReturnType<typeof jointAccessors>,
) {
    const jointDualOne = tgpu
        .fn([d.u32])((jid) => {
            "use gpu";
            if (J.jActive(jid) !== 1) return;
            const recBase = jid * d.u32(JOINT_REC_VEC4);
            const a = bitcastF32toU32(J.jrec(jid, 0).x);
            const b = bitcastF32toU32(J.jrec(jid, 0).y);
            const aWorld = a === WORLD_ANCHOR;
            // all-static gate (joint.ts updateJointDual): mirrors the construction-time both-static
            // rejection (jointInit) — the world anchor counts as static, aWorld short-circuits so
            // solverStatic never reads the sentinel eid.
            if (A.solverStatic(b) && (aWorld || A.solverStatic(a))) return;
            const qA = std.select(A.bQuat(a), d.vec4f(0, 0, 0, 1), aWorld);
            const rA = J.jrec(jid, 1).xyz;
            const stiffLin = J.jrec(jid, 1).w;
            const rB = J.jrec(jid, 2).xyz;
            const stiffAng = J.jrec(jid, 2).w;
            const torqueArm = J.jTorqueArm(jid);
            const alpha = A.layout.$.params.alpha;

            let penLin = d.vec3f(J.jPenLin(jid));
            if (std.dot(penLin, penLin) > 0) {
                const pA = std.select(std.add(A.bPos(a), A.qRotateW(qA, rA)), rA, aWorld);
                const pB = std.add(A.bPos(b), A.qRotateW(A.bQuat(b), rB));
                let C = d.vec3f(std.sub(pA, pB));
                if (stiffLin > RIGID_THRESHOLD) {
                    C = std.sub(C, std.mul(J.jC0Lin(jid), alpha));
                    J.layout.$.jointRecords[recBase + 6] = d.vec4f(
                        std.add(std.mul(penLin, C), J.jLamLin(jid)),
                        0,
                    ); // λ ← K·C + λ
                }
                penLin = std.min(
                    std.add(penLin, std.mul(std.abs(C), A.layout.$.params.betaLin)),
                    d.vec3f(std.min(stiffLin, PENALTY_MAX)),
                );
                J.layout.$.jointRecords[recBase + 4] = d.vec4f(penLin, 0);
            }
            let penAng = d.vec3f(J.jPenAng(jid));
            if (std.dot(penAng, penAng) > 0) {
                let C = d.vec3f(std.mul(A.qSubW(qA, A.bQuat(b)), torqueArm));
                if (stiffAng > RIGID_THRESHOLD) {
                    C = std.sub(C, std.mul(J.jC0Ang(jid), alpha));
                    J.layout.$.jointRecords[recBase + 7] = d.vec4f(
                        std.add(std.mul(penAng, C), J.jLamAng(jid)),
                        0,
                    );
                }
                penAng = std.min(
                    std.add(penAng, std.mul(std.abs(C), A.layout.$.params.betaAng)),
                    d.vec3f(std.min(stiffAng, PENALTY_MAX)),
                );
                J.layout.$.jointRecords[recBase + 5] = d.vec4f(penAng, 0);
            }
            // motor dual — λ clamped to ±maxTorque; the penalty ramps toward PENALTY_MAX only while λ is
            // strictly inside the force bounds (solver.cpp's lambda-inside-bounds gate)
            const motorMax = J.jMotorMax(jid);
            if (motorMax > 0) {
                const axis = J.jMotorAxis(jid);
                const dB = std.dot(A.qSubW(A.bQuat(b), A.bInitQ(b)), axis);
                const dA = std.select(
                    std.dot(A.qSubW(A.bQuat(a), A.bInitQ(a)), axis),
                    d.f32(0),
                    aWorld,
                );
                const c = dB - dA - J.jMotorSpeed(jid) * A.layout.$.params.dt;
                const lam = std.clamp(
                    J.jMotorLam(jid) + J.jMotorPen(jid) * c,
                    std.neg(motorMax),
                    motorMax,
                );
                let pen = J.jMotorPen(jid);
                if (lam > std.neg(motorMax) && lam < motorMax) {
                    pen = std.min(pen + std.abs(c) * A.layout.$.params.betaAng, PENALTY_MAX);
                }
                J.layout.$.jointRecords[recBase + 11] = d.vec4f(J.jMotorSpeed(jid), lam, pen, 0);
            }
        })
        .$name("jointDualOne");

    return { jointDualOne };
}

/** the shared factory instantiations the typed kernels consume. The `roRo`-backed instance
 *  (`contactContrib`/`springContrib`/`jointContrib`/`solvePose`) is both CPU-differential-testable and the
 *  typed primal's real pipeline consumer. @internal */
const cfRoRo = contactMath(roRo);
const cfRoRw = contactMath(roRw);
export const solverRoRo = {
    ...cfRoRo,
    ...contribMath(roRo, jointRo, cfRoRo.contactForce),
};
/** the dual pass's variant: bodies read-only, manifolds read-write. */
export const solverRoRw = {
    ...cfRoRw,
    ...dualMath(roRw, cfRoRw.contactForce),
};
/** the joint-dual pass's variant: bodies read-only, joint records read-write. */
export const solverJointRw = jointDualMath(roRo, jointRw);

/**
 * every pass's WGSL, for the device-free structural tests + the emitted-WGSL differential
 * (`step.test.ts`). Each raw entry is the exact text its pipeline compiles; `aabb` is a fresh resolve of
 * the typed kernel rather than the pipeline's own (same text, its own namespace — the kernel splices no
 * raw WGSL, so a suffixed name would be self-consistent).
 * @internal
 */
export const stepWgsl = {
    aabb: (): string => tgpu.resolve([aabbKernel], { names: "strict" }),
    broadphase: (): string => tgpu.resolve([broadphaseKernel], { names: "strict" }),
    broadphaseSmall: (): string => tgpu.resolve([broadphaseSmallKernel], { names: "strict" }),
    collideBox: (): string => tgpu.resolve([collideBoxKernel], { names: "strict" }),
    collideRounded: (): string => tgpu.resolve([collideRoundedKernel], { names: "strict" }),
    collideHull: (): string => tgpu.resolve([collideHullKernel], { names: "strict" }),
    collideRoundedPoly: (): string => tgpu.resolve([collideRoundedPolyKernel], { names: "strict" }),
    inertial: (): string => tgpu.resolve([inertialKernel], { names: "strict" }),
    primal: (): string => tgpu.resolve([primalKernel], { names: "strict" }),
    commit: (): string => tgpu.resolve([commitKernel], { names: "strict" }),
    dual: (): string => tgpu.resolve([dualKernel], { names: "strict" }),
    solveLds: (): string => tgpu.resolve([solveLdsKernel], { names: "strict" }),
    coloring: (): string => tgpu.resolve([coloringKernel], { names: "strict" }),
    repair: (): string => tgpu.resolve([repairKernel], { names: "strict" }),
    jointInit: (): string => tgpu.resolve([jointInitKernel], { names: "strict" }),
    jointDual: (): string => tgpu.resolve([jointDualKernel], { names: "strict" }),
    velocity: (): string => tgpu.resolve([velocityKernel], { names: "strict" }),
    compose: (): string => tgpu.resolve([composeKernel], { names: "strict" }),
    csrCount: (): string => tgpu.resolve([csrCountKernel], { names: "strict" }),
    csrScatter: (): string => tgpu.resolve([csrScatterKernel], { names: "strict" }),
    csrColorSmall: (): string => tgpu.resolve([csrColorSmallKernel], { names: "strict" }),
    // the four size-parameterized passes: representative arguments, since the structural properties
    // (duplicate declarations, integer discipline, what each pass declares of the shared group) are
    // size-independent
    csrScan: (): string => csrScanWgsl(1024, 1024),
    packCount: (): string => packCountWgsl({ gen: 0, mask: 1 }, 1024),
    packScan: (): string => packScanWgsl(4, 1024),
    packScatter: (): string => packScatterWgsl({ gen: 0, mask: 1 }, 1024, 1024),
};

/** the shared accessor variants, for the structural tests: one instantiation per storage access pair.
 * @internal */
export const solverVariants = { roRo, roRw, rwRw };

// ── aabb: dense body oriented-box AABB → the bvh prim buffer (the broadphase BVH input) ──
// Each prim is the body's world-AABB [pos − e, pos + e], e = the oriented-box extent (boxExtent). 2
// vec4/prim (min.xyz+pad, max.xyz+pad — bvh/core prim layout). Only [0, count) is written; the build
// sentinel-pads the tail.
const aabbLayout = tgpu
    .bindGroupLayout({
        prims: { storage: d.arrayOf(d.vec4f), access: "mutable" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = live count, [1+d] = the d-th live eid
    })
    .$idx(0);

// one thread per dense slot in [0, count); prim index = the slot, body = the eid at eids[1+slot]. So
// prim s is body eids[1+s]'s box-AABB, and the broadphase maps a leaf's prim index back through eids.
const aabbKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= aabbLayout.$.eids[0]) return;
        const i = aabbLayout.$.eids[1 + slot];
        const p = roRo.bPos(i);
        // pad the prim by the speculative band (static skin) + the per-axis velocity sweep |vel|·dt (Phase
        // 4.8.4, the webphysics velocity-fattened-tree form) so the broadphase finds a fast approaching pair
        // before contact. The static skin is prim-only (the query box stays tight on it → combined static
        // slack = SPECULATIVE_DISTANCE); the velocity pad is on both prim + query → combined
        // ≈ (|vA|+|vB|)·dt ≥ |vRel|·dt.
        const e = std.add(
            std.add(roRo.boxExtent(i), d.vec3f(SPECULATIVE_DISTANCE)),
            std.mul(std.abs(roRo.bVelL(i)), roRo.layout.$.params.dt),
        );
        aabbLayout.$.prims[2 * slot] = d.vec4f(std.sub(p, e), 0);
        aabbLayout.$.prims[2 * slot + 1] = d.vec4f(std.add(p, e), 0);
    })
    .$name("aabbMain");

// ── broadphase per-candidate accumulate + block emit (shared by the descent + the small-N scan) ──
// TGSL functions over `nbr`/`nd2`/`count` — the mains' accumulator — threaded by `d.ref` pointer (arrays)
// and the `uniformLoad`-style widened-signature escape (the scalar `count`, which `d.ref` refuses). Both
// mains call the SAME two functions, so the ownership rule, the nearest-K + static-pin prune, the sort, and
// the block write are one source of truth — the small-N O(n²) scan differs ONLY in how it enumerates
// candidates, the precondition for warmstart carrying across a regime flip (identical blocks). Broadphase
// (descent) and broadphase-small (scan) each declare their own group-0 layout (different extra bindings —
// `nodes` vs `prims`), so this factory closes over whichever layout its caller passes — the
// factory-closure law, applied to a group-0 layout rather than the shared solver group.
const NbrArr = d.arrayOf(d.u32, PAIRS_PER_BODY);
const Nd2Arr = d.arrayOf(d.f32, PAIRS_PER_BODY);

/** the two group-0 layouts `broadOutput` closes over — identical `pairList`/`counters`/`eids` shape,
 *  differing only in `nodes` (the descent) vs `prims` (the small-N scan). */
type BroadLayout = typeof broadphaseLayout | typeof broadphaseSmallLayout;

function broadOutput(layout: BroadLayout) {
    const broadCandidateFn = tgpu
        .fn([d.u32, d.vec3f, d.u32, d.ptrFn(NbrArr), d.ptrFn(Nd2Arr), d.ptrFn(d.u32)])(
            (dOwn, pi, dj, nbr, nd2, count) => {
                "use gpu";
                const je = layout.$.eids[1 + dj];
                const staticJ = roRo.bMass(je) <= 0;
                // dOwn (dynamic) owns: every dyn-static pair, and a dyn-dyn pair only when dj has the
                // lower slot (so the higher-slot body owns it — emitted once).
                if (staticJ || dj < dOwn) {
                    const dc = std.sub(pi, roRo.bPos(je));
                    let d2 = std.dot(dc, dc);
                    if (staticJ) d2 = d.f32(-1); // pin: a big static's center is far, never evicted
                    if (count.$ < PAIRS_PER_BODY) {
                        nbr.$[count.$] = dj;
                        nd2.$[count.$] = d2;
                        count.$ = count.$ + 1;
                    } else {
                        // loud: a candidate exceeded the cap (now a graceful nearest-K drop)
                        std.atomicAdd(layout.$.counters[3], 1);
                        // farthest kept candidate (tie → higher slot, deterministic); replace only if the
                        // newcomer is nearer, so the dropped one is always the least-important (farthest).
                        let wi = d.u32(0);
                        let wd2 = nd2.$[0];
                        let wdj = nbr.$[0];
                        for (let w = d.u32(1); w < PAIRS_PER_BODY; w++) {
                            if (nd2.$[w] > wd2 || (nd2.$[w] === wd2 && nbr.$[w] > wdj)) {
                                wi = w;
                                wd2 = nd2.$[w];
                                wdj = nbr.$[w];
                            }
                        }
                        if (d2 < wd2 || (d2 === wd2 && dj < wdj)) {
                            // evicted a static — only if > K statics (the pin keeps this 0 with one ground)
                            if (roRo.bMass(layout.$.eids[1 + wdj]) <= 0)
                                std.atomicAdd(layout.$.counters[7], 1);
                            nbr.$[wi] = dj;
                            nd2.$[wi] = d2;
                        } else if (staticJ) {
                            std.atomicAdd(layout.$.counters[7], 1); // a static newcomer was dropped
                        }
                    }
                }
            },
        )
        .$name("broadCandidate");
    // the widened call signature is the uniformLoad / considerCapAxis typing gap (engine/utils/tgsl.ts):
    // `d.ref` refuses a scalar, so `count` (mutated only through this pointer) is passed bare — nbr/nd2 go
    // through real `d.ref` (arrays), so only the last param needs widening.
    const broadCandidate = broadCandidateFn as typeof broadCandidateFn &
        ((dOwn: number, pi: d.v3f, dj: number, nbr: unknown, nd2: unknown, count: number) => void);

    const broadEmit = tgpu
        .fn([d.u32, d.u32, d.ptrFn(NbrArr), d.u32])((i, blockBase, nbr, count) => {
            "use gpu";
            // insertion-sort the neighbor slots ascending (stable order ⇒ stable per-pair slot across frames)
            for (let s = d.u32(1); s < count; s++) {
                const key = nbr.$[s];
                let k = s;
                while (true) {
                    if (k === 0) break;
                    if (nbr.$[k - 1] <= key) break;
                    nbr.$[k] = nbr.$[k - 1];
                    k = k - 1;
                }
                nbr.$[k] = key;
            }
            // write the per-eid block: [0, count) the owned pairs (oriented a > b by eid),
            // [count, PAIRS_PER_BODY) cleared to INVALID so a stale record (a prior frame / a recycled
            // eid) is never read as a live pair.
            for (let k = d.u32(0); k < PAIRS_PER_BODY; k++) {
                if (k < count) {
                    const je = layout.$.eids[1 + nbr.$[k]];
                    layout.$.pairList[blockBase + k] = d.vec2u(std.max(i, je), std.min(i, je));
                } else {
                    layout.$.pairList[blockBase + k] = d.vec2u(INVALID_PAIR, INVALID_PAIR);
                }
            }
        })
        .$name("broadEmit");

    return { broadCandidate, broadEmit };
}

// ── broadphase: LBVH box-overlap descent → each live body's per-eid FIXED pair block ──
// One thread per dense body dOwn (the block owner) descends the BVH built over the sphere-AABBs and writes
// dOwn's overlapping neighbors into ITS OWN fixed block `pairList[eid·PAIRS_PER_BODY …]` (a cheap `vec2u`
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
const broadphaseLayout = tgpu
    .bindGroupLayout({
        nodes: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        pairList: { storage: d.arrayOf(d.vec2u), access: "mutable" },
        counters: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
    })
    .$idx(0);

const nodeLeft = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((n) => {
        "use gpu";
        return bitcastF32toU32(broadphaseLayout.$.nodes[2 * n].w);
    })
    .$name("nodeLeft");
const nodeRight = tgpu
    .fn(
        [d.u32],
        d.u32,
    )((n) => {
        "use gpu";
        return bitcastF32toU32(broadphaseLayout.$.nodes[2 * n + 1].w);
    })
    .$name("nodeRight");
const nodeMin = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((n) => {
        "use gpu";
        return broadphaseLayout.$.nodes[2 * n].xyz;
    })
    .$name("nodeMin");
const nodeMax = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((n) => {
        "use gpu";
        return broadphaseLayout.$.nodes[2 * n + 1].xyz;
    })
    .$name("nodeMax");
const aabbOverlap = tgpu
    .fn(
        [d.vec3f, d.vec3f, d.u32],
        d.bool,
    )((qmin, qmax, n) => {
        "use gpu";
        return std.all(std.le(qmin, nodeMax(n))) && std.all(std.ge(qmax, nodeMin(n)));
    })
    .$name("aabbOverlap");

const { broadCandidate: descendCandidate, broadEmit: descendEmit } = broadOutput(broadphaseLayout);

const broadphaseKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const dOwn = input.gid.x; // this body's dense slot
        if (dOwn >= broadphaseLayout.$.eids[0]) return;
        const i = broadphaseLayout.$.eids[1 + dOwn]; // i = this body's eid
        const blockBase = i * d.u32(PAIRS_PER_BODY); // the owner-EID fixed block — the stable warmstart slot
        // static bodies own no pair — the dynamic partner owns every dyn-static pair (so a ground touching
        // N boxes never owns a huge block). Clear the block to INVALID + return.
        if (roRo.bMass(i) <= 0) {
            for (let k = d.u32(0); k < PAIRS_PER_BODY; k++) {
                broadphaseLayout.$.pairList[blockBase + k] = d.vec2u(INVALID_PAIR, INVALID_PAIR);
            }
            return;
        }
        const pi = roRo.bPos(i);
        // the query box is the tight box-extent + the velocity sweep |vel|·dt (Phase 4.8.4); the static
        // SPECULATIVE_DISTANCE skin lives on the prim only (the aabb pass), so combined static slack stays
        // SPECULATIVE_DISTANCE while the velocity slack ≈ (|vA|+|vB|)·dt covers the relative closing.
        const ei = std.add(
            roRo.boxExtent(i),
            std.mul(std.abs(roRo.bVelL(i)), roRo.layout.$.params.dt),
        );
        const qmin = std.sub(pi, ei);
        const qmax = std.add(pi, ei);

        // collect dOwn's owned neighbors as the K NEAREST by center-dist² (importance prune, webphysics
        // broadPhase.ts ~560-694): when a tight pile gives a body > PAIRS_PER_BODY band-neighbors, keep the
        // nearest (the ones that actually become contacts) and drop the farthest, NOT an arbitrary
        // BVH-traversal-order drop — that drop can evict a real support, the root of the dense-pile churn
        // AND the static-ground fall-through. Static supports are pinned (d2 = −1, never the farthest) so
        // the ground is never evicted. Sorted ascending below for a stable per-pair slot.
        const nbr = NbrArr();
        const nd2 = Nd2Arr();
        let count = d.u32(0);
        count = 0; // force `var` (pointerDiscipline) — mutated only through broadCandidate's pointer

        const stack = d.arrayOf(d.u32, 64)();
        let sp = d.u32(0);
        let node = bvhRoot(broadphaseLayout.$.eids[0]);
        while (true) {
            if (nodeLeft(node) === BVH_INVALID) {
                // leaf — overlap confirmed by the descent above
                const dj = nodeRight(node); // neighbor's dense slot (prim index)
                if (dj !== dOwn) descendCandidate(dOwn, pi, dj, d.ref(nbr), d.ref(nd2), count);
                if (sp === 0) break;
                sp = sp - 1;
                node = stack[sp];
                continue;
            }
            const l = nodeLeft(node);
            const r = nodeRight(node);
            const okL = aabbOverlap(qmin, qmax, l);
            const okR = aabbOverlap(qmin, qmax, r);
            if (okL && okR) {
                if (sp < 64) {
                    stack[sp] = r;
                    sp = sp + 1;
                }
                node = l;
            } else if (okL) {
                node = l;
            } else if (okR) {
                node = r;
            } else {
                if (sp === 0) break;
                sp = sp - 1;
                node = stack[sp];
            }
        }

        descendEmit(i, blockBase, d.ref(nbr), count);
    })
    .$name("broadphaseMain");

// ── broadphase (small-N regime, C1.0): one-dispatch O(n²) scan → the same per-eid FIXED pair blocks ──
// At a live count ≤ the smallN threshold, record() replaces the whole BVH build + descent (~29 dependent
// phases of near-pure structure tax at gameplay counts) with this single
// dispatch: each live body's lane scans the dense live set against the SAME aabb-pass prims the BVH leaves
// carry, so the overlap test, pads, ownership, prune, and block write are identical to the descent and the
// pair blocks come out byte-identical — warmstart carries across a regime flip. O(n²) is exact at any N
// (only slow past the threshold), so the frame-stale regime switch is correctness-safe in both directions.
const TILE = 64;

const broadphaseSmallLayout = tgpu
    .bindGroupLayout({
        prims: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        pairList: { storage: d.arrayOf(d.vec2u), access: "mutable" },
        counters: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
    })
    .$idx(0);

// the n-body LDS tiling (Bullet 3 / GPU-gems n² pattern): each round the workgroup cooperatively stages
// TILE prims into workgroup memory, then every lane tests its own query box against the staged tile — a
// naive per-lane serial scan of global prims is latency-bound (measured 0.2–0.4 ms at 1k, worse than the
// BVH front-end it replaces), the staged form reads each prim from global memory once per workgroup.
const tMin = tgpu.workgroupVar(d.arrayOf(d.vec4f, TILE));
const tMax = tgpu.workgroupVar(d.arrayOf(d.vec4f, TILE));
const wgN = tgpu.workgroupVar(d.u32);

const { broadCandidate: scanCandidate, broadEmit: scanEmit } = broadOutput(broadphaseSmallLayout);

const broadphaseSmallKernel = tgpu
    .computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
    })((input) => {
        "use gpu";
        // every lane must reach the tile barriers, so out-of-range / static lanes stay in the loop as
        // inactive rather than returning. workgroupUniformLoad makes the live count uniform for Tint's
        // uniformity analysis (eids[0] is workgroup-uniform in fact, but a raw storage read can't prove it).
        if (input.lid.x === 0) wgN.$ = broadphaseSmallLayout.$.eids[0];
        const n = uniformLoad(wgN.$);
        const dOwn = input.gid.x;
        const inRange = dOwn < n;
        // i = this body's eid (0 placeholder when idle)
        const i = std.select(0, broadphaseSmallLayout.$.eids[1 + std.min(dOwn, n - 1)], inRange);
        const blockBase = i * d.u32(PAIRS_PER_BODY); // the owner-EID fixed block — the stable warmstart slot
        // static bodies own no pair — the dynamic partner owns every dyn-static pair (so a ground touching
        // N boxes never owns a huge block); their block is cleared to INVALID by the emit below (count stays 0).
        const act = inRange && roRo.bMass(i) > 0;
        const pi = roRo.bPos(i);
        // own query box = own prim shrunk by the static skin: the prim is pos ± (boxExtent +
        // SPECULATIVE_DISTANCE + |vel|·dt), and the skin is prim-only (the descent's query stays tight on
        // it), so shrinking recovers exactly the descent's query box — combined static slack stays
        // SPECULATIVE_DISTANCE, velocity slack ≈ (|vA|+|vB|)·dt. A candidate's prim is what the BVH leaf
        // bounds were, so the test below is the descent's leaf test verbatim.
        const dq = std.select(0, dOwn, inRange); // idle tail lanes read prim 0 (unused) — keeps access in bounds
        const qmin = std.add(
            broadphaseSmallLayout.$.prims[2 * dq].xyz,
            d.vec3f(d.f32(SPECULATIVE_DISTANCE)),
        );
        const qmax = std.sub(
            broadphaseSmallLayout.$.prims[2 * dq + 1].xyz,
            d.vec3f(d.f32(SPECULATIVE_DISTANCE)),
        );

        const nbr = NbrArr();
        const nd2 = Nd2Arr();
        let count = d.u32(0);
        count = 0; // force `var` (pointerDiscipline) — mutated only through broadCandidate's pointer

        for (let base = d.u32(0); base < n; base = base + TILE) {
            const src = base + input.lid.x;
            if (src < n) {
                tMin.$[input.lid.x] = d.vec4f(broadphaseSmallLayout.$.prims[2 * src]);
                tMax.$[input.lid.x] = d.vec4f(broadphaseSmallLayout.$.prims[2 * src + 1]);
            }
            std.workgroupBarrier();
            const len = std.min(TILE, n - base);
            if (act) {
                for (let t = d.u32(0); t < len; t++) {
                    const dj = base + t;
                    if (dj === dOwn) continue;
                    if (
                        !(
                            std.all(std.le(qmin, tMax.$[t].xyz)) &&
                            std.all(std.ge(qmax, tMin.$[t].xyz))
                        )
                    )
                        continue;
                    scanCandidate(dOwn, pi, dj, d.ref(nbr), d.ref(nd2), count);
                }
            }
            std.workgroupBarrier();
        }
        if (!inRange) return;
        scanEmit(i, blockBase, d.ref(nbr), count);
    })
    .$name("broadphaseSmallMain");

// ── narrowphase (collide): shape-pair SAT over the per-eid pair blocks; in-place warmstart ──
// One thread per pair SLOT in the live bodies' fixed blocks: the dispatch is `liveCount · PAIRS_PER_BODY`
// lanes, lane → (dOwn = lane/K, k = lane%K), owner eid = eids[1+dOwn], slot = eid·K + k (the fixed per-eid
// block base — webphysics `bodyBase`). The slot's manifold lives at recBase = slot*CONTACTS_PER_PAIR in
// `pairContacts`, persistent across frames; because the base is the owner's eid (not a prefix-sum offset),
// the slot is STABLE unless the owner's own candidate set flickers — local warmstart fragility, not the
// global collapse a compaction has (avbd.md "Storage"). Per slot: read pairList[slot]; an
// INVALID (unused) slot or a separating pair clears the block (kind 0, so the solve skips it + it cold-starts
// next frame, matching the oracle dropping a 0-contact manifold) — gated by `ownsLifecycle` (only the box
// pipeline owns dead-slot clearing). A 0-contact SAT result always clears (every pipeline reaching that
// point owns this pair for the frame, by the class gate). Otherwise, for each fresh contact scan THIS
// SLOT's prev records (read before any write — `mergeWarmstart`) for a record with the SAME pair (a,b) +
// feature key, and carry its λ/k decayed (Eq. 19) + sticking arms (manifold.ts initManifold); `writeContacts`
// then overwrites the slot's records in place. The sphere test (dot(dp,dp) <= r², r = radiusA + radiusB)
// filters the AABB-overlap superset back to the exact reference contact set.
// Built as FOUR kernels by shape-pair class — box×box, rounded×rounded, polytope×polytope (collideHull),
// and rounded×polytope (collideRoundedPolytope) — each calling its SAT fn directly rather than splicing a
// chunk: `tgpu.resolve` walks each kernel's own call graph, so a box-only kernel never pulls in the hull
// SAT's `hullData` dependency (the DXC pipeline split falls out of one resolve per kernel, not manual
// chunk composition). All four dispatch over the SAME live pair slots (indirect off pairArgs) and act only
// on their class; the BOX kernel OWNS slot lifecycle — it clears every INVALID + separated slot regardless
// of class (the sphere filter is shape-aware), so the other three only fill/clear their OWN class's live
// slots and every slot is written exactly once (the gates are mutually exclusive + exhaustive). The cost is
// 3 extra cheap early-out dispatches per step (collide runs once per fixed step, not per iteration), traded
// for not needing a per-class partition pass.
const collideLayout = tgpu
    .bindGroupLayout({
        counters: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        pairList: { storage: d.arrayOf(d.vec2u), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        hullData: { storage: d.arrayOf(d.u32), access: "readonly" }, // packed convex-hull geometry (./hull packHulls)
    })
    .$idx(0);

/**
 * a real TGSL reference into `hullData`, called (its result discarded) by the two kernels that reach the
 * hull SAT. `collideHull`/`collideRoundedPolytope` reach `hullData` only through the WGSL-BODIED hull-geometry
 * readers (`hullRef`/`hVertL`/…, collide.ts), which name it as a free identifier invisible to
 * `tgpu.resolve`'s dependency walk (no `.$uses` can carry an external binding) — so nothing in either
 * kernel's traced call graph would otherwise make the resolver emit `hullData`'s own binding declaration.
 * This is the one genuine reference that does — an inert `& 0` fold keeps the add a true no-op, routed to
 * `counters[15]` (reserved, unread by any real logic — avbd.md's `counters[0..7]` are all live gauges) so
 * the touch adds no atomic contention on a counter every lane actually contends on.
 */
const touchHullData = tgpu
    .fn([d.u32])((i) => {
        "use gpu";
        std.atomicAdd(collideLayout.$.counters[15], collideLayout.$.hullData[i] & 0);
    })
    .$name("touchHullData");

// clear a slot's whole manifold block to inactive (kind 0) — the solve + warmstart skip a 0-meta record.
const clearBlock = tgpu
    .fn([d.u32])((recBase) => {
        "use gpu";
        const rc = roRw.layout.$.params.recordCap;
        for (let s = d.u32(0); s < CONTACTS_PER_PAIR; s++) {
            roRw.layout.$.pairContacts[C_META * rc + recBase + s] = d.vec4f(0);
        }
    })
    .$name("clearBlock");

/** the prelude every shape-pair class shares: the pair's poses/shapes + the sphere-filter separation test
 *  (Phase 4.8.3/4.8.4). `live: false` means separated past the band — the caller clears (box) or returns
 *  (the other three). qa/qb/sa/sb are pure reads with no side effect, so reading them unconditionally
 *  (rather than only on the live branch) keeps one construction site — a deviation from the reference's
 *  read order, not its behavior. */
const CollidePrelude = d
    .struct({
        live: d.bool,
        pa: d.vec3f,
        qa: d.vec4f,
        sa: d.u32,
        pb: d.vec3f,
        qb: d.vec4f,
        sb: d.u32,
        roundedA: d.bool,
        roundedB: d.bool,
        dRel: d.vec3f,
    })
    .$name("CollidePrelude");

const collidePrelude = tgpu
    .fn(
        [d.u32, d.u32],
        CollidePrelude,
    )((ia, ib) => {
        "use gpu";
        const pa = roRw.bPos(ia);
        const pb = roRw.bPos(ib);
        // the bounding-sphere radius is shape-aware: core span + the rounding radius (Phase 6.3). A box's
        // radius is 0, so this is length(bHalf) unchanged.
        const ra = std.length(roRw.bHalf(ia)) + roRw.bRadius(ia);
        const rb = std.length(roRw.bHalf(ib)) + roRw.bRadius(ib);
        const dp = std.sub(pa, pb);
        // dRel = (vA−vB)·dt — the velocity sweep (Phase 4.8.4): widens the sphere filter + the SAT band so
        // a fast approaching pair reaches the SAT and generates its swept contact (mirrors the C++ + oracle).
        const dRel = std.mul(std.sub(roRw.bVelL(ia), roRw.bVelL(ib)), roRw.layout.$.params.dt);
        // the sphere filter is the reference broadphase (mirrors the C++ + oracle); the static band (Phase
        // 4.8.3) + the relative sweep |vRel|·dt (4.8.4) let a pair within reach this step pass to the SAT.
        const r = ra + rb + d.f32(SPECULATIVE_DISTANCE) + std.length(dRel);
        const qa = roRw.bQuat(ia);
        const qb = roRw.bQuat(ib);
        const sa = roRw.bShape(ia);
        const sb = roRw.bShape(ib);
        const roundedA = sa === 1 || sa === 2;
        const roundedB = sb === 1 || sb === 2;
        const live = std.dot(dp, dp) <= r * r;
        return CollidePrelude({ live, pa, qa, sa, pb, qb, sb, roundedA, roundedB, dRel });
    })
    .$name("collidePrelude");

/** the per-contact warmstart merge (loop 1, avbd.md "Storage"): per fresh SAT contact, scan THIS slot's
 *  prev records for the same pair (a,b) + feature key and carry its λ/k decayed + sticking arms. */
const Warmstart = d
    .struct({
        lam: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR),
        pen: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR),
        rA: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR),
        rB: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR),
        merged: d.u32,
    })
    .$name("Warmstart");

const mergeWarmstart = tgpu
    .fn(
        [SatResult, d.u32, d.u32, d.bool, d.bool, d.u32],
        Warmstart,
    )((sat, ia, ib, roundedA, roundedB, recBase) => {
        "use gpu";
        const out = Warmstart({
            lam: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR)(),
            pen: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR)(),
            rA: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR)(),
            rB: d.arrayOf(d.vec3f, CONTACTS_PER_PAIR)(),
            merged: 0,
        });
        for (let k = d.u32(0); k < sat.count; k++) {
            const feat = sat.feat[k];
            let lam3 = d.vec3f(0);
            let pen3 = d.vec3f(roRw.layout.$.params.penalty);
            let rA = d.vec3f(sat.rA[k]);
            let rB = d.vec3f(sat.rB[k]);
            for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
                const wm = roRw.cc(recBase + ls, C_META);
                if (
                    bitcastF32toU32(wm.x) === CONSTRAINT_CONTACT &&
                    bitcastF32toU32(wm.y) === ia &&
                    bitcastF32toU32(wm.z) === ib &&
                    bitcastF32toU32(wm.w) === feat
                ) {
                    const oldLam = roRw.cc(recBase + ls, C_LAMBDA);
                    lam3 = std.mul(
                        oldLam.xyz,
                        roRw.layout.$.params.alpha * roRw.layout.$.params.gamma,
                    );
                    pen3 = std.clamp(
                        std.mul(roRw.cc(recBase + ls, C_PEN).xyz, roRw.layout.$.params.gamma),
                        d.vec3f(PENALTY_MIN),
                        d.vec3f(PENALTY_MAX),
                    );
                    // a sticking contact keeps its frozen arms ONLY for box-box pairs, where the feature key
                    // identifies a persistent vertex/edge. Any pair INVOLVING a rounded (sphere/capsule)
                    // shape — even vs a box — has a sliding closest point under a constant feature key, so
                    // freezing its arm anchors a stale point: any spin rotates the frozen arm → a tangential
                    // c0 → torque → runaway spin (Phase 6.3). A rounded shape re-collides fresh arms vs a
                    // box too.
                    if (oldLam.w > 0.5 && !(roundedA || roundedB)) {
                        rA = roRw.cc(recBase + ls, C_RA).xyz;
                        rB = roRw.cc(recBase + ls, C_RB).xyz;
                    }
                    out.merged = out.merged + 1;
                    break;
                }
            }
            out.lam[k] = d.vec3f(lam3);
            out.pen[k] = d.vec3f(pen3);
            out.rA[k] = d.vec3f(rA);
            out.rB[k] = d.vec3f(rB);
        }
        return out;
    })
    .$name("mergeWarmstart");

/** loop 2: overwrite the slot's records in place — [0, sat.count) live, the rest inactive (kind 0) */
const writeContacts = tgpu
    .fn([SatResult, Warmstart, d.u32, d.u32, d.u32, d.vec3f, d.vec4f, d.vec3f, d.vec4f, d.f32])(
        (sat, w, ia, ib, recBase, pa, qa, pb, qb, friction) => {
            "use gpu";
            const rc = roRw.layout.$.params.recordCap;
            const n0 = sat.basis.r0;
            for (let k = d.u32(0); k < CONTACTS_PER_PAIR; k++) {
                const rec = recBase + k;
                if (k < sat.count) {
                    const rA = w.rA[k];
                    const rB = w.rB[k];
                    // the arms anchor the CORE feature point; the contact surface is offset ±radius along
                    // the normal (rounded narrowphase, Phase 6.3 — keeps the radius part geometric, off the
                    // spin). Reconstruct the surface points for the true gap. A box has radius 0, so this
                    // is the bare arm.
                    const xA = std.sub(
                        std.add(pa, roRw.qRotateW(qa, rA)),
                        std.mul(n0, roRw.bRadius(ia)),
                    );
                    const xB = std.add(
                        std.add(pb, roRw.qRotateW(qb, rB)),
                        std.mul(n0, roRw.bRadius(ib)),
                    );
                    const dlt = std.sub(xA, xB);
                    const c0 = d.vec3f(
                        std.dot(sat.basis.r0, dlt) + d.f32(COLLISION_MARGIN),
                        std.dot(sat.basis.r1, dlt),
                        std.dot(sat.basis.r2, dlt),
                    );
                    roRw.layout.$.pairContacts[C_META * rc + rec] = d.vec4f(
                        std.bitcastU32toF32(CONSTRAINT_CONTACT),
                        std.bitcastU32toF32(ia),
                        std.bitcastU32toF32(ib),
                        std.bitcastU32toF32(sat.feat[k]),
                    );
                    roRw.layout.$.pairContacts[C_NORMAL * rc + rec] = d.vec4f(n0, 0);
                    roRw.layout.$.pairContacts[C_RA * rc + rec] = d.vec4f(rA, 0);
                    roRw.layout.$.pairContacts[C_RB * rc + rec] = d.vec4f(rB, 0);
                    roRw.layout.$.pairContacts[C_C0 * rc + rec] = d.vec4f(c0, 0);
                    roRw.layout.$.pairContacts[C_PEN * rc + rec] = d.vec4f(w.pen[k], friction);
                    roRw.layout.$.pairContacts[C_LAMBDA * rc + rec] = d.vec4f(w.lam[k], 0);
                } else {
                    roRw.layout.$.pairContacts[C_META * rc + rec] = d.vec4f(0); // inactive
                }
            }
        },
    )
    .$name("writeContacts");

// box×box — the common case + the slot-lifecycle owner (clears INVALID + separated slots for every class).
const collideBoxKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= collideLayout.$.eids[0]) return; // past the live body count (the tail of the last workgroup)
        const slot =
            collideLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY);
        const recBase = slot * d.u32(CONTACTS_PER_PAIR);
        const pair = collideLayout.$.pairList[slot];
        if (pair.x === INVALID_PAIR) {
            clearBlock(recBase); // INVALID — unused block slot; box owns dead-slot clearing
            return;
        }
        const ia = pair.x;
        const ib = pair.y;
        const p = collidePrelude(ia, ib);
        if (!p.live) {
            clearBlock(recBase); // separated past the band → cold next frame
            return;
        }
        if (!(p.sa === 0 && p.sb === 0)) return; // class gate: box×box only
        const sat = collideBoxBox(
            p.pa,
            p.qa,
            std.mul(roRw.bHalf(ia), 2),
            p.pb,
            p.qb,
            std.mul(roRw.bHalf(ib), 2),
            p.dRel,
        );
        if (sat.count === 0) {
            clearBlock(recBase);
            return;
        }
        const w = mergeWarmstart(sat, ia, ib, p.roundedA, p.roundedB, recBase);
        const friction = std.sqrt(roRw.bFriction(ia) * roRw.bFriction(ib));
        writeContacts(sat, w, ia, ib, recBase, p.pa, p.qa, p.pb, p.qb, friction);
        std.atomicAdd(collideLayout.$.counters[0], sat.count); // total active contacts (the GPU correctness gates read this)
        if (w.merged > 0) std.atomicAdd(collideLayout.$.counters[6], w.merged); // warmstarted-contact count
    })
    .$name("collideBoxMain");

// rounded×rounded — sphere/capsule pairs (one segment-segment closest point).
const collideRoundedKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= collideLayout.$.eids[0]) return;
        const slot =
            collideLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY);
        const recBase = slot * d.u32(CONTACTS_PER_PAIR);
        const pair = collideLayout.$.pairList[slot];
        if (pair.x === INVALID_PAIR) return;
        const ia = pair.x;
        const ib = pair.y;
        const p = collidePrelude(ia, ib);
        if (!p.live) return;
        if (!(p.roundedA && p.roundedB)) return; // class gate: both rounded only
        const sat = collideRounded(
            p.pa,
            p.qa,
            std.mul(roRw.bHalf(ia), 2),
            roRw.bRadius(ia),
            p.pb,
            p.qb,
            std.mul(roRw.bHalf(ib), 2),
            roRw.bRadius(ib),
            p.dRel,
        );
        if (sat.count === 0) {
            clearBlock(recBase);
            return;
        }
        const w = mergeWarmstart(sat, ia, ib, p.roundedA, p.roundedB, recBase);
        const friction = std.sqrt(roRw.bFriction(ia) * roRw.bFriction(ib));
        writeContacts(sat, w, ia, ib, recBase, p.pa, p.qa, p.pb, p.qb, friction);
        std.atomicAdd(collideLayout.$.counters[0], sat.count);
        if (w.merged > 0) std.atomicAdd(collideLayout.$.counters[6], w.merged);
    })
    .$name("collideRoundedMain");

// hull — box×hull, hull×hull (collideHull). The polytope×polytope SAT, its own kernel (the 4-way split,
// gpu.md "DXC shader compilation"): collideHull's big face/edge SAT resolves apart from the rounded
// segment-clip below, so neither pipeline pays the other's superlinear compile cost (the combined hull
// kernel was the standing ~920 ms long pole). Gate: both non-rounded, not both box ⇒ at least one hull.
const collideHullKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= collideLayout.$.eids[0]) return;
        const slot =
            collideLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY);
        const recBase = slot * d.u32(CONTACTS_PER_PAIR);
        const pair = collideLayout.$.pairList[slot];
        if (pair.x === INVALID_PAIR) return;
        const ia = pair.x;
        const ib = pair.y;
        const p = collidePrelude(ia, ib);
        if (!p.live) return;
        if (!(!p.roundedA && !p.roundedB && !(p.sa === 0 && p.sb === 0))) return; // at least one hull
        touchHullData(0); // forces hullData's binding into scope (see touchHullData)
        const sat = collideHull(
            polyMake(p.sa, p.pa, p.qa, std.mul(roRw.bHalf(ia), 2), roRw.bHullId(ia)),
            polyMake(p.sb, p.pb, p.qb, std.mul(roRw.bHalf(ib), 2), roRw.bHullId(ib)),
            p.dRel,
        );
        if (sat.count === 0) {
            clearBlock(recBase);
            return;
        }
        const w = mergeWarmstart(sat, ia, ib, p.roundedA, p.roundedB, recBase);
        const friction = std.sqrt(roRw.bFriction(ia) * roRw.bFriction(ib));
        writeContacts(sat, w, ia, ib, recBase, p.pa, p.qa, p.pb, p.qb, friction);
        std.atomicAdd(collideLayout.$.counters[0], sat.count);
        if (w.merged > 0) std.atomicAdd(collideLayout.$.counters[6], w.merged);
    })
    .$name("collideHullMain");

// rounded×polytope — sphere/capsule vs box/hull (collideRoundedPolytope). The other half of the old hull
// kernel, its own pipeline. Gate: exactly one shape is rounded. Mutually exclusive with the box/rounded/hull
// gates, so every live slot is still written by exactly one pipeline.
const collideRoundedPolyKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= collideLayout.$.eids[0]) return;
        const slot =
            collideLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY);
        const recBase = slot * d.u32(CONTACTS_PER_PAIR);
        const pair = collideLayout.$.pairList[slot];
        if (pair.x === INVALID_PAIR) return;
        const ia = pair.x;
        const ib = pair.y;
        const p = collidePrelude(ia, ib);
        if (!p.live) return;
        if (p.roundedA === p.roundedB) return; // exactly one shape is rounded
        touchHullData(0); // forces hullData's binding into scope (see touchHullData)
        const sat = collideRoundedPolytope(
            p.pa,
            p.qa,
            std.mul(roRw.bHalf(ia), 2),
            roRw.bRadius(ia),
            p.sa,
            roRw.bHullId(ia),
            p.pb,
            p.qb,
            std.mul(roRw.bHalf(ib), 2),
            roRw.bRadius(ib),
            p.sb,
            roRw.bHullId(ib),
            p.dRel,
        );
        if (sat.count === 0) {
            clearBlock(recBase);
            return;
        }
        const w = mergeWarmstart(sat, ia, ib, p.roundedA, p.roundedB, recBase);
        const friction = std.sqrt(roRw.bFriction(ia) * roRw.bFriction(ib));
        writeContacts(sat, w, ia, ib, recBase, p.pa, p.qa, p.pb, p.qb, friction);
        std.atomicAdd(collideLayout.$.counters[0], sat.count);
        if (w.merged > 0) std.atomicAdd(collideLayout.$.counters[6], w.merged);
    })
    .$name("collideRoundedPolyMain");

// ── inertial: inertial target (Eq. 2) + adaptive warmstart reposition (solver.cpp step 3) ──
const inertialLayout = tgpu
    .bindGroupLayout({ eids: { storage: d.arrayOf(d.u32), access: "readonly" } })
    .$idx(0);

const inertialKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= inertialLayout.$.eids[0]) return;
        const i = inertialLayout.$.eids[1 + slot];
        const dt = rwRw.layout.$.params.dt;
        const g = rwRw.layout.$.params.gravity;
        const dynamic = !rwRw.solverStatic(i);

        const pos = rwRw.bPos(i);
        const quat = rwRw.bQuat(i);
        const vel = rwRw.bVelL(i);
        const velA = rwRw.bVelA(i);
        const prevV = rwRw.bPrevV(i);

        // a static / kinematic body (solverStatic): no gravity in the inertial target, no
        // warmstart reposition — frozen inertial = initial = current pose, so dq = 0 (oracle solver.ts).
        // inertial target: full gravity (Eq. 2). The warmstart-start pos reuses the same predicted
        // step, scaling only the gravity term by accelWeight; the angular warmstart equals inertialQ.
        const predicted = std.add(pos, std.mul(vel, dt));
        const inertialQ = rwRw.qAddW(quat, std.mul(velA, dt));
        let inertialL = d.vec3f(predicted); // copy — `predicted` is itself a reference, needs a real `var`
        if (dynamic) inertialL = std.add(inertialL, d.vec3f(0, g * dt * dt, 0));

        // adaptive accelWeight (VBD): scales the warmstart-start gravity term, not the inertial target
        const accel = std.div(std.sub(vel, prevV), dt);
        let accelWeight = std.clamp((accel.y * std.sign(g)) / std.abs(g), 0, 1);
        // biome-ignore lint/suspicious/noSelfCompare: the WGSL NaN idiom (x != x), transpiled verbatim
        if (accelWeight !== accelWeight) accelWeight = 0; // NaN -> 0

        const cap = rwRw.layout.$.params.eidCap;
        rwRw.layout.$.bodies[B_INERTL * cap + i] = d.vec4f(inertialL, 0);
        rwRw.layout.$.bodies[B_INERTQ * cap + i] = d.vec4f(inertialQ); // copy — reference-yielding call result
        // initialLin = x⁻ (the contact constraint reads this as dq=0)
        rwRw.layout.$.bodies[B_INITL * cap + i] = d.vec4f(pos, 0);
        rwRw.layout.$.bodies[B_INITQ * cap + i] = d.vec4f(quat); // initialAng

        if (dynamic) {
            const warmPos = std.add(predicted, d.vec3f(0, g * accelWeight * dt * dt, 0));
            rwRw.layout.$.bodies[B_POS * cap + i] = d.vec4f(warmPos, 0);
            rwRw.layout.$.bodies[B_QUAT * cap + i] = d.vec4f(inertialQ);
        }
    })
    .$name("inertialMain");

/** the color index commit + primal read at a per-color STATIC bind group (the locked design):
 *  typegpu 0.11.9 has no dynamic-offset uniform binding, so `COLOR_CAP` one-buffer
 *  immutable groups replace the raw dynamic-offset UBO — one physical buffer each dodges the 256-B
 *  256-byte `minUniformBufferOffsetAlignment` rule. */
const ColorIdx = d.struct({ value: d.u32 }).$name("ColorIdx");

// ── primal: colored Gauss-Seidel, contact-force stamp + 6×6 LDLᵀ → the double-buffer scratch ──
// The primal reads the committed `bodies` (read-only) and writes each solved body's new pose into the
// `solveOut` scratch, NOT back into `bodies`; the commit pass applies it for the current color. So within
// one color's dispatch no body's `bodies` slot is both read (by a contact) and written, and a same-color
// pair reduces to the paper's deferred-within-color (clean Jacobi), not a racy write-in-place.
//
// Own I/O splits across two groups: `primalOwnLayout` (group 0, unchanged — csr/csrList/
// constraintCsr/constraintList, forced by `solverRoRo.solvePose`'s internal reads, so the primal needs no
// explicit reference of its own) + `primalColorLayout` (group {@link PRIMAL_COLOR_GROUP} — colors/eids/
// solveOut + the per-color static uniform, group 0's commit shape repeated since group 0 is taken here).
export const PRIMAL_COLOR_GROUP = 3;

const primalColorLayout = tgpu
    .bindGroupLayout({
        colors: { storage: d.arrayOf(d.u32), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        solveOut: { storage: d.arrayOf(d.vec4f), access: "mutable" },
        color: { uniform: ColorIdx },
    })
    .$idx(PRIMAL_COLOR_GROUP);

const primalKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= primalColorLayout.$.eids[0]) return;
        const bid = primalColorLayout.$.eids[1 + slot];
        if (roRo.solverStatic(bid)) return; // static / kinematic — no primal
        if (primalColorLayout.$.colors[bid] !== primalColorLayout.$.color.value) return; // colored GS
        const np = solverRoRo.solvePose(bid);
        // double-buffer: write the solved pose to the scratch, not back into bodies — bodies is read-only
        // in this pass, so a same-color contact pair never races on the pose; the commit applies it per color.
        const cap = roRo.layout.$.params.eidCap;
        primalColorLayout.$.solveOut[SO_POS * cap + bid] = d.vec4f(np.pos, 0);
        primalColorLayout.$.solveOut[SO_QUAT * cap + bid] = d.vec4f(np.quat);
    })
    .$name("primalMain");

// ── commit: apply the current color's solved poses from the scratch into `bodies` (the deferred commit) ──
// The write half of the double-buffer (paper Algorithm 1 lines 22-24, webphysics `commitBodySolveKernel`).
// One dispatch after each color's primal: copy `solveOut` → `bodies` for the bodies of `color.x` only, so
// the next color's primal reads the committed pose. Gated identically to the primal (live count, static,
// color), so every body it writes the primal just solved into `solveOut` this color — no stale read, no
// clear needed. A same-color pair is now a clean Jacobi: both solved from the color-start pose, committed
// together, matching the oracle's `primalColored`.
const commitLayout = tgpu
    .bindGroupLayout({
        solveOut: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        colors: { storage: d.arrayOf(d.u32), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        color: { uniform: ColorIdx },
    })
    .$idx(0);

const commitKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= commitLayout.$.eids[0]) return;
        const bid = commitLayout.$.eids[1 + slot];
        if (rwRw.solverStatic(bid)) return; // static / kinematic — no primal, no commit
        if (commitLayout.$.colors[bid] !== commitLayout.$.color.value) return; // only this color commits
        const cap = rwRw.layout.$.params.eidCap;
        rwRw.layout.$.bodies[B_POS * cap + bid] = d.vec4f(
            commitLayout.$.solveOut[SO_POS * cap + bid],
        );
        rwRw.layout.$.bodies[B_QUAT * cap + bid] = d.vec4f(
            commitLayout.$.solveOut[SO_QUAT * cap + bid],
        );
    })
    .$name("commitMain");

// ── dual: λ ← F + the conditional penalty ramp + the friction stick flag (manifold.ts updateDual) ──
// One dispatch over the contact rows, once per iteration after the primal colors.
// Reads the post-primal pose, recomputes the same cone-clamped force the primal used, stores it as
// λ, and ramps the penalty: the normal stiffness only while the contact is active (F[0] < 0), the
// tangent stiffness only inside the friction cone (sticking). The stick flag rides λ.w for the
// Phase-3 warmstart merge (unused within a frame). Each contact is independent — one thread, no race.
const dualLayout = tgpu
    .bindGroupLayout({ eids: { storage: d.arrayOf(d.u32), access: "readonly" } })
    .$idx(0);

// one thread per pair SLOT in the live bodies' per-eid blocks (lane → d → owner eid → slot = eid·K + k)
const dualKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= dualLayout.$.eids[0]) return; // past the live body count
        solverRoRw.dualSlot(
            dualLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY),
        );
    })
    .$name("dualMain");

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
// The cap (`params.maxColors`) is separate from the 32-wide mask (avbd.md — two
// numbers): a body that finds no free color within the cap folds to `bid % maxColors`, degrading that
// pair to Jacobi for the step (a soft-contact conflict the iterative primal tolerates — the invariant
// is a low *measured* conflict rate, not zero). The neighbor scan reads the body's CSR contact list
// (csrList[csr[bid] .. +csr[eidCap+bid]]), not every contact — the same O(valence) read the primal
// does. Deterministic integer logic; the CPU reference is tests/avbd/coloring.ts (the executable
// spec) and the GPU reproduces it in the gym `pile` scenario (coloring-conflict counter).
const coloringLayout = tgpu
    .bindGroupLayout({
        // [0, eidCap) offsets, [eidCap, 2·eidCap) counts
        csr: { storage: d.arrayOf(d.u32), access: "readonly" },
        csrList: { storage: d.arrayOf(d.u32), access: "readonly" },
        colors: { storage: d.arrayOf(d.u32), access: "mutable" },
        colorScratch: { storage: d.arrayOf(d.u32), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        // colorCount[0] = the used-color count this step (max dynamic color + 1), the readback-bounded
        // color loop's input (Phase 4.9 Lever 1). Cleared each step before this pass; each dynamic body
        // atomicMaxes its chosen color + 1. One slot, low contention — the counters class.
        colorCount: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        // authored-constraint adjacency (springs Phase 6.1 + joints Phase 6.2) — the edge enters the
        // coloring so a constraint-connected dynamic pair prefers different colors (avoidance, kind-
        // agnostic). A soft spring tolerates a same-color clean-Jacobi pair; a hard joint must NOT be
        // same-color, so the joint edge gets a second repair pass on top of this avoidance.
        constraintCsr: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintList: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    })
    .$idx(0);

const coloringKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= coloringLayout.$.eids[0]) return;
        const bid = coloringLayout.$.eids[1 + slot];
        if (roRo.bMass(bid) <= 0) {
            coloringLayout.$.colors[bid] = d.u32(0xffffffff); // static — uncolored, not dispatched
            return;
        }

        const colorsN = std.max(d.u32(1), std.min(roRo.layout.$.params.maxColors, d.u32(32)));
        const lo = coloringLayout.$.csr[bid];
        const hi = lo + coloringLayout.$.csr[roRo.layout.$.params.eidCap + bid];
        let usedMask = d.u32(0);
        for (let k = lo; k < hi; k++) {
            const m = roRo.cc(coloringLayout.$.csrList[k], C_META);
            const a = bitcastF32toU32(m.y);
            const b = bitcastF32toU32(m.z);
            let other = a; // every CSR contact touches bid; pick the neighbor
            if (a === bid) other = b;
            if (other <= bid) continue; // higher-id symmetry break — no atomics
            if (roRo.solverStatic(other)) continue; // static / kinematic neighbor: no scheduling constraint
            const pc = coloringLayout.$.colorScratch[other];
            if (pc < 32) usedMask = usedMask | (d.u32(1) << pc);
        }
        // authored-constraint neighbors (springs + joints): same higher-id-symmetry avoidance as contacts,
        // kind-agnostic (the partner eid is constraintList[e+2].x for both). A constraint-less body has count 0.
        const slo = coloringLayout.$.constraintCsr[bid];
        const shi = slo + coloringLayout.$.constraintCsr[roRo.layout.$.params.eidCap + bid];
        for (let e = slo; e < shi; e++) {
            const other = bitcastF32toU32(
                coloringLayout.$.constraintList[e * d.u32(CONSTRAINT_VEC4) + 2].x,
            );
            if (other <= bid) continue;
            if (roRo.solverStatic(other)) continue;
            const pc = coloringLayout.$.colorScratch[other];
            if (pc < 32) usedMask = usedMask | (d.u32(1) << pc);
        }

        let chosen = coloringLayout.$.colorScratch[bid]; // incremental: keep the prior color when still free
        let needsNew = chosen >= colorsN;
        if (!needsNew) needsNew = (usedMask & (d.u32(1) << chosen)) !== 0;
        if (needsNew) {
            let found = false;
            for (let c = d.u32(0); c < colorsN; c++) {
                if ((usedMask & (d.u32(1) << c)) === 0) {
                    chosen = c;
                    found = true;
                    break;
                }
            }
            if (!found) chosen = bid % colorsN; // fold past the cap — a tolerated same-color conflict
        }
        coloringLayout.$.colors[bid] = chosen;
        // publish the used-color count (max dynamic color + 1) for the readback-bounded color loop — the
        // primal next frame dispatches min(maxColors, usedColors + COLOR_MARGIN) color-passes (Lever 1).
        std.atomicMax(coloringLayout.$.colorCount[0], chosen + 1);
    })
    .$name("coloringMain");

// ── repair: the joint hard-conflict coloring repair (Phase 6.2, webphysics repairHardBodyColors) ──
// The greedy avoids ALL constraint neighbors but folds past the cap (a tolerated same-color Jacobi). A SOFT
// spring survives that; a HARD (dynamic-dynamic) joint pair degrading to same-color Jacobi destabilizes, so
// after the greedy this runs JOINT_REPAIR_ROUNDS rounds: each round snapshots colors → colorScratch, then
// each lower-eid endpoint of a same-color joint pair recolors to a free color (excluding all its constraint
// neighbors). Reading the stable snapshot keeps it race-free + deterministic, like the greedy. GPU==oracle
// can't validate this (the oracle runs the GPU's coloring), so the gym gates the observable invariant: a
// dynamic joint pair ends colors[a] != colors[b].
const repairLayout = tgpu
    .bindGroupLayout({
        colors: { storage: d.arrayOf(d.u32), access: "mutable" },
        colorScratch: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintCsr: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintList: { storage: d.arrayOf(d.vec4f), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        colorCount: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    })
    .$idx(0);

const repairKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= repairLayout.$.eids[0]) return;
        const bid = repairLayout.$.eids[1 + slot];
        if (roRo.bMass(bid) <= 0) return; // static — uncolored, never a hard mover

        const colorsN = std.max(d.u32(1), std.min(roRo.layout.$.params.maxColors, d.u32(32)));
        const myColor = repairLayout.$.colorScratch[bid];
        const slo = repairLayout.$.constraintCsr[bid];
        const shi = slo + repairLayout.$.constraintCsr[roRo.layout.$.params.eidCap + bid];
        let usedMask = d.u32(0);
        let hardConflict = false;
        for (let e = slo; e < shi; e++) {
            const base = e * d.u32(CONSTRAINT_VEC4);
            const other = bitcastF32toU32(repairLayout.$.constraintList[base + 2].x);
            const kind = bitcastF32toU32(repairLayout.$.constraintList[base + 2].y);
            if (other >= roRo.layout.$.params.eidCap || roRo.bMass(other) <= 0) continue; // static: no constraint
            const oc = repairLayout.$.colorScratch[other];
            if (oc < 32) usedMask = usedMask | (d.u32(1) << oc);
            // the lower-eid endpoint of a same-color joint pair is the one that moves (higher-id stays fixed)
            if (kind === KIND_JOINT && other > bid && oc === myColor) hardConflict = true;
        }
        if (!hardConflict) return; // already conflict-free — its color stays counted
        let chosen = myColor;
        let found = false;
        for (let c = d.u32(0); c < colorsN; c++) {
            if ((usedMask & (d.u32(1) << c)) === 0) {
                chosen = c;
                found = true;
                break;
            }
        }
        if (!found) chosen = bid % colorsN; // fold (a tolerated soft conflict — no free color left)
        repairLayout.$.colors[bid] = chosen;
        std.atomicMax(repairLayout.$.colorCount[0], chosen + 1); // keep the readback-bounded loop's count ≥ this
    })
    .$name("repairMain");

// ── joint init: warmstart the per-joint dual state + capture C(x⁻) (Phase 6.2, joint.ts initJoint) ──
// One thread per joint, before the main loop (after the contact collide, before inertial init so it reads
// the step-start pose x⁻). Runs the recycle-version guard + the one-time construction guard, computes the
// torque arm GPU-side, captures C₀, then decays λ/penalty (Eq. 19) and clamps to material stiffness.
// ── joint init: warmstart the per-joint dual state + capture C(x⁻) (Phase 6.2, joint.ts initJoint) ──
// One thread per joint, before the main loop (after the contact collide, before inertial init so it reads
// the step-start pose x⁻). Runs the recycle-version guard + the one-time construction guard, computes the
// torque arm GPU-side, captures C₀, then decays λ/penalty (Eq. 19) and clamps to material stiffness.
const jointInitLayout = tgpu
    .bindGroupLayout({
        jointVersions: { storage: d.arrayOf(d.u32), access: "readonly" },
        counters: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        // per-eid seed flag (the pack sets it after seeding a body's slot from its slabs): a joint must
        // not read a body's pose until it's seeded. jointInit is per-JOINT, NOT gated on the live count,
        // so on the very first fixed step (which can run before the first pack at 60 Hz) the bodies slots
        // are still zero-init — without this gate the fresh anchor guard would see half = 0 (reach a tiny
        // margin) + offset anchors and WRONGLY reject every valid joint.
        seeded: { storage: d.arrayOf(d.u32), access: "readonly" },
    })
    .$idx(0);

const jointInitKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const jid = input.gid.x;
        if (jid >= roRo.layout.$.params.jointCount) return;
        const r0 = jointRw.jrec(jid, 0);
        const a = bitcastF32toU32(r0.x);
        const b = bitcastF32toU32(r0.y);
        const recBase = jid * d.u32(JOINT_REC_VEC4);
        // a == WORLD_ANCHOR is the world (no body): rA is a world point, orientation identity,
        // mass/size/radius 0, always seeded, no version. b is always a real body.
        const aWorld = a === WORLD_ANCHOR;

        // recycle-version guard: a despawned-then-recycled endpoint must not realias the joint to a new
        // body. A version mismatch deactivates it (its primal stamp + dual then no-op).
        const aBad = !aWorld && jointInitLayout.$.jointVersions[a] !== bitcastF32toU32(r0.z);
        if (aBad || jointInitLayout.$.jointVersions[b] !== bitcastF32toU32(r0.w)) {
            jointRw.layout.$.jointRecords[recBase + 3].y = std.bitcastU32toF32(0);
            return;
        }
        if ((!aWorld && jointInitLayout.$.seeded[a] === 0) || jointInitLayout.$.seeded[b] === 0)
            return; // not seeded yet — retry after the pack

        // both-static guard: a joint NO dynamic body can resolve is never satisfiable by the primal, so
        // its dual ramps penalty + lambda unbounded (avbd.md "Joint guards"). Checked EVERY frame — a
        // persistent gauge, not a one-frame blip a lagged Mirror readback can miss.
        const aStatic = aWorld || roRo.bMass(a) <= 0;
        if (aStatic && roRo.bMass(b) <= 0) {
            jointRw.layout.$.jointRecords[recBase + 3].y = std.bitcastU32toF32(0);
            std.atomicAdd(jointInitLayout.$.counters[1], 1); // observable: a both-endpoints-static joint, deactivated
            return;
        }
        const act = jointRw.jActive(jid);
        if (act === 0) return; // already deactivated — stays off until re-authored

        const rA = jointRw.jrec(jid, 1).xyz;
        const stiffLin = jointRw.jrec(jid, 1).w;
        const rB = jointRw.jrec(jid, 2).xyz;
        const stiffAng = jointRw.jrec(jid, 2).w;
        // torqueArm = ‖sizeA + sizeB‖² (full size = 2·halfExtents) — GPU-computed so the CPU needn't carry
        // sizes. The world anchor contributes size 0 + identity orientation + its world point rA.
        const sizeA = std.select(std.mul(roRo.bHalf(a), 2), d.vec3f(0, 0, 0), aWorld);
        const sizeB = std.mul(roRo.bHalf(b), 2);
        const sizeSum = std.add(sizeA, sizeB);
        const torqueArm = std.dot(sizeSum, sizeSum);
        const qA = std.select(roRo.bQuat(a), d.vec4f(0, 0, 0, 1), aWorld);
        const pA = std.select(std.add(roRo.bPos(a), roRo.qRotateW(qA, rA)), rA, aWorld);
        const pB = std.add(roRo.bPos(b), roRo.qRotateW(roRo.bQuat(b), rB));

        if (act === 2) {
            // anchor-coincidence guard: the anchors must START coincident; a gross mismatch injects
            // energy through BDF1 recovery. Reach = length(halfExtents) + the rounded radius. Pose-
            // dependent, so only on the fresh frame.
            const reachA = std.select(
                std.length(roRo.bHalf(a)) + roRo.bRadius(a),
                d.f32(0),
                aWorld,
            );
            const reach = reachA + std.length(roRo.bHalf(b)) + roRo.bRadius(b) + COLLISION_MARGIN;
            if (std.length(std.sub(pA, pB)) > reach) {
                jointRw.layout.$.jointRecords[recBase + 3].y = std.bitcastU32toF32(0);
                std.atomicAdd(jointInitLayout.$.counters[2], 1); // observable: joints rejected by the anchor guard
                return;
            }
        }
        jointRw.layout.$.jointRecords[recBase + 3] = d.vec4f(
            torqueArm,
            std.bitcastU32toF32(1),
            0,
            0,
        );

        // C(x⁻) at the step-start pose (this pass runs before inertial init predicts the pose)
        jointRw.layout.$.jointRecords[recBase + 8] = d.vec4f(std.sub(pA, pB), 0);
        jointRw.layout.$.jointRecords[recBase + 9] = d.vec4f(
            std.mul(roRo.qSubW(qA, roRo.bQuat(b)), torqueArm),
            0,
        );

        // warmstart λ + penalty (Eq. 19): λ ← α·γ·λ, k ← clamp(γ·k, MIN, MAX) then clamp to material stiffness
        const ag = roRo.layout.$.params.alpha * roRo.layout.$.params.gamma;
        jointRw.layout.$.jointRecords[recBase + 6] = d.vec4f(std.mul(jointRw.jLamLin(jid), ag), 0);
        jointRw.layout.$.jointRecords[recBase + 7] = d.vec4f(std.mul(jointRw.jLamAng(jid), ag), 0);
        const penLin = std.min(
            std.clamp(
                std.mul(jointRw.jPenLin(jid), roRo.layout.$.params.gamma),
                d.vec3f(PENALTY_MIN),
                d.vec3f(PENALTY_MAX),
            ),
            d.vec3f(stiffLin),
        );
        const penAng = std.min(
            std.clamp(
                std.mul(jointRw.jPenAng(jid), roRo.layout.$.params.gamma),
                d.vec3f(PENALTY_MIN),
                d.vec3f(PENALTY_MAX),
            ),
            d.vec3f(stiffAng),
        );
        jointRw.layout.$.jointRecords[recBase + 4] = d.vec4f(penLin, 0);
        jointRw.layout.$.jointRecords[recBase + 5] = d.vec4f(penAng, 0);

        // motor warmstart (Eq. 19): decay λ + penalty. The static axis/speed/maxTorque (col 10, 11.x) are
        // not rewritten by this pass, so they persist from setJoints across frames.
        if (jointRw.jMotorMax(jid) > 0) {
            const mp = std.clamp(
                jointRw.jMotorPen(jid) * roRo.layout.$.params.gamma,
                PENALTY_MIN,
                PENALTY_MAX,
            );
            jointRw.layout.$.jointRecords[recBase + 11] = d.vec4f(
                jointRw.jMotorSpeed(jid),
                jointRw.jMotorLam(jid) * ag,
                mp,
                0,
            );
        }
    })
    .$name("jointInitMain");

// ── joint dual: advance λ + the penalty ramp per joint each iteration (Phase 6.2, joint.ts updateJointDual) ──
// One thread per joint, after each iteration's primal (like the contact dual). The rigid (∞-stiffness)
// rows store λ ← K·C + λ; both row triples ramp the penalty by β|C|, clamped to PENALTY_MAX.
// joint-dual has no own I/O (its only inputs are the shared roRo group + jointRecords) — typegpu's
// resolve only tracks a bind group layout that the emitted WGSL actually references, so an empty group-0
// layout with no touch is invisible to it and the pipeline compiles with a missing bind group at runtime
// (real-device only — no resolve-time signal). `counters` (fixed-size, already bound this way by the
// collide pass's `touchHullData`) gives group 0 a real reference; the forcing touch adds 0 to the same
// reserved never-read slot, so it's a no-op wherever it lands.
const jointDualLayout = tgpu
    .bindGroupLayout({ counters: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" } })
    .$idx(0);

const jointDualKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const jid = input.gid.x;
        if (jid >= roRo.layout.$.params.jointCount) return;
        std.atomicAdd(jointDualLayout.$.counters[15], 0);
        solverJointRw.jointDualOne(jid);
    })
    .$name("jointDualMain");

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
// solve-lds's own I/O: `colors`/`eids` (the looped path's own bindings) + `denseMap`, the eid → LDS-slot
// map REBOUND over the `solveOut` scratch (unused by this path — 2·eidCap vec4s ≥ the eidCap u32s the map
// needs; the kernel sits exactly at the 10-storage floor, so the map can't be a new buffer without
// evicting one). `csr`/`csrList`/`constraintCsr`/`constraintList` need no binding here — `solvePose`
// (below) already references them via `primalOwnLayout`, the same group 0 the typed primal reuses.
export const LDS_IO_GROUP = 3;

const ldsLayout = tgpu
    .bindGroupLayout({
        colors: { storage: d.arrayOf(d.u32), access: "readonly" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        denseMap: { storage: d.arrayOf(d.u32), access: "mutable" },
    })
    .$idx(LDS_IO_GROUP);

const SOLVE_WG = 256;

const lpx = tgpu.workgroupVar(d.arrayOf(d.f32, LDS_CAP));
const lpy = tgpu.workgroupVar(d.arrayOf(d.f32, LDS_CAP));
const lpz = tgpu.workgroupVar(d.arrayOf(d.f32, LDS_CAP));
const lq = tgpu.workgroupVar(d.arrayOf(d.vec4f, LDS_CAP));
const wgCount = tgpu.workgroupVar(d.u32);
const wgColorMax = tgpu.workgroupVar(d.atomic(d.u32));
const wgColors = tgpu.workgroupVar(d.u32);

// the LDS-backed pose readers the shared solve/dual math goes through (one
// authored kernel re-emits per reader set — storage this stage's other kernels, workgroup memory here).
// Every live eid's denseMap entry is written at kernel start, so a slot ≥ LDS_CAP means the body
// overflowed residency → its storage pose (constant this step: an overflow body is never solved) is the
// consistent fallback.
const ldsBPos = tgpu
    .fn(
        [d.u32],
        d.vec3f,
    )((i) => {
        "use gpu";
        const s = ldsLayout.$.denseMap[i];
        if (s < d.u32(LDS_CAP)) return d.vec3f(lpx.$[s], lpy.$[s], lpz.$[s]);
        return rwRw.bCol(B_POS, i).xyz;
    })
    .$name("bPos");

const ldsBQuat = tgpu
    .fn(
        [d.u32],
        d.vec4f,
    )((i) => {
        "use gpu";
        const s = ldsLayout.$.denseMap[i];
        if (s < d.u32(LDS_CAP)) return d.vec4f(lq.$[s]);
        return rwRw.bCol(B_QUAT, i);
    })
    .$name("bQuat");

/** the LDS-resident solver instantiation — `contactMath`/`contribMath`/`dualMath`/`jointDualMath` re-run
 *  over a reader set whose `bPos`/`bQuat` shadow `rwRw`'s storage readers with the workgroup-memory ones
 *  above; everything else (bMass, bHalf, solverStatic, cc, the quat math, `layout`) stays `rwRw`'s. @internal */
const ldsA = { ...rwRw, bPos: ldsBPos, bQuat: ldsBQuat };
const ldsContactForce = contactMath(ldsA).contactForce;
const solverLds = {
    contactForce: ldsContactForce,
    ...contribMath(ldsA, jointRw, ldsContactForce),
    ...dualMath(ldsA, ldsContactForce),
    ...jointDualMath(ldsA, jointRw),
};

const solveLdsKernel = tgpu
    .computeFn({ workgroupSize: [SOLVE_WG], in: { lid: d.builtin.localInvocationId } })((input) => {
        "use gpu";
        const lane = input.lid.x;
        if (lane === 0) wgCount.$ = ldsLayout.$.eids[0];
        const count = uniformLoad(wgCount.$);
        const n = std.min(count, d.u32(LDS_CAP));

        // load: eid → dense map + the resident poses + the used-color count (max dynamic color + 1 — the
        // looped path's readback-bounded count, computed GPU-fresh; colors past it hold no bodies, so the
        // loop bound is a dispatch-count choice, never a math change)
        for (let dOwn = lane; dOwn < count; dOwn = dOwn + d.u32(SOLVE_WG)) {
            const eid = ldsLayout.$.eids[1 + dOwn];
            ldsLayout.$.denseMap[eid] = dOwn;
            if (dOwn < d.u32(LDS_CAP)) {
                const p = rwRw.bCol(B_POS, eid).xyz;
                lpx.$[dOwn] = p.x;
                lpy.$[dOwn] = p.y;
                lpz.$[dOwn] = p.z;
                lq.$[dOwn] = rwRw.bCol(B_QUAT, eid);
            }
            if (rwRw.bMass(eid) > 0)
                std.atomicMax(wgColorMax.$, ldsLayout.$.colors[eid] + d.u32(1));
        }
        std.storageBarrier(); // denseMap visible before any bPos/bQuat routes through it
        std.workgroupBarrier(); // resident poses + wgColorMax
        if (lane === 0)
            wgColors.$ = std.min(std.atomicLoad(wgColorMax.$), rwRw.layout.$.params.maxColors);
        const colorsToRun = uniformLoad(wgColors.$);

        for (let it = d.u32(0); it < rwRw.layout.$.params.iterations; it++) {
            for (let c = d.u32(0); c < colorsToRun; c++) {
                // primal: each lane solves its ≤2 bodies of this color from the committed LDS poses,
                // staging the results in registers — the double-buffer (a folded same-color pair reads
                // the color-start pose on both sides, the clean Jacobi the looped solveOut/commit pair gives)
                let np0 = NewPose({ pos: d.vec3f(0, 0, 0), quat: d.vec4f(0, 0, 0, 0) });
                let w0 = false;
                const d0 = lane;
                if (d0 < n) {
                    const bid = ldsLayout.$.eids[1 + d0];
                    if (!rwRw.solverStatic(bid) && ldsLayout.$.colors[bid] === c) {
                        np0 = NewPose(solverLds.solvePose(bid));
                        w0 = true;
                    }
                }
                let np1 = NewPose({ pos: d.vec3f(0, 0, 0), quat: d.vec4f(0, 0, 0, 0) });
                let w1 = false;
                const d1 = lane + d.u32(SOLVE_WG);
                if (d1 < n) {
                    const bid = ldsLayout.$.eids[1 + d1];
                    if (!rwRw.solverStatic(bid) && ldsLayout.$.colors[bid] === c) {
                        np1 = NewPose(solverLds.solvePose(bid));
                        w1 = true;
                    }
                }
                std.workgroupBarrier();
                // commit: the color's staged poses land in LDS together
                if (w0) {
                    lpx.$[d0] = np0.pos.x;
                    lpy.$[d0] = np0.pos.y;
                    lpz.$[d0] = np0.pos.z;
                    lq.$[d0] = d.vec4f(np0.quat);
                }
                if (w1) {
                    lpx.$[d1] = np1.pos.x;
                    lpy.$[d1] = np1.pos.y;
                    lpz.$[d1] = np1.pos.z;
                    lq.$[d1] = d.vec4f(np1.quat);
                }
                std.workgroupBarrier();
            }
            // dual + joint dual: the standalone passes' lane mappings, strided over one workgroup. They
            // read the post-color LDS poses and write λ/penalty into the persistent storage records,
            // which the next iteration's primal reads — once-per-iteration storage traffic, not per color.
            const slots = count * d.u32(PAIRS_PER_BODY);
            for (let s = lane; s < slots; s = s + d.u32(SOLVE_WG)) {
                solverLds.dualSlot(
                    ldsLayout.$.eids[1 + idiv(s, PAIRS_PER_BODY)] * d.u32(PAIRS_PER_BODY) +
                        (s % d.u32(PAIRS_PER_BODY)),
                );
            }
            for (let j = lane; j < rwRw.layout.$.params.jointCount; j = j + d.u32(SOLVE_WG)) {
                solverLds.jointDualOne(j);
            }
            std.storageBarrier(); // record λ/penalty writes → the next iteration's contactForce reads
        }

        // write back the solved poses. Statics (incl. kinematic characters) are skipped — the looped path
        // never writes them either, and a character's char-pass pose must not be re-stamped with w = 0.
        for (let dOwn = lane; dOwn < n; dOwn = dOwn + d.u32(SOLVE_WG)) {
            const eid = ldsLayout.$.eids[1 + dOwn];
            if (rwRw.solverStatic(eid)) continue;
            rwRw.layout.$.bodies[B_POS * rwRw.layout.$.params.eidCap + eid] = d.vec4f(
                lpx.$[dOwn],
                lpy.$[dOwn],
                lpz.$[dOwn],
                0,
            );
            rwRw.layout.$.bodies[B_QUAT * rwRw.layout.$.params.eidCap + eid] = d.vec4f(lq.$[dOwn]);
        }
    })
    .$name("solveLdsMain");

// ── compose: scatter the interpolated body pose into the eid-indexed transform firehose ──
// The bodied-entity half of the Body/Transform contract (roadmap): a `Body` is a `Part` whose world
// matrix physics owns. `Body.excludes [Transform]`, so the Transform compose writes a stale slot for a
// body eid; this pass runs after it and overwrites `transforms[eids[d]]` with the live pose. Scale is
// 2·halfExtents — the cube mesh is unit (-0.5..0.5), so the render box matches the collision box (the
// body pose itself is scale-free; this is render-only). Writes the decomposed `Xform` (the same struct
// the Transform compose gathers); readers reconstruct the world transform via xformWgsl().
//
// Render interpolation (Phase 5): the solver steps at the fixed
// rate but compose runs every render frame, so at >60Hz it would repeat a fixed-step pose then jump
// (stutter). Blend prev→curr by `interp.alpha` (= time.fixedAlpha, the fraction past the last fixed tick).
// The prev pose needs no extra column or snapshot pass: the inertial pass already saves x⁻ (the pre-warmstart
// pose = last frame's settled pose) into B_INITL/B_INITQ before warmstart mutates B_POS, so prev = bInit*,
// curr = bPos/bQuat. lerp position, nlerp quat on the shortest arc. For a static or freshly-seeded body
// B_INITL == B_POS, so it's a no-op; at alpha = 1 this is exactly the bare current pose.
/** interpolation alpha uniform — `time.fixedAlpha`, the fraction past the last fixed tick */
const Interp = d.struct({ alpha: d.f32 }).$name("Interp");

// nlerp toward the shortest arc: flip prev into curr's hemisphere, lerp, renormalize
const nlerpShortest = tgpu
    .fn(
        [d.vec4f, d.vec4f, d.f32],
        d.vec4f,
    )((prev, curr, t) => {
        "use gpu";
        const flip = std.select(d.f32(1), d.f32(-1), std.dot(prev, curr) < 0);
        const q = std.mix(std.mul(prev, flip), curr, t);
        const len = std.length(q);
        return std.select(d.vec4f(0, 0, 0, 1), std.div(q, len), len > 1e-12);
    })
    .$name("nlerpShortest");

const composeLayout = tgpu
    .bindGroupLayout({
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        transforms: { storage: d.arrayOf(Xform), access: "mutable" },
        interp: { uniform: Interp },
    })
    .$idx(0);

const composeKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= composeLayout.$.eids[0]) return;
        const i = composeLayout.$.eids[1 + slot];
        const a = composeLayout.$.interp.alpha;
        const p = std.mix(roRo.bInitL(i), roRo.bPos(i), a); // prev = x⁻ (pre-warmstart), curr = solved pose
        const q = nlerpShortest(roRo.bInitQ(i), roRo.bQuat(i), a);
        // render scale maps the unit mesh to the body's shape (Phase 6.3): the cube/sphere meshes are unit
        // half-extent 0.5 → scale 2·extent; the capsule mesh is y∈[-1,1] (half-extents 0.5,1,0.5), so its
        // bounding box (2r, hc+r, 2r) maps with y-scale hc+r (the caps distort under a non-proportional
        // ratio — a fixed-mesh limitation, render-only; the collider is exact).
        const shape = roRo.bShape(i);
        const radius = roRo.bRadius(i);
        let s = std.mul(roRo.bHalf(i), 2);
        if (shape === 1) {
            s = d.vec3f(2 * radius, 2 * radius, 2 * radius);
        } else if (shape === 2) {
            s = d.vec3f(2 * radius, roRo.bHalf(i).y + radius, 2 * radius);
        }
        // element-schema copy — p/q/s are reference-yielding call results (mix/nlerpShortest/select)
        composeLayout.$.transforms[i] = Xform({
            pos: d.vec3f(p),
            quat: d.vec4f(q),
            scale: d.vec3f(s),
        });
    })
    .$name("composeMain");

// ── velocity: BDF1 recovery (solver.cpp step 5). prevVel updated for the next adaptive warmstart ──
const velocityLayout = tgpu
    .bindGroupLayout({ eids: { storage: d.arrayOf(d.u32), access: "readonly" } })
    .$idx(0);

const velocityKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const slot = input.gid.x;
        if (slot >= velocityLayout.$.eids[0]) return;
        const i = velocityLayout.$.eids[1 + slot];
        const cap = rwRw.layout.$.params.eidCap;
        rwRw.layout.$.bodies[B_PREVV * cap + i] = d.vec4f(rwRw.bVelL(i), 0); // prevVel = vel
        if (rwRw.solverStatic(i)) return; // static / kinematic — keep the frozen velocity
        const velL = std.div(std.sub(rwRw.bPos(i), rwRw.bInitL(i)), rwRw.layout.$.params.dt);
        const velA = std.div(rwRw.qSubW(rwRw.bQuat(i), rwRw.bInitQ(i)), rwRw.layout.$.params.dt);
        rwRw.layout.$.bodies[B_VELL * cap + i] = d.vec4f(velL, 0);
        rwRw.layout.$.bodies[B_VELA * cap + i] = d.vec4f(velA, 0);
    })
    .$name("velocityMain");

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

const csrCountLayout = tgpu
    .bindGroupLayout({
        csr: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" }, // counts in [eidCap, 2·eidCap)
        eids: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = live count, [1+d] = the d-th live eid
    })
    .$idx(0);

const csrCountKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= csrCountLayout.$.eids[0]) return; // past the live body count
        const recBase =
            (csrCountLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY)) *
            d.u32(CONTACTS_PER_PAIR);
        for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
            const m = roRo.cc(recBase + ls, C_META);
            if (bitcastF32toU32(m.x) !== CONSTRAINT_CONTACT) continue; // inactive record
            std.atomicAdd(
                csrCountLayout.$.csr[roRo.layout.$.params.eidCap + bitcastF32toU32(m.y)],
                1,
            ); // body a
            std.atomicAdd(
                csrCountLayout.$.csr[roRo.layout.$.params.eidCap + bitcastF32toU32(m.z)],
                1,
            ); // body b
        }
    })
    .$name("csrCountMain");

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

const csrScatterLayout = tgpu
    .bindGroupLayout({
        // [0,eidCap) offsets (read via atomicLoad — same binding, atomic type), [eidCap,…) append cursor
        csr: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        csrList: { storage: d.arrayOf(d.u32), access: "mutable" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = live count, [1+d] = the d-th live eid
    })
    .$idx(0);

const csrScatterKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const lane = input.gid.x;
        const dOwn = idiv(lane, PAIRS_PER_BODY);
        if (dOwn >= csrScatterLayout.$.eids[0]) return; // past the live body count
        const recBase =
            (csrScatterLayout.$.eids[1 + dOwn] * d.u32(PAIRS_PER_BODY) + (lane % PAIRS_PER_BODY)) *
            d.u32(CONTACTS_PER_PAIR);
        for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
            const ci = recBase + ls;
            const m = roRo.cc(ci, C_META);
            if (bitcastF32toU32(m.x) !== CONSTRAINT_CONTACT) continue; // inactive record
            const a = bitcastF32toU32(m.y);
            const b = bitcastF32toU32(m.z);
            csrScatterLayout.$.csrList[
                std.atomicLoad(csrScatterLayout.$.csr[a]) +
                    std.atomicAdd(csrScatterLayout.$.csr[roRo.layout.$.params.eidCap + a], 1)
            ] = ci;
            csrScatterLayout.$.csrList[
                std.atomicLoad(csrScatterLayout.$.csr[b]) +
                    std.atomicAdd(csrScatterLayout.$.csr[roRo.layout.$.params.eidCap + b], 1)
            ] = ci;
        }
    })
    .$name("csrScatterMain");

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
const csrColorSmallLayout = tgpu
    .bindGroupLayout({
        csr: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        csrList: { storage: d.arrayOf(d.u32), access: "mutable" },
        colors: { storage: d.arrayOf(d.u32), access: "mutable" },
        colorScratch: { storage: d.arrayOf(d.u32), access: "mutable" },
        eids: { storage: d.arrayOf(d.u32), access: "readonly" },
        colorCount: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
        constraintCsr: { storage: d.arrayOf(d.u32), access: "readonly" },
        constraintList: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    })
    .$idx(0);

const csrColorWgN = tgpu.workgroupVar(d.u32);
const csrColorSum = tgpu.workgroupVar(d.arrayOf(d.u32, PACK_WG));
const csrColorWgMax = tgpu.workgroupVar(d.atomic(d.u32)); // max dynamic color + 1 (zero-inits per dispatch)

const csrColorSmallKernel = tgpu
    .computeFn({ workgroupSize: [PACK_WG], in: { lid: d.builtin.localInvocationId } })((input) => {
        "use gpu";
        // every lane reaches every barrier (no early returns — idle lanes skip strided loops), and the live
        // count flows through uniformLoad for Tint's uniformity analysis (the small-broadphase pattern).
        const t = input.lid.x;
        if (t === 0) csrColorWgN.$ = csrColorSmallLayout.$.eids[0];
        const n = uniformLoad(csrColorWgN.$);

        // clear: live eids' counts (the only counts the scan reads — see the header)
        for (let dOwn = t; dOwn < n; dOwn = dOwn + PACK_WG) {
            std.atomicStore(
                csrColorSmallLayout.$.csr[
                    roRo.layout.$.params.eidCap + csrColorSmallLayout.$.eids[1 + dOwn]
                ],
                0,
            );
        }
        std.storageBarrier();

        // count
        const slots = n * d.u32(PAIRS_PER_BODY);
        for (let s = t; s < slots; s = s + PACK_WG) {
            const recBase =
                (csrColorSmallLayout.$.eids[1 + idiv(s, PAIRS_PER_BODY)] * d.u32(PAIRS_PER_BODY) +
                    (s % PAIRS_PER_BODY)) *
                d.u32(CONTACTS_PER_PAIR);
            for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
                const m = roRo.cc(recBase + ls, C_META);
                if (bitcastF32toU32(m.x) !== CONSTRAINT_CONTACT) continue; // inactive record
                std.atomicAdd(
                    csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + bitcastF32toU32(m.y)],
                    1,
                ); // body a
                std.atomicAdd(
                    csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + bitcastF32toU32(m.z)],
                    1,
                ); // body b
            }
        }
        std.storageBarrier();

        // scan: exclusive prefix over the live counts → offsets; counts reset to 0 (the scatter cursor)
        const chunkSz = idiv(n + d.u32(PACK_WG) - 1, d.u32(PACK_WG));
        const lo = t * chunkSz;
        const hi = std.min(lo + chunkSz, n);
        let sum = d.u32(0);
        for (let dOwn = lo; dOwn < hi; dOwn++) {
            sum =
                sum +
                std.atomicLoad(
                    csrColorSmallLayout.$.csr[
                        roRo.layout.$.params.eidCap + csrColorSmallLayout.$.eids[1 + dOwn]
                    ],
                );
        }
        csrColorSum.$[t] = sum;
        std.workgroupBarrier();
        if (t === 0) {
            let acc0 = d.u32(0);
            for (let i = d.u32(0); i < PACK_WG; i++) {
                const c = csrColorSum.$[i];
                csrColorSum.$[i] = acc0;
                acc0 = acc0 + c;
            }
        }
        std.workgroupBarrier();
        let acc = csrColorSum.$[t];
        for (let dOwn = lo; dOwn < hi; dOwn++) {
            const e = csrColorSmallLayout.$.eids[1 + dOwn];
            std.atomicStore(csrColorSmallLayout.$.csr[e], acc);
            acc = acc + std.atomicLoad(csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + e]);
            std.atomicStore(csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + e], 0);
        }
        std.storageBarrier();

        // scatter
        for (let s = t; s < slots; s = s + PACK_WG) {
            const recBase =
                (csrColorSmallLayout.$.eids[1 + idiv(s, PAIRS_PER_BODY)] * d.u32(PAIRS_PER_BODY) +
                    (s % PAIRS_PER_BODY)) *
                d.u32(CONTACTS_PER_PAIR);
            for (let ls = d.u32(0); ls < CONTACTS_PER_PAIR; ls++) {
                const ci = recBase + ls;
                const m = roRo.cc(ci, C_META);
                if (bitcastF32toU32(m.x) !== CONSTRAINT_CONTACT) continue; // inactive record
                const a = bitcastF32toU32(m.y);
                const b = bitcastF32toU32(m.z);
                csrColorSmallLayout.$.csrList[
                    std.atomicLoad(csrColorSmallLayout.$.csr[a]) +
                        std.atomicAdd(csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + a], 1)
                ] = ci;
                csrColorSmallLayout.$.csrList[
                    std.atomicLoad(csrColorSmallLayout.$.csr[b]) +
                        std.atomicAdd(csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + b], 1)
                ] = ci;
            }
        }
        std.storageBarrier();

        // greedy: the incremental coloring (coloringKernel's logic), priors read from colors (untouched
        // this step), chosen staged in colorScratch so every lane's prior reads complete before any commit
        for (let dOwn = t; dOwn < n; dOwn = dOwn + PACK_WG) {
            const bid = csrColorSmallLayout.$.eids[1 + dOwn];
            if (roRo.bMass(bid) <= 0) {
                csrColorSmallLayout.$.colorScratch[bid] = d.u32(0xffffffff); // static — uncolored
                continue;
            }
            const colorsN = std.max(d.u32(1), std.min(roRo.layout.$.params.maxColors, d.u32(32)));
            const clo = std.atomicLoad(csrColorSmallLayout.$.csr[bid]);
            const chi =
                clo + std.atomicLoad(csrColorSmallLayout.$.csr[roRo.layout.$.params.eidCap + bid]);
            let usedMask = d.u32(0);
            for (let k = clo; k < chi; k++) {
                const m = roRo.cc(csrColorSmallLayout.$.csrList[k], C_META);
                const a = bitcastF32toU32(m.y);
                const b = bitcastF32toU32(m.z);
                let other = a; // every CSR contact touches bid; pick the neighbor
                if (a === bid) other = b;
                if (other <= bid) continue; // higher-id symmetry break — no atomics
                if (roRo.bMass(other) <= 0) continue; // static neighbor: no scheduling constraint
                const pc = csrColorSmallLayout.$.colors[other];
                if (pc < 32) usedMask = usedMask | (d.u32(1) << pc);
            }
            const slo = csrColorSmallLayout.$.constraintCsr[bid];
            const shi =
                slo + csrColorSmallLayout.$.constraintCsr[roRo.layout.$.params.eidCap + bid];
            for (let e = slo; e < shi; e++) {
                const other = bitcastF32toU32(
                    csrColorSmallLayout.$.constraintList[e * d.u32(CONSTRAINT_VEC4) + 2].x,
                );
                if (other <= bid) continue;
                if (roRo.bMass(other) <= 0) continue;
                const pc = csrColorSmallLayout.$.colors[other];
                if (pc < 32) usedMask = usedMask | (d.u32(1) << pc);
            }
            let chosen = csrColorSmallLayout.$.colors[bid]; // incremental: keep the prior color when still free
            let needsNew = chosen >= colorsN;
            if (!needsNew) needsNew = (usedMask & (d.u32(1) << chosen)) !== 0;
            if (needsNew) {
                let found = false;
                for (let c = d.u32(0); c < colorsN; c++) {
                    if ((usedMask & (d.u32(1) << c)) === 0) {
                        chosen = c;
                        found = true;
                        break;
                    }
                }
                if (!found) chosen = bid % colorsN; // fold past the cap — a tolerated same-color conflict
            }
            csrColorSmallLayout.$.colorScratch[bid] = chosen;
            std.atomicMax(csrColorWgMax.$, chosen + 1);
        }
        std.storageBarrier();
        std.workgroupBarrier();

        // commit the staged colors + publish the used-color count (word 0; word 1 is packScan's live count)
        for (let dOwn = t; dOwn < n; dOwn = dOwn + PACK_WG) {
            const bid = csrColorSmallLayout.$.eids[1 + dOwn];
            csrColorSmallLayout.$.colors[bid] = csrColorSmallLayout.$.colorScratch[bid];
        }
        if (t === 0)
            std.atomicStore(csrColorSmallLayout.$.colorCount[0], std.atomicLoad(csrColorWgMax.$));
    })
    .$name("csrColorSmallMain");

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
        PACK_B_CONSTS_WGSL +
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
 * `a`'s anchor `rA`: a linear anchor pin + an angular relative-orientation lock. `a`/`b` are body eids,
 * `rA`/`rB` the anchors in each body's local frame. **`a = {@link WORLD}` (any `a < 0`) makes `rA` a
 * world-space point** with no body A: the constraint pins `b` to a fixed world anchor with no anchor body
 * (so no anchor↔b contact), `b` dangling freely; drive the point each frame with {@link PhysicsStep.setJointAnchor}.
 * Defaults match `joint()`/`Joint::Joint`: rigid linear (`stiffnessLin = ∞`) + free rotation
 * (`stiffnessAng = 0`) = a spherical joint; pass `stiffnessAng = Infinity` for a fixed joint. Two
 * construction-time guards reject an energy-injecting authoring (deactivate, the GPU analog of `joint()`'s
 * throw): the two endpoints must NOT both be non-dynamic (`mass ≤ 0`, the world counting as static), since a joint
 * no dynamic body can resolve ramps its dual penalty + λ unbounded (a `counters[1]` bump); and the anchors
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

interface Pass {
    pipeline: GPUComputePipeline;
    layout: GPUBindGroupLayout;
    /** which shared solver variant this kernel's accessors resolved against — the bind group the
     *  dispatch helpers set at {@link SOLVER_GROUP}. `undefined` = a kernel that splices no accessor
     *  (the CSR scan, the pack passes, which address `bodies` by column through their own binding). */
    variant?: VariantTag;
}

type VariantTag = "roRo" | "roRw" | "rwRw";

const variants: Record<VariantTag, ReturnType<typeof accessors>> = { roRo, roRw, rwRw };

async function buildPass(
    device: GPUDevice,
    label: string,
    code: string,
    entries: GPUBindGroupLayoutEntry[],
    variant?: VariantTag,
): Promise<Pass> {
    const layout = device.createBindGroupLayout({ label, entries });
    const groups = [layout];
    if (variant) groups.push(Compute.root.unwrap(variants[variant].layout));
    const pipeline = await device.createComputePipelineAsync({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: groups }),
        compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    return { pipeline, layout, variant };
}

const ro: GPUBufferBindingType = "read-only-storage";
const rw: GPUBufferBindingType = "storage";
const buf = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
});

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
    /** the aabb prim pass — a typed pipeline, bound to its own I/O + the shared solver group */
    private readonly _aabbPipe: TgpuComputePipeline;
    /** the two broadphase regimes (C1.0) — typed pipelines, bound to their own I/O + the shared roRo group */
    private readonly _broadphasePipe: TgpuComputePipeline;
    private readonly _broadphaseSmallPipe: TgpuComputePipeline;
    // the narrowphase is four typed pipelines (box / rounded / hull / rounded-poly), each calling only its
    // own shape-pair SAT fn (`tgpu.resolve`'s per-kernel call-graph walk is the DXC pipeline split — no
    // manual chunk composition). Dispatched in order each step; rebuilt (re-`.with()`'d, not recompiled)
    // whenever `setHulls` grows the `hullData` buffer.
    private _collidePipes: TgpuComputePipeline[];
    /** the inertial target + adaptive warmstart pass — a typed pipeline, own I/O + the shared rwRw group */
    private readonly _inertialPipe: TgpuComputePipeline;
    /** the per-color primal: `COLOR_CAP` per-color STATIC bind groups (typegpu has no dynamic-offset
     * uniform binding), like `_commitColorPipes` — own I/O splits across `primalOwnLayout` (group 0),
     * the shared roRo group, `jointRo` (read-only joint records), and `primalColorLayout` (colors/eids/
     * solveOut/color); rebuilt in `buildSolveBindGroups` alongside the growable csr/constraint buffers. */
    private _primalColorPipes!: TgpuComputePipeline[];
    private _primalBase?: TgpuComputePipeline;
    /** the per-color commit: applies `solveOut` → `bodies` for the current color (the double-buffer
     * write) — a typed pipeline. `COLOR_CAP` per-color STATIC bind groups (typegpu has no dynamic-offset
     * uniform binding) let the color loop select `_commitColorPipes[c]` by index at encode; one physical
     * buffer per color dodges the 256-byte `minUniformBufferOffsetAlignment` rule. */
    private readonly _commitColorPipes: TgpuComputePipeline[];
    /** one 4-byte uniform buffer per color, seeded once at construction with its own index — shared by
     * `_commitColorPipes` and `_primalColorPipes`. */
    private readonly _colorIdxBufs: GPUBuffer[];
    /** the contact dual update — a typed pipeline, own I/O + the shared roRw group. */
    private readonly _dualPipe: TgpuComputePipeline;
    /** the LDS-resident solve (C1.2) — a typed pipeline; own I/O splits across `primalOwnLayout` (group 0,
     * forced by `solverLds.solvePose`'s internal reads), the shared rwRw group, `jointRw`, and `ldsLayout`
     * (colors/eids/denseMap). Rebuilt in `buildSolveBindGroups` alongside the same growable buffers. */
    private _solveLdsPipe!: TgpuComputePipeline;
    private _solveLdsBase?: TgpuComputePipeline;
    /** the incremental-greedy body coloring — a typed pipeline; re-`.with()`'d (never recompiled) whenever
     * `setSprings`/`setJoints` grows `constraintList` (the collide-pipe growth precedent) */
    private _coloringPipe!: TgpuComputePipeline;
    /** the joint hard-conflict coloring repair (Phase 6.2), run JOINT_REPAIR_ROUNDS times after the greedy —
     * a typed pipeline, re-`.with()`'d alongside `_coloringPipe` */
    private _repairPipe!: TgpuComputePipeline;
    /** per-joint warmstart + C₀ capture (Phase 6.2), before the main loop — a typed pipeline; re-`.with()`'d
     * (never recompiled) whenever `setJoints` grows `jointRecords` (the `_jointBG` regrowth, alongside
     * `_coloringPipe`/`_repairPipe`'s `constraintList` regrowth). */
    private _jointInitPipe!: TgpuComputePipeline;
    /** per-joint dual update (Phase 6.2), after each iteration's primal — a typed pipeline, re-`.with()`'d
     * alongside `_jointInitPipe`. */
    private _jointDualPipe!: TgpuComputePipeline;
    private readonly _velocityPipe: TgpuComputePipeline;
    /** the compose pipeline, bound by {@link prepareCompose} once the external firehose exists. */
    private _composePipe: TgpuComputePipeline | null = null;
    private _composeBase?: TgpuComputePipeline;
    private readonly _csrCountPipe: TgpuComputePipeline;
    private readonly _csrScan: Pass;
    private readonly _csrScatterPipe: TgpuComputePipeline;
    /** the fused small-N CSR + coloring tail (C1.1) — replaces the 3 CSR passes + colorize in the small
     * regime. A typed pipeline; re-`.with()`'d alongside `_coloringPipe`/`_repairPipe` (binds the same
     * growable `constraintCsr`/`constraintList`). */
    private _csrColorSmallPipe!: TgpuComputePipeline;
    // the pack (membership-scan compaction + one-time seed; count → scan → scatter, C1.3) is plugin-only
    // (membership-gated / slab-sourced); null for the standalone gates, which set `eids` via
    // `gateSetCount` + seed `bodies` by `writeBuffer` directly.
    private readonly _packCount: Pass | null;
    private readonly _packScan: Pass | null;
    private readonly _packScatter: Pass | null;
    private readonly _sharedBG: Record<VariantTag, GPUBindGroup>;
    // the joint-records group (JOINT_GROUP): jointRecords grows on `setJoints`, so both are rebuilt
    // (alongside `_jointInitPipe`/`_jointDualPipe`/`_primalColorPipes`/`_solveLdsPipe`'s re-`.with()`)
    // whenever it does. `_jointBG` is mutable (joint-init/joint-dual/solve-lds write it); `_jointBGRo` is
    // read-only (the typed primal only reads it).
    private _jointBG!: GPUBindGroup;
    private _jointBGRo!: GPUBindGroup;
    private readonly _csrScanBG: GPUBindGroup;
    // built lazily on the first pack call (the slab .gpu + membership buffers aren't allocated at warm —
    // parallel warms); rebuilt if a source identity changes (stable in practice — allocated once)
    private _packCountBG: GPUBindGroup | null = null;
    private _packScanBG: GPUBindGroup | null = null;
    private _packScatterBG: GPUBindGroup | null = null;
    private _gatherInputs: Inputs | null = null;
    // rebuilt if the external firehose buffer's identity changes (it doesn't in practice —
    // TransformsPlugin allocates it once); tracks `_composePipe`'s built-against identity
    private _composeDst: GPUBuffer | null = null;
    private _composeCompiled = false;
    private _composePreparation: Promise<void> | null = null;
    private _composeError: unknown;
    private _iterations = 10;
    // dispatched-color cap: the primal dispatches `_maxColors` color-passes per iteration (empty colors
    // no-op via the early-out), the coloring folds bodies past it (avbd.md "Dispatch count"). 32 = no cap.
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
    // this instance's precompile label prefix — an app can stand up several worlds (the gym `pile` builds
    // two, `constraints` three) and the queue rejects a duplicate label. Captured once at construction
    // because `phys-compose` registers lazily on the first `compose()`, long after it.
    private readonly _scope: string;
    private readonly _registrations: Promise<void>[] = [];

    private constructor(
        device: GPUDevice,
        eidCap: number,
        maxBodies: number,
        bvh: Bvh,
        passes: {
            csrScan: Pass;
            packCount: Pass | null;
            packScan: Pass | null;
            packScatter: Pass | null;
        },
    ) {
        this.device = device;
        this._scope = precompileScope("phys");
        const register = (label: string, force: () => unknown): void => {
            this._registrations.push(precompile(label, force));
        };
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
            size: STEP_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._interpUbo = device.createBuffer({
            label: "phys-interp",
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._csrScan = passes.csrScan;
        this._packCount = passes.packCount;
        this._packScan = passes.packScan;
        this._packScatter = passes.packScatter;

        // the shared solver group (params + bodies + pairContacts), one bind group per access variant —
        // every kernel's accessors resolved against one of these, so the dispatch helpers set it at
        // SOLVER_GROUP and no kernel re-declares the three bindings.
        const root = Compute.root;
        const shared = {
            params: this._stepUbo,
            bodies: this.bodies,
            pairContacts: this.pairContacts,
        };
        this._sharedBG = {
            roRo: root.unwrap(root.createBindGroup(roRo.layout, shared)),
            roRw: root.unwrap(root.createBindGroup(roRw.layout, shared)),
            rwRw: root.unwrap(root.createBindGroup(rwRw.layout, shared)),
        };
        this._aabbPipe = root
            .createComputePipeline({ compute: aabbKernel })
            .$name("phys-aabb")
            .with(root.createBindGroup(aabbLayout, { prims: bvh.prims, eids: this.eids }))
            .with(roRo.layout, this._sharedBG.roRo);
        // a typed pipeline is created synchronously, so Dawn defers its shader compile to the first
        // dispatch — a first-frame hitch where the raw passes' `createComputePipelineAsync` compiled at
        // build. Force it at warm with a 0-workgroup dispatch, the shipped drain. Both groups are
        // already bound here, so the drain's bound-nothing guard is inert — a forcer that allocates its
        // own buffers must return the *dispatch* for that guard to fire.
        register(`${this._scope}-aabb`, () => {
            this._aabbPipe.dispatchWorkgroups(0);
            return this._aabbPipe;
        });
        this._broadphasePipe = root
            .createComputePipeline({ compute: broadphaseKernel })
            .$name("phys-broadphase")
            .with(
                root.createBindGroup(broadphaseLayout, {
                    nodes: bvh.nodes,
                    pairList: this.pairList,
                    counters: this.counters,
                    eids: this.eids,
                }),
            )
            .with(roRo.layout, this._sharedBG.roRo);
        register(`${this._scope}-broadphase`, () => {
            this._broadphasePipe.dispatchWorkgroups(0);
            return this._broadphasePipe;
        });
        this._broadphaseSmallPipe = root
            .createComputePipeline({ compute: broadphaseSmallKernel })
            .$name("phys-broadphase-small")
            .with(
                root.createBindGroup(broadphaseSmallLayout, {
                    prims: bvh.prims,
                    pairList: this.pairList,
                    counters: this.counters,
                    eids: this.eids,
                }),
            )
            .with(roRo.layout, this._sharedBG.roRo);
        register(`${this._scope}-broadphase-small`, () => {
            this._broadphaseSmallPipe.dispatchWorkgroups(0);
            return this._broadphaseSmallPipe;
        });
        this._collidePipes = this._makeCollidePipes();
        for (const [i, label] of ["box", "rounded", "hull", "rounded-poly"].entries()) {
            register(`${this._scope}-collide-${label}`, () => {
                const pipe = this._collidePipes[i];
                pipe.dispatchWorkgroups(0);
                return pipe;
            });
        }
        this._inertialPipe = root
            .createComputePipeline({ compute: inertialKernel })
            .$name("phys-inertial")
            .with(root.createBindGroup(inertialLayout, { eids: this.eids }))
            .with(rwRw.layout, this._sharedBG.rwRw);
        register(`${this._scope}-inertial`, () => {
            this._inertialPipe.dispatchWorkgroups(0);
            return this._inertialPipe;
        });
        this._velocityPipe = root
            .createComputePipeline({ compute: velocityKernel })
            .$name("phys-velocity")
            .with(root.createBindGroup(velocityLayout, { eids: this.eids }))
            .with(rwRw.layout, this._sharedBG.rwRw);
        register(`${this._scope}-velocity`, () => {
            this._velocityPipe.dispatchWorkgroups(0);
            return this._velocityPipe;
        });
        this._csrCountPipe = root
            .createComputePipeline({ compute: csrCountKernel })
            .$name("phys-csr-count")
            .with(root.createBindGroup(csrCountLayout, { csr: this.csr, eids: this.eids }))
            .with(roRo.layout, this._sharedBG.roRo);
        register(`${this._scope}-csr-count`, () => {
            this._csrCountPipe.dispatchWorkgroups(0);
            return this._csrCountPipe;
        });
        this._csrScatterPipe = root
            .createComputePipeline({ compute: csrScatterKernel })
            .$name("phys-csr-scatter")
            .with(
                root.createBindGroup(csrScatterLayout, {
                    csr: this.csr,
                    csrList: this.csrList,
                    eids: this.eids,
                }),
            )
            .with(roRo.layout, this._sharedBG.roRo);
        register(`${this._scope}-csr-scatter`, () => {
            this._csrScatterPipe.dispatchWorkgroups(0);
            return this._csrScatterPipe;
        });
        this._colorIdxBufs = [];
        for (let c = 0; c < COLOR_CAP; c++) {
            const buf = device.createBuffer({
                label: `phys-color-idx-${c}`,
                size: d.sizeOf(ColorIdx),
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(buf, 0, new Uint32Array([c]));
            this._colorIdxBufs.push(buf);
        }
        const commitBase = root
            .createComputePipeline({ compute: commitKernel })
            .$name("phys-commit");
        this._commitColorPipes = this._colorIdxBufs.map((color) =>
            commitBase
                .with(
                    root.createBindGroup(commitLayout, {
                        solveOut: this.solveOut,
                        colors: this.colors,
                        eids: this.eids,
                        color,
                    }),
                )
                .with(rwRw.layout, this._sharedBG.rwRw),
        );
        register(`${this._scope}-commit`, () => {
            const pipe = this._commitColorPipes[0];
            pipe.dispatchWorkgroups(0);
            return pipe;
        });
        this._dualPipe = root
            .createComputePipeline({ compute: dualKernel })
            .$name("phys-dual")
            .with(root.createBindGroup(dualLayout, { eids: this.eids }))
            .with(roRw.layout, this._sharedBG.roRw);
        register(`${this._scope}-dual`, () => {
            this._dualPipe.dispatchWorkgroups(0);
            return this._dualPipe;
        });
        this._csrScanBG = device.createBindGroup({
            label: "phys-csr-scan",
            layout: this._csrScan.layout,
            entries: [
                { binding: 0, resource: { buffer: this.eids } },
                { binding: 1, resource: { buffer: this.csr } },
            ],
        });
        // the solve bind groups last — they bind the growable constraintList / jointRecords, so a set* that
        // grows either rebuilds them (primal + solve-lds + csrColorSmall + jointInit + jointDual)
        this.buildSolveBindGroups();
        // force-compiled once here (like the collide pipes) — a later constraintList/jointRecords grow
        // re-`.with()`s these pipelines in buildSolveBindGroups without re-registering a precompile
        // forcer, matching setHulls' growth policy for the collide pipes.
        register(`${this._scope}-primal`, () => {
            const pipe = this._primalColorPipes[0];
            pipe.dispatchWorkgroups(0);
            return pipe;
        });
        register(`${this._scope}-solve-lds`, () => {
            this._solveLdsPipe.dispatchWorkgroups(0);
            return this._solveLdsPipe;
        });
        register(`${this._scope}-coloring`, () => {
            this._coloringPipe.dispatchWorkgroups(0);
            return this._coloringPipe;
        });
        register(`${this._scope}-repair`, () => {
            this._repairPipe.dispatchWorkgroups(0);
            return this._repairPipe;
        });
        register(`${this._scope}-csr-color-small`, () => {
            this._csrColorSmallPipe.dispatchWorkgroups(0);
            return this._csrColorSmallPipe;
        });
        register(`${this._scope}-joint-init`, () => {
            this._jointInitPipe.dispatchWorkgroups(0);
            return this._jointInitPipe;
        });
        register(`${this._scope}-joint-dual`, () => {
            this._jointDualPipe.dispatchWorkgroups(0);
            return this._jointDualPipe;
        });
    }

    // the coloring + repair pipelines bind the growable `constraintList` (setSprings/setJoints), so — like
    // the collide pipes' `hullData` growth — a grow re-`.with()`s onto the new buffer without recompiling.
    private _makeColoringPipe(): TgpuComputePipeline {
        const root = Compute.root;
        this._coloringBase ??= root
            .createComputePipeline({ compute: coloringKernel })
            .$name("phys-coloring");
        const group0 = root.createBindGroup(coloringLayout, {
            csr: this.csr,
            csrList: this.csrList,
            colors: this.colors,
            colorScratch: this.colorScratch,
            eids: this.eids,
            colorCount: this.colorCount,
            constraintCsr: this.constraintCsr,
            constraintList: this.constraintList,
        });
        return this._coloringBase.with(group0).with(roRo.layout, this._sharedBG.roRo);
    }
    private _coloringBase?: TgpuComputePipeline;

    private _makeRepairPipe(): TgpuComputePipeline {
        const root = Compute.root;
        this._repairBase ??= root
            .createComputePipeline({ compute: repairKernel })
            .$name("phys-repair");
        const group0 = root.createBindGroup(repairLayout, {
            colors: this.colors,
            colorScratch: this.colorScratch,
            constraintCsr: this.constraintCsr,
            constraintList: this.constraintList,
            eids: this.eids,
            colorCount: this.colorCount,
        });
        return this._repairBase.with(group0).with(roRo.layout, this._sharedBG.roRo);
    }
    private _repairBase?: TgpuComputePipeline;

    // the four collide kernels compile once (memoized here); only the group-0 bind group depends on
    // `hullData`, which grows (a new buffer) on demand — so a `setHulls` growth re-`.with()`s the four
    // pipelines onto the new bind group without recompiling any of them.
    private _collideBase?: TgpuComputePipeline[];

    private _makeCollidePipes(): TgpuComputePipeline[] {
        const root = Compute.root;
        this._collideBase ??= [
            root.createComputePipeline({ compute: collideBoxKernel }).$name("phys-collide-box"),
            root
                .createComputePipeline({ compute: collideRoundedKernel })
                .$name("phys-collide-rounded"),
            root.createComputePipeline({ compute: collideHullKernel }).$name("phys-collide-hull"),
            root
                .createComputePipeline({ compute: collideRoundedPolyKernel })
                .$name("phys-collide-rounded-poly"),
        ];
        const group0 = root.createBindGroup(collideLayout, {
            counters: this.counters,
            pairList: this.pairList,
            eids: this.eids,
            hullData: this.hullData,
        });
        return this._collideBase.map((pipe) =>
            pipe.with(group0).with(roRw.layout, this._sharedBG.roRw),
        );
    }

    /**
     * Upload the packed convex-hull geometry (`./hull` packHulls) the collide pass reads for ShapeKind.Hull
     * bodies. The buffer grows on demand (rebinding the four collide pipelines onto the new buffer).
     * Idempotent — a no-op when the data is unchanged is the caller's job (the plugin uploads only when the
     * `Hulls` registry changes).
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
            this._collidePipes = this._makeCollidePipes();
        }
        this.device.queue.writeBuffer(this.hullData, 0, data as Uint32Array<ArrayBuffer>);
    }

    private buildSolveBindGroups(): void {
        const root = Compute.root;
        // `primalOwnLayout` (csr/csrList/constraintCsr/constraintList) is forced onto both the typed
        // primal and solve-lds by `solverRoRo`/`solverLds`'s internal reads — one bind group, shared.
        const ownGroup = root.createBindGroup(primalOwnLayout, {
            csr: this.csr,
            csrList: this.csrList,
            constraintCsr: this.constraintCsr,
            constraintList: this.constraintList,
        });
        // the joint-records group grows with `jointRecords` (setJoints) — rebuilt here alongside every
        // pipeline that binds it (mirroring the constraintList regrowth above).
        this._jointBG = root.unwrap(
            root.createBindGroup(jointRw.layout, { jointRecords: this.jointRecords }),
        );
        this._jointBGRo = root.unwrap(
            root.createBindGroup(jointRo.layout, { jointRecords: this.jointRecords }),
        );

        this._primalBase ??= root
            .createComputePipeline({ compute: primalKernel })
            .$name("phys-primal");
        this._primalColorPipes = this._colorIdxBufs.map((color) =>
            this._primalBase!.with(ownGroup)
                .with(roRo.layout, this._sharedBG.roRo)
                .with(jointRo.layout, this._jointBGRo)
                .with(
                    root.createBindGroup(primalColorLayout, {
                        colors: this.colors,
                        eids: this.eids,
                        solveOut: this.solveOut,
                        color,
                    }),
                ),
        );

        this._solveLdsBase ??= root
            .createComputePipeline({ compute: solveLdsKernel })
            .$name("phys-solve-lds");
        this._solveLdsPipe = this._solveLdsBase
            .with(ownGroup)
            .with(rwRw.layout, this._sharedBG.rwRw)
            .with(jointRw.layout, this._jointBG)
            .with(
                root.createBindGroup(ldsLayout, {
                    colors: this.colors,
                    eids: this.eids,
                    // the eid → dense map rides the solveOut scratch (unused by the LDS path; 2·eidCap
                    // vec4s ≥ the eidCap u32s the map needs), rebound as array<u32>
                    denseMap: this.solveOut,
                }),
            );

        // the coloring + repair + csrColorSmall pipelines bind the same growable constraintCsr/constraintList
        // — re-`.with()`'d here too (a no-op the first time this runs, from the constructor, before
        // `_coloringBase`/`_repairBase`/`_csrColorSmallBase` exist)
        this._coloringPipe = this._makeColoringPipe();
        this._repairPipe = this._makeRepairPipe();
        this._csrColorSmallPipe = this._makeCsrColorSmallPipe();
        this._jointInitPipe = this._makeJointInitPipe();
        this._jointDualPipe = this._makeJointDualPipe();
    }

    private _csrColorSmallBase?: TgpuComputePipeline;
    private _makeCsrColorSmallPipe(): TgpuComputePipeline {
        const root = Compute.root;
        this._csrColorSmallBase ??= root
            .createComputePipeline({ compute: csrColorSmallKernel })
            .$name("phys-csr-color-small");
        const group0 = root.createBindGroup(csrColorSmallLayout, {
            csr: this.csr,
            csrList: this.csrList,
            colors: this.colors,
            colorScratch: this.colorScratch,
            eids: this.eids,
            colorCount: this.colorCount,
            constraintCsr: this.constraintCsr,
            constraintList: this.constraintList,
        });
        return this._csrColorSmallBase.with(group0).with(roRo.layout, this._sharedBG.roRo);
    }

    private _jointInitBase?: TgpuComputePipeline;
    private _makeJointInitPipe(): TgpuComputePipeline {
        const root = Compute.root;
        this._jointInitBase ??= root
            .createComputePipeline({ compute: jointInitKernel })
            .$name("phys-joint-init");
        const group0 = root.createBindGroup(jointInitLayout, {
            jointVersions: this.jointVersions,
            counters: this.counters,
            seeded: this.seeded,
        });
        return this._jointInitBase
            .with(group0)
            .with(roRo.layout, this._sharedBG.roRo)
            .with(jointRw.layout, this._jointBG);
    }

    private _jointDualBase?: TgpuComputePipeline;
    private _makeJointDualPipe(): TgpuComputePipeline {
        const root = Compute.root;
        this._jointDualBase ??= root
            .createComputePipeline({ compute: jointDualKernel })
            .$name("phys-joint-dual");
        return this._jointDualBase
            .with(root.createBindGroup(jointDualLayout, { counters: this.counters }))
            .with(roRo.layout, this._sharedBG.roRo)
            .with(jointRw.layout, this._jointBG);
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
        const [csrScan, packCount, packScan, packScatter] = await Promise.all([
            buildPass(device, "phys-csr-scan", csrScanWgsl(maxBodies, eidCap), [
                buf(0, ro),
                buf(1, rw),
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
        const step = new PhysicsStep(device, eidCap, maxBodies, bvh, {
            csrScan,
            packCount,
            packScan,
            packScatter,
        });
        const registrations = await Promise.allSettled(step._registrations.splice(0));
        const rejected = registrations.find(
            (result): result is PromiseRejectedResult => result.status === "rejected",
        );
        if (rejected) throw rejected.reason;
        return step;
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
        const ab = new ArrayBuffer(STEP_BYTES);
        new Uint32Array(ab, 0, 4).set([this.recordCap, p.iterations, this.eidCap, this._maxColors]);
        new Float32Array(ab, 16, 4).set([h, p.gravity, p.alpha, p.penalty]);
        // betaAng (the joint angular ramp, Phase 6.2) defaults to the canonical 100 when unset (contacts/springs
        // don't read it); jointCount is preserved (setJoints writes it separately, so a reconfigure mustn't zero it).
        new Float32Array(ab, 32, 4).set([1 / (h * h), p.betaLin, p.gamma, p.betaAng ?? 100]);
        // jointCount (48) + substeps (52). setJoints rewrites only jointCount (48), preserving substeps (52).
        new Uint32Array(ab, STEP_JOINT_COUNT, 2).set([this._jointCount, this._substeps]);
        // offsets 56/60 (pad2/pad3) stay 0 — the ArrayBuffer is zero-init.
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
     * force the next pack to re-seed an eid's solver slot (ecs.md "An eid is a borrow")
     * — call when an eid is recycled to a NEW `Body` within one update (a destroy + same-update create).
     * The pack clears `seeded` only for a non-member eid, so a same-update realias — where the eid stays a
     * `Body` member the whole time — never clears it, and the new body would inherit the destroyed one's
     * seeded pose + velocity. Clearing the flag makes the next pack seed the slot fresh from the authored
     * slabs; evicting `_kinPrev` drops the destroyed body's kinematic pose history so a moving-platform
     * velocity isn't derived across the realias. The CPU stamp diff (the plugin's PackSystem) detects the
     * realias; this resets the GPU-side seed state it can't otherwise observe.
     */
    reseed(eid: number): void {
        this.device.queue.writeBuffer(this.seeded, eid * 4, new Uint32Array(1));
        this._kinPrev.delete(eid);
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
        this.device.queue.writeBuffer(
            this._stepUbo,
            STEP_JOINT_COUNT,
            new Uint32Array([this._jointCount]),
        );
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
     * `usedColors <= 0` (no readback yet / empty scene) keeps the full cap, the safe cold-start.
     * The margin covers the readback's 1-2 frame staleness; a
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
            this._packCount!,
            this._packCountBG!,
            this._packWgs,
            Compute.span?.("phys:pack"),
        );
        this.pass(encoder, this._packScan!, this._packScanBG!, 1, Compute.span?.("phys:pack"));
        this.pass(
            encoder,
            this._packScatter!,
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
            {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:aabb"),
                });
                this._aabbPipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
                pass.end();
            }
            // broadphase regime (C1.0): at a frame-stale live count ≤ smallN the one-dispatch O(n²) scan
            // covers the whole front-end (the BVH build's ~28 dependent phases are structure tax at gameplay
            // counts); past it, the BVH build + descent. Both write identical
            // per-eid pair blocks (shared candidate/emit WGSL), so warmstart carries across a regime flip,
            // and both are exact at any N, so the stale switch is correctness-safe. Same span name — the
            // profiler's phys:broadphase is the front-end pair search whichever regime ran.
            const small = this._liveBound > 0 && this._liveBound <= this._smallN;
            this._smallRan = small;
            if (small) {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:broadphase"),
                });
                this._broadphaseSmallPipe
                    .with(pass)
                    .dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
                pass.end();
            } else {
                this._bvh.build(encoder);
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:broadphase"),
                });
                this._broadphasePipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
                pass.end();
            }
            // narrowphase (collide): SAT + in-place warmstart over the live bodies' per-eid pair blocks. FOUR
            // typed pipelines (box / rounded / hull / rounded-poly) by shape-pair class (the DXC pipeline
            // split falls out of `tgpu.resolve`'s per-kernel call graph). All dispatch indirect off pairArgs
            // (= ceil(liveCount·PAIRS_PER_BODY/64) workgroups, written by packScan) over the SAME slots; each
            // lane → d → owner eid → slot = eid·K + k, early-out past the live body count + the class gate.
            // One compute pass — each pipeline carries its own pre-bound groups (`.with()`), so the
            // consecutive dispatches' write→read hazards on pairContacts are barrier-ordered (the box
            // pipeline clears dead slots first), as with the primal/commit pair below. Same span name → the
            // profiler sums all four.
            {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:collide"),
                });
                for (const pipe of this._collidePipes) {
                    pipe.with(pass).dispatchWorkgroupsIndirect(this.pairArgs, 0);
                }
                pass.end();
            }

            // CSR + coloring tail: the small regime runs ONE single-WG fused dispatch (C1.1 — the multi-WG
            // passes' boundaries are near-pure structure tax at gameplay counts; see csrColorSmallWgsl()).
            // The fused coloring runs before jointInit/inertial while the BVH regime's colorize runs after —
            // safe, the coloring reads only mass + this step's contact/constraint adjacency, none of which
            // those passes write. The joint hard-conflict repair keeps its own snapshot+pass rounds in both
            // regimes. The BVH regime keeps the multi-WG passes (work-bound at scale, where a single WG
            // would serialize).
            if (small) {
                {
                    const pass = encoder.beginComputePass({
                        timestampWrites: Compute.span?.("phys:csr"),
                    });
                    this._csrColorSmallPipe.with(pass).dispatchWorkgroups(1);
                    pass.end();
                }
                if (this._jointCount > 0) this.repairColors(encoder);
            } else {
                this.buildCsr(encoder);
            }

            // joint warmstart + C₀ capture (Phase 6.2) — before inertial init, so it reads the step-start pose
            // x⁻ (the contact warmstart in collide above reads it the same way). One thread per joint, direct
            // dispatch (the count is CPU-authored, not GPU-resident); skipped entirely when there are no joints.
            if (this._jointCount > 0) {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:joint"),
                });
                this._jointInitPipe.with(pass).dispatchWorkgroups(Math.ceil(this._jointCount / 64));
                pass.end();
            }

            {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:inertial"),
                });
                this._inertialPipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
                pass.end();
            }

            // real incremental-greedy coloring ahead of the primal — the dispatch collapse: the primal
            // loops `maxColors` colors, not one per body (physics.md "Dispatch count"). The coloring reads
            // this step's CSR adjacency + last step's colors snapshot, folds bodies past the cap. In the
            // small regime the fused tail above already colored.
            if (!small) this.colorize(encoder);

            // LDS-resident solve regime (C1.2): at a frame-stale live count ≤ ldsN the whole iters × colors
            // primal/commit/dual block below runs as ONE single-workgroup dispatch with every live body's
            // pose in workgroup memory (solveLdsKernel — the per-color dependent round trip the looped path
            // pays in storage becomes an in-kernel barrier on LDS). Same solve math (solvePose / dualSlot /
            // jointDualOne are shared chunks), so GPU == oracle holds on either path; the full block reports
            // under phys:primal (phys:dual / phys:joint read 0 in this regime, the phys:csr precedent).
            const lds = this._liveBound > 0 && this._liveBound <= this._ldsN;
            this._ldsRan = lds;
            if (lds) {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:primal"),
                });
                this._solveLdsPipe.with(pass).dispatchWorkgroups(1);
                pass.end();
            } else {
                for (let it = 0; it < this._iterations; it++) {
                    // timestamp every iteration — the profiler sums same-named spans, so this reports the FULL
                    // primal GPU time (all iterations × color dispatches), not just it 0.
                    const pass = encoder.beginComputePass({
                        timestampWrites: Compute.span?.("phys:primal"),
                    });
                    // `_colorsToRun` colors/iteration (the readback-bounded count, Phase 4.9 Lever 1 — full cap until
                    // boundColors is fed a usedColors readback), one compute pass (the color rides a per-color
                    // static bind group, no advanceColor dispatch). Each color is primal-then-commit: the primal
                    // solves that color's bodies into `solveOut` reading the committed `bodies`, then the commit
                    // applies `solveOut` → `bodies` for that color so the next color's primal sees it (the
                    // double-buffer, Phase 4.5 Stage B). Consecutive same-pass dispatches with a write→read hazard
                    // (primal→commit on solveOut, commit→next-primal on bodies) are ordered by the implementation's
                    // barrier — the single-step gate confirms. Each dispatches DIRECT off `_bodyGroups` (the
                    // frame-stale live count + BODY_MARGIN, rung 0 — an indirect dispatch costs ≈ 2× and this loop
                    // is the dominant block); over-dispatched workgroups and colors past the live set early-out on
                    // `eids[0]` and no-op.
                    for (let c = 0; c < this._colorsToRun; c++) {
                        this._primalColorPipes[c].with(pass).dispatchWorkgroups(this._bodyGroups);
                        this._commitColorPipes[c].with(pass).dispatchWorkgroups(this._bodyGroups);
                    }
                    pass.end();

                    // dual update: λ ← F + the penalty ramp over the live contacts, reading the pose the primal
                    // colors just wrote. One thread per per-eid pair slot (indirect off pairArgs, looping its records,
                    // inactive records early-out). Updates λ/k in place in pairContacts — which IS the persistent
                    // store, so next frame's collide warmstarts off it.
                    {
                        const dualPass = encoder.beginComputePass({
                            timestampWrites: Compute.span?.("phys:dual"),
                        });
                        this._dualPipe.with(dualPass).dispatchWorkgroupsIndirect(this.pairArgs, 0);
                        dualPass.end();
                    }

                    // joint dual: advance λ + the penalty ramp per joint, reading the pose this iteration's primal
                    // wrote (like the contact dual). One thread per joint, in place in the persistent jointRecords.
                    if (this._jointCount > 0) {
                        const jointPass = encoder.beginComputePass({
                            timestampWrites: Compute.span?.("phys:joint"),
                        });
                        this._jointDualPipe
                            .with(jointPass)
                            .dispatchWorkgroups(Math.ceil(this._jointCount / 64));
                        jointPass.end();
                    }
                }
            }

            {
                const pass = encoder.beginComputePass({
                    timestampWrites: Compute.span?.("phys:velocity"),
                });
                this._velocityPipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
                pass.end();
            }
            // No cache pass: warmstart is in place — the dual wrote this sub-step's final λ/k into pairContacts,
            // the persistent store, so the next sub-step's (or frame's) collide reads it at the same slot (Phase 4.7).
        }
    }

    /** bind and validate the compose pipeline once the external transforms firehose exists. */
    async prepareCompose(transforms: GPUBuffer): Promise<void> {
        if (this._composeDst === transforms && this._composePipe) return;
        if (this._composeError) throw this._composeError;
        if (this._composePreparation) {
            await this._composePreparation;
            if (this._composeDst === transforms && this._composePipe) return;
        }
        const root = Compute.root;
        this._composeBase ??= root
            .createComputePipeline({ compute: composeKernel })
            .$name("phys-compose");
        const pipe = this._composeBase
            .with(
                root.createBindGroup(composeLayout, {
                    eids: this.eids,
                    transforms,
                    interp: this._interpUbo,
                }),
            )
            .with(roRo.layout, this._sharedBG.roRo);
        if (!this._composeCompiled) {
            const preparation = precompile(`${this._scope}-compose`, () => {
                pipe.dispatchWorkgroups(0);
                return pipe;
            }).then(() => {
                this._composeCompiled = true;
                this._composePipe = pipe;
                this._composeDst = transforms;
            });
            this._composePreparation = preparation;
            try {
                await preparation;
            } catch (error) {
                this._composeError = error;
                throw error;
            } finally {
                if (this._composePreparation === preparation) this._composePreparation = null;
            }
            return;
        }
        this._composePipe = pipe;
        this._composeDst = transforms;
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
     * Await {@link prepareCompose} with this `transforms` buffer before the first call.
     */
    compose(encoder: GPUCommandEncoder, transforms: GPUBuffer, alpha = 1): void {
        if (this._composeError) throw this._composeError;
        if (this._composeDst !== transforms || !this._composePipe) {
            throw new Error("PhysicsStep.compose: await prepareCompose(transforms) before use");
        }
        this._interpData[0] = alpha;
        this.device.queue.writeBuffer(this._interpUbo, 0, this._interpData);
        const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.("phys:compose") });
        this._composePipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
        pass.end();
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
        {
            const pass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("phys:coloring"),
            });
            this._coloringPipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
            pass.end();
        }
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
            const pass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("phys:coloring"),
            });
            this._repairPipe.with(pass).dispatchWorkgroupsIndirect(this.dispatchArgs, 0);
            pass.end();
        }
    }

    /**
     * build the per-body CSR contact adjacency from this step's contacts (count → scan → scatter), so
     * the primal + coloring read only a body's own contacts. Run by `record` after the collide in the
     * BVH regime (the small regime fuses CSR + coloring into one dispatch — csrColorSmallWgsl()); also the
     * standalone entry the coloring crux test calls (it seeds the contact graph, then builds the CSR the
     * coloring reads). The count + scatter dispatch indirect off pairArgs (one thread per per-eid pair slot,
     * looping its records, skipping inactive records); the scan is the single-workgroup parallel prefix.
     */
    buildCsr(encoder: GPUCommandEncoder): void {
        // zero only the count region [eidCap, 2·eidCap); the offset region is fully rewritten by the scan
        encoder.clearBuffer(this.csr, this.eidCap * 4, this.eidCap * 4);
        {
            const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.("phys:csr") });
            this._csrCountPipe.with(pass).dispatchWorkgroupsIndirect(this.pairArgs, 0);
            pass.end();
        }
        this.pass(encoder, this._csrScan, this._csrScanBG, 1, Compute.span?.("phys:csr"));
        {
            const pass = encoder.beginComputePass({ timestampWrites: Compute.span?.("phys:csr") });
            this._csrScatterPipe.with(pass).dispatchWorkgroupsIndirect(this.pairArgs, 0);
            pass.end();
        }
    }

    private pass(
        encoder: GPUCommandEncoder,
        p: Pass,
        bg: GPUBindGroup,
        groups: number,
        timestampWrites?: GPUComputePassTimestampWrites,
    ): void {
        const pass = encoder.beginComputePass({ timestampWrites });
        pass.setPipeline(p.pipeline);
        pass.setBindGroup(0, bg);
        if (p.variant) pass.setBindGroup(SOLVER_GROUP, this._sharedBG[p.variant]);
        pass.dispatchWorkgroups(groups);
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
        this._interpUbo.destroy();
        for (const buf of this._colorIdxBufs) buf.destroy();
    }
}
