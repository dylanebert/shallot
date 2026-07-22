import { Compute, capacity, type Plugin, type State, type System, Time } from "../../engine";
import { BVH_FEATURES } from "../bvh/core";
import { type Mirror, MirrorPlugin, mirror } from "../mirror";
import {
    Body,
    bodyTraits,
    ComposeSystem,
    ConstraintSystem,
    installBackend,
    Joint,
    jointTraits,
    type PhysicsBackend,
    Spring,
    StepSystem,
    springTraits,
    uninstallBackend,
} from "../physics";
import { Hulls } from "../physics/core";
import { SlabPlugin } from "../slab";
import { packHulls } from "./hull";
import { diffStamps } from "./recycle";
import { type Inputs, PENALTY_MIN, PhysicsStep } from "./step";

// AVBD physics — the rigid-body solver re-added to the lean engine, validated against the f64 oracle
// (tests/avbd). This plugin implements the substrate's `PhysicsBackend` handle (`standard/physics`); the
// solver itself is `step.ts` (`PhysicsStep`), the SAT is `collide.ts`. Runs the full `warmstart`
// augmented-Lagrangian layer (λ accumulation + the conditional penalty ramp + friction + cross-frame
// persistence: the collide merges last frame's λ/k by feature key, γ decay).
//
// Storage is eid-indexed over `capacity`, persistent across frames — a body's solver state lives at its
// eid slot and survives spawn/despawn. The whole CPU side is firehose: NO per-entity iteration. Each
// frame a single GPU `pack` pass (PackSystem, draw group) scans capacity gated on the Body membership
// bit and (a) compacts the live eids into the dense→eid map (`eids[0]` = count, `eids[1+d]` = eid) the
// solver passes read, and (b) one-time-seeds any newly-spawned body's slot from its authored slabs
// (gated on a GPU `seeded` flag — existing bodies untouched). The fixed-group StepSystem (substrate)
// solves from last frame's pack output (a 1-frame structural latency — a new body joins the solve next
// frame).
//
// Body / Transform contract (roadmap): `Body.excludes [Transform]` (substrate trait). `Body` carries
// pos/quat (spawn pose, then physics-owned) + mass/halfExtents/friction on slab; this backend owns the
// GPU pose after spawn. The substrate's `ComposeSystem` scatters the live pose into the `transforms`
// firehose (`compose` below), after the Transform compose and before the renderer reads geometry, so a
// `Body`+`Part` renders at the physics-owned pose. A CPU consumer reads the live pose through
// `Physics.backend.readBody` (the pose-read seam pick.ts + the character sweep use) or the raw
// `Avbd.step.bodies` escape hatch.

const GRAVITY = -10;
const ALPHA = 0.99;
// penalty ramp rate (Eq. 17) + warmstart decay (Eq. 19), the canonical AVBD set. The warmstart layer
// carries λ/k across frames, so the ramp converges from the persisted state — the canonical 1e4 holds
// a resting box to ~mg/k.
const BETA_LIN = 1e4;
// the joint angular penalty-ramp rate (Phase 6.2, joint.ts betaAng) — the canonical AVBD value; contacts
// ignore it, so it only matters once a scene authors joints (via Avbd.step.setJoints).
const BETA_ANG = 100;
const GAMMA = 0.999;
// solve iterations — a perf/robustness tradeoff knob, NOT a correctness gate: every fixed count explodes
// on *some* taller stack (iters=4 churns a 16384-body 12-layer pile, iters=10 a taller one), so the choice
// is a deliberate ship tradeoff, not a bug. Production ships 6: it settles a 10-storey stack (the collapse
// showcase wall) where 4 (the paper's count) under-converges and pancakes during the settle, ~1ms@1k on
// lovelace (0.996ms measured, vs 0.575 at 4). The f64 oracle + GPU gates VALIDATE at iters=10
// (corpus.oracle.ts, the gym seeded gates), where the math is proven correct independent of this ship
// value. Raise per-scene via `Avbd.step.configure` for a known-harder pile. physics.md "f32 precision"
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
// dispatch count is bounded by the cap not the body count (avbd.md "Dispatch count"). The reference +
// webphysics both cap at 8; the convergence probe found realistic piles color in ≤4, well under it.
const MAX_COLORS = 8;

/** the running AVBD state: custom tooling + the gym read the GPU pose from `step.bodies` (indexed by
 *  `step.eids`), tune the solver via `step.configure`, or author joints imperatively via `step.setJoints`. */
export const Avbd: { step: PhysicsStep | null } = {
    step: null,
};

// the frame-stale readback of `step.colorCount` — word 0 the greedy's used-color count (the readback-bounded
// color loop, Phase 4.9 Lever 1), word 1 the clamped live body count (the color loop's direct dispatch,
// rung 0). Mirror is the sanctioned GPU→CPU readback; StepSystem's step() reads the snapshot each fixed
// tick and bounds the primal's color count + dispatch size. MirrorSystem (draw, last) flushes it after
// PackSystem wrote word 1, so the snapshot step() reads is from a prior frame — exactly the frame-stale
// input both bounds want. Allocated in warm (after Mirror.reset), released in dispose.
let colorMirror: Mirror | null = null;

// the create-stamp each Body eid was last packed at, the state diffStamps diffs against to catch a
// same-update realias the GPU pack's non-member seed reset misses (recycle.ts). Reset in warm/dispose.
const stamps = new Map<number, number>();

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
        throw new Error("[avbd] pack sources missing — declare SlabPlugin as a dependency");
    }
    return { membership, pos, quat, half, mass, friction, shape };
}

// the GPU firehose pack: scan capacity gated on Body membership → the dense→eid map + the one-time seed
// of any newly-spawned body. Draw group, so it runs after SlabSystem + MembershipSystem (both `first`
// in draw) have flushed the Body slabs + the membership mirror — the pack reads fresh GPU data. Submits
// its own encoder (works headless, no renderer needed); its writes are visible to the next fixed step +
// to `compose` (below, later submits). No CPU per-entity iteration — the membership scan + seed are GPU.
const PackSystem: System = {
    name: "pack",
    group: "draw",
    annotations: { mode: "always" },
    update(state: State) {
        const step = Avbd.step;
        if (!step || !Compute.device) return;
        // the CPU pre-pack stamp diff: force a re-seed on every eid recycled to a new body since the last
        // pack (a same-update destroy+create the GPU pack's non-member seed reset can't see, diffStamps).
        for (const eid of diffStamps(state.query([Body]), (e) => state.stamp(e), stamps)) {
            step.reseed(eid);
        }
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

// the bodies SoA columns `readBody` reads off the Mirror snapshot (must match step.ts B_POS / B_QUAT / B_VELL)
const B_POS = 0;
const B_QUAT = 1;
const B_VELL = 6;

// the ONE body-pose Mirror `readBody` reads through, shared by every CPU consumer (the character sweep,
// pick.ts). Lazily allocated once `Avbd.step` exists (after warm + Mirror.reset).
let bodyMirror: Mirror | null = null;
// a cached view over the Mirror's reused snapshot buffer (physics.md: `snapshot.bytes` is the SAME
// ArrayBuffer object across readbacks), so `readBody` doesn't re-wrap a Float32Array every call.
let cachedBuf: ArrayBuffer | null = null;
let cachedView: Float32Array | null = null;

// this backend's `PhysicsBackend` implementation — installed at `Physics.backend` in `warm()`, uninstalled
// in `dispose()`. Every method reads `Avbd.step` fresh (never captured), so it stays correct across a
// State rebuild that tears down + recreates the step.
const backendHandle: PhysicsBackend = {
    step() {
        const s = Avbd.step;
        if (!s || !Compute.device) return;
        // readback-bounded color loop (Phase 4.9 Lever 1) + direct color-loop dispatch (rung 0): bound the
        // primal's dispatched color count to the frame-stale used-color count and size the color loop's
        // direct dispatch off the frame-stale live count, both riding one snapshot ([0] = usedColors from
        // colorize, [1] = liveCount from packScan). No snapshot yet (first frames) → both keep the full
        // cap (the safe cold-start).
        if (colorMirror?.snapshot) {
            const counts = new Uint32Array(colorMirror.snapshot.bytes);
            s.boundColors(counts[0]);
            s.boundBodies(counts[1]);
        }
        const encoder = Compute.device.createCommandEncoder({ label: "physics-step" });
        s.record(encoder);
        Compute.device.queue.submit([encoder.finish()]);
    },
    readBody(eid) {
        const s = Avbd.step;
        if (!s) return null;
        if (!bodyMirror) bodyMirror = mirror(s.bodies);
        const snap = bodyMirror.snapshot;
        if (!snap) return null;
        if (cachedBuf !== snap.bytes) {
            cachedBuf = snap.bytes;
            cachedView = new Float32Array(snap.bytes);
        }
        const f = cachedView as Float32Array;
        const cap = s.eidCap;
        const po = (B_POS * cap + eid) * 4;
        const qo = (B_QUAT * cap + eid) * 4;
        const vo = (B_VELL * cap + eid) * 4;
        return {
            pos: [f[po], f[po + 1], f[po + 2]],
            quat: [f[qo], f[qo + 1], f[qo + 2], f[qo + 3]],
            vel: [f[vo], f[vo + 1], f[vo + 2]],
        };
    },
    setKinematic(eid, pos, quat, teleport, vel) {
        Avbd.step?.setKinematic(eid, pos, quat, teleport, vel);
    },
    setVelocity(eid, vx, vy, vz) {
        Avbd.step?.setVelocity(eid, vx, vy, vz);
    },
    setSprings(springs) {
        Avbd.step?.setSprings(springs);
    },
    setJoints(joints) {
        Avbd.step?.setJoints(joints);
    },
    get gravity() {
        return Avbd.step?.gravity ?? 0;
    },
    get dt() {
        return Avbd.step?.dt ?? 0;
    },
    compose(encoder, transforms, alpha) {
        Avbd.step?.compose(encoder, transforms, alpha);
    },
};

/**
 * the AVBD rigid-body physics backend: installs `Body`/`Spring`/`Joint` and the GPU solver, and implements
 * the physics substrate's `PhysicsBackend` handle at `Physics.backend`. Opt-in — add it to a scene to run
 * physics (it's not in the default plugins). The specialized swap-in for GPU-resident scale; the default
 * for most game-scale scenes is `standard/tumble`'s `TumblePlugin`.
 *
 * @example
 * ```
 * export const config: Config = { plugins: [AvbdPlugin], scene: "scenes/scene.scene" };
 * ```
 */
export const AvbdPlugin: Plugin = {
    name: "Avbd",
    components: { Body, Spring, Joint },
    systems: [ConstraintSystem, StepSystem, PackSystem, ComposeSystem],
    // MirrorPlugin: the readback-bounded color loop reads `step.colorCount` through a Mirror (above), and
    // `readBody` reads `step.bodies` through one.
    dependencies: [SlabPlugin, MirrorPlugin],
    // the LBVH broadphase's bounds reduction + radix sort prefer subgroup ops, falling back to an
    // LDS arm where absent (WebKit) — preferred, not required, so a no-subgroup device still runs
    // physics (bvh/core; createBvh reads `device.features` to pick the arm).
    preferredFeatures: BVH_FEATURES,
    traits: {
        Body: bodyTraits,
        Spring: springTraits,
        Joint: jointTraits,
    },

    initialize() {
        Avbd.step = null;
    },

    async warm(state: State) {
        if (!Compute.device) return;
        lastHullCount = -1; // force a hull re-upload into the fresh step (the registry persists across states)
        stamps.clear(); // a fresh step re-seeds every body; the stamp diff arms against the new step's slots
        bodyMirror = null;
        cachedBuf = null;
        cachedView = null;
        // the membership gate templates the pack's per-eid skip test. `build` fixes every component's
        // bit up front, so `bit(Body)` is valid here; `capacity` is the eid range the pack walks.
        const { gen, mask } = state.membership.bit(Body);
        Avbd.step = await PhysicsStep.create(Compute.device, capacity, MAX_BODIES, {
            gen,
            mask,
        });
        // static per-step params — the live count is GPU-resident (the pack writes it), not a config field.
        Avbd.step.configure({
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
        colorMirror = mirror(Avbd.step.colorCount);
        installBackend(backendHandle);
    },

    dispose() {
        stamps.clear();
        colorMirror?.dispose();
        colorMirror = null;
        bodyMirror?.dispose();
        bodyMirror = null;
        cachedBuf = null;
        cachedView = null;
        Avbd.step?.destroy();
        Avbd.step = null;
        uninstallBackend();
    },
};
