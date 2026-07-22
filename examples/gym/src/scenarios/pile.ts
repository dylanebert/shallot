import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Color,
    Compute,
    DirectionalLight,
    GlazePlugin,
    InputPlugin,
    type Mirror,
    MirrorPlugin,
    mirror,
    Orbit,
    OrbitPlugin,
    Part,
    PartPlugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    ShapeKind,
    SlabPlugin,
    type State,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
// the production narrowphase WGSL (composed for the gates), reached at src — the rounded narrowphase gate
// runs it on the device and diffs against the f64 oracle `narrowphase`, like the rest reach tests/ below.
import {
    Avbd,
    BODY_VEC4,
    COLLIDE_WGSL,
    HULL_WGSL,
    type JointDef,
    LDS_CAP,
    LDS_N,
    MAX_CONTACTS,
    PENALTY_MIN,
    PhysicsStep,
    packHulls,
    SMALL_N,
} from "@dylanebert/shallot/avbd/core";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { Hulls } from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";
// the f64 oracle (tests/, out of the published src/) is the executable spec the GPU gates compare against,
// reached by relative path. accel.ts reaches the BVH oracle the same way.
import { collide, SPECULATIVE_DISTANCE } from "../../../../packages/shallot/tests/avbd/collide";
import { CORPUS } from "../../../../packages/shallot/tests/avbd/corpus";
import { boxHull } from "../../../../packages/shallot/tests/avbd/hull";
import { type Quat, sub, type Vec3 } from "../../../../packages/shallot/tests/avbd/math";
import {
    capsule,
    body as makeBody,
    type Body as OracleBody,
    sphere,
} from "../../../../packages/shallot/tests/avbd/rigid";
import { narrowphase } from "../../../../packages/shallot/tests/avbd/rounded";
import { makeSolver, step as oracleStep } from "../../../../packages/shallot/tests/avbd/solver";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// pile — contact-settling rigidbodies, the AVBD solver's canonical GPU-correctness home (physics.md "GPU
// correctness is the gym pile scenario"). The first of the three §6 scenarios (pile / constraints / character),
// it's ONE parametric scene of composable atoms: N bodies × `shape` (box/sphere/capsule/hull) × `layout`
// (grid / heap / pyramid) × `boundary` (flat ground / drum pen). The pile gathers into the eid-indexed
// solver, steps under the warmstart augmented-Lagrangian layer, and renders at the physics-owned pose via the
// firehose compose (a Body is a Part whose matrix physics scatters into `transforms` — BodyComposeSystem).
//
// The `boundary` atom is what makes a ROUNDED rest a real gate (the §6.6 lift): a sphere/capsule pile rolls
// off a flat ground (no-tunnel only), but a `drum` (a static wall pen) holds it so it settles — so the
// settled-pile band becomes a real correctness gate for the rounded narrowphase, not box-only.
//
// The gate set, by what each covers:
//   • live-band / firehose-pose / spawn-despawn — the live sim end to end (gather → step → readback →
//     firehose compose), read through Mirror. spawn-despawn is the only gate exercising the GPU pack/seed
//     path (a mid-sim membership change must not disturb the settled pile — the Phase-4.7 contract).
//   • free-fall (closed form) + single-step-exact + the six corpus topologies — the solve math: inject an
//     oracle-advanced state, step GPU + oracle one frame each, compare pose/velocity to a derived
//     single-step tolerance (no chaos accumulation, so it's tight). Memoryless, iters-independent.
//   • stack-warmstart / friction-slide / speculative-stop / swept-stop — GPU-only behaviors the per-step
//     compare can't isolate: multi-pair λ/k carry, kinetic-friction deceleration, the speculative band,
//     the velocity sweep. Each GPU == oracle on the real device.
// The seeded gates drive a fresh isolated PhysicsStep, so they never touch the live sim.
//
// Perf is reported, not gated: the `measured` reporter publishes the per-step GPU spans + dispatch count +
// memory + fall-through signals as structured `Check.data`; scripts/physics-bench.ts sweeps the scene and
// reads that payload. On WebGPU the binding cost is dispatch *encode*, not the GPU span (physics.md
// "Dispatch count is the binding cost").
//
// The `shape` + `joints` params give the OTHER narrowphase + the joint passes standing PERF coverage (a
// non-box pile runs the rounded/hull collide pipelines through `record()`; `joints` chains the rows so
// jointInit/jointDual/repair run). The box-only oracle suite (single-step / corpus GPU == oracle) seeds its
// OWN box bodies, so it is independent of the live pile's shape. What the live shape DOES gate is the
// settled-pile band: box (any boundary) + rounded-in-`drum` must SETTLE (the real rounded rest gate);
// rounded-on-`flat` rolls off, so it gets the no-tunnel band only. A rounded pile (sphere/capsule) ALSO
// carries the tight single-step rounded narrowphase gate (the production
// collideRounded/collideRoundedPolytope WGSL byte-exact vs the f64 oracle `narrowphase`); the rolling /
// conservation math stays the `rounded.oracle.ts` CPU tier. The hull single-step gate stays in `sat`;
// constraint + character behaviors are the sibling `constraints` / `character` gym scenarios.

const G = -10;
const DT = 1 / 60;
const ALPHA = 0.99;
const BETA_LIN = 1e4; // canonical penalty ramp — warmstart carries λ/k across frames
const BETA_ANG = 100; // joint angular penalty ramp (Phase 6.2; contacts ignore it)
const GAMMA = 0.999; // warmstart decay (Eq. 19)
const ITERS = 10; // the gym's robustness config — DECOUPLED from the shipped iters=6 (a perf tradeoff)
const MAX_COLORS = 8; // dispatched-color cap (matches AvbdPlugin)
const GROUND_HALF_Y = 0.5;

let liveMirror: Mirror | null = null;
let xformMirror: Mirror | null = null;
let countersMirror: Mirror | null = null;
let dynamicCount = 0; // dynamic bodies (mass > 0 — every body but the ground)
// the ground footprint (centered at the origin), so the band check tells a box that fell THROUGH the floor
// (below the surface, over the footprint) from one that slid OFF the edge (past it).
let groundExtent = 0;
let groundTop = 0;
// the live iters the build resolved from params — the HUD reads it, not the module ITERS.
let cfgIters = ITERS;
// the regime thresholds the build resolved from params — the regime-cross gate predicts against them.
let cfgSmallN = SMALL_N;
let cfgLdsN = LDS_N;
// sub-steps per fixed step the live pile uses (the dense-pile convergence lever) — the HUD reads it.
let cfgSubsteps = 1;
// the collider the live pile uses (resolved from the `shape` param at build). box runs the full correctness
// gate set; sphere/capsule (rounded narrowphase) + hull (convex-hull narrowphase, a unit box-hull) are PERF
// + no-tunnel cells — the tight rounded/hull rest gates are §6.6 (the gym `sat` scenario). A non-box
// pile gives the always-compiled rounded/hull collide pipelines standing per-step coverage through `record()`.
let pileShape: number = ShapeKind.Box;
// whether the pile rows are joint-chained (`joints` param) — pins each row into a spherical-joint chain so the
// joint passes (jointInit/jointDual/repair + the primal joint stamp + the constraint CSR) run in the perf sweep.
let pileJoints = false;
// the pile layout (grid / heap / pyramid) + boundary (flat ground / drum wall-pen), resolved from params at
// build. drum is the rounded-rest container: a sphere/capsule pile a flat ground can't hold settles inside it.
let pileLayout = "grid";
let pileBoundary = "flat";
// the registered box-hull id reused by every hull-shape body (registry is keyed by name, so this is stable).
let pileHullId = -1;
// shape names indexed by ShapeKind value (0 box … 3 hull) — the `shape` param resolves to an index via indexOf.
const SHAPE_NAME = ["box", "sphere", "capsule", "hull"];
// the body eids the scene created (ground + boxes), captured at build. Solver state is eid-indexed, so the
// checks read `bodies[col*cap + eid]` / `transforms[eid]` by eid.
let bodyEids: number[] = [];
// spawn-despawn: the original bodies' settled poses snapshotted before a mid-sim spawn. After spawn +
// settle they must hold — eid-persistence keeps them put (the old dense compact teleported the whole pile).
let preSpawnPose: Map<number, Vec3> | null = null;
let spawnedEid = -1;
// regime-cross: the step's regimes + live count captured just before the spawn. When a regime threshold
// (smallN / ldsN) sits between the pre- and post-spawn live counts, the spawn must flip the step off its
// specialized path and despawning back must restore it — the frame-stale gate's two-sided safety, exercised.
let preSpawnRegimes: { small: boolean; lds: boolean } | null = null;
let preSpawnLive = 0;
let liveState: State | null = null;

function rng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// a random orientation within `maxTilt` of upright (random axis, angle ∈ [0, maxTilt]) — the heap's noisy
// drop. Bounded (NOT uniform SO(3)): a box tilted ≤ ~50° lands on a face and reaches rest, where a
// uniform-random box lands on an edge/corner and rocks forever (marginal stability). The drop + collisions
// still tumble them into a chaotic resting pile; the bound only keeps it settleable.
// Returns [x, y, z, w].
function tiltQuat(rand: () => number, maxTilt: number): Quat {
    const u = rand() * 2 - 1; // cos(polar) — uniform axis on the unit sphere
    const phi = 2 * Math.PI * rand();
    const r = Math.sqrt(1 - u * u);
    const ax = r * Math.cos(phi);
    const ay = u;
    const az = r * Math.sin(phi);
    const angle = rand() * maxTilt;
    const s = Math.sin(angle / 2);
    return [ax * s, ay * s, az * s, Math.cos(angle / 2)];
}

// the authored pile centers + per-body orientations (as quaternions), by layout. grid = a dense axis-aligned
// lattice (face contacts from frame 0, identity orientation); heap = a chaotic drop (full random orientation
// + off-cell scatter + interleaved layer heights, so boxes tumble onto each other and wedge — not rotated
// columns); pyramid = a Box2D-Pyramid box stack (rows narrowing from a base, each upper box bricked over the
// seam of the two below so it rests on both — a stack that must hold at rest). Pure over `rand`, so the
// drum + ground fit the result.
const IDENTITY: Quat = [0, 0, 0, 1];
function layoutPositions(
    n: number,
    cols: number,
    spacing: number,
    layout: string,
    rand: () => number,
): { pts: Vec3[]; quats: Quat[] } {
    const pts: Vec3[] = [];
    const quats: Quat[] = [];
    if (layout === "pyramid") {
        const base = Math.max(1, Math.ceil((Math.sqrt(8 * n + 1) - 1) / 2));
        for (let r = 0, placed = 0; r < base && placed < n; r++) {
            const rowCount = base - r;
            const y = GROUND_HALF_Y + 0.5 + r; // unit boxes, stacked one apart, the row resting on the one below
            for (let c = 0; c < rowCount && placed < n; c++, placed++) {
                pts.push([(c - (rowCount - 1) / 2) * spacing, y, 0]); // centered → the brick offset is half a spacing
                quats.push(IDENTITY);
            }
        }
        return { pts, quats };
    }
    if (layout === "heap") {
        // a chaotic-but-settleable rubble drop: each box at a bounded random tilt (≤ MAX_TILT — noisy, yet
        // it lands on a face and reaches rest, unlike uniform SO(3) which rocks on edges forever), dropped
        // from a lattice cell lifted well above the ground with the layer heights jittered. The cells are
        // wider than a tilted box's footprint (no spawn overlap → no perpetual jostling), so the chaos is the
        // DROP — they fall, tilt, topple onto each other and wedge into an irregular pile that then settles.
        const maxTilt = (50 * Math.PI) / 180;
        for (let i = 0; i < n; i++) {
            const gx = i % cols;
            const gz = Math.floor(i / cols) % cols;
            const gy = Math.floor(i / (cols * cols));
            const x = (gx - (cols - 1) / 2) * spacing + (rand() - 0.5) * 0.1;
            const z = (gz - (cols - 1) / 2) * spacing + (rand() - 0.5) * 0.1;
            const y = GROUND_HALF_Y + 0.5 + 3.0 + gy * 1.5 + rand() * 1.0;
            pts.push([x, y, z]);
            quats.push(tiltQuat(rand, maxTilt));
        }
        return { pts, quats };
    }
    for (let i = 0; i < n; i++) {
        const gx = i % cols;
        const gz = Math.floor(i / cols) % cols;
        const gy = Math.floor(i / (cols * cols));
        const x = (gx - (cols - 1) / 2) * spacing + (rand() - 0.5) * 0.05;
        const z = (gz - (cols - 1) / 2) * spacing + (rand() - 0.5) * 0.05;
        const y = GROUND_HALF_Y + 0.5 + gy * 1.05 + gz * 0.02;
        pts.push([x, y, z]);
        quats.push(IDENTITY);
    }
    return { pts, quats };
}

// author the resolved collider onto a Body eid: the shape tag + the halfExtents geometry + the render mesh.
// Shared by the pile + the spawn-despawn body. box → the unit cube (shape/mesh defaults already correct, the
// untouched default path); sphere/capsule → a core + radius (the rounded narrowphase); hull → the unit box-hull
// (the convex-hull narrowphase, rendered as its AABB cube — the gym `sat` convention, since the bench gates on
// numbers not pixels). Call AFTER any Part add so the mesh slab exists. The seed derives the per-shape moment.
function applyShape(eid: number, viz: boolean): void {
    if (pileShape === ShapeKind.Sphere) {
        Body.shape.set(eid, ShapeKind.Sphere);
        Body.halfExtents.set(eid, 0, 0, 0, 0.5); // core point, radius rides .w
        if (viz) Part.mesh.set(eid, Meshes.id("sphere") ?? 0);
    } else if (pileShape === ShapeKind.Capsule) {
        Body.shape.set(eid, ShapeKind.Capsule);
        Body.halfExtents.set(eid, 0, 0.35, 0, 0.3); // core segment along Y + radius .w
        if (viz) Part.mesh.set(eid, Meshes.id("capsule") ?? 0);
    } else if (pileShape === ShapeKind.Hull) {
        Body.shape.set(eid, ShapeKind.Hull);
        Body.halfExtents.set(eid, 0.5, 0.5, 0.5, pileHullId); // AABB half .xyz, hull id .w
    } else {
        Body.halfExtents.set(eid, 0.5, 0.5, 0.5, 0); // box — shape + mesh defaults already correct
    }
}

const scenario: Scenario = {
    name: "pile",
    params: [
        { key: "count", type: "number", default: 48, min: 1, max: 65536, step: 8, rebuild: true },
        // the bench's sweep surface (scripts/physics-bench.ts). ground extent is independent of count:
        // 0 = auto (fit the pile), > 0 = an explicit static-ground half-extent (a large ground over a
        // tight pile is the broadphase-robustness case — the ground AABB must not degrade the LBVH).
        // layers/gap are the pile-density knobs. layout shapes the stack: grid (dense axis-aligned, face
        // contacts from frame 0), heap (chaotic full-random-orientation drop — boxes tumble + wedge), pyramid
        // (a Box2D-Pyramid box stack that must hold at rest). boundary contains it: flat (the single ground) or drum (a static wall pen
        // that holds a rounded pile so it settles — the rounded rest gate).
        { key: "ground", type: "number", default: 0, min: 0, max: 2000, rebuild: true },
        { key: "layers", type: "number", default: 4, min: 1, max: 256, rebuild: true },
        { key: "gap", type: "number", default: 1.1, min: 1, max: 4, step: 0.05, rebuild: true },
        {
            key: "layout",
            type: "select",
            default: "grid",
            options: ["grid", "heap", "pyramid"],
            rebuild: true,
        },
        {
            key: "boundary",
            type: "select",
            default: "flat",
            options: ["flat", "drum"],
            rebuild: true,
        },
        // the collider the dropped pile uses — box (default; the full correctness gate set runs ONLY here),
        // sphere/capsule (the rounded narrowphase), or hull (the convex-hull narrowphase, a unit box-hull). A
        // non-box pile is a PERF + no-tunnel cell; scripts/physics-bench.ts sweeps this so the rounded/hull
        // collide pipelines (which always compile, ~920 ms for hull) get standing per-step coverage.
        {
            key: "shape",
            type: "select",
            default: "box",
            options: ["box", "sphere", "capsule", "hull"],
            rebuild: true,
        },
        // pin each pile row into a chain of spherical joints — exercises the joint passes (jointInit/jointDual/
        // repair + the primal joint stamp + the constraint CSR) in the standing perf sweep. Forces an axis-aligned
        // grid (no yaw) so the authored anchors are exactly coincident (the construction guard passes).
        { key: "joints", type: "bool", default: false, rebuild: true },
        // solve knobs — reconfigured onto the live step after build. The seeded oracle gates build fresh
        // steps at ITERS regardless, so tuning these never moves the correctness gates.
        { key: "iters", type: "number", default: ITERS, min: 1, max: 40, rebuild: true },
        { key: "colors", type: "number", default: MAX_COLORS, min: 1, max: 32, rebuild: true },
        // the broadphase regime threshold (StepParams.smallN) — 0 forces the BVH path at every count,
        // the bench's A/B lever for the small-N crossover sweep
        { key: "smallN", type: "number", default: SMALL_N, min: 0, max: 65536, rebuild: true },
        // the LDS-resident solve threshold (StepParams.ldsN) — 0 forces the looped color passes,
        // the bench's A/B lever for the C1.2 solve regime
        { key: "ldsN", type: "number", default: LDS_N, min: 0, max: LDS_CAP, rebuild: true },
        // sub-steps per fixed step (StepParams.substeps) — the dense-pile convergence lever: h = dt/substeps
        // bounds per-step penetration so the penalty ramp clears it where raising iters saturates. 1 = the
        // single-sub-step path. The heap+drum repro goes green at substeps >= 2 (roadmap "dense-pile
        // contact convergence"); the seeded `substep-parity` gate pins GPU == oracle at substeps = 2.
        { key: "substeps", type: "number", default: 1, min: 1, max: 8, rebuild: true },
        { key: "seed", type: "number", default: 1, rebuild: true },
        { key: "viz", type: "bool", default: true, rebuild: true },
    ],

    async build(_canvas, p: Params) {
        const count = (p.count as number) ?? 48;
        const viz = (p.viz as boolean) ?? true;
        const rand = rng((p.seed as number) ?? 1);
        bodyEids = [];

        pileShape = Math.max(0, SHAPE_NAME.indexOf((p.shape as string) ?? "box"));
        pileJoints = (p.joints as boolean) ?? false;
        pileLayout = (p.layout as string) ?? "grid";
        pileBoundary = (p.boundary as string) ?? "flat";
        // joints chain adjacent-in-x bodies, so they need the axis-aligned grid — its anchors start exactly
        // coincident (the construction guard); a heap/pyramid would offset them and the guard would reject.
        if (pileJoints) pileLayout = "grid";
        // a hull body references a registered hull by id (halfExtents.w); reuse one unit box-hull for the whole
        // pile (the registry keys by name, so this is stable across rebuilds). box/sphere/capsule need no registry.
        if (pileShape === ShapeKind.Hull)
            pileHullId = Hulls.register({ name: "gym-pile-hull", ...boxHull([1, 1, 1]) });

        const layers = Math.max(1, Math.round((p.layers as number) ?? 4));
        const gap = (p.gap as number) ?? 1.1;
        // spacing by layout: a heap drops tilt-randomized boxes one cell apart (1.6 — wider than a ≤50°-
        // tilted box's ~1.4 footprint, so the spawn never overlaps and the solver isn't perpetually digging
        // boxes apart; the drop + topple supplies the chaos); a pyramid stacks unit boxes shoulder-to-
        // shoulder with a hair of slack (1.02); a grid keeps the authored gap.
        const spacing =
            pileLayout === "heap" ? Math.max(gap, 1.6) : pileLayout === "pyramid" ? 1.02 : gap;
        const cols = Math.max(1, Math.ceil(Math.sqrt(count / layers)));
        // the authored centers + orientations (pure), computed up front so the drum + auto-ground fit the extent.
        const { pts: pilePos, quats } = layoutPositions(count, cols, spacing, pileLayout, rand);
        let extent = 0;
        for (const pt of pilePos) extent = Math.max(extent, Math.abs(pt[0]), Math.abs(pt[2]));
        // ground extent: 0 = fit the pile (its extent + a margin); > 0 = an explicit half-extent. A drum pen
        // sits a touch outside the pile so a rounded pile is contained + settles; a flat ground stays generous.
        const drum = pileBoundary === "drum";
        const drumHalf = extent + 1.5; // pile edge → wall inner face: a body half + slack to settle into
        const groundParam = (p.ground as number) ?? 0;
        const groundHalf =
            groundParam > 0 ? groundParam : drum ? drumHalf + 1 : Math.max(20, extent + 2);
        groundExtent = groundHalf;
        groundTop = GROUND_HALF_Y;
        // capacity sizes the eid-indexed solver state (+ the `bodies` Mirror); scale to the live count so a
        // small scene stays lean (count=48 → 1024) and a stress pile gets the eid space it needs. +16 pads the
        // ground + the 4 drum walls + the spawn-despawn body.
        const cap = Math.max(1024, count + 16);

        // AvbdPlugin before PartPlugin: BodyComposeSystem and the Part pack both run after BeginFrameSystem;
        // the compose declares before:[PrepassSystem] but the pack has no edge to it, so registration order
        // breaks the tie — physics first puts the firehose write ahead of the cull read (render.md ordering).
        const { state, dispose } = await run({
            defaults: false,
            capacity: cap,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                AvbdPlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        // the iters/colors perform levers, onto the live solve. The gym tests at the ROBUSTNESS iters
        // (ITERS=10), DECOUPLED from the plugin's shipped iters=6 (a perf tradeoff): reconfigure the live
        // step to the gym's cfg so the live pile + the HUD stay self-consistent regardless of the shipped value.
        cfgIters = Math.max(1, Math.round((p.iters as number) ?? ITERS));
        const cfgColors = Math.max(1, Math.round((p.colors as number) ?? MAX_COLORS));
        cfgSmallN = Math.max(0, Math.round((p.smallN as number) ?? SMALL_N));
        cfgLdsN = Math.max(0, Math.round((p.ldsN as number) ?? LDS_N));
        // sub-stepping is the dense-pile convergence lever, SCENE-configured (roadmap decision: not an
        // always-on engine default — it 2-4×'s the dispatch-count budget, physics.md "Dispatch count"). A
        // chaotic `heap` is the trigger: it ships count-aware substeps (≈2 ≤ 128 bodies, 4 above — the
        // real-GPU `no-overlap` thresholds) so the heap's no-overlap gate holds at the ship config; clean
        // grid/pyramid piles converge at substeps=1. An explicit `--param substeps` overrides.
        const ssParam = Math.max(1, Math.round((p.substeps as number) ?? 1));
        cfgSubsteps = ssParam > 1 ? ssParam : pileLayout === "heap" ? (count > 192 ? 4 : 2) : 1;
        Avbd.step?.configure({
            dt: DT,
            gravity: G,
            alpha: ALPHA,
            penalty: PENALTY_MIN,
            betaLin: BETA_LIN,
            betaAng: BETA_ANG,
            gamma: GAMMA,
            iterations: cfgIters,
            maxColors: cfgColors,
            smallN: cfgSmallN,
            ldsN: cfgLdsN,
            substeps: cfgSubsteps,
        });

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);

        // ground first so it takes the lowest dense index — broadphase orients each pair A = higher index,
        // so a dynamic box (higher eid) is always body A against the static ground (physics.md).
        const ground = state.create();
        bodyEids.push(ground);
        state.add(ground, Body);
        Body.pos.set(ground, 0, 0, 0, 0);
        Body.halfExtents.set(ground, groundHalf, GROUND_HALF_Y, groundHalf, 0);
        Body.mass.set(ground, 0); // static
        if (viz) {
            state.add(ground, Part);
            state.add(ground, Color);
            Color.rgba.set(ground, 0.18, 0.2, 0.24, 1);
        }

        // drum boundary: 4 INVISIBLE static walls pen the pile in, so a chaotic / rounded pile a flat ground
        // can't hold settles into a contained heap (its stragglers caught by the walls, not lost off the edge
        // — the contained-settle gate). Statics like the ground, created BEFORE the dynamics so a dynamic body
        // is always the higher-eid body A against a wall (physics.md pair orientation). Not in bodyEids — the
        // gates check the pile; the walls only contain it. Collision-only (NO Part) — the default surface is
        // opaque (transparency is a surface `blend` property, not a per-instance Color alpha), so an invisible
        // wall IS a Part-less Body, the idiomatic invisible collider; only the ground + the pile render.
        if (drum) {
            const wallHalfY = GROUND_HALF_Y + Math.max(3, layers + 2); // tall enough to hold the pile in
            const wt = 0.5; // wall thickness (half)
            const span = drumHalf + wt; // half-length along the wall so the corners close
            const wall = (x: number, z: number, hx: number, hz: number): void => {
                const w = state.create();
                state.add(w, Body);
                Body.pos.set(w, x, GROUND_HALF_Y + wallHalfY, z, 0); // base flush with the ground top
                Body.halfExtents.set(w, hx, wallHalfY, hz, 0);
                Body.mass.set(w, 0);
            };
            wall(drumHalf + wt, 0, wt, span);
            wall(-drumHalf - wt, 0, wt, span);
            wall(0, drumHalf + wt, span, wt);
            wall(0, -drumHalf - wt, span, wt);
        }

        // place the dynamic bodies at the precomputed centers + orientations. A heap carries a full random
        // orientation so it tumbles + wedges into a chaotic pile; grid/pyramid stay axis-aligned. A wide-slab
        // pile settles in a bounded number of frames; a tall stack takes far longer to resolve.
        dynamicCount = count;
        for (let i = 0; i < count; i++) {
            const [x, y, z] = pilePos[i];
            const body = state.create();
            bodyEids.push(body);
            state.add(body, Body);
            Body.pos.set(body, x, y, z, 0);
            const q = quats[i];
            Body.quat.set(body, q[0], q[1], q[2], q[3]);
            Body.mass.set(body, 1);
            if (viz) {
                state.add(body, Part);
                state.add(body, Color);
                Color.rgba.set(body, 0.3 + rand() * 0.5, 0.4 + rand() * 0.5, 0.6 + rand() * 0.4, 1);
            }
            applyShape(body, viz); // shape tag + halfExtents geometry + per-shape mesh (after Part for the mesh)
        }

        // joint-chain each pile row's adjacent-in-x bodies into a spherical chain — the joint perf cell. The
        // anchor is the exact midpoint of the two authored centers (rA = (pB−pA)/2, rB = −rA), so the anchors
        // start coincident (length(pA−pB) = 0) and the construction guard passes even for a sphere (halfExtents
        // .xyz = 0 → a tiny reach). A row boundary (i % cols == 0) joins non-adjacent bodies, so skip it.
        if (pileJoints) {
            const joints: JointDef[] = [];
            for (let i = 1; i < count; i++) {
                if (i % cols === 0) continue;
                const pa = pilePos[i - 1];
                const pb = pilePos[i];
                const rA: Vec3 = [(pb[0] - pa[0]) / 2, (pb[1] - pa[1]) / 2, (pb[2] - pa[2]) / 2];
                joints.push({
                    a: bodyEids[1 + i - 1],
                    b: bodyEids[1 + i],
                    rA,
                    rB: [-rA[0], -rA[1], -rA[2]],
                }); // spherical (default) — the linear pin, rotation free
            }
            Avbd.step?.setJoints(joints);
        }

        const cam = state.create();
        // Transform is required: OrbitSystem queries [Orbit, OrbitSmooth, Transform] and drives the camera
        // pose through Transform; without it the camera never gets a view and nothing renders.
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 7);
        Orbit.distance.set(cam, Math.max(12, cols * 3 + 8));
        Orbit.minDistance.set(cam, 3);
        Orbit.maxDistance.set(cam, Math.max(200, cols * 4));

        // let warm complete + a couple steps run, then mirror the eid-indexed body state (the CPU pose-read
        // path), the transform firehose (the bodied-entity compose output), and the overflow counters.
        await frames(3);
        if (Avbd.step) {
            liveMirror = mirror(Avbd.step.bodies);
            countersMirror = mirror(Avbd.step.counters);
        }
        const xforms = Compute.buffers.get("transforms");
        if (xforms) xformMirror = mirror(xforms);

        // settle the pile, then exercise spawn-despawn: snapshot the settled bodies, spawn a new body in a
        // ground corner (away from the pile), settle again. eid-indexed state persists, so the existing
        // bodies must stay put. Settle frames scale with count but stay bounded so a stress run still builds
        // within the harness ready-window; at the bound a huge pile isn't fully settled (the settled-pile
        // asserts are calibrated for the default count — the per-step spans stay valid regardless). A chaotic
        // `heap` gets extra frames to resolve its tumble + settle below the no-overlap bound (sub-stepping
        // converges the penetration fast, so this is enough for no-overlap + no-tunnel).
        // 2× the per-step cost at substeps ≥ 2 keeps this within the harness ready window only if bounded.
        const chaosSettle = pileLayout === "heap" ? 600 : 0;
        await frames(Math.min(240 + count * 4, 3000) + chaosSettle);
        preSpawnPose = await capturePose();
        preSpawnRegimes = Avbd.step ? Avbd.step.regimes : null;
        preSpawnLive = bodyEids.length;
        liveState = state;
        const extra = state.create();
        bodyEids.push(extra); // not in preSpawnPose, so spawn-despawn checks only the originals
        dynamicCount += 1;
        state.add(extra, Body);
        Body.pos.set(extra, groundHalf - 5, GROUND_HALF_Y + 0.5, groundHalf - 5, 0);
        Body.mass.set(extra, 1);
        if (viz) {
            state.add(extra, Part);
            state.add(extra, Color);
            Color.rgba.set(extra, 0.95, 0.55, 0.2, 1);
        }
        applyShape(extra, viz); // the spawned body matches the pile collider (exercises the seed for that shape)
        spawnedEid = extra;
        await frames(Math.min(120 + count * 2, 1500));

        return {
            state,
            dispose() {
                liveMirror?.dispose();
                xformMirror?.dispose();
                countersMirror?.dispose();
                liveMirror = null;
                xformMirror = null;
                countersMirror = null;
                preSpawnPose = null;
                preSpawnRegimes = null;
                liveState = null;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        // box + no joints is the full correctness surface (the 17 gates). A non-box or jointed pile is a PERF +
        // no-tunnel cell: the live band relaxes its settle requirement (a rolling sphere settles slowly; the live
        // rounded rest is the drum boundary, §6.6) and the box-only oracle gates (single-step/corpus, which seed
        // their own box bodies) + the strict spawn-drift are skipped. firehose-pose + measured are shape-agnostic.
        // A rounded pile (sphere/capsule) instead carries the rounded narrowphase gate — the production
        // collideRounded/collideRoundedPolytope WGSL diffed byte-exact against the f64 oracle `narrowphase`.
        const boxStd = pileShape === ShapeKind.Box && !pileJoints;
        const rounded = pileShape === ShapeKind.Sphere || pileShape === ShapeKind.Capsule;
        checks.push(await liveBand());
        checks.push(await firehosePose());
        if (boxStd) {
            checks.push(await noOverlap());
            checks.push(await spawnDespawn());
            const cross = await regimeCross();
            if (cross) checks.push(cross);
            checks.push(...(await seededGates()));
            checks.push(...(await corpusGates()));
        }
        if (rounded) checks.push(...(await roundedGates()));
        checks.push(await measured());
        return checks;
    },

    // short fixed-shape lines so the pinned panel never reflows; the profiler holds the fixed-group spans
    // steady across the fixed/draw cadence, so these read stable (not oscillating to 0).
    live(): string {
        return [
            `pile — AVBD warmstart (${SHAPE_NAME[pileShape]}, ${pileLayout}${pileBoundary === "drum" ? ", drum" : ""}${pileJoints ? ", joints" : ""})`,
            `bodies   ${bodyEids.length} (${dynamicCount} dyn)`,
            `colors   ${Avbd.step?.dispatchedColors ?? 0} / ${cfgIters} it${cfgSubsteps > 1 ? ` × ${cfgSubsteps} ss` : ""}`,
            `primal   ${span("phys:primal")} ms`,
            `dual     ${span("phys:dual")} ms`,
            `collide  ${span("phys:collide")} ms`,
        ].join("\n");
    },
};

// ── atoms ──────────────────────────────────────────────────────────────────

const dist = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const gpuPos = (s: Float32Array, i: number, cap: number): Vec3 => {
    const o = (0 * cap + i) * 4; // posLin (col 0)
    return [s[o], s[o + 1], s[o + 2]];
};
const gpuVel = (s: Float32Array, i: number, cap: number): Vec3 => {
    const o = (6 * cap + i) * 4; // velLin (col 6)
    return [s[o], s[o + 1], s[o + 2]];
};
const span = (name: string): string => {
    const ms = Profile.gpu.get(name);
    return ms === undefined ? "…" : ms.toFixed(3);
};

interface LiveStats {
    dyn: number;
    nonFinite: number;
    throughFloor: number; // below the surface, over the footprint — a real fall-through
    offEdge: number; // below the surface, past the footprint — slid off
    minY: number;
    maxY: number;
    maxSpeed: number;
}

// read the live pile back through Mirror, by eid. The shared read both the live-band gate and the measured
// reporter use — the eid-indexed SoA `s[(col*cap + eid)*4 + lane]`, cap = total floats / (cols*4).
async function liveStats(): Promise<LiveStats | null> {
    if (!liveMirror) return null;
    await settle(liveMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return null;
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    const at = (col: number, eid: number): number => (col * cap + eid) * 4;
    const floorY = groundTop - 0.1; // a box center below this has sunk through the surface (it rests at +0.5)
    const out: LiveStats = {
        dyn: 0,
        nonFinite: 0,
        throughFloor: 0,
        offEdge: 0,
        minY: Infinity,
        maxY: -Infinity,
        maxSpeed: 0,
    };
    for (const eid of bodyEids) {
        if (s[at(9, eid) + 3] <= 0) continue; // mass.w (col 9) <= 0 — static (the ground)
        out.dyn++;
        const px = s[at(0, eid)];
        const py = s[at(0, eid) + 1];
        const pz = s[at(0, eid) + 2];
        const vx = s[at(6, eid)];
        const vy = s[at(6, eid) + 1];
        const vz = s[at(6, eid) + 2];
        if (![px, py, pz, vx, vy, vz].every(Number.isFinite)) {
            out.nonFinite++;
            continue;
        }
        out.minY = Math.min(out.minY, py);
        out.maxY = Math.max(out.maxY, py);
        out.maxSpeed = Math.max(out.maxSpeed, Math.hypot(vx, vy, vz));
        if (py < floorY) {
            // over the footprint (centered at the origin) = a real fall-through; past it = slid off the edge
            const over = Math.abs(px) <= groundExtent && Math.abs(pz) <= groundExtent;
            if (over) out.throughFloor++;
            else out.offEdge++;
        }
    }
    return out;
}

// snapshot the dynamic bodies' positions by eid (the eid-indexed pose read), for spawn-despawn.
async function capturePose(): Promise<Map<number, Vec3>> {
    const pose = new Map<number, Vec3>();
    if (!liveMirror) return pose;
    await settle(liveMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return pose;
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    for (const eid of bodyEids) pose.set(eid, gpuPos(s, eid, cap));
    return pose;
}

interface StepResult {
    state: Float32Array;
    contacts: number; // counters[0] — total active contacts
    merged: number; // counters[6] — warmstarted contacts (exact-persistence metric)
    colors: Uint32Array; // the dense coloring the GPU solved with — fed to the oracle's colored schedule
}
// `mid` splits the run into two submits and runs its hook between them — a queue-ordered CPU write
// (setVelocity) lands before the remaining steps.
type RunGPU = (
    bodies: OracleBody[],
    steps: number,
    colorCap?: number,
    mid?: { at: number; run(): void },
) => Promise<StepResult>;

// dense readback layout for a `cap`-sized gate run: [bodies SoA | 8 counters (32 B) | dense colors].
const readBytes = (cap: number): number => cap * BODY_VEC4 * 16 + 32 + cap * 4;

// the dense-index seed colorize starts from (identity, so dense slot d keeps color d on conflict-free fixtures).
const bootstrap = (n: number): Uint32Array<ArrayBuffer> => {
    const a = new Uint32Array(n);
    for (let i = 0; i < n; i++) a[i] = i;
    return a;
};

// write one oracle body into the dense bodies SoA at dense index i: `arr[(col*cap + i)*4 + lane]`.
function seedBody(arr: Float32Array, i: number, b: OracleBody, cap: number): void {
    const w = (col: number, x: number, y: number, z: number, ww: number): void => {
        const o = (col * cap + i) * 4;
        arr[o] = x;
        arr[o + 1] = y;
        arr[o + 2] = z;
        arr[o + 3] = ww;
    };
    w(0, b.posLin[0], b.posLin[1], b.posLin[2], 0); // posLin
    w(1, b.posAng[0], b.posAng[1], b.posAng[2], b.posAng[3]); // posAng (quat)
    w(2, b.posLin[0], b.posLin[1], b.posLin[2], 0); // inertialLin (overwritten by the inertial pass)
    w(3, b.posAng[0], b.posAng[1], b.posAng[2], b.posAng[3]); // inertialAng
    w(4, b.posLin[0], b.posLin[1], b.posLin[2], 0); // initialLin
    w(5, b.posAng[0], b.posAng[1], b.posAng[2], b.posAng[3]); // initialAng
    w(6, b.velLin[0], b.velLin[1], b.velLin[2], 0); // velLin
    w(7, b.velAng[0], b.velAng[1], b.velAng[2], 0); // velAng
    w(8, b.prevVelLin[0], b.prevVelLin[1], b.prevVelLin[2], 0); // prevVelLin
    w(9, b.moment[0], b.moment[1], b.moment[2], b.mass); // moment.xyz / mass.w
    w(10, b.size[0] / 2, b.size[1] / 2, b.size[2] / 2, b.friction); // halfExtents.xyz / friction.w
    w(11, 0, 0, 0, 0); // B_ROUND: shape/radius/hullId 0 for these box gates (.w lane unused)
}

// the shared GPU-step engine for the seeded + corpus gates. Seeds `bodies` + an identity dense→eid map
// directly (no GPU pack), runs `steps` GPU steps in one encoder, reads back the dense state + the last
// step's counters + the coloring the GPU solved with. `cold()` clears the prior run's persistent store.
async function runStep(
    phys: PhysicsStep,
    read: GPUBuffer,
    cap: number,
    bodies: OracleBody[],
    steps: number,
    colorCap: number,
    mid?: { at: number; run(): void },
    smallN = SMALL_N,
    ldsN = LDS_N,
    substeps = 1,
): Promise<StepResult> {
    const device = Compute.device;
    const Full = cap * BODY_VEC4 * 16;
    const Counters = Full + 32;
    const End = readBytes(cap);
    const arr = new Float32Array(cap * BODY_VEC4 * 4);
    for (let i = 0; i < bodies.length; i++) seedBody(arr, i, bodies[i], cap);
    device.queue.writeBuffer(phys.bodies, 0, arr);
    device.queue.writeBuffer(phys.colors, 0, bootstrap(cap));
    phys.configure({
        dt: DT,
        gravity: G,
        alpha: ALPHA,
        penalty: PENALTY_MIN,
        betaLin: BETA_LIN,
        gamma: GAMMA,
        iterations: ITERS,
        maxColors: colorCap,
        smallN,
        ldsN,
        substeps,
    });
    phys.gateSetCount(bodies.length);
    phys.cold();
    let enc = device.createCommandEncoder();
    for (let s = 0; s < steps; s++) {
        if (mid && s === mid.at) {
            device.queue.submit([enc.finish()]);
            mid.run();
            enc = device.createCommandEncoder();
        }
        phys.record(enc);
    }
    enc.copyBufferToBuffer(phys.bodies, 0, read, 0, Full);
    enc.copyBufferToBuffer(phys.counters, 0, read, Full, 32);
    enc.copyBufferToBuffer(phys.colors, 0, read, Counters, cap * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ, 0, End);
    const mapped = read.getMappedRange(0, End);
    const state = new Float32Array(mapped.slice(0, Full));
    const counters = new Uint32Array(mapped.slice(Full, Counters));
    const colors = new Uint32Array(mapped.slice(Counters, End));
    read.unmap();
    return { state, contacts: counters[0], merged: counters[6], colors };
}

const clone = (b: OracleBody): OracleBody =>
    makeBody(
        [...b.size] as Vec3,
        b.mass,
        b.friction,
        [...b.posLin] as Vec3,
        [...b.velLin] as Vec3,
        [...b.posAng] as Quat,
    );

// step the f64 oracle `steps` frames. With a coloring supplied (the GPU's actual one, for a single-step
// compare) the oracle runs that colored schedule; without it, the dense-index seed (advancing a scene).
function oracle(
    bodies: OracleBody[],
    steps: number,
    colors?: number[],
    substeps = 1,
): OracleBody[] {
    const clones = bodies.map((b) => clone(b));
    const s = makeSolver(clones, {
        layer: "warmstart",
        penaltyStiffness: PENALTY_MIN,
        betaLin: BETA_LIN,
        gamma: GAMMA,
        iterations: ITERS,
        alpha: ALPHA,
        dt: DT,
        gravity: G,
        substeps,
    });
    const sched = { kind: "colored" as const, colors: colors ?? [...bootstrap(clones.length)] };
    for (let i = 0; i < steps; i++) oracleStep(s, sched);
    return clones;
}

// the static ground's 0xffffffff (uncolored) remapped to 0 so the oracle's `0..maxColor` loop stays bounded
// (static is skipped in the primal). Feeds the GPU's real coloring to the oracle's colored schedule.
const oracleColors = (raw: Uint32Array, n: number): number[] =>
    Array.from({ length: n }, (_, i) => (raw[i] === 0xffffffff ? 0 : raw[i]));

function oracleContacts(bodies: OracleBody[]): number {
    let total = 0;
    for (let ia = bodies.length - 1; ia >= 1; ia--) {
        for (let ib = ia - 1; ib >= 0; ib--) {
            const A = bodies[ia];
            const B = bodies[ib];
            if (A.mass <= 0 && B.mass <= 0) continue;
            const dp = sub(A.posLin, B.posLin);
            // the speculative-band broadphase pad (Phase 4.8.3) — matches the GPU sphere test, so the
            // expected count includes a speculative pair (gap within the band) the GPU also keeps.
            const r = A.radius + B.radius + SPECULATIVE_DISTANCE;
            if (dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2] > r * r) continue;
            total += collide(
                { pos: A.posLin, quat: A.posAng, size: A.size },
                { pos: B.posLin, quat: B.posAng, size: B.size },
            ).contacts.length;
        }
    }
    return total;
}

// single-step-exact: advance the oracle k frames into contact, snapshot, then step a fresh GPU + a fresh
// oracle ONE frame each from that identical state and compare. A fresh solver cold-starts (pairContacts
// cleared, oracle manifolds empty), so the one step is memoryless — no chaos accumulation → tight.
// f32 (GPU) vs f64 (oracle), one frame of ITERS sweeps: dx ~1e-5 abs, velocity = dx/dt ×60 → ~1e-3.
const POS_TOL = 2e-4;
const VEL_TOL = 2e-2;

async function compareSingleStep(
    name: string,
    scene: () => OracleBody[],
    ks: number[],
    cap: number,
    run: RunGPU,
): Promise<Check> {
    let maxPos = 0;
    let maxVel = 0;
    let countMatch = true;
    for (const k of ks) {
        const snapshot = oracle(scene(), k).map(clone);
        const { state, contacts, colors } = await run(snapshot, 1);
        const ora = oracle(snapshot, 1, oracleColors(colors, snapshot.length));
        if (contacts !== oracleContacts(snapshot)) countMatch = false;
        for (let i = 0; i < snapshot.length; i++) {
            maxPos = Math.max(maxPos, dist(gpuPos(state, i, cap), ora[i].posLin));
            maxVel = Math.max(maxVel, dist(gpuVel(state, i, cap), ora[i].velLin));
        }
    }
    return {
        name,
        pass: countMatch && maxPos < POS_TOL && maxVel < VEL_TOL,
        detail: `pos err ${maxPos.toExponential(2)}, vel err ${maxVel.toExponential(2)}, contacts ${
            countMatch ? "match" : "MISMATCH"
        }`,
    };
}

// substep-parity (the sub-stepping gate, roadmap "dense-pile contact convergence"): one fixed step at
// substeps>1 must reproduce the oracle's substeps loop — `record()` runs `substeps` complete sub-steps of
// h=dt/substeps against the persistent warmstart store, matching tests/avbd/solver.ts `subStep` × substeps.
// Same memoryless single-step compare as compareSingleStep, but BOTH sides run the same `substeps`. Pose/vel
// only (no contact-count compare — the GPU `counters` accumulate across the sub-steps' collide passes, while
// the oracle counts the start pose once). Tolerance scales with the sub-step count: each sub-step is an
// independent f32 ITERS sweep, so the f32 error accumulates ~linearly in substeps.
async function substepParity(
    phys: PhysicsStep,
    read: GPUBuffer,
    cap: number,
    scene: () => OracleBody[],
    substeps: number,
): Promise<Check> {
    let maxPos = 0;
    let maxVel = 0;
    for (const k of [0, 3, 8]) {
        const snapshot = oracle(scene(), k).map(clone);
        const { state, colors } = await runStep(
            phys,
            read,
            cap,
            snapshot,
            1,
            MAX_COLORS,
            undefined,
            SMALL_N,
            LDS_N,
            substeps,
        );
        // the box stack's contact graph is stable across sub-steps, so the GPU re-colors identically each
        // sub-step — the last sub-step's coloring (read back) is every sub-step's, so the oracle's single
        // colored schedule matches the GPU's per-sub-step coloring.
        const ora = oracle(snapshot, 1, oracleColors(colors, snapshot.length), substeps);
        for (let i = 0; i < snapshot.length; i++) {
            maxPos = Math.max(maxPos, dist(gpuPos(state, i, cap), ora[i].posLin));
            maxVel = Math.max(maxVel, dist(gpuVel(state, i, cap), ora[i].velLin));
        }
    }
    return {
        name: `substep-parity (substeps=${substeps}, GPU == oracle)`,
        pass: maxPos < POS_TOL * substeps && maxVel < VEL_TOL * substeps,
        detail: `pos err ${maxPos.toExponential(2)}, vel err ${maxVel.toExponential(2)} (tol ${(POS_TOL * substeps).toExponential(1)} / ${(VEL_TOL * substeps).toExponential(1)})`,
    };
}

// ── live-sim gates ───────────────────────────────────────────────────────────

// the running pile read back through Mirror: every dynamic body finite, above the ground (no tunnel),
// settled (bounded speed), bounded. Each failure mode is named distinctly so a red names its cause. Chaotic
// long-horizon scenes gate only on this band — two float impls can't bit-match a settling pile.
async function liveBand(): Promise<Check> {
    const s = await liveStats();
    if (!s) return { name: "live-band", pass: false, detail: "no physics step" };
    // a settled-pile band needs a pile that ACTUALLY settles: a box does on any boundary; a rounded pile does
    // ONCE a drum contains it (on a flat ground it rolls — no-tunnel only). The §6.6 lift is exactly this:
    // the rounded-in-drum rest becomes a real settle gate, not box-only. A jointed chain swings (no settle).
    // The no-tunnel invariants (finite, no through-floor / off-edge, bounded) hold at every shape.
    const rounded = pileShape === ShapeKind.Sphere || pileShape === ShapeKind.Capsule;
    const settles =
        !pileJoints && (pileShape === ShapeKind.Box || (rounded && pileBoundary === "drum"));
    const boxStd = pileShape === ShapeKind.Box && !pileJoints;
    const ceil = GROUND_HALF_Y + dynamicCount + (boxStd ? 2 : 4);
    // 0.6 m/s is the box band; a drum-contained rounded pile keeps a little residual roll, so its settled band
    // is a touch looser (observed on lovelace) yet well below a rolling/exploding speed. maxSpeed is fastest-of-N.
    const settled = settles ? s.maxSpeed < (rounded ? 1.0 : 0.6) : true;
    const bounded = s.maxY < ceil;
    const pass =
        s.dyn > 0 &&
        s.nonFinite === 0 &&
        s.throughFloor === 0 &&
        s.offEdge === 0 &&
        settled &&
        bounded;
    const modes = [
        s.nonFinite > 0 ? `${s.nonFinite} NON-FINITE` : "",
        s.throughFloor > 0 ? `${s.throughFloor} through-floor` : "",
        s.offEdge > 0 ? `${s.offEdge} off-edge` : "",
        settled ? "" : "not-settled",
        bounded ? "" : "unbounded",
    ].filter(Boolean);
    return {
        name: settles
            ? `live-band (settled ${SHAPE_NAME[pileShape]}${pileBoundary === "drum" ? " drum" : ""})`
            : `live-band (no tunnel, ${SHAPE_NAME[pileShape]})`,
        pass,
        detail: `${s.dyn} dynamic, y∈[${s.minY.toFixed(2)}, ${s.maxY.toFixed(2)}], maxSpeed ${s.maxSpeed.toFixed(3)} m/s${
            modes.length ? ` — ${modes.join(", ")}` : ""
        }`,
    };
}

// no-overlap (the invalid-config detector): a SETTLED pile must not have bodies severely interpenetrating.
// AVBD rests a box ~COLLISION_MARGIN (0.01 m) into contact; penetration ≫ that is under-convergence — the
// solver left an overlap unresolved. Runs the f64 oracle SAT (the SAME narrowphase the solver matches —
// exact OBB penetration, NOT an AABB proxy that would false-positive on a tilted-but-touching box) over the
// settled dynamic box pairs, center-distance pruned, and asserts the deepest penetration is within a bound.
// Box-only (the oracle SAT is box-box); a settled pile with a box buried in another reds here. O(n²) prune,
// so capped at a moderate count.
const OVERLAP_BOUND = 0.1; // ~10× the contact margin — a converged pile rests well under it; ≫ it is a bug
async function noOverlap(): Promise<Check> {
    if (!liveMirror) return { name: "no-overlap", pass: false, detail: "no physics step" };
    await settle(liveMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return { name: "no-overlap", pass: false, detail: "no snapshot" };
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    const boxes: { pos: Vec3; quat: Quat; size: Vec3; r: number }[] = [];
    for (const eid of bodyEids) {
        if (s[(9 * cap + eid) * 4 + 3] <= 0) continue; // mass.w (col 9) <= 0 — static
        const pos = gpuPos(s, eid, cap);
        const qo = (1 * cap + eid) * 4; // B_QUAT (col 1)
        const ho = (10 * cap + eid) * 4; // B_HF (col 10) — halfExtents.xyz
        const hx = s[ho];
        const hy = s[ho + 1];
        const hz = s[ho + 2];
        boxes.push({
            pos,
            quat: [s[qo], s[qo + 1], s[qo + 2], s[qo + 3]],
            size: [2 * hx, 2 * hy, 2 * hz],
            r: Math.hypot(hx, hy, hz), // bounding-sphere radius for the center-distance prune
        });
    }
    if (boxes.length > 4096) {
        return {
            name: "no-overlap",
            pass: true,
            detail: `skipped (${boxes.length} boxes > 4096 — O(n²))`,
        };
    }
    let maxPen = 0;
    for (let i = 0; i < boxes.length; i++) {
        for (let j = i + 1; j < boxes.length; j++) {
            const a = boxes[i];
            const b = boxes[j];
            if (dist(a.pos, b.pos) > a.r + b.r + SPECULATIVE_DISTANCE) continue; // broadphase prune
            const { separation } = collide(
                { pos: a.pos, quat: a.quat, size: a.size },
                { pos: b.pos, quat: b.quat, size: b.size },
            );
            maxPen = Math.max(maxPen, -separation); // separation < 0 ⇒ penetrating; depth = -separation
        }
    }
    return {
        name: "no-overlap (settled bodies don't severely interpenetrate)",
        pass: maxPen <= OVERLAP_BOUND,
        detail: `max penetration ${maxPen.toExponential(2)} m (bound ${OVERLAP_BOUND}); ${boxes.length} dynamic boxes`,
    };
}

// the bodied-entity firehose compose: physics scatters each body's pose into transforms[eid] so a Body+Part
// renders without a Transform. The composed Xform's pos lane must equal the stepped pose.
async function firehosePose(): Promise<Check> {
    if (!liveMirror || !xformMirror) {
        return { name: "firehose-pose", pass: false, detail: "no transform mirror" };
    }
    await settle(liveMirror);
    await settle(xformMirror);
    const body = liveMirror.snapshot && new Float32Array(liveMirror.snapshot.bytes);
    const xf = xformMirror.snapshot && new Float32Array(xformMirror.snapshot.bytes);
    if (!body || !xf) return { name: "firehose-pose", pass: false, detail: "no snapshot" };
    const cap = body.length / (BODY_VEC4 * 4);
    let maxErr = 0;
    let maxSpeed = 0;
    let checked = 0;
    for (const eid of bodyEids) {
        const bodyPos = gpuPos(body, eid, cap);
        const m = eid * 12; // Xform stride (48 B); pos in the first lane (floats 0..2)
        const xfPos: Vec3 = [xf[m], xf[m + 1], xf[m + 2]];
        maxErr = Math.max(maxErr, dist(bodyPos, xfPos));
        const v = gpuVel(body, eid, cap);
        maxSpeed = Math.max(maxSpeed, Math.hypot(v[0], v[1], v[2]));
        checked++;
    }
    // both mirrors lag the staging ring identically, so a SETTLED pile reads the same-frame pose to a tight
    // bound. A moving pile (a rolling sphere/capsule that won't settle in bounded frames) can have the two
    // mirrors a frame apart, so the floor is its travel-per-frame (maxSpeed·DT) widened ×3 for the skew —
    // derived, not tuned. A real bug (wrong eid, dropped scatter) is O(scene) ≫ either floor, so it still trips.
    const tol = Math.max(1e-3, maxSpeed * DT * 3);
    return {
        name: "firehose-pose (transforms == body pose)",
        pass: checked > 0 && maxErr < tol,
        detail: `${checked} bodies, max translation err ${maxErr.toExponential(2)} (tol ${tol.toExponential(2)}, maxSpeed ${maxSpeed.toFixed(2)})`,
    };
}

// spawn-despawn (the eid-persistence proof, the one gate on the live pack/seed path). A body was spawned
// mid-sim; eid-indexed state persists, so the bodies that existed before the spawn must hold their settled
// pose — only the new body's slot was seeded. The old dense compact re-gathered the WHOLE pile from its
// spawn-pose slabs on any membership change → a settled heap would teleport back to its loose start grid.
async function spawnDespawn(): Promise<Check> {
    if (!liveMirror || !preSpawnPose) {
        return { name: "spawn-despawn", pass: false, detail: "no pre-spawn snapshot" };
    }
    await settle(liveMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return { name: "spawn-despawn", pass: false, detail: "no snapshot" };
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    // mean drift, not max: a reset re-gathers EVERY body from its spawn slab → all N teleport, mean huge.
    // Residual settle-creep moves only a few edge bodies a little → mean tiny.
    let totalDrift = 0;
    for (const [eid, pre] of preSpawnPose) totalDrift += dist(gpuPos(s, eid, cap), pre);
    const meanDrift = preSpawnPose.size > 0 ? totalDrift / preSpawnPose.size : 0;
    // the spawned body was seeded (finite pose, on/above the ground — not unseeded garbage)
    const spawnPos = gpuPos(s, spawnedEid, cap);
    const seeded =
        Number.isFinite(spawnPos[0]) &&
        Number.isFinite(spawnPos[1]) &&
        spawnPos[1] > GROUND_HALF_Y - 0.1;
    return {
        name: "spawn-despawn (existing bodies keep their pose)",
        pass: meanDrift < 0.15 && seeded,
        detail: `${preSpawnPose.size} bodies, mean drift ${meanDrift.toFixed(4)} m across the spawn; spawned body y ${spawnPos[1].toFixed(2)} ${seeded ? "seeded" : "NOT seeded"}`,
    };
}

// regime-cross (the two-sided safety of the frame-stale regime gates, C1.0/C1.2). Runs only when a
// threshold (smallN / ldsN) sits between the pre- and post-spawn live counts: the spawn must have flipped
// the step off its specialized path, despawning the extra must flip it back, and the originals' poses must
// hold across both flips (eid-indexed state + warmstart survive a regime change). Reads the step's
// `regimes` witness — spans can't distinguish the paths (the profiler holds a non-firing pass's last span).
async function regimeCross(): Promise<Check | null> {
    const step = Avbd.step;
    if (!step || !liveState || !preSpawnRegimes || spawnedEid < 0 || !liveMirror || !preSpawnPose) {
        return null;
    }
    const expect = (n: number) => ({ small: n > 0 && n <= cfgSmallN, lds: n > 0 && n <= cfgLdsN });
    const eq = (a: { small: boolean; lds: boolean }, b: { small: boolean; lds: boolean }) =>
        a.small === b.small && a.lds === b.lds;
    const pre = expect(preSpawnLive);
    const post = expect(preSpawnLive + 1);
    if (eq(pre, post)) return null; // no threshold straddles the spawn — the default cell never crosses
    const up = step.regimes;
    liveState.destroy(spawnedEid);
    await frames(60); // the gate's 1-2 frame readback staleness, then fresh steps on the restored path
    const back = step.regimes;
    await settle(liveMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return { name: "regime-cross", pass: false, detail: "no snapshot" };
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    let totalDrift = 0;
    for (const [eid, p] of preSpawnPose) totalDrift += dist(gpuPos(s, eid, cap), p);
    const meanDrift = preSpawnPose.size > 0 ? totalDrift / preSpawnPose.size : 0;
    const show = (r: { small: boolean; lds: boolean }): string =>
        `small=${r.small ? 1 : 0} lds=${r.lds ? 1 : 0}`;
    const pass = eq(preSpawnRegimes, pre) && eq(up, post) && eq(back, pre) && meanDrift < 0.15;
    return {
        name: "regime-cross (spawn flips the regime, despawn restores it)",
        pass,
        detail:
            `pre ${show(preSpawnRegimes)} (want ${show(pre)}) → spawned ${show(up)} (want ${show(post)}) ` +
            `→ despawned ${show(back)} (want ${show(pre)}); mean drift ${meanDrift.toFixed(4)} m`,
    };
}

// ── seeded GPU == oracle gates (a fresh PhysicsStep, no live-sim interference) ──

const GATE_CAP = 8;
const CORPUS_CAP = 24; // ≥ the largest topology (wide-pile = ground + 18)

async function seededGates(): Promise<Check[]> {
    const device = Compute.device;
    const phys = await PhysicsStep.create(device, GATE_CAP, GATE_CAP);
    const read = device.createBuffer({
        size: readBytes(GATE_CAP),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const run: RunGPU = (bodies, steps, colorCap = MAX_COLORS, mid) =>
        runStep(phys, read, GATE_CAP, bodies, steps, colorCap, mid);
    const checks: Check[] = [];
    try {
        checks.push(await freeFall(run));
        checks.push(await kickGate(run, phys));
        checks.push(await spinGate(run, phys));
        checks.push(
            await compareSingleStep(
                "single-step-exact (GPU == oracle)",
                stackScene,
                [0, 3, 8],
                GATE_CAP,
                run,
            ),
        );
        // sub-stepping (roadmap "dense-pile contact convergence"): the record() sub-step loop reproduces the
        // oracle's substeps loop. Runs the same stack scene one fixed step at substeps=2, GPU == oracle.
        checks.push(await substepParity(phys, read, GATE_CAP, stackScene, 2));
        checks.push(await stackWarmstart(run));
        checks.push(await frictionSlide(run));
        checks.push(await speculativeStop(run));
        checks.push(await sweptStop(run));
        checks.push(await interpolation(phys));
    } finally {
        phys.destroy();
        read.destroy();
    }
    return checks;
}

// the per-topology GPU == oracle gate: the six corpus scenes, each single-step-exact at distinct contact
// configurations (free-fall start, in contact, mid-topple). Memoryless → tight; iters-independent.
// Forced onto the BVH broadphase + the looped color solve (smallN 0, ldsN 0): gate counts sit under the
// default thresholds, so every other gate rides the small-N broadphase + the LDS-resident solve — this
// harness keeps the BVH descent AND the looped primal/commit/dual single-step-exact-gated per run.
async function corpusGates(): Promise<Check[]> {
    const device = Compute.device;
    const phys = await PhysicsStep.create(device, CORPUS_CAP, CORPUS_CAP);
    const read = device.createBuffer({
        size: readBytes(CORPUS_CAP),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const run: RunGPU = (bodies, steps, colorCap = MAX_COLORS) =>
        runStep(phys, read, CORPUS_CAP, bodies, steps, colorCap, undefined, 0, 0);
    const checks: Check[] = [];
    try {
        for (const sc of CORPUS) {
            checks.push(
                await compareSingleStep(
                    `corpus single-step: ${sc.name}`,
                    sc.bodies,
                    [0, 8, 30],
                    CORPUS_CAP,
                    run,
                ),
            );
        }
    } finally {
        phys.destroy();
        read.destroy();
    }
    return checks;
}

// a short penetrating stack on the ground — the single-step-exact scene.
const stackScene = (): OracleBody[] => [
    makeBody([20, 1, 20], 0, 0.5, [0, 0, 0]),
    makeBody([1, 1, 1], 1, 0.5, [0, 1.0, 0]),
    makeBody([1, 1, 1], 1, 0.5, [0.05, 1.95, 0.03]),
    makeBody([1, 1, 1], 1, 0.5, [-0.02, 2.9, 0.01]),
];

// free-fall (no contacts): the primal lands on the inertial target, so the GPU matches closed-form
// semi-implicit Euler exactly. No oracle — pure bedrock.
async function freeFall(run: RunGPU): Promise<Check> {
    const Steps = 30;
    const { state } = await run([makeBody([1, 1, 1], 1, 0.5, [0, 10, 0])], Steps);
    let v = 0;
    let y = 10;
    for (let i = 0; i < Steps; i++) {
        v += G * DT;
        y += v * DT;
    }
    const gy = gpuPos(state, 0, GATE_CAP)[1];
    const gvy = gpuVel(state, 0, GATE_CAP)[1];
    return {
        name: "free-fall (closed form)",
        pass: Math.abs(gy - y) < 1e-3 && Math.abs(gvy - v) < 1e-3,
        detail: `y ${gy.toFixed(4)} vs ${y.toFixed(4)}, vy ${gvy.toFixed(4)} vs ${v.toFixed(4)}`,
    };
}

// setVelocity (the launch impulse): a free body falls 2 steps, a queue-ordered setVelocity between
// submits overwrites its velocity upward, 4 more steps integrate it — closed-form semi-implicit Euler
// from the kicked velocity, same bedrock as free-fall. Gates the write's offset (B_VELL at the eid slot)
// and its queue ordering (lands before the next step's inertial pass, exactly once).
async function kickGate(run: RunGPU, phys: PhysicsStep): Promise<Check> {
    const Pre = 2;
    const Post = 4;
    const Kick = 8;
    const { state } = await run([makeBody([1, 1, 1], 1, 0.5, [0, 10, 0])], Pre + Post, MAX_COLORS, {
        at: Pre,
        run: () => phys.setVelocity(0, 0, Kick, 0),
    });
    let v = 0;
    let y = 10;
    for (let i = 0; i < Pre; i++) {
        v += G * DT;
        y += v * DT;
    }
    v = Kick;
    for (let i = 0; i < Post; i++) {
        v += G * DT;
        y += v * DT;
    }
    const gy = gpuPos(state, 0, GATE_CAP)[1];
    const gvy = gpuVel(state, 0, GATE_CAP)[1];
    return {
        name: "setVelocity (kick, closed form)",
        pass: Math.abs(gy - y) < 1e-3 && Math.abs(gvy - v) < 1e-3,
        detail: `y ${gy.toFixed(4)} vs ${y.toFixed(4)}, vy ${gvy.toFixed(4)} vs ${v.toFixed(4)}`,
    };
}

// setAngularVelocity (the spin twin of kickGate): a free body, a queue-ordered setAngularVelocity between
// submits, one inertial step integrates it — single-step-exact against the closed-form quaternion integrator.
// With no constraints, B_QUAT lands at qAdd(identity, ω·dt) = normalize((0, ½ω·dt, 0, 1)), and BDF1 recovery
// returns B_VELA.y = ω/√(1+¼ω²dt²) (the documented small-angle quat-integrator decay, physics.md). Gates the
// write's offset (B_VELA at the eid slot), its queue ordering (consumed by the next inertial pass, exactly
// once), and the +Y sign — the spindle's reported wrong-way startup is a SCENE/joint transient, not this API.
async function spinGate(run: RunGPU, phys: PhysicsStep): Promise<Check> {
    const W = 3; // rad/s about +Y
    const { state } = await run([makeBody([1, 1, 1], 1, 0.5, [0, 10, 0])], 1, MAX_COLORS, {
        at: 0,
        run: () => phys.setAngularVelocity(0, 0, W, 0),
    });
    const h = 0.5 * W * DT; // ½·ω·dt — the qAdd integrator's half-step y-term
    const inv = 1 / Math.hypot(h, 1); // 1/|(0, h, 0, 1)|
    const qy = h * inv; // expected B_QUAT.y
    const wy = W * inv; // expected B_VELA.y (the BDF1-recovered ω, decayed by 1/√(1+¼ω²dt²))
    const qo = (1 * GATE_CAP + 0) * 4; // B_QUAT
    const vo = (7 * GATE_CAP + 0) * 4; // B_VELA
    const gqy = state[qo + 1];
    const gqw = state[qo + 3];
    const gwy = state[vo + 1];
    const Tol = 1e-4; // single-step f32 rotation — tight, no chaos accumulation
    return {
        name: "setAngularVelocity (spin, closed form)",
        pass:
            Math.abs(gqy - qy) < Tol &&
            Math.abs(gqw - inv) < Tol &&
            Math.abs(gwy - wy) < Tol &&
            gqy > 0 &&
            gwy > 0,
        detail: `quat.y ${gqy.toFixed(5)} vs ${qy.toFixed(5)}, w.y ${gwy.toFixed(4)} vs ${wy.toFixed(4)} (+Y, no wrong-way)`,
    };
}

// stack-warmstart: a 3-box vertical stack settled over 60 steps of continuous warmstart — the multi-pair
// carry gate (three persistent pairs hold λ/k across frames). GPU == oracle final pose (loose — the bottom
// box bears 3× weight, so the f32/f64 equilibrium gap accumulates up the column) + exact persistence: a
// settled contact set is stable, so the lossless per-pair warmstart carries EVERY contact (merged == total).
async function stackWarmstart(run: RunGPU): Promise<Check> {
    const scene = (): OracleBody[] => [
        makeBody([20, 1, 20], 0, 0.5, [0, 0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [0, 1.0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [0, 2.0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [0, 3.0, 0]),
    ];
    const Steps = 60;
    const { state, contacts, merged } = await run(scene(), Steps);
    const ora = oracle(scene(), Steps);
    let maxErr = 0;
    let maxSpeed = 0;
    let finite = true;
    for (let i = 1; i < 4; i++) {
        const gp = gpuPos(state, i, GATE_CAP);
        const gv = gpuVel(state, i, GATE_CAP);
        if (![...gp, ...gv].every(Number.isFinite)) finite = false;
        maxErr = Math.max(maxErr, dist(gp, ora[i].posLin));
        maxSpeed = Math.max(maxSpeed, Math.hypot(gv[0], gv[1], gv[2]));
    }
    const persisted = contacts > 0 && merged === contacts;
    return {
        name: "stack-warmstart (GPU == oracle, multi-pair carry)",
        pass: finite && maxSpeed < 0.05 && maxErr < 3e-2 && persisted,
        detail: `max-pos err ${maxErr.toExponential(2)}, max speed ${maxSpeed.toFixed(3)} m/s, warmstart ${merged}/${contacts}`,
    };
}

// friction-slide: a box launched horizontally, run multi-step under warmstart — kinetic friction must
// decelerate it to rest. The GPU gate for kinetic friction: a sliding contact whose tangent penalty ramps
// unboundedly (gating on the post-clamp force) fades friction to ~0, so the box never stops. Only a
// multi-step slide catches it. μ=0.5 ⇒ stop ≈ v0²/2μg ≈ 2.5 m from 5 m/s; the bug slides ≫ 5 m.
async function frictionSlide(run: RunGPU): Promise<Check> {
    const scene = (): OracleBody[] => [
        makeBody([20, 1, 20], 0, 0.5, [0, 0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [-3, 1.0, 0], [5, 0, 0]),
    ];
    const Steps = 80;
    const { state } = await run(scene(), Steps);
    const ora = oracle(scene(), Steps);
    const gp = gpuPos(state, 1, GATE_CAP);
    const gv = gpuVel(state, 1, GATE_CAP);
    const posErr = dist(gp, ora[1].posLin);
    const speed = Math.hypot(gv[0], gv[1], gv[2]);
    const slid = gp[0] - -3;
    const finite = [...gp, ...gv].every(Number.isFinite);
    return {
        name: "friction-slide (kinetic friction stops the box; GPU == oracle)",
        pass: finite && speed < 0.1 && slid > 0.5 && slid < 5 && posErr < 3e-2,
        detail: `slid ${slid.toFixed(3)} m, end speed ${speed.toFixed(3)} m/s, pos err ${posErr.toExponential(2)}`,
    };
}

// speculative-stop (Phase 4.8.3): a box dropped fast from WITHIN the speculative band onto the ground. The
// SAT generates a contact while still separated (gap < SPECULATIVE_DISTANCE), carrying the true +gap, and
// the repulsion-only constraint lands the box AT the surface in one step — no penetration pop, no tunnel.
// Overlap-only CD tunnelled ~0.64 below in one step (falling 100 m/s ≈ 1.7 m/step).
async function speculativeStop(run: RunGPU): Promise<Check> {
    const touchingY = 1.0; // box-center height where the unit box face meets the ground top (0.5 + 0.5)
    const gap = SPECULATIVE_DISTANCE * 0.75; // 0.03 — inside the band, so the speculative contact fires
    const bodies: OracleBody[] = [
        makeBody([20, 1, 20], 0, 0.5, [0, 0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [0, touchingY + gap, 0], [0, -100, 0]),
    ];
    const { state, contacts, colors } = await run(bodies, 1);
    const ora = oracle(bodies, 1, oracleColors(colors, bodies.length));
    const gp = gpuPos(state, 1, GATE_CAP);
    const gv = gpuVel(state, 1, GATE_CAP);
    const posErr = dist(gp, ora[1].posLin);
    const velErr = dist(gv, ora[1].velLin);
    const finite = [...gp, ...gv].every(Number.isFinite);
    const landed = gp[1] > touchingY - 0.05 && gp[1] < touchingY + gap;
    return {
        name: "speculative-stop (in-band box lands at contact; GPU == oracle)",
        pass: finite && contacts > 0 && landed && posErr < POS_TOL && velErr < VEL_TOL,
        detail: `box y ${gp[1].toFixed(4)} (gap-start ${(touchingY + gap).toFixed(2)}), ${contacts} contact(s), pos err ${posErr.toExponential(2)}, vel err ${velErr.toExponential(2)}`,
    };
}

// swept-stop (Phase 4.8.4): a box dropped fast from a gap BEYOND the static band (0.5 ≫ SPECULATIVE_
// DISTANCE). Without the velocity sweep the box (v·dt ≈ 1.7 m) tunnels clean through in one step; the
// sweep widens the SAT band to max(SPECULATIVE_DISTANCE, |closing|·dt), so the frame-start SAT generates
// the swept contact and the repulsion-only constraint arrests the approach — caught above the surface.
async function sweptStop(run: RunGPU): Promise<Check> {
    const touchingY = 1.0;
    const gap = 0.5; // ≫ SPECULATIVE_DISTANCE (0.04): the static band generates nothing at frame start
    const bodies: OracleBody[] = [
        makeBody([20, 1, 20], 0, 0.5, [0, 0, 0]),
        makeBody([1, 1, 1], 1, 0.5, [0, touchingY + gap, 0], [0, -100, 0]),
    ];
    const { state, contacts, colors } = await run(bodies, 1);
    const ora = oracle(bodies, 1, oracleColors(colors, bodies.length));
    const gp = gpuPos(state, 1, GATE_CAP);
    const gv = gpuVel(state, 1, GATE_CAP);
    const posErr = dist(gp, ora[1].posLin);
    const velErr = dist(gv, ora[1].velLin);
    const finite = [...gp, ...gv].every(Number.isFinite);
    const caught = gp[1] > touchingY - 0.05 && gp[1] < touchingY + gap + 0.05;
    return {
        name: "swept-stop (fast box beyond the band caught, no tunnel; GPU == oracle)",
        pass: finite && contacts > 0 && caught && posErr < POS_TOL && velErr < VEL_TOL,
        detail: `box y ${gp[1].toFixed(4)} (entry ${(touchingY + gap).toFixed(2)}), ${contacts} contact(s), pos err ${posErr.toExponential(2)}, vel err ${velErr.toExponential(2)}`,
    };
}

// render interpolation (Phase 5): the compose blends prev→curr pose by alpha (prev = B_INITL/B_INITQ, the
// inertial pass's x⁻; curr = B_POS/B_QUAT). The settled-pile firehose-pose gate can't pin this — prev ≈ curr
// there, so it passes even with interpolation removed. This seeds DISTINCT prev/curr columns directly (which
// runStep's seedBody can't — it forces prev == curr) and reads the composed transform back at alpha 0/0.5/1.
// Body 0 pins position lerp + the alpha direction; body 1 pins the quat nlerp incl. the shortest-arc flip
// (prev = identity, curr = −rotY(90°): the SAME rotation in the opposite hemisphere, so dot < 0 must flip,
// else the nlerp takes the long way). At alpha 0.5 the chord midpoint normalizes to the arc midpoint exactly
// (nlerp == slerp at 0.5 by symmetry), so the expected rotation is rotY(alpha·90°) at all three test points.
async function interpolation(phys: PhysicsStep): Promise<Check> {
    const device = Compute.device;
    const cap = phys.eidCap;
    const Bytes = cap * 48; // Xform per eid (48 B)
    const xforms = device.createBuffer({
        label: "interp-xforms",
        size: Bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
        label: "interp-read",
        size: Bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const prev0: Vec3 = [-2, 1, 3];
    const curr0: Vec3 = [4, 7, -5];
    const c = Math.cos(Math.PI / 4); // = sin(π/4); the −rotY(90°) quat's y/w components
    try {
        phys.configure({
            dt: DT,
            gravity: G,
            alpha: ALPHA,
            penalty: PENALTY_MIN,
            betaLin: BETA_LIN,
            gamma: GAMMA,
            iterations: ITERS,
            maxColors: MAX_COLORS,
        });
        const arr = new Float32Array(cap * BODY_VEC4 * 4);
        const set = (i: number, col: number, x: number, y: number, z: number, w: number): void => {
            const o = (col * cap + i) * 4;
            arr[o] = x;
            arr[o + 1] = y;
            arr[o + 2] = z;
            arr[o + 3] = w;
        };
        // body 0 — position lerp (quats identity, so rotation is inert)
        set(0, 0, curr0[0], curr0[1], curr0[2], 0); // B_POS   = curr
        set(0, 4, prev0[0], prev0[1], prev0[2], 0); // B_INITL = prev
        set(0, 1, 0, 0, 0, 1); // B_QUAT  = identity
        set(0, 5, 0, 0, 0, 1); // B_INITQ = identity
        set(0, 10, 0.5, 0.5, 0.5, 0.5); // halfExtents → scale 1 (col0 = R·x̂ directly)
        // body 1 — quat nlerp + shortest-arc flip (position fixed)
        set(1, 0, 0, 10, 0, 0);
        set(1, 4, 0, 10, 0, 0);
        set(1, 1, 0, -c, 0, -c); // B_QUAT  = −rotY(90°), the opposite hemisphere of identity
        set(1, 5, 0, 0, 0, 1); // B_INITQ = identity
        set(1, 10, 0.5, 0.5, 0.5, 0.5);
        device.queue.writeBuffer(phys.bodies, 0, arr);
        phys.gateSetCount(2);

        let posErr = 0;
        let rotErr = 0;
        for (const a of [0, 0.5, 1]) {
            const enc = device.createCommandEncoder();
            phys.compose(enc, xforms, a);
            enc.copyBufferToBuffer(xforms, 0, read, 0, Bytes);
            device.queue.submit([enc.finish()]);
            await read.mapAsync(GPUMapMode.READ, 0, Bytes);
            const m = new Float32Array(read.getMappedRange(0, Bytes));
            // body 0 pos lane (Xform floats 0..2) vs the position lerp
            const t: Vec3 = [m[0], m[1], m[2]];
            const want: Vec3 = [
                prev0[0] + (curr0[0] - prev0[0]) * a,
                prev0[1] + (curr0[1] - prev0[1]) * a,
                prev0[2] + (curr0[2] - prev0[2]) * a,
            ];
            posErr = Math.max(posErr, dist(t, want));
            // body 1's stored quat (eid 1 record base 12, quat at +4) → R·x̂ (sign-invariant, unlike raw
            // quat components) vs rotY(alpha·90°)·x̂ = (cos θ, 0, −sin θ): pins the nlerp incl. shortest-arc flip
            const th = (a * Math.PI) / 2;
            const qx = m[16];
            const qy = m[17];
            const qz = m[18];
            const qw = m[19];
            const col0: Vec3 = [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy + qw * qz),
                2 * (qx * qz - qw * qy),
            ];
            rotErr = Math.max(rotErr, dist(col0, [Math.cos(th), 0, -Math.sin(th)]));
            read.unmap();
        }
        // f32 lerp + nlerp-normalize; every bug class (no interp, inverted alpha, wrong source column,
        // missing flip) produces O(1) error, orders above this.
        const tol = 1e-4;
        return {
            name: "interpolation (compose blends prev→curr by alpha)",
            pass: posErr < tol && rotErr < tol,
            detail: `pos err ${posErr.toExponential(2)}, rot err ${rotErr.toExponential(2)}`,
        };
    } finally {
        xforms.destroy();
        read.destroy();
    }
}

// ── perf reporter ────────────────────────────────────────────────────────────

// the complete per-step GPU breakdown, in record() execution order. `Profile.gpu` is a single post-window
// snapshot ≈ one physics step (the window-MEAN in metrics.gpu.passes is inflated + non-monotonic by the
// drain/hold cadence — NOT per-step). The binding cost is dispatch encode, not these spans (physics.md
// "Dispatch count"); the primal span sums the capped `ITERS × dispatchedColors` color dispatches.
const STEP_PASSES = [
    "phys:pack",
    "phys:aabb",
    "bvh:bounds",
    "bvh:morton",
    "bvh:sort",
    "bvh:build",
    "phys:broadphase",
    "phys:collide",
    "phys:csr",
    "phys:coloring",
    "phys:joint", // jointInit + jointDual (0 when the scene authors no joints)
    "phys:inertial",
    "phys:primal",
    "phys:dual",
    "phys:velocity",
    "phys:compose",
] as const;

// always-pass perf REPORTER (not a correctness gate): publishes the per-step spans + dispatch count +
// memory + the fall-through signals as structured `Check.data`, the seam scripts/physics-bench.ts reads.
async function measured(): Promise<Check> {
    const step = Avbd.step;
    const get = (name: string): number => Profile.gpu.get(name) ?? 0;
    const data: Record<string, number> = {};
    for (const n of STEP_PASSES) data[n] = get(n);
    const full = STEP_PASSES.reduce((s, n) => s + data[n], 0);
    data.dispatchedColors = step?.dispatchedColors ?? 0;
    data.bytes = step?.bytes ?? 0;
    data.bodies = bodyEids.length;
    data.dynamicCount = dynamicCount;
    data.shape = pileShape; // 0 box / 1 sphere / 2 capsule / 3 hull — the bench labels the cell
    data.joints = pileJoints ? 1 : 0;
    data.layout = ["grid", "heap", "pyramid"].indexOf(pileLayout); // the bench labels the cell
    data.boundary = pileBoundary === "drum" ? 1 : 0;

    const live = await liveStats();
    if (live) {
        data.minY = live.minY;
        data.throughFloor = live.throughFloor;
        data.offEdge = live.offEdge;
    }
    if (countersMirror) {
        await settle(countersMirror);
        const c = countersMirror.snapshot && new Uint32Array(countersMirror.snapshot.bytes);
        if (c) {
            data.descentBlock = c[3]; // per-body block overflow
            data.bodyPool = c[5]; // body-pool overflow
            data.contacts = c[6]; // warmstarted contacts ≈ live contact records (the audit's traffic input)
            data.staticDrop = c[7]; // static-support dropped → the direct fall-through signal
        }
    }

    const resolved = get("phys:collide") > 0 && get("phys:primal") > 0;
    const label = (n: string): string => n.replace("phys:", "").replace("bvh:", "bvh.");
    const parts = STEP_PASSES.map((n) => `${label(n)} ${data[n].toFixed(3)}`).join(" · ");
    return {
        name: "measured (solver spans)",
        pass: true, // a reporter — the bench gates on the payload, this never fails the gym
        detail: resolved ? `step ~${full.toFixed(3)} ms = ${parts} ms` : "no solver spans resolved",
        data,
    };
}

// ── rounded narrowphase gate (the sphere/capsule pile rows, roadmap Examples Stage 3) ───────────────────
// The sphere/capsule narrowphase codegen check, run as the sphere/capsule pile rows. A standalone compute
// kernel runs the production `collideRounded` (both rounded) + `collideRoundedPolytope` (one rounded, one
// box) WGSL over a closed-form config set and diffs against the f64 oracle `narrowphase` (tests/avbd/rounded),
// byte-exact (f32 tol): count, the shared B→A normal, every contact's feature key + local arms (so a
// capsule-box's two endpoint contacts are both gated). It seeds its OWN config bodies, so it is independent
// of the live pile and shape-identical for sphere and capsule (the whole rounded narrowphase, run on either
// rounded row). The CORE-arm storage byte-exactness here is the codegen half of the rotational-stability rule
// (physics.md): a baked-radius arm would diverge from the oracle's core arm and red the gate. The live rounded
// settle is the drum boundary above; the rolling/conservation math is the `rounded.oracle.ts` CPU tier.
const ROUNDED_TOL = 1e-4; // f32 vs f64 over ~20 ops on magnitudes <= 10 — the gym `sat`'s derived margin

interface Cfg {
    name: string;
    a: OracleBody;
    b: OracleBody;
    dRel: [number, number, number];
}
const Z90: [number, number, number, number] = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
const roundedConfigs: Cfg[] = [
    {
        name: "sphere-sphere vertical",
        a: sphere(0.5, 1, 0.5, [0, 0.9, 0]),
        b: sphere(0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-sphere diagonal",
        a: sphere(1, 1, 0.5, [1.74, 2.32, 0]),
        b: sphere(2, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-capsule mid",
        a: sphere(0.5, 1, 0.5, [0.9, 0, 0]),
        b: capsule(1, 0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-capsule endpoint",
        a: sphere(0.5, 1, 0.5, [0, 1.9, 0]),
        b: capsule(1, 0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "capsule-capsule collinear",
        a: capsule(1, 0.5, 1, 0.5, [0, 2.9, 0]),
        b: capsule(1, 0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "capsule-capsule crossed",
        a: capsule(1, 0.5, 1, 0.5, [0, 1.9, 0], [0, 0, 0], Z90),
        b: capsule(1, 0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "separated past band",
        a: sphere(0.5, 1, 0.5, [0, 1.2, 0]),
        b: sphere(0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "within speculative band",
        a: sphere(0.5, 1, 0.5, [0, 1.02, 0]),
        b: sphere(0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "velocity sweep (fast approach)",
        a: sphere(0.5, 1, 0.5, [0, 1.2, 0]),
        b: sphere(0.5, 1, 0.5, [0, 0, 0]),
        dRel: [0, -0.5, 0],
    },
    // rounded × box (rounded-box) — sphere-box features + the box-as-A swap, capsule-box endpoint sampling
    {
        name: "sphere-box face",
        a: sphere(0.5, 1, 0.5, [0, 1.4, 0]),
        b: makeBody([2, 2, 2], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-box edge",
        a: sphere(0.5, 1, 0.5, [1.27, 1.36, 0]),
        b: makeBody([2, 2, 2], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-box corner",
        a: sphere(0.5, 1, 0.5, [1.3, 1.3, 1.15]),
        b: makeBody([2, 2, 2], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-box interior (deep)",
        a: sphere(0.5, 1, 0.5, [0.3, 0, 0]),
        b: makeBody([2, 2, 2], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "sphere-box, box-as-A (swap)",
        a: makeBody([2, 2, 2], 0, 0.5, [0, 0, 0]),
        b: sphere(0.5, 1, 0.5, [0, 1.4, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "capsule-box flat (2 contacts)",
        a: capsule(1, 0.5, 1, 0.5, [0, 1.4, 0], [0, 0, 0], Z90),
        b: makeBody([4, 2, 4], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
    {
        name: "capsule-box vertical (1 contact)",
        a: capsule(1, 0.5, 1, 0.5, [0, 2.4, 0]),
        b: makeBody([4, 2, 4], 0, 0.5, [0, 0, 0]),
        dRel: [0, 0, 0],
    },
];

// the standalone GPU kernel: one thread per config, run the production narrowphase, write count + the shared
// basis.r0 (one normal per rounded manifold) + every contact's feature key + local arms. All contacts written
// so a capsule-box's 2nd endpoint is gated too.
const ROUNDED_OUT_STRIDE = 1 + 3 + MAX_CONTACTS * 7; // count, normal(3), MAX × (feat, rA.xyz, rB.xyz)
const ROUNDED_CFG_STRIDE = 32; // 8 vec4: posA, quatA, sizeRadA, posB, quatB, sizeRadB, dRel, shapes
const ROUNDED_KERNEL_WGSL = `${COLLIDE_WGSL}${HULL_WGSL}
struct Cfg { posA: vec4<f32>, quatA: vec4<f32>, sizeRadA: vec4<f32>, posB: vec4<f32>, quatB: vec4<f32>, sizeRadB: vec4<f32>, dRel: vec4<f32>, shapes: vec4<f32> };
@group(0) @binding(0) var<storage, read> cfgs: array<Cfg>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
@group(0) @binding(3) var<storage, read> hullData: array<u32>; // unused here (box polys), declared for HULL_WGSL

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.x) { return; }
    let c = cfgs[i];
    let sA = u32(c.shapes.x);
    let sB = u32(c.shapes.y);
    // dispatch like the production collide pass: both rounded → collideRounded, mixed → collideRoundedPolytope
    var r: SatResult;
    if (sA != 0u && sB != 0u) {
        r = collideRounded(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w,
                           c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, c.dRel.xyz);
    } else {
        r = collideRoundedPolytope(c.posA.xyz, c.quatA, c.sizeRadA.xyz, c.sizeRadA.w, sA, 0u,
                                   c.posB.xyz, c.quatB, c.sizeRadB.xyz, c.sizeRadB.w, sB, 0u, c.dRel.xyz);
    }
    let base = i * params.y;
    out[base] = bitcast<f32>(r.count);
    out[base + 1u] = r.basis.r0.x; out[base + 2u] = r.basis.r0.y; out[base + 3u] = r.basis.r0.z;
    for (var k = 0u; k < ${MAX_CONTACTS}u; k = k + 1u) {
        let o = base + 4u + k * 7u;
        out[o] = bitcast<f32>(r.feat[k]);
        out[o + 1u] = r.rA[k].x; out[o + 2u] = r.rA[k].y; out[o + 3u] = r.rA[k].z;
        out[o + 4u] = r.rB[k].x; out[o + 5u] = r.rB[k].y; out[o + 6u] = r.rB[k].z;
    }
}`;

interface GpuRounded {
    count: number;
    normal: [number, number, number];
    contacts: { feature: number; rA: [number, number, number]; rB: [number, number, number] }[];
}

// run all configs once through the production-shape kernel, decode the per-config results.
async function runRoundedKernel(): Promise<GpuRounded[]> {
    const device = Compute.device;
    const n = roundedConfigs.length;
    const cfgData = new Float32Array(n * ROUNDED_CFG_STRIDE);
    for (let i = 0; i < n; i++) {
        const c = roundedConfigs[i];
        const o = i * ROUNDED_CFG_STRIDE;
        cfgData.set([c.a.posLin[0], c.a.posLin[1], c.a.posLin[2], 0], o);
        cfgData.set(c.a.posAng, o + 4);
        cfgData.set([c.a.size[0], c.a.size[1], c.a.size[2], c.a.roundRadius], o + 8);
        cfgData.set([c.b.posLin[0], c.b.posLin[1], c.b.posLin[2], 0], o + 12);
        cfgData.set(c.b.posAng, o + 16);
        cfgData.set([c.b.size[0], c.b.size[1], c.b.size[2], c.b.roundRadius], o + 20);
        cfgData.set([c.dRel[0], c.dRel[1], c.dRel[2], 0], o + 24);
        cfgData.set([c.a.shape, c.b.shape, 0, 0], o + 28); // shape tags (f32, exact for 0/1/2)
    }

    const cfgBuf = device.createBuffer({
        label: "rnd-cfg",
        size: cfgData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outBuf = device.createBuffer({
        label: "rnd-out",
        size: n * ROUNDED_OUT_STRIDE * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramBuf = device.createBuffer({
        label: "rnd-params",
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const readBuf = device.createBuffer({
        label: "rnd-read",
        size: n * ROUNDED_OUT_STRIDE * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // these configs are all box/sphere/capsule (box polys → no hull read), but HULL_WGSL declares the binding,
    // so bind the packed registry (a 1-u32 stub when no hulls are registered) to satisfy it.
    const hullBytes = packHulls();
    const hullBuf = device.createBuffer({
        label: "rnd-hulls",
        size: hullBytes.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(cfgBuf, 0, cfgData);
    device.queue.writeBuffer(paramBuf, 0, new Uint32Array([n, ROUNDED_OUT_STRIDE, 0, 0]));
    device.queue.writeBuffer(hullBuf, 0, hullBytes as Uint32Array<ArrayBuffer>);

    const pipeline = await device.createComputePipelineAsync({
        label: "rounded-kernel",
        layout: "auto",
        compute: {
            module: device.createShaderModule({
                label: "rounded-module",
                code: ROUNDED_KERNEL_WGSL,
            }),
            entryPoint: "main",
        },
    });
    const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: cfgBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: paramBuf } },
            { binding: 3, resource: { buffer: hullBuf } },
        ],
    });
    const enc = device.createCommandEncoder({ label: "rounded-enc" });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 64));
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, n * ROUNDED_OUT_STRIDE * 4);
    device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bytes = readBuf.getMappedRange().slice(0);
    readBuf.unmap();
    cfgBuf.destroy();
    outBuf.destroy();
    paramBuf.destroy();
    hullBuf.destroy();
    readBuf.destroy();

    const f = new Float32Array(bytes);
    const u = new Uint32Array(bytes);
    const out: GpuRounded[] = [];
    for (let i = 0; i < n; i++) {
        const b = i * ROUNDED_OUT_STRIDE;
        const count = u[b];
        const contacts: GpuRounded["contacts"] = [];
        for (let k = 0; k < count; k++) {
            const o = b + 4 + k * 7;
            contacts.push({
                feature: u[o],
                rA: [f[o + 1], f[o + 2], f[o + 3]],
                rB: [f[o + 4], f[o + 5], f[o + 6]],
            });
        }
        out.push({ count, normal: [f[b + 1], f[b + 2], f[b + 3]], contacts });
    }
    return out;
}

// diff one GPU result against the oracle narrowphase: count, the shared B→A normal, then EACH contact's arms
// matched by feature key (so a capsule-box's two endpoint contacts are both checked). Returns "" on agreement,
// else the first divergence.
function diffRounded(cfg: Cfg, got: GpuRounded): { detail: string; err: number } {
    const { contacts, basis } = narrowphase(cfg.a, cfg.b, cfg.dRel);
    if (got.count !== contacts.length)
        return { detail: `count ${got.count} vs oracle ${contacts.length}`, err: 0 };
    if (contacts.length === 0) return { detail: "", err: 0 };
    let err = 0;
    const cmp = (g: number[], w: number[], label: string): string => {
        for (let i = 0; i < 3; i++) {
            const d = Math.abs(g[i] - w[i]);
            err = Math.max(err, d);
            if (d >= ROUNDED_TOL)
                return `${label}[${i}] gpu ${g[i].toExponential(3)} vs oracle ${w[i].toExponential(3)}`;
        }
        return "";
    };
    let detail = cmp(got.normal, basis[0], "normal");
    for (const wc of contacts) {
        if (detail) break;
        const gc = got.contacts.find((c) => c.feature >>> 0 === wc.feature >>> 0);
        if (!gc) {
            detail = `oracle feature 0x${(wc.feature >>> 0).toString(16)} missing on GPU`;
            break;
        }
        detail =
            cmp(gc.rA, wc.rA, `rA(${wc.feature & 0xff})`) ||
            cmp(gc.rB, wc.rB, `rB(${wc.feature & 0xff})`);
    }
    return { detail, err };
}

// the rounded narrowphase gate — one check per config (GPU == oracle) + a max-error roll-up.
async function roundedGates(): Promise<Check[]> {
    const checks: Check[] = [];
    const results = await runRoundedKernel();
    let maxErr = 0;
    for (let i = 0; i < roundedConfigs.length; i++) {
        const { detail, err } = diffRounded(roundedConfigs[i], results[i]);
        maxErr = Math.max(maxErr, err);
        checks.push({
            name: `rounded narrowphase — ${roundedConfigs[i].name}`,
            pass: detail === "",
            detail:
                detail ||
                `${results[i].count} contact(s), normal + arms match (err ${err.toExponential(2)})`,
        });
    }
    checks.push({
        name: "rounded narrowphase — max-abs-error (TOL 1e-4)",
        pass: maxErr < ROUNDED_TOL,
        detail: maxErr.toExponential(2),
    });
    return checks;
}

register(scenario);
