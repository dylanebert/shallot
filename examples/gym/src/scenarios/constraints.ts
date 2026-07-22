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
    Joint,
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
    SlabPlugin,
    Spring,
    type State,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
import {
    Avbd,
    BODY_VEC4,
    JOINT_REC_VEC4,
    type JointDef,
    PENALTY_MIN,
    PhysicsStep,
    type SpringDef,
    WORLD,
} from "@dylanebert/shallot/avbd/core";
import { Profile, ProfilePlugin } from "@dylanebert/shallot/extras";
import { joint as oracleJoint } from "../../../../packages/shallot/tests/avbd/joint";
// the f64 oracle (tests/, out of the published src/) is the executable spec the GPU constraint solve compares
// against — reached by relative path, like the pile scenario reaches the rigid oracle.
import type { Quat, Vec3 } from "../../../../packages/shallot/tests/avbd/math";
import {
    body as makeBody,
    type Body as OracleBody,
} from "../../../../packages/shallot/tests/avbd/rigid";
import { makeSolver, step as oracleStep } from "../../../../packages/shallot/tests/avbd/solver";
import { spring as oracleSpring } from "../../../../packages/shallot/tests/avbd/spring";
import { type Check, frames, type Params, register, type Scenario, settle } from "../gym";

// constraints — the authored Force scenario (Phase 6, the §6.6 promotion of the `springs` + `joints` labs).
// Springs (soft, Phase 6.1) + joints (hard, Phase 6.2) share one scene because they share the constraint
// adjacency (`constraintCsr` / `constraintList`); the solver branches on the kind tag. Both author as ECS
// COMPONENTS — a `Spring` / `Joint` is its own entity holding two `@name` body refs + local anchors
// (`ecs.md`, like `Tween.target`) — so this gym scenario exercises the ConstraintSystem upload-on-change
// path the imperative labs (`setSprings`/`setJoints`) don't. The GPU stamps each in the primal, a per-joint
// jointInit/jointDual pair warmstarts + ramps the hard ones, the coloring repairs hard same-color conflicts.
//
// The gate set, by what each covers:
//   • single-step GPU == oracle (the tight regression gate, NEW vs the labs): seed a spring + a joint rig in
//     a fresh PhysicsStep, step ONE cold frame on the GPU + the f64 oracle, compare pose + velocity. Cold ⇒
//     memoryless ⇒ tight (the spring mg/k pull + the joint linear pin reproduced to ~1e-7).
//   • the live structural invariants (what two float impls can't bit-match over a settle, so they're the
//     observable that ISN'T a per-step compare): the spring mg/k hold + the chain's mirror-symmetric angular
//     stamp; the fixed cantilever's angular lock; the spherical pendulum's held-arm swing; the rope's
//     inextensible drape; the recycle-version realias drop; the construction + both-static guards; the
//     coloring split (a dynamic joint pair must get different colors — GPU == oracle can't see this, the
//     oracle runs the GPU's coloring).
//   • measured — the per-step spans + the joint pass cost as structured `Check.data`, the bench seam.
//
// Spring rigs sit on the z = -10 lane, joint rigs on z = 0 (a shared floor under the joint drops), so they
// don't interfere. Like the pile scenario, the live view is the orbit camera + the HUD; the interactive god
// grab stays in the sandbox gravity gun (the gym apps don't depend on the pick/grab helper).

const G = -10;
const DT = 1 / 60;
const ALPHA = 0.99;
const BETA_LIN = 1e4;
const BETA_ANG = 100; // the joint angular penalty ramp (springs/contacts ignore it)
const GAMMA = 0.999;
const MAX_COLORS = 8;
const ITERS = 10; // the gym's robustness config (the rigid joint converges tighter than the shipped iters=6)

const cfg = {
    dt: DT,
    gravity: G,
    alpha: ALPHA,
    penalty: PENALTY_MIN,
    betaLin: BETA_LIN,
    betaAng: BETA_ANG,
    gamma: GAMMA,
    iterations: ITERS,
    maxColors: MAX_COLORS,
};

// ── spring rig constants (the hold/chain rigs, z = -10) ──
const SPRING_Z = -10;
const ANCHOR_Y = 14;
const REST = 4;
const STIFFNESS = 100;
const BLOCK = 2; // full size of the hanging block (mass 8 at density 1)
const BLOCK_MASS = BLOCK * BLOCK * BLOCK;
const EXT = (BLOCK_MASS * Math.abs(G)) / STIFFNESS; // mg/k = 0.8 — the static spring extension past rest
const Y_EQ = ANCHOR_Y - REST - EXT; // 9.2 — the hold rig's closed-form equilibrium

// ── joint rig constants (cantilever/pendulum/rope/recycle/guards, z = 0) ──
const H = 12;
const SPAN = 1; // cantilever link spacing
const ARM = 2.5; // pendulum arm length
const THETA0 = 0.6; // pendulum release angle (rad)
const DROP = 2; // recycle-rig hang distance below the anchor
const ROPE_LINKS = 8;
const ROPE_SEG = 0.7;
const ROPE_X = 10;

// captured eids (solver state is eid-indexed — the gates read by eid)
let holdBlock = -1;
let chainDyn: number[] = [];
let cantBoxes: number[] = [];
let pendBob = -1;
let recycleBox = -1;
let guardBox = -1;
let ropeLinks: number[] = [];
let recycleHeldY = Number.NaN; // the recycle box's y while held (captured before the version bump)
let liveMirror: Mirror | null = null;
let colorMirror: Mirror | null = null;
let countersMirror: Mirror | null = null;

const STATIC: [number, number, number] = [0.3, 0.32, 0.36];

function box(
    state: State,
    pos: Vec3,
    half: Vec3,
    mass: number,
    color: [number, number, number],
    quat: Quat = [0, 0, 0, 1],
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// author a Spring as its own entity (the component path the ConstraintSystem uploads). The anchors are
// vec4 lanes (w unused); stateless, so no warmstart.
function addSpring(
    state: State,
    a: number,
    b: number,
    rA: Vec3,
    rB: Vec3,
    stiffness: number,
    rest: number,
): void {
    const e = state.create();
    state.add(e, Spring);
    Spring.a.set(e, a);
    Spring.b.set(e, b);
    Spring.rA.set(e, rA[0], rA[1], rA[2], 0);
    Spring.rB.set(e, rB[0], rB[1], rB[2], 0);
    Spring.stiffness.set(e, stiffness);
    Spring.rest.set(e, rest);
}

// author a Joint as its own entity (the component path the ConstraintSystem uploads). `stiffnessAng` 0 =
// spherical (free rotation), ∞ = fixed (orientation locked); the linear pin is always rigid (the JointDef default).
function addJoint(state: State, a: number, b: number, rA: Vec3, rB: Vec3, stiffnessAng = 0): void {
    const e = state.create();
    state.add(e, Joint);
    Joint.a.set(e, a);
    Joint.b.set(e, b);
    Joint.rA.set(e, rA[0], rA[1], rA[2], 0);
    Joint.rB.set(e, rB[0], rB[1], rB[2], 0);
    Joint.stiffnessAng.set(e, stiffnessAng);
}

const qz = (rad: number): Quat => [0, 0, Math.sin(rad / 2), Math.cos(rad / 2)];

const scenario: Scenario = {
    name: "constraints",
    params: [],

    async build(_canvas, _p: Params) {
        const { state, dispose } = await run({
            defaults: false,
            capacity: 64,
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

        state.add(state.create(), AmbientLight);
        state.add(state.create(), DirectionalLight);
        Avbd.step?.configure(cfg);

        // a floor under the joint rigs so the deactivation-test boxes (recycle + guard) drop and LAND rather
        // than fall forever — "released, fell, landed" reads as intentional; the held rigs hang above it.
        box(state, [3, -0.5, 0], [16, 0.5, 8], 0, [0.28, 0.3, 0.34]);

        // ── spring rigs (z = -10) ──
        const half: Vec3 = [BLOCK / 2, BLOCK / 2, BLOCK / 2];
        // hold rig: a block placed AT the static equilibrium y_eq holds it (the spring force mg cancels gravity)
        const holdAnchor = box(state, [-5, ANCHOR_Y, SPRING_Z], [0.5, 0.5, 0.5], 0, STATIC);
        holdBlock = box(state, [-5, Y_EQ, SPRING_Z], half, BLOCK_MASS, [0.4, 0.75, 0.5]);
        addSpring(state, holdAnchor, holdBlock, [0, 0, 0], [0, 0, 0], STIFFNESS, REST);
        // chain rig: static — dyn — dyn — static with OFFSET anchors, so the links rotate as it sags (the
        // angular spring stamp). It's symmetric about x = 6.25, so correct physics keeps the mirror.
        const cy = 12;
        const chainL = box(state, [4, cy, SPRING_Z], [0.5, 0.5, 0.5], 0, STATIC);
        const cm1 = box(state, [5.5, cy, SPRING_Z], [0.5, 0.5, 0.5], 1, [0.5, 0.55, 0.85]);
        const cm2 = box(state, [7, cy, SPRING_Z], [0.5, 0.5, 0.5], 1, [0.5, 0.55, 0.85]);
        const chainR = box(state, [8.5, cy, SPRING_Z], [0.5, 0.5, 0.5], 0, STATIC);
        chainDyn = [cm1, cm2];
        const o: Vec3 = [0.5, 0, 0];
        const oNeg: Vec3 = [-0.5, 0, 0];
        addSpring(state, chainL, cm1, o, oNeg, STIFFNESS, 0.5);
        addSpring(state, cm1, cm2, o, oNeg, STIFFNESS, 0.5);
        addSpring(state, cm2, chainR, o, oNeg, STIFFNESS, 0.5);

        // ── fixed cantilever (z = 0): a static anchor + 3 dynamic boxes pinned horizontally by FIXED joints;
        //    the angular lock holds the row straight against gravity ──
        const cx = -5;
        const cAnchor = box(state, [cx, H, 0], [0.3, 0.3, 0.3], 0, STATIC);
        const c1 = box(state, [cx + SPAN, H, 0], [0.3, 0.3, 0.3], 1, [0.85, 0.6, 0.35]);
        const c2 = box(state, [cx + 2 * SPAN, H, 0], [0.3, 0.3, 0.3], 1, [0.8, 0.55, 0.4]);
        const c3 = box(state, [cx + 3 * SPAN, H, 0], [0.3, 0.3, 0.3], 1, [0.75, 0.5, 0.45]);
        cantBoxes = [c1, c2, c3];
        const fixed = Number.POSITIVE_INFINITY;
        const armHalf: Vec3 = [SPAN / 2, 0, 0];
        const armHalfNeg: Vec3 = [-SPAN / 2, 0, 0];
        addJoint(state, cAnchor, c1, armHalf, armHalfNeg, fixed);
        addJoint(state, c1, c2, armHalf, armHalfNeg, fixed);
        addJoint(state, c2, c3, armHalf, armHalfNeg, fixed);

        // ── spherical pendulum (z = 0): a static pivot + a dynamic bob released at θ0; the bob's local anchor
        //    rB = [0, ARM, 0] lands on the pivot (coincident), the spherical joint leaves rotation free → it swings ──
        const px = 1;
        const pivot = box(state, [px, H, 0], [0.3, 0.3, 0.3], 0, STATIC);
        pendBob = box(
            state,
            [px + ARM * Math.sin(THETA0), H - ARM * Math.cos(THETA0), 0],
            [0.3, 0.3, 0.3],
            1,
            [0.4, 0.75, 0.5],
            qz(THETA0),
        );
        addJoint(state, pivot, pendBob, [0, 0, 0], [0, ARM, 0]); // spherical (default)

        // ── recycle rig (z = 0): a box held DROP below a static anchor by a spherical joint; bumping its
        //    version mid-run must DEACTIVATE the joint (a despawned-then-recycled endpoint must not realias) ──
        const rx = 7;
        const rAnchor = box(state, [rx, H, 0], [0.3, 0.3, 0.3], 0, STATIC);
        recycleBox = box(state, [rx, H - DROP, 0], [0.3, 0.3, 0.3], 1, [0.5, 0.55, 0.85]);
        addJoint(state, rAnchor, recycleBox, [0, 0, 0], [0, DROP, 0]);

        // ── construction guard (z = 0): a joint with GROSSLY non-coincident anchors (~4.2 m apart). jointInit
        //    must REJECT it (deactivate + counters[2]) rather than let the rigid pin yank the box; it free-falls ──
        const gx = 4;
        const gAnchor = box(state, [gx, H, 0], [0.3, 0.3, 0.3], 0, [0.5, 0.3, 0.3]);
        guardBox = box(state, [gx, H - 3, 0], [0.3, 0.3, 0.3], 1, [0.85, 0.4, 0.4]);
        addJoint(state, gAnchor, guardBox, [0, 0, 0], [3, 0, 0]); // anchors ~4.2 m apart → rejected

        // ── both-static guard (z = 0): a joint between TWO static bodies — no dynamic body can resolve it, so
        //    its dual would ramp unbounded; jointInit must REJECT it (deactivate + counters[1], a per-frame gauge) ──
        const sx = -6;
        const sA = box(state, [sx, 4, 0], [0.3, 0.3, 0.3], 0, [0.45, 0.38, 0.32]);
        const sB = box(state, [sx + 0.7, 4, 0], [0.3, 0.3, 0.3], 0, [0.45, 0.38, 0.32]);
        addJoint(state, sA, sB, [0.35, 0, 0], [-0.35, 0, 0]); // both static → counters[1]++

        // ── rope (z = 0): a static mount + a chain of THIN dynamic links on SPHERICAL joints, hung at a
        //    diagonal — flexible (rotation-free pins) + inextensible (rigid linear pins hold each segment) ──
        const ropeAngle = (35 * Math.PI) / 180;
        const dir: Vec3 = [Math.sin(ropeAngle), -Math.cos(ropeAngle), 0];
        const phi = Math.atan2(dir[1], dir[0]);
        const qDir = qz(phi);
        const pin = ROPE_SEG / 2;
        const linkHalf: Vec3 = [pin - 0.05, 0.06, 0.06]; // a thin bar aligned to the chain, a small end-gap
        let prev = box(state, [ROPE_X, H, 0], [0.15, 0.15, 0.15], 0, STATIC); // fixed mount
        ropeLinks = [];
        for (let i = 0; i < ROPE_LINKS; i++) {
            const d = ROPE_SEG * (i + 0.5);
            const link = box(
                state,
                [ROPE_X + d * dir[0], H + d * dir[1], 0],
                linkHalf,
                1,
                [0.55, 0.5 + 0.04 * i, 0.85 - 0.04 * i],
                qDir,
            );
            addJoint(state, prev, link, i === 0 ? [0, 0, 0] : [pin, 0, 0], [-pin, 0, 0]);
            ropeLinks.push(link);
            prev = link;
        }

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 7);
        Orbit.pitch.set(cam, Math.PI / 12);
        Orbit.distance.set(cam, 34);

        await frames(3);
        if (Avbd.step) {
            liveMirror = mirror(Avbd.step.bodies);
            colorMirror = mirror(Avbd.step.colors);
            countersMirror = mirror(Avbd.step.counters);
        }
        await frames(180); // settle the cantilever rigid, let the pendulum swing + the rope drape

        // capture the recycle box's HELD height, then bump its version → the joint deactivates → it free-falls
        if (liveMirror) {
            await settle(liveMirror);
            const snap = liveMirror.snapshot;
            if (snap) {
                const s = new Float32Array(snap.bytes);
                const cap = s.length / (BODY_VEC4 * 4);
                recycleHeldY = s[(0 * cap + recycleBox) * 4 + 1];
            }
        }
        Avbd.step?.recycleVersion(recycleBox);
        await frames(240); // the deactivated box free-falls well clear of the held pose (lands on the floor)

        return {
            state,
            dispose() {
                liveMirror?.dispose();
                colorMirror?.dispose();
                countersMirror?.dispose();
                liveMirror = null;
                colorMirror = null;
                countersMirror = null;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const checks: Check[] = [];
        checks.push(...(await liveGates()));
        checks.push(await singleStep());
        checks.push(await reauthor());
        checks.push(await sphereGrab());
        checks.push(await measured());
        return checks;
    },

    live(): string {
        if (!liveMirror?.snapshot) return "constraints — warming";
        const s = new Float32Array(liveMirror.snapshot.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const y = (eid: number): string => s[(0 * cap + eid) * 4 + 1].toFixed(2);
        return [
            "constraints — authored springs + joints",
            `hold y ${y(holdBlock)} (eq ${Y_EQ.toFixed(2)})  chain y ${chainDyn.map(y).join(" ")}`,
            `cantilever y ${cantBoxes.map(y).join(" ")} (held ${H})  pendulum y ${y(pendBob)}`,
            `rope tip y ${y(ropeLinks[ropeLinks.length - 1] ?? -1)}  recycle y ${y(recycleBox)}`,
        ].join("\n");
    },
};

// ── atoms ──────────────────────────────────────────────────────────────────

const dist = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const gpuPos = (s: Float32Array, i: number, cap: number): Vec3 => {
    const o = (0 * cap + i) * 4;
    return [s[o], s[o + 1], s[o + 2]];
};
const gpuVel = (s: Float32Array, i: number, cap: number): Vec3 => {
    const o = (6 * cap + i) * 4;
    return [s[o], s[o + 1], s[o + 2]];
};

// ── live structural invariants (read the settled GPU state through Mirror) ──

async function liveGates(): Promise<Check[]> {
    if (!liveMirror) return [{ name: "constraints", pass: false, detail: "no physics step" }];
    await settle(liveMirror);
    if (colorMirror) await settle(colorMirror);
    if (countersMirror) await settle(countersMirror);
    const snap = liveMirror.snapshot;
    if (!snap) return [{ name: "constraints", pass: false, detail: "no snapshot" }];
    const s = new Float32Array(snap.bytes);
    const cap = s.length / (BODY_VEC4 * 4);
    const pos = (eid: number): Vec3 => gpuPos(s, eid, cap);
    const speed = (eid: number): number => Math.hypot(...gpuVel(s, eid, cap));
    const finite = (eid: number): boolean => pos(eid).every(Number.isFinite);
    const colors = colorMirror?.snapshot ? new Uint32Array(colorMirror.snapshot.bytes) : null;
    const counters = countersMirror?.snapshot
        ? new Uint32Array(countersMirror.snapshot.bytes)
        : null;
    const checks: Check[] = [];

    // hold: the block placed at y_eq holds the mg/k fixed point. A wrong spring (sign/scale) moves the
    // equilibrium, so the block drifts O(0.1+ m); 0.05 discriminates it.
    const hy = pos(holdBlock)[1];
    checks.push({
        name: "spring hold (the mg/k equilibrium)",
        pass: finite(holdBlock) && Math.abs(hy - Y_EQ) < 0.05 && speed(holdBlock) < 0.05,
        detail: `y ${hy.toFixed(4)} vs y_eq ${Y_EQ.toFixed(2)} (Δ ${Math.abs(hy - Y_EQ).toExponential(2)}), speed ${speed(holdBlock).toFixed(3)} m/s`,
    });

    // chain mirror symmetry (the angular spring stamp): the chain is symmetric about its midpoint, so correct
    // physics keeps cm2.qz = −cm1.qz — a sign-flipped torque breaks it to O(qz)≈0.2. finite+bounded + the
    // center-anchored hold don't reach the offset-anchor torque; this does.
    const qzLane = (eid: number): number => s[(1 * cap + eid) * 4 + 2];
    const [z1, z2] = chainDyn.map(qzLane);
    const asym = Math.abs(z1 + z2);
    const tilted = Math.abs(z1) > 0.05; // the links actually rotated (else the gate is vacuous)
    checks.push({
        name: "spring chain mirror symmetry (cm2.qz = −cm1.qz — the angular stamp)",
        pass: chainDyn.every(finite) && tilted && asym < 1e-3,
        detail: `cm1.qz ${z1.toFixed(4)}, cm2.qz ${z2.toFixed(4)} — |z1+z2| ${asym.toExponential(2)}, tilt ${Math.abs(z1).toFixed(3)}`,
    });

    // fixed cantilever: the angular lock holds the 3 boxes near their start height. A sag/collapse = a broken
    // angular stamp / rigid pin; the AL residual + GPU f32 keep it within a few cm, so 0.15 m discriminates.
    const cantHeld = cantBoxes.every((eid) => finite(eid) && Math.abs(pos(eid)[1] - H) < 0.15);
    checks.push({
        name: "fixed cantilever held straight (the angular lock vs gravity)",
        pass: cantHeld,
        detail: `box y [${cantBoxes.map((eid) => pos(eid)[1].toFixed(3)).join(", ")}] (start ${H})`,
    });

    // spherical pendulum: the rigid linear pin holds the bob at the arm length while rotation stays free — the
    // bob hangs below the pivot at radius ARM and swings (bounded).
    const pivotPos: Vec3 = [1, H, 0];
    const bp = pos(pendBob);
    const armLen = dist(bp, pivotPos);
    checks.push({
        name: "spherical pendulum on a held arm (the linear pin, rotation free)",
        pass: finite(pendBob) && bp[1] < H && Math.abs(armLen - ARM) < 0.1 && Math.abs(bp[2]) < 0.2,
        detail: `bob ${bp.map((v) => v.toFixed(2)).join(", ")}, arm ${armLen.toFixed(3)} (rest ${ARM}), below pivot ${bp[1] < H}`,
    });

    // rope: a spherical chain drapes — FLEXIBLE (hangs well below the mount) + INEXTENSIBLE (the rigid pins
    // cap the straight-line reach at the total link length; an extensible/exploding rope exceeds it).
    const ropeAnchor: Vec3 = [ROPE_X, H, 0];
    const tip = pos(ropeLinks[ropeLinks.length - 1]);
    const reach = dist(tip, ropeAnchor);
    const maxLen = ROPE_LINKS * ROPE_SEG; // 5.6 — the inextensible upper bound
    checks.push({
        name: "rope (spherical chain drapes — flexible + inextensible)",
        pass: ropeLinks.every(finite) && reach < maxLen + 0.3 && tip[1] < H - 2,
        detail: `tip ${tip.map((v) => v.toFixed(2)).join(", ")}, reach ${reach.toFixed(2)} (≤ ${maxLen.toFixed(1)}), hangs below ${tip[1] < H - 2}`,
    });

    // recycle-version realias: bumping the box's version deactivated its joint, so it free-fell from its held
    // pose. A still-held box (joint ignored the version) is RIGID at recycleHeldY (Δ ≈ 0); a freed box falls
    // many metres. 3 m is between the held jitter and the multi-metre free-fall, so it discriminates robustly.
    const fellY = pos(recycleBox)[1];
    checks.push({
        name: "recycle-version deactivates the joint (the realias guard)",
        pass: finite(recycleBox) && Number.isFinite(recycleHeldY) && fellY < recycleHeldY - 3,
        detail: `held y ${recycleHeldY.toFixed(2)} → dropped to ${fellY.toFixed(2)} (Δ ${(recycleHeldY - fellY).toFixed(2)} m)`,
    });

    // construction guard: the grossly non-coincident joint was rejected by jointInit, so its box is a free
    // body that dropped to the floor. A guard that let the rigid pin act would yank/explode it.
    const guardY = pos(guardBox)[1];
    checks.push({
        name: "construction guard rejects a non-coincident joint (the GPU analog of joint()'s throw)",
        pass: finite(guardBox) && guardY < H - 3 - 5,
        detail: `box y ${guardY.toFixed(2)} (start ${H - 3}, dropped to the floor — the bad joint deactivated)`,
    });

    // coloring split (the joint hard-conflict repair): the cantilever's dynamic–dynamic pairs (c1–c2, c2–c3)
    // and every adjacent rope link pair must get different colors — a hard constraint degrading to same-color
    // Jacobi destabilizes. GPU == oracle can't see this (the oracle runs the GPU's coloring), so it's observable.
    if (colors) {
        const [k1, k2, k3] = cantBoxes.map((eid) => colors[eid]);
        const cantSplit = k1 !== k2 && k2 !== k3;
        let ropeSplit = true;
        for (let i = 1; i < ropeLinks.length; i++)
            if (colors[ropeLinks[i]] === colors[ropeLinks[i - 1]]) ropeSplit = false;
        checks.push({
            name: "coloring split (dynamic joint pairs differ — the hard-conflict repair)",
            pass: cantSplit && ropeSplit,
            detail: `cantilever [${k1}, ${k2}, ${k3}], rope [${ropeLinks.map((e) => colors[e]).join(",")}] (adjacent must differ)`,
        });
    }

    // both-static guard: the joint between two static bodies was rejected by jointInit (its dual would ramp
    // unbounded — energy injection). Both ends are static so there's no body-motion tell; the signal is
    // counters[1], which the guard re-bumps EVERY frame, so a direct read is reliable. 0 = the guard never fired.
    checks.push({
        name: "both-static joint rejected loudly (the energy-injection guard, counters[1])",
        pass: counters !== null && counters[1] >= 1,
        detail: `counters[1] = ${counters ? counters[1] : "n/a"} (a joint between two non-dynamic bodies is deactivated each frame)`,
    });

    return checks;
}

// ── seeded single-step GPU == oracle (a fresh PhysicsStep, no live-sim interference) ──

const GATE_CAP = 8;
const POS_TOL = 2e-4; // one cold step of ITERS sweeps, f32 (GPU) vs f64 (oracle)
const VEL_TOL = 2e-2; // ≈ POS_TOL / DT

// the dense readback layout: [bodies SoA | colors]. (the spring is stateless + the joint cold-inits, so one
// step is memoryless — the gym pile's single-step argument.)
const readBytes = (cap: number): number => cap * BODY_VEC4 * 16 + cap * 4;

const bootstrap = (n: number): Uint32Array<ArrayBuffer> => {
    const a = new Uint32Array(n);
    for (let i = 0; i < n; i++) a[i] = i;
    return a;
};

// write one oracle body into the dense bodies SoA at dense index i (`arr[(col*cap + i)*4 + lane]`) — the
// columns the inertial + primal + constraint stamps read. Mirrors step.ts seedBody.
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
    w(2, b.posLin[0], b.posLin[1], b.posLin[2], 0); // inertialLin (the inertial pass overwrites)
    w(3, b.posAng[0], b.posAng[1], b.posAng[2], b.posAng[3]); // inertialAng
    w(4, b.posLin[0], b.posLin[1], b.posLin[2], 0); // initialLin
    w(5, b.posAng[0], b.posAng[1], b.posAng[2], b.posAng[3]); // initialAng
    w(6, b.velLin[0], b.velLin[1], b.velLin[2], 0); // velLin
    w(7, b.velAng[0], b.velAng[1], b.velAng[2], 0); // velAng
    w(8, b.prevVelLin[0], b.prevVelLin[1], b.prevVelLin[2], 0); // prevVelLin
    w(9, b.moment[0], b.moment[1], b.moment[2], b.mass); // moment.xyz / mass.w
    w(10, b.size[0] / 2, b.size[1] / 2, b.size[2] / 2, b.friction); // halfExtents.xyz / friction.w
    w(11, 0, 0, 0, 0); // B_ROUND: shape/radius/hullId 0 (box); .w lane unused
}

// the f64 oracle: a spring + a joint rig from the same bodies, ONE cold step with the GPU's coloring. The
// dense → eid identity map means a SpringDef/JointDef index is the oracle body index too.
function oracle(
    bodies: OracleBody[],
    springs: SpringDef[],
    joints: JointDef[],
    colors: number[],
): OracleBody[] {
    const s = makeSolver(bodies, {
        penaltyStiffness: PENALTY_MIN,
        betaLin: BETA_LIN,
        betaAng: BETA_ANG,
        gamma: GAMMA,
        iterations: ITERS,
        alpha: ALPHA,
        dt: DT,
        gravity: G,
    });
    for (const sp of springs)
        s.springs.push(
            oracleSpring(
                bodies[sp.a],
                bodies[sp.b],
                sp.rA as Vec3,
                sp.rB as Vec3,
                sp.stiffness,
                sp.rest,
            ),
        );
    for (const jt of joints)
        s.joints.push(
            oracleJoint(
                bodies[jt.a],
                bodies[jt.b],
                jt.rA as Vec3,
                jt.rB as Vec3,
                Number.POSITIVE_INFINITY,
                jt.stiffnessAng ?? 0,
            ),
        );
    oracleStep(s, { kind: "colored", colors });
    return bodies;
}

const oracleColors = (raw: Uint32Array, n: number): number[] =>
    Array.from({ length: n }, (_, i) => (raw[i] === 0xffffffff ? 0 : raw[i]));

// seed a spring rig (anchor + displaced block) + a joint rig (pivot + bob at θ0, anchors coincident) into a
// fresh PhysicsStep, step ONE cold frame, and compare the block + bob pose/velocity to the f64 oracle.
async function singleStep(): Promise<Check> {
    const device = Compute.device;
    const phys = await PhysicsStep.create(device, GATE_CAP, GATE_CAP);
    const read = device.createBuffer({
        size: readBytes(GATE_CAP),
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // dense indices: 0 anchor (static), 1 block (spring), 2 pivot (static), 3 bob (joint). The block sits
    // above equilibrium (spring compressed → real force); the bob's local anchor lands on the pivot.
    const scene = (): OracleBody[] => [
        makeBody([1, 1, 1], 0, 0.5, [-3, 10, 0]),
        makeBody([2, 2, 2], 8, 0.5, [-3, 8, 0]),
        makeBody([0.6, 0.6, 0.6], 0, 0.5, [1, 10, 0]),
        makeBody(
            [0.6, 0.6, 0.6],
            1,
            0.5,
            [1 + ARM * Math.sin(THETA0), 10 - ARM * Math.cos(THETA0), 0],
            [0, 0, 0],
            qz(THETA0),
        ),
    ];
    const springs: SpringDef[] = [
        { a: 0, b: 1, rA: [0, 0, 0], rB: [0, 0, 0], stiffness: STIFFNESS, rest: REST },
    ];
    const joints: JointDef[] = [{ a: 2, b: 3, rA: [0, 0, 0], rB: [0, ARM, 0] }];
    try {
        const bodies = scene();
        const arr = new Float32Array(GATE_CAP * BODY_VEC4 * 4);
        for (let i = 0; i < bodies.length; i++) seedBody(arr, i, bodies[i], GATE_CAP);
        device.queue.writeBuffer(phys.bodies, 0, arr);
        device.queue.writeBuffer(phys.colors, 0, bootstrap(GATE_CAP));
        phys.configure(cfg);
        phys.setSprings(springs);
        phys.setJoints(joints);
        phys.gateSetCount(bodies.length);
        phys.cold();
        const enc = device.createCommandEncoder();
        phys.record(enc);
        const Full = GATE_CAP * BODY_VEC4 * 16;
        enc.copyBufferToBuffer(phys.bodies, 0, read, 0, Full);
        enc.copyBufferToBuffer(phys.colors, 0, read, Full, GATE_CAP * 4);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ, 0, readBytes(GATE_CAP));
        const mapped = read.getMappedRange(0, readBytes(GATE_CAP));
        const state = new Float32Array(mapped.slice(0, Full));
        const colors = new Uint32Array(mapped.slice(Full, readBytes(GATE_CAP)));
        read.unmap();

        const ora = oracle(scene(), springs, joints, oracleColors(colors, bodies.length));
        let maxPos = 0;
        let maxVel = 0;
        for (const i of [1, 3]) {
            maxPos = Math.max(maxPos, dist(gpuPos(state, i, GATE_CAP), ora[i].posLin as Vec3));
            maxVel = Math.max(maxVel, dist(gpuVel(state, i, GATE_CAP), ora[i].velLin as Vec3));
        }
        return {
            name: "single-step GPU == oracle (spring mg/k pull + joint pin)",
            pass: maxPos < POS_TOL && maxVel < VEL_TOL,
            detail: `pos err ${maxPos.toExponential(2)} (tol ${POS_TOL}), vel err ${maxVel.toExponential(2)} (tol ${VEL_TOL})`,
        };
    } finally {
        phys.destroy();
        read.destroy();
    }
}

// re-author under load — the sandbox grab/release flow: a settled rigid-pin rope is yanked hard by an
// appended world-anchor grab joint (pins stretch past the reach guard), then released by re-uploading
// the SAME rope defs. setJoints keeps unchanged slots' live records, so the release must neither re-run
// the construction guards (a fresh re-init re-judges the stretched pins and REJECTS them — the chain
// disconnects, the sandbox "collapse on release") nor zero the warmstart. The act lanes are the direct
// observable: every rope joint must still be active after the release.
async function reauthor(): Promise<Check> {
    const device = Compute.device;
    const phys = await PhysicsStep.create(device, 32, 32);
    const Cap = 32;
    const Full = Cap * BODY_VEC4 * 16;
    // the sandbox's weighted rope verbatim: 18 thin light links + a 50 kg weight — the mass ratio that
    // honestly stretches the top pins past the reach guard under a hard pull
    const Links = 18;
    const Seg = 0.21;
    const Pin = Seg / 2;
    const Top = 12;
    const Wr = 0.4; // weight half-size
    const Joints = Links + 1; // link pins + the weight pin
    const Recs = Joints * JOINT_REC_VEC4 * 16; // the rope joints' records (act lanes)
    const read = device.createBuffer({
        size: 2 * Full + Recs,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    // dense/eid 0 = static mount at the top pin; 1..Links = links; Links+1 = the weight
    const weightY = Top - Seg * Links - Wr;
    const scene = (): OracleBody[] => {
        const bodies = [makeBody([0.3, 0.3, 0.3], 0, 0.5, [0, Top, 0])];
        for (let i = 0; i < Links; i++) {
            bodies.push(makeBody([0.1, 0.17, 0.1], 0.2, 0.5, [0, Top - Seg * i - Pin, 0]));
        }
        bodies.push(makeBody([2 * Wr, 2 * Wr, 2 * Wr], 50, 0.5, [0, weightY, 0]));
        return bodies;
    };
    const rope: JointDef[] = [{ a: 0, b: 1, rA: [0, 0, 0], rB: [0, Pin, 0] }];
    for (let i = 1; i < Links; i++) {
        rope.push({ a: i, b: i + 1, rA: [0, -Pin, 0], rB: [0, Pin, 0] });
    }
    rope.push({ a: Links, b: Links + 1, rA: [0, -Pin, 0], rB: [0, Wr, 0] });
    const weight = Links + 1;
    try {
        const bodies = scene();
        const arr = new Float32Array(Cap * BODY_VEC4 * 4);
        for (let i = 0; i < bodies.length; i++) seedBody(arr, i, bodies[i], Cap);
        device.queue.writeBuffer(phys.bodies, 0, arr);
        device.queue.writeBuffer(phys.colors, 0, bootstrap(Cap));
        phys.configure(cfg);
        phys.setJoints(rope);
        phys.gateSetCount(bodies.length);
        phys.cold();
        const run = (steps: number): void => {
            const enc = device.createCommandEncoder();
            for (let i = 0; i < steps; i++) phys.record(enc);
            device.queue.submit([enc.finish()]);
        };
        run(60); // settle (a vertical chain hangs at its spawn equilibrium)
        // grab the tip (the sandbox gun: a soft world-anchor joint appended after the rope)
        phys.setJoints([
            ...rope,
            { a: WORLD, b: weight, rA: [0, weightY, 0], rB: [0, 0, 0], stiffnessLin: 5000 },
        ]);
        // yank hard sideways — the anchor races away faster than the chain can follow, stretching pins
        for (let i = 1; i <= 6; i++) {
            phys.setJointAnchor(rope.length, i * 2.5, weightY + i * 0.4, 0);
            run(10);
        }
        const encPre = device.createCommandEncoder();
        encPre.copyBufferToBuffer(phys.bodies, 0, read, Full, Full); // the held pose, pre-release
        device.queue.submit([encPre.finish()]);
        phys.setJoints(rope); // release — the grab pattern's restore
        run(10);
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(phys.bodies, 0, read, 0, Full); // just past the release
        enc.copyBufferToBuffer(phys.jointRecords, 0, read, 2 * Full, Recs);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ, 0, 2 * Full + Recs);
        const mapped = read.getMappedRange(0, 2 * Full + Recs);
        const state = new Float32Array(mapped.slice(0, Full));
        const pre = new Float32Array(mapped.slice(Full, 2 * Full));
        const recs = new Uint32Array(mapped.slice(2 * Full, 2 * Full + Recs));
        read.unmap();

        // three observables of the kept records: every rope joint still active (a fresh re-init re-runs
        // the construction guards — a rejected pin reads act 0), the chain doesn't slacken at the release
        // (a λ/penalty reset drops the 50 kg weight while the penalties re-ramp — the sandbox's visible
        // "collapse on release"), and the chain stays intact under the loaded re-upload.
        const acts: number[] = [];
        for (let i = 0; i < Joints; i++) acts.push(recs[i * JOINT_REC_VEC4 * 4 + 13]);
        const live = acts.every((a) => a === 1);
        const reachPre = dist(gpuPos(pre, weight, Cap), [0, Top, 0]);
        const reachPost = dist(gpuPos(state, weight, Cap), [0, Top, 0]);
        const maxLen = Links * Seg + 2 * Wr;
        return {
            name: "re-author under load keeps live joints (the grab release — no guard re-run, no λ reset)",
            pass: live && reachPost <= reachPre + 0.05 && reachPost < maxLen + 0.5,
            detail: `acts [${acts.join(",")}], weight reach ${reachPre.toFixed(2)} → ${reachPost.toFixed(2)} past the release (rope ${maxLen.toFixed(1)})`,
        };
    } finally {
        phys.destroy();
        read.destroy();
    }
}

// a world-anchor grab on a SPHERE: a rounded body carries its size in bRadius with halfExtents (0,0,0),
// so a construction-guard reach computed off halfExtents alone is ~zero — every sphere grab whose anchor
// sits even slightly off the centre (the gun's reel between grab and init) is rejected, and spheres read
// as ungrabbable. The guard's reach must include the radius: an anchor offset inside the sphere's
// surface initializes (act 1) and tugs the body toward it.
async function sphereGrab(): Promise<Check> {
    const device = Compute.device;
    const phys = await PhysicsStep.create(device, GATE_CAP, GATE_CAP);
    const Full = GATE_CAP * BODY_VEC4 * 16;
    const RecBytes = JOINT_REC_VEC4 * 16;
    const read = device.createBuffer({
        size: Full + RecBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const R = 0.4;
    const Mass = 12;
    try {
        // seed the box columns, then patch the rounded-shape columns: moment ⅖mr², halfExtents (0,0,0),
        // B_ROUND = (kind Sphere, radius) — the authored shape the gate's direct seed path bypasses
        const arr = new Float32Array(GATE_CAP * BODY_VEC4 * 4);
        seedBody(arr, 0, makeBody([2 * R, 2 * R, 2 * R], Mass, 0.5, [0, 10, 0]), GATE_CAP);
        const m = 0.4 * Mass * R * R;
        const patch = (col: number, v: Float32Array): void => {
            arr.set(v, (col * GATE_CAP + 0) * 4);
        };
        patch(9, new Float32Array([m, m, m, Mass]));
        patch(10, new Float32Array([0, 0, 0, 0.5]));
        const round = new Float32Array(4);
        new Uint32Array(round.buffer)[0] = 1; // ShapeKind.Sphere
        round[1] = R;
        patch(11, round);
        device.queue.writeBuffer(phys.bodies, 0, arr);
        device.queue.writeBuffer(phys.colors, 0, bootstrap(GATE_CAP));
        phys.configure(cfg);
        // the grab: a soft world anchor 0.2 m off the centre — INSIDE the sphere's surface (reach must
        // count the radius; off halfExtents alone the reach is ~1 cm and this rejects)
        phys.setJoints([{ a: WORLD, b: 0, rA: [0.2, 10, 0], rB: [0, 0, 0], stiffnessLin: 5000 }]);
        phys.gateSetCount(1);
        phys.cold();
        const enc = device.createCommandEncoder();
        for (let s = 0; s < 30; s++) phys.record(enc);
        enc.copyBufferToBuffer(phys.bodies, 0, read, 0, Full);
        enc.copyBufferToBuffer(phys.jointRecords, 0, read, Full, RecBytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ, 0, Full + RecBytes);
        const mapped = read.getMappedRange(0, Full + RecBytes);
        const state = new Float32Array(mapped.slice(0, Full));
        const recs = new Uint32Array(mapped.slice(Full, Full + RecBytes));
        read.unmap();

        const act = recs[13];
        const pos = gpuPos(state, 0, GATE_CAP);
        // held: the grab arrests the fall and tugs +x toward the anchor; rejected, the sphere free-falls
        return {
            name: "sphere grab initializes (the guard's reach counts the rounded radius)",
            pass: act === 1 && pos[0] > 0.02 && pos[1] > 8,
            detail: `act ${act}, sphere ${pos[0].toFixed(3)}, ${pos[1].toFixed(2)} (start 0, 10; free-fall y ≈ 8.75)`,
        };
    } finally {
        phys.destroy();
        read.destroy();
    }
}

// ── perf reporter ────────────────────────────────────────────────────────────

const STEP_PASSES = [
    "phys:pack",
    "phys:aabb",
    "bvh:sort",
    "bvh:build",
    "phys:broadphase",
    "phys:collide",
    "phys:csr",
    "phys:coloring",
    "phys:joint",
    "phys:inertial",
    "phys:primal",
    "phys:dual",
    "phys:velocity",
    "phys:compose",
] as const;

// always-pass perf REPORTER (not a correctness gate): the per-step spans (incl. the joint pass) as structured
// `Check.data`, the seam scripts/physics-bench.ts reads. phys:joint is the constraint scenario's headline cost.
async function measured(): Promise<Check> {
    const get = (name: string): number => Profile.gpu.get(name) ?? 0;
    const data: Record<string, number> = {};
    for (const n of STEP_PASSES) data[n] = get(n);
    const full = STEP_PASSES.reduce((sum, n) => sum + data[n], 0);
    data.dispatchedColors = Avbd.step?.dispatchedColors ?? 0;
    data.bytes = Avbd.step?.bytes ?? 0;
    const resolved = get("phys:primal") > 0;
    const label = (n: string): string => n.replace("phys:", "").replace("bvh:", "bvh.");
    const parts = STEP_PASSES.map((n) => `${label(n)} ${data[n].toFixed(3)}`).join(" · ");
    return {
        name: "measured (solver spans)",
        pass: true,
        detail: resolved ? `step ~${full.toFixed(3)} ms = ${parts} ms` : "no solver spans resolved",
        data,
    };
}

register(scenario);
