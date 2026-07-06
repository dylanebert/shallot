import {
    AmbientLight,
    Body,
    Camera,
    CameraMode,
    Color,
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
    PhysicsPlugin,
    type Plugin,
    RenderPlugin,
    run,
    Sear,
    SearPlugin,
    Shadow,
    ShapeKind,
    SlabPlugin,
    type State,
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    BODY_VEC4,
    type JointDef,
    PENALTY_MIN,
    Physics,
    StepSystem,
} from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// spikecatch — the spindle catch stripped to ONE pair (games/spindle roadmap §1: rope sheds off the thin
// spikes). The whole scene: a thin horizontal SPIKE (a thin rod along X) with a sphere-link ROPE draped
// PERPENDICULAR over it. The rod BOBS up and down, its speed ramped SMOOTHLY (a gentle frequency chirp — low
// jerk, no sudden toss), so the catch's behavior is read as the motion intensifies.
//
// Rope validity (the load-bearing detail): shallot has NO collision filtering between jointed bodies, so the
// spindle's overlapping-capsule "hole-free tube" (core+radius > link spacing) is ALSO interpenetrating —
// every pinned pair has a contact shoving it apart while the joint holds it together = injected energy that
// tosses the rope. Here the rope is SPHERE links spaced just OVER 2·radius: tightly packed (near-continuous)
// but NON-overlapping and bend-invariant (spheres can't overlap under any bend if centers stay > 2r apart),
// so the only contacts are with the rod — a valid rope. `requireValidRope` throws LOUDLY if the spacing
// would make adjacent links overlap (the energy-injecting construction the engine can't yet filter). A LONG
// rope drapes over the LOW spike so its ends rest on the floor — the grounded ends anchor it against sliding
// off the side.
//
//   kinematic — spike posed by setKinematic (identity quat ⇒ never rotates), exact smooth bob.
//   dynamic   — a REAL dynamic body driven by a feed-forward velocity (smooth, no jerk) + a soft gravity
//               trim, held level by an angular-velocity servo to identity. It carries the rope by contact,
//               the rope can push back — the game's compliant-spike counterpart to the kinematic forced pose.

const G = -10;
const DT = 1 / 60;

const ROD_HALF_X = 0.35; // rod half-length along X
// the spike sits LOW so a long rope draped over it reaches the floor on both sides — the grounded ends
// anchor the rope (friction) so it can't slide off the rod's side. With `links` ≥ ~14 each arm (≈ links·SEG/2)
// exceeds this height and the ends rest flat on the ground.
const ROD_BASE_Y = 0.55;
const ROD_MASS = 2.0; // dynamic-mode rod mass (heavy enough that the rope load barely perturbs the driven bob)

// the rope: SPHERE links, spaced just over 2·radius — tightly packed, NON-overlapping, bend-invariant.
const ROPE_RADIUS = 0.05;
const ROPE_SEG = 0.105; // > 2·radius (0.10) ⇒ a 0.005 m gap, no inter-link collision under any bend
let ropeLinks = 9; // set from the `links` param — 1 isolates the bare carry (one sphere on the rod, no drape)

// the smooth bob: y = base + AMP·sin(phase), phase = 2π·(F0·t + ½·FRATE·t²). FRATE is GENTLE so the speed
// (and jerk) ramp slowly — no sudden jerk that tosses the rope. Peak vertical speed AMP·2π·f climbs as f does.
const AMP = 0.09;
const F0 = 0.25;
const FRATE = 0.4;
const RUN_S = 9.0;
const TAIL_S = 0.6; // hold the rod still after the bob so the final state is stable (and matches a screenshot)
const SETTLE = 1.6; // seconds for the long rope to drape + its ends to settle on the floor before the bob

let spikeEid = -1;
let kinematic = false;
let rodThick = 0.04;
let bodyMirror: Mirror | null = null;
let ropeBodies: number[] = [];
let result: {
    draped: number; // how far below the rod top the rope's lowest link hung after settling (a real drape > 0)
    earlySettle: number; // max link speed early in the still-spike settle window
    lateSettle: number; // …and late. A valid rope DECAYS (late ≤ early); energy injection GROWS (late ≫ early)
    held: boolean; // the rope was still up on the rod at the end of the bob (vs shed to the floor)
    endRopeY: number;
    maxRodTilt: number; // max |rod q.xyz| — ~0 means the rod stayed level (no rotation dumping the rope)
    finite: boolean;
} | null = null;

function bobY(t: number): number {
    const phase = 2 * Math.PI * (F0 * t + 0.5 * FRATE * t * t);
    return ROD_BASE_Y + AMP * Math.sin(phase);
}
// analytic vertical velocity of the bob — the dynamic drive's feed-forward (smooth, no jerk).
function bobVel(t: number): number {
    const phase = 2 * Math.PI * (F0 * t + 0.5 * FRATE * t * t);
    return AMP * Math.cos(phase) * 2 * Math.PI * (F0 + FRATE * t);
}

const DrivePlugin: Plugin = {
    name: "SpikeCatchDrive",
    systems: [
        {
            name: "spikecatch-drive",
            group: "fixed",
            before: [StepSystem],
            update(state: State) {
                if (spikeEid < 0 || !Physics.step) return;
                const t = state.time.elapsed - SETTLE;
                const target = t > 0 && t < RUN_S ? bobY(t) : ROD_BASE_Y; // hold base before + after the bob
                if (kinematic) {
                    Physics.step.setKinematic(spikeEid, [0, target, 0], [0, 0, 0, 1]); // identity ⇒ no rotation
                } else {
                    // dynamic: feed-forward velocity (the analytic bob derivative) drives the SAME smooth
                    // motion with no deadbeat jerk; a soft position trim counters gravity drift. A REAL dynamic
                    // body — carries by contact, the rope can push back — the fair counterpart to the kinematic
                    // forced pose. The rod is held LEVEL by an angular-velocity SERVO that drives the
                    // orientation back to identity each tick (the spindle's drive approach): an asymmetric rope
                    // load torques the rod, and a one-shot setAngularVelocity(0) only resets next tick's start
                    // — it doesn't undo this tick's contact tilt, so the tilt accumulates and dumps the rope.
                    // Driving ω = −errorVec/dt (errorVec = 2·sign(qw)·q.xyz, the small-angle orientation error)
                    // cancels the tilt deadbeat. Reads the live pose off the Mirror (1 frame stale, fine).
                    const vff = t > 0 && t < RUN_S ? bobVel(t) : 0;
                    const snap = bodyMirror?.snapshot;
                    let y = ROD_BASE_Y;
                    let qx = 0;
                    let qy = 0;
                    let qz = 0;
                    let qw = 1;
                    if (snap) {
                        const s = new Float32Array(snap.bytes);
                        const cap = s.length / (BODY_VEC4 * 4);
                        y = s[(0 * cap + spikeEid) * 4 + 1];
                        qx = s[(1 * cap + spikeEid) * 4];
                        qy = s[(1 * cap + spikeEid) * 4 + 1];
                        qz = s[(1 * cap + spikeEid) * 4 + 2];
                        qw = s[(1 * cap + spikeEid) * 4 + 3];
                    }
                    Physics.step.setVelocity(spikeEid, 0, vff + (target - y) / 0.15, 0); // ff + soft trim
                    const sgn = qw < 0 ? -1 : 1; // q and −q are the same rotation; pick the short way to identity
                    Physics.step.setAngularVelocity(
                        spikeEid,
                        (-2 * sgn * qx) / DT,
                        (-2 * sgn * qy) / DT,
                        (-2 * sgn * qz) / DT,
                    );
                }
            },
        } satisfies System,
    ],
};

function box(
    state: State,
    pos: readonly [number, number, number],
    half: readonly [number, number, number],
    mass: number,
    color: readonly [number, number, number],
    friction = 0.8,
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, friction);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// throw LOUDLY if a joint would connect two sphere links whose colliders OVERLAP — the energy-injecting
// construction (a contact and the joint fighting) the engine has no collision filter to suppress. A valid
// rope keeps centers > 2·radius apart.
function requireValidRope(seg: number, radius: number): void {
    if (seg <= 2 * radius) {
        throw new Error(
            `spikecatch: invalid rope — link spacing ${seg} ≤ 2·radius ${2 * radius}: adjacent colliders overlap, ` +
                `so each joint fights a contact (injected energy). Space links > 2·radius, or the engine must ` +
                `filter collision between jointed links.`,
        );
    }
}

const scenario: Scenario = {
    name: "spikecatch",
    params: [
        { key: "kinematic", type: "number", default: 0, min: 0, max: 1, step: 1 },
        { key: "thick", type: "number", default: 0.04, min: 0.01, max: 0.2, step: 0.01 }, // rod cross-section
        { key: "links", type: "number", default: 18, min: 1, max: 30, step: 1 }, // 1 = bare carry, no drape
        { key: "iters", type: "number", default: 10, min: 1, max: 30, step: 1 },
    ],

    async build(_canvas, p: Params) {
        result = null;
        ropeBodies = [];
        kinematic = (p.kinematic as number) > 0;
        rodThick = p.thick as number;
        ropeLinks = p.links as number;
        requireValidRope(ROPE_SEG, ROPE_RADIUS);

        const { state, dispose } = await run({
            defaults: false,
            capacity: 48,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PhysicsPlugin,
                DrivePlugin,
                PartPlugin,
                SearPlugin,
                GlazePlugin,
            ],
        });

        const amb = state.create();
        state.add(amb, AmbientLight);
        AmbientLight.color.set(amb, 0xd0dcec);
        AmbientLight.intensity.set(amb, 0.6);
        const sun = state.create();
        state.add(sun, DirectionalLight);
        DirectionalLight.color.set(sun, 0xfff4e0);
        DirectionalLight.intensity.set(sun, 1.4);
        DirectionalLight.direction.set(sun, -0.4, -1, -0.55, 0);
        state.add(sun, Shadow);
        Shadow.distance.set(sun, 30);

        Physics.step?.configure({
            dt: DT,
            gravity: G,
            alpha: 0.99,
            penalty: PENALTY_MIN,
            betaLin: 1e4,
            gamma: 0.999,
            iterations: p.iters as number,
            maxColors: 8,
        });

        box(state, [0, -0.5, 0], [20, 0.5, 20], 0, [0.28, 0.3, 0.34]); // floor, top at 0

        // the thin rod (dynamic mass, or kinematic = mass 0), along X, at the bob center height
        spikeEid = box(
            state,
            [0, ROD_BASE_Y, 0],
            [ROD_HALF_X, rodThick, rodThick],
            kinematic ? 0 : ROD_MASS,
            [0.8, 0.42, 0.16],
        );

        // the rope joints (dynamic mode drives the rod by velocity, no pins; kinematic by setKinematic).
        const joints: JointDef[] = [];

        // the rope: sphere links along Z, centered so the middle link crosses the rod at the origin and the
        // rest drapes down both ±Z sides. Spheres ⇒ no orientation, joints anchor at the midpoint between
        // centers (a spherical pin keeping them ROPE_SEG apart, free to bend).
        const sphereMesh = Meshes.id("sphere") ?? 0;
        const rodTop = ROD_BASE_Y + rodThick;
        let prev = -1;
        for (let i = 0; i < ropeLinks; i++) {
            const pz = (i - (ropeLinks - 1) / 2) * ROPE_SEG;
            const eid = state.create();
            state.add(eid, Body);
            Body.shape.set(eid, ShapeKind.Sphere);
            Body.pos.set(eid, 0, rodTop + ROPE_RADIUS + 0.01, pz, 0);
            Body.halfExtents.set(eid, 0, 0, 0, ROPE_RADIUS);
            Body.mass.set(eid, 0.2);
            Body.friction.set(eid, 0.8);
            state.add(eid, Part);
            Part.mesh.set(eid, sphereMesh);
            state.add(eid, Color);
            Color.rgba.set(eid, 0.72, 0.57, 0.35, 1);
            if (prev >= 0)
                joints.push({
                    a: prev,
                    b: eid,
                    rA: [0, 0, ROPE_SEG / 2],
                    rB: [0, 0, -ROPE_SEG / 2],
                });
            prev = eid;
            ropeBodies.push(eid);
        }
        Physics.step?.setJoints(joints);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 8);
        Orbit.distance.set(cam, 3.4);

        await frames(3);
        if (Physics.step) bodyMirror = mirror(Physics.step.bodies);

        const yOf = (s: Float32Array, cap: number, eid: number): number =>
            s[(0 * cap + eid) * 4 + 1];
        const speedOf = (s: Float32Array, cap: number, eid: number): number =>
            Math.hypot(
                s[(6 * cap + eid) * 4],
                s[(6 * cap + eid) * 4 + 1],
                s[(6 * cap + eid) * 4 + 2],
            );

        // loop on real SIM time (frames(1) is one rAF ≈ 240 fps headless, but the bob is clocked off
        // state.time.elapsed). Sample at ~120 Hz, tracking the metrics inline:
        //   settle energy — early vs late max link speed while the spike is STILL (a valid rope decays, an
        //     energy-injecting one grows); rod tilt — must stay ~level; drapeLow / endRopeY — the rope's
        //     lowest link as it settles, and its mean height at the end (held on the rod vs shed to the floor).
        let finite = true;
        let earlySettle = 0;
        let lateSettle = 0;
        let maxRodTilt = 0;
        let drapeLow = Number.POSITIVE_INFINITY;
        let endRopeY = 0;
        const targetT = SETTLE + RUN_S + TAIL_S;
        let lastRec = -1;
        while (state.time.elapsed < targetT) {
            await frames(1);
            const snap = bodyMirror?.snapshot;
            if (!snap) continue;
            const now = state.time.elapsed;
            if (now - lastRec < 1 / 120) continue;
            lastRec = now;
            const s = new Float32Array(snap.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            const tBob = now - SETTLE;
            maxRodTilt = Math.max(
                maxRodTilt,
                Math.hypot(
                    s[(1 * cap + spikeEid) * 4],
                    s[(1 * cap + spikeEid) * 4 + 1],
                    s[(1 * cap + spikeEid) * 4 + 2],
                ),
            );
            if (tBob > -SETTLE + 0.2 && tBob < -SETTLE + 0.45)
                for (const e of ropeBodies) earlySettle = Math.max(earlySettle, speedOf(s, cap, e));
            if (tBob > -0.25 && tBob < 0)
                for (const e of ropeBodies) lateSettle = Math.max(lateSettle, speedOf(s, cap, e));
            let meanY = 0;
            for (const e of ropeBodies) {
                const y = yOf(s, cap, e);
                if (!Number.isFinite(y)) finite = false;
                if (tBob > -0.15 && tBob < 0) drapeLow = Math.min(drapeLow, y); // lowest link just before the bob
                meanY += y;
            }
            endRopeY = meanY / ropeBodies.length; // last sample wins
        }
        const draped = ROD_BASE_Y + rodThick - drapeLow;

        result = {
            draped,
            earlySettle,
            lateSettle,
            held: endRopeY > 0.3,
            endRopeY,
            maxRodTilt,
            finite,
        };

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                spikeEid = -1;
                ropeBodies = [];
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const r = result;
        if (!r) return [{ name: "spikecatch", pass: false, detail: "no measurement" }];
        const mode = kinematic ? "kinematic" : "dynamic";
        const data = {
            draped: +r.draped.toFixed(3),
            earlySettle: +r.earlySettle.toFixed(3),
            lateSettle: +r.lateSettle.toFixed(3),
            endRopeY: +r.endRopeY.toFixed(3),
            held: r.held ? 1 : 0,
            thick: rodThick,
        };
        return [
            { name: "finite", pass: r.finite, detail: "rope stays finite (no NaN)" },
            {
                // the spike must stay roughly LEVEL — a gross tilt from an asymmetric rope load dumps the rope
                // off the side. The angular servo holds it to a small transient peak (~0.10 ≈ 12°) at which
                // the grounded rope demonstrably stays; gate at 0.15 (≈17°) for margin.
                name: "level-spike",
                pass: r.maxRodTilt < 0.15,
                detail: `[${mode}] max rod tilt |q.xyz| ${r.maxRodTilt.toFixed(3)} (< 0.15 ≈ 17° ⇒ stayed level)`,
            },
            {
                // a VALID rope's residual motion on a STILL spike decays (or holds); energy injection from
                // overlapping jointed colliders GROWS. So late settle speed must not exceed early by much.
                name: "valid-rope (energy non-growing)",
                pass: r.lateSettle <= r.earlySettle + 0.1,
                detail: `[${mode}] settle speed early ${data.earlySettle} → late ${data.lateSettle} m/s (late ≤ early+0.1 ⇒ no injection)`,
            },
            {
                name: "drapes",
                pass: r.draped > 3 * ROPE_RADIUS,
                detail: `[${mode}] draped ${data.draped} m below the rod top (> ${(3 * ROPE_RADIUS).toFixed(3)}) | thick ${rodThick}`,
            },
            {
                // DIAGNOSTIC (data, like kincarry) — the carry result: the rope ends up on the rod (held)
                // vs shed to the floor. Dynamic carries it; a kinematic spike loses it (kex backlog).
                name: "holds (diagnostic)",
                pass: true,
                detail: `[${mode}] held ${data.held} | endRopeY ${data.endRopeY} (> 0.3 ⇒ still on the rod)`,
                data,
            },
        ];
    },

    live(): string {
        const snap = bodyMirror?.snapshot;
        if (!snap) return "spikecatch — warming";
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const sy = s[(0 * cap + spikeEid) * 4 + 1];
        let ry = 0;
        for (const e of ropeBodies) ry += s[(0 * cap + e) * 4 + 1];
        ry /= ropeBodies.length;
        return `spikecatch — ${kinematic ? "kinematic" : "dynamic"} bob, thick ${rodThick}\nrod y ${sy.toFixed(2)}  rope y ${ry.toFixed(2)}`;
    },
};

register(scenario);
