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
    qRotate,
    StepSystem,
    WORLD,
} from "@dylanebert/shallot/physics/core";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// dyncarry — the spindle game's drum assembly, isolated to validate it. A thin DYNAMIC box drum on a
// two-spherical-pin axle, studded with a crown of thin up-out spikes (each held by two spherical pins — a
// FIXED joint's small-angle angular row injects energy at a large relative orientation, the chaos this
// scene first caught), the whole rigid assembly DRIVEN to the Y-rotation field each tick. Plus a capsule
// rope strand. The asserts gate: SETTLED stability (no vibration / explosion), the axle holding, a CLEAN
// STARTUP (the drum reaches +ω with no wrong-way dip — the rigid-drive fix), and the overlapping-capsule
// rope holding together. Params (rate / iters / rings / perRing / embed / lift / rope) bisect it.
// `kincarry` stays as the kinematic-null reference.

const G = -10;
const DT = 1 / 60;
const DRUM_HW = 0.12; // thin square cross-section half-width
const DRUM_HH = 1.8; // half-height
const SPIKE_REACH = 0.5; // tip distance from the base along the up-out axis
const SPIKE_TILT = 0.8; // up-angle from horizontal (rad) — the squid-jig rake
const SPIKE_R = 0.025; // half-thickness — thin, so it slips between rope links and grabs
const SPIKE_MASS = 0.05;
const DRUM_MASS = 4;
const BOX_HALF = 0.4;

// align a box's local +x to a unit direction, shortest arc (no roll) — quat [x,y,z,w].
function aimQuat(dx: number, dy: number, dz: number): [number, number, number, number] {
    if (dx > 0.99999) return [0, 0, 0, 1];
    if (dx < -0.99999) return [0, 0, 1, 0];
    const ax = 0;
    const ay = -dz;
    const az = dy; // x̂ × dir
    const al = Math.hypot(ax, ay, az);
    const ang = Math.acos(dx);
    const s = Math.sin(ang / 2) / al;
    return [ax * s, ay * s, az * s, Math.cos(ang / 2)];
}

// align local +Y to a unit direction (capsule core orientation), shortest arc — quat [x,y,z,w].
function aimQuatY(dx: number, dy: number, dz: number): [number, number, number, number] {
    if (dy > 0.99999) return [0, 0, 0, 1];
    if (dy < -0.99999) return [0, 0, 1, 0];
    const ax = dz;
    const az = -dx; // (0,1,0) × dir
    const al = Math.hypot(ax, 0, az);
    const ang = Math.acos(dy);
    const s = Math.sin(ang / 2) / al;
    return [ax * s, 0, az * s, Math.cos(ang / 2)];
}

const ROPE_SEG = 0.12;
const ROPE_PIN = ROPE_SEG / 2;
const ROPE_RADIUS = 0.045;
const ROPE_CORE = 0.02; // core+radius 0.065 > pin → overlapping continuous capsule tube

let drumEid = -1;
let boxEid = -1;
let assembly: number[] = []; // drum + spikes — the bodies whose vibration signals instability
let ropeBodies: number[] = []; // a capsule rope strand — validates the overlapping-link construction
let spinRate = 0;
let bodyMirror: Mirror | null = null;
let result: {
    settledSpeed: number; // max body speed over the LAST samples — should be ~0 at rest (vibration ⇒ unstable)
    peakSpeed: number;
    drumDrift: number;
    tangential: number;
    minDrumSpin: number; // min drum angular velocity (Y) over startup — negative ⇒ a wrong-way spin glitch
    minSpikeOrbit: number; // min spike ORBITAL rate (Y) over startup — negative ⇒ a spike winding backward
    ropeSettled: number; // max rope-link speed over the settled window — high ⇒ the capsule chain blew apart
    finite: boolean;
} | null = null;

// drive the whole rigid assembly (drum + spikes) to the Y-rotation field each tick: angular (0, ω, 0) +
// linear ω×r at each body's ACTUAL (x, z) off the axis (the GPU pose via the Mirror, the authored spawn until
// it's live). Driving every body (not just the drum) starts it rotating coherently — a drum-only drive lets
// the spikes lag + the assembly settle under gravity first (a startup stall). Reading the ACTUAL position
// (not an open-loop θ = ω·elapsed) keeps the prescribed linear velocity tangent to where the body really is,
// so a phase lag — deferred spawn (the game boots the world several frames in, at θ=0 but a large elapsed),
// spin-up inertia, a rope load — can't misalign the tangent and wind a spike the wrong way. 0 = no drive.
const SpinPlugin: Plugin = {
    name: "DynSpin",
    systems: [
        {
            name: "dyn-spin",
            group: "fixed",
            before: [StepSystem],
            update() {
                if (spinRate === 0 || !Physics.step || assembly.length === 0) return;
                const snap = bodyMirror?.snapshot;
                const s = snap ? new Float32Array(snap.bytes) : null;
                const cap = s ? s.length / (BODY_VEC4 * 4) : 0;
                for (const eid of assembly) {
                    const x = s ? s[(0 * cap + eid) * 4] : Body.pos.x.get(eid);
                    const z = s ? s[(0 * cap + eid) * 4 + 2] : Body.pos.z.get(eid);
                    Physics.step.setAngularVelocity(eid, 0, spinRate, 0);
                    Physics.step.setVelocity(eid, spinRate * z, 0, -spinRate * x);
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
    quat: readonly [number, number, number, number] = [0, 0, 0, 1],
    friction = 0.6,
): number {
    const eid = state.create();
    state.add(eid, Body);
    Body.shape.set(eid, ShapeKind.Box);
    Body.pos.set(eid, pos[0], pos[1], pos[2], 0);
    Body.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
    Body.halfExtents.set(eid, half[0], half[1], half[2], 0);
    Body.mass.set(eid, mass);
    Body.friction.set(eid, friction);
    state.add(eid, Part);
    state.add(eid, Color);
    Color.rgba.set(eid, color[0], color[1], color[2], 1);
    return eid;
}

// one upward-out spike held to the drum (the game's spike()): a thin pole whose base sits past the drum
// surface and tip rakes up + out (the squid-jig catch). Held by TWO spherical joints (base + tip) along its
// axis. NOT a fixed joint: its angular row is a small-angle quaternion difference that can't hold a
// 90°/180° relative orientation and injects energy. Two linear point-pins fix the spike's position +
// direction with no angular row — robust at any orientation; the only free DOF is roll about the spike's
// own thin axis, invisible at this thickness. Every drum-side anchor is in drum-local frame, so the spike
// orbits rigidly with the driven drum.
function spawnSpike(
    state: State,
    drumCenterY: number,
    az: number,
    baseY: number,
    embed: number,
    joints: JointDef[],
): number {
    const dx = Math.cos(az);
    const dz = -Math.sin(az);
    // sit the base ON the box surface along this azimuth (face DRUM_HW, corner DRUM_HW·√2), then `embed`
    // inside it — flush, no gap. The spike rakes up + out immediately, so only the base nub overlaps.
    const inner = DRUM_HW / Math.max(Math.abs(dx), Math.abs(dz)) - embed;
    const ct = Math.cos(SPIKE_TILT);
    const st = Math.sin(SPIKE_TILT);
    const dir: [number, number, number] = [dx * ct, st, dz * ct]; // unit up-out axis
    const base: [number, number, number] = [dx * inner, baseY, dz * inner];
    const half = SPIKE_REACH / 2;
    const center: [number, number, number] = [
        base[0] + dir[0] * half,
        base[1] + dir[1] * half,
        base[2] + dir[2] * half,
    ];
    const quat = aimQuat(dir[0], dir[1], dir[2]);
    const eid = box(state, center, [half, SPIKE_R, SPIKE_R], SPIKE_MASS, [0.8, 0.42, 0.16], quat);
    // pin the base (local −half) and tip (local +half) to drum-local points; rA = world point − drum center
    // (drum at identity quat), rB = the local point. Both coincident with the spike ends at spawn.
    const pin = (lx: number): void => {
        const w = qRotate(quat[0], quat[1], quat[2], quat[3], lx, 0, 0);
        joints.push({
            a: drumEid,
            b: eid,
            rA: [center[0] + w[0], center[1] + w[1] - drumCenterY, center[2] + w[2]],
            rB: [lx, 0, 0],
        });
    };
    pin(-half);
    pin(half);
    return eid;
}

// a capsule rope strand on the floor along +x — overlapping rounded links (core+radius > pin) chained by
// spherical pins at the core ends. Validates that the continuous-capsule construction holds (no blast).
function spawnRope(state: State, sx: number, links: number, joints: JointDef[]): void {
    const quat = aimQuatY(1, 0, 0); // core along +x
    let prev = -1;
    for (let i = 0; i < links; i++) {
        const eid = state.create();
        state.add(eid, Body);
        Body.shape.set(eid, ShapeKind.Capsule);
        Body.pos.set(eid, sx + ROPE_SEG * (i + 0.5), ROPE_RADIUS + 0.02, 0, 0);
        Body.quat.set(eid, quat[0], quat[1], quat[2], quat[3]);
        Body.halfExtents.set(eid, 0, ROPE_CORE, 0, ROPE_RADIUS);
        Body.mass.set(eid, 0.4);
        Body.friction.set(eid, 0.7);
        state.add(eid, Part);
        state.add(eid, Color);
        Color.rgba.set(eid, 0.72, 0.57, 0.35, 1);
        if (prev >= 0)
            joints.push({ a: prev, b: eid, rA: [0, ROPE_PIN, 0], rB: [0, -ROPE_PIN, 0] });
        prev = eid;
        ropeBodies.push(eid);
    }
}

const scenario: Scenario = {
    name: "dyncarry",
    params: [
        { key: "rate", type: "number", default: 3, min: 0, max: 30, step: 0.5 },
        { key: "iters", type: "number", default: 6, min: 1, max: 30, step: 1 },
        { key: "rings", type: "number", default: 9, min: 0, max: 20, step: 1 },
        { key: "perRing", type: "number", default: 5, min: 1, max: 12, step: 1 },
        // how far the spike base sits inside the drum surface (flush + a nub of overlap)
        { key: "embed", type: "number", default: 0.02, min: 0, max: 0.1, step: 0.005 },
        { key: "lift", type: "number", default: 0, min: 0, max: 1, step: 0.05 },
        { key: "dist", type: "number", default: 0.85, min: 0, max: 2, step: 0.05 },
        { key: "rope", type: "number", default: 1, min: 0, max: 1, step: 1 }, // spawn a capsule rope strand
    ],

    async build(_canvas, p: Params) {
        result = null;
        assembly = [];
        ropeBodies = [];
        spinRate = p.rate as number;
        const dist = p.dist as number;
        const lift = p.lift as number;
        const drumCY = DRUM_HH + lift; // drum center; base at `lift` above the floor (top y=0)

        const { state, dispose } = await run({
            defaults: false,
            capacity: 128,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                MirrorPlugin,
                TransformsPlugin,
                InputPlugin,
                OrbitPlugin,
                RenderPlugin,
                PhysicsPlugin,
                SpinPlugin,
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

        // the drum + axle (two spherical pins at top + base on the Y axis)
        drumEid = box(
            state,
            [0, drumCY, 0],
            [DRUM_HW, DRUM_HH, DRUM_HW],
            DRUM_MASS,
            [0.55, 0.6, 0.7],
        );
        Body.friction.set(drumEid, 0.8);
        assembly.push(drumEid);
        const joints: JointDef[] = [
            { a: WORLD, b: drumEid, rA: [0, drumCY + DRUM_HH, 0], rB: [0, DRUM_HH, 0] },
            { a: WORLD, b: drumEid, rA: [0, drumCY - DRUM_HH, 0], rB: [0, -DRUM_HH, 0] },
        ];

        // the spikes: regular rings up the drum (a bottle-brush crown), each raking up + out (the squid-jig
        // catch). Alternate rings offset half a step so adjacent rings' spikes interleave for denser cover.
        const rings = p.rings as number;
        const perRing = p.perRing as number;
        const embed = p.embed as number;
        for (let r = 0; r < rings; r++) {
            const baseY = 0.4 + (rings > 1 ? r / (rings - 1) : 0) * 2.6 + lift;
            const off = (r % 2) * (Math.PI / perRing);
            for (let k = 0; k < perRing; k++) {
                const az = (k / perRing) * Math.PI * 2 + off;
                assembly.push(spawnSpike(state, drumCY, az, baseY, embed, joints));
            }
        }

        // a capsule rope strand on the floor, clear of the drum + spikes (reach ~0.6) — construction check
        if ((p.rope as number) > 0) spawnRope(state, 2, 20, joints);

        Physics.step?.setJoints(joints);

        boxEid =
            dist > 0
                ? box(
                      state,
                      [dist, 0.4, 0],
                      [BOX_HALF, BOX_HALF, BOX_HALF],
                      1,
                      [0.85, 0.55, 0.35],
                      undefined,
                      0.6,
                  )
                : -1;

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 5);
        Orbit.distance.set(cam, 8);

        await frames(3);
        if (Physics.step) bodyMirror = mirror(Physics.step.bodies);

        const speedOf = (s: Float32Array, cap: number, eid: number): number => {
            const o = (6 * cap + eid) * 4;
            return Math.hypot(s[o], s[o + 1], s[o + 2]);
        };

        // startup window: the drum's recovered angular velocity (B_VELA.y) should climb to +ω without
        // dipping negative — a negative dip is the wrong-way stall the rigid drive is meant to remove. The
        // SPIKES are the real tell (the drum is on-axis, so its drive is angular-only): each spike ORBITS the
        // Y axis, so its orbital rate ω_orb = (z·vx − x·vz)/(x²+z²) (from its actual pos + B_VELL) should also
        // hold +ω — a negative dip is a spike winding the wrong way (the analytic-θ drive's misalignment).
        let minDrumSpin = spinRate;
        let minSpikeOrbit = spinRate;
        const spikes = assembly.filter((e) => e !== drumEid);
        for (let k = 0; k < 25; k++) {
            await frames(1);
            const snap = bodyMirror?.snapshot;
            if (!snap) continue;
            const s = new Float32Array(snap.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            const wy = s[(7 * cap + drumEid) * 4 + 1]; // B_VELA.y
            if (Number.isFinite(wy)) minDrumSpin = Math.min(minDrumSpin, wy);
            for (const e of spikes) {
                const x = s[(0 * cap + e) * 4];
                const z = s[(0 * cap + e) * 4 + 2];
                const vx = s[(6 * cap + e) * 4];
                const vz = s[(6 * cap + e) * 4 + 2];
                const r2 = x * x + z * z;
                if (r2 < 1e-4) continue; // on-axis, no orbit defined
                const orbit = (z * vx - x * vz) / r2;
                if (Number.isFinite(orbit)) minSpikeOrbit = Math.min(minSpikeOrbit, orbit);
            }
        }

        let peakSpeed = 0;
        let settledSpeed = 0;
        let maxDrift = 0;
        let maxTan = 0;
        let ropeSettled = 0;
        let finite = true;
        for (let k = 0; k < 24; k++) {
            await frames(15);
            const snap = bodyMirror?.snapshot;
            if (!snap) continue;
            const s = new Float32Array(snap.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            let frameMax = 0;
            for (const e of assembly) {
                const v = speedOf(s, cap, e);
                if (!Number.isFinite(v)) finite = false;
                frameMax = Math.max(frameMax, v);
            }
            peakSpeed = Math.max(peakSpeed, frameMax);
            if (k >= 18) settledSpeed = Math.max(settledSpeed, frameMax); // last ~6 samples = settled window
            for (const e of ropeBodies) {
                const v = speedOf(s, cap, e);
                if (!Number.isFinite(v)) finite = false;
                if (k >= 18) ropeSettled = Math.max(ropeSettled, v);
            }
            const dx = s[(0 * cap + drumEid) * 4];
            const dz = s[(0 * cap + drumEid) * 4 + 2];
            maxDrift = Math.max(maxDrift, Math.hypot(dx, dz));
            if (boxEid >= 0) {
                const bx = s[(0 * cap + boxEid) * 4];
                const bz = s[(0 * cap + boxEid) * 4 + 2];
                if (Number.isFinite(bx) && Number.isFinite(bz)) {
                    maxTan = Math.max(maxTan, Math.abs(Math.atan2(bz, bx)));
                }
            }
        }
        result = {
            settledSpeed,
            peakSpeed,
            drumDrift: maxDrift,
            tangential: maxTan,
            minDrumSpin,
            minSpikeOrbit,
            ropeSettled,
            finite,
        };

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                drumEid = -1;
                boxEid = -1;
                assembly = [];
                ropeBodies = [];
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const r = result;
        if (!r) return [{ name: "dyncarry", pass: false, detail: "no measurement" }];
        const data = {
            settledSpeed: +r.settledSpeed.toFixed(4),
            peakSpeed: +r.peakSpeed.toFixed(4),
            drumDrift: +r.drumDrift.toFixed(4),
            tangential: +r.tangential.toFixed(4),
            minDrumSpin: +r.minDrumSpin.toFixed(3),
            minSpikeOrbit: +r.minSpikeOrbit.toFixed(3),
            ropeSettled: +r.ropeSettled.toFixed(4),
        };
        const driven = spinRate > 0;
        const checks: Check[] = [
            {
                name: "finite",
                pass: r.finite,
                detail: "assembly + rope stay finite (no NaN/explosion)",
            },
            {
                // the stability gate: at rest (rate 0) the assembly settles to ~0 vibration; flickering shows
                // as a high settled speed. Driven, the forced spin carries the spike tips (≈ω·r), so the gate
                // relaxes to "bounded, not exploding". This is the check that caught the chaotic joint setup.
                name: "settled-stable",
                pass: driven ? r.settledSpeed < 5 && r.peakSpeed < 50 : r.settledSpeed < 0.2,
                detail: `settledSpeed ${data.settledSpeed} ${driven ? "(driven, < 5)" : "(rest, < 0.2)"} | peak ${data.peakSpeed}`,
                data,
            },
            {
                name: "axle-holds",
                pass: r.drumDrift < 0.1,
                detail: `drum drift off axis ${data.drumDrift} (< 0.1)`,
            },
        ];
        if (driven) {
            // the startup gate: the rigid drive holds the drum at ~ω from the first tick (min 2.77 at ω=3);
            // the pre-fix drum-only drive let the un-driven spikes load the drum's joints during spin-up,
            // sagging it to ~1.7 (0.57ω). Gate at 0.8·ω — above the pre-fix sag, below the rigid hold — so
            // it pins the rigid drive (verified by A/B: drum-only goes red here).
            checks.push({
                name: "clean-startup",
                pass: r.minDrumSpin > 0.8 * spinRate,
                detail: `min drum spin ${data.minDrumSpin} rad/s over startup (> ${(0.8 * spinRate).toFixed(2)} = 0.8·ω)`,
            });
            // the spikes must ORBIT +ω from the first tick too (not just the on-axis drum): a negative dip is
            // a spike winding the wrong way during spin-up — the visible glitch the actual-position drive
            // removes (the analytic-θ drive misaligned the prescribed tangent once the spike's angle lagged).
            checks.push({
                name: "spikes-no-reverse",
                pass: r.minSpikeOrbit > -0.05 * spinRate,
                detail: `min spike orbit ${data.minSpikeOrbit} rad/s over startup (> ${(-0.05 * spinRate).toFixed(2)} — no wrong-way)`,
            });
        }
        if (ropeBodies.length > 0) {
            // the capsule-rope gate: the overlapping-link chain settles flat, not blown apart
            checks.push({
                name: "rope-stable",
                pass: r.ropeSettled < 0.5,
                detail: `rope settled speed ${data.ropeSettled} (< 0.5 — the capsule chain holds)`,
            });
        }
        return checks;
    },

    live(): string {
        const snap = bodyMirror?.snapshot;
        if (!snap) return "dyncarry — warming";
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const dq = s[(1 * cap + drumEid) * 4 + 1].toFixed(2);
        return `dyncarry — driven spike drum + capsule rope\ndrum quat.y ${dq}  bodies ${assembly.length}`;
    },
};

register(scenario);
