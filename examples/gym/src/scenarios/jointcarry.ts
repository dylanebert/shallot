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
    type System,
    Transform,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    BODY_VEC4,
    PENALTY_MIN,
    Physics,
    StepSystem,
    WORLD,
} from "@dylanebert/shallot/physics/core";
import { Meshes } from "@dylanebert/shallot/render/core";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// jointcarry — the spindle game's catch mechanism, isolated (games/spindle roadmap §1). kincarry proved a
// rotating KINEMATIC body carries neither contacts nor (as this scene first measured) joints — its dq ≡ 0 in
// the solve, so a jointed body just floats and spirals in (kex backlog.md "kinematic angular carry —
// declined"). The supported path is a DYNAMIC driven body: real rotation, so the joint couples velocity and
// carries. Here a dynamic box (a square column on a two-pin world axle, spun by a forced angular velocity
// each tick — the proven motor drive) rotates about Y; a dynamic ball is pinned to it by a spherical joint at
// radius R, OUTSIDE the corner sweep (a clear gap — no contact, so the carry is 100% the joint). The gate:
// the ball orbits at ~the drive rate (swept angle tracks the spinner's, magnitude-for-magnitude) and its
// radius holds — where the kinematic version left it essentially still.

const G = -10;
const DT = 1 / 60;
const CY = 1.5; // axis height
const SPIN_HW = 0.3; // spindle half-width (square cross-section — corners make the spin visible; round hides it)
const SPIN_HH = 1.0; // spindle half-height
const SPIN_MASS = 4; // dynamic — heavy enough that the light ball's joint reaction barely perturbs the driven spin
const BALL_R = 0.2; // dynamic ball radius
const R = 0.8; // pin radius from the axis — ball surface (0.6) clears the box's corner sweep (0.3·√2≈0.42): no contact

const cfg = {
    dt: DT,
    gravity: G,
    alpha: 0.99,
    penalty: PENALTY_MIN,
    betaLin: 1e4,
    gamma: 0.999,
    iterations: 10,
    maxColors: 8,
};

let spinnerEid = -1;
let ballEid = -1;
let spinRate = 3; // rad/s — read by the spin system; set from the param in build
let bodyMirror: Mirror | null = null;
let result: {
    sweptBall: number;
    sweptSpinner: number;
    minR: number;
    maxR: number;
    finite: boolean;
} | null = null;

// drive the DYNAMIC spindle: force its angular velocity about Y each fixed tick BEFORE the step (a motor that
// overrides the joint reaction). COM is on the axis, so linear velocity is forced to 0; the two-pin world
// axle holds it in place against gravity, leaving Y-spin free. A dynamic body integrates this (dq ≠ 0), so
// the joint to the ball couples real motion and carries it — the kinematic body's dq ≡ 0 is exactly what
// failed.
const SpinPlugin: Plugin = {
    name: "JointCarrySpin",
    systems: [
        {
            name: "jointcarry-spin",
            group: "fixed",
            before: [StepSystem],
            update() {
                if (spinnerEid < 0 || !Physics.step) return;
                Physics.step.setAngularVelocity(spinnerEid, 0, spinRate, 0);
                Physics.step.setVelocity(spinnerEid, 0, 0, 0);
            },
        } satisfies System,
    ],
};

// wrap a raw angle delta into (-π, π] so the per-sample accumulation survives the atan2 branch cut.
function wrapPi(d: number): number {
    return d - 2 * Math.PI * Math.round(d / (2 * Math.PI));
}

const scenario: Scenario = {
    name: "jointcarry",
    params: [{ key: "rate", type: "number", default: 3, min: 0, max: 30, step: 0.5 }],

    async build(_canvas, p: Params) {
        result = null;
        spinRate = p.rate as number;

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

        Physics.step?.configure(cfg);

        // floor (top y=0) just for the visual
        const floor = state.create();
        state.add(floor, Body);
        Body.shape.set(floor, ShapeKind.Box);
        Body.pos.set(floor, 0, -0.5, 0, 0);
        Body.halfExtents.set(floor, 20, 0.5, 20, 0);
        Body.mass.set(floor, 0);
        state.add(floor, Part);
        state.add(floor, Color);
        Color.rgba.set(floor, 0.28, 0.3, 0.34, 1);

        // the dynamic spindle (a square column — corners make the rotation legible), driven + axle-pinned
        spinnerEid = state.create();
        state.add(spinnerEid, Body);
        Body.shape.set(spinnerEid, ShapeKind.Box);
        Body.pos.set(spinnerEid, 0, CY, 0, 0);
        Body.halfExtents.set(spinnerEid, SPIN_HW, SPIN_HH, SPIN_HW, 0);
        Body.mass.set(spinnerEid, SPIN_MASS);
        Body.friction.set(spinnerEid, 0.5);
        state.add(spinnerEid, Part);
        state.add(spinnerEid, Color);
        Color.rgba.set(spinnerEid, 0.55, 0.6, 0.7, 1);

        // the dynamic ball, pinned OUTSIDE the spindle surface (no contact — pure joint carry)
        ballEid = state.create();
        state.add(ballEid, Body);
        Body.shape.set(ballEid, ShapeKind.Sphere);
        Body.pos.set(ballEid, R, CY, 0, 0);
        Body.halfExtents.set(ballEid, 0, 0, 0, BALL_R);
        Body.mass.set(ballEid, 1);
        Body.friction.set(ballEid, 0.4);
        state.add(ballEid, Part);
        Part.mesh.set(ballEid, Meshes.id("sphere") ?? 0);
        state.add(ballEid, Color);
        Color.rgba.set(ballEid, 0.85, 0.55, 0.35, 1);

        // the axle (two world spherical joints pinning the spindle's top + base to fixed world points — lock
        // translation + tilt, leave Y-spin free) + the attach (a spherical joint pinning the ball's center to
        // spindle-local [R,0,0], coincident at the identity spawn pose). Rigid linear (default) holds them;
        // spherical (stiffnessAng 0) leaves rotation free. The driven spindle's real rotation drags the ball.
        Physics.step?.setJoints([
            { a: WORLD, b: spinnerEid, rA: [0, CY + SPIN_HH, 0], rB: [0, SPIN_HH, 0] },
            { a: WORLD, b: spinnerEid, rA: [0, CY - SPIN_HH, 0], rB: [0, -SPIN_HH, 0] },
            { a: spinnerEid, b: ballEid, rA: [R, 0, 0], rB: [0, 0, 0], stiffnessAng: 0 },
        ]);

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 4);
        Orbit.distance.set(cam, 6);

        await frames(5);
        if (Physics.step) bodyMirror = mirror(Physics.step.bodies);

        // sample the ball's orbit + the spindle's drive angle over the run, accumulating each through the
        // atan2 branch cut. swept magnitudes compared (the orbit is opposite-signed to the drive — atan2(−sinθ)).
        let sweptBall = 0;
        let sweptSpinner = 0;
        let minR = R;
        let maxR = R;
        let finite = true;
        let prevB: number | null = null;
        let prevS: number | null = null;
        for (let k = 0; k < 40; k++) {
            await frames(15);
            const snap = bodyMirror?.snapshot;
            if (!snap) continue;
            const s = new Float32Array(snap.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            const x = s[(0 * cap + ballEid) * 4];
            const z = s[(0 * cap + ballEid) * 4 + 2];
            const qy = s[(1 * cap + spinnerEid) * 4 + 1];
            const qw = s[(1 * cap + spinnerEid) * 4 + 3];
            if (![x, z, qy, qw].every(Number.isFinite)) {
                finite = false;
                continue;
            }
            const b = Math.atan2(z, x);
            const sp = 2 * Math.atan2(qy, qw);
            if (prevB !== null) sweptBall += wrapPi(b - prevB);
            if (prevS !== null) sweptSpinner += wrapPi(sp - prevS);
            prevB = b;
            prevS = sp;
            const r = Math.hypot(x, z);
            minR = Math.min(minR, r);
            maxR = Math.max(maxR, r);
        }
        result = { sweptBall, sweptSpinner, minR, maxR, finite };

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                spinnerEid = -1;
                ballEid = -1;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const r = result;
        if (!r) return [{ name: "jointcarry", pass: false, detail: "no measurement" }];
        const ball = Math.abs(r.sweptBall);
        const spin = Math.abs(r.sweptSpinner);
        // window long enough to be unambiguous (the spindle swept multiple revolutions)
        const drove = spin > 2 * Math.PI;
        // carry: the ball's orbit tracks the drive magnitude-for-magnitude (a small constant phase lag from
        // the joint's compliance cancels in the swept difference). Rigid-joint convergence at iters=10 holds
        // the ratio near 1; allow a generous band — the contrast with kincarry (~0) is the point.
        const ratio = spin > 1e-6 ? ball / spin : 0;
        const tracks = ratio > 0.8 && ratio < 1.1;
        // radius holds: the rigid pin keeps the ball at R against gravity (sub-cm convergence + a little sag)
        const heldRadius = r.minR > R - 0.1 && r.maxR < R + 0.1;
        const data = {
            sweptBall: +r.sweptBall.toFixed(3),
            sweptSpinner: +r.sweptSpinner.toFixed(3),
            ratio: +ratio.toFixed(3),
            minR: +r.minR.toFixed(3),
            maxR: +r.maxR.toFixed(3),
        };
        return [
            { name: "finite", pass: r.finite, detail: "ball stays finite (no NaN/blowup)" },
            { name: "drove", pass: drove, detail: `spindle swept ${data.sweptSpinner} rad (> 2π)` },
            {
                name: "carries",
                pass: tracks,
                detail: `ball/spinner swept ratio ${data.ratio} (orbit tracks the drive) | ball ${data.sweptBall} spinner ${data.sweptSpinner}`,
                data,
            },
            {
                name: "holds-radius",
                pass: heldRadius,
                detail: `radius ∈ [${data.minR}, ${data.maxR}] vs pin ${R}`,
            },
        ];
    },

    live(): string {
        const snap = bodyMirror?.snapshot;
        if (!snap) return "jointcarry — warming";
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const bx = s[(0 * cap + ballEid) * 4].toFixed(2);
        const bz = s[(0 * cap + ballEid) * 4 + 2].toFixed(2);
        const cq = s[(1 * cap + spinnerEid) * 4 + 1].toFixed(2);
        return `jointcarry — ball joint-pinned to a spinning kinematic spindle\nball xz ${bx},${bz}  spindle quat.y ${cq}`;
    },
};

register(scenario);
