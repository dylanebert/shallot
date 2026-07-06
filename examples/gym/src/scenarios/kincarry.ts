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
import { BODY_VEC4, PENALTY_MIN, Physics, StepSystem } from "@dylanebert/shallot/physics/core";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// kincarry — the fundamental behind the spindle game, stripped to the bone: does the CORNER of a rotating
// KINEMATIC cube push a contacting DYNAMIC box? No ropes, no joints, no grab. A kinematic body's pose is set
// each fixed step (`setKinematic`); physics.md is explicit that kinematic ANGULAR carry is NOT tracked — the
// body's COM linear velocity carries a platform, rotation does not. So the open question this scenario
// isolates: with no tracked angular velocity, does the rotating cube's geometry still depenetrate (and thus
// push) a dynamic box its corner sweeps into, or does nothing happen? Understanding only — no fix here.
//
// Layout: a static floor (top y=0.5), a kinematic cube spinning about Y at the origin (bottom on the floor,
// so Y-rotation keeps it flat), and one dynamic box placed at radius `dist` so ONLY the cube's corner reach
// (half·√2 ≈ 0.707) crosses it — the flat face (reach 0.5) never does. The probe Mirrors the body buffer and
// tracks the box: radial push-out, tangential carry (orbit angle), and peak speed.
//
// FINDING (lovelace, 2026-06-21, default params rate 2.5 / dist 0.95): the corner does NOT carry the box.
// displacement 0.038, radialOut 0.039, tangential 0.0025 rad (~0.14°), maxSpeed 0.08 m/s — i.e. a hair of
// soft depenetration and essentially zero tangential drag, with the corner visibly interpenetrating the box
// (it passes THROUGH faster than the soft contact pushes out). Root cause is the documented design: a
// kinematic body carries a contact by its COM LINEAR velocity only (`setKinematic` derives that from the
// pose delta — zero for pure rotation), and angular velocity is not tracked, so the solver attributes no
// relative velocity to the spinning surface and the contact resolves like a static penetration the rotation
// outruns. So the spindle's "spin winds rope on by friction/sweep" does NOT work on kinematic rotation
// alone — it needs either a joint-pin (pin a contacting body to the drum so the drum's POSE rotation carries
// it) or engine support for tracked kinematic angular velocity. Not fixed here — understanding only.

const G = -10;
const DT = 1 / 60;
const CUBE_HALF = 0.5; // corner reach = 0.5·√2 ≈ 0.707; flat-face reach = 0.5
const BOX_HALF = 0.4;
const FLOOR_TOP = 0.5;
const CUBE_Y = FLOOR_TOP + CUBE_HALF; // 1.0 — bottom flat on the floor
const BOX_Y = FLOOR_TOP + BOX_HALF; // 0.9

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

let cubeEid = -1;
let boxEid = -1;
let spinRate = 2.5; // rad/s — read by the spin system; set from the param in build
let bodyMirror: Mirror | null = null;
let result: {
    displacement: number;
    radialOut: number;
    tangential: number;
    maxSpeed: number;
    finite: boolean;
} | null = null;

// drive the kinematic cube: rotate its quat about Y each fixed tick and upload via setKinematic, BEFORE the
// step so the pose lands in this tick's solve. COM is stationary, so the derived linear velocity is 0 — the
// only motion is the rotation, exactly the kinematic-angular-carry case under test.
const SpinPlugin: Plugin = {
    name: "KinSpin",
    systems: [
        {
            name: "kin-spin",
            group: "fixed",
            before: [StepSystem],
            update(state: State) {
                if (cubeEid < 0 || !Physics.step) return;
                const a = state.time.elapsed * spinRate;
                Physics.step.setKinematic(
                    cubeEid,
                    [0, CUBE_Y, 0],
                    [0, Math.sin(a / 2), 0, Math.cos(a / 2)],
                );
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
    friction = 0.4,
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

const scenario: Scenario = {
    name: "kincarry",
    params: [
        { key: "rate", type: "number", default: 2.5, min: 0, max: 30, step: 0.5 },
        // radius of the dynamic box from the cube's axis. default 0.95 → near face at 0.55, so the flat face
        // (0.5) never touches and only the corner (0.707) sweeps into it — a pure corner-push test.
        { key: "dist", type: "number", default: 0.95, min: 0.6, max: 2, step: 0.05 },
        { key: "friction", type: "number", default: 0.2, min: 0, max: 1, step: 0.05 },
    ],

    async build(_canvas, p: Params) {
        result = null;
        spinRate = p.rate as number;
        const dist = p.dist as number;

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

        box(state, [0, -0.5 + FLOOR_TOP, 0], [20, 0.5, 20], 0, [0.28, 0.3, 0.34]); // floor, top at 0.5
        cubeEid = box(
            state,
            [0, CUBE_Y, 0],
            [CUBE_HALF, CUBE_HALF, CUBE_HALF],
            0,
            [0.55, 0.6, 0.7],
        ); // kinematic
        boxEid = box(
            state,
            [dist, BOX_Y, 0],
            [BOX_HALF, BOX_HALF, BOX_HALF],
            1,
            [0.85, 0.55, 0.35],
            p.friction as number,
        );

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 4);
        Orbit.distance.set(cam, 6);

        await frames(3);
        if (Physics.step) bodyMirror = mirror(Physics.step.bodies);

        // sample the box over ~3 s of sim and track the worst-case motion away from its (dist, 0) spawn.
        let maxR = dist;
        let maxTan = 0;
        let maxSpeed = 0;
        let finite = true;
        let fx = dist;
        let fz = 0;
        for (let k = 0; k < 20; k++) {
            await frames(15);
            const snap = bodyMirror?.snapshot;
            if (!snap) continue;
            const s = new Float32Array(snap.bytes);
            const cap = s.length / (BODY_VEC4 * 4);
            const x = s[(0 * cap + boxEid) * 4];
            const z = s[(0 * cap + boxEid) * 4 + 2];
            const vx = s[(6 * cap + boxEid) * 4];
            const vy = s[(6 * cap + boxEid) * 4 + 1];
            const vz = s[(6 * cap + boxEid) * 4 + 2];
            if (![x, z, vx, vy, vz].every(Number.isFinite)) {
                finite = false;
                continue;
            }
            fx = x;
            fz = z;
            maxR = Math.max(maxR, Math.hypot(x, z));
            maxTan = Math.max(maxTan, Math.abs(Math.atan2(z, x))); // orbit angle off the +x spawn ray
            maxSpeed = Math.max(maxSpeed, Math.hypot(vx, vy, vz));
        }
        result = {
            displacement: Math.hypot(fx - dist, fz),
            radialOut: maxR - dist,
            tangential: maxTan,
            maxSpeed,
            finite,
        };

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                cubeEid = -1;
                boxEid = -1;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const r = result;
        if (!r) return [{ name: "kincarry", pass: false, detail: "no measurement" }];
        // The runs-clean gate (a real regression guard): the sim stays finite and the box doesn't tunnel.
        // The behavioral finding (does the corner push? carry?) rides as data on a diagnostic check — this is
        // an understand-first scenario, so the numbers are the deliverable, not yet a hard pass/fail.
        const data = {
            displacement: +r.displacement.toFixed(4),
            radialOut: +r.radialOut.toFixed(4),
            tangential: +r.tangential.toFixed(4),
            maxSpeed: +r.maxSpeed.toFixed(4),
        };
        return [
            { name: "finite", pass: r.finite, detail: "box stays finite (no NaN/tunnel)" },
            {
                name: "corner-push",
                pass: true,
                detail: `displacement ${data.displacement} | radialOut ${data.radialOut} | tangential ${data.tangential} rad | maxSpeed ${data.maxSpeed}`,
                data,
            },
        ];
    },

    live(): string {
        const snap = bodyMirror?.snapshot;
        if (!snap) return "kincarry — warming";
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const bx = s[(0 * cap + boxEid) * 4].toFixed(2);
        const bz = s[(0 * cap + boxEid) * 4 + 2].toFixed(2);
        const cq = s[(1 * cap + cubeEid) * 4 + 1].toFixed(2); // cube quat.y — nonzero ⇒ spinning
        return `kincarry — rotating kinematic cube vs dynamic box\nbox xz ${bx},${bz}  cube quat.y ${cq}`;
    },
};

register(scenario);
