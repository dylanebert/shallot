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
import { AvbdPlugin } from "@dylanebert/shallot/avbd";
import { Avbd, BODY_VEC4, type JointDef, PENALTY_MIN, WORLD } from "@dylanebert/shallot/avbd/core";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import { StepSystem } from "@dylanebert/shallot/physics/core";
import { type Check, frames, type Params, register, type Scenario } from "../gym";

// motor — the angular motor constraint (kex roadmap "Correct driven-rotation behavior"), the spindle game's
// drive done right: a driven body HOLDS its target ω under load. Two drives select via the
// `drive` param against the SAME heavy rotational load (a light spindle rigidly coupled to a heavy coaxial
// flywheel — the rigid joint to a large inertia is the forced drive's stall mechanism, the rope's aggregate
// joints in miniature):
//   • forced  — setAngularVelocity each tick (the old-drum approach): consumed once by the inertial
//               prediction, so the rigid coupling redistributes it to the flywheel every step and the spin
//               creeps far below target — it STALLS.
//   • motor   — a 1-DOF force-clamped angular motor joint (WORLD → spindle): competes inside every iteration
//               up to maxTorque, so it spins the whole assembly up and HOLDS the target.
// The default (`drive=motor`) is the gate — it compiles the motor WGSL on the real device and asserts the
// spindle holds target ω under the load. `--param drive=forced` is the contrast (asserts it stalls), the
// same joint-vs-raw-contact carry gap the game's catch mechanism depends on. The per-step motor math is
// the f64 oracle's job (motor.oracle.ts).

const G = -10;
const DT = 1 / 60;
const CY = 2.0; // spindle axis height
const SPIN_HW = 0.3; // spindle half-width (light, thin — its own moment is negligible vs the flywheel)
const SPIN_HH = 1.0; // spindle half-height
const SPIN_MASS = 2;
const FW_HH = 0.2; // flywheel half-height (a wide thin disk)
const FW_HR = 1.5; // flywheel half-radius (wide → large moment of inertia)
const FW_MASS = 100; // HEAVY — its I_y ≈ 150 ≫ the spindle's ≈ 0.12, so the forced drive can't accelerate it
const MAX_TORQUE = 1500; // the motor clamp — ≫ the torque to spin the assembly up, so the motor holds target

const cfg = {
    dt: DT,
    gravity: G,
    alpha: 0.99,
    penalty: PENALTY_MIN,
    betaLin: 1e4,
    betaAng: 100,
    gamma: 0.999,
    iterations: 10,
    maxColors: 8,
};

let spinnerEid = -1;
let flywheelEid = -1;
let driveMode: "motor" | "forced" = "motor";
let rate = 4; // rad/s target
let bodyMirror: Mirror | null = null;
// omega2 = ω after a live setMotor speed change (motor drive only) — exercises the record-lane write
let result: { omega: number; swept: number; omega2: number; finite: boolean } | null = null;

// the forced drive (only installed for drive=forced): force the spindle's Y angular velocity each tick before
// the step. The COM is on the axle, so linear velocity is zeroed; the rigid coupling to the heavy flywheel is
// what the once-per-tick re-injection cannot accelerate — it stalls.
const ForcedPlugin: Plugin = {
    name: "MotorForcedDrive",
    systems: [
        {
            name: "motor-forced",
            group: "fixed",
            before: [StepSystem],
            update() {
                if (driveMode !== "forced" || spinnerEid < 0 || !Avbd.step) return;
                Avbd.step.setAngularVelocity(spinnerEid, 0, rate, 0);
                Avbd.step.setVelocity(spinnerEid, 0, 0, 0);
            },
        } satisfies System,
    ],
};

const scenario: Scenario = {
    name: "motor",
    params: [
        {
            key: "drive",
            type: "select",
            default: "motor",
            options: ["motor", "forced"],
            rebuild: true,
        },
        { key: "rate", type: "number", default: 4, min: 0, max: 20, step: 0.5 },
    ],

    async build(_canvas, p: Params) {
        result = null;
        driveMode = (p.drive as string) === "forced" ? "forced" : "motor";
        rate = p.rate as number;

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
                ForcedPlugin,
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

        Avbd.step?.configure(cfg);

        const floor = state.create();
        state.add(floor, Body);
        Body.shape.set(floor, ShapeKind.Box);
        Body.pos.set(floor, 0, -0.5, 0, 0);
        Body.halfExtents.set(floor, 20, 0.5, 20, 0);
        Body.mass.set(floor, 0);
        state.add(floor, Part);
        state.add(floor, Color);
        Color.rgba.set(floor, 0.28, 0.3, 0.34, 1);

        // the light spindle (the driven body) on a two-pin world axle (locks translation + tilt, frees Y-spin)
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

        // the heavy coaxial flywheel, fixed-jointed to the spindle's top — the rotational load. Its large moment
        // is the inertia the forced drive can't spin up (stall) but the motor's sustained clamped torque can.
        const fwY = CY + SPIN_HH + FW_HH;
        flywheelEid = state.create();
        state.add(flywheelEid, Body);
        Body.shape.set(flywheelEid, ShapeKind.Box);
        Body.pos.set(flywheelEid, 0, fwY, 0, 0);
        Body.halfExtents.set(flywheelEid, FW_HR, FW_HH, FW_HR, 0);
        Body.mass.set(flywheelEid, FW_MASS);
        Body.friction.set(flywheelEid, 0.5);
        state.add(flywheelEid, Part);
        state.add(flywheelEid, Color);
        Color.rgba.set(flywheelEid, 0.7, 0.5, 0.4, 1);

        // axle (two world spherical pins) + the rigid coupling (a FIXED joint, stiffnessAng ∞, locking the
        // flywheel to the spindle so they spin as one) + the motor (a WORLD → spindle force-clamped Y drive,
        // installed only for drive=motor; forced uses the plugin above instead).
        const joints: JointDef[] = [
            { a: WORLD, b: spinnerEid, rA: [0, CY + SPIN_HH, 0], rB: [0, SPIN_HH, 0] },
            { a: WORLD, b: spinnerEid, rA: [0, CY - SPIN_HH, 0], rB: [0, -SPIN_HH, 0] },
            {
                a: spinnerEid,
                b: flywheelEid,
                rA: [0, SPIN_HH, 0],
                rB: [0, -FW_HH, 0],
                stiffnessAng: Number.POSITIVE_INFINITY,
            },
        ];
        if (driveMode === "motor") {
            // stiffnessLin 0 — a PURE motor (no linear rows; the axle pins hold the COM), the config the spindle
            // game ships. The motor's 1-DOF angular drive is the only term this joint contributes.
            joints.push({
                a: WORLD,
                b: spinnerEid,
                rA: [0, CY, 0],
                rB: [0, 0, 0],
                stiffnessLin: 0,
                motor: { axis: [0, 1, 0], speed: rate, maxTorque: MAX_TORQUE },
            });
        }
        Avbd.step?.setJoints(joints);
        const motorIndex = joints.length - 1; // the motor joint's setJoints index (last), for the setMotor leg

        const cam = state.create();
        state.add(cam, Transform);
        state.add(cam, Camera);
        state.add(cam, Sear);
        state.add(cam, Orbit);
        Camera.mode.set(cam, CameraMode.Perspective);
        Orbit.yaw.set(cam, Math.PI / 5);
        Orbit.pitch.set(cam, Math.PI / 4);
        Orbit.distance.set(cam, 9);

        // warm: let the assembly spin up (the motor reaches target; the forced drive creeps)
        await frames(120);
        if (Avbd.step) bodyMirror = mirror(Avbd.step.bodies);

        // sample the spindle's angular speed about Y from B_VELA (col 7) over a window — the reliable signal; the
        // swept angle is its integral (the heavy flywheel sags the assembly a hair, so a quat-Y readout would be
        // polluted by the tilt, but ω.y stays clean). `finite` flips on any NaN.
        const sampleDt = 5 * DT;
        let finite = true;
        const measure = async (): Promise<{ omega: number; swept: number }> => {
            let omega = 0;
            let swept = 0;
            let samples = 0;
            for (let k = 0; k < 30; k++) {
                await frames(5);
                const snap = bodyMirror?.snapshot;
                if (!snap) continue;
                const s = new Float32Array(snap.bytes);
                const cap = s.length / (BODY_VEC4 * 4);
                const wy = s[(7 * cap + spinnerEid) * 4 + 1];
                if (!Number.isFinite(wy)) {
                    finite = false;
                    continue;
                }
                omega += wy;
                swept += wy * sampleDt;
                samples++;
            }
            return { omega: samples > 0 ? omega / samples : 0, swept };
        };

        const m1 = await measure();
        // live-drive leg (motor only): ramp speed + maxTorque via setMotor (no re-author — exercises both record
        // lane writes), then confirm ω follows the new target. The forced drive has no motor joint to retarget.
        let omega2 = 0;
        if (driveMode === "motor") {
            Avbd.step?.setMotor(motorIndex, rate * 1.5, MAX_TORQUE * 2);
            await frames(90);
            omega2 = (await measure()).omega;
        }
        result = { omega: m1.omega, swept: m1.swept, omega2, finite };

        return {
            state,
            dispose() {
                bodyMirror?.dispose();
                bodyMirror = null;
                spinnerEid = -1;
                flywheelEid = -1;
                dispose();
            },
        };
    },

    async assert(): Promise<Check[]> {
        const r = result;
        if (!r) return [{ name: "motor", pass: false, detail: "no measurement" }];
        const rate2 = rate * 1.5; // the live setMotor target (the leg below)
        const data = {
            omega: +r.omega.toFixed(3),
            swept: +r.swept.toFixed(3),
            omega2: +r.omega2.toFixed(3),
            rate,
        };
        if (driveMode === "motor") {
            // HOLDS: the motor spins the heavy assembly up to +target and holds it (steady ω ≈ +rate). Signed
            // bounds — the lower catches a stall OR a sign flip, the upper catches the deltaAngle overshoot (the
            // qsub-nonlinearity bug that an absolute reference would cause).
            const holds = r.omega > rate * 0.85 && r.omega < rate * 1.15;
            const spun = Math.abs(r.swept) > Math.PI; // genuinely rotated (multiple revolutions in the window)
            // TRACKS: after a live setMotor speed change, ω follows the new target — exercises the record-lane write
            const tracks = r.omega2 > rate2 * 0.85 && r.omega2 < rate2 * 1.15;
            return [
                { name: "finite", pass: r.finite, detail: "spindle stays finite (no NaN/blowup)" },
                {
                    name: "holds",
                    pass: holds,
                    detail: `motor holds ω ${data.omega} ≈ target ${rate} under the heavy load`,
                    data,
                },
                { name: "spun", pass: spun, detail: `swept ${data.swept} rad (> π)` },
                {
                    name: "tracks",
                    pass: tracks,
                    detail: `setMotor retargets ω ${data.omega2} ≈ new target ${rate2}`,
                },
            ];
        }
        // STALLS: the forced drive can't accelerate the rigidly-coupled flywheel — steady ω stays far below
        // target. This is the witness the motor resolves.
        const stalled = Math.abs(r.omega) < rate * 0.4;
        return [
            { name: "finite", pass: r.finite, detail: "spindle stays finite" },
            {
                name: "stalls",
                pass: stalled,
                detail: `forced drive stalls: ω ${data.omega} ≪ target ${rate} under the heavy load`,
                data,
            },
        ];
    },

    live(): string {
        const snap = bodyMirror?.snapshot;
        if (!snap) return "motor — warming";
        const s = new Float32Array(snap.bytes);
        const cap = s.length / (BODY_VEC4 * 4);
        const wy = s[(7 * cap + spinnerEid) * 4 + 1].toFixed(2);
        return `motor (${driveMode}) — spindle ω.y ${wy} / target ${rate}`;
    },
};

register(scenario);
