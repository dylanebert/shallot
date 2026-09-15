import { resolve } from "node:path";
import {
    Body,
    body,
    build,
    devices,
    InputPlugin,
    PhysicsPlugin,
    physicsStepConfig,
    pressKey,
    readBody,
    releaseKey,
    type State,
} from "@dylanebert/shallot";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { check } from "@dylanebert/shallot/harness/check";
import {
    Car,
    readVehicle,
    SPIN_FRAME,
    SUSPENSION_FRAME,
    VEHICLE_CONFIG,
    Vehicle,
    type VehicleObservation,
    VehicleRole,
} from "./car";

const SCENE = resolve(import.meta.dir, "../public/scenes/drive-a-vehicle.scene");
const HORIZON = 120;
const BOX3D_LINEAR_SLOP = 0.005;

type BodySnapshot = NonNullable<ReturnType<typeof readBody>>;
type Arm = readonly string[];
type Vec3 = readonly [number, number, number];
type Quat = readonly [number, number, number, number];
type AuthoredBody = { pos: Vec3; quat: Quat };

type TickSample = {
    tick: number;
    keys: Arm;
    bodies: BodySnapshot[];
    authored: AuthoredBody[];
    totalSpeeds: number[];
    horizontalSpeeds: number[];
    wheelGroundSeparations: number[];
    wheelAltitudes: number[];
    wheelJointSeparations: number[];
    observation: VehicleObservation;
};

type Trace = {
    keys: Arm;
    chassis: number;
    wheels: number[];
    bounds: VehicleBounds;
    initial: TickSample;
    warmup: TickSample[];
    samples: TickSample[];
};

type VehicleBounds = {
    dt: number;
    gravity: number;
    substeps: number;
    slop: number;
    clearAirTicks: number;
    groundTop: number;
    wheelRadius: number;
    speedCeiling: number;
    verticalSpeedCeiling: number;
    totalSpeedCeiling: number;
    wheelSpeedCeiling: number;
    altitudeCeiling: number;
    chassisGroundFloor: number;
    wheelAltitudeCeiling: number;
    suspensionExcursion: number;
    driveAcceleration: number;
    driveImpulse: number;
    effectiveTorque: number;
    torqueBudget: number;
};

async function vehicle() {
    // This is the manifest's selected scene and local plugin, consumed with the same public build seam
    // as the project. The exact-project Chromium row below proves the manifest selection itself.
    return build({
        defaults: false,
        plugins: [PhysicsPlugin, InputPlugin, Car],
        scene: SCENE,
    });
}

function role(state: State, wanted: number): number {
    for (const eid of state.query([Vehicle, Body])) {
        if (Vehicle.role.get(eid) === wanted) return eid;
    }
    return -1;
}

function chassis(state: State): number {
    const eid = role(state, VehicleRole.Chassis);
    if (eid < 0) throw new Error("actual vehicle scene has no chassis Body");
    return eid;
}

function wheels(state: State): number[] {
    const result = [...state.query([Vehicle, Body])].filter((eid) => {
        const value = Vehicle.role.get(eid);
        return value === VehicleRole.FrontWheel || value === VehicleRole.RearWheel;
    });
    if (result.length !== 4)
        throw new Error(
            `actual vehicle scene has ${result.length} live wheels; expected exactly four`,
        );
    return result;
}

function step(app: Awaited<ReturnType<typeof vehicle>>, ticks: number): void {
    const config = physicsStepConfig(app.state);
    for (let i = 0; i < ticks; i++) app.state.step(config.dt);
}

function tuplePos(eid: number): Vec3 {
    return [Body.pos.x.get(eid), Body.pos.y.get(eid), Body.pos.z.get(eid)];
}

function tupleQuat(eid: number): Quat {
    return [Body.quat.x.get(eid), Body.quat.y.get(eid), Body.quat.z.get(eid), Body.quat.w.get(eid)];
}

function distance(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function speed(velocity: BodySnapshot["vel"]): number {
    return Math.hypot(...velocity);
}

function horizontalSpeed(velocity: BodySnapshot["vel"]): number {
    return Math.hypot(velocity[0], velocity[2]);
}

function bounds(
    app: Awaited<ReturnType<typeof vehicle>>,
    chassisEid: number,
    wheelEids: readonly number[],
    observation: VehicleObservation,
): VehicleBounds {
    const ground = role(app.state, VehicleRole.Ground);
    if (ground < 0) throw new Error("actual vehicle scene has no ground Body");
    const stepConfig = physicsStepConfig(app.state);
    const groundTop = Body.pos.y.get(ground) + Body.halfExtents.y.get(ground);
    const wheelRadius = Body.halfExtents.w.get(wheelEids[0]);
    const wheelBottom = Body.pos.y.get(wheelEids[0]) - wheelRadius;
    const initialClearance = wheelBottom - groundTop;
    const mass = [chassisEid, ...wheelEids].reduce((sum, eid) => sum + Body.mass.get(eid), 0);
    const rear = observation.wheels.filter((wheel) => wheel.role === VehicleRole.RearWheel);
    if (rear.length !== 2)
        throw new Error(`actual vehicle has ${rear.length} effective rear joints`);
    const suspensionExcursion = Math.max(
        ...observation.wheels.map((wheel) =>
            Math.max(Math.abs(wheel.suspension.lower), Math.abs(wheel.suspension.upper)),
        ),
    );
    const wheelFrameAnchor = Math.max(
        ...observation.wheels.map((wheel) => Math.abs(wheel.localFrameA.p.y)),
    );
    const slop = BOX3D_LINEAR_SLOP;
    const clearAirSeconds = Math.sqrt(
        Math.max(0, (2 * (initialClearance - slop)) / Math.abs(stepConfig.gravity)),
    );
    const anchorDrop = Math.abs(Body.pos.y.get(chassisEid) - Body.pos.y.get(wheelEids[0]));
    const chassisHalfHeight = Body.halfExtents.y.get(chassisEid);
    const chassisInitialClearance = Body.pos.y.get(chassisEid) - chassisHalfHeight - groundTop;
    const fixedStepDrop = 0.5 * Math.abs(stepConfig.gravity) * stepConfig.dt ** 2;
    const chassisGroundFloor =
        groundTop + wheelRadius + anchorDrop - suspensionExcursion - 4 * slop - fixedStepDrop;
    const verticalSpeedCeiling =
        Math.sqrt(2 * Math.abs(stepConfig.gravity) * Math.max(0, chassisInitialClearance)) +
        Math.abs(stepConfig.gravity) * stepConfig.dt +
        (4 * slop) / stepConfig.dt;
    const altitudeCeiling =
        Math.max(
            Body.pos.y.get(chassisEid),
            groundTop + wheelRadius + anchorDrop + suspensionExcursion,
        ) +
        2 * wheelRadius +
        slop;
    const wheelAltitudeCeiling =
        altitudeCeiling + wheelFrameAnchor + suspensionExcursion + wheelRadius + slop;
    const effectiveTorque = Math.max(...rear.map((wheel) => Math.abs(wheel.spin.maxTorque)));
    const rimSpeed = Math.abs(VEHICLE_CONFIG.throttle) * wheelRadius;
    const driveAcceleration =
        (rear.length * Math.abs(VEHICLE_CONFIG.maxSpinTorque)) / (wheelRadius * mass);
    return {
        dt: stepConfig.dt,
        gravity: stepConfig.gravity,
        substeps: stepConfig.substeps,
        slop,
        clearAirTicks: Math.max(0, Math.ceil(clearAirSeconds / stepConfig.dt) - 1),
        groundTop,
        wheelRadius,
        speedCeiling: rimSpeed + (2 * slop) / stepConfig.dt,
        verticalSpeedCeiling,
        totalSpeedCeiling: Math.hypot(rimSpeed + (2 * slop) / stepConfig.dt, verticalSpeedCeiling),
        wheelSpeedCeiling:
            rimSpeed + (suspensionExcursion + slop) / stepConfig.dt + (2 * slop) / stepConfig.dt,
        altitudeCeiling,
        chassisGroundFloor,
        wheelAltitudeCeiling,
        suspensionExcursion,
        driveAcceleration,
        driveImpulse:
            driveAcceleration * stepConfig.dt + slop / (stepConfig.dt / stepConfig.substeps),
        effectiveTorque,
        torqueBudget: Math.abs(VEHICLE_CONFIG.maxSpinTorque),
    };
}

function capture(
    app: Awaited<ReturnType<typeof vehicle>>,
    eids: readonly number[],
    wheelEids: readonly number[],
    tick: number,
    keys: Arm,
    groundTop: number,
): TickSample {
    const bodies = eids.map((eid) => {
        const bodyState = readBody(app.state, eid);
        if (!bodyState) throw new Error(`vehicle body ${eid} has no live stepped pose`);
        return bodyState;
    });
    const observation = readVehicle(app.state);
    if (!observation) throw new Error("vehicle joints were not wired before trajectory sampling");
    const authored = eids.map((eid) => ({ pos: tuplePos(eid), quat: tupleQuat(eid) }));
    const wheelGroundSeparations = wheelEids.map(
        (eid) => bodies[eids.indexOf(eid)].pos[1] - Body.halfExtents.w.get(eid) - groundTop,
    );
    const wheelAltitudes = wheelEids.map((eid) => bodies[eids.indexOf(eid)].pos[1]);
    const wheelJointSeparations = wheelEids.map((eid) => {
        const wheel = observation.wheels.find((value) => value.eid === eid);
        if (!wheel) throw new Error(`vehicle observation lost wheel ${eid}`);
        return wheel.linearSeparation;
    });
    return {
        tick,
        keys,
        bodies,
        authored,
        totalSpeeds: bodies.map((bodyState) => speed(bodyState.vel)),
        horizontalSpeeds: bodies.map((bodyState) => horizontalSpeed(bodyState.vel)),
        wheelGroundSeparations,
        wheelAltitudes,
        wheelJointSeparations,
        observation,
    };
}

function diagnostic(trace: Trace, index: number, boundsValue: VehicleBounds): string {
    const samples = [trace.initial, ...trace.samples];
    const actualIndex = index + 1;
    const from = Math.max(0, actualIndex - 1);
    const to = Math.min(samples.length, actualIndex + 2);
    const sample = samples.slice(from, to).map((value) => ({
        tick: value.tick,
        activeKeys: value.keys,
        bodies: value.bodies.map((bodyState, bodyIndex) => ({
            body: bodyIndex === 0 ? "chassis" : `wheel ${bodyIndex}`,
            pos: bodyState.pos,
            vel: bodyState.vel,
            quat: bodyState.quat,
            quatNorm: Math.hypot(...bodyState.quat),
            authored: value.authored[bodyIndex],
            totalSpeed: value.totalSpeeds[bodyIndex],
            horizontalSpeed: value.horizontalSpeeds[bodyIndex],
        })),
        wheelGroundSeparations: value.wheelGroundSeparations,
        wheelAltitudes: value.wheelAltitudes,
        wheelJointSeparations: value.wheelJointSeparations,
        effectiveVehicle: value.observation,
    }));
    return JSON.stringify({
        firstFailingTick: index,
        activeKeys: trace.keys,
        bounds: boundsValue,
        sample,
    });
}

function fail(trace: Trace, index: number, boundsValue: VehicleBounds, reason: string): never {
    throw new Error(
        `${reason}; first failing body/invariant at tick ${index}; diagnostics=${diagnostic(trace, index, boundsValue)}`,
    );
}

function diagnosticSample(value: TickSample): object {
    return {
        tick: value.tick,
        activeKeys: value.keys,
        bodies: value.bodies.map((bodyState, bodyIndex) => ({
            body: bodyIndex === 0 ? "chassis" : `wheel ${bodyIndex}`,
            pos: bodyState.pos,
            vel: bodyState.vel,
            quat: bodyState.quat,
            quatNorm: Math.hypot(...bodyState.quat),
            authored: value.authored[bodyIndex],
            totalSpeed: value.totalSpeeds[bodyIndex],
            horizontalSpeed: value.horizontalSpeeds[bodyIndex],
        })),
        wheelGroundSeparations: value.wheelGroundSeparations,
        wheelAltitudes: value.wheelAltitudes,
        wheelJointSeparations: value.wheelJointSeparations,
        effectiveVehicle: value.observation,
    };
}

function pairedDiagnostic(
    label: string,
    left: Trace,
    right: Trace,
    index: number,
    boundsValue: VehicleBounds,
): string {
    const window = (trace: Trace) => {
        const samples = [trace.initial, ...trace.samples];
        const actualIndex = Math.min(samples.length - 1, index + 1);
        return samples
            .slice(Math.max(0, actualIndex - 1), Math.min(samples.length, actualIndex + 2))
            .map(diagnosticSample);
    };
    return JSON.stringify({
        firstDivergenceTick: index,
        label,
        bounds: boundsValue,
        arms: { left: window(left), right: window(right) },
    });
}

function warmupPairedDiagnostic(label: string, left: Trace, right: Trace, index: number): string {
    const from = Math.max(0, index - 1);
    const to = Math.min(left.warmup.length, index + 2);
    return JSON.stringify({
        firstDivergenceWarmup: index,
        label,
        arms: {
            left: left.warmup.slice(from, to).map(diagnosticSample),
            right: right.warmup.slice(from, to).map(diagnosticSample),
        },
    });
}

function firstDivergence(
    left: Trace,
    right: Trace,
    differs: (left: TickSample, right: TickSample) => boolean,
): number | null {
    const count = Math.min(left.samples.length, right.samples.length);
    for (let index = 0; index < count; index++) {
        if (differs(left.samples[index], right.samples[index])) return index;
    }
    return null;
}

function validateTrace(trace: Trace, idle: boolean): void {
    let previous = trace.initial;
    for (const sample of trace.samples) {
        const bodies = sample.bodies;
        for (const [index, bodyState] of bodies.entries()) {
            const values = [...bodyState.pos, ...bodyState.vel, ...bodyState.quat];
            if (values.some((value) => !Number.isFinite(value)))
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `${index === 0 ? "chassis" : `wheel ${index}`} has non-finite state`,
                );
            if (Math.abs(Math.hypot(...bodyState.quat) - 1) > 1e-5)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `${index === 0 ? "chassis" : `wheel ${index}`} quaternion is not unit`,
                );
            const displacement = distance(bodyState.pos, previous.bodies[index].pos);
            if (
                displacement >
                Math.max(sample.totalSpeeds[index], previous.totalSpeeds[index]) * trace.bounds.dt +
                    trace.bounds.slop
            )
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `${index === 0 ? "chassis" : `wheel ${index}`} moved ${displacement.toFixed(4)}m without endpoint velocity support`,
                );
        }
        const chassisState = bodies[0];
        if (sample.horizontalSpeeds[0] > trace.bounds.speedCeiling)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `chassis horizontal speed ${sample.horizontalSpeeds[0].toFixed(3)} exceeded ${trace.bounds.speedCeiling.toFixed(3)}`,
            );
        if (Math.abs(chassisState.vel[1]) > trace.bounds.verticalSpeedCeiling)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `chassis vertical speed ${Math.abs(chassisState.vel[1]).toFixed(3)} exceeded ${trace.bounds.verticalSpeedCeiling.toFixed(3)}`,
            );
        if (sample.totalSpeeds[0] > trace.bounds.totalSpeedCeiling)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `chassis total speed ${sample.totalSpeeds[0].toFixed(3)} exceeded ${trace.bounds.totalSpeedCeiling.toFixed(3)}`,
            );
        if (chassisState.pos[1] < trace.bounds.chassisGroundFloor)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `chassis center altitude ${chassisState.pos[1].toFixed(3)} fell below derived ground floor ${trace.bounds.chassisGroundFloor.toFixed(3)}`,
            );
        if (chassisState.pos[1] > trace.bounds.altitudeCeiling)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `chassis altitude ${chassisState.pos[1].toFixed(3)} exceeded ${trace.bounds.altitudeCeiling.toFixed(3)}`,
            );
        const up = rotate(chassisState.quat, [0, 1, 0]);
        if (up[1] < 0.5)
            fail(trace, sample.tick, trace.bounds, `chassis inverted with up axis ${up}`);
        if (sample.observation.wheels.length !== 4)
            fail(
                trace,
                sample.tick,
                trace.bounds,
                `effective vehicle exposed ${sample.observation.wheels.length} wheels, expected four`,
            );
        for (const [wheelIndex] of bodies.slice(1).entries()) {
            if (sample.totalSpeeds[wheelIndex + 1] > trace.bounds.wheelSpeedCeiling)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} speed exceeded ${trace.bounds.wheelSpeedCeiling.toFixed(3)}`,
                );
            if (sample.wheelAltitudes[wheelIndex] > trace.bounds.wheelAltitudeCeiling)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} altitude exceeded ${trace.bounds.wheelAltitudeCeiling.toFixed(3)}`,
                );
            const contactTolerance =
                4 * trace.bounds.slop + Math.max(0, -chassisState.vel[1]) * trace.bounds.dt;
            if (sample.wheelGroundSeparations[wheelIndex] < -contactTolerance)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} penetrated ground by ${(-sample.wheelGroundSeparations[wheelIndex]).toFixed(3)}m`,
                );
            if (
                Math.abs(sample.wheelJointSeparations[wheelIndex]) >
                trace.bounds.suspensionExcursion + trace.bounds.slop
            )
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} joint separation ${sample.wheelJointSeparations[wheelIndex].toFixed(4)} exceeded suspension excursion`,
                );
            const wheel = sample.observation.wheels.find(
                (value) => value.eid === trace.wheels[wheelIndex],
            );
            if (!wheel)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} is absent from effective observation`,
                );
            // Target and applied values remain in every retained observation; causality is judged from
            // sampled motion below, so command mutations red the dynamic invariant rather than a setup seam.
            if (Math.abs(wheel.spin.torque) > trace.bounds.torqueBudget + 1e-4)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `wheel ${wheelIndex} applied spin torque ${wheel.spin.torque} exceeded recipe budget ${trace.bounds.torqueBudget}`,
                );
        }
        previous = sample;
    }
    // Keep dynamic idle causality ahead of command-report checks: an uncommanded normal-speed motor
    // must first red for motion, not merely because its target was still observable at release.
    if (idle) {
        for (const sample of trace.samples) {
            const drift = Math.hypot(
                sample.bodies[0].pos[0] - trace.initial.bodies[0].pos[0],
                sample.bodies[0].pos[2] - trace.initial.bodies[0].pos[2],
            );
            if (
                drift > 4 * trace.bounds.slop ||
                sample.horizontalSpeeds[0] > (2 * trace.bounds.slop) / trace.bounds.dt
            )
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `idle acquired horizontal drift/speed after contact: ${drift.toFixed(4)}m / ${sample.horizontalSpeeds[0].toFixed(4)}m/s`,
                );
        }
    }
    const released = trace.samples[trace.samples.length - 1];
    for (const wheel of released.observation.wheels) {
        if (Math.abs(wheel.spin.target) > 1e-5 || Math.abs(wheel.steering.target) > 1e-5)
            fail(
                trace,
                released.tick,
                trace.bounds,
                "vehicle command target remained active after key release",
            );
    }
}

function rotate(q: Quat, v: Vec3): [number, number, number] {
    const [x, y, z, w] = q;
    const [vx, vy, vz] = v;
    return [
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ];
}

function quat(value: { v: { x: number; y: number; z: number }; s: number }): Quat {
    return [value.v.x, value.v.y, value.v.z, value.s];
}

function multiply(a: Quat, b: Quat): Quat {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ];
}

function assertFramePremise(chassisEid: number, wheelEids: readonly number[]): void {
    const chassisPos = tuplePos(chassisEid);
    const suspensionAxis = rotate(quat(SUSPENSION_FRAME), [1, 0, 0]);
    const parallelAxis = rotate(quat(SPIN_FRAME), [0, 0, 1]);
    if (distance(suspensionAxis, [0, 1, 0]) > 1e-5)
        throw new Error(`wheel suspension frame x axis is not world Y: ${suspensionAxis}`);
    if (distance(parallelAxis, [0, 1, 0]) > 1e-5)
        throw new Error(`upright parallel frame z axis is not world Y: ${parallelAxis}`);
    for (const eid of wheelEids) {
        const q = tupleQuat(eid);
        if (Math.abs(Math.hypot(...q) - 1) > 1e-5)
            throw new Error(`wheel ${eid} stored a non-unit authored quaternion: ${q}`);
        const axle = rotate(multiply(q, quat(SPIN_FRAME)), [0, 0, 1]);
        if (distance(axle, [0, 0, 1]) > 1e-5)
            throw new Error(`wheel ${eid} spin frame does not realize the world-Z axle: ${axle}`);
        const anchor = [
            Body.pos.x.get(eid) - chassisPos[0],
            Body.pos.y.get(eid) - chassisPos[1],
            Body.pos.z.get(eid) - chassisPos[2],
        ] as const;
        const expected = [
            Vehicle.role.get(eid) === VehicleRole.FrontWheel ? 1.5 : -1.5,
            -1,
            Math.sign(Body.pos.z.get(eid)) * 0.8,
        ] as const;
        if (distance(anchor, expected) > 1e-5)
            throw new Error(
                `wheel ${eid} anchor is not coincident with its authored chassis frame: ${anchor}`,
            );
    }
}

async function runTrace(keys: Arm): Promise<Trace> {
    const app = await vehicle();
    try {
        const chassisEid = chassis(app.state);
        const wheelEids = wheels(app.state);
        const eids = [chassisEid, ...wheelEids];
        step(app, 1);
        const firstObservation = readVehicle(app.state);
        if (!firstObservation)
            throw new Error("actual vehicle did not wire after its first fixed step");
        const bound = bounds(app, chassisEid, wheelEids, firstObservation);
        const warmup: TickSample[] = [capture(app, eids, wheelEids, -1, [], bound.groundTop)];
        step(app, 1);
        warmup.push(capture(app, eids, wheelEids, -0, [], bound.groundTop));
        const initial = warmup[warmup.length - 1];
        for (const key of keys) pressKey(app.state, key);
        const samples: TickSample[] = [];
        for (let tick = 0; tick < HORIZON; tick++) {
            app.state.step(bound.dt);
            samples.push(capture(app, eids, wheelEids, tick, keys, bound.groundTop));
        }
        for (const key of keys) releaseKey(app.state, key);
        app.state.step(bound.dt);
        samples.push(capture(app, eids, wheelEids, HORIZON, [], bound.groundTop));
        const trace = {
            keys,
            chassis: chassisEid,
            wheels: wheelEids.slice(),
            bounds: bound,
            initial,
            warmup,
            samples,
        };
        validateTrace(trace, keys.length === 0);
        return trace;
    } finally {
        app.dispose();
    }
}

function assertSameWarmup(traces: readonly Trace[]): void {
    const reference = traces[0];
    for (const trace of traces.slice(1)) {
        for (let index = 0; index < reference.warmup.length; index++) {
            const a = reference.warmup[index];
            const b = trace.warmup[index];
            for (let bodyIndex = 0; bodyIndex < a.bodies.length; bodyIndex++) {
                if (
                    distance(a.bodies[bodyIndex].pos, b.bodies[bodyIndex].pos) > 1e-5 ||
                    distance(a.bodies[bodyIndex].vel, b.bodies[bodyIndex].vel) > 1e-5 ||
                    Math.hypot(
                        a.bodies[bodyIndex].quat[0] - b.bodies[bodyIndex].quat[0],
                        a.bodies[bodyIndex].quat[1] - b.bodies[bodyIndex].quat[1],
                        a.bodies[bodyIndex].quat[2] - b.bodies[bodyIndex].quat[2],
                        a.bodies[bodyIndex].quat[3] - b.bodies[bodyIndex].quat[3],
                    ) > 1e-5
                )
                    throw new Error(
                        `fresh vehicle arms diverged before input injection at warm-up ${index}; diagnostics=${warmupPairedDiagnostic("warm-up", reference, trace, index)}`,
                    );
            }
        }
    }
}

function heading(q: Quat): number {
    const forward = rotate(q, [1, 0, 0]);
    return Math.atan2(forward[2], forward[0]);
}

function causalDelta(trace: Trace, idle: Trace, index: number): Vec3 {
    return [
        trace.samples[index].bodies[0].pos[0] - idle.samples[index].bodies[0].pos[0],
        trace.samples[index].bodies[0].pos[1] - idle.samples[index].bodies[0].pos[1],
        trace.samples[index].bodies[0].pos[2] - idle.samples[index].bodies[0].pos[2],
    ];
}

check(
    "drive-a-vehicle authored frames and effective joints are valid",
    {
        claim: "the actual vehicle scene's authored wheel frames and production joint observations establish the semantic suspension, axle, anchor and upright premises",
    },
    async () => {
        const app = await vehicle();
        try {
            const chassisEid = chassis(app.state);
            const wheelEids = wheels(app.state);
            step(app, 2);
            assertFramePremise(chassisEid, wheelEids);
            const observation = readVehicle(app.state);
            if (observation?.wheels.length !== 4 || observation?.upright === null)
                throw new Error("actual vehicle did not expose four wheels and an upright joint");
        } finally {
            app.dispose();
        }
    },
);

check(
    "drive-a-vehicle actual scene follows a bounded causal trajectory",
    {
        claim: "the actual drive-a-vehicle scene follows a finite, ground-bound, input-causal stepped trajectory rather than receiving an uncommanded launch",
        size: "integration",
        subject: ["examples/drive-a-vehicle"],
    },
    async () => {
        const idle = await runTrace([]);
        const forward = await runTrace(["KeyW"]);
        const reverse = await runTrace(["KeyS"]);
        const left = await runTrace(["KeyW", "KeyA"]);
        const right = await runTrace(["KeyW", "KeyD"]);
        assertSameWarmup([idle, forward, reverse, left, right]);
        const deadline = Math.min(HORIZON - 1, idle.bounds.clearAirTicks + 10);
        const causalMinimum = Math.max(4 * idle.bounds.slop, idle.bounds.wheelRadius * 0.1);
        const forwardDelta = causalDelta(forward, idle, deadline);
        const reverseDelta = causalDelta(reverse, idle, deadline);
        const forwardFirst = firstDivergence(
            forward,
            idle,
            (a, b) => Math.abs(a.bodies[0].pos[0] - b.bodies[0].pos[0]) > causalMinimum,
        );
        const reverseFirst = firstDivergence(
            reverse,
            idle,
            (a, b) => Math.abs(a.bodies[0].pos[0] - b.bodies[0].pos[0]) > causalMinimum,
        );
        if (forwardDelta[0] < causalMinimum || reverseDelta[0] > -causalMinimum)
            throw new Error(
                `W/S command-minus-idle causality failed by deadline ${deadline}; first divergences W=${forwardFirst ?? deadline} S=${reverseFirst ?? deadline}; W=${pairedDiagnostic("W vs idle", forward, idle, forwardFirst ?? deadline, idle.bounds)}; S=${pairedDiagnostic("S vs idle", reverse, idle, reverseFirst ?? deadline, idle.bounds)}`,
            );
        const leftDelta = causalDelta(left, forward, HORIZON - 1);
        const rightDelta = causalDelta(right, forward, HORIZON - 1);
        const leftFirst = firstDivergence(
            left,
            forward,
            (a, b) => Math.abs(a.bodies[0].pos[2] - b.bodies[0].pos[2]) > causalMinimum,
        );
        const rightFirst = firstDivergence(
            right,
            forward,
            (a, b) => Math.abs(a.bodies[0].pos[2] - b.bodies[0].pos[2]) > causalMinimum,
        );
        if (
            leftDelta[2] > -causalMinimum ||
            rightDelta[2] < causalMinimum ||
            leftDelta[2] * rightDelta[2] >= 0
        )
            throw new Error(
                `W+A/W+D command-minus-forward steering failed by tick ${HORIZON - 1}; first divergences A=${leftFirst ?? HORIZON - 1} D=${rightFirst ?? HORIZON - 1}; A=${pairedDiagnostic("W+A vs W", left, forward, leftFirst ?? HORIZON - 1, idle.bounds)}; D=${pairedDiagnostic("W+D vs W", right, forward, rightFirst ?? HORIZON - 1, idle.bounds)}`,
            );
        const forwardRear = forward.samples[HORIZON - 1].observation.wheels.filter(
            (wheel) => wheel.role === VehicleRole.RearWheel,
        );
        if (forwardRear.length !== 2)
            throw new Error(`straight W arm exposed ${forwardRear.length} rear wheels`);
        const rearDelta = Math.abs(
            forwardRear[0].linearSeparation - forwardRear[1].linearSeparation,
        );
        if (rearDelta > idle.bounds.slop)
            throw new Error(`straight W arm rear-wheel separation was asymmetric by ${rearDelta}`);
        const drivenDelta = Math.hypot(
            forward.samples[0].bodies[0].vel[0] - idle.samples[0].bodies[0].vel[0],
            forward.samples[0].bodies[0].vel[2] - idle.samples[0].bodies[0].vel[2],
        );
        if (drivenDelta > idle.bounds.driveImpulse)
            throw new Error(
                `first W-minus-idle horizontal velocity divergence exceeded ${idle.bounds.driveImpulse.toFixed(4)}m/s; diagnostics=${pairedDiagnostic("first W vs idle impulse", forward, idle, 0, idle.bounds)}`,
            );
    },
);

check(
    "vehicle throttle displaces the actual chassis",
    {
        claim: "the actual drive-a-vehicle recipe turns W into chassis displacement through its production wheel motor",
    },
    async () => {
        const driven = await runTrace(["KeyW"]);
        const before = driven.initial.bodies[0];
        const after = driven.samples[24].bodies[0];
        if (after.pos[0] <= before.pos[0] || after.vel[0] <= 0.1)
            throw new Error(
                `actual throttle did not produce an early forward response: displacement=${(after.pos[0] - before.pos[0]).toFixed(4)}m velocity=${after.vel[0].toFixed(4)}m/s`,
            );
    },
);

check(
    "vehicle opposite steering gives opposite heading",
    {
        claim: "opposite A/D steering commands produce opposite signed chassis heading through the actual front wheel targets",
    },
    async () => {
        const left = await runTrace(["KeyW", "KeyA"]);
        const right = await runTrace(["KeyW", "KeyD"]);
        const leftVelocity = left.samples[20].bodies[0].vel[2];
        const rightVelocity = right.samples[20].bodies[0].vel[2];
        const leftHeading = heading(left.samples[HORIZON - 1].bodies[0].quat);
        const rightHeading = heading(right.samples[HORIZON - 1].bodies[0].quat);
        if (
            leftVelocity >= -0.001 ||
            rightVelocity <= 0.001 ||
            leftVelocity * rightVelocity >= 0 ||
            leftHeading >= -0.02 ||
            rightHeading <= 0.02 ||
            leftHeading * rightHeading >= 0
        )
            throw new Error(
                `actual steering lateral/yaw response was not opposite: lateral=${leftVelocity},${rightVelocity} heading=${leftHeading},${rightHeading}; diagnostics=${pairedDiagnostic("opposite W+A/W+D steering", left, right, HORIZON - 1, left.bounds)}`,
            );
    },
);

check(
    "vehicle upright joint recovers a bounded roll disturbance",
    {
        claim: "the actual vehicle's production upright parallel joint restores chassis up-axis after a bounded public-seam roll impulse",
    },
    async () => {
        const app = await vehicle();
        try {
            const eid = chassis(app.state);
            step(app, 2);
            const observation = readVehicle(app.state);
            if (!observation) throw new Error("vehicle observation is absent");
            const live = body(app.state, eid);
            if (!live) throw new Error("actual chassis never became a live body");
            live.applyAngularImpulse({ x: 10, y: 0, z: 0 }, true);
            for (let i = 0; i < 30; i++) app.state.step(physicsStepConfig(app.state).dt);
            const after = readBody(app.state, eid);
            if (!after) throw new Error("disturbed chassis could not be read");
            const up = rotate(after.quat, [0, 1, 0]);
            if (up[1] < 0.9)
                throw new Error(`bounded roll did not recover upright chassis: up=${up}`);
            const current = readVehicle(app.state);
            if (!current) throw new Error("vehicle observation disappeared during recovery");
        } finally {
            app.dispose();
        }
    },
);

check(
    "drive-a-vehicle exact project composes its selected scene and plugin",
    {
        claim: "the exact drive-a-vehicle manifest builds its selected scene and local Car role plugin before disposal",
        size: "integration",
        requires: ["chromium"],
        host: "mac",
        subject: ["examples/drive-a-vehicle"],
    },
    async () =>
        runBrowserCheck((port) => [
            process.execPath,
            resolve(import.meta.dir, "../../../scripts/fixtures/recipe-composition-serve.ts"),
            "--port",
            String(port),
            "--project",
            resolve(import.meta.dir, ".."),
            "--recipe",
            "vehicle",
        ]),
);

check(
    "vehicle command wakes the actual sleeping chassis",
    {
        claim: "a production throttle command moves the actual sleeping chassis without a recipe caller wake loop",
    },
    async () => {
        const app = await vehicle();
        try {
            const eid = chassis(app.state);
            step(app, 2);
            const parked = body(app.state, eid);
            if (!parked) throw new Error("actual chassis never became a live body");
            parked.setAwake(false);
            const before = readBody(app.state, eid);
            pressKey(app.state, "KeyW");
            step(app, 8);
            releaseKey(app.state, "KeyW");
            const after = readBody(app.state, eid);
            if (!before || !after) throw new Error("sleeping actual chassis could not be read");
            const displacement = Math.hypot(
                after.pos[0] - before.pos[0],
                after.pos[2] - before.pos[2],
            );
            if (displacement < 0.001)
                throw new Error(
                    `sleeping actual command produced only ${displacement.toFixed(3)}m`,
                );
            if (devices(app.state).keys.held.has("KeyW"))
                throw new Error("W remained held after release");
        } finally {
            app.dispose();
        }
    },
);
