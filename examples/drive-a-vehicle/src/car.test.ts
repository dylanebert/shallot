import { resolve } from "node:path";
import {
    Body,
    body,
    build,
    InputPlugin,
    PhysicsPlugin,
    physicsStepConfig,
    pressKey,
    readBody,
    releaseKey,
    type State,
} from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import {
    Car,
    readVehicle,
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
    samples: TickSample[];
};

type VehicleBounds = {
    dt: number;
    gravity: number;
    substeps: number;
    slop: number;
    // A count, not an inclusive final tick: admitted samples satisfy tick < clearAirTicks.
    clearAirTicks: number;
    clearAirSteps: number;
    clearAirWheelTimes: number[];
    clearAirWheelSeparations: number[];
    clearAirWheelVerticalVelocities: number[];
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
    const rimSpeed = Math.abs(VEHICLE_CONFIG.throttle) * wheelRadius;
    const driveAcceleration =
        (rear.length * Math.abs(VEHICLE_CONFIG.maxSpinTorque)) / (wheelRadius * mass);
    return {
        dt: stepConfig.dt,
        gravity: stepConfig.gravity,
        substeps: stepConfig.substeps,
        slop,
        clearAirTicks: 0,
        clearAirSteps: 0,
        clearAirWheelTimes: [],
        clearAirWheelSeparations: [],
        clearAirWheelVerticalVelocities: [],
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
        torqueBudget: Math.abs(VEHICLE_CONFIG.maxSpinTorque),
    };
}

function earliestContactSeconds(
    wheelGroundSeparation: number,
    verticalVelocity: number,
    gravity: number,
    slop: number,
): number {
    const remainingClearance = wheelGroundSeparation - slop;
    if (!(remainingClearance > 0)) return 0;
    const downwardAcceleration = Math.abs(gravity);
    if (!(downwardAcceleration > 0))
        return verticalVelocity < 0
            ? remainingClearance / -verticalVelocity
            : Number.POSITIVE_INFINITY;
    const discriminant =
        verticalVelocity * verticalVelocity + 2 * downwardAcceleration * remainingClearance;
    return Math.max(
        0,
        (verticalVelocity + Math.sqrt(Math.max(0, discriminant))) / downwardAcceleration,
    );
}

function deriveClearAirBounds(bound: VehicleBounds, initial: TickSample): VehicleBounds {
    const clearAirWheelSeparations = initial.wheelGroundSeparations.slice();
    const clearAirWheelVerticalVelocities = initial.bodies
        .slice(1)
        .map((bodyState) => bodyState.vel[1]);
    const clearAirWheelTimes = clearAirWheelSeparations.map((separation, index) =>
        earliestContactSeconds(
            separation,
            clearAirWheelVerticalVelocities[index],
            bound.gravity,
            bound.slop,
        ),
    );
    const clearAirSeconds = Math.min(...clearAirWheelTimes);
    const clearAirSteps = clearAirSeconds / bound.dt;
    return {
        ...bound,
        clearAirTicks: Math.max(0, Math.ceil(clearAirSteps - 1)),
        clearAirSteps,
        clearAirWheelTimes,
        clearAirWheelSeparations,
        clearAirWheelVerticalVelocities,
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
    return JSON.stringify({
        firstFailingTick: index,
        activeKeys: trace.keys,
        bounds: boundsValue,
        sample: samples
            .slice(Math.max(0, actualIndex - 1), Math.min(samples.length, actualIndex + 2))
            .map(diagnosticSample),
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
    const fromRestWheelTimes = trace.bounds.clearAirWheelSeparations.map((separation) =>
        earliestContactSeconds(separation, 0, trace.bounds.gravity, trace.bounds.slop),
    );
    const fromRestSteps = Math.min(...fromRestWheelTimes) / trace.bounds.dt;
    const hasLiveDownwardWheel = trace.bounds.clearAirWheelVerticalVelocities.some(
        (velocity) => velocity < -1e-6,
    );
    if (idle && hasLiveDownwardWheel && !(trace.bounds.clearAirSteps < fromRestSteps))
        fail(
            trace,
            -1,
            trace.bounds,
            `clear-air horizon ignored live downward wheel velocity: live=${trace.bounds.clearAirSteps.toFixed(4)} steps, from-rest=${fromRestSteps.toFixed(4)} steps`,
        );
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
        // clearAirTicks is a count with a non-inclusive endpoint: sample tick t is the (t + 1)th
        // fixed step after trace.initial, and every admitted step must remain strictly before possible
        // wheel contact derived from the live initial wheel/ground separation.
        if (idle && sample.tick < trace.bounds.clearAirTicks) {
            const admittedStep = sample.tick + 1;
            if (!(admittedStep < trace.bounds.clearAirSteps))
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `clear-air admitted step ${admittedStep} was not strictly before possible contact at ${trace.bounds.clearAirSteps.toFixed(4)}`,
                );
            const initialChassis = trace.initial.bodies[0];
            const precedingChassis = previous.bodies[0];
            const xDelta = chassisState.pos[0] - initialChassis.pos[0];
            const zDelta = chassisState.pos[2] - initialChassis.pos[2];
            const clearAirSpeed = trace.bounds.slop / trace.bounds.dt;
            if (Math.abs(xDelta) > trace.bounds.slop || Math.abs(zDelta) > trace.bounds.slop)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `clear-air chassis X/Z moved beyond slop: ${xDelta.toFixed(4)}m / ${zDelta.toFixed(4)}m`,
                );
            if (sample.horizontalSpeeds[0] > clearAirSpeed)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `clear-air chassis horizontal speed ${sample.horizontalSpeeds[0].toFixed(4)} exceeded ${clearAirSpeed.toFixed(4)}`,
                );
            if (chassisState.pos[1] > precedingChassis.pos[1] + trace.bounds.slop)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `clear-air chassis Y increased by ${(chassisState.pos[1] - precedingChassis.pos[1]).toFixed(4)}m from preceding sample beyond slop`,
                );
            if (chassisState.vel[1] > clearAirSpeed)
                fail(
                    trace,
                    sample.tick,
                    trace.bounds,
                    `clear-air chassis upward velocity ${chassisState.vel[1].toFixed(4)} exceeded ${clearAirSpeed.toFixed(4)}`,
                );
        }
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
        const preliminaryBound = bounds(app, chassisEid, wheelEids, firstObservation);
        step(app, 1);
        const initial = capture(app, eids, wheelEids, -0, [], preliminaryBound.groundTop);
        const bound = deriveClearAirBounds(preliminaryBound, initial);
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
            samples,
        };
        validateTrace(trace, keys.length === 0);
        return trace;
    } finally {
        app.dispose();
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
    "drive-a-vehicle actual scene follows a bounded causal trajectory",
    {
        claim: "the actual drive-a-vehicle scene follows a finite, ground-bound, input-causal stepped trajectory rather than receiving an uncommanded launch",
    },
    async () => {
        const idle = await runTrace([]);
        const forward = await runTrace(["KeyW"]);
        const reverse = await runTrace(["KeyS"]);
        const left = await runTrace(["KeyW", "KeyA"]);
        const right = await runTrace(["KeyW", "KeyD"]);
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
        const leftHeading = heading(left.samples[HORIZON - 1].bodies[0].quat);
        const rightHeading = heading(right.samples[HORIZON - 1].bodies[0].quat);
        if (leftHeading >= -0.02 || rightHeading <= 0.02)
            throw new Error(
                `W+A/W+D heading was not opposite: ${leftHeading},${rightHeading}; diagnostics=${pairedDiagnostic("opposite W+A/W+D heading", left, right, HORIZON - 1, idle.bounds)}`,
            );
        if (forward.samples[HORIZON - 1].observation.upright === null)
            throw new Error("actual vehicle exposed no upright joint");
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
            if (!observation.upright) throw new Error("actual vehicle exposed no upright joint");
            const live = body(app.state, eid);
            if (!live) throw new Error("actual chassis never became a live body");
            // Size the roll impulse from the chassis roll inertia and the upright spring so the
            // critically damped response, theta(t) = w0 t exp(-omega t), peaks at rollBudget. After the
            // spring's settling time 4 / (zeta omega), that response is down to residual; the wheels only
            // add inertia, so the real chassis must roll no further and recover at least as far.
            const rollBudget = Math.PI / 6;
            const { uprightHertz: hertz, uprightDampingRatio: dampingRatio } = VEHICLE_CONFIG;
            const omega = 2 * Math.PI * hertz;
            const settleTime = 4 / (dampingRatio * omega);
            const residual =
                rollBudget * Math.E * omega * settleTime * Math.exp(-omega * settleTime);
            const rollInertia = live.getMassData().inertia.cx.x;
            live.applyAngularImpulse(
                { x: rollInertia * rollBudget * omega * Math.E, y: 0, z: 0 },
                true,
            );
            const { dt } = physicsStepConfig(app.state);
            const roll = () => {
                const sample = readBody(app.state, eid);
                if (!sample) throw new Error("disturbed chassis could not be read");
                return Math.acos(Math.min(1, rotate(sample.quat, [0, 1, 0])[1]));
            };
            let peak = 0;
            for (let i = 0; i < Math.ceil(settleTime / dt); i++) {
                app.state.step(dt);
                peak = Math.max(peak, roll());
            }
            if (peak > rollBudget || peak <= residual)
                throw new Error(
                    `roll impulse peaked at ${peak} rad, outside (${residual}, ${rollBudget}]`,
                );
            const settled = roll();
            if (settled > residual)
                throw new Error(
                    `bounded roll did not recover upright chassis: roll=${settled} rad`,
                );
            const current = readVehicle(app.state);
            if (!current) throw new Error("vehicle observation disappeared during recovery");
        } finally {
            app.dispose();
        }
    },
);
