import { resolve } from "node:path";
import {
    Body,
    body,
    build,
    devices,
    InputPlugin,
    PhysicsPlugin,
    pressKey,
    readBody,
    releaseKey,
    type State,
    Time,
} from "@dylanebert/shallot";
import { runBrowserCheck } from "@dylanebert/shallot/harness";
import { check } from "@dylanebert/shallot/harness/check";
import { Car, SPIN_FRAME, SUSPENSION_FRAME, Vehicle, VehicleRole } from "./car";

const SCENE = resolve(import.meta.dir, "../public/scenes/drive-a-vehicle.scene");
const HORIZON = 120;
const SUBSTEPS = 4;
const BOX3D_LINEAR_SLOP = 0.005;

type BodySnapshot = NonNullable<ReturnType<typeof readBody>>;
type Arm = readonly string[];
type Trace = {
    keys: Arm;
    chassis: number;
    wheels: number[];
    bounds: VehicleBounds;
    samples: BodySnapshot[][];
};

type VehicleBounds = {
    dt: number;
    substepDt: number;
    slop: number;
    clearAirTicks: number;
    groundTop: number;
    wheelRadius: number;
    speedCeiling: number;
    altitudeCeiling: number;
    driveImpulse: number;
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
    if (result.length < 4)
        throw new Error(`actual vehicle scene has ${result.length} live wheels; expected four`);
    return result;
}

function step(app: Awaited<ReturnType<typeof vehicle>>, ticks: number): void {
    for (let i = 0; i < ticks; i++) app.state.step(Time.FIXED_DT);
}

function live(app: Awaited<ReturnType<typeof vehicle>>, eids: readonly number[]): BodySnapshot[] {
    return eids.map((eid) => {
        const bodyState = readBody(app.state, eid);
        if (!bodyState) throw new Error(`vehicle body ${eid} has no live stepped pose`);
        return bodyState;
    });
}

function quat(value: {
    v: { x: number; y: number; z: number };
    s: number;
}): [number, number, number, number] {
    return [value.v.x, value.v.y, value.v.z, value.s];
}

function multiply(
    a: readonly [number, number, number, number],
    b: readonly [number, number, number, number],
): [number, number, number, number] {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ];
}

function rotate(
    q: readonly [number, number, number, number],
    v: readonly [number, number, number],
): [number, number, number] {
    const [x, y, z, w] = q;
    const [vx, vy, vz] = v;
    return [
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - z * w) * vy + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx + 2 * (y * z + x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    ];
}

function distance(
    a: readonly [number, number, number],
    b: readonly [number, number, number],
): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function bounds(
    app: Awaited<ReturnType<typeof vehicle>>,
    chassisEid: number,
    wheelEids: readonly number[],
): VehicleBounds {
    const ground = role(app.state, VehicleRole.Ground);
    if (ground < 0) throw new Error("actual vehicle scene has no ground Body");
    const groundTop = Body.pos.y.get(ground) + Body.halfExtents.y.get(ground);
    const wheelRadius = Body.halfExtents.w.get(wheelEids[0]);
    const wheelBottom = Body.pos.y.get(wheelEids[0]) - wheelRadius;
    const initialClearance = wheelBottom - groundTop;
    const mass = [chassisEid, ...wheelEids].reduce((sum, eid) => sum + Body.mass.get(eid), 0);
    const drivenWheels = wheelEids.filter(
        (eid) => Vehicle.role.get(eid) === VehicleRole.RearWheel,
    ).length;
    const torque = 5;
    const targetSpin = 14;
    const dt = Time.FIXED_DT;
    const substepDt = dt / SUBSTEPS;
    const slop = BOX3D_LINEAR_SLOP;
    const clearAirSeconds = Math.sqrt(Math.max(0, (2 * (initialClearance - slop)) / 10));
    const anchorDrop = Math.abs(Body.pos.y.get(chassisEid) - Body.pos.y.get(wheelEids[0]));
    const upperTravel = 0.2;
    const altitudeCeiling =
        Math.max(Body.pos.y.get(chassisEid), groundTop + wheelRadius + anchorDrop + upperTravel) +
        2 * wheelRadius +
        slop;
    const rimSpeed = Math.abs(targetSpin) * wheelRadius;
    const driveAcceleration = (drivenWheels * torque) / (wheelRadius * mass);
    return {
        dt,
        substepDt,
        slop,
        clearAirTicks: Math.max(0, Math.ceil(clearAirSeconds / dt) - 1),
        groundTop,
        wheelRadius,
        // The motor target is a wheel rim speed, so the chassis must stay within that cap plus
        // two solver slops; this is deliberately independent of the trajectory horizon.
        speedCeiling: rimSpeed + (2 * slop) / dt,
        altitudeCeiling,
        driveImpulse: driveAcceleration * dt + slop / substepDt,
    };
}

function diagnostic(trace: Trace, index: number, boundsValue: VehicleBounds): string {
    const from = Math.max(0, index - 1);
    const to = Math.min(trace.samples.length, index + 2);
    const sample = trace.samples.slice(from, to).map((bodies, offset) => ({
        tick: from + offset,
        chassis: { pos: bodies[0].pos, vel: bodies[0].vel, quat: bodies[0].quat },
        wheels: bodies.slice(1).map((bodyState) => ({ pos: bodyState.pos, quat: bodyState.quat })),
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
        `${reason}; first failing tick ${index}; diagnostics=${diagnostic(trace, index, boundsValue)}`,
    );
}

function assertFramePremise(chassisEid: number, wheelEids: readonly number[]): void {
    const chassisPos = [
        Body.pos.x.get(chassisEid),
        Body.pos.y.get(chassisEid),
        Body.pos.z.get(chassisEid),
    ] as const;
    const suspensionAxis = rotate(quat(SUSPENSION_FRAME), [1, 0, 0]);
    const parallelAxis = rotate(quat(SPIN_FRAME), [0, 0, 1]);
    if (distance(suspensionAxis, [0, 1, 0]) > 1e-5)
        throw new Error(`wheel suspension frame x axis is not world Y: ${suspensionAxis}`);
    if (distance(parallelAxis, [0, 1, 0]) > 1e-5)
        throw new Error(`upright parallel frame z axis is not world Y: ${parallelAxis}`);

    for (const eid of wheelEids) {
        const q: [number, number, number, number] = [
            Body.quat.x.get(eid),
            Body.quat.y.get(eid),
            Body.quat.z.get(eid),
            Body.quat.w.get(eid),
        ];
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

function validateTrace(
    trace: Trace,
    boundsValue: VehicleBounds,
    idle: boolean,
    strictTrajectory: boolean,
): void {
    let previous = trace.samples[0][0];
    for (let tick = 0; tick < trace.samples.length; tick++) {
        const bodies = trace.samples[tick];
        for (const [index, bodyState] of bodies.entries()) {
            const values = [...bodyState.pos, ...bodyState.vel, ...bodyState.quat];
            if (values.some((value) => !Number.isFinite(value)))
                fail(
                    trace,
                    tick,
                    boundsValue,
                    `${index === 0 ? "chassis" : `wheel ${index}`} has non-finite state`,
                );
            if (Math.abs(Math.hypot(...bodyState.quat) - 1) > 1e-5)
                fail(
                    trace,
                    tick,
                    boundsValue,
                    `${index === 0 ? "chassis" : `wheel ${index}`} quaternion is not unit`,
                );
        }
        const chassisState = bodies[0];
        const horizontalSpeed = Math.hypot(chassisState.vel[0], chassisState.vel[2]);
        if (horizontalSpeed > boundsValue.speedCeiling)
            fail(
                trace,
                tick,
                boundsValue,
                `chassis horizontal speed ${horizontalSpeed.toFixed(3)} exceeded ${boundsValue.speedCeiling.toFixed(3)}`,
            );
        // Box3D may leave a few linear slops of overlap while the suspension resolves contact; the
        // tolerance is solver-derived, not an altitude/speed relaxation. A falling body still gets only
        // one fixed-step catch-up allowance and therefore reds dramatically through-ground motion.
        const contactTolerance =
            4 * boundsValue.slop + Math.max(0, -chassisState.vel[1]) * boundsValue.dt;
        for (const [wheelIndex, wheelState] of bodies.slice(1).entries()) {
            const wheelBottom = wheelState.pos[1] - boundsValue.wheelRadius;
            if (wheelBottom < boundsValue.groundTop - contactTolerance)
                fail(
                    trace,
                    tick,
                    boundsValue,
                    `wheel ${wheelIndex} penetrated ground: bottom ${wheelBottom.toFixed(3)} below ${boundsValue.groundTop.toFixed(3)} with contact tolerance ${contactTolerance.toFixed(3)}`,
                );
        }
        const up = rotate(chassisState.quat, [0, 1, 0]);
        if (up[1] < 0.5) fail(trace, tick, boundsValue, `chassis inverted with up axis ${up}`);
        const displacement = distance(chassisState.pos, previous.pos);
        const totalSpeed = Math.hypot(...chassisState.vel);
        const previousTotalSpeed = Math.hypot(...previous.vel);
        if (
            strictTrajectory &&
            tick > 0 &&
            displacement >
                Math.max(totalSpeed, previousTotalSpeed) * boundsValue.dt + boundsValue.slop
        )
            fail(
                trace,
                tick,
                boundsValue,
                `chassis moved ${displacement.toFixed(4)}m in one tick without endpoint velocity support`,
            );
        if (strictTrajectory && idle && tick <= boundsValue.clearAirTicks) {
            const horizontal = Math.hypot(chassisState.pos[0], chassisState.pos[2]);
            const horizontalSpeed = Math.hypot(chassisState.vel[0], chassisState.vel[2]);
            if (
                horizontal > boundsValue.slop ||
                horizontalSpeed > boundsValue.slop / boundsValue.dt ||
                chassisState.pos[1] > previous.pos[1] + boundsValue.slop ||
                chassisState.vel[1] > boundsValue.slop / boundsValue.dt
            )
                fail(
                    trace,
                    tick,
                    boundsValue,
                    "idle clear-air phase gained uncommanded horizontal or upward energy",
                );
        }
        if (chassisState.pos[1] > boundsValue.altitudeCeiling)
            fail(
                trace,
                tick,
                boundsValue,
                `chassis altitude ${chassisState.pos[1].toFixed(3)} exceeded ${boundsValue.altitudeCeiling.toFixed(3)}`,
            );
        previous = chassisState;
    }
}

async function runTrace(
    keys: Arm,
    horizon = HORIZON,
    strictTrajectory = horizon === HORIZON,
): Promise<Trace> {
    const app = await vehicle();
    try {
        const chassisEid = chassis(app.state);
        const wheelEids = wheels(app.state);
        assertFramePremise(chassisEid, wheelEids);
        const bound = bounds(app, chassisEid, wheelEids);
        step(app, 2);
        const samples: BodySnapshot[][] = [live(app, [chassisEid, ...wheelEids])];
        for (const key of keys) pressKey(app.state, key);
        for (let tick = 0; tick < horizon; tick++) {
            app.state.step(Time.FIXED_DT);
            samples.push(live(app, [chassisEid, ...wheelEids]));
        }
        for (const key of keys) releaseKey(app.state, key);
        app.state.step(Time.FIXED_DT);
        samples.push(live(app, [chassisEid, ...wheelEids]));
        const trace = {
            keys,
            chassis: chassisEid,
            wheels: wheelEids.slice(),
            bounds: bound,
            samples,
        };
        validateTrace(trace, bound, keys.length === 0, strictTrajectory);
        return trace;
    } finally {
        app.dispose();
    }
}

let vehicleQueue: Promise<unknown> = Promise.resolve();
function enqueueVehicle<T>(task: () => Promise<T>): Promise<T> {
    const result = vehicleQueue.then(task);
    vehicleQueue = result.then(
        () => undefined,
        () => undefined,
    );
    return result;
}

function runVehicleTrace(
    keys: Arm,
    horizon = HORIZON,
    strictTrajectory = horizon === HORIZON,
): Promise<Trace> {
    return enqueueVehicle(() => runTrace(keys, horizon, strictTrajectory));
}

function yaw(quatValue: readonly [number, number, number, number]): number {
    const [x, y, z, w] = quatValue;
    return Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z));
}

function angleDelta(after: number, before: number): number {
    return Math.atan2(Math.sin(after - before), Math.cos(after - before));
}

check(
    "drive-a-vehicle actual scene follows a bounded causal trajectory",
    {
        claim: "the actual drive-a-vehicle scene follows a finite, ground-bound, input-causal stepped trajectory rather than receiving an uncommanded launch",
        size: "integration",
        subject: ["examples/drive-a-vehicle"],
    },
    async () => {
        const idle = await runVehicleTrace([]);
        const forward = await runVehicleTrace(["KeyW"]);
        const reverse = await runVehicleTrace(["KeyS"]);
        const left = await runVehicleTrace(["KeyW", "KeyA"]);
        const right = await runVehicleTrace(["KeyW", "KeyD"]);
        const idleStart = idle.samples[0][0];
        const idleEnd = idle.samples[idle.samples.length - 2][0];
        const forwardEnd = forward.samples[forward.samples.length - 2][0];
        const reverseEnd = reverse.samples[reverse.samples.length - 2][0];
        const leftEnd = left.samples[left.samples.length - 2][0];
        const rightEnd = right.samples[right.samples.length - 2][0];
        const forwardDelta = forwardEnd.pos[0] - idleStart.pos[0];
        const reverseDelta = reverseEnd.pos[0] - idleStart.pos[0];
        if (forwardDelta < 0.25 || reverseDelta > -0.25)
            throw new Error(
                `W/S did not produce opposite longitudinal displacement: ${forwardDelta}, ${reverseDelta}`,
            );
        const leftLateral = leftEnd.pos[2] - idleEnd.pos[2];
        const rightLateral = rightEnd.pos[2] - idleEnd.pos[2];
        if (leftLateral > -0.25 || rightLateral < 0.25 || leftLateral * rightLateral >= 0)
            throw new Error(
                `W+A/W+D did not produce opposite lateral effects: ${leftLateral}, ${rightLateral}`,
            );
        const drivenDelta = Math.hypot(
            forward.samples[1][0].vel[0] - idle.samples[1][0].vel[0],
            forward.samples[1][0].vel[2] - idle.samples[1][0].vel[2],
        );
        const bound = idle.bounds;
        if (drivenDelta > bound.driveImpulse)
            throw new Error(
                `first driven tick changed horizontal velocity by ${drivenDelta.toFixed(4)}m/s, beyond ${bound.driveImpulse.toFixed(4)}m/s`,
            );
        const before = yaw(idle.samples[0][0].quat);
        const after = yaw(forward.samples[1][0].quat);
        if (!Number.isFinite(angleDelta(after, before)))
            throw new Error("vehicle heading was non-finite");
    },
);

check(
    "vehicle throttle displaces the actual chassis",
    {
        claim: "the actual drive-a-vehicle recipe turns W into chassis displacement through its production wheel motor",
    },
    async () => {
        const driven = await runVehicleTrace(["KeyW"], 24);
        const before = driven.samples[0][0];
        const after = driven.samples[24][0];
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
        const left = await runVehicleTrace(["KeyW", "KeyA"], 20);
        const right = await runVehicleTrace(["KeyW", "KeyD"], 20);
        const leftVelocity = left.samples[20][0].vel[2];
        const rightVelocity = right.samples[20][0].vel[2];
        if (leftVelocity >= -0.001 || rightVelocity <= 0.001 || leftVelocity * rightVelocity >= 0)
            throw new Error(
                `actual steering lateral velocities were not opposite: ${leftVelocity}, ${rightVelocity}`,
            );
    },
);

check(
    "drive-a-vehicle exact project composes its selected scene and plugin",
    {
        claim: "the exact drive-a-vehicle manifest builds its selected scene and local Car role plugin before disposal",
        size: "integration",
        requires: ["chromium"],
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
    () =>
        enqueueVehicle(async () => {
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
        }),
);
