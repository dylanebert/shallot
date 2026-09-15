import {
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
import { check } from "@dylanebert/shallot/harness/check";
import { Car, Vehicle, VehicleRole } from "./car";

const SCENE = `<scene>
    <a id="ground" vehicle="role: ground" body="pos: 0 0 0; half-extents: 30 0.5 30; mass: 0; friction: 1" />
    <a id="chassis" vehicle="role: chassis" body="pos: 0 2.5 0; half-extents: 2 0.5 1; mass: 4" />
    <a vehicle="role: front-wheel" body="shape: 1; pos: 1.5 1.5 0.8; half-extents: 0 0 0 0.4; mass: 0.5; friction: 3; quat: 90 0 0" />
    <a vehicle="role: front-wheel" body="shape: 1; pos: 1.5 1.5 -0.8; half-extents: 0 0 0 0.4; mass: 0.5; friction: 3; quat: 90 0 0" />
    <a vehicle="role: rear-wheel" body="shape: 1; pos: -1.5 1.5 0.8; half-extents: 0 0 0 0.4; mass: 0.5; friction: 3; quat: 90 0 0" />
    <a vehicle="role: rear-wheel" body="shape: 1; pos: -1.5 1.5 -0.8; half-extents: 0 0 0 0.4; mass: 0.5; friction: 3; quat: 90 0 0" />
</scene>`;

async function vehicle() {
    return build({
        defaults: false,
        plugins: [PhysicsPlugin, InputPlugin, Car],
        scene: SCENE,
    });
}

function chassis(state: State): number {
    for (const eid of state.query([Vehicle])) {
        if (Vehicle.role.get(eid) === VehicleRole.Chassis) return eid;
    }
    throw new Error("vehicle scene has no chassis");
}

function step(app: Awaited<ReturnType<typeof vehicle>>, ticks: number): void {
    for (let i = 0; i < ticks; i++) app.state.step(Time.FIXED_DT);
}

function yaw(quat: readonly [number, number, number, number]): number {
    const [x, y, z, w] = quat;
    return Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z));
}

function angleDelta(after: number, before: number): number {
    return Math.atan2(Math.sin(after - before), Math.cos(after - before));
}

check(
    "vehicle throttle displaces the chassis",
    {
        claim: "the drive-a-vehicle recipe turns W into chassis displacement through its production wheel motor",
    },
    async () => {
        async function run(throttle: boolean): Promise<readonly [number, number]> {
            const app = await vehicle();
            try {
                const eid = chassis(app.state);
                step(app, 2);
                const before = readBody(app.state, eid);
                if (throttle) pressKey(app.state, "KeyW");
                step(app, 12);
                if (throttle) releaseKey(app.state, "KeyW");
                const after = readBody(app.state, eid);
                if (!before || !after) throw new Error("chassis never became a live body");
                return [after.pos[0] - before.pos[0], after.pos[2] - before.pos[2]];
            } finally {
                app.dispose();
            }
        }
        const idle = await run(false);
        const driven = await run(true);
        const commandDisplacement = Math.hypot(driven[0] - idle[0], driven[1] - idle[1]);
        if (commandDisplacement < 0.0005)
            throw new Error(
                `throttle displacement changed only ${commandDisplacement.toFixed(4)}m`,
            );
    },
);

check(
    "vehicle opposite steering gives opposite heading",
    {
        claim: "opposite A/D steering commands produce opposite signed chassis heading through the front wheel targets",
    },
    async () => {
        async function run(steering: "KeyA" | "KeyD" | null): Promise<number> {
            const app = await vehicle();
            try {
                step(app, 2);
                const before = readBody(app.state, chassis(app.state));
                pressKey(app.state, "KeyW");
                if (steering) pressKey(app.state, steering);
                step(app, 8);
                const result = readBody(app.state, chassis(app.state));
                if (!before || !result) throw new Error("steering body was not live");
                return angleDelta(yaw(result.quat), yaw(before.quat));
            } finally {
                app.dispose();
            }
        }
        const baseline = await run(null);
        const leftYaw = (await run("KeyA")) - baseline;
        const rightYaw = (await run("KeyD")) - baseline;
        if (Math.abs(leftYaw) < 0.001 || Math.abs(rightYaw) < 0.001 || leftYaw * rightYaw >= 0)
            throw new Error(`steering headings were not opposite: ${leftYaw}, ${rightYaw}`);
    },
);

check(
    "vehicle command wakes a sleeping chassis",
    {
        claim: "a production throttle command moves a sleeping chassis without a recipe caller wake loop",
    },
    async () => {
        const app = await vehicle();
        try {
            const eid = chassis(app.state);
            step(app, 90);
            const parked = body(app.state, eid);
            if (!parked) throw new Error("chassis never became a live body");
            parked.setAwake(false);
            const before = readBody(app.state, eid);
            pressKey(app.state, "KeyW");
            step(app, 12);
            const after = readBody(app.state, eid);
            if (!before || !after) throw new Error("sleeping chassis could not be read");
            const displacement = Math.hypot(
                after.pos[0] - before.pos[0],
                after.pos[2] - before.pos[2],
            );
            if (displacement < 0.001)
                throw new Error(`sleeping command produced only ${displacement.toFixed(3)}m`);
            if (!devices(app.state).keys.held.has("KeyW")) throw new Error("W was not delivered");
        } finally {
            app.dispose();
        }
    },
);
