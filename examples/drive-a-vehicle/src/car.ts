import {
    Body,
    body,
    createParallelJoint,
    createWheelJoint,
    devices,
    InputPlugin,
    PhysicsPlugin,
    type Plugin,
    type State,
    type System,
    slab,
    u32,
} from "@dylanebert/shallot";

/** The scene's one recipe-local role component keeps content declarative while the plugin owns wiring. */
export const Vehicle = { role: slab(u32) };

export const VehicleRole = {
    Ground: 0,
    Chassis: 1,
    FrontWheel: 2,
    RearWheel: 3,
} as const;

const THROTTLE = 14; // rad/s at full throttle: enough to move the sample car without hiding the suspension.
const STEER = Math.PI / 5; // a readable steering lock for the orbit view.

// These are authored vehicle frame values, not derived solver math. The wheel body pose is authored in the
// scene; the joint's x suspension axis and z spin axis stay fixed for this recipe's four-wheel layout.
export const SUSPENSION_FRAME = { v: { x: 0, y: 0, z: 0.70710677 }, s: 0.70710677 };
export const SPIN_FRAME = { v: { x: -0.70710677, y: 0, z: 0 }, s: 0.70710677 };
type Wheel = ReturnType<typeof createWheelJoint>;
type VehicleRuntime = {
    ground: number;
    chassis: number;
    front: number[];
    rear: number[];
    frontJoints: Wheel[];
    rearJoints: Wheel[];
    steer: number;
    wired: boolean;
};

const runtimes = new WeakMap<State, VehicleRuntime>();

function role(state: State, wanted: number): number {
    for (const eid of state.query([Vehicle, Body])) {
        if (Vehicle.role.get(eid) === wanted) return eid;
    }
    return -1;
}

function wheelConfig(sx: number, sz: number) {
    return {
        localFrameA: { p: { x: 1.5 * sx, y: -1, z: 0.8 * sz }, q: SUSPENSION_FRAME },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: SPIN_FRAME },
        enableSuspensionSpring: true,
        suspensionHertz: 4,
        suspensionDampingRatio: 0.7,
        enableSuspensionLimit: true,
        lowerSuspensionLimit: -0.2,
        upperSuspensionLimit: 0.2,
    };
}

function wire(state: State, runtime: VehicleRuntime): boolean {
    if (runtime.wired) return true;
    const bodyEids = [runtime.ground, runtime.chassis, ...runtime.front, ...runtime.rear];
    if (bodyEids.some((eid) => eid < 0 || !body(state, eid))) return false;

    runtime.frontJoints = runtime.front.map((eid, index) => {
        const sx = 1;
        const sz = index % 2 === 0 ? 1 : -1;
        return createWheelJoint(state, runtime.chassis, eid, {
            ...wheelConfig(sx, sz),
            enableSteering: true,
            steeringHertz: 10,
            steeringDampingRatio: 0.7,
            maxSteeringTorque: 5,
            targetSteeringAngle: 0,
            enableSteeringLimit: true,
            lowerSteeringLimit: -STEER,
            upperSteeringLimit: STEER,
        });
    });
    runtime.rearJoints = runtime.rear.map((eid, index) => {
        const sx = -1;
        const sz = index % 2 === 0 ? 1 : -1;
        return createWheelJoint(state, runtime.chassis, eid, {
            ...wheelConfig(sx, sz),
            enableSpinMotor: true,
            spinSpeed: 0,
            maxSpinTorque: 5,
        });
    });
    createParallelJoint(state, runtime.ground, runtime.chassis, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: SPIN_FRAME },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: SPIN_FRAME },
        hertz: 0.5,
        dampingRatio: 1,
        collideConnected: true,
    });
    runtime.wired = true;
    return true;
}

const driver: System = {
    name: "vehicle-driver",
    group: "simulation",
    update(state) {
        const runtime = runtimes.get(state);
        if (!runtime || !wire(state, runtime)) return;

        const keys = devices(state).keys.held;
        let throttle = 0;
        if (keys.has("KeyW")) throttle -= THROTTLE;
        if (keys.has("KeyS")) throttle += THROTTLE;
        let steer = 0;
        if (keys.has("KeyA")) steer += STEER;
        if (keys.has("KeyD")) steer -= STEER;
        for (const joint of runtime.rearJoints) joint.setSpinMotorSpeed(throttle);
        if (steer !== runtime.steer) {
            for (const joint of runtime.frontJoints) joint.setTargetSteeringAngle(steer);
            runtime.steer = steer;
        }
    },
};

export const Car = {
    name: "Car",
    components: { Vehicle },
    traits: {
        Vehicle: {
            defaults: () => ({ role: VehicleRole.Ground }),
            enums: { role: VehicleRole },
        },
    },
    dependencies: [InputPlugin, PhysicsPlugin],
    systems: [driver],
    warm(state: State) {
        runtimes.set(state, {
            ground: role(state, VehicleRole.Ground),
            chassis: role(state, VehicleRole.Chassis),
            front: [...state.query([Vehicle, Body])].filter(
                (eid) => Vehicle.role.get(eid) === VehicleRole.FrontWheel,
            ),
            rear: [...state.query([Vehicle, Body])].filter(
                (eid) => Vehicle.role.get(eid) === VehicleRole.RearWheel,
            ),
            frontJoints: [],
            rearJoints: [],
            steer: 0,
            wired: false,
        });
    },
} satisfies Plugin;

export default Car;
