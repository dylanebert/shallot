import {
    Body,
    body,
    createParallelJoint,
    createWheelJoint,
    devices,
    InputPlugin,
    mountOverlay,
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

// These are authored recipe controls and joint tunings, consumed by both wiring and driving. They are
// intentionally one frozen data object so trajectory bounds can observe the effective handles instead of
// repeating production literals in the check.
export const VEHICLE_CONFIG = Object.freeze({
    throttle: 14, // rad/s at full throttle: enough to move the sample car without hiding the suspension.
    steeringLock: Math.PI / 5, // a readable steering lock for the orbit view.
    suspensionHertz: 4,
    suspensionDampingRatio: 0.7,
    suspensionLower: -0.2,
    suspensionUpper: 0.2,
    steeringHertz: 10,
    steeringDampingRatio: 0.7,
    maxSteeringTorque: 5,
    maxSpinTorque: 5,
    uprightHertz: 0.5,
    uprightDampingRatio: 1,
});

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
    upright: ReturnType<typeof createParallelJoint> | null;
    steer: number;
    wired: boolean;
};

type VehicleFrame = ReturnType<Wheel["getLocalFrameA"]>;
export type VehicleObservation = Readonly<{
    ground: number;
    chassis: number;
    wheels: readonly Readonly<{
        eid: number;
        role: number;
        localFrameA: VehicleFrame;
        localFrameB: VehicleFrame;
        suspension: Readonly<{
            enabled: boolean;
            hertz: number;
            dampingRatio: number;
            limitEnabled: boolean;
            lower: number;
            upper: number;
        }>;
        steering: Readonly<{
            enabled: boolean;
            hertz: number;
            dampingRatio: number;
            limitEnabled: boolean;
            lower: number;
            upper: number;
            target: number;
            angle: number;
            torque: number;
        }>;
        spin: Readonly<{
            enabled: boolean;
            target: number;
            maxTorque: number;
            speed: number;
            torque: number;
        }>;
        linearSeparation: number;
    }>[];
    upright: Readonly<{
        localFrameA: VehicleFrame;
        localFrameB: VehicleFrame;
        hertz: number;
        dampingRatio: number;
        maxTorque: number;
        angularSeparation: number;
    }> | null;
}>;

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
        suspensionHertz: VEHICLE_CONFIG.suspensionHertz,
        suspensionDampingRatio: VEHICLE_CONFIG.suspensionDampingRatio,
        enableSuspensionLimit: true,
        lowerSuspensionLimit: VEHICLE_CONFIG.suspensionLower,
        upperSuspensionLimit: VEHICLE_CONFIG.suspensionUpper,
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
            steeringHertz: VEHICLE_CONFIG.steeringHertz,
            steeringDampingRatio: VEHICLE_CONFIG.steeringDampingRatio,
            maxSteeringTorque: VEHICLE_CONFIG.maxSteeringTorque,
            targetSteeringAngle: 0,
            enableSteeringLimit: true,
            lowerSteeringLimit: -VEHICLE_CONFIG.steeringLock,
            upperSteeringLimit: VEHICLE_CONFIG.steeringLock,
        });
    });
    runtime.rearJoints = runtime.rear.map((eid, index) => {
        const sx = -1;
        const sz = index % 2 === 0 ? 1 : -1;
        return createWheelJoint(state, runtime.chassis, eid, {
            ...wheelConfig(sx, sz),
            enableSpinMotor: true,
            spinSpeed: 0,
            maxSpinTorque: VEHICLE_CONFIG.maxSpinTorque,
        });
    });
    runtime.upright = createParallelJoint(state, runtime.ground, runtime.chassis, {
        localFrameA: { p: { x: 0, y: 0, z: 0 }, q: SPIN_FRAME },
        localFrameB: { p: { x: 0, y: 0, z: 0 }, q: SPIN_FRAME },
        hertz: VEHICLE_CONFIG.uprightHertz,
        dampingRatio: VEHICLE_CONFIG.uprightDampingRatio,
        collideConnected: true,
    });
    runtime.wired = true;
    return true;
}

function copyFrame(frame: VehicleFrame): VehicleFrame {
    return {
        p: { x: frame.p.x, y: frame.p.y, z: frame.p.z },
        q: { v: { x: frame.q.v.x, y: frame.q.v.y, z: frame.q.v.z }, s: frame.q.s },
    };
}

/** Read-only effective vehicle wiring and command state for bounded recipe evidence. */
export function readVehicle(state: State): VehicleObservation | null {
    const runtime = runtimes.get(state);
    if (!runtime?.wired) return null;
    const joints = [
        ...runtime.frontJoints.map((joint, index) => ({
            eid: runtime.front[index],
            role: VehicleRole.FrontWheel,
            joint,
        })),
        ...runtime.rearJoints.map((joint, index) => ({
            eid: runtime.rear[index],
            role: VehicleRole.RearWheel,
            joint,
        })),
    ];
    return {
        ground: runtime.ground,
        chassis: runtime.chassis,
        wheels: joints.map(({ eid, role, joint }) => ({
            eid,
            role,
            localFrameA: copyFrame(joint.getLocalFrameA()),
            localFrameB: copyFrame(joint.getLocalFrameB()),
            suspension: {
                enabled: joint.isSuspensionEnabled(),
                hertz: joint.getSuspensionHertz(),
                dampingRatio: joint.getSuspensionDampingRatio(),
                limitEnabled: joint.isSuspensionLimitEnabled(),
                lower: joint.getLowerSuspensionLimit(),
                upper: joint.getUpperSuspensionLimit(),
            },
            steering: {
                enabled: joint.isSteeringEnabled(),
                hertz: joint.getSteeringHertz(),
                dampingRatio: joint.getSteeringDampingRatio(),
                limitEnabled: joint.isSteeringLimitEnabled(),
                lower: joint.getLowerSteeringLimit(),
                upper: joint.getUpperSteeringLimit(),
                target: joint.getTargetSteeringAngle(),
                angle: joint.getSteeringAngle(),
                torque: joint.getSteeringTorque(),
            },
            spin: {
                enabled: joint.isSpinMotorEnabled(),
                target: joint.getSpinMotorSpeed(),
                maxTorque: joint.getMaxSpinTorque(),
                speed: joint.getSpinSpeed(),
                torque: joint.getSpinTorque(),
            },
            linearSeparation: joint.getLinearSeparation(),
        })),
        upright:
            runtime.upright === null
                ? null
                : {
                      localFrameA: copyFrame(runtime.upright.getLocalFrameA()),
                      localFrameB: copyFrame(runtime.upright.getLocalFrameB()),
                      ...runtime.upright.getConstraintTuning(),
                      maxTorque: runtime.upright.getMaxTorque(),
                      angularSeparation: runtime.upright.getAngularSeparation(),
                  },
    };
}

const CONTROL_PANEL = Symbol.for("shallot.examples.drive-a-vehicle.controls");
type ControlState = State & { [CONTROL_PANEL]?: HTMLDivElement };

function mountControls(state: State): void {
    if (typeof document === "undefined") return;
    const owner = state as ControlState;
    if (owner[CONTROL_PANEL]) return;
    const overlay = mountOverlay(document.querySelector("canvas"), state);
    const panel = document.createElement("div");
    panel.dataset.recipeControls = "";
    panel.style.cssText =
        "position:absolute;top:20px;left:20px;pointer-events:none;padding:10px 12px;" +
        "display:grid;row-gap:6px;column-gap:16px;border:1px solid rgba(255,255,255,0.12);" +
        "border-radius:6px;background:rgba(14,17,20,0.72);color:#ffffff;" +
        "font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace";
    for (const [control, action] of [
        ["W / S", "Throttle"],
        ["A / D", "Steer"],
    ] as const) {
        const row = document.createElement("div");
        row.dataset.controlRow = "";
        row.style.cssText =
            "display:grid;grid-template-columns:max-content max-content;column-gap:16px";
        for (const text of [control, action]) {
            const cell = document.createElement("span");
            cell.textContent = text;
            cell.style.color = "#ffffff";
            row.append(cell);
        }
        panel.append(row);
    }
    overlay.append(panel);
    owner[CONTROL_PANEL] = panel;
    state.onDispose(() => {
        if (owner[CONTROL_PANEL] === panel) delete owner[CONTROL_PANEL];
    });
}

const controls: System = {
    name: "vehicle-controls",
    group: "draw",
    update(state) {
        mountControls(state);
    },
};

const driver: System = {
    name: "vehicle-driver",
    group: "simulation",
    update(state) {
        const runtime = runtimes.get(state);
        if (!runtime || !wire(state, runtime)) return;

        const keys = devices(state).keys.held;
        let throttle = 0;
        if (keys.has("KeyW")) throttle -= VEHICLE_CONFIG.throttle;
        if (keys.has("KeyS")) throttle += VEHICLE_CONFIG.throttle;
        let steer = 0;
        if (keys.has("KeyA")) steer += VEHICLE_CONFIG.steeringLock;
        if (keys.has("KeyD")) steer -= VEHICLE_CONFIG.steeringLock;
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
    systems: [driver, controls],
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
            upright: null,
            steer: 0,
            wired: false,
        });
    },
} satisfies Plugin;

export default Car;
