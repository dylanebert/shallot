import {
    Body,
    BodyType,
    Devices,
    HeightFields,
    Hulls,
    InputPlugin,
    ParallelJoint,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";
import {
    StandardPhysicsPlugin,
    SyncPhysicsConstraintsSystem,
    setAwake,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
export const DrivingState = {
    create: () => ({
        chassis: 0,
        wheels: [] as number[],
        keys: { W: 0, A: 0, S: 0, D: 0 },
        speed: 30,
    }),
};
export const DrivingPlugin: Plugin = {
    name: "Driving",
    dependencies: [StandardPhysicsPlugin, InputPlugin],
    recovery: recoverResource(DrivingState),
    systems: [
        {
            name: "driving-input",
            group: "setup",
            update(world) {
                const state = world.resource(DrivingState);
                if (!state.chassis) return;
                const devices = world.resource(Devices);
                for (const key of ["W", "A", "S", "D"] as const)
                    state.keys[key] = Number(devices.enabled && devices.keys.held.has(`Key${key}`));
            },
        },
        {
            name: "driving-before-step",
            group: "fixed",
            before: [SyncPhysicsConstraintsSystem],
            update(world) {
                const state = world.resource(DrivingState);
                if (!state.chassis) return;
                const { W, A, S, D } = state.keys;
                if (W || A || S || D) setAwake(world, state.chassis, true);
                const joints = world.storage(WheelJoint);
                for (const [i, eid] of state.wheels.entries()) {
                    if (i < 2)
                        joints.targetSteeringAngle.set(eid, f(f(0.25 * f(Math.PI)) * (A - D)));
                    else joints.spinSpeed.set(eid, f(-state.speed * (W - S)));
                }
            },
        },
    ],
};
function driving(world: World): number[] {
    const state = world.resource(DrivingState);
    Object.assign(state, DrivingState.create());
    const ground = world.create();
    world.add(ground, Transform, { translation: [-20, 0, -20] });
    world.add(ground, Body);
    const geometry = world.resource(HeightFields).add({
        name: "Wave",
        wave: {
            rowCount: 50,
            columnCount: 50,
            scale: [4, 2, 4],
            rowFrequency: f(0.02),
            columnFrequency: f(0.04),
            makeHoles: false,
        },
    });
    world.add(ground, Shape, { kind: ShapeKind.HeightField, geometry });
    const chassis = world.create();
    state.chassis = chassis;
    world.add(chassis, Transform, { translation: [0, 2.5, 0] });
    world.add(chassis, Body, { type: BodyType.Dynamic });
    world.add(chassis, Shape, {
        density: 0.5,
        geometry: world.resource(Hulls).add({ name: "box", halfExtents: [2, 0.5, 1] }),
    });
    const q: [number, number, number, number] = [-f(Math.SQRT1_2), 0, 0, f(Math.SQRT1_2)];
    world.add(world.create(), ParallelJoint, {
        a: ground,
        b: chassis,
        localRotationA: q,
        localRotationB: q,
        collideConnected: 1,
        drawScale: 2,
        hertz: 0.5,
        dampingRatio: 1,
    });
    const bodies = [chassis];
    for (const [i, position] of (
        [
            [1.5, 2, 0.8],
            [1.5, 2, -0.8],
            [-1.5, 2, 0.8],
            [-1.5, 2, -0.8],
        ] as [number, number, number][]
    ).entries()) {
        const wheel = world.create();
        bodies.push(wheel);
        world.add(wheel, Transform, {
            translation: position,
            rotation: [f(Math.SQRT1_2), 0, 0, f(Math.SQRT1_2)],
        });
        world.add(wheel, Body, { type: BodyType.Dynamic, allowFastRotation: 1 });
        world.add(wheel, Shape, { kind: ShapeKind.Sphere, radius: 0.4, density: 2, friction: 3 });
        const joint = world.create();
        state.wheels.push(joint);
        world.add(joint, WheelJoint, {
            a: chassis,
            b: wheel,
            localAnchorA: [position[0], -0.5, position[2]],
            localRotationA: [0, 0, f(Math.SQRT1_2), f(Math.SQRT1_2)],
            localRotationB: q,
            enableSuspensionLimit: 1,
            lowerSuspensionLimit: -0.2,
            upperSuspensionLimit: 0.2,
            suspensionHertz: 4,
            suspensionDampingRatio: 0.7,
            enableSpinMotor: Number(i >= 2),
            maxSpinTorque: 5,
            enableSteering: Number(i < 2),
            steeringHertz: 10,
            steeringDampingRatio: 0.7,
            maxSteeringTorque: 5,
            enableSteeringLimit: 1,
            lowerSteeringLimit: f(f(f(Math.PI) / 180) * -45),
            upperSteeringLimit: f(f(f(Math.PI) / 180) * 45),
        });
    }
    return bodies;
}
const controls: Controls = Object.fromEntries(
    (
        [
            ["Suspension Min", "lowerSuspensionLimit", -0.2, -10, 10, 0.1],
            ["Suspension Max", "upperSuspensionLimit", 0.2, -10, 10, 0.1],
            ["Suspension Hertz", "suspensionHertz", 4, 0, 10, 0.1],
            ["Suspension Damping", "suspensionDampingRatio", 0.7, 0, 2, 0.1],
            ["Max Torque", "maxSpinTorque", 5, 0, 100, 1],
            ["Speed", "spinSpeed", 30, 0, 100, 1],
            ["Steering Hertz", "steeringHertz", 10, 0, 10, 0.1],
            ["Steering Damping", "steeringDampingRatio", 0.7, 0, 2, 0.1],
            ["Steering Torque", "maxSteeringTorque", 5, 0, 20, 0.1],
            ["Steering Min Deg", "lowerSteeringLimit", -45, -90, 0, 1],
            ["Steering Max Deg", "upperSteeringLimit", 45, 0, 90, 1],
        ] as const
    ).map(([label, field, value, min, max, step]) => [
        label,
        {
            label,
            value,
            min,
            max,
            step,
            decimals: step === 0.1 ? 1 : 0,
            set(world, _bodies, value, values) {
                const state = world.resource(DrivingState);
                if (label === "Speed") {
                    state.speed = f(value);
                    return;
                }
                if (label === "Max Torque") return; // The sample's motor torque row changes only its stored value.
                const joints = world.storage(WheelJoint);
                if (label === "Suspension Min") value = Math.min(value, values["Suspension Max"]!);
                if (label === "Suspension Max") value = Math.max(value, values["Suspension Min"]!);
                values[label] = value;
                for (const [i, eid] of state.wheels.entries()) {
                    if (label.startsWith("Steering") && i >= 2) continue;
                    joints[field].set(
                        eid,
                        field.endsWith("SteeringLimit") ? f(f(f(Math.PI) / 180) * f(value)) : value,
                    );
                }
            },
        } satisfies Controls[string],
    ]),
);
export const drivingScene = {
    construct: driving,
    gridExtent: 200,
    gridSpacing: 4,
    camera: { yaw: 25, pitch: 20, distance: 12, target: [0, 2.5, 0] },
    controls,
};
