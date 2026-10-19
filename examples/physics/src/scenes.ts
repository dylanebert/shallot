import {
    Body,
    BodyType,
    FilterJoint,
    Hulls,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    SphericalJoint,
    Transform,
    WheelJoint,
    type World,
} from "@dylanebert/shallot";

import { PhysicsWorldDefinition, setAwake } from "@dylanebert/shallot/standard/physics";

import { eventScenes } from "./events";

export type Scene =
    | "boxStack"
    | "bridge"
    | "revolute"
    | "prismatic"
    | "wheel"
    | "filter"
    | "parallel"
    | keyof typeof eventScenes;
export const categories: { name: string; scenes: { id: Scene; name: string }[] }[] = [
    {
        name: "Events",
        scenes: [
            { id: "contact", name: "Contact" },
            { id: "hit", name: "Hit" },
            { id: "joint", name: "Joint" },
            { id: "sensor", name: "Sensor Visit" },
        ],
    },
    {
        name: "Joints",
        scenes: [
            { id: "bridge", name: "Bridge" },
            { id: "filter", name: "Filter" },
            { id: "motor", name: "Motor Joint" },
            { id: "parallel", name: "Parallel Spring" },
            { id: "prismatic", name: "Prismatic" },
            { id: "revolute", name: "Revolute" },
            { id: "wheel", name: "Wheel" },
        ],
    },
    { name: "Stacking", scenes: [{ id: "boxStack", name: "Box Stack" }] },
];
export interface Control {
    label: string;
    value: number;
    min?: number;
    max?: number;
    step?: number;
    button?: boolean;
    when?: readonly string[];
    set(
        world: World,
        bodies: readonly number[],
        value: number,
        values: Record<string, number>,
    ): void;
}
export type Controls = Record<string, Control>;
function control(
    label: string,
    value: number,
    set: Control["set"],
    min?: number,
    max?: number,
    step?: number,
    when?: string | readonly string[],
): Control {
    return { label, value, set, min, max, step, when: typeof when === "string" ? [when] : when };
}
export function controlValues(scene: Scene): Record<string, number> {
    return Object.fromEntries(
        Object.entries(scenes[scene].controls ?? {}).map(([name, control]) => [
            name,
            control.value,
        ]),
    );
}
export function visibleControls(scene: Scene, values: Record<string, number>): Controls {
    const controls: Controls | undefined = scenes[scene].controls;
    return Object.fromEntries(
        Object.entries(controls ?? {}).filter(
            ([, control]) => !control.when || control.when.every((name) => values[name]),
        ),
    );
}
export function changeControl(
    world: World,
    scene: Scene,
    bodies: readonly number[],
    values: Record<string, number>,
    name: string,
    value: number,
): void {
    const controls: Controls | undefined = scenes[scene].controls;
    const selected = controls?.[name];
    if (!selected) throw new Error(`Unknown ${scene} control: ${name}`);
    values[name] = value;
    selected.set(world, bodies, value, values);
}
export const scenes = {
    ...eventScenes,
    boxStack: {
        construct: boxStack,
        camera: { yaw: 40, pitch: 10, distance: 46, target: [0, 23, 0] },
        controls: undefined,
    },
    bridge: {
        construct: bridge,
        camera: { yaw: 40, pitch: 30, distance: 29, target: [-1.25, 17, 0] },
        controls: {
            "Gravity scale": control(
                "Gravity scale",
                1,
                (world, bodies, value) => setGravityScale(world, bodies, value),
                -1,
                1,
                0.1,
            ),
        },
    },
    revolute: {
        construct: revolute,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Limit: control("Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Lower Angle": control(
                "Lower Angle",
                -35,
                (world, bodies, value, values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    values["Lower Angle"] = value = Math.min(value, values["Upper Angle"]!);
                    joint.lowerAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Limit",
            ),
            "Upper Angle": control(
                "Upper Angle",
                35,
                (world, bodies, value, values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    values["Upper Angle"] = value = Math.max(value, values["Lower Angle"]!);
                    joint.upperAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Torque": control(
                "Max Torque",
                5000,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.maxMotorTorque.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                50000,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.motorSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            Spring: control("Spring", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([RevoluteJoint]);
                if (!eid) return;
                const joint = world.storage(RevoluteJoint);
                joint.enableSpring.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            Hertz: control(
                "Hertz",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Spring",
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Spring",
            ),
            Rotation: control(
                "Rotation",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([RevoluteJoint]);
                    if (!eid) return;
                    const joint = world.storage(RevoluteJoint);
                    joint.targetAngle.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -180,
                180,
                1,
                "Spring",
            ),
        },
    },
    prismatic: {
        construct: prismatic,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Limit: control("Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Lower Translation": control(
                "Lower Translation",
                -1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    values["Lower Translation"] = value = Math.min(
                        value,
                        values["Upper Translation"]!,
                    );
                    joint.lowerTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Limit",
            ),
            "Upper Translation": control(
                "Upper Translation",
                1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    values["Upper Translation"] = value = Math.max(
                        value,
                        values["Lower Translation"]!,
                    );
                    joint.upperTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Force": control(
                "Max Force",
                20,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.maxMotorForce.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                100000,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.motorSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            Spring: control("Spring", 1, (world, bodies, value, _values) => {
                const [eid] = world.query([PrismaticJoint]);
                if (!eid) return;
                const joint = world.storage(PrismaticJoint);
                joint.enableSpring.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            Hertz: control(
                "Hertz",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Spring",
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Spring",
            ),
            Translation: control(
                "Translation",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([PrismaticJoint]);
                    if (!eid) return;
                    const joint = world.storage(PrismaticJoint);
                    joint.targetTranslation.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -20,
                20,
                0.1,
                "Spring",
            ),
        },
    },
    wheel: {
        construct: wheel,
        camera: { yaw: 25, pitch: 20, distance: 7, target: [0, 2, 0] },
        controls: {
            "Suspension Limit": control("Suspension Limit", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSuspensionLimit.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Min##Limit": control(
                "Min suspension",
                -1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    values["Min##Limit"] = value = Math.min(value, values["Max##Limit"]!);
                    joint.lowerSuspensionLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Suspension Limit",
            ),
            "Max##Limit": control(
                "Max suspension",
                1,
                (world, bodies, value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    values["Max##Limit"] = value = Math.max(value, values["Min##Limit"]!);
                    joint.upperSuspensionLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                0.1,
                "Suspension Limit",
            ),
            Motor: control("Motor", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSpinMotor.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Max Torque": control(
                "Max Torque",
                20,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.maxSpinTorque.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                100,
                1,
                "Motor",
            ),
            Speed: control(
                "Speed",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.spinSpeed.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                -10,
                10,
                1,
                "Motor",
            ),
            "Suspension Spring": control(
                "Suspension Spring",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.enableSuspensionSpring.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
            ),
            "Hertz##Suspension": control(
                "Hertz suspension",
                2,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.suspensionHertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Suspension Spring",
            ),
            "Damping##Suspension": control(
                "Damping suspension",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.suspensionDampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Suspension Spring",
            ),
            Steering: control("Steering", 0, (world, bodies, value, _values) => {
                const [eid] = world.query([WheelJoint]);
                if (!eid) return;
                const joint = world.storage(WheelJoint);
                joint.enableSteering.set(eid, value);
                for (const body of bodies) setAwake(world, body, true);
            }),
            "Hertz##Steering": control(
                "Hertz steering",
                1,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.steeringHertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                10,
                0.1,
                "Steering",
            ),
            "Damping##Steering": control(
                "Damping steering",
                0.7,
                (world, bodies, _value, values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint); // Box3D's steering damping control reapplies suspension damping, not steering damping.
                    joint.suspensionDampingRatio.set(eid, values["Damping##Suspension"]!);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
                "Steering",
            ),
            "Degrees##Steering": control(
                "Degrees steering",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.targetSteeringAngle.set(
                        eid,
                        Math.fround(Math.fround(value * Math.fround(Math.PI)) / 180),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -90,
                90,
                1,
                "Steering",
            ),
            "Steering Limit": control(
                "Steering Limit",
                0,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.enableSteeringLimit.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                undefined,
                undefined,
                undefined,
                "Steering",
            ),
            "Min Degrees": control(
                "Min Degrees",
                -45,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.lowerSteeringLimit.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                -90,
                0,
                1,
                ["Steering", "Steering Limit"],
            ),
            "Max Degrees": control(
                "Max Degrees",
                45,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([WheelJoint]);
                    if (!eid) return;
                    const joint = world.storage(WheelJoint);
                    joint.upperSteeringLimit.set(
                        eid,
                        Math.fround(Math.fround(Math.fround(Math.PI) / 180) * value),
                    );
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                90,
                1,
                ["Steering", "Steering Limit"],
            ),
        },
    },
    parallel: {
        construct: parallel,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: {
            Hertz: control(
                "Hertz",
                10,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([ParallelJoint]);
                    if (!eid) return;
                    const joint = world.storage(ParallelJoint);
                    joint.hertz.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                5,
                0.1,
            ),
            Damping: control(
                "Damping",
                0.7,
                (world, bodies, value, _values) => {
                    const [eid] = world.query([ParallelJoint]);
                    if (!eid) return;
                    const joint = world.storage(ParallelJoint);
                    joint.dampingRatio.set(eid, value);
                    for (const body of bodies) setAwake(world, body, true);
                },
                0,
                2,
                0.1,
            ),
        },
    },
    filter: {
        construct: filter,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: undefined,
    },
} as const;

export function createScene(world: World, scene: Scene): number[] {
    world.resource(PhysicsWorldDefinition).workerCount = 1;
    return scenes[scene].construct(world);
}

function ground(world: World, extent: number): number {
    const geometry = world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 0.5] });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0], scale: [2 * extent, 2, 2 * extent] });
    world.add(ground, Body);
    world.add(ground, Shape, { geometry });
    return geometry;
}

function boxStack(world: World): number[] {
    const geometry = ground(world, 40);
    const bodies: number[] = [];
    for (let i = 0; i < 40; i++) {
        const body = world.create();
        world.add(body, Transform, { translation: [0, 0.75 + 1.25 * i, 0] });
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape, { geometry, rollingResistance: 0.1 });
        bodies.push(body);
    }
    return bodies;
}

function bridge(world: World): number[] {
    const geometry = ground(world, 60);
    const bodies: number[] = [];
    const anchor = world.create();
    world.add(anchor, Body);
    let previous = anchor;
    function joint(
        a: number,
        b: number,
        ax: number,
        ay: number,
        bx: number,
        by: number,
        z: number,
    ): void {
        const eid = world.create();
        world.add(eid, SphericalJoint, {
            a,
            b,
            localAnchorA: [ax, ay, z],
            localAnchorB: [bx, by, z],
            constraintHertz: 1000,
            enableSpring: 1,
            hertz: 2,
            dampingRatio: 1,
        });
    }
    for (let i = 0; i < 150; i++) {
        const body = world.create();
        world.add(body, Transform, {
            translation: [-20 + 0.125 * (1 + 2 * i), 20, 0],
            scale: [0.25, 0.25, 1],
        });
        world.add(body, Body, { type: BodyType.Dynamic, linearDamping: 0.1, angularDamping: 0.1 });
        world.add(body, Shape, { geometry, density: 20 });
        for (const z of [-0.5, 0.5])
            joint(previous, body, i === 0 ? -20 : 0.125, i === 0 ? 20 : 0, -0.125, 0, z);
        bodies.push(body);
        previous = body;
    }
    for (const z of [-0.5, 0.5]) joint(previous, anchor, 0.125, 0, 17.5, 20, z);
    return bodies;
}

function hangingBody(world: World, height: number, geometry: number): number {
    const body = world.create();
    world.add(body, Transform, { translation: [0, height, 0] });
    world.add(body, Body, { type: BodyType.Dynamic });
    world.add(body, Shape, { geometry });
    return body;
}

function anchor(world: World): number {
    const body = world.create();
    world.add(body, Transform, { translation: [0, -1, 0] });
    world.add(body, Body);
    return body;
}

function revolute(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world.resource(Hulls).add({ name: "pendulum", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    world.add(world.create(), RevoluteJoint, {
        a,
        b,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        drawScale: 2,
        lowerAngle: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * -35),
        upperAngle: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * 35),
        hertz: 2,
        dampingRatio: 0.7,
        maxMotorTorque: 5000,
    });
    return [b];
}

function prismatic(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world.resource(Hulls).add({ name: "slider", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    world.storage(Body).gravityScale.set(b, 0);
    world.add(world.create(), PrismaticJoint, {
        a,
        b,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        constraintHertz: 120,
        lowerTranslation: -1,
        upperTranslation: 1,
        enableSpring: 1,
        hertz: 2,
        dampingRatio: 0.7,
        maxMotorForce: 20,
    });
    return [b];
}

const diagonal = 0.7071067690849304;
function wheel(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const geometry = world
        .resource(Hulls)
        .add({ name: "wheel", cylinder: { height: 0.25, radius: 0.4, yOffset: 0, sides: 12 } });
    const b = hangingBody(world, 2, geometry);
    world.storage(Transform).rotation.set(b, diagonal, 0, 0, diagonal);
    world.add(world.create(), WheelJoint, {
        a,
        b,
        localAnchorA: [0, 3, 0],
        localRotationA: [0, 0, diagonal, diagonal],
        localRotationB: [-diagonal, 0, 0, diagonal],
        collideConnected: 1,
        enableSuspensionSpring: 0,
        suspensionHertz: 2,
        suspensionDampingRatio: 0.7,
        lowerSuspensionLimit: -1,
        upperSuspensionLimit: 1,
        maxSpinTorque: 20,
        maxSteeringTorque: 20,
        lowerSteeringLimit: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * -45),
        upperSteeringLimit: Math.fround(Math.fround(Math.fround(Math.PI) / 180) * 45),
    });
    return [b];
}

function filter(world: World): number[] {
    const geometry = ground(world, 20);
    const bodies: number[] = [];
    for (const x of [2, -2]) {
        const body = hangingBody(world, 4, geometry);
        world.storage(Transform).translation.set(body, x, 4, 0);
        bodies.push(body);
    }
    world.add(world.create(), FilterJoint, { a: bodies[0]!, b: bodies[1]! });
    return bodies;
}

function parallel(world: World): number[] {
    ground(world, 20);
    const a = anchor(world);
    const walls = [
        { halfExtents: [20, 5, 0.1], position: [0, 5, -20] },
        { halfExtents: [20, 5, 0.1], position: [0, 5, 20] },
        { halfExtents: [0.1, 5, 20], position: [-20, 5, 0] },
        { halfExtents: [0.1, 5, 20], position: [20, 5, 0] },
    ] as const;
    for (const wall of walls) {
        const geometry = world.resource(Hulls).add({
            name: "wall",
            halfExtents: wall.halfExtents,
            transform: { position: wall.position, rotation: [0, 0, 0, 1] },
        });
        world.add(world.create(), Shape, { body: a, geometry });
    }
    const geometry = world
        .resource(Hulls)
        .add({ name: "parallel spring", halfExtents: [0.5, 1.5, 0.25] });
    const b = hangingBody(world, 4, geometry);
    // b3MakeQuatFromAxisAngle(axisX, 0.25f * B3_PI), using Box3D's deterministic trig.
    const x = 0.3837169408798218,
        w = 0.9234507083892822;
    world.storage(Transform).rotation.set(b, x, 0, 0, w);
    const f = Math.fround;
    world.add(world.create(), ParallelJoint, {
        a,
        b,
        localRotationA: [-diagonal, 0, 0, diagonal],
        localRotationB: [
            f(f(w * -diagonal) - f(x * diagonal)),
            0,
            0,
            f(f(w * diagonal) + f(x * -diagonal)),
        ],
        drawScale: 2,
        collideConnected: 1,
        hertz: 10,
        dampingRatio: 0.7,
    });
    return [b];
}

export function setGravityScale(world: World, bodies: readonly number[], scale: number): void {
    const gravity = world.storage(Body).gravityScale;
    for (const body of bodies) gravity.set(body, scale);
}
