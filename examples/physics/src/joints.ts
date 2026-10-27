import {
    Body,
    BodyType,
    DistanceJoint,
    GlobalTransform,
    Hulls,
    MotorJoint,
    type Plugin,
    PrismaticJoint,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    SphericalJoint,
    type System,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import {
    applyLinearImpulse,
    computeCosSin,
    explode,
    getJointLinearSeparation,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAwake,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

type V3 = [number, number, number];
function body(world: World, position: V3, dynamic = true): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function box(world: World, eid: number, halfExtents: V3, translation: V3 = [0, 0, 0]): void {
    const geometry = world.resource(Hulls).add({
        name: "box",
        halfExtents,
        transform: { position: translation, rotation: [0, 0, 0, 1] },
    });
    const shape = world.has(eid, Shape) ? world.create() : eid;
    world.add(shape, Shape, { geometry, body: eid === shape ? 0 : eid });
}
function ground(world: World): number {
    const eid = body(world, [0, -1, 0], false);
    box(world, eid, [20, 1, 20]);
    return eid;
}
function ballAndChain(world: World): number[] {
    let parent = body(world, [0, 0, 0], false);
    const bodies: number[] = [];
    for (let i = 0; i <= 32; i++) {
        const last = i === 32;
        const eid = body(world, [last ? 34 : 0.5 + i, 0, 0]);
        world.add(
            eid,
            Shape,
            last
                ? { kind: ShapeKind.Sphere, radius: 2 }
                : {
                      kind: ShapeKind.Capsule,
                      capsuleCenter1: [-0.5, 0, 0],
                      capsuleCenter2: [0.5, 0, 0],
                      radius: 0.125,
                  },
        );
        world.add(world.create(), SphericalJoint, {
            a: parent,
            b: eid,
            localAnchorA: [i ? 0.5 : 0, 0, 0],
            localAnchorB: [last ? -2 : -0.5, 0, 0],
            enableMotor: 1,
            maxMotorTorque: 10,
        });
        bodies.push(eid);
        parent = eid;
    }
    return bodies;
}
function foldOut(world: World): number[] {
    let parent = ground(world);
    const bodies: number[] = [];
    const q: [number, number, number, number] = [
        -Math.fround(Math.SQRT1_2),
        0,
        0,
        Math.fround(Math.SQRT1_2),
    ];
    for (let i = 0; i < 6; i++) {
        const eid = body(world, [0, 0.5 + i, 0]);
        box(world, eid, [2, 0.1, 0.25]);
        world.storage(Shape).density.set(eid, 2000);
        const x = i & 1 ? 2 : -2;
        world.add(world.create(), RevoluteJoint, {
            a: parent,
            b: eid,
            localAnchorA: [x, i ? 0.5 : 1, 0],
            localAnchorB: [x, -0.5, 0],
            localRotationA: q,
            localRotationB: q,
            constraintHertz: 240,
            drawScale: 0.5,
            enableMotor: 1,
            maxMotorTorque: 1000,
        });
        bodies.push(eid);
        parent = eid;
    }
    return bodies;
}
function topDownFriction(world: World): number[] {
    const ground = body(world, [0, 0, 0], false);
    box(world, ground, [10, 0.5, 4]);
    box(world, ground, [0.5, 10, 4], [-10, 10, 0]);
    box(world, ground, [0.5, 10, 4], [10, 10, 0]);
    box(world, ground, [10, 0.5, 4], [0, 20, 0]);
    const bodies: number[] = [];
    for (let i = 0; i < 10; i++)
        for (let j = 0; j < 10; j++) {
            const eid = body(world, [-5 + j, 15 - i, 0]);
            world.storage(Body).gravityScale.set(eid, 0);
            const remainder = (10 * i + j) % 4;
            if (remainder === 0)
                world.add(eid, Shape, {
                    kind: ShapeKind.Capsule,
                    capsuleCenter1: [-0.25, 0, 0],
                    capsuleCenter2: [0.25, 0, 0],
                    radius: 0.25,
                });
            else if (remainder === 1)
                world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.35 });
            else box(world, eid, [0.35, 0.35, 0.35]);
            world.storage(Shape).restitution.set(eid, 0.8);
            world.add(world.create(), MotorJoint, {
                a: ground,
                b: eid,
                collideConnected: 1,
                maxVelocityForce: 1000,
                maxVelocityTorque: 1000,
            });
            bodies.push(eid);
        }
    return bodies;
}
function weld(world: World): number[] {
    ground(world);
    const anchor = body(world, [0, -1, 0], false);
    const eid = body(world, [0, 4, 0]);
    world.storage(Body).gravityScale.set(eid, 0);
    box(world, eid, [0.5, 1.5, 0.25]);
    world.add(world.create(), WeldJoint, {
        a: anchor,
        b: eid,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        constraintHertz: 240,
        angularHertz: 2,
        angularDampingRatio: 0.7,
        drawScale: 2,
    });
    return [eid];
}
const weldControls = Object.fromEntries(
    (
        [
            ["Linear Hertz", "linearHertz", 0, 10],
            ["Linear Damping", "linearDampingRatio", 0, 2],
            ["Angular Hertz", "angularHertz", 2, 10],
            ["Angular Damping", "angularDampingRatio", 0.7, 2],
        ] as const
    ).map(([label, field, value, max]) => [
        label,
        {
            label,
            value,
            min: 0,
            max,
            step: 0.1,
            set(world, bodies, value) {
                for (const eid of world.query([WeldJoint]))
                    world.storage(WeldJoint)[field].set(eid, value);
                for (const eid of bodies) setAwake(world, eid, true);
            },
        } satisfies Controls[string],
    ]),
);
type JointControl = [string, string, number, number?, number?, number?, string?];
const distanceRows: JointControl[] = [
    ["Length", "length", 1, 0.1, 4, 0.1],
    ["Spring", "enableSpring", 0],
    ["Tension", "lowerSpringForce", 2000, 0, 4000, 0.001, "Spring"],
    ["Compression", "upperSpringForce", 100, 0, 200, 0.001, "Spring"],
    ["Hertz", "hertz", 5, 0, 15, 0.1, "Spring"],
    ["Damping", "dampingRatio", 0.5, 0, 4, 0.1, "Spring"],
    ["Limit", "enableLimit", 0],
    ["Min", "minLength", 1, 0.1, 4, 0.1, "Limit"],
    ["Max", "maxLength", 1, 0.1, 4, 0.1, "Limit"],
];
function distanceChain(
    world: World,
    anchor: number,
    count: number,
    values: Record<string, number>,
): number[] {
    let parent = anchor;
    const bodies: number[] = [];
    const length = Math.fround(values.Length!);
    for (let i = 0; i < count; i++) {
        const eid = body(world, [Math.fround(length * (i + 1)), 20, 0]);
        world.storage(Body).angularDamping.set(eid, 1);
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.25, density: 20 });
        world.add(world.create(), DistanceJoint, {
            a: parent,
            b: eid,
            localAnchorA: i ? [0, 0, 0] : [0, 20, 0],
            ...Object.fromEntries(
                distanceRows.map(([label, field]) => [
                    field,
                    label === "Tension" ? -values[label]! : values[label]!,
                ]),
            ),
        });
        bodies.push(eid);
        parent = eid;
    }
    return bodies;
}
function distance(world: World): number[] {
    ground(world);
    const anchor = body(world, [0, 0, 0], false);
    return distanceChain(
        world,
        anchor,
        1,
        Object.fromEntries(distanceRows.map(([label, , value]) => [label, value])),
    );
}
const distanceControls: Controls = Object.fromEntries(
    distanceRows.map(([label, field, value, min, max, step, when]) => [
        when ? `${label}##${when}` : label,
        {
            label,
            value,
            min,
            max,
            step,
            when: when ? [when] : undefined,
            set(world, bodies, value) {
                for (const eid of world.query([DistanceJoint]))
                    world
                        .storage(DistanceJoint)
                        [field as "length"].set(eid, label === "Tension" ? -value : value);
                for (const eid of bodies) setAwake(world, eid, true);
            },
        } satisfies Controls[string],
    ]),
);
distanceControls.Count = {
    label: "Count",
    value: 1,
    min: 1,
    max: 20,
    step: 1,
    set(world, bodies, value, values) {
        const [joint] = world.query([DistanceJoint]);
        const anchor = world.storage(DistanceJoint).a.get(joint!);
        for (const eid of world.query([DistanceJoint])) world.destroy(eid);
        for (const eid of bodies) world.destroy(eid);
        (bodies as number[]).splice(
            0,
            bodies.length,
            ...distanceChain(
                world,
                anchor,
                value,
                Object.fromEntries(
                    distanceRows.map(([label, , , , , , when]) => [
                        label,
                        values[when ? `${label}##${when}` : label]!,
                    ]),
                ),
            ),
        );
    },
};
const radians = (degrees: number) =>
    Math.fround(Math.fround(Math.fround(Math.PI) / 180) * Math.fround(degrees));
const sphericalRows: JointControl[] = [
    ["Cone Limit", "enableConeLimit", 0],
    ["Cone Angle", "coneAngle", 30, 0, 90, 1, "Cone Limit"],
    ["Twist Limit", "enableTwistLimit", 0],
    ["Lower Twist", "lowerTwistAngle", -35, -180, 180, 1, "Twist Limit"],
    ["Upper Twist", "upperTwistAngle", 35, -180, 180, 1, "Twist Limit"],
    ["Motor", "enableMotor", 0],
    ["Max Torque", "maxMotorTorque", 20, 0, 10000, 1, "Motor"],
    ...["X", "Y", "Z"].map(
        (axis): JointControl => [`Velocity ${axis}`, "motorVelocity", 0, -10, 10, 1, "Motor"],
    ),
    ["Spring", "enableSpring", 1],
    ["Hertz", "hertz", 2, 0, 10, 0.1, "Spring"],
    ["Damping", "dampingRatio", 0.7, 0, 2, 0.1, "Spring"],
    ...["X", "Y", "Z"].map(
        (axis): JointControl => [`Rotation ${axis}`, "targetRotation", 0, -180, 180, 1, "Spring"],
    ),
];
function spherical(world: World): number[] {
    ground(world);
    const anchor = body(world, [0, -1, 0], false);
    const eid = body(world, [0, 4, 0]);
    world.storage(Body).gravityScale.set(eid, 0);
    box(world, eid, [0.5, 1.5, 0.25]);
    world.storage(Shape).density.set(eid, 100);
    world.add(world.create(), SphericalJoint, {
        a: anchor,
        b: eid,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        drawScale: 2,
        coneAngle: radians(30),
        lowerTwistAngle: radians(-35),
        upperTwistAngle: radians(35),
        hertz: 2,
        dampingRatio: 0.7,
        maxMotorTorque: 20,
        enableSpring: 1,
    });
    return [eid];
}
const sphericalControls: Controls = Object.fromEntries(
    sphericalRows.map(([label, field, value, min, max, step, when]) => [
        label,
        {
            label,
            value,
            min,
            max,
            step,
            when: when ? [when] : undefined,
            set(world, bodies, value, values) {
                const [eid] = world.query([SphericalJoint]);
                const joint = world.storage(SphericalJoint);
                if (label === "Lower Twist")
                    values[label] = value = Math.min(value, values["Upper Twist"]!);
                if (label === "Upper Twist")
                    values[label] = value = Math.max(value, values["Lower Twist"]!);
                if (field === "motorVelocity")
                    joint.motorVelocity.set(
                        eid!,
                        values["Velocity X"]!,
                        values["Velocity Y"]!,
                        values["Velocity Z"]!,
                    );
                else if (field === "targetRotation") {
                    const f = Math.fround;
                    const angles = ["X", "Y", "Z"].map((axis) =>
                        f(0.5 * radians(values[`Rotation ${axis}`]!)),
                    );
                    const cs = angles.map((a) => computeCosSin(world, a));
                    const [sx, sy, sz] = cs.map((pair) => pair[1]);
                    const [cx, cy, cz] = cs.map((pair) => pair[0]);
                    joint.targetRotation.set(
                        eid!,
                        f(f(cz! * f(cy! * sx!)) - f(sz! * f(sy! * cx!))),
                        f(f(cz! * f(sy! * cx!)) + f(sz! * f(cy! * sx!))),
                        f(f(sz! * f(cy! * cx!)) - f(cz! * f(sy! * sx!))),
                        f(f(cz! * f(cy! * cx!)) + f(sz! * f(sy! * sx!))),
                    );
                } else
                    joint[field as "hertz"].set(
                        eid!,
                        field.includes("Angle") ? radians(value) : value,
                    );
                for (const body of bodies) setAwake(world, body, true);
            },
        } satisfies Controls[string],
    ]),
);
function motionLocks(world: World): number[] {
    ground(world);
    const anchor = body(world, [0, 0, 0], false);
    return [DistanceJoint, PrismaticJoint, RevoluteJoint, WeldJoint].map((kind, i) => {
        const x = -12.5 + 5 * i;
        const eid = body(world, [x, 10, 0]);
        world.storage(Body).enableSleep.set(eid, 0);
        world.storage(Body).angularDamping.set(eid, 0.5);
        box(world, eid, [1, 1, 0.5]);
        world.storage(Shape).density.set(eid, 1);
        const base = {
            a: anchor,
            b: eid,
            forceThreshold: 20000,
            torqueThreshold: 10000,
            collideConnected: 1,
        };
        if (kind === DistanceJoint)
            world.add(world.create(), DistanceJoint, {
                ...base,
                localAnchorA: [x, 13, 0],
                localAnchorB: [0, 1, 0],
                length: 2,
            });
        else if (kind === WeldJoint)
            world.add(world.create(), WeldJoint, {
                ...base,
                localAnchorA: [x - 1, 10, 0],
                localAnchorB: [-1, 0, 0],
                angularHertz: 2,
                angularDampingRatio: 0.5,
            });
        else
            world.add(world.create(), kind, {
                ...base,
                localAnchorA: [x - 1, 10, 0],
                localAnchorB: [-1, 0, 0],
            });
        return eid;
    });
}
const motionLockControls: Controls = Object.fromEntries(
    ["Linear X", "Linear Y", "Linear Z", "Angular X", "Angular Y", "Angular Z"].map((axis, bit) => [
        `Lock ${axis}`,
        {
            label: `Lock ${axis}`,
            value: 0,
            set(world, bodies, value) {
                const locks = world.storage(Body).motionLocks;
                for (const eid of bodies) {
                    locks.set(
                        eid,
                        value ? locks.get(eid) | (1 << bit) : locks.get(eid) & ~(1 << bit),
                    );
                    setAwake(world, eid, true);
                }
            },
        } satisfies Controls[string],
    ]),
);

export const DoorState = {
    create: () => ({ joints: [] as number[], errors: [0, 0], impulse: 0, door: 0 }),
};
const BeforeDoor: System = {
    name: "door-before-step",
    group: "fixed",
    after: [SyncPhysicsConstraintsSystem],
    before: [StepPhysicsSystem],
    update(world) {
        const state = world.resource(DoorState);
        if (!state.impulse) return;
        const pose = world.storage(GlobalTransform);
        const q = pose.rotation,
            p = pose.translation,
            eid = state.door;
        const f = Math.fround;
        const x = q.x.get(eid),
            y = q.y.get(eid),
            z = q.z.get(eid),
            w = q.w.get(eid);
        const tx = 0,
            ty = f(z * 1.5),
            tz = f(-y * 1.5);
        const dx = f(0.75 + f(f(y * tz) - f(z * ty)));
        const dy = f(f(w * ty) + f(f(z * tx) - f(x * tz)));
        const dz = f(f(w * tz) + f(f(x * ty) - f(y * tx)));
        applyLinearImpulse(
            world,
            eid,
            { x: 0, y: 0, z: -state.impulse },
            { x: p.x.get(eid) + dx, y: p.y.get(eid) + dy, z: p.z.get(eid) + dz },
        );
        state.impulse = 0;
    },
};
const AfterDoor: System = {
    name: "door-after-step",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const state = world.resource(DoorState);
        state.joints.forEach((eid, i) => {
            state.errors[i] = Math.max(state.errors[i]!, getJointLinearSeparation(world, eid) ?? 0);
        });
    },
};
export const JointSamplePlugin: Plugin = {
    name: "JointSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [BeforeDoor, AfterDoor],
    recovery: recoverResource(DoorState),
};
function doorJoints(
    world: World,
    anchor: number,
    eid: number,
    two: number,
    hertz: number,
    damping: number,
): void {
    const state = world.resource(DoorState);
    for (const joint of state.joints) world.destroy(joint);
    state.joints = [];
    const q: [number, number, number, number] = [
        -Math.fround(Math.SQRT1_2),
        0,
        0,
        Math.fround(Math.SQRT1_2),
    ];
    for (let i = 0; i < (two ? 2 : 1); i++) {
        const joint = world.create();
        world.add(joint, RevoluteJoint, {
            a: anchor,
            b: eid,
            localAnchorA: [-0.75, 1 + 3 * i, 0],
            localAnchorB: [-0.75, -1.5 + 3 * i, 0],
            localRotationA: q,
            localRotationB: q,
            constraintHertz: hertz,
            constraintDampingRatio: damping,
            enableLimit: 1,
            lowerAngle: radians(-90),
            upperAngle: radians(90),
            enableSpring: 1,
            hertz: 1,
            dampingRatio: 0.5,
            maxMotorTorque: 100,
            drawScale: 2,
        });
        state.joints.push(joint);
    }
}
function door(world: World): number[] {
    const anchor = ground(world);
    const eid = body(world, [0, 1.5, 0]);
    world.storage(Body).gravityScale.set(eid, 2);
    box(world, eid, [0.75, 1.5, 0.1]);
    world.storage(Shape).density.set(eid, 1000);
    const state = world.resource(DoorState);
    state.door = eid;
    doorJoints(world, anchor, eid, 1, 120, 0);
    return [eid];
}
const doorControls: Controls = {
    Impulse: {
        label: "Impulse",
        value: 0,
        button: true,
        set(world, _bodies, _value, values) {
            const state = world.resource(DoorState);
            state.impulse = values.Magnitude!;
            state.errors.fill(0);
        },
    },
    Magnitude: { label: "Magnitude", value: 50000, min: 1000, max: 100000, step: 1, set() {} },
    Limit: {
        label: "Limit",
        value: 1,
        set(world, _bodies, value) {
            for (const eid of world.resource(DoorState).joints)
                world.storage(RevoluteJoint).enableLimit.set(eid, value);
        },
    },
    "Two joints": {
        label: "Two joints",
        value: 1,
        set(world, bodies, value, values) {
            const [joint] = world.resource(DoorState).joints;
            const anchor = world.storage(RevoluteJoint).a.get(joint!);
            doorJoints(world, anchor, bodies[0]!, value, values.Hertz!, values.Damping!);
        },
    },
    ...Object.fromEntries(
        (
            [
                ["Hertz", "constraintHertz", 120, 15, 240, 1],
                ["Damping", "constraintDampingRatio", 0, 0, 10, 0.1],
            ] as const
        ).map(([label, field, value, min, max, step]) => [
            label,
            {
                label,
                value,
                min,
                max,
                step,
                set(world, _bodies, value) {
                    for (const eid of world.resource(DoorState).joints)
                        world.storage(RevoluteJoint)[field].set(eid, value);
                },
            } satisfies Controls[string],
        ]),
    ),
    ...Object.fromEntries(
        [0, 1].map((i) => [
            `Translation error ${i + 1}`,
            {
                label: `Translation error ${i + 1}`,
                value: 0,
                read(world) {
                    const state = world.resource(DoorState);
                    return state.joints[i] === undefined ? null : state.errors[i]!;
                },
                set() {},
            } satisfies Controls[string],
        ]),
    ),
};

export const jointScenes = {
    door: {
        construct: door,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: doorControls,
    },
    distance: {
        construct: distance,
        gridExtent: 25,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 15, distance: 40, target: [0, 10, 0] },
        controls: distanceControls,
    },
    spherical: {
        construct: spherical,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: sphericalControls,
    },
    motionLocks: {
        construct: motionLocks,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 30, distance: 40, target: [0, 5, 0] },
        controls: motionLockControls,
    },
    ballAndChain: {
        construct: ballAndChain,
        gridExtent: 40,
        gridSpacing: 2,
        camera: { yaw: 25, pitch: 40, distance: 60, target: [8, -15, 0] },
        controls: undefined,
    },
    foldOut: {
        construct: foldOut,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 10, distance: 22, target: [0, 5, 0] },
        controls: undefined,
    },
    topDownFriction: {
        construct: topDownFriction,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 0, distance: 28, target: [0, 10, 0] },
        controls: {
            Explode: {
                label: "Explode",
                value: 0,
                button: true,
                set(world) {
                    explode(world, {
                        maskBits: 0xffffffffffffffffn,
                        position: { x: 0, y: 10, z: 0 },
                        radius: 10,
                        falloff: 5,
                        impulsePerArea: 10000,
                    });
                },
            },
        } satisfies Controls,
    },
    weld: {
        construct: weld,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: weldControls,
    },
} as const;
