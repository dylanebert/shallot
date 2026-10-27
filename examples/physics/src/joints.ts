import {
    Body,
    BodyType,
    DistanceJoint,
    GlobalTransform,
    Hulls,
    MotorJoint,
    PhysicsMeshes,
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
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAwake,
} from "@dylanebert/shallot/standard/physics";
import earcut from "earcut";
import { DrivingPlugin, drivingScene } from "./driving";
import { MotorizedJoint } from "./roles";
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
    dependencies: [StandardPhysicsPlugin, DrivingPlugin],
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

// Box3D GearLift: extruded stairwell, paired toothed shafts and two gate chains.
function gearLift(world: World): number[] {
    const f = Math.fround;
    const pi = f(Math.PI);
    ground(world);
    const anchor = body(world, [0, 0, 0], false);
    const ring = [
        [-11.3, -0.2167],
        [9.3375, -0.2167],
        [9.3375, 7.1917],
        [8.8083, 7.1917],
        [8.8083, 0.3125],
        [0.3417, 0.3125],
        [0.3417, 0.8417],
        [-0.1875, 0.8417],
        [-0.1875, 1.3708],
        [-0.7167, 1.3708],
        [-0.7167, 1.9],
        [-1.2458, 1.9],
        [-1.2458, 2.4292],
        [-1.775, 2.4292],
        [-1.775, 2.9583],
        [-2.3042, 2.9583],
        [-2.3042, 3.4875],
        [-2.8333, 3.4875],
        [-2.8333, 4.0167],
        [-3.3625, 4.0167],
        [-3.3625, 4.5458],
        [-3.8917, 4.5458],
        [-3.8917, 5.075],
        [-4.4208, 5.075],
        [-4.4208, 5.6042],
        [-4.95, 5.6042],
        [-4.95, 6.1333],
        [-5.4792, 6.1333],
        [-5.4792, 6.6625],
        [-6.0083, 6.6625],
        [-6.0083, 7.1917],
        [-11.3, 7.1917],
    ].map(([x, y]) => [f(x!), f(y!)] as [number, number]);
    const vertices = ring.flatMap(([x, y]) => [x, y, -2, x, y, 2]);
    const indices: number[] = [];
    for (let i = 0; i < 32; i++) {
        const j = (i + 1) % 32;
        indices.push(2 * i, 2 * j, 2 * j + 1, 2 * i, 2 * j + 1, 2 * i + 1);
    }
    const caps = earcut(ring.flat());
    for (let k = 0; k < caps.length; k += 3) {
        const a = caps[k]!,
            b = caps[k + 1]!,
            c = caps[k + 2]!;
        const [ax, ay] = ring[a]!,
            [bx, by] = ring[b]!,
            [cx, cy] = ring[c]!;
        const positive = f(f(f(bx - ax) * f(cy - ay)) - f(f(by - ay) * f(cx - ax))) > 0;
        indices.push(2 * a + 1, 2 * (positive ? b : c) + 1, 2 * (positive ? c : b) + 1);
        indices.push(2 * a, 2 * (positive ? c : b), 2 * (positive ? b : c));
    }
    const mesh = world.resource(PhysicsMeshes).add({
        name: "stairwell",
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        identifyEdges: true,
    });
    world.add(anchor, Shape, { kind: ShapeKind.Mesh, geometry: mesh });
    const lx = ring[0]![0],
        ly = ring[0]![1],
        ux = ring[1]![0],
        uy = ring[2]![1];
    box(
        world,
        anchor,
        [f(0.5 * f(ux - lx)), f(0.5 * f(uy - ly)), f(0.05)],
        [f(0.5 * f(lx + ux)), f(0.5 * f(ly + uy)), f(-2 - f(0.05))],
    );
    function hull(name: string, points: number[]): number {
        return world
            .resource(Hulls)
            .add({ name, points: new Float32Array(points), maxVertexCount: points.length / 3 });
    }
    function cylinder(radius: number, lo: number, hi: number, sides: number): number {
        const points: number[] = [];
        for (let i = 0; i < sides; i++) {
            const angle = f(f(f(2 * pi) * i) / sides);
            const x = f(radius * f(Math.cos(angle))),
                y = f(radius * f(Math.sin(angle)));
            points.push(x, y, lo, x, y, hi);
        }
        return hull("shaft", points);
    }
    const near = cylinder(1, -1.625, -1.375, 24),
        far = cylinder(1, 1.375, 1.625, 24);
    const axle = cylinder(f(0.2), -1.5, 1.5, 12);
    function shape(eid: number, geometry: number): number {
        const sid = world.has(eid, Shape) ? world.create() : eid;
        world.add(sid, Shape, { geometry, body: sid === eid ? 0 : eid, friction: 0.1 });
        return sid;
    }
    const bodies: number[] = [];
    function dynamic(position: V3): number {
        const eid = body(world, position);
        bodies.push(eid);
        return eid;
    }
    function gear(position: V3, radius: number): number {
        const eid = dynamic(position);
        shape(eid, near);
        shape(eid, far);
        shape(eid, axle);
        const delta = f(f(2 * pi) / 16);
        for (const depth of [-1.5, 1.5])
            for (let i = 0; i < 16; i++) {
                const q = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(i * delta));
                const z = q[2],
                    w = q[3];
                function rotate(x: number, y: number): [number, number] {
                    const tx = f(f(-z * y) + f(w * x)),
                        ty = f(f(z * x) + f(w * y));
                    return [f(x + f(2 * f(-z * ty))), f(y + f(2 * f(z * tx)))];
                }
                const center = rotate(radius, 0);
                const points: number[] = [];
                const hx = f(0.11),
                    base = f(0.09),
                    tip = f(base - f(0.03));
                for (const [x, y, z] of [
                    [-hx, -base, -0.125],
                    [-hx, base, -0.125],
                    [-hx, base, 0.125],
                    [-hx, -base, 0.125],
                    [hx, -tip, -0.125],
                    [hx, tip, -0.125],
                    [hx, tip, 0.125],
                    [hx, -tip, 0.125],
                ] as V3[]) {
                    const p = rotate(x, y);
                    points.push(f(center[0] + p[0]), f(center[1] + p[1]), f(depth + z));
                }
                shape(eid, hull("tooth", points));
            }
        return eid;
    }
    const driver = gear([-4.25, 9.75, 0], f(1 + f(0.09)));
    const follower = gear([-2.25, 10.75, 0], f(1 + f(0.11)));
    const motor = world.create();
    world.add(motor, RevoluteJoint, {
        a: anchor,
        b: driver,
        localAnchorA: [-4.25, 9.75, 0],
        enableMotor: 1,
        maxMotorTorque: 30000,
        motorSpeed: -0.3,
    });
    world.add(motor, MotorizedJoint);
    world.add(world.create(), RevoluteJoint, {
        a: anchor,
        b: follower,
        localAnchorA: [-2.25, 10.75, 0],
        localRotationA: rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(0.25 * pi)),
        enableMotor: 1,
        maxMotorTorque: 0.5,
        enableLimit: 1,
        lowerAngle: f(f(-0.3) * pi),
        upperAngle: f(f(0.8) * pi),
    });
    const attachX = f(f(f(-2.25 + 1) + f(2 * f(0.11))) + f(0.03));
    const doorY = f(10.75 - f(f(80 * f(0.07)) + 1.5));
    const lastLinks: number[] = [];
    for (const depth of [-1.5, 1.5]) {
        let previous = follower,
            y = f(10.75 - f(0.07));
        for (let i = 0; i < 40; i++) {
            const eid = dynamic([attachX, y, depth]);
            world.add(eid, Shape, {
                kind: ShapeKind.Capsule,
                capsuleCenter1: [0, -0.07, 0],
                capsuleCenter2: [0, 0.07, 0],
                radius: 0.05,
            });
            const pivotY = f(y + f(0.07));
            world.add(world.create(), RevoluteJoint, {
                a: previous,
                b: eid,
                localAnchorA: i
                    ? [0, f(pivotY - f(y + f(2 * f(0.07)))), 0]
                    : [f(attachX + 2.25), f(pivotY - 10.75), depth],
                localAnchorB: [0, f(pivotY - y), 0],
                enableMotor: 1,
                maxMotorTorque: 0.05,
                drawScale: 0.2,
            });
            previous = eid;
            y = f(y - f(2 * f(0.07)));
        }
        lastLinks.push(previous);
    }
    const door = dynamic([attachX, doorY, 0]);
    box(world, door, [0.05, 1.5, 1.95]);
    world.storage(Shape).density.set(door, 500);
    world.storage(Shape).friction.set(door, 0.1);
    for (const [i, link] of lastLinks.entries()) {
        const y = world.storage(Transform).translation.y.get(link);
        world.add(world.create(), RevoluteJoint, {
            a: link,
            b: door,
            localAnchorA: [0, f(f(doorY + 1.5) - y), 0],
            localAnchorB: [0, 1.5, i ? 1.5 : -1.5],
            enableMotor: 1,
            maxMotorTorque: 50,
        });
    }
    const slide: [number, number, number, number] = [0, 0, f(Math.SQRT1_2), f(Math.SQRT1_2)];
    world.add(world.create(), PrismaticJoint, {
        a: anchor,
        b: door,
        localAnchorA: [attachX, doorY, 0],
        localRotationA: slide,
        localRotationB: slide,
        maxMotorForce: 200,
        enableMotor: 1,
        collideConnected: 1,
    });
    const phi = f(f(1 + f(Math.sqrt(5))) / 2);
    const [dc, ds] = computeCosSin(world, f(f(2 * pi) / phi));
    let c = 1,
        s = 0;
    const points: number[] = [];
    for (let i = 0; i < 10; i++) {
        const z = f(1 - f((2 * i + 1) / 10)),
            r = f(Math.sqrt(f(1 - f(z * z))));
        points.push(f(f(f(0.3) * r) * c), f(f(f(0.3) * r) * s), f(f(0.3) * z));
        const previous = c;
        c = f(f(dc * c) - f(ds * s));
        s = f(f(ds * previous) + f(dc * s));
    }
    const rock = hull("rock", points);
    let seed = 12345;
    function randomInt(): number {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return seed % 32768;
    }
    function random(lo: number, hi: number): number {
        return f(f(f(hi - lo) * f(randomInt() / 32767)) + lo);
    }
    let x = -5;
    for (let i = 0; i < 12; i++) {
        let y = f(6.5 - f(0.25 * i));
        for (let j = 0; j < 10; j++) {
            const z = random(f(-1.65), f(0.35));
            const u1 = random(0, 1),
                u2 = random(0, f(2 * pi)),
                u3 = random(0, f(2 * pi));
            const a = f(Math.sqrt(f(1 - u1))),
                b = f(Math.sqrt(u1));
            const [c2, s2] = computeCosSin(world, u2),
                [c3, s3] = computeCosSin(world, u3);
            const eid = dynamic([x, y, z]);
            world.storage(Transform).rotation.set(eid, f(a * s2), f(a * c2), f(b * s3), f(b * c3));
            randomInt(); // The sample consumes a random colour for each rock.
            world.add(eid, Shape, { geometry: rock, rollingResistance: 0.3 });
            y = f(y + f(0.2));
        }
        x = f(x + f(0.3));
    }
    return bodies;
}
const gearLiftControls = Object.fromEntries(
    (
        [
            ["Motor", "enableMotor", 1],
            ["Max Torque", "maxMotorTorque", 30000, 0, 100000, 1],
            ["Speed", "motorSpeed", -0.3, -0.3, 0.3, 0.01],
        ] as const
    ).map(([label, field, value, min, max, step]) => [
        label,
        {
            label,
            value,
            min,
            max,
            step,
            when: label === "Motor" ? undefined : ["Motor"],
            set(world, _bodies, value) {
                const joint = [...world.query([RevoluteJoint, MotorizedJoint])][0]!;
                const storage = world.storage(RevoluteJoint);
                storage[field].set(joint, value);
                setAwake(world, storage.b.get(joint), true);
            },
        } satisfies Controls[string],
    ]),
);

export const jointScenes = {
    driving: drivingScene,
    gearLift: {
        construct: gearLift,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 15, pitch: 15, distance: 26, target: [-1.5, 5, 0] },
        controls: gearLiftControls,
    },
    door: {
        construct: door,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: doorControls,
    },
    distance: {
        construct: distance,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 15, distance: 14, target: [2, 17, 0] },
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
