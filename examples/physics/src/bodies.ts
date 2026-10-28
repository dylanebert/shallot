import {
    Body,
    BodyMotionLock,
    BodyType,
    GlobalTransform,
    Hulls,
    type Plugin,
    PrismaticJoint,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import {
    applyLinearImpulseToCenter,
    computeCosSin,
    explode,
    getMassData,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
    setAwake,
    setBodyType,
    setLinearVelocity,
    setMassData,
    setTargetTransform,
    setTransform,
    stepPhysics,
} from "@dylanebert/shallot/standard/physics";
import { CastPlugin, castScene } from "./cast";
import { gyroscopicPrecession, PrecessionPlugin } from "./precession";
import type { Controls } from "./scenes";

type V3 = [number, number, number];
const f = Math.fround;
export const BodySampleState = {
    create: () => ({
        ring: 0,
        ringVelocity: [0, 0, 0] as V3,
        target: 0,
        offset: 0,
        spin: 0,
        weeble: 0,
        link: 0,
        platform: 0,
        type: BodyType.Dynamic as number,
        magnitude: 20000,
        actions: [] as string[],
        time: 0,
        initialized: false,
    }),
};
export const BodySamplePlugin: Plugin = {
    name: "BodySample",
    dependencies: [StandardPhysicsPlugin, PrecessionPlugin, CastPlugin],
    recovery: recoverResource(BodySampleState),
    systems: [
        {
            name: "class-ring-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                if (world.resource(BodySampleState).ring)
                    for (let i = 0; i < 16; i++) stepPhysics(world, 1 / 960, 8);
            },
        },
        {
            name: "body-sample-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const state = world.resource(BodySampleState);
                if (state.ring && !state.initialized) {
                    const [x, y, z] = state.ringVelocity;
                    setAngularVelocity(world, state.ring, { x, y, z });
                    state.initialized = true;
                }
                if (state.weeble && !state.initialized) {
                    const mass = getMassData(world, state.weeble)!;
                    const shift = f(mass.mass * f(1.5 * 1.5));
                    mass.inertia.cx.x = f(mass.inertia.cx.x + shift);
                    mass.inertia.cz.z = f(mass.inertia.cz.z + shift);
                    mass.center = { x: 0, y: -1.5, z: 0 };
                    setMassData(world, state.weeble, mass);
                    state.initialized = true;
                }
                for (const action of state.actions) {
                    if (action === "Teleport") {
                        setTransform(
                            world,
                            state.weeble,
                            [0, 5, 0],
                            rotationFromAxisAngle(
                                world,
                                { x: 0, y: 0, z: 1 },
                                f(f(0.95) * f(Math.PI)),
                            ),
                        );
                        setAwake(world, state.weeble, true);
                    } else if (action === "Explode") {
                        explode(world, {
                            position: { x: 0, y: -0.1, z: 0 },
                            radius: 8,
                            falloff: 0.1,
                            maskBits: 0xffffffffffffffffn,
                            impulsePerArea: state.magnitude,
                        });
                    }
                }
                state.actions.length = 0;
                if (state.link)
                    applyLinearImpulseToCenter(world, state.link, { x: 0, y: 0.1, z: 0 });
                if (state.platform && state.type === BodyType.Kinematic) {
                    const global = world.storage(GlobalTransform);
                    const x = global.translation.x.get(state.platform);
                    const v = [
                        global.linearVelocity.x.get(state.platform),
                        global.linearVelocity.y.get(state.platform),
                        global.linearVelocity.z.get(state.platform),
                    ];
                    if ((x < -14 && v[0]! < 0) || (x > 6 && v[0]! > 0))
                        setLinearVelocity(world, state.platform, { x: -v[0]!, y: v[1]!, z: v[2]! });
                }
                if (state.spin && !state.initialized) {
                    setAngularVelocity(world, state.spin, { x: 0.01, y: 0.01, z: 10 });
                    state.initialized = true;
                }
                if (state.offset && !state.initialized) {
                    const mass = getMassData(world, state.offset)!;
                    mass.center = { x: 0.5, y: 0.5, z: 1 };
                    setMassData(world, state.offset, mass);
                    setAngularVelocity(world, state.offset, { x: 1, y: -1, z: 2 });
                    state.initialized = true;
                }
                if (!state.target) return;
                if (state.time > 2) {
                    const t = f(state.time - 2);
                    setTargetTransform(
                        world,
                        state.target,
                        [f(4 * f(Math.cos(t))), f(f(2 * f(f(Math.sin(f(2 * t))) + 1)) + 1), 0],
                        rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(2 * t)),
                    );
                }
                state.time = f(state.time + f(1 / 60));
            },
        },
    ],
};
function body(world: World, position: V3, type: number = BodyType.Dynamic): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position });
    world.add(eid, Body, { type });
    return eid;
}
function box(world: World, eid: number, halfExtents: V3, position: V3 = [0, 0, 0]): void {
    const geometry = world
        .resource(Hulls)
        .add({ name: "box", halfExtents, transform: { position, rotation: [0, 0, 0, 1] } });
    const shape = world.has(eid, Shape) ? world.create() : eid;
    world.add(shape, Shape, { geometry, body: shape === eid ? 0 : eid });
}
function ground(world: World, extent = 20): void {
    box(world, body(world, [0, -1, 0], BodyType.Static), [extent, 1, extent]);
}
function spinningBook(world: World): number[] {
    ground(world, 10);
    return [-2, 0, 2].map((x, i) => {
        const eid = body(world, [x, 2, 0]);
        world.storage(Body).gravityScale.set(eid, 0);
        world
            .storage(Body)
            .angularVelocity.set(eid, i === 0 ? 5 : 0.01, i === 1 ? 5 : 0.01, i === 2 ? -5 : 0.01);
        box(world, eid, [0.35, 0.08, 0.5]);
        return eid;
    });
}
function gyroscopicTorque(world: World): number[] {
    ground(world);
    const eid = body(world, [0, 2, 0]);
    world
        .storage(Transform)
        .rotation.set(
            eid,
            ...rotationFromAxisAngle(world, { x: 1, y: 0, z: 0 }, f(-0.5 * f(Math.PI))),
        );
    world.storage(Body).gravityScale.set(eid, 0);
    world.resource(BodySampleState).spin = eid;
    world.add(eid, Shape, {
        geometry: world.resource(Hulls).add({
            name: "cylinder",
            cylinder: { height: 0.6, radius: 0.15, yOffset: 0, sides: 32 },
        }),
    });
    box(world, eid, [1, 0.05, 0.1]);
    return [eid];
}
function kinematic(world: World): number[] {
    ground(world);
    const eid = body(world, [4, 3, 0], BodyType.Kinematic);
    box(world, eid, [0.1, 1, 0.2]);
    world.resource(BodySampleState).target = eid;
    return [eid];
}
function lockMixing(world: World): number[] {
    ground(world);
    return (
        [
            [[0, 2, 0], 0, BodyType.Dynamic],
            [[2, 2, 0], BodyMotionLock.angularX | BodyMotionLock.angularZ, BodyType.Dynamic],
            [
                [-2, 2, 0],
                BodyMotionLock.linearX | BodyMotionLock.linearY | BodyMotionLock.linearZ,
                BodyType.Dynamic,
            ],
            [[0, 1, 2], 63, BodyType.Dynamic],
            [[0, 1, -3], 0, BodyType.Static],
        ] as [V3, number, number][]
    ).map(([position, locks, type]) => {
        const eid = body(world, position, type);
        world.storage(Body).motionLocks.set(eid, locks);
        box(world, eid, [1, 1, 1]);
        return eid;
    });
}
function fixedRotation(world: World): number[] {
    ground(world);
    return [0, 1].map((i) => {
        const eid = body(world, [i ? 0.3 : 0, 0.5, 0], i ? BodyType.Dynamic : BodyType.Static);
        if (i) {
            world.storage(Body).gravityScale.set(eid, 0);
            world.storage(Body).enableSleep.set(eid, 0);
            world
                .storage(Body)
                .motionLocks.set(
                    eid,
                    BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
                );
        }
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, 0, 0],
            capsuleCenter2: [0, 1, 0],
            radius: i ? 0.2 : 0.3,
        });
        return eid;
    });
}
function classRing(world: World): number[] {
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    ground(world, 100);
    const eid = body(world, [0, 1, 0]);
    const rotation = rotationFromAxisAngle(
        world,
        { x: 1, y: 0, z: 0 },
        f(f(13 * f(Math.PI)) / 180),
    );
    world.storage(Transform).rotation.set(eid, ...rotation);
    world.storage(Body).allowFastRotation.set(eid, 1);
    world.storage(Body).enableContactRecycling.set(eid, 0);
    const [cosine, sine] = computeCosSin(world, f(f(2 * f(Math.PI)) / 24));
    const vertices: V3[] = [];
    let x = f(1 - f(0.1)),
        y = 0;
    for (let i = 0; i < 24; i++) {
        vertices.push([x, y, 0]);
        const nextX = f(f(cosine * x) - f(sine * y));
        y = f(f(sine * x) + f(cosine * y));
        x = nextX;
    }
    for (let i = 0; i < 24; i++) {
        const shape = i === 0 ? eid : world.create();
        world.add(shape, Shape, {
            body: i === 0 ? 0 : eid,
            kind: ShapeKind.Capsule,
            capsuleCenter1: vertices[i]!,
            capsuleCenter2: vertices[(i + 1) % 24]!,
            radius: 0.1,
            density: 1,
        });
    }
    world.add(world.create(), Shape, {
        body: eid,
        kind: ShapeKind.Sphere,
        sphereCenter: [0, -0.65, 0],
        radius: 0.3,
        density: 2,
    });
    const tx = f(rotation[0] * 100),
        ty = f(rotation[3] * 100);
    world.resource(BodySampleState).ringVelocity = [
        0,
        f(100 + f(2 * f(-f(rotation[0] * tx)))),
        f(2 * f(rotation[0] * ty)),
    ];
    world.resource(BodySampleState).ring = eid;
    return [eid];
}
function offsetKinematic(world: World): number[] {
    ground(world);
    const eid = body(world, [0, 3, 0], BodyType.Kinematic);
    box(world, eid, [0.5, 0.5, 1], [0.5, 0.5, 1]);
    world.resource(BodySampleState).offset = eid;
    return [eid];
}
function bodyType(world: World): number[] {
    ground(world);
    const groundId = [...world.query([Body])][0]!;
    const attachment = body(world, [-2, 3, 0]);
    box(world, attachment, [0.5, 2, 0.5]);
    world.storage(Shape).density.set(attachment, 1);
    const second = body(world, [3, 3, 0]);
    box(world, second, [0.5, 2, 0.5]);
    world.storage(Shape).density.set(second, 1);
    const platform = body(world, [-4, 5, 0]);
    const shape = world.resource(Hulls).add({
        name: "box",
        halfExtents: [0.5, 4, 0.5],
        transform: {
            position: [4, 0, 0],
            rotation: rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(0.5 * f(Math.PI))),
        },
    });
    world.add(platform, Shape, { geometry: shape, density: 2 });
    for (const [a, localAnchorB] of [
        [attachment, [2, 0, 0]],
        [second, [7, 0, 0]],
    ] as [number, V3][]) {
        world.add(world.create(), RevoluteJoint, {
            a,
            b: platform,
            localAnchorA: [0, 2, 0],
            localAnchorB,
            maxMotorTorque: 50,
            enableMotor: 1,
        });
    }
    world.add(world.create(), PrismaticJoint, {
        a: groundId,
        b: platform,
        localAnchorA: [0, 6, 0],
        localAnchorB: [4, 0, 0],
        maxMotorForce: 1000,
        enableMotor: 1,
        lowerTranslation: -10,
        upperTranslation: 10,
        enableLimit: 1,
    });
    const payload = body(world, [-3, 8, 0]);
    box(world, payload, [0.75, 0.75, 0.75]);
    world.storage(Shape).density.set(payload, 2);
    const secondPayload = body(world, [2, 8, 0]);
    box(world, secondPayload, [0.75, 0.75, 0.75]);
    world.storage(Shape).density.set(secondPayload, 2);
    const debris = body(world, [8, 0.2, 0]);
    world.add(debris, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, 0, 0],
        capsuleCenter2: [1, 0, 0],
        radius: 0.25,
        density: 2,
    });
    const floater = body(world, [-8, 12, 0]);
    world.storage(Body).gravityScale.set(floater, 0);
    world.add(floater, Shape, {
        kind: ShapeKind.Sphere,
        sphereCenter: [0, 0.5, 0],
        radius: 0.25,
        density: 2,
    });
    Object.assign(world.resource(BodySampleState), { platform });
    return [attachment, second, platform, payload, secondPayload, debris, floater];
}
const bodyTypeControls = {
    Type: {
        label: "Type",
        value: BodyType.Dynamic,
        select: true,
        options: [
            { label: "Static", value: BodyType.Static },
            { label: "Kinematic", value: BodyType.Kinematic },
            { label: "Dynamic", value: BodyType.Dynamic },
        ],
        set(world, bodies, value) {
            setBodyType(world, bodies[2]!, value);
            if (value === BodyType.Kinematic) {
                setLinearVelocity(world, bodies[2]!, { x: -3, y: 0, z: 0 });
                setAngularVelocity(world, bodies[2]!, { x: 0, y: 0, z: 0 });
            }
            setBodyType(world, bodies[1]!, value);
            if (value === BodyType.Kinematic) {
                setLinearVelocity(world, bodies[1]!, { x: 0, y: 0, z: 0 });
                setAngularVelocity(world, bodies[1]!, { x: 0, y: 0, z: 0 });
            }
            for (const i of [4, 5, 6]) setBodyType(world, bodies[i]!, value);
            world.resource(BodySampleState).type = value;
        },
    },
    Enable: {
        label: "Enable",
        value: 1,
        set(world, bodies, value) {
            for (const i of [0, 4, 6]) world.storage(Body).isEnabled.set(bodies[i]!, value);
        },
    },
} satisfies Controls;
function weeble(world: World): number[] {
    ground(world, 30);
    const eid = body(world, [0, 3, 0]);
    world.add(eid, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, -1, 0],
        capsuleCenter2: [0, 1, 0],
        radius: 1,
        rollingResistance: 0.1,
    });
    world.resource(BodySampleState).weeble = eid;
    return [eid];
}
const weebleControls = {
    Teleport: {
        label: "Teleport",
        value: 0,
        button: true,
        set(world) {
            world.resource(BodySampleState).actions.push("Teleport");
        },
    },
    Explode: {
        label: "Explode",
        value: 0,
        button: true,
        set(world) {
            world.resource(BodySampleState).actions.push("Explode");
        },
    },
    Magnitude: {
        label: "Magnitude",
        value: 20000,
        min: -100000,
        max: 100000,
        step: 1,
        decimals: 0,
        set(world, _bodies, value) {
            world.resource(BodySampleState).magnitude = value;
        },
    },
} satisfies Controls;
function disable(world: World): number[] {
    ground(world);
    const bodies: number[] = [];
    for (let i = 0; i < 4; i++) {
        const eid = body(
            world,
            [0, f(f(4 - i) * 0.5 + 1), 0],
            i ? BodyType.Dynamic : BodyType.Kinematic,
        );
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, 0, 0],
            capsuleCenter2: [0, -0.5, 0],
            radius: 0.1,
        });
        if (i)
            world.add(world.create(), WeldJoint, {
                a: bodies[i - 1]!,
                b: eid,
                localAnchorA: [0, -0.5, 0],
                angularHertz: 10,
                angularDampingRatio: 1,
            });
        bodies.push(eid);
    }
    const ball = body(world, [3, 3, 0]);
    world.add(ball, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
    bodies.push(ball);
    world.resource(BodySampleState).link = bodies[2]!;
    return bodies;
}
const disableControls: Controls = Object.fromEntries(
    (
        [
            ["Enable Link", 2],
            ["Enable Ball", 4],
        ] as const
    ).map(([label, index]) => [
        label,
        {
            label,
            value: 1,
            set(world: World, bodies: readonly number[], value: number) {
                world.storage(Body).isEnabled.set(bodies[index]!, value);
            },
        },
    ]),
);
export const bodyScenes = {
    cast: castScene,
    classRing: {
        construct: classRing,
        gridExtent: 100,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 6, target: [0, 1, 0] },
        controls: undefined,
    },
    gyroscopicPrecession,
    bodyType: {
        construct: bodyType,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 15, pitch: 20, distance: 24, phoneDistance: 60, target: [0, 5, 0] },
        controls: bodyTypeControls,
    },
    weeble: {
        construct: weeble,
        gridExtent: 30,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 23, target: [0, 2, 0] },
        controls: weebleControls,
    },
    disable: {
        construct: disable,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 20, distance: 8, target: [1.5, 1.5, 0] },
        controls: disableControls,
    },
    spinningBook: {
        construct: spinningBook,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 25, distance: 8, target: [0, 2, 0] },
        controls: undefined,
    },
    gyroscopicTorque: {
        construct: gyroscopicTorque,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 25, distance: 5, target: [0, 2, 0] },
        controls: undefined,
    },
    kinematic: {
        construct: kinematic,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 20, distance: 12, target: [0, 3, 0] },
        controls: undefined,
    },
    lockMixing: {
        construct: lockMixing,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 30, distance: 13, target: [0, 1, 0] },
        controls: undefined,
    },
    fixedRotation: {
        construct: fixedRotation,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 20, distance: 4, target: [0, 1, 0] },
        controls: undefined,
    },
    offsetKinematic: {
        construct: offsetKinematic,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 30, pitch: 25, distance: 7, target: [0.5, 3.5, 1] },
        controls: undefined,
    },
};
