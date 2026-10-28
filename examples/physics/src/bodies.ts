import {
    Body,
    BodyMotionLock,
    BodyType,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    getMassData,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
    setMassData,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";

type V3 = [number, number, number];
const f = Math.fround;
export const BodySampleState = {
    create: () => ({ target: 0, offset: 0, spin: 0, time: 0, initialized: false }),
};
export const BodySamplePlugin: Plugin = {
    name: "BodySample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(BodySampleState),
    systems: [
        {
            name: "body-sample-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const state = world.resource(BodySampleState);
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
function offsetKinematic(world: World): number[] {
    ground(world);
    const eid = body(world, [0, 3, 0], BodyType.Kinematic);
    box(world, eid, [0.5, 0.5, 1], [0.5, 0.5, 1]);
    world.resource(BodySampleState).offset = eid;
    return [eid];
}
export const bodyScenes = {
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
