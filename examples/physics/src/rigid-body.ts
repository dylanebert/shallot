import {
    Body,
    BodyMotionLock,
    BodyType,
    Camera,
    Devices,
    GlobalTransform,
    HeightFields,
    Hulls,
    InputPlugin,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "@dylanebert/shallot/standard/physics";
import building from "./building.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";
import { pose, rotate, v } from "./mover";
import {
    BODY_RADIUS,
    FEET_HEIGHT,
    lateRigid,
    prepareRigid,
    rigidCharacter,
    stepRigid,
    TOTAL_HEIGHT,
} from "./rigid-character";
import stairs from "./stairs.json" with { type: "json" };
import level from "./test_map01.json" with { type: "json" };
import voxel1 from "./voxel_mesh_01.json" with { type: "json" };
import voxel2 from "./voxel_mesh_02.json" with { type: "json" };

const f = Math.fround;
export const RigidBodyState = {
    create: () => ({
        active: false,
        character: rigidCharacter(),
        keys: { W: 0, A: 0, S: 0, D: 0, Space: 0, Shift: 0 },
        forward: v(0, 0, -1),
        right: v(1, 0, 0),
    }),
};
function body(
    world: World,
    position: [number, number, number],
    type: number = BodyType.Static,
    rotation: [number, number, number, number] = [0, 0, 0, 1],
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, rotation });
    world.add(eid, Body, { type });
    return eid;
}
function box(
    world: World,
    owner: number,
    halfExtents: [number, number, number],
    friction: number,
    density = 1000,
    position: [number, number, number] = [0, 0, 0],
) {
    const shape = world.create();
    world.add(shape, Shape, {
        body: owner,
        geometry: world
            .resource(Hulls)
            .add({ name: "box", halfExtents, transform: { position, rotation: [0, 0, 0, 1] } }),
        friction,
        density,
    });
    return shape;
}
function construct(world: World): number[] {
    const state = world.resource(RigidBodyState);
    Object.assign(state, RigidBodyState.create());
    state.active = true;
    const s = state.character;
    const character = body(world, [7.5, 2, 9], BodyType.Dynamic);
    s.body = character;
    const bodies = [character];
    const b = world.storage(Body);
    b.motionLocks.set(
        character,
        BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
    );
    b.enableSleep.set(character, 0);
    b.enableContactRecycling.set(character, 0);
    b.gravityScale.set(character, 1.5);
    const halfW = f(BODY_RADIUS * 0.5),
        halfY = f(FEET_HEIGHT * 0.5);
    const volume = f(f(f(8 * halfW) * halfY) * halfW);
    s.feet = box(world, character, [halfW, halfY, halfW], 0, f(f(500 * f(0.4)) / volume), [
        0,
        f(f(-TOTAL_HEIGHT * 0.5) + halfY),
        0,
    ]);
    // biome-ignore lint/suspicious/noApproximativeNumericConstant: samples/sample_character.cpp specifies 0.707f.
    const radius = f(BODY_RADIUS * f(0.707));
    const bottom = f(f(f(-TOTAL_HEIGHT * 0.5) + f(FEET_HEIGHT * 0.5)) + radius),
        top = f(f(TOTAL_HEIGHT * 0.5) - radius);
    const h = f(top - bottom);
    const capsuleVolume = f(f(f(f(Math.PI) * radius) * radius) * f(h + f(f(4 * radius) / 3)));
    s.capsule = world.create();
    world.add(s.capsule, Shape, {
        body: character,
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, bottom, 0],
        capsuleCenter2: [0, top, 0],
        radius,
        friction: 0,
        restitution: 0,
        density: f(f(500 * f(0.6)) / capsuleVolume),
    });
    const meshes: [string, typeof level, [number, number, number], [number, number, number]][] = [
        ["test map", level, [0, 0, 0], [1, 1, 1]],
        ["stairs", stairs, [-10, 0, 0], [0.75, 0.75, -1.5]],
        ["building", building, [-5, 0, -10], [1, 1, 1]],
        ["voxel 1", voxel1, [10, 0, -10], [1, 1, 1]],
        ["voxel 2", voxel2, [10, 0, 10], [1, 1, 1]],
    ];
    for (const [name, data, position, scale] of meshes) {
        const owner = body(world, position);
        bodies.push(owner);
        world.add(world.create(), Shape, {
            body: owner,
            kind: ShapeKind.Mesh,
            scale,
            geometry: world.resource(PhysicsMeshes).add({
                name,
                ...decodeMesh(data),
                identifyEdges: true,
                weldVertices: true,
                weldTolerance: 0.002,
            }),
        });
    }
    const terrain = body(world, [20, 0, 0]);
    bodies.push(terrain);
    world.add(world.create(), Shape, {
        body: terrain,
        kind: ShapeKind.HeightField,
        geometry: world.resource(HeightFields).add({
            name: "wave",
            wave: {
                rowCount: 50,
                columnCount: 50,
                scale: [1, 1, 1],
                rowFrequency: f(0.02),
                columnFrequency: f(0.04),
                makeHoles: true,
            },
        }),
    });
    for (const [x, y, z, angle, width] of [
        [6, 1, 4, -20, 3],
        [6, 2, -4, -50, 2.5],
    ]) {
        const owner = body(
            world,
            [x!, y!, z!],
            BodyType.Static,
            rotationFromAxisAngle(world, v(0, 0, 1), f(angle! * f(Math.PI / 180))),
        );
        bodies.push(owner);
        box(world, owner, [width!, 0.15, 1.5], 0.6);
    }
    for (let i = 0; i < 3; i++) {
        const owner = body(world, [f(-4 + f(3.5 * i)), 1.2, -5]);
        bodies.push(owner);
        box(world, owner, [1.2, 0.15, 1.2], 0.6);
    }
    for (let i = 0; i < 5; i++) {
        const lip = f(f(0.05) + f(f(0.08) * i));
        const owner = body(world, [-8, lip, f(-1 + 2 * i)]);
        bodies.push(owner);
        box(world, owner, [1, lip, 0.6], 0.6);
    }
    const wall = body(world, [0, 1.5, 10]);
    bodies.push(wall);
    box(world, wall, [4, 1.5, 0.2], 0.6);
    for (let i = 0; i < 3; i++) {
        const owner = body(world, [f(3 + f(1.5 * i)), 0.5, 0], BodyType.Dynamic);
        bodies.push(owner);
        box(world, owner, [0.4, 0.4, 0.4], 0.6);
    }
    const sphere = body(world, [-3, 1, 0], BodyType.Dynamic);
    bodies.push(sphere);
    world.add(world.create(), Shape, {
        body: sphere,
        kind: ShapeKind.Sphere,
        radius: 0.5,
        friction: 0.6,
    });
    return bodies;
}
export const RigidBodyPlugin: Plugin = {
    name: "RigidBodySample",
    dependencies: [StandardPhysicsPlugin, InputPlugin],
    recovery: recoverResource(RigidBodyState),
    systems: [
        {
            name: "rigid-body-input",
            group: "setup",
            update(world) {
                const s = world.resource(RigidBodyState);
                if (!s.active) return;
                const devices = world.resource(Devices);
                for (const key of ["W", "A", "S", "D", "Space", "Shift"] as const)
                    s.keys[key] = Number(
                        devices.enabled &&
                            devices.keys.held.has(
                                key === "Shift"
                                    ? "ShiftLeft"
                                    : key === "Space"
                                      ? "Space"
                                      : `Key${key}`,
                            ),
                    );
                const [camera] = world.query([Camera, GlobalTransform]);
                if (!camera) return;
                const q = pose(world, camera).q;
                s.forward = rotate(q, v(0, 0, -1));
                s.right = rotate(q, v(1, 0, 0));
            },
        },
        {
            name: "rigid-body-prepare",
            group: "fixed",
            before: StandardPhysicsPlugin.systems!.filter((system) => system.group === "fixed"),
            update(world) {
                const s = world.resource(RigidBodyState);
                if (s.active) prepareRigid(world, s.character, s.forward, s.right, s.keys);
            },
        },
        {
            name: "rigid-body-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(RigidBodyState);
                if (s.active) stepRigid(world, s.character);
            },
        },
        {
            name: "rigid-body-after-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(RigidBodyState);
                if (s.active) lateRigid(world, s.character);
            },
        },
    ],
};
export const rigidBodyScene = {
    construct,
    gridExtent: 3300,
    gridSpacing: 1,
    controls: undefined,
    camera: { target: [7.5, 2, 9] as [number, number, number], distance: 6, yaw: 0, pitch: 25 },
};
