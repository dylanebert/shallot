import {
    Body,
    BodyMotionLock,
    BodyType,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    PrismaticJoint,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import {
    createHull,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
} from "@dylanebert/shallot/standard/physics";
import building from "./building.json" with { type: "json" };
import { Geometry } from "./geometry";
import jitter from "./issues-points.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const IssueVelocities = {
    create: () => [] as { body: number; velocity: { x: number; y: number; z: number } }[],
};
export const IssuesPlugin: Plugin = {
    name: "Issues",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(IssueVelocities),
    systems: [
        {
            name: "issues-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const pending = world.resource(IssueVelocities);
                for (const { body, velocity } of pending) setAngularVelocity(world, body, velocity);
                pending.length = 0;
            },
        },
    ],
};
function body(world: World, translation: V3, dynamic = false): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function box(world: World, eid: number, halfExtents: V3): void {
    world.add(eid, Shape, { geometry: world.resource(Hulls).add({ name: "box", halfExtents }) });
}
function ground(world: World, extent: number): void {
    box(world, body(world, [0, -1, 0]), [extent, 1, extent]);
}
function crash(world: World): number[] {
    const g = body(world, [0, -1, 0]);
    world.add(g, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "grid",
            grid: {
                xCount: 20,
                zCount: 20,
                cellWidth: 2,
                materialCount: 0,
                identifyEdges: true,
            },
        }),
    });
    const bodies = [body(world, [2, 4, 0], true), body(world, [-2, 4, 0], true)];
    for (const eid of bodies) box(world, eid, [0.5, 0.5, 0.5]);
    return bodies;
}
function multiplePrismatic(world: World): number[] {
    let a = body(world, [0, 0, 0]);
    const bodies: number[] = [];
    for (let i = 0; i < 6; i++) {
        const b = body(world, [0, f(f(0.6) + f(f(1.2) * i)), 0], true);
        box(world, b, [0.5, 0.5, 0.5]);
        world.add(world.create(), PrismaticJoint, {
            a,
            b,
            localAnchorA: [0, i === 0 ? 0 : 0.6, 0],
            localAnchorB: [0, -0.6, 0],
            drawScale: 2,
            constraintHertz: 240,
            lowerTranslation: -6,
            upperTranslation: 6,
            enableLimit: 1,
        });
        a = b;
        bodies.push(b);
    }
    return bodies;
}
export function hullCrashPoints() {
    return [
        [100, -142.292389, 130.826111],
        [99.5354385, -71.3011093, 130.826111],
        [99.5930862, -80.1112213, -100],
        [100, -142.292389, -100],
        [99.5930862, -80.1112213, 130.826111],
    ].map(([x, y, z]) => ({ x: f(f(0.01) * f(x!)), y: f(f(0.01) * f(y!)), z: f(f(0.01) * f(z!)) }));
}
function hullCrash(world: World): number[] {
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    world.resource(Geometry).hulls = [createHull(hullCrashPoints(), 5, world)];
    return [];
}
function convexJitter(world: World): number[] {
    ground(world, 10);
    const bodies: number[] = [];
    for (const [i, points] of jitter.entries()) {
        const b =
            i === 0 ? [-459.292877, 217.398331, 1.00115335] : [-402.321838, 157.310364, 16.816925];
        const eid = body(
            world,
            [f(f(0.01) * f(b[0]!)), f(f(f(0.01) * f(b[2]!)) + 2), f(f(0.01) * f(b[1]!))],
            i === 1,
        );
        world
            .storage(Transform)
            .rotation.set(
                eid,
                0,
                i === 0 ? -0.707106769 : -0.00152086187,
                0,
                i === 0 ? 0.707106769 : 0.999998868,
            );
        world.add(eid, Shape, {
            geometry: world.resource(Hulls).add({
                name: "convex",
                points: new Float32Array(
                    points.flatMap(([x, y, z]) => [
                        f(f(0.01) * f(x!)),
                        f(f(0.01) * f(z!)),
                        f(f(0.01) * f(y!)),
                    ]),
                ),
                maxVertexCount: points.length,
            }),
            rollingResistance: i === 1 ? 0.1 : 0,
        });
        if (i === 1) bodies.push(eid);
    }
    return bodies;
}
function sboxMover(world: World): number[] {
    const g = body(world, [-10, 0, -10]);
    world.add(g, Shape, {
        kind: ShapeKind.HeightField,
        geometry: world.resource(HeightFields).add({
            name: "grid",
            heights: new Float32Array(40 * 40),
            countX: 40,
            countZ: 40,
            scale: [0.5, 1, 0.5],
            globalMinimumHeight: -256,
            globalMaximumHeight: 256,
        }),
    });
    const platform = body(world, [0, 0, 0]);
    world.add(platform, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "platform",
            vertices: new Float32Array([
                1, 1, 1, -1, 1, 1, -2.5, 0, 2.5, 2.5, 0, 2.5, 1, 1, -1, -1, 1, -1, -2.5, 0, -2.5,
                2.5, 0, -2.5,
            ]),
            indices: new Uint32Array([
                0, 1, 3, 1, 2, 3, 0, 4, 1, 1, 4, 5, 0, 3, 7, 4, 0, 7, 4, 7, 5, 6, 5, 7, 1, 5, 2, 6,
                2, 5, 3, 2, 7, 6, 7, 2,
            ]),
            useMedianSplit: true,
            identifyEdges: true,
        }),
    });
    const eid = body(world, [0, 3.5, 0], true);
    world
        .storage(Body)
        .motionLocks.set(
            eid,
            BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
        );
    world.storage(Body).enableContactRecycling.set(eid, 0);
    box(world, eid, [0.25, 1, 0.25]);
    return [eid];
}
function capsuleMesh(world: World): number[] {
    box(world, body(world, [0, 0, 0]), [50, 0.1, 50]);
    const mesh = body(world, [0, 0.1, 0]);
    world.add(mesh, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "building",
            ...decodeMesh(building),
            identifyEdges: true,
            useMedianSplit: true,
        }),
    });
    const eid = body(world, [0, 4, 10], true);
    const bodies = world.storage(Body);
    bodies.motionLocks.set(
        eid,
        BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
    );
    bodies.enableSleep.set(eid, 0);
    bodies.enableContactRecycling.set(eid, 0);
    world.add(eid, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, -0.5, 0],
        capsuleCenter2: [0, 0.5, 0],
        radius: 0.3,
        friction: 0.3,
    });
    return [eid];
}
function slideTwistOffCenterShape(world: World): number[] {
    ground(world, 50);
    const angle = f(20 * f(f(Math.PI) / 180));
    const q = rotationFromAxisAngle(world, { x: 1, y: 0, z: 0 }, angle);
    const c = f(f(1 - f(2 * f(q[0] * q[0])))),
        s = f(2 * f(q[3] * q[0]));
    const plane = body(world, [0, 4, 0]);
    world.storage(Transform).rotation.set(plane, ...q);
    box(world, plane, [10, 0.5, 10]);
    world.storage(Shape).friction.set(plane, 0.6);
    const offsetY = f(0.5 + f(2 * f(-q[0] * f(f(q[0] * 0.5) + q[3]))));
    const offsetZ = f(1 + f(2 * f(q[0] * f(-q[0] + f(q[3] * 0.5)))));
    const eid = body(world, [-1, f(5 - offsetY), -offsetZ], true);
    world.storage(Transform).rotation.set(eid, ...q);
    world.add(eid, Shape, {
        geometry: world.resource(Hulls).add({
            name: "offset box",
            halfExtents: [1, 0.5, 1],
            transform: { position: [1, 0.5, 1], rotation: [0, 0, 0, 1] },
        }),
        friction: 0.3,
    });
    world
        .resource(IssueVelocities)
        .push({ body: eid, velocity: { x: 0, y: f(25 * c), z: f(25 * s) } });
    return [eid];
}
function hugeBox(world: World): number[] {
    ground(world, 400);
    const eid = body(world, [0, 100, 0], true);
    world
        .storage(Transform)
        .rotation.set(eid, ...rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(0.2)));
    box(world, eid, [100, 100, 100]);
    return [eid];
}
function issueHeightfield(world: World): number[] {
    const g = body(world, [0, 0, 0]);
    world.add(g, Shape, {
        kind: ShapeKind.HeightField,
        geometry: world.resource(HeightFields).add({
            name: "grid",
            heights: new Float32Array(100 * 100),
            countX: 100,
            countZ: 100,
            scale: [1, 1, 1],
            globalMinimumHeight: -256,
            globalMaximumHeight: 256,
        }),
    });
    const eid = body(world, [50, 20, 50], true);
    world
        .storage(Transform)
        .rotation.set(
            eid,
            ...rotationFromAxisAngle(world, { x: 1, y: 0, z: 0 }, f(f(Math.PI) / 4)),
        );
    world.add(eid, Shape, {
        geometry: world.resource(Hulls).add({
            name: "cylinder",
            cylinder: { height: 20, radius: 5, yOffset: -10, sides: 24 },
        }),
    });
    return [eid];
}
export const issueScenes = {
    crash: {
        construct: crash,
        gridHeight: -1,
        gridExtent: 20,
        gridSpacing: 2,
        camera: { yaw: 25, pitch: 25, distance: 12, target: [0, 2, 0] },
        controls: {
            "Add Joint": {
                label: "Add Joint",
                value: 0,
                button: true,
                set(world, bodies) {
                    world.add(world.create(), WeldJoint, { a: bodies[0]!, b: bodies[1]! });
                },
            },
        } satisfies Controls,
    },
    multiplePrismatic: {
        construct: multiplePrismatic,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 20, distance: 15, target: [0, 3.6, 0] },
        controls: undefined,
    },
    hullCrash: {
        construct: hullCrash,
        gridExtent: 4,
        gridSpacing: 0.5,
        camera: { yaw: 65, pitch: 20, distance: 4, target: [1, -1, 0.15] },
        controls: undefined,
    },
    convexJitter: {
        construct: convexJitter,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 20, distance: 5, target: [-4.3, 2, 1.8] },
        controls: undefined,
    },
    sboxMover: {
        construct: sboxMover,
        gridCenter: [-0.25, -0.25],
        gridExtent: 9.75,
        gridSpacing: 0.5,
        camera: { yaw: 35, pitch: 25, distance: 10, target: [0, 1.5, 0] },
        controls: undefined,
    },
    capsuleMesh: {
        construct: capsuleMesh,
        gridHeight: 0.1,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 20, pitch: 20, distance: 28, target: [0, 2, 4] },
        controls: undefined,
    },
    slideTwistOffCenterShape: {
        construct: slideTwistOffCenterShape,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 30, pitch: 25, distance: 32, target: [0, 4, 0] },
        controls: undefined,
    },
    hugeBox: {
        construct: hugeBox,
        gridExtent: 400,
        gridSpacing: 20,
        camera: { yaw: 30, pitch: 20, distance: 550, phoneDistance: 850, target: [0, 100, 0] },
        controls: undefined,
    },
    issueHeightfield: {
        construct: issueHeightfield,
        gridCenter: [49.5, 49.5],
        gridExtent: 49.5,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 25, distance: 55, phoneDistance: 80, target: [50, 10, 50] },
        controls: undefined,
    },
};
