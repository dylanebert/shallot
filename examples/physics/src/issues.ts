import {
    Body,
    BodyMotionLock,
    BodyType,
    GlobalTransform,
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
    setLinearVelocity,
} from "@dylanebert/shallot/standard/physics";
import building from "./building.json" with { type: "json" };
import { Geometry } from "./geometry";
import { ghostFloor } from "./ghost-floor";
import wheel from "./issue-wheel.json" with { type: "json" };
import jitter from "./issues-points.json" with { type: "json" };
import { decodeMesh } from "./mesh-data";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const IssueVelocities = {
    create: () => [] as { body: number; velocity: { x: number; y: number; z: number } }[],
};
export const IssueState = {
    create: () => ({
        ghost: -1,
        bounce: -1,
        directionX: 1,
        directionZ: 1,
        speedX: f(350 * f(0.0254)),
        speedZ: f(20 * f(0.0254)),
        launches: 0,
        maxLaunchSpeed: 0,
        wasLaunched: false,
        markers: [] as V3[],
        bounced: false,
        maxBounce: 0,
        failed: false,
    }),
};
const IssueStatePlugin: Plugin = { name: "IssueState", recovery: recoverResource(IssueState) };
export const IssuesPlugin: Plugin = {
    name: "Issues",
    dependencies: [StandardPhysicsPlugin, IssueStatePlugin],
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
                const s = world.resource(IssueState);
                if (s.ghost < 0) return;
                const t = world.storage(GlobalTransform),
                    e = s.ghost;
                const x = t.translation.x.get(e),
                    z = t.translation.z.get(e);
                if (x > 3.5) s.directionX = -1;
                else if (x < -3.5) s.directionX = 1;
                if (z > 0.5) s.directionZ = -1;
                else if (z < -0.5) s.directionZ = 1;
                setLinearVelocity(world, e, {
                    x: f(s.directionX * s.speedX),
                    y: t.linearVelocity.y.get(e),
                    z: f(s.directionZ * s.speedZ),
                });
            },
        },
        {
            name: "issues-after-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(IssueState),
                    t = world.storage(GlobalTransform);
                if (s.ghost >= 0) {
                    const e = s.ghost,
                        y = t.translation.y.get(e),
                        vy = t.linearVelocity.y.get(e);
                    const grounded = y < f(f(f(36 * f(0.0254)) + f(0.01)) + f(4 * f(0.0254))),
                        launched = vy > 0.5;
                    if (grounded && launched && !s.wasLaunched) {
                        s.launches++;
                        s.maxLaunchSpeed = Math.max(s.maxLaunchSpeed, vy);
                        if (s.markers.length < 64)
                            s.markers.push([t.translation.x.get(e), y, t.translation.z.get(e)]);
                    }
                    s.wasLaunched = launched;
                }
                if (s.bounce >= 0) {
                    const y = t.translation.y.get(s.bounce),
                        vy = t.linearVelocity.y.get(s.bounce);
                    if (!s.bounced && vy > 0) s.bounced = true;
                    if (s.bounced) {
                        s.maxBounce = Math.max(s.maxBounce, y);
                        if (y > f(10 + f(0.05))) s.failed = true;
                    }
                }
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
function ghostCollisions(world: World): number[] {
    for (let chunk = 0; chunk < 2; chunk++) {
        const e = body(world, [0, 0, 0]);
        world.add(e, Shape, {
            kind: ShapeKind.Mesh,
            geometry: world.resource(PhysicsMeshes).add({ name: "chunk", ...ghostFloor(chunk) }),
        });
    }
    const half = f(16 * f(0.0254)),
        height = f(36 * f(0.0254));
    const e = body(world, [-3.5, f(height + f(0.1)), 0], true);
    const b = world.storage(Body);
    b.motionLocks.set(
        e,
        BodyMotionLock.angularX | BodyMotionLock.angularY | BodyMotionLock.angularZ,
    );
    b.enableSleep.set(e, 0);
    b.enableContactRecycling.set(e, 0);
    b.gravityScale.set(e, 2.03);
    world.add(e, Shape, {
        geometry: world.resource(Hulls).add({ name: "box", halfExtents: [half, height, half] }),
        friction: 0,
        restitution: 0,
        density: f(500 / f(f(f(8 * half) * height) * half)),
        enableSpeculativeContact: 0,
    });
    world.resource(IssueState).ghost = e;
    return [e];
}
function wheelStack(world: World): number[] {
    ground(world, 10);
    const points: number[] = [];
    for (const [offset, count] of wheel.spans) {
        const hull = createHull(
            wheel.points
                .slice(offset!, offset! + count!)
                .map(([x, y, z]) => ({ x: f(x!), y: f(y!), z: f(z!) })),
            count!,
            world,
        );
        if (hull)
            for (const p of hull.points.slice(0, hull.vertexCount))
                if (points.length < 512 * 3) points.push(p.x, p.y, p.z);
    }
    const geometry = world.resource(Hulls).add({
        name: "wheel",
        points: new Float32Array(points),
        maxVertexCount: points.length / 3,
    });
    const result: number[] = [];
    const height = f(0.171),
        spacing = f(height + f(0.006)),
        start = f(f(0.5 * height) + f(0.004));
    for (let i = 0; i < 30; i++) {
        const e = body(world, [0, f(start + f(i * spacing)), 0], true);
        world.add(e, Shape, { geometry, friction: 0.6 });
        result.push(e);
    }
    Object.assign(world.resource(PhysicsWorldDefinition), {
        contactHertz: 240,
        contactDampingRatio: 10,
        contactSpeed: 3,
    });
    return result;
}
function restitutionOvershoot(world: World): number[] {
    box(world, body(world, [0, -0.25, 0]), [0.375, 0.25, 0.375]);
    const e = body(world, [0, 10, 0], true);
    box(world, e, [0.5, 0.5, 0.5]);
    world.storage(Shape).restitution.set(e, 1);
    world.resource(IssueState).bounce = e;
    return [e];
}
export const issueScenes = {
    ghostCollisions: {
        construct: ghostCollisions,
        gridExtent: 6.5024,
        gridExtentZ: 1.6256,
        gridSpacing: 0.8128,
        gridCenter: [0, 0],
        camera: {
            yaw: 20,
            pitch: 30,
            distance: 15,
            phoneDistance: 28,
            target: [0, 0.5, 0],
            phoneTarget: [0, 5.5, 0],
        },
        controls: {
            "Walk Speed X (inch/s)": {
                label: "Walk speed X, in/s",
                value: 350,
                min: 100,
                max: 400,
                step: 1,
                decimals: 0,
                set(world, _b, value) {
                    world.resource(IssueState).speedX = f(f(value) * f(0.0254));
                },
            },
            "Walk Speed Z (inch/s)": {
                label: "Walk speed Z, in/s",
                value: 20,
                min: 10,
                max: 100,
                step: 1,
                decimals: 0,
                set(world, _b, value) {
                    world.resource(IssueState).speedZ = f(f(value) * f(0.0254));
                },
            },
            "Reset Counters": {
                label: "Reset Counters",
                value: 0,
                button: true,
                set(world) {
                    const s = world.resource(IssueState);
                    s.launches = 0;
                    s.maxLaunchSpeed = 0;
                    s.markers = [];
                },
            },
            Launches: {
                label: "Launches",
                value: 0,
                decimals: 0,
                read(world) {
                    return world.resource(IssueState).launches;
                },
                set() {},
            },
            Worst: {
                label: "Worst, m/s",
                value: 0,
                decimals: 2,
                read(world) {
                    return world.resource(IssueState).maxLaunchSpeed;
                },
                set() {},
            },
        } satisfies Controls,
    },
    wheelStack: {
        construct: wheelStack,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 15, distance: 9, phoneDistance: 12, target: [0, 2.6, 0] },
        controls: undefined,
    },
    restitutionOvershoot: {
        construct: restitutionOvershoot,
        gridExtent: 0.375,
        gridSpacing: 0.75,
        camera: {
            yaw: 20,
            pitch: 10,
            distance: 28,
            phoneDistance: 28,
            target: [0, 5, 0],
            phoneTarget: [0, 9, 0],
        },
        controls: {
            Drop: {
                label: "Drop height, m",
                value: 10,
                decimals: 2,
                read() {
                    return 10;
                },
                set() {},
            },
            Maximum: {
                label: "Max bounce, m",
                value: 0,
                decimals: 2,
                read(world) {
                    return world.resource(IssueState).maxBounce;
                },
                set() {},
            },
            Result: {
                label: "At or below drop",
                value: 0,
                read(world) {
                    return world.resource(IssueState).failed ? "No" : "Yes";
                },
                set() {},
            },
        } satisfies Controls,
    },

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
