import {
    Body,
    BodyType,
    GlobalTransform,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    computeCosSin,
    isAwake,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    stepPhysics,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const ContinuousState = {
    create: () => ({
        bullet: -1,
        meshDrop: false,
        ground: -1,
        walls: [] as number[],
        bodies: [] as number[],
        dropBox: 0,
        cylinder: 0,
        type: 0,
        amplitude: 0.5,
        collide: true,
        auto: false,
        failure: false,
        paused: false,
        steps: 0,
        moving: false,
        seed: null as number | null,
    }),
};
export const ContinuousStepObserver = {
    create: () => ({ observe: undefined as ((world: World) => void) | undefined }),
};
function meshSeed(world: World): number {
    const state = world.resource(ContinuousState);
    return state.seed === null ? Math.floor(world.time.realElapsed * 1e6) : state.seed++;
}
export const ContinuousSamplePlugin: Plugin = {
    name: "ContinuousSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(ContinuousState),
    systems: [
        {
            name: "mesh-drop-before-step",
            group: "fixed",
            before: [StepPhysicsSystem],
            update(world) {
                const state = world.resource(ContinuousState);
                // Native move events include a body on the step in which it falls asleep.
                if (state.meshDrop) state.moving = state.bodies.some((eid) => isAwake(world, eid));
            },
        },
        {
            name: "mesh-drop-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const state = world.resource(ContinuousState);
                if (!state.meshDrop || state.paused) return;
                world.resource(ContinuousStepObserver).observe?.(world);
                state.steps++;
                const transform = world.storage(GlobalTransform);
                for (const eid of state.bodies) {
                    if (state.failure) break;
                    // Only the cylinder has an offset mass center, along its local Y axis.
                    const center = state.type === 2 ? 0.2 : 0;
                    const x = transform.rotation.x.get(eid),
                        z = transform.rotation.z.get(eid);
                    const y = transform.translation.y.get(eid) + center * (1 - 2 * (x * x + z * z));
                    if (y < -2) {
                        state.paused = state.failure = true;
                        state.auto = false;
                    }
                }
                if (state.auto) {
                    if (!state.moving && !state.bodies.some((eid) => isAwake(world, eid)))
                        generate(world);
                    for (let i = 0; i < 20; i++) {
                        stepPhysics(world, 1 / 60, 4);
                        world.resource(ContinuousStepObserver).observe?.(world);
                    }
                    state.steps += 20;
                    if (state.steps > (state.type === 1 ? 3000 : 1000)) {
                        state.paused = state.failure = true;
                        state.auto = false;
                    }
                }
                if (state.paused)
                    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
            },
        },
    ],
};
function box(world: World, halfExtents: V3): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(world: World, translation: V3, dynamic = true): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function ground(world: World, extent: number): void {
    const eid = body(world, [0, -1, 0], false);
    world.add(eid, Shape, { geometry: box(world, [extent, 1, extent]) });
}
function thinWall(world: World): number[] {
    ground(world, 40);
    const wall = body(world, [0, 10, 0], false);
    world
        .storage(Transform)
        .rotation.set(
            wall,
            ...rotationFromAxisAngle(world, { x: 1, y: 0, z: 0 }, f(90 * f(f(Math.PI) / 180))),
        );
    world.add(wall, Shape, { geometry: box(world, [10, 0.1, 10]) });
    return [-5, 0, 5].map((x, index) => {
        const eid = body(world, [x, 10, 20]);
        world.storage(Body).linearVelocity.set(eid, 0, 0, -180);
        world.storage(Body).angularVelocity.set(eid, 20, index === 0 ? 0 : index === 1 ? -5 : 5, 0);
        world.add(eid, Shape, {
            ...(index === 0
                ? { kind: ShapeKind.Sphere, radius: 0.1 }
                : index === 1
                  ? {
                        kind: ShapeKind.Capsule,
                        capsuleCenter1: [-0.3, 0, 0],
                        capsuleCenter2: [0.3, 0, 0],
                        radius: 0.1,
                    }
                  : { geometry: box(world, [0.4, 0.1, 0.1]) }),
            rollingResistance: 0.1,
        });
        return eid;
    });
}
function bounceHouse(world: World): number[] {
    ground(world, 10);
    const base = body(world, [0, -1, 0], false);
    for (const [position, halfExtents] of [
        [
            [10, 5, 0],
            [0.1, 5, 10],
        ],
        [
            [-10, 5, 0],
            [0.1, 5, 10],
        ],
        [
            [0, 5, -10],
            [10, 5, 0.1],
        ],
        [
            [0, 5, 10],
            [10, 5, 0.1],
        ],
    ] as [V3, V3][]) {
        const shape = world.create();
        world.add(shape, Shape, {
            body: base,
            geometry: world.resource(Hulls).add({
                name: "wall",
                halfExtents,
                transform: { position, rotation: [0, 0, 0, 1] },
            }),
        });
    }
    const eid = body(world, [-8, 4, 0]);
    world.storage(Body).gravityScale.set(eid, 0);
    world.storage(Body).linearVelocity.set(eid, 120, 0, 120);
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, friction: 0, restitution: 1 });
    return [eid];
}
function spinningStick(world: World): number[] {
    ground(world, 10);
    const wall = body(world, [0, 0.5, 0], false);
    world.add(wall, Shape, { geometry: box(world, [0.125, 0.5, 10]) });
    let seed = 12345;
    const random = () => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(100 * f((seed % 32768) / 32767)) - 50);
    };
    const eid = body(world, [0, 20, 0.5]);
    world.storage(Body).linearVelocity.set(eid, 0, -100, 0);
    world.storage(Body).angularVelocity.set(eid, random(), random(), random());
    world.add(eid, Shape, { geometry: box(world, [2, 0.1, 0.1]), rollingResistance: 0.1 });
    return [eid];
}
function needleMesh(world: World): number[] {
    const base = body(world, [0, 0, 0], false);
    for (const [height, x, z] of [
        [0.99, 0.2, 0.2],
        [1.01, 0.2, -0.2],
        [0.98, -0.2, -0.2],
        [1.02, -0.2, 0.2],
    ]) {
        const vertices = [f(x!), f(height!), f(z!)];
        let angle = 0;
        const delta = f(f(2 * f(Math.PI)) / 8);
        for (let i = 0; i < 8; i++) {
            const [c, s] = computeCosSin(world, angle);
            vertices.push(f(f(f(0.1) * c) + f(x!)), 0, f(f(f(0.1) * s) + f(z!)));
            angle = f(angle + delta);
        }
        const indices: number[] = [];
        let previous = 8;
        for (let i = 1; i <= 8; i++) {
            indices.push(0, i, previous);
            previous = i;
        }
        const shape = world.create();
        world.add(shape, Shape, {
            body: base,
            kind: ShapeKind.Mesh,
            geometry: world.resource(PhysicsMeshes).add({
                name: "needle",
                vertices: new Float32Array(vertices),
                indices: new Uint32Array(indices),
                useMedianSplit: true,
            }),
        });
    }
    const eid = body(world, [0, 5, 0]);
    world.storage(Body).linearVelocity.set(eid, 0, -10, 0);
    world.add(eid, Shape, { geometry: box(world, [0.3, 0.01, 0.3]) });
    return [eid];
}
function humpMesh(world: World): number[] {
    ground(world, 20);
    const base = body(world, [0, 0, 0], false);
    const vertices: number[] = [];
    for (const x of [-4, 4])
        for (const z of [-8, 0, 8]) vertices.push(x, z === 0 ? f(f(0.05) * 8) : 0, z);
    world.add(base, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "hump",
            vertices: new Float32Array(vertices),
            indices: new Uint32Array([0, 1, 4, 4, 3, 0, 1, 2, 5, 5, 4, 1]),
            useMedianSplit: true,
            identifyEdges: true,
        }),
    });
    const eid = body(world, [0, 5, 0]);
    world.storage(Body).linearVelocity.set(eid, 0, -50, 0);
    world.add(eid, Shape, { geometry: box(world, [0.5, 0.05, 1]) });
    return [eid];
}
function isFast(world: World): number[] {
    ground(world, 40);
    const geometry = box(world, [0.5, 10, 0.5]);
    return [-12, 0, 12].map((x, index) => {
        const eid = body(world, [x, 20, 0]);
        world.storage(Body).gravityScale.set(eid, 0);
        world
            .storage(Body)
            .angularVelocity.set(
                eid,
                index === 2 ? 4 : 0,
                index === 1 ? 4 : 0,
                index === 0 ? 4 : 0,
            );
        world.add(eid, Shape, { geometry });
        return eid;
    });
}
function launch(world: World): number {
    const state = world.resource(ContinuousState);
    if (state.bullet !== -1) world.destroy(state.bullet);
    const phi = f(f(1 + f(Math.sqrt(5))) / 2);
    const [dc, ds] = computeCosSin(world, f(f(2 * f(Math.PI)) / phi));
    let c = 1,
        s = 0;
    const points: number[] = [];
    for (let i = 0; i < 10; i++) {
        const z = f(1 - f((2 * i + 1) / 10));
        const r = f(Math.sqrt(f(1 - f(z * z))));
        points.push(f(f(0.25 * r) * c), f(f(0.25 * r) * s), f(0.25 * z));
        const previous = c;
        c = f(f(dc * c) - f(ds * s));
        s = f(f(ds * previous) + f(dc * s));
    }
    const eid = body(world, [0, 1, -10]);
    world.storage(Body).isBullet.set(eid, 1);
    world.storage(Body).linearVelocity.set(eid, 0, 0, 600);
    world.storage(Body).angularVelocity.set(eid, 0, 0, 20);
    world.add(eid, Shape, {
        geometry: world
            .resource(Hulls)
            .add({ name: "rock", points: new Float32Array(points), maxVertexCount: 10 }),
    });
    state.bullet = eid;
    return eid;
}
function stall(world: World): number[] {
    ground(world, 500);
    const base = body(world, [0, 2, 0], false);
    world.add(base, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "torus",
            torus: { radialResolution: 200, tubularResolution: 200, radius: 2, thickness: 1 },
        }),
    });
    return [launch(world)];
}
function launchStack(world: World): void {
    const state = world.resource(ContinuousState);
    if (state.bullet !== -1) world.destroy(state.bullet);
    const eid = body(world, [20.5, 5.5, 0]);
    world.storage(Body).isBullet.set(eid, 1);
    world.storage(Body).linearVelocity.set(eid, -500, 0, 0);
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.25, density: 10000 });
    state.bullet = eid;
}
function bulletVersusStack(world: World): number[] {
    ground(world, 50);
    const wall = body(world, [0, -1, 0], false);
    world.add(wall, Shape, {
        geometry: world.resource(Hulls).add({
            name: "wall",
            halfExtents: [0.1, 5, 10],
            transform: { position: [-1, 5, 0], rotation: [0, 0, 0, 1] },
        }),
    });
    const geometry = box(world, [0.5, 0.5, 0.5]);
    return Array.from({ length: 10 }, (_, row) => {
        const eid = body(world, [0, f(0.5 + f(f(1.1) * row)), 0]);
        world.add(eid, Shape, { geometry });
        return eid;
    });
}
function meshGround(world: World): void {
    const state = world.resource(ContinuousState);
    for (const eid of state.walls) world.destroy(eid);
    state.walls.length = 0;
    if (state.ground !== -1) world.destroy(state.ground);
    const eid = body(world, [0, 0, 0], false);
    state.ground = eid;
    world.add(eid, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "wave",
            wave: {
                xCount: 40,
                zCount: 40,
                cellWidth: 1,
                amplitude: state.amplitude,
                rowFrequency: 0.1,
                columnFrequency: 0.2,
            },
        }),
        filterCategoryLow: 1,
        filterCategoryHigh: 0,
    });
    for (const [position, halfExtents] of [
        [
            [0, 1, -20],
            [20, 1, 0.1],
        ],
        [
            [0, 1, 20],
            [20, 1, 0.1],
        ],
        [
            [-20, 1, 0],
            [0.1, 1, 20],
        ],
        [
            [20, 1, 0],
            [0.1, 1, 20],
        ],
    ] as [V3, V3][]) {
        const shape = world.create();
        state.walls.push(shape);
        world.add(shape, Shape, {
            body: eid,
            filterCategoryLow: 1,
            filterCategoryHigh: 0,
            geometry: world.resource(Hulls).add({
                name: "wall",
                halfExtents,
                transform: { position, rotation: [0, 0, 0, 1] },
            }),
        });
    }
}
function generate(world: World): void {
    const state = world.resource(ContinuousState);
    for (const eid of state.bodies) world.destroy(eid);
    state.bodies.length = 0;
    state.steps = 0;
    let seed = meshSeed(world) >>> 0;
    const random = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(hi - lo) * f((seed % 32768) / 32767)) + lo);
    };
    const geometry = state.type === 0 ? state.dropBox : state.cylinder;
    for (let i = 0; i < 32; i++)
        for (let j = 0; j < 32; j++) {
            const eid = body(world, [0.5 * (i - 16), 5, 0.5 * (j - 16)]);
            world
                .storage(Body)
                .linearVelocity.set(eid, random(-1, 1), random(-1, 1), random(-1, 1));
            world
                .storage(Body)
                .angularVelocity.set(eid, random(-5, 5), random(-5, 5), random(-5, 5));
            world.add(eid, Shape, {
                ...(state.type === 1
                    ? {
                          kind: ShapeKind.Capsule,
                          capsuleCenter1: [0, -0.2, 0] as V3,
                          capsuleCenter2: [0, 0.2, 0] as V3,
                          radius: 0.05,
                      }
                    : state.type === 3
                      ? { kind: ShapeKind.Sphere, radius: 0.05 }
                      : { geometry }),
                rollingResistance: state.type === 1 ? 0.4 : 0.1,
                ...(state.collide
                    ? {}
                    : {
                          filterCategoryLow: 2,
                          filterCategoryHigh: 0,
                          filterMaskLow: 1,
                          filterMaskHigh: 0,
                      }),
            });
            state.bodies.push(eid);
        }
}
function meshDrop(world: World): number[] {
    const state = world.resource(ContinuousState);
    state.meshDrop = true;
    meshGround(world);
    state.cylinder = world.resource(Hulls).add({
        name: "cylinder",
        cylinder: { height: 0.4, radius: 0.05, yOffset: 0, sides: 6 },
    });
    state.dropBox = box(world, [0.02, 0.2, 0.04]);
    generate(world);
    return world.resource(ContinuousState).bodies;
}
const meshControls = {
    Type: {
        label: "Type",
        value: 0,
        select: true,
        options: ["box", "capsule", "cylinder", "sphere"].map((label, value) => ({ label, value })),
        set(world, _bodies, value) {
            world.resource(ContinuousState).type = value;
            generate(world);
        },
    },
    Amplitude: {
        label: "Amplitude",
        value: 0.5,
        min: 0,
        max: 1,
        step: 0.001,
        set(world, _bodies, value) {
            world.resource(ContinuousState).amplitude = value;
            meshGround(world);
            generate(world);
        },
    },
    Collide: {
        label: "Collide",
        value: 1,
        set(world, _bodies, value) {
            world.resource(ContinuousState).collide = value !== 0;
            generate(world);
        },
    },
    Generate: {
        label: "Generate",
        value: 0,
        button: true,
        set(world) {
            generate(world);
        },
    },
    "Auto Generate": {
        label: "Auto Generate",
        value: 0,
        button: true,
        set(world) {
            const state = world.resource(ContinuousState);
            state.auto = !state.auto;
            state.steps = 0;
        },
    },
} satisfies Controls;
export const continuousScenes = {
    bulletVersusStack: {
        construct: bulletVersusStack,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 25, pitch: 20, distance: 36, target: [6, 4, 0] },
        controls: {
            Launch: {
                label: "Launch",
                value: 0,
                button: true,
                set(world) {
                    launchStack(world);
                },
            },
        } satisfies Controls,
    },
    meshDrop: {
        construct: meshDrop,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 50, distance: 62, phoneDistance: 130, target: [0, 0, 0] },
        controls: meshControls,
    },
    thinWall: {
        construct: thinWall,
        gridExtent: 40,
        gridSpacing: 2,
        camera: { yaw: 30, pitch: 25, distance: 48, target: [0, 8, 5] },
        controls: undefined,
    },
    bounceHouse: {
        construct: bounceHouse,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 55, distance: 36, target: [0, 3, 0] },
        controls: undefined,
    },
    spinningStick: {
        construct: spinningStick,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 20, distance: 34, target: [0, 9, 0] },
        controls: undefined,
    },
    needleMesh: {
        construct: needleMesh,
        gridExtent: 10,
        gridSpacing: 0.5,
        camera: { yaw: 35, pitch: 20, distance: 7, target: [0, 2, 0] },
        controls: undefined,
    },
    humpMesh: {
        construct: humpMesh,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 30, distance: 22, target: [0, 1, 0] },
        controls: undefined,
    },
    isFast: {
        construct: isFast,
        gridExtent: 40,
        gridSpacing: 2,
        camera: { yaw: 15, pitch: 15, distance: 66, phoneDistance: 100, target: [0, 18, 0] },
        controls: undefined,
    },
    stall: {
        construct: stall,
        gridExtent: 500,
        gridSpacing: 2,
        camera: { yaw: 145, pitch: 25, distance: 20, target: [0, 2, -2] },
        controls: {
            Launch: {
                label: "Launch",
                value: 0,
                button: true,
                set(world) {
                    launch(world);
                },
            },
        } satisfies Controls,
    },
} as const;
