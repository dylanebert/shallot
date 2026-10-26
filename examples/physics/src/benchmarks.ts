import {
    Body,
    BodyType,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    Shape,
    ShapeKind,
    SphericalJoint,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    computeCosSin,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";
import { human } from "./human";
import { fallingTrees, TreeSamplePlugin, treeControls } from "./trees";

const f = Math.fround;
function box(world: World, halfExtents: [number, number, number]): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(
    world: World,
    geometry: number,
    translation: [number, number, number],
    dynamic = true,
    density = 1000,
    enableSleep = true,
): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, {
        type: dynamic ? BodyType.Dynamic : BodyType.Static,
        enableSleep: Number(enableSleep),
    });
    world.add(eid, Shape, { geometry, density });
    return eid;
}
function ground(world: World, extent: number): void {
    body(world, box(world, [extent, 1, extent]), [0, -1, 0], false);
}
function largePyramid(world: World): number[] {
    ground(world, 400);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 100; i++)
        for (let j = i; j < 100; j++)
            bodies.push(
                body(
                    world,
                    geometry,
                    [(i + 1) * 0.5 + (j - i) - 50, (2 * i + 1) * 0.5, 0],
                    true,
                    100,
                ),
            );
    return bodies;
}
function widePyramid(world: World): number[] {
    ground(world, 100);
    const geometry = box(world, [f(0.975), f(0.975), f(0.975)]);
    const bodies: number[] = [];
    for (let i = 0; i < 15; i++)
        for (let j = Math.floor(i / 2); j < 15 - Math.floor((i + 1) / 2); j++)
            for (let k = Math.floor(i / 2); k < 15 - Math.floor((i + 1) / 2); k++)
                bodies.push(
                    body(world, geometry, [
                        -15 + 2 * j + (i & 1),
                        1 + 2.5 * i,
                        -15 + 2 * k + (i & 1),
                    ]),
                );
    return bodies;
}
function manyPyramids(world: World): number[] {
    ground(world, 77);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    let z = -76;
    const delta = f(152 / 13);
    for (let row = 0; row < 14; row++) {
        for (let column = 0; column < 14; column++) {
            const center = -77 + column * 11 + 1;
            for (let i = 0; i < 10; i++)
                for (let j = i; j < 10; j++)
                    bodies.push(
                        body(
                            world,
                            geometry,
                            [(i + 1) * 0.5 + (j - i) + center - 0.5, (2 * i + 1) * 0.5, z],
                            true,
                            100,
                            false,
                        ),
                    );
        }
        z = f(z + delta);
    }
    return bodies;
}
function fallingBoxes(world: World): number[] {
    ground(world, 100);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 50; i++)
        for (let j = 0; j < 8; j++)
            for (let k = 0; k < 8; k++)
                bodies.push(body(world, geometry, [-8 + 2 * j, 2 * i + 2.5, -8 + 2 * k]));
    return bodies;
}
function jointGrid(world: World): number[] {
    world.resource(PhysicsWorldDefinition).enableSleep = false;
    const bodies: number[] = [];
    function joint(a: number, b: number, horizontal: boolean): void {
        const eid = world.create();
        world.add(eid, SphericalJoint, {
            a,
            b,
            localAnchorA: horizontal ? [0.5, 0, 0] : [0, -0.5, 0],
            localAnchorB: horizontal ? [-0.5, 0, 0] : [0, 0.5, 0],
        });
    }
    for (let k = 0; k < 100; k++) {
        for (let i = 0; i < 100; i++) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [k, -i, 0] });
            world.add(eid, Body, {
                type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                enableSleep: 0,
            });
            world.add(eid, Shape, {
                kind: ShapeKind.Sphere,
                radius: 0.4,
                filterCategoryLow: 2,
                filterCategoryHigh: 0,
                filterMaskLow: 0xfffffffd,
            });
            if (i > 0) joint(bodies[bodies.length - 1]!, eid, false);
            if (k > 0) joint(bodies[bodies.length - 100]!, eid, true);
            bodies.push(eid);
        }
    }
    return bodies;
}
export const BenchmarkSample = {
    create: () => ({
        degrees: 0,
        pusher: 0,
        scene: "",
        step: 0,
        columnCount: 0,
        columnIndex: 0,
        groups: Array.from({ length: 100 }, () => [] as ReturnType<typeof human>[]),
    }),
};
const BeforeBenchmark: System = {
    name: "benchmark-before-step",
    group: "fixed",
    after: [SyncPhysicsConstraintsSystem],
    before: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(BenchmarkSample);
        if (sample.scene !== "junkyard") return;
        const dt = f(1 / 60);
        sample.degrees = f(sample.degrees + f(-6 * dt));
        const [c, s] = computeCosSin(world, f(f(sample.degrees * f(Math.PI)) / 180));
        setTargetTransform(
            world,
            sample.pusher,
            [f(35 * c), 0, f(35 * s)],
            [0, 0, 0, 1],
            dt,
            false,
        );
    },
};
const SpawnBenchmark: System = {
    name: "benchmark-spawn",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem],
    update(world) {
        const sample = world.resource(BenchmarkSample);
        if (sample.scene !== "rain") return;
        if ((sample.step++ & 0x2f) !== 0) return;
        const filling = sample.columnCount < 10;
        const column = filling ? sample.columnCount++ : sample.columnIndex;
        for (let row = 0; row < 10; row++) {
            const index = row * 10 + column;
            for (const group of sample.groups[index]!) {
                world.destroy(group.joints.at(-1)!);
                for (const joint of group.joints.slice(0, -1)) world.destroy(joint);
                for (const body of group.bodies) world.destroy(body);
            }
            const groups: ReturnType<typeof human>[] = [];
            let x = -75 + 15 * (column + 0.5);
            for (let i = 0; i < 3; i++, x = f(x + 0.75))
                groups.push(human(world, [x, 20, -75 + 15 * (row + 0.5)], index));
            sample.groups[index] = groups;
        }
        if (!filling) sample.columnIndex = (sample.columnIndex + 1) % 10;
    },
};
function rain(world: World): number[] {
    const meshes = world.resource(PhysicsMeshes);
    const grid = meshes.add({
        name: "grid",
        grid: { xCount: 8, zCount: 8, cellWidth: f(15 / 8), materialCount: 1, identifyEdges: true },
    });
    const torus = meshes.add({
        name: "torus",
        torus: { radialResolution: 16, tubularResolution: 16, radius: 3.75, thickness: 1 },
    });
    for (let i = 0; i < 10; i++)
        for (let j = 0; j < 10; j++) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [-67.5 + 15 * i, 0, -67.5 + 15 * j] });
            world.add(eid, Body);
            world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry: grid });
            const other = world.create();
            world.add(other, Shape, { body: eid, kind: ShapeKind.Mesh, geometry: torus });
        }
    Object.assign(world.resource(BenchmarkSample), {
        scene: "rain",
        step: 0,
        columnCount: 0,
        columnIndex: 0,
    });
    return [];
}
export const BenchmarkSamplePlugin: Plugin = {
    name: "BenchmarkSample",
    dependencies: [StandardPhysicsPlugin, TreeSamplePlugin],
    systems: [SpawnBenchmark, BeforeBenchmark],
    recovery: (world) => ({
        snapshot: () => structuredClone(world.resource(BenchmarkSample)),
        restore: (state: ReturnType<typeof BenchmarkSample.create>) =>
            Object.assign(world.resource(BenchmarkSample), structuredClone(state)),
    }),
};
function shape(world: World, parent: number, geometry: number): void {
    const eid = world.create();
    world.add(eid, Shape, { body: parent, geometry });
}
function junkyard(world: World): number[] {
    const base = body(world, box(world, [120, 1, 120]), [0, -1, 0], false);
    for (const [halfExtents, position] of [
        [
            [1, 8, 50],
            [-50, 8, 0],
        ],
        [
            [1, 8, 50],
            [50, 8, 0],
        ],
        [
            [50, 8, 1],
            [0, 8, -50],
        ],
        [
            [50, 8, 1],
            [0, 8, 50],
        ],
    ] as const) {
        shape(
            world,
            base,
            world.resource(Hulls).add({
                name: "wall",
                halfExtents,
                transform: { position, rotation: [0, 0, 0, 1] },
            }),
        );
    }
    const phi = f(f(1 + f(Math.sqrt(5))) / 2);
    const [dc, ds] = computeCosSin(world, f(f(2 * f(Math.PI)) / phi));
    let c = 1,
        s = 0;
    const points: number[] = [];
    for (let i = 0; i < 10; i++) {
        const z = f(1 - f((2 * i + 1) / 10));
        const r = f(Math.sqrt(f(1 - f(z * z))));
        points.push(f(f(1.5 * r) * c), f(f(1.5 * r) * s), f(1.5 * z));
        const previous = c;
        c = f(f(dc * c) - f(ds * s));
        s = f(f(ds * previous) + f(dc * s));
    }
    const rock = world
        .resource(Hulls)
        .add({ name: "rock", points: new Float32Array(points), maxVertexCount: 10 });
    const bodies: number[] = [];
    for (let y = 0; y < 24; y++)
        for (let x = 0; x <= 20; x++)
            for (let z = 0; z <= 20; z++)
                bodies.push(body(world, rock, [-40 + 4 * x, 4 * y + 25, -40 + 4 * z]));
    const cylinder = world
        .resource(Hulls)
        .add({ name: "pusher", cylinder: { height: 24, radius: 4, yOffset: 0, sides: 16 } });
    const pusher = body(world, cylinder, [35, 0, 0]);
    world.storage(Body).type.set(pusher, BodyType.Kinematic);
    Object.assign(world.resource(BenchmarkSample), { scene: "junkyard", degrees: 0, pusher });
    bodies.push(pusher);
    return bodies;
}
function washer(world: World): number[] {
    world.resource(PhysicsWorldDefinition).contactRecycleDistance = f(0.05);
    ground(world, 60);
    const drum = world.create();
    world.add(drum, Transform, { translation: [0, 21, 0] });
    world.add(drum, Body, {
        type: BodyType.Kinematic,
        angularVelocity: [0, 0, f(f(f(Math.PI) / 180) * 25)],
        linearVelocity: [0.001, -0.002, 0],
    });
    const angle = f(f(Math.PI) / 18);
    const q = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, angle);
    const qo = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(f(0.1) * angle));
    function rotate(v: [number, number], z: number, w: number, inverse = false): [number, number] {
        const tx = f(-f(z * v[1])),
            ty = f(z * v[0]);
        const ux = inverse ? f(tx - f(w * v[0])) : f(tx + f(w * v[0]));
        const uy = inverse ? f(ty - f(w * v[1])) : f(ty + f(w * v[1]));
        return [f(v[0] + f(2 * f(-f(z * uy)))), f(v[1] + f(2 * f(z * ux)))];
    }
    function segment(a: [number, number], b: [number, number], r1: number, r2: number): void {
        const points: number[] = [];
        for (const z of [-10, 10])
            for (const v of [a, b])
                for (const r of [r1, r2]) points.push(f(r * v[0]), f(r * v[1]), z);
        shape(
            world,
            drum,
            world
                .resource(Hulls)
                .add({ name: "drum", points: new Float32Array(points), maxVertexCount: 8 }),
        );
    }
    let u1: [number, number] = [1, 0];
    for (let i = 0; i < 36; i++) {
        const u2: [number, number] = i === 35 ? [1, 0] : rotate(u1, q[2], q[3]);
        segment(rotate(u1, qo[2], qo[3], true), rotate(u2, qo[2], qo[3]), 16, 18);
        if (i % 9 === 0) segment(u1, u2, 14, 16);
        u1 = u2;
    }
    const geometry = box(world, [f(0.2), f(0.2), f(0.2)]);
    const bodies = [drum];
    let x = f(-f(f(2 * f(0.2)) * 20));
    for (let i = 0; i < 20; i++, x = f(x + f(4 * f(0.2)))) {
        let y = f(f(-f(f(2 * f(0.2)) * 20)) + 21);
        for (let j = 0; j < 20; j++, y = f(y + f(4 * f(0.2)))) {
            let z = f(-f(f(2 * f(0.2)) * 20));
            for (let k = 0; k < 20; k++, z = f(z + f(4 * f(0.2))))
                bodies.push(body(world, geometry, [x, y, z]));
        }
    }
    return bodies;
}
function convexPile(world: World): number[] {
    ground(world, 250);
    let state = 42;
    function random(): number {
        state = (Math.imul(state, 2147001325) + 715136305) >>> 0;
        return f(f((state & 0xffff) / 65535) - 0.5);
    }
    const points: number[] = [];
    for (let i = 0; i < 32; i++) {
        let x: number, y: number, z: number, length: number;
        do {
            x = random();
            y = random();
            z = random();
            length = f(f(f(x * x) + f(y * y)) + f(z * z));
        } while (length > 0.25);
        const inv = f(1 / f(Math.sqrt(length)));
        points.push(f(2 * f(inv * x)), f(2 * f(inv * y)), f(2 * f(inv * z)));
    }
    const geometry = world
        .resource(Hulls)
        .add({ name: "convex", points: new Float32Array(points), maxVertexCount: 32 });
    const bodies: number[] = [];
    for (let layer = 0; layer < 80; layer++)
        for (let z = 0; z < 8; z++)
            for (let x = 0; x < 8; x++)
                bodies.push(body(world, geometry, [(x - 4) * 4, 2 + 4 * layer, (z - 4) * 4]));
    return bodies;
}
function candyCups(world: World): number[] {
    ground(world, 60);
    const points: number[] = [];
    const delta = f(f(2 * f(Math.PI)) / 8);
    let alpha = 0;
    for (let i = 0; i < 8; i++) {
        const [c, s] = computeCosSin(world, alpha);
        points.push(f(f(0.6) * c), 0, f(f(0.6) * s), f(f(0.95) * c), 1, f(f(0.95) * s));
        alpha = f(alpha + delta);
    }
    const geometry = world
        .resource(Hulls)
        .add({ name: "cup", points: new Float32Array(points), maxVertexCount: 16 });
    const bodies: number[] = [];
    for (let i = 0; i < 16; i++)
        for (let j = 0; j < 16; j++)
            for (let k = 0; k < 16; k++)
                bodies.push(body(world, geometry, [-10 + 2.5 * j, i, -10 + 2.5 * k]));
    return bodies;
}
export const benchmarkScenes = {
    fallingTrees: {
        construct: fallingTrees,
        gridExtent: 100,
        gridSpacing: 10,
        camera: { yaw: 20, pitch: 0, distance: 140, target: [0, 15, 0] },
        controls: treeControls,
    },
    candyCups: {
        construct: candyCups,
        gridExtent: 60,
        gridSpacing: 5,
        camera: { yaw: 45, pitch: 20, distance: 70, target: [0, 0, 0] },
        controls: undefined,
    },
    convexPile: {
        construct: convexPile,
        gridExtent: 250,
        gridSpacing: 10,
        camera: { yaw: 45, pitch: 20, distance: 100, target: [0, 40, 0] },
        controls: undefined,
    },
    washer: {
        construct: washer,
        gridExtent: 60,
        gridSpacing: 5,
        camera: { yaw: 15, pitch: 20, distance: 60, target: [0, 15, 0] },
        controls: undefined,
    },
    junkyard: {
        construct: junkyard,
        gridExtent: 120,
        gridSpacing: 10,
        camera: { yaw: 45, pitch: 20, distance: 140, target: [0, 20, 0] },
        controls: undefined,
    },
    rain: {
        construct: rain,
        gridExtent: 75,
        gridSpacing: 15,
        camera: { yaw: 25, pitch: 10, distance: 70, target: [0, 0, 0] },
        controls: undefined,
    },
    largePyramid: {
        construct: largePyramid,
        gridExtent: 400,
        gridSpacing: 10,
        camera: { yaw: 40, pitch: -10, distance: 110, target: [0, 40, 0] },
        controls: undefined,
    },
    widePyramid: {
        construct: widePyramid,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: 0, pitch: 5, distance: 80, target: [0, 18, 0] },
        controls: undefined,
    },
    manyPyramids: {
        construct: manyPyramids,
        gridExtent: 77,
        gridSpacing: 7,
        camera: { yaw: -10, pitch: 10, distance: 120, target: [0, 5, 0] },
        controls: undefined,
    },
    fallingBoxes: {
        construct: fallingBoxes,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: 45, pitch: 10, distance: 80, target: [0, 20, 0] },
        controls: undefined,
    },
    jointGrid: {
        construct: jointGrid,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: -25, pitch: 25, distance: 94, target: [30, -30, 30] },
        controls: undefined,
    },
} as const;
