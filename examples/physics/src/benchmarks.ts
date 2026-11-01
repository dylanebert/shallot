import {
    Body,
    BodyType,
    FilterJoint,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    SphericalJoint,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    applyWind,
    computeCosSin,
    defaultExplosionDef,
    explode,
    isAwake,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setTargetTransform,
    stepPhysics,
} from "@dylanebert/shallot/standard/physics";
import { BenchmarkVillagePlugin, benchmarkVillageScene } from "./benchmark-village";
import { human } from "./human";
import { QuerySamplePlugin, queryScenes } from "./queries";
import { MotorizedJoint } from "./roles";
import { benchmarkSensor, SensorSamplePlugin } from "./sensor";
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
function largeWorld(world: World): number[] {
    Object.assign(world.resource(BenchmarkSample), { scene: "largeWorld", step: 0 });
    world.resource(PhysicsWorldDefinition).capacity = {
        staticShapeCount: 1_000_000,
        staticBodyCount: 1_000_000,
        dynamicShapeCount: 100,
        dynamicBodyCount: 100,
        contactCount: 1024,
    };
    const geometry = box(world, [5, 0.25, 5]);
    const bodies: number[] = [];
    for (let i = 0; i < 1000; i++)
        for (let j = 0; j < 1000; j++) {
            const eid = body(
                world,
                geometry,
                [-5000 + (i + 0.5) * 10, 0, -5000 + (j + 0.5) * 10],
                false,
            );
            world.storage(Shape).invokeContactCreation.set(eid, 1);
            bodies.push(eid);
        }
    return bodies;
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
        sleepPairs: [] as [number, number][],
        sleepJoint: -1,
        sleepIndex: -1,
        sleepComplete: false,
        windShapes: [] as number[],
        noise: [0, 0, 0],
        explosionPending: false,
        destructionBodies: [] as number[],
        randomSeed: 12345,

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
        if (sample.scene === "destruction" && sample.explosionPending) {
            sample.explosionPending = false;
            explode(world, {
                ...defaultExplosionDef(),
                radius: 2.5,
                falloff: 1.25,
                position: { x: 0, y: 5, z: 0 },
                impulsePerArea: 1000,
            });
        }
        if (sample.scene === "chains") {
            const wind = {
                x: f(20 * f(1 + sample.noise[0]!)),
                y: f(20 * sample.noise[1]!),
                z: f(20 * sample.noise[2]!),
            };
            for (const eid of sample.windShapes) applyWind(world, eid, wind, 1, 1, 20, false);
            for (let i = 0; i < 3; i++) {
                let x = sample.randomSeed;
                x ^= x << 13;
                x ^= x >>> 17;
                x ^= x << 5;
                sample.randomSeed = x >>> 0;
                const rand = f(
                    f(f(f(0.3) - f(-0.3)) * f((sample.randomSeed % 32768) / 32767)) + f(-0.3),
                );
                sample.noise[i] = f(f(f(1 - f(0.05)) * sample.noise[i]!) + f(f(0.05) * rand));
            }
        }
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
        if (sample.scene === "sleep") {
            sample.sleepComplete = false;
            if (sample.sleepJoint >= 0) {
                world.destroy(sample.sleepJoint);
                sample.sleepJoint = -1;
                return;
            }
            sample.sleepIndex = sample.sleepPairs.findIndex(([a]) => isAwake(world, a) === false);
            if (sample.sleepIndex >= 0) {
                const [a, b] = sample.sleepPairs[sample.sleepIndex]!;
                sample.sleepJoint = world.create();
                world.add(sample.sleepJoint, FilterJoint, { a, b });
            }
            return;
        }
        if (sample.scene === "largeWorld") {
            const step = sample.step++;
            if (step === 0 || step % 5 !== 0 || step > 500) return;
            const index = step / 5 - 1;
            const eid = world.create();
            world.add(eid, Transform, {
                translation: [
                    -4000 + ((index % 10) + 0.5) * 800,
                    1.5,
                    -4000 + (Math.floor(index / 10) + 0.5) * 800,
                ],
            });
            world.add(eid, Body, { type: BodyType.Dynamic });
            world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5 });
            return;
        }
        if (sample.scene !== "rain") return;
        if ((sample.step++ & 0x2f) !== 0) return;
        const filling = sample.columnCount < 10;
        const column = filling ? sample.columnCount++ : sample.columnIndex;
        for (let row = 0; row < 10; row++) {
            const index = row * 10 + column;
            for (const group of sample.groups[index]!) {
                for (const joint of group.joints.slice(-3)) world.destroy(joint);
                for (const joint of group.joints.slice(0, -3)) world.destroy(joint);
                for (const [index, body] of group.bodies.entries()) {
                    // Body removal destroys its attached shapes in Box3D's linked-list order.
                    world.remove(body, Body);
                    world.destroy(body);
                    if (group.shapes[index]) world.destroy(group.shapes[index]!);
                }
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
function gridGround(world: World): number {
    const geometry = world.resource(PhysicsMeshes).add({
        name: "grid",
        grid: { xCount: 40, zCount: 40, cellWidth: 1, materialCount: 0, identifyEdges: true },
    });
    const eid = world.create();
    world.add(eid, Body);
    world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry });
    return eid;
}
function explosion(world: World): number[] {
    const base = gridGround(world);
    for (const [halfExtents, position] of [
        [
            [20, 1, 0.1],
            [0, 1, -20],
        ],
        [
            [20, 1, 0.1],
            [0, 1, 20],
        ],
        [
            [0.1, 1, 20],
            [-20, 1, 0],
        ],
        [
            [0.1, 1, 20],
            [20, 1, 0],
        ],
    ] as const)
        shape(
            world,
            base,
            world.resource(Hulls).add({
                name: "wall",
                halfExtents,
                transform: { position, rotation: [0, 0, 0, 1] },
            }),
        );
    const geometry = world
        .resource(Hulls)
        .add({ name: "cylinder", cylinder: { height: 0.5, radius: 0.2, yOffset: 0, sides: 15 } });
    const bodies: number[] = [];
    for (let i = -16; i <= 16; i++)
        for (let k = -16; k <= 16; k++) {
            const eid = body(world, geometry, [i, 0, k]);
            world.storage(Shape).explosionScale.set(eid, 2);
            bodies.push(eid);
        }
    return bodies;
}
function spawnDestruction(world: World): number[] {
    const sample = world.resource(BenchmarkSample);
    const a = f(2.5 / 20);
    const geometry = box(world, [f(f(0.8) * a), f(f(0.8) * a), f(f(0.8) * a)]);
    const bodies: number[] = [];
    for (let i = 0; i < 20; i++)
        for (let j = 0; j < 20; j++)
            for (let k = 0; k < 20; k++) {
                let x = sample.randomSeed;
                x ^= x << 13;
                x ^= x >>> 17;
                x ^= x << 5;
                sample.randomSeed = x >>> 0;
                if ((sample.randomSeed % 32768) % 2 === 0) continue;
                bodies.push(
                    body(world, geometry, [
                        f((2 * i - 19) * a),
                        f((2 * j + 1) * a),
                        f((2 * k - 19) * a),
                    ]),
                );
            }
    sample.destructionBodies = bodies;
    sample.explosionPending = true;
    return bodies;
}
function destruction(world: World): number[] {
    gridGround(world);
    Object.assign(world.resource(BenchmarkSample), {
        scene: "destruction",
        step: 0,
        randomSeed: 12345,
    });
    return spawnDestruction(world);
}
const AfterDestruction: System = {
    name: "destruction-destroy",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(BenchmarkSample);
        if (sample.scene !== "destruction" || ++sample.step % 140 !== 0) return;
        for (const eid of sample.destructionBodies) world.destroy(eid);
    },
};
const SpawnDestruction: System = {
    name: "destruction-spawn",
    group: "fixed",
    after: [AfterDestruction],
    update(world) {
        const sample = world.resource(BenchmarkSample);
        if (sample.scene === "destruction" && sample.step % 140 === 0) spawnDestruction(world);
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
const StepSleep: System = {
    name: "sleep-step",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const s = world.resource(BenchmarkSample);
        if (s.scene !== "sleep" || s.sleepJoint >= 0) return;
        stepPhysics(world, 1 / 60, 4);
        s.sleepComplete = true;
    },
};
function sleep(world: World): number[] {
    const s = world.resource(BenchmarkSample);
    Object.assign(s, {
        scene: "sleep",
        sleepPairs: [],
        sleepJoint: -1,
        sleepIndex: -1,
        sleepComplete: false,
    });
    const def = world.resource(PhysicsWorldDefinition);
    def.enableAutomaticStep = false;
    def.capacity = {
        staticShapeCount: 4,
        staticBodyCount: 4,
        dynamicShapeCount: 2200,
        dynamicBodyCount: 2200,
        contactCount: 12000,
    };
    body(world, box(world, [128, 1, 4]), [0, -1, 0], false);
    const hull = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let p = 0; p < 10; p++) {
        const pair: [number, number] = [0, 0];
        for (let i = 0; i < 20; i++)
            for (let j = i; j < 20; j++) {
                const e = body(world, hull, [
                    (i + 1) * 0.5 + (j - i) - 108 + p * 24 - 10,
                    (2 * i + 1) * 0.5,
                    0,
                ]);
                world.storage(Body).sleepThreshold.set(e, 1);
                bodies.push(e);
                if (i === 0 && j === 0) pair[0] = e;
                else if (i === 1 && j === 1) pair[1] = e;
            }
        s.sleepPairs.push(pair);
    }
    return bodies;
}
export const BenchmarkSamplePlugin: Plugin = {
    name: "BenchmarkSample",
    dependencies: [
        StandardPhysicsPlugin,
        TreeSamplePlugin,
        SensorSamplePlugin,
        QuerySamplePlugin,
        BenchmarkVillagePlugin,
    ],
    systems: [SpawnBenchmark, BeforeBenchmark, AfterDestruction, SpawnDestruction, StepSleep],
    recovery: recoverResource(BenchmarkSample),
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
function chains(world: World): number[] {
    const geometry = world.resource(PhysicsMeshes).add({
        name: "wave",
        wave: {
            xCount: 80,
            zCount: 80,
            cellWidth: 1,
            amplitude: 0.5,
            rowFrequency: 0.05,
            columnFrequency: 0.01,
        },
    });
    const ground = world.create();
    world.add(ground, Body);
    world.add(ground, Shape, { kind: ShapeKind.Mesh, geometry });
    const sample = world.resource(BenchmarkSample);
    Object.assign(sample, { scene: "chains", randomSeed: 12345, noise: [0, 0, 0], windShapes: [] });
    const bodies: number[] = [];
    for (let row = 0; row < 25; row++)
        for (let column = 0; column < 25; column++) {
            let previous = 0;
            for (let i = 0; i < 4; i++) {
                const eid = world.create();
                world.add(eid, Transform, {
                    translation: [-25 + 2 * row, (1 - 2 * i) * 0.25 + 3, -25 + 2 * column],
                });
                world.add(eid, Body, {
                    type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                    enableSleep: 0,
                });
                world.add(eid, Shape, {
                    kind: ShapeKind.Capsule,
                    capsuleCenter1: [0, -0.25, 0],
                    capsuleCenter2: [0, 0.25, 0],
                    radius: 0.125,
                });
                if (i === 3) sample.windShapes.push(eid);
                if (i > 0) {
                    const joint = world.create();
                    world.add(joint, SphericalJoint, {
                        a: previous,
                        b: eid,
                        localAnchorA: [0, -0.25, 0],
                        localAnchorB: [0, 0.25, 0],
                        enableSpring: 1,
                        hertz: 1,
                        dampingRatio: 0.7,
                        enableMotor: 1,
                        maxMotorTorque: 1,
                    });
                }
                previous = eid;
                bodies.push(eid);
            }
        }
    return bodies;
}
function benchmarkMeshDrop(world: World): number[] {
    const mesh = world.resource(PhysicsMeshes).add({
        name: "terrain",
        wave: {
            xCount: 120,
            zCount: 120,
            cellWidth: 1,
            amplitude: 0.5,
            rowFrequency: 0.1,
            columnFrequency: 0.2,
        },
    });
    const base = world.create();
    world.add(base, Body);
    world.add(base, Shape, {
        kind: ShapeKind.Mesh,
        geometry: mesh,
        filterCategoryLow: 1,
        filterCategoryHigh: 0,
    });
    const hull = box(world, [0.02, 0.2, 0.04]);
    let seed = 3963634789;
    const rand = (lo: number, hi: number) => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(f(f(hi) - f(lo)) * f((seed % 32768) / 32767)) + f(lo));
    };
    const bodies: number[] = [];
    for (let i = 0; i < 100; i++)
        for (let j = 0; j < 100; j++) {
            const v: [number, number, number] = [rand(-1, 1), rand(-1, 1), rand(-1, 1)];
            const w: [number, number, number] = [rand(-5, 5), rand(-5, 5), rand(-5, 5)];
            const e = body(world, hull, [-49.5 + i, 5, -49.5 + j]);
            world.storage(Body).linearVelocity.set(e, ...v);
            world.storage(Body).angularVelocity.set(e, ...w);
            const shapes = world.storage(Shape);
            shapes.rollingResistance.set(e, 0.1);
            shapes.filterCategoryLow.set(e, 2);
            shapes.filterCategoryHigh.set(e, 0);
            shapes.filterMaskLow.set(e, 1);
            shapes.filterMaskHigh.set(e, 0);
            bodies.push(e);
        }
    return bodies;
}
function spinner(world: World): number[] {
    world.resource(PhysicsWorldDefinition).capacity = {
        staticShapeCount: 4,
        staticBodyCount: 4,
        dynamicShapeCount: 2000,
        dynamicBodyCount: 2000,
        contactCount: 20000,
    };
    const vertices: number[] = [],
        indices: number[] = [];
    for (let i = 0; i < 90; i++) {
        const [c, s] = computeCosSin(world, f(f(f(2 * f(Math.PI)) * i) / 90));
        const x = f(12 * c),
            y = f(f(12 * s) + 12);
        vertices.push(x, y, -1, x, y, 1);
        const j = (i + 1) % 90;
        indices.push(
            2 * i,
            2 * i + 1,
            2 * j + 1,
            2 * i,
            2 * j + 1,
            2 * j,
            180,
            2 * i,
            2 * j,
            181,
            2 * j + 1,
            2 * i + 1,
        );
    }
    vertices.push(0, 12, -1, 0, 12, 1);
    const base = world.create();
    world.add(base, Body);
    world.add(base, Shape, {
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "cylinder",
            vertices: new Float32Array(vertices),
            indices: new Uint32Array(indices),
            identifyEdges: true,
        }),
        friction: 0.1,
    });
    const paddle = body(world, box(world, [0.3, 6.1, 1.1]), [0, 6, 0], true, 1000, false);
    world.storage(Shape).friction.set(paddle, 0);
    const motor = world.create();
    world.add(motor, RevoluteJoint, {
        a: base,
        b: paddle,
        localAnchorA: [0, 6, 0],
        enableMotor: 1,
        motorSpeed: 8,
        maxMotorTorque: f(3.4028234663852886e38),
    });
    world.add(motor, MotorizedJoint);
    const bodies = [paddle];
    const limit = f(f(Math.sqrt(44)) - f(2 * f(0.55)));
    const zLimit = f(1 - f(2 * f(0.2)));
    let x = -limit,
        y = 2,
        z = -zLimit;
    for (let i = 0; i < 1500; i++) {
        const e = world.create();
        world.add(e, Transform, { translation: [x, y, z] });
        world.add(e, Body, { type: BodyType.Dynamic });
        world.add(e, Shape, {
            kind: ShapeKind.Sphere,
            radius: 0.2,
            friction: 0.1,
            restitution: 0.1,
            density: 0.25,
        });
        bodies.push(e);
        z = f(z + f(0.55));
        if (z > zLimit) {
            z = -zLimit;
            x = f(x + f(0.55));
            if (x > limit) {
                x = -limit;
                y = f(y + f(0.55));
            }
        }
    }
    return bodies;
}
export const benchmarkScenes = {
    benchmarkVillage: benchmarkVillageScene,
    benchmarkMeshDrop: {
        construct: benchmarkMeshDrop,
        gridExtent: 60,
        gridSpacing: 4,
        camera: { yaw: 0, pitch: 35, distance: 140, phoneDistance: 220, target: [0, 0, 0] },
        controls: undefined,
    },
    spinner: {
        construct: spinner,
        gridExtent: 16,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 5, distance: 38, phoneDistance: 58, target: [0, 12, 0] },
        controls: undefined,
    },
    sleep: {
        construct: sleep,
        iterationComplete: (world: World) => world.resource(BenchmarkSample).sleepComplete,
        gridExtent: 128,
        gridSpacing: 4,
        camera: { yaw: 0, pitch: 35, distance: 260, phoneDistance: 420, target: [0, 5, 0] },
        controls: undefined,
    },
    largeWorld: {
        construct: largeWorld,
        gridExtent: 0,
        gridSpacing: 10,
        camera: { yaw: 0, pitch: 10, distance: 250, target: [0, 0, 0] },
        controls: undefined,
    },
    ...queryScenes,
    benchmarkSensor: {
        construct: benchmarkSensor,
        gridExtent: 130,
        gridSpacing: 10,
        camera: { yaw: 0, pitch: 0, distance: 250, target: [0, 110, 0] },
        controls: undefined,
    },
    chains: {
        construct: chains,
        gridExtent: 40,
        gridSpacing: 5,
        camera: { yaw: 0, pitch: 15, distance: 50, target: [0, 5, 0] },
        controls: undefined,
    },
    explosion: {
        construct: explosion,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 20, distance: 30, target: [0, 0, 0] },
        controls: {
            Magnitude: { label: "Magnitude", value: 1000, min: 0, max: 2000, step: 1, set() {} },
            Explode: {
                label: "Explode",
                value: 0,
                button: true,
                set(
                    world: World,
                    _bodies: readonly number[],
                    _value: number,
                    values: Record<string, number>,
                ) {
                    explode(world, {
                        ...defaultExplosionDef(),
                        radius: 16,
                        position: { x: 0, y: -4, z: 0 },
                        impulsePerArea: values.Magnitude!,
                    });
                },
            },
        },
    },
    destruction: {
        construct: destruction,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 40, distance: 30, target: [0, 0, 0] },
        controls: undefined,
    },
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
