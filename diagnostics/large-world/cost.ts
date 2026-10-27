import { Body, createApp, Hulls, Shape, StandardPhysicsPlugin, Transform, Meshes, MeshInstance, MeshMaterial, Materials, StandardMaterial, shapeMesh, shapeEdges } from "@dylanebert/shallot";
import { hashPhysics, PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { Gizmo, GizmoAsset, drawLine, registerGizmo } from "@dylanebert/shallot/extras";
import { AppPlugin, Viewer } from "../../examples/physics/src/app";

// Bundle for Node, then run with a grid count (1000 is the release sample) and optional "draw".
const grid = Number(process.argv[2] ?? 1000);
const shared = process.argv[3] === "shared";
const draw = process.argv[3] === "draw" || shared;
const workerCount = Number(process.argv[4] ?? 1);
if (!Number.isInteger(grid) || grid < 1) throw new Error("grid count must be a positive integer");
if (![0, 1].includes(workerCount)) throw new Error("worker count must be 0 (unshared) or 1 (shared), both one solver thread");
const report = (phase: string, start: number) => console.log(JSON.stringify({ phase, milliseconds: performance.now() - start, floorBodies: grid * grid, solverThreads: 1, workerCount, ...process.memoryUsage() }));
const opened = performance.now();
const config = {
    defaults: draw,
    plugins: draw ? [AppPlugin] : [StandardPhysicsPlugin],
    setup(world: import("@dylanebert/shallot").World) {
        const definition = world.resource(PhysicsWorldDefinition);
        definition.workerCount = workerCount;
        definition.maximumMemory = 4294967296;
        definition.capacity = {
            staticShapeCount: grid * grid,
            staticBodyCount: grid * grid,
            dynamicShapeCount: 100,
            dynamicBodyCount: 100,
            contactCount: 1024,
        };
    },
};
const app = draw ? await openGpuApp(config) : await createApp(config);
report("open", opened);
try {
    const world = app.world;
    const geometry = world.resource(Hulls).add({ name: "floor", halfExtents: [5, 0.25, 5] });
    const start = performance.now();
    for (let i = 0; i < grid; i++) for (let j = 0; j < grid; j++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [-5 * grid + (i + 0.5) * 10, 0, -5 * grid + (j + 0.5) * 10] });
        world.add(eid, Body);
        world.add(eid, Shape, { geometry, invokeContactCreation: 1 });
    }
    report("author", start);
    const tick = performance.now();
    try { world.tick(); }
    catch (error) { report("first tick failed", tick); throw error; }
    report("first tick (sync and solve)", tick);
    const steady = performance.now();
    world.tick();
    report("second tick", steady);
    let hash = 0n;
    const hashing = performance.now();
    for (let i = 0; i < 20; i++) hash ^= hashPhysics(world);
    report("20 hashes", hashing);
    console.log(`hash sentinel: ${hash}`);
    if (draw) {
        const view = performance.now();
        if (shared) {
            const first = [...world.query([Shape])][0]!;
            const shape = world.get(first, Shape)!;
            const mesh = world.resource(Meshes).add({ name: "floor", ...shapeMesh(world, shape) });
            const material = world.resource(Materials).add(StandardMaterial({ baseColor: [0.085, 0.08, 0.072, 1] }));
            const edges = new GizmoAsset();
            for (const [a, b] of shapeEdges(world, shape)) drawLine(edges, a, b, 0x000000);
            const handle = registerGizmo(world, edges);
            for (const eid of world.query([Shape])) {
                world.add(eid, MeshInstance, { mesh });
                world.add(eid, MeshMaterial, material);
                world.add(eid, Gizmo, { handle, width: 1, opacity: 0.45 });
            }
        } else world.resource(Viewer)(world, "boxStack");
        report("example view", view);
        const frame = performance.now();
        world.step(0);
        await world.frameFence;
        report("example frame", frame);
    }
} finally { app.dispose(); }
