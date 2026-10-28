import {
    Body,
    BodyType,
    createApp,
    Hulls,
    Shape,
    ShapeKind,
    SphericalJoint,
    Transform,
} from "@dylanebert/shallot";
import {
    getWorldCounters,
    type PhysicsWorldCounters,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

export async function counterPile(overflow: boolean, workerCount = 0) {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            const definition = world.resource(PhysicsWorldDefinition);
            definition.workerCount = workerCount;
            definition.enableAutomaticStep = false;
            definition.enableSleep = false;
        },
    });
    const world = app.world;
    const floor = world.resource(Hulls).add({ name: "floor", halfExtents: [20, 1, 20] });
    const box = world.resource(Hulls).add({ name: "box", halfExtents: [0.5, 0.5, 0.5] });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0] });
    world.add(ground, Body);
    world.add(ground, Shape, { geometry: floor });
    const count = overflow ? 26 : 3;
    for (let i = 0; i < count; i++) {
        const body = world.create();
        const y = overflow ? Math.fround(0.5 + Math.fround(Math.fround(0.02) * i)) : 0.5 + i;
        world.add(body, Transform, { translation: [0, y, 0] });
        world.add(body, Body, { type: BodyType.Dynamic });
        world.add(body, Shape, { geometry: box });
    }
    world.tick();
    if (!getWorldCounters(world)) throw new Error("counters unavailable after sync");
    return app;
}

export async function counterSweep(workerCount: number) {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            world.resource(PhysicsWorldDefinition).workerCount = workerCount;
        },
    });
    const world = app.world;
    const floor = world.resource(Hulls).add({ name: "floor", halfExtents: [20, 1, 20] });
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0] });
    world.add(ground, Body);
    world.add(ground, Shape, { geometry: floor });
    const bullet = world.create();
    world.add(bullet, Transform, { translation: [0, 2, 0] });
    world.add(bullet, Body, { type: BodyType.Dynamic, linearVelocity: [0, -200, 0] });
    world.add(bullet, Shape, { kind: ShapeKind.Sphere, radius: 0.1 });
    const rows: string[] = [];
    let distanceIterations = 0;
    for (let i = 0; i < 5; i++) {
        world.tick();
        const c = getWorldCounters(world)!;
        distanceIterations = Math.max(distanceIterations, c.distanceIterations);
        rows.push(counterRow(c));
    }
    world.add(world.create(), SphericalJoint, { a: ground, b: bullet });
    world.tick();
    rows.push(counterRow(getWorldCounters(world)!));
    app.dispose();
    return { rows, distanceIterations };
}

export function counterRow(c: PhysicsWorldCounters): string {
    return [
        c.bodyCount,
        c.shapeCount,
        c.contactCount,
        c.jointCount,
        c.islandCount,
        c.staticTreeHeight,
        c.treeHeight,
        c.satCallCount,
        c.satCacheHitCount,
        ...c.colorCounts,
        ...c.manifoldCounts,
        c.awakeContactCount,
        c.recycledContactCount,
        c.distanceIterations,
        c.pushBackIterations,
        c.rootIterations,
    ].join(" ");
}
