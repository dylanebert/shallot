import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, hashPhysics, StandardPhysicsPlugin, type World } from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";
import { BenchmarkSamplePlugin } from "../../examples/physics/src/benchmarks";
import { EventSamplePlugin } from "../../examples/physics/src/events";
import { sceneSwitcher } from "../../examples/physics/src/sample";
import { createScene } from "../../examples/physics/src/scenes";

setDefaultTimeout(0);

function constructed(world: World) {
    for (const name of ["physics-sync", "world-definition", "constraints"]) {
        StandardPhysicsPlugin.systems!.find((system) => system.name === name)!.update!(world);
    }
    return { hash: hashPhysics(world), entities: world.entities().length };
}

test("largeWorld switched construction equals a fresh build before its first step", async () => {
    const plugins = [StandardPhysicsPlugin, EventSamplePlugin, BenchmarkSamplePlugin];
    const app = await createApp({
        defaults: false,
        plugins,
        setup(world) {
            world.resource(PhysicsWorldDefinition).maximumMemory = 4 * 1024 ** 3;
        },
    });
    const fresh = await createApp({
        defaults: false,
        plugins,
        setup(world) {
            createScene(world, "largeWorld");
        },
    });
    try {
        const select = sceneSwitcher(app.world);
        select("junkyard");
        app.world.tick();
        select("largeWorld");
        expect(constructed(app.world)).toEqual(constructed(fresh.world));
    } finally {
        app.dispose();
        fresh.dispose();
    }
});
