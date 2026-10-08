import { expect, test } from "bun:test";
import { type Plugin, swapPlugins, World } from "../../engine";

async function load(version: string): Promise<typeof import("./global-transform")> {
    return import(`./global-transform.ts?${version}`);
}

function install(world: World, plugin: Plugin): void {
    for (const entry of plugin.components ?? []) world.registry.register(entry, plugin.name);
    world.registerRecovery(
        plugin.name,
        typeof plugin.recovery === "function" ? plugin.recovery(world) : plugin.recovery,
    );
    for (const system of plugin.systems ?? []) world.addSystem(system, plugin.name);
}

test("a compatible re-evaluated TransformPlugin swaps boundary behavior without registering twice", async () => {
    const old = await load("old");
    const next = await load("new");
    const world = new World();
    try {
        install(world, old.TransformPlugin);
        const eid = world.create();
        world.add(eid, old.Transform, { translation: [3, 0, 0, 0] });
        world.tick();
        const runtime = world.resource(old.TransformRuntime);
        expect(await swapPlugins(world, [old.TransformPlugin], [next.TransformPlugin])).toEqual({
            ok: true,
        });
        expect(world.resource(next.TransformRuntime)).toBe(runtime);
        expect(world.hasSystem(old.GlobalTransformTickStartSystem)).toBe(true);
        expect(world.hasSystem(next.GlobalTransformTickStartSystem)).toBe(false);
        expect(old.GlobalTransformTickStartSystem.update).toBe(
            next.GlobalTransformTickStartSystem.update,
        );
        world.storage(next.Transform).translation.x.set(eid, 7);
        world.tick();
        expect(world.storage(next.GlobalTransform).translation.x.get(eid)).toBe(7);
        expect(() => world.snapshot()).not.toThrow();
    } finally {
        world.dispose();
    }
});

test("a reloaded recovery factory that cannot carry its state returns a rebuild result, never throws", async () => {
    const old = await load("recovery-old");
    const next = await load("recovery-new");
    const world = new World();
    try {
        install(world, old.TransformPlugin);
        const incompatible: Plugin = {
            ...next.TransformPlugin,
            recovery() {
                throw new Error("runtime format cannot be carried");
            },
        };
        expect(await swapPlugins(world, [old.TransformPlugin], [incompatible])).toEqual({
            ok: false,
            reason: "Transform: recovery threw — runtime format cannot be carried",
        });
    } finally {
        world.dispose();
    }
});
