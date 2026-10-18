import { expect, test } from "bun:test";
import { World } from "../../engine";
import {
    Compounds,
    HeightFields,
    Hulls,
    PhysicsMeshes,
    PhysicsPlugin,
    ShapeMaterials,
} from "./index";

test("core physics restores every handle, duplicate labels, removed slots and allocation state with reusable images", () => {
    const world = new World();
    if (typeof PhysicsPlugin.recovery !== "function")
        throw new Error("Physics must own geometry recovery");
    world.registerRecovery(PhysicsPlugin.name, PhysicsPlugin.recovery(world));
    const hulls = world.resource(Hulls);
    const cube = structuredClone(hulls.get(0)!);
    const removed = hulls.add({ ...structuredClone(cube), name: "same" });
    const live = hulls.add({ ...structuredClone(cube), name: "same" });
    hulls.remove(removed);
    const mesh = world.resource(PhysicsMeshes).add({
        name: "same",
        vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 1, 2]),
    });
    const field = world.resource(HeightFields).add({
        name: "same",
        heights: new Float32Array(4),
        countX: 2,
        countZ: 2,
        scale: [1, 1, 1],
        globalMinimumHeight: 0,
        globalMaximumHeight: 1,
    });
    const compound = world.resource(Compounds).add({
        name: "same",
        hulls: [{ hull: live, transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1] } }],
    });
    const material = world.resource(ShapeMaterials).add({ name: "same", materials: [] });
    world.resource(PhysicsMeshes).add(structuredClone(world.resource(PhysicsMeshes).get(mesh)!));
    world.resource(HeightFields).add(structuredClone(world.resource(HeightFields).get(field)!));
    world.resource(Compounds).add(structuredClone(world.resource(Compounds).get(compound)!));
    world
        .resource(ShapeMaterials)
        .add(structuredClone(world.resource(ShapeMaterials).get(material)!));
    const collections = [
        hulls,
        world.resource(PhysicsMeshes),
        world.resource(HeightFields),
        world.resource(Compounds),
        world.resource(ShapeMaterials),
    ];
    const handles = [live, mesh, field, compound, material];
    const images = collections.map((assets) => structuredClone([...assets.entries()]));
    const saved = world.snapshot();
    hulls.get(live)!.points![0] = 88;
    for (const assets of collections) assets.clear();
    hulls.add({ ...cube, name: "future" });
    world.restore(saved);
    expect(world.resource(Hulls)).toBe(hulls);
    for (let i = 0; i < collections.length; i++) {
        expect([...collections[i].entries()]).toEqual(images[i]);
        expect(collections[i].has(handles[i])).toBe(true);
    }
    expect(hulls.has(removed)).toBe(false);
    expect(hulls.get(0)).toEqual(cube);
    hulls.get(live)!.points![3] = 99;
    world.restore(saved);
    expect(hulls.get(live)).toEqual({ ...cube, name: "same" });
    expect(hulls.add({ ...cube, name: "same" })).toBe(removed);
    expect(hulls.add({ ...cube, name: "same" })).toBe(3);
    world.dispose();
});
