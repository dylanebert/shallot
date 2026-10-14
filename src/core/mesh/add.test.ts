import { expect, test } from "bun:test";
import { runInNewContext } from "node:vm";
import { World } from "../../engine";
import { Meshes } from "./index";
import { cube } from "./primitives";

test("Meshes.add takes typed arrays from another realm as typed arrays, refusing them before MeshPlugin initializes as it does local ones", () => {
    const local = cube();
    const foreign = {
        name: "foreign",
        vertices: runInNewContext("new Float32Array(source)", { source: [...local.vertices] }),
        indices: local.indices,
    };
    expect(foreign.vertices instanceof Float32Array).toBe(false);
    const world = new World();
    try {
        const meshes = world.resource(Meshes);
        expect(() => meshes.add({ ...local, name: "local" })).toThrow(/needs MeshPlugin/);
        expect(() => meshes.add(foreign)).toThrow(/needs MeshPlugin/);
        expect(meshes.size).toBe(0);
    } finally {
        world.dispose();
    }
});
