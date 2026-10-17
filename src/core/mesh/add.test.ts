import { expect, test } from "bun:test";
import { runInNewContext } from "node:vm";
import { World } from "../../engine";
import { type Mesh, Meshes } from "./index";
import { cube } from "./primitives";

test("mesh removal clears every reader and handle reuse appends the new mesh in insertion order", () => {
    const world = new World();
    const meshes = world.resource(Meshes);
    const mesh = (name: string) =>
        ({ name, vertices: {}, indices: {}, indexBase: 0, indexCount: 3 }) as Mesh;
    try {
        const first = mesh("first");
        const second = mesh("second");
        const handle = meshes.add(first);
        const other = meshes.add(second);
        expect(meshes.remove(handle)).toBe(first);
        expect(meshes.get(handle)).toBeUndefined();
        expect(meshes.has(handle)).toBe(false);
        expect([...meshes.entries()]).toEqual([[other, second]]);
        expect([...meshes.values()]).toEqual([second]);
        expect([...meshes]).toEqual([second]);
        expect(meshes.size).toBe(1);
        const replacement = mesh("replacement");
        expect(meshes.add(replacement)).toBe(handle);
        expect(meshes.get(handle)).toBe(replacement);
        expect([...meshes.entries()]).toEqual([
            [other, second],
            [handle, replacement],
        ]);
        expect([...meshes.values()]).toEqual([second, replacement]);
        meshes.clear();
        expect([...meshes]).toEqual([]);
        expect(meshes.has(handle)).toBe(false);
        expect(meshes.get(other)).toBeUndefined();
        expect(meshes.size).toBe(0);
    } finally {
        world.dispose();
    }
});

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
