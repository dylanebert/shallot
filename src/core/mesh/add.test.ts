import { expect, test } from "bun:test";
import { runInNewContext } from "node:vm";
import { World } from "../../engine";
import { type Mesh, Meshes, meshResourcesKey } from "./index";
import { cube } from "./primitives";

test("mesh removal clears every reader and index reuse appends the new mesh in insertion order", () => {
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
        const next = meshes.add(replacement);
        expect(next).not.toBe(handle);
        expect(meshes.get(handle)).toBeUndefined();
        expect(meshes.remove(handle)).toBeUndefined();
        expect(meshes.get(next)).toBe(replacement);
        expect([...meshes.entries()]).toEqual([
            [other, second],
            [next, replacement],
        ]);
        expect([...meshes.values()]).toEqual([second, replacement]);
        meshes.clear();
        expect([...meshes]).toEqual([]);
        expect(meshes.has(handle)).toBe(false);
        expect(meshes.get(other)).toBeUndefined();
        expect(meshes.size).toBe(0);
        const afterClear = meshes.add(first);
        expect([handle, other, next]).not.toContain(afterClear);
        for (const stale of [handle, other, next]) expect(meshes.get(stale)).toBeUndefined();
    } finally {
        world.dispose();
    }
});

test("steady add and remove turnover keeps reusing one dense row past a generation wrap", () => {
    const world = new World();
    const meshes = world.resource(Meshes);
    const mesh = {
        name: "turnover",
        vertices: {},
        indices: {},
        indexBase: 0,
        indexCount: 3,
    } as Mesh;
    try {
        let handle = meshes.add(mesh);
        for (let i = 0; i < 2049; i++) {
            meshes.remove(handle);
            handle = meshes.add(mesh);
        }
        expect(world.resource(meshResourcesKey).meshes.capacity).toBe(1);
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
