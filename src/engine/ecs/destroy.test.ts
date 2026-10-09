import { expect, test } from "bun:test";
import { component, f16, f32, vec4 } from "./component";
import { World } from "./world";

const Held = component("destroy-held-values", {
    scalar: f32,
    vector: vec4,
    half: f16,
});

const Unregistered = {
    scalar: f32,
    vector: vec4,
    half: f16,
};

test("destroy clears every held field and publishes its change mark", () => {
    const world = new World();
    try {
        const eid = world.create();
        world.add(eid, Held, { scalar: 17, vector: [1, 2, 3, 4], half: 1.5 });
        const storage = world.storage(Held);
        const scalarColumn = world.fieldStorage(Held, "scalar");
        const vectorColumn = world.fieldStorage(Held, "vector");
        const halfColumn = world.fieldStorage(Held, "half");
        scalarColumn.dirty.fill(0);
        vectorColumn.dirty.fill(0);
        halfColumn.dirty.fill(0);

        world.destroy(eid);

        expect(storage.scalar.get(eid)).toBe(0);
        expect(scalarColumn.column[eid]).toBe(0);
        expect(Array.from(storage.vector.read(eid, new Float32Array(4)))).toEqual([0, 0, 0, 0]);
        expect(Array.from(vectorColumn.column.slice(eid * 4, eid * 4 + 4))).toEqual([0, 0, 0, 0]);
        expect(storage.half.get(eid)).toBe(0);
        expect(halfColumn.column[eid]).toBe(0);
        for (const field of [scalarColumn, vectorColumn, halfColumn]) {
            expect(field.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        }
    } finally {
        world.dispose();
    }
});

test("reusing a destroyed eid does not expose fields of its unregistered component", () => {
    const world = new World();
    try {
        const previous = world.create();
        world.add(previous, Unregistered, {
            scalar: 23,
            vector: [5, 6, 7, 8],
            half: 2.5,
        });
        const storage = world.storage(Unregistered);
        const eidField = world.fieldStorage(Unregistered, "scalar");
        const vectorColumn = world.fieldStorage(Unregistered, "vector");
        const halfColumn = world.fieldStorage(Unregistered, "half");

        world.destroy(previous);
        const reused = world.create();
        expect(reused).toBe(previous);
        world.add(reused, Unregistered);

        expect(storage.scalar.get(reused)).toBe(0);
        expect(eidField.column[reused]).toBe(0);
        expect(Array.from(storage.vector.read(reused, new Float32Array(4)))).toEqual([0, 0, 0, 0]);
        expect(Array.from(vectorColumn.column.slice(reused * 4, reused * 4 + 4))).toEqual([
            0, 0, 0, 0,
        ]);
        expect(storage.half.get(reused)).toBe(0);
        expect(halfColumn.column[reused]).toBe(0);
    } finally {
        world.dispose();
    }
});
