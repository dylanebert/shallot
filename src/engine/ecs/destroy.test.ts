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

const Removed = component("remove-clears-held-values", { value: f32, vector: vec4 });
const Retained = component("remove-keeps-other-values", { value: f32 });
const RestoreX = component("destroy-restore-owner-x", { scalar: f32, vector: vec4 });
const RestoreY = component("destroy-restore-owner-y", { scalar: f32, vector: vec4 });

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

test("restore rebuilds owners when another component reuses a restored mask bit", () => {
    const world = new World();
    try {
        const x = world.storage(RestoreX);
        const y = world.storage(RestoreY);
        const eid = world.create();
        const beforeX = world.snapshot();
        world.add(eid, RestoreX, { scalar: 13, vector: [1, 2, 3, 4] });
        const withX = world.snapshot();

        world.restore(beforeX);
        const firstYHolder = world.create();
        world.add(firstYHolder, RestoreY, { scalar: 29, vector: [5, 6, 7, 8] });
        world.restore(withX);
        const yHolder = world.create();
        expect(yHolder).toBe(firstYHolder);
        world.add(yHolder, RestoreY, { scalar: 31, vector: [9, 10, 11, 12] });
        const xScalar = world.fieldStorage(RestoreX, "scalar");
        const xVector = world.fieldStorage(RestoreX, "vector");
        xScalar.dirty.fill(0);
        xVector.dirty.fill(0);
        const yScalar = world.fieldStorage(RestoreY, "scalar");
        const yVector = world.fieldStorage(RestoreY, "vector");
        yScalar.dirty.fill(0);
        yVector.dirty.fill(0);

        world.destroy(eid);

        expect(x.scalar.get(eid)).toBe(0);
        expect(xScalar.column[eid]).toBe(0);
        expect(Array.from(x.vector.read(eid, new Float32Array(4)))).toEqual([0, 0, 0, 0]);
        expect(Array.from(xVector.column.slice(eid * 4, eid * 4 + 4))).toEqual([0, 0, 0, 0]);
        expect(xScalar.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        expect(xVector.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        expect(y.scalar.get(yHolder)).toBe(31);
        expect(yScalar.column[yHolder]).toBe(31);
        expect(Array.from(y.vector.read(yHolder, new Float32Array(4)))).toEqual([9, 10, 11, 12]);
        expect(Array.from(yVector.column.slice(yHolder * 4, yHolder * 4 + 4))).toEqual([
            9, 10, 11, 12,
        ]);
    } finally {
        world.dispose();
    }
});

test("held components survive mask generations, reload and restore", () => {
    const world = new World();
    try {
        const components = Array.from({ length: 33 }, (_, index) =>
            component(`destroy-mask-generation-${index}`, { value: f32 }),
        );
        const Tag = component("destroy-mask-generation-tag", {});
        const Original = component("destroy-held-hot-reload", { value: f32 });
        const Reloaded = component("destroy-held-hot-reload", { value: f32 });
        const eid = world.create();
        for (let index = 0; index < components.length; index++) {
            world.add(eid, components[index], { value: index + 1 });
        }
        world.add(eid, Tag);
        world.add(eid, Original, { value: 99 });
        expect(world.storage(Reloaded).value.get(eid)).toBe(99);
        const snapshot = world.snapshot();

        world.remove(eid, components[0]);
        world.restore(snapshot);
        world.destroy(eid);

        for (const Component of components) {
            expect(world.storage(Component).value.get(eid)).toBe(0);
        }
        expect(world.storage(Reloaded).value.get(eid)).toBe(0);
    } finally {
        world.dispose();
    }
});

test("remove clears only its component after membership observers read the old values", () => {
    const world = new World();
    try {
        const eid = world.create();
        world.add(eid, Removed, { value: 31, vector: [1, 2, 3, 4] });
        world.add(eid, Retained, { value: 47 });
        const removed = world.storage(Removed);
        const retained = world.storage(Retained);
        const valueColumn = world.fieldStorage(Removed, "value");
        const vectorColumn = world.fieldStorage(Removed, "vector");
        const retainedColumn = world.fieldStorage(Retained, "value");
        valueColumn.dirty.fill(0);
        vectorColumn.dirty.fill(0);
        retainedColumn.dirty.fill(0);
        const observed: { value: number; vector: number[] }[] = [];
        world.observeMembership(Removed, (changed, present) => {
            if (!present) {
                observed.push({
                    value: removed.value.get(changed),
                    vector: Array.from(removed.vector.read(changed, new Float32Array(4))),
                });
            }
        });

        world.remove(eid, Removed);

        expect(observed).toEqual([{ value: 31, vector: [1, 2, 3, 4] }]);
        expect(removed.value.get(eid)).toBe(0);
        expect(valueColumn.column[eid]).toBe(0);
        expect(Array.from(removed.vector.read(eid, new Float32Array(4)))).toEqual([0, 0, 0, 0]);
        expect(Array.from(vectorColumn.column.slice(eid * 4, eid * 4 + 4))).toEqual([0, 0, 0, 0]);
        expect(valueColumn.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        expect(vectorColumn.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        expect(retained.value.get(eid)).toBe(47);
        expect(retainedColumn.column[eid]).toBe(47);
        expect(retainedColumn.dirty[eid >>> 5] & (1 << (eid & 31))).toBe(0);
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
