import { expect, test } from "bun:test";
import { entity, f16, f32, u32, vec2, vec3, vec4, World } from "./index";

const Record = {
    scalar: f32,
    count: u32,
    half: f16,
    pair: vec2,
    point: vec3,
    quad: vec4,
    target: entity,
};

test("get returns the values add wrote, vectors at their authored length", () => {
    const world = new World();
    try {
        const target = world.create();
        const eid = world.create();
        world.add(eid, Record, {
            scalar: 3,
            count: 7,
            half: 1.5,
            pair: [4, 5],
            point: [6, 7, 8],
            quad: [9, 10, 11, 12],
            target,
        });
        const values = world.get(eid, Record);
        expect(values).toEqual({
            scalar: 3,
            count: 7,
            half: 1.5,
            pair: [4, 5],
            point: [6, 7, 8],
            quad: [9, 10, 11, 12],
            target,
        });
        const copy = world.create();
        world.add(copy, Record, values);
        expect(world.get(copy, Record)).toEqual(values!);
    } finally {
        world.dispose();
    }
});

test("get reads a destroyed entity field as 0", () => {
    const world = new World();
    try {
        const target = world.create();
        const eid = world.create();
        world.add(eid, Record, { target });
        world.destroy(target);
        expect(world.get(eid, Record)!.target).toBe(0);
    } finally {
        world.dispose();
    }
});

test("get returns a fresh record that does not alias storage", () => {
    const world = new World();
    try {
        const eid = world.create();
        world.add(eid, Record, { pair: [1, 2] });
        const first = world.get(eid, Record)!;
        first.pair[0] = 9;
        expect(world.get(eid, Record)!.pair).toEqual([1, 2]);
        expect(world.get(eid, Record)).not.toBe(world.get(eid, Record));
    } finally {
        world.dispose();
    }
});

test("get is undefined for a component the entity never held or removed", () => {
    const world = new World();
    try {
        const eid = world.create();
        expect(world.get(eid, Record)).toBeUndefined();
        world.add(eid, Record, { scalar: 2 });
        world.remove(eid, Record);
        world.storage(Record).scalar.set(eid, 5);
        expect(world.get(eid, Record)).toBeUndefined();
    } finally {
        world.dispose();
    }
});

test("get publishes no field change", () => {
    const world = new World();
    try {
        const eid = world.create();
        world.add(eid, Record, { scalar: 2 });
        world.clearChanges();
        world.get(eid, Record);
        for (const entry of world.storageEntries())
            for (const field of entry.fields.values())
                expect(field.dirty.some(Boolean)).toBe(false);
    } finally {
        world.dispose();
    }
});
