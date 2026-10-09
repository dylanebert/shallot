import { expect, test } from "bun:test";
import type { ScalarField, Vector2Field, Vector4Field } from "./component";
import { component } from "./component";
import { entity, f16, f32, vec2, vec4, World } from "./index";
import { WorldField } from "./storage";

test("bulk field writes copy typed rows, preserve other rows, publish scalar-equivalent marks and refuse mismatches", () => {
    const column = new WorldField(vec4, 16);
    const storage = column.bind();
    storage.set(3, 9, 8, 7, 6);
    column.dirty.fill(0);
    const eids = new Uint32Array([99, 2, 7, 99]).subarray(1, 3);
    const source = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    storage.writeEncoded(eids, source);
    expect(Array.from(storage.column.slice(8, 12))).toEqual([1, 2, 3, 4]);
    expect(Array.from(storage.column.slice(28, 32))).toEqual([5, 6, 7, 8]);
    expect(Array.from(storage.column.slice(12, 16))).toEqual([9, 8, 7, 6]);
    expect(column.dirty[0]).toBe((1 << 2) | (1 << 7));
    expect(Array.from(column.dirty)).toEqual([(1 << 2) | (1 << 7)]);
    expect(() => storage.writeEncoded(eids, new Float32Array(7))).toThrow(/write.*length.*4 lanes/);
    expect(() => storage.writeEncoded(eids, new Uint32Array(8))).toThrow(
        /write.*Float32Array.*Uint32Array/,
    );
    expect(Array.from(column.dirty)).toEqual([(1 << 2) | (1 << 7)]);
});

test("binding a component freezes its schema against later mutation", () => {
    const Component = { value: f32 };
    const world = new World();
    const second = new World();
    world.registry.register(component("FrozenComponent", Component));
    world.storage(Component);
    second.registry.register(component("FrozenComponent", Component));
    second.storage(Component);
    world.dispose();
    second.dispose();

    expect(Object.isFrozen(Component)).toBe(true);
    expect(() => {
        (Component as Record<string, unknown>).added = f32;
    }).toThrow();
    expect(() => {
        (Component.value as unknown as { lanes: number }).lanes = 2;
    }).toThrow();
    expect(Object.isFrozen(Component.value)).toBe(true);
});

test("schema field access reuses cached columns without repeating schema sorts", () => {
    const Component = { value: f32 };
    const world = new World();
    const eid = world.create();
    world.storage(Component);
    const originalSort = Array.prototype.sort;
    let sortCalls = 0;
    let first = 0;
    let second = 0;

    Array.prototype.sort = function <T>(this: T[], compareFn?: (a: T, b: T) => number): T[] {
        sortCalls++;
        return originalSort.call(this, compareFn);
    };
    try {
        world.storage(Component).value.set(eid, 11);
        first = world.storage(Component).value.get(eid);
        world.storage(Component).value.set(eid, 12);
        second = world.storage(Component).value.get(eid);
    } finally {
        Array.prototype.sort = originalSort;
        world.dispose();
    }

    expect([first, second]).toEqual([11, 12]);
    expect(sortCalls).toBe(0);
});

test("resolving another world's storage cannot redirect retained or newly resolved fields", () => {
    const Component = { value: f32 };
    const a = new World();
    const b = new World();
    const eid = a.create();
    expect(b.create()).toBe(eid);
    const av = a.storage(Component).value;
    av.set(eid, 11);
    b.storage(Component).value.set(eid, 22);
    expect(av.get(eid)).toBe(11);
    expect(a.storage(Component).value.get(eid)).toBe(11);
    expect(b.storage(Component).value.get(eid)).toBe(22);
    let duringA: number[] = [];
    a.addSystem({
        update(world) {
            duringA = [
                b.storage(Component).value.get(eid),
                world.storage(Component).value.get(eid),
            ];
        },
    });
    a.step();
    expect(duringA).toEqual([22, 11]);
    expect("get" in Component.value).toBe(false);
    a.dispose();
    b.dispose();
});

// These declarations have metadata only; tsc must reject deleted schema-bound access.
function schemaBoundTypeControl(): void {
    const Component = { value: f32 };
    // @ts-expect-error resolve entity data through state.storage(Component), not its declaration.
    Component.value.get(1);
}
void schemaBoundTypeControl;

test("scalar and vector setters write their arguments to the expected columns and publish changes", () => {
    const eid = 5;
    const scalarColumn = new WorldField(f32, 16);
    const pairColumn = new WorldField(vec2, 16);
    const quadColumn = new WorldField(vec4, 16);
    const scalar = scalarColumn.bind();
    const pair = pairColumn.bind();
    const quad = quadColumn.bind();
    scalarColumn.dirty.fill(0);
    pairColumn.dirty.fill(0);
    quadColumn.dirty.fill(0);

    scalar.set(eid, 1);
    pair.set(eid, 2, 3);
    quad.set(eid, 4, 5, 6, 7);

    expect(Array.from(scalar.column.slice(eid, eid + 1))).toEqual([1]);
    expect(Array.from(pair.column.slice(eid * 2, eid * 2 + 2))).toEqual([2, 3]);
    expect(Array.from(quad.column.slice(eid * 4, eid * 4 + 4))).toEqual([4, 5, 6, 7]);
    expect(Array.from(scalarColumn.dirty)).toEqual([1 << eid]);
    expect(Array.from(pairColumn.dirty)).toEqual([1 << eid]);
    expect(Array.from(quadColumn.dirty)).toEqual([1 << eid]);
});

test("field getters return zero beyond plain, encoded and entity column capacity", () => {
    const world = new World();
    const scalar = new WorldField(f32, 16).bind();
    const pair = new WorldField(vec2, 16).bind();
    const half = new WorldField(f16, 16).bind();
    const target = new WorldField(entity, 16, world).bind();

    expect(scalar.get(16)).toBe(0);
    expect(pair.x.get(16)).toBe(0);
    expect(half.get(16)).toBe(0);
    expect(target.get(16)).toBe(0);
    world.dispose();
});

for (const type of [f32, vec2, vec4] as const) {
    test(`${type.name} retained handles publish exactly the written eids across growth and raw writes require markChanged`, () => {
        const column = new WorldField(type, 16);
        const handle = column.bind() as ScalarField | Vector2Field | Vector4Field;
        const lane = "x" in handle ? handle.x : handle;
        const replaced = handle.column;
        // Retain the handles, never the raw array, across growth.
        lane.set(40, 5);
        expect(handle.column).not.toBe(replaced);
        expect(lane.column).toBe(handle.column);
        column.dirty.fill(0);
        handle.set(2, 1, 2, 3, 4);
        lane.set(7, 8);
        handle.writeEncoded(new Uint32Array([33]), new Float32Array(type.lanes).fill(9));
        lane.writeEncoded(new Uint32Array([35]), new Float32Array([10]));
        handle.column[41 * type.lanes] = 11;
        handle.markChanged(41);
        lane.column[43 * type.lanes] = 12;
        lane.markChanged(43);
        handle.column[45 * type.lanes] = 13;
        expect(Array.from(column.dirty)).toEqual([
            (1 << 2) | (1 << 7),
            (1 << 1) | (1 << 3) | (1 << 9) | (1 << 11),
        ]);
        column.dirty.fill(0);
        lane.column[45 * type.lanes] = 14;
        expect(Array.from(column.dirty)).toEqual([0, 0]);
        expect(lane.get(45)).toBe(14);
    });
}

test("restore marks changes confined to each vector lane and a scalar lane", () => {
    const State = component("RestoreStrideChanges", { vector: vec4, scalar: f32 });
    const world = new World();
    world.registry.register(State);
    const eid = world.create();
    world.add(eid, State);
    const { vector, scalar } = world.storage(State);
    vector.set(eid, 1, 2, 3, 4);
    scalar.set(eid, 5);
    world.clearChanges();
    const image = world.snapshot();
    const vectorField = world.fieldStorage(State, "vector");
    const scalarField = world.fieldStorage(State, "scalar");

    for (let lane = 0; lane < 4; lane++) {
        vector.column[eid * 4 + lane] = 10 + lane;
        world.clearChanges();
        world.restore(image);
        expect(vectorField.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
        expect(scalarField.dirty[eid >>> 5] & (1 << (eid & 31))).toBe(0);
        world.clearChanges();
    }

    scalar.column[eid] = 6;
    world.clearChanges();
    world.restore(image);
    expect(scalarField.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
    expect(vectorField.dirty[eid >>> 5] & (1 << (eid & 31))).toBe(0);
    world.dispose();
});

test("restore publishes changed field values but leaves equal values unmarked after clear", () => {
    const State = component("RestoreChangedField", { value: f32 });
    const world = new World();
    world.registry.register(State);
    const changed = world.create();
    const equal = world.create();
    world.add(changed, State);
    world.add(equal, State);
    const value = world.storage(State).value;
    value.set(changed, 3);
    value.set(equal, 5);
    world.clearChanges();
    const image = world.snapshot();

    value.set(changed, 7);
    world.clearChanges();
    world.restore(image);

    const field = world.fieldStorage(State, "value");
    expect(field.dirty[changed >>> 5] & (1 << (changed & 31))).not.toBe(0);
    expect(field.dirty[equal >>> 5] & (1 << (equal & 31))).toBe(0);
});

test("restore republishes a lane marked in the image even when its value is equal", () => {
    const State = component("RestorePendingField", { value: f32 });
    const world = new World();
    world.registry.register(State);
    const eid = world.create();
    world.add(eid, State);
    const value = world.storage(State).value;
    value.set(eid, 3);
    const image = world.snapshot();
    world.clearChanges();

    world.restore(image);

    const field = world.fieldStorage(State, "value");
    expect(value.get(eid)).toBe(3);
    expect(field.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
});

test("restore zeroes and publishes a lane written by an entity created after the image", () => {
    const State = component("RestorePastImageField", { value: f32 });
    const world = new World();
    world.registry.register(State);
    const saved = world.create();
    world.add(saved, State);
    const value = world.storage(State).value;
    value.set(saved, 3);
    const image = world.snapshot();

    const added = world.create();
    world.add(added, State);
    value.set(added, 9);
    world.clearChanges();
    world.restore(image);

    const field = world.fieldStorage(State, "value");
    expect(value.get(added)).toBe(0);
    expect(field.dirty[added >>> 5] & (1 << (added & 31))).not.toBe(0);
});

test("restore publishes a raw-bit change between negative and positive zero", () => {
    const State = component("RestoreSignedZeroField", { value: f32 });
    const world = new World();
    world.registry.register(State);
    const eid = world.create();
    world.add(eid, State);
    const value = world.storage(State).value;
    value.set(eid, 0);
    const image = world.snapshot();

    value.set(eid, -0);
    world.clearChanges();
    world.restore(image);

    const field = world.fieldStorage(State, "value");
    expect(Object.is(value.column[eid], 0)).toBe(true);
    expect(field.dirty[eid >>> 5] & (1 << (eid & 31))).not.toBe(0);
});
