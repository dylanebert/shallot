import { expect, test } from "bun:test";
import type { Pair, Quad, Single } from "./component";
import { f32, State, vec2, vec4 } from "./index";
import { WorldField } from "./storage";

test("bulk field writes copy typed rows, preserve other rows, publish scalar-equivalent marks and refuse mismatches", () => {
    const column = new WorldField(vec4, 16);
    const storage = column.bind();
    storage.set(3, 9, 8, 7, 6);
    column.dirty.fill(0);
    const eids = new Uint32Array([99, 2, 7, 99]).subarray(1, 3);
    const source = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    storage.write(eids, source);
    expect(Array.from(storage.column.slice(8, 12))).toEqual([1, 2, 3, 4]);
    expect(Array.from(storage.column.slice(28, 32))).toEqual([5, 6, 7, 8]);
    expect(Array.from(storage.column.slice(12, 16))).toEqual([9, 8, 7, 6]);
    expect(column.dirty[0]).toBe((1 << 2) | (1 << 7));
    expect(Array.from(column.dirty)).toEqual([(1 << 2) | (1 << 7)]);
    expect(() => storage.write(eids, new Float32Array(7))).toThrow(/write.*length.*4 lanes/);
    expect(() => storage.write(eids, new Uint32Array(8))).toThrow(
        /write.*Float32Array.*Uint32Array/,
    );
    expect(Array.from(column.dirty)).toEqual([(1 << 2) | (1 << 7)]);
});

test("binding a component freezes its schema against later mutation", () => {
    const Component = { value: f32 };
    const state = new State();
    const second = new State();
    state.registry.register("FrozenComponent", Component);
    state.of(Component);
    second.registry.register("FrozenComponent", Component);
    second.of(Component);
    state.dispose();
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
    const state = new State();
    const eid = state.create();
    state.of(Component);
    const originalSort = Array.prototype.sort;
    let sortCalls = 0;
    let first = 0;
    let second = 0;

    Array.prototype.sort = function <T>(this: T[], compareFn?: (a: T, b: T) => number): T[] {
        sortCalls++;
        return originalSort.call(this, compareFn);
    };
    try {
        state.of(Component).value.set(eid, 11);
        first = state.of(Component).value.get(eid);
        state.of(Component).value.set(eid, 12);
        second = state.of(Component).value.get(eid);
    } finally {
        Array.prototype.sort = originalSort;
        state.dispose();
    }

    expect([first, second]).toEqual([11, 12]);
    expect(sortCalls).toBe(0);
});

test("resolving another world's storage cannot redirect retained or newly resolved fields", () => {
    const Component = { value: f32 };
    const a = new State();
    const b = new State();
    const eid = a.create();
    expect(b.create()).toBe(eid);
    const av = a.of(Component).value;
    av.set(eid, 11);
    b.of(Component).value.set(eid, 22);
    expect(av.get(eid)).toBe(11);
    expect(a.of(Component).value.get(eid)).toBe(11);
    expect(b.of(Component).value.get(eid)).toBe(22);
    let duringA: number[] = [];
    a.addSystem({
        update(state) {
            duringA = [b.of(Component).value.get(eid), state.of(Component).value.get(eid)];
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
    // @ts-expect-error resolve entity data through state.of(Component), not its declaration.
    Component.value.get(1);
}
void schemaBoundTypeControl;

test("scalar and vector field writes reach columns without a temporary value array", () => {
    const Scalar = { value: f32 };
    const Pair = { value: vec2 };
    const Quad = { value: vec4 };
    const state = new State();
    const eid = state.create();
    const scalar = state.of(Scalar).value;
    const pair = state.of(Pair).value;
    const quad = state.of(Quad).value;
    const original = WorldField.prototype.set;
    const firstArguments: unknown[] = [];

    WorldField.prototype.set = function (this: WorldField, ...args: Parameters<WorldField["set"]>) {
        firstArguments.push(args[1]);
        return Reflect.apply(original, this, args);
    };
    try {
        scalar.set(eid, 1);
        pair.set(eid, 2, 3);
        quad.set(eid, 4, 5, 6, 7);
    } finally {
        WorldField.prototype.set = original;
        state.dispose();
    }

    expect(firstArguments.map(Array.isArray)).toEqual([false, false, false]);
});

for (const type of [f32, vec2, vec4] as const) {
    test(`${type.name} retained handles publish exactly the written eids across growth and raw writes require markChanged`, () => {
        const column = new WorldField(type, 16);
        const handle = column.bind() as Single | Pair | Quad;
        const lane = "x" in handle ? handle.x : handle;
        const replaced = handle.column;
        // Retain the handles, never the raw array, across growth.
        lane.set(40, 5);
        expect(handle.column).not.toBe(replaced);
        expect(lane.column).toBe(handle.column);
        column.dirty.fill(0);
        handle.set(2, 1, 2, 3, 4);
        lane.set(7, 8);
        handle.write(new Uint32Array([33]), new Float32Array(type.lanes).fill(9));
        lane.write(new Uint32Array([35]), new Float32Array([10]));
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
