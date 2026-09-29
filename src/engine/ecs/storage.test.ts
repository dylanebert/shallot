import { expect, test } from "bun:test";
import { f32, field, State, vec2, vec4 } from "./index";
import { WorldField } from "./storage";

test("bulk field writes copy typed rows, preserve other rows, publish scalar-equivalent marks and refuse mismatches", () => {
    const column = new WorldField(field(vec4), 16);
    const storage = column.bind();
    const observed: number[] = [];
    column.observe((eid) => observed.push(eid));
    storage.set(3, 9, 8, 7, 6);
    storage.dirty.fill(0);
    observed.length = 0;
    const eids = new Uint32Array([99, 2, 7, 99]).subarray(1, 3);
    const source = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    storage.write(eids, source);
    expect(Array.from(storage.column.slice(8, 12))).toEqual([1, 2, 3, 4]);
    expect(Array.from(storage.column.slice(28, 32))).toEqual([5, 6, 7, 8]);
    expect(Array.from(storage.column.slice(12, 16))).toEqual([9, 8, 7, 6]);
    expect(storage.dirty[0]).toBe((1 << 2) | (1 << 7));
    expect(observed).toEqual([2, 7]);
    expect(() => storage.write(eids, new Float32Array(7))).toThrow(/write.*length.*4 lanes/);
    expect(() => storage.write(eids, new Uint32Array(8))).toThrow(
        /write.*Float32Array.*Uint32Array/,
    );
    expect(observed).toEqual([2, 7]);
});

test("binding a component freezes its schema against later mutation", () => {
    const Component = { value: field(f32) };
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
        (Component as Record<string, unknown>).added = field(f32);
    }).toThrow();
    expect(() => {
        (Component.value as unknown as { type: unknown }).type = vec2;
    }).toThrow();
    expect(Object.isFrozen(Component.value)).toBe(true);
});

test("schema field access reuses cached columns without repeating schema sorts", () => {
    const Component = { value: field(f32) };
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
        Component.value.set(eid, 11);
        first = Component.value.get(eid);
        Component.value.set(eid, 12);
        second = Component.value.get(eid);
    } finally {
        Array.prototype.sort = originalSort;
        state.dispose();
    }

    expect([first, second]).toEqual([11, 12]);
    expect(sortCalls).toBe(0);
});

test("scalar and vector field writes reach columns without a temporary value array", () => {
    const Scalar = { value: field(f32) };
    const Pair = { value: field(vec2) };
    const Quad = { value: field(vec4) };
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
