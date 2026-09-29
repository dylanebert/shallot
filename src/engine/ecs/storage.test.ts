import { expect, test } from "bun:test";
import { f32, field, State, vec2, vec4 } from "./index";
import { WorldField } from "./storage";

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
