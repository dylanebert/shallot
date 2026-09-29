import { expect, test } from "bun:test";
import { f32, State, sparse, vec2, vec4 } from "./index";
import { WorldField } from "./storage";

test("scalar and vector field writes reach columns without a temporary value array", () => {
    const Scalar = { value: sparse(f32) };
    const Pair = { value: sparse(vec2) };
    const Quad = { value: sparse(vec4) };
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
