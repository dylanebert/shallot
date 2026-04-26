import { test, expect, describe, beforeEach } from "bun:test";
import { State } from "../src";
import { clearRegistry } from "../src/engine/ecs/component";
import { Gizmos, GizmosPlugin } from "../src/extras/gizmos";
import { RenderPlugin } from "../src/standard/render";
import { count, all } from "./helpers/state";

describe("Gizmos", () => {
    let state: State;

    beforeEach(() => {
        clearRegistry();
        state = new State();
        state.register(RenderPlugin);
        state.register(GizmosPlugin);
    });

    test("defaults grid to 1", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Gizmos);

        expect(Gizmos.grid[eid]).toBe(1);
    });

    test("grid value can be changed to 0", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Gizmos);

        Gizmos.grid[eid] = 0;

        expect(Gizmos.grid[eid]).toBe(0);
    });

    test("entity with Gizmos is queryable", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Gizmos);

        expect(count(state, [Gizmos])).toBe(1);
        expect(all(state, [Gizmos])).toEqual([eid]);
    });

    test("removing component clears entity from query", () => {
        const eid = state.addEntity();
        state.addComponent(eid, Gizmos);
        state.removeComponent(eid, Gizmos);

        expect(count(state, [Gizmos])).toBe(0);
    });

    test("multiple entities with different grid values", () => {
        const a = state.addEntity();
        state.addComponent(a, Gizmos);

        const b = state.addEntity();
        state.addComponent(b, Gizmos);
        Gizmos.grid[b] = 0;

        expect(Gizmos.grid[a]).toBe(1);
        expect(Gizmos.grid[b]).toBe(0);
        expect(count(state, [Gizmos])).toBe(2);
    });

    test("plugin registers without error", () => {
        clearRegistry();
        const s = new State();

        expect(() => {
            s.register(RenderPlugin);
            s.register(GizmosPlugin);
        }).not.toThrow();
    });
});
