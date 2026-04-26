import { test, expect, describe, beforeEach } from "bun:test";
import { State } from "../src";
import { clearRegistry, registerComponent } from "../src/engine/ecs/component";
import { clearRelations } from "../src/engine/ecs/relation";
import { all } from "./helpers/state";

describe("Entity lifecycle invariants", () => {
    const A = { x: [] as number[] };
    const B = { y: [] as number[] };

    beforeEach(() => {
        clearRegistry();
        clearRelations();
        registerComponent("inv-a", A);
        registerComponent("inv-b", B);
    });

    test("destroyed entity absent from all queries", () => {
        const state = new State();
        const eid = state.addEntity();
        state.addComponent(eid, A);
        state.addComponent(eid, B);
        expect(all(state, [A])).toContain(eid);
        expect(all(state, [B])).toContain(eid);

        state.removeEntity(eid);
        expect(all(state, [A])).not.toContain(eid);
        expect(all(state, [B])).not.toContain(eid);
    });

    test("removed component excludes entity from matching queries", () => {
        const state = new State();
        const eid = state.addEntity();
        state.addComponent(eid, A);
        state.addComponent(eid, B);
        expect(all(state, [A, B])).toContain(eid);

        state.removeComponent(eid, B);
        expect(all(state, [A, B])).not.toContain(eid);
        expect(all(state, [A])).toContain(eid);
    });

    test("entity ID reuse does not leak components", () => {
        const state = new State();
        const eid1 = state.addEntity();
        state.addComponent(eid1, A);
        A.x[eid1] = 999;

        state.removeEntity(eid1);
        const eid2 = state.addEntity();

        expect(state.hasComponent(eid2, A)).toBe(false);
        expect(all(state, [A])).not.toContain(eid2);
    });

    test("query results update immediately after mutation", () => {
        const state = new State();
        const eid = state.addEntity();

        state.addComponent(eid, A);
        expect(all(state, [A])).toContain(eid);

        state.removeComponent(eid, A);
        expect(all(state, [A])).not.toContain(eid);

        state.addComponent(eid, A);
        expect(all(state, [A])).toContain(eid);

        state.removeEntity(eid);
        expect(all(state, [A])).not.toContain(eid);
    });
});
