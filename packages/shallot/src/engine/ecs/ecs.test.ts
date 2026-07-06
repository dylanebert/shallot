import { beforeEach, describe, expect, test } from "bun:test";
import { build, type Plugin, State, sparse, u32 } from "../..";
import { clear, idOf, register } from "./core";

const Health = { current: [] as number[], max: [] as number[] };
const Score = { value: [] as number[] };

const HealthPlugin: Plugin = {
    name: "Health",
    components: { Health },
    traits: { Health: { defaults: () => ({ current: 100, max: 100 }) } },
};

const ScorePlugin: Plugin = {
    name: "Score",
    components: { Score },
};

describe("state.add / remove / destroy", () => {
    beforeEach(() => {
        clear();
    });

    test("state.add applies defaults to bare typed-array fields", async () => {
        const { state } = await build({ plugins: [HealthPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Health);

        expect(Health.current[eid]).toBe(100);
        expect(Health.max[eid]).toBe(100);
    });

    test("state.add no-op when component is already attached", async () => {
        const { state } = await build({ plugins: [HealthPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Health);
        Health.current[eid] = 42;
        state.add(eid, Health); // second add is a no-op; should NOT reapply defaults

        expect(Health.current[eid]).toBe(42);
    });

    test("state.remove detaches the component cleanly", async () => {
        const { state } = await build({ plugins: [HealthPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Health);
        expect(state.has(eid, Health)).toBe(true);

        state.remove(eid, Health);
        expect(state.has(eid, Health)).toBe(false);
    });

    test("state.destroy clears every attached component", async () => {
        const { state } = await build({
            plugins: [HealthPlugin, ScorePlugin],
            defaults: false,
        });

        const eid = state.create();
        state.add(eid, Health);
        state.add(eid, Score);

        state.destroy(eid);

        expect(state.exists(eid)).toBe(false);
    });
});

// the component-identity half of the reload-safety tier (testing.md "Reload tier"): a module reload
// recreates the component object; these assert the id + membership + query + storage all re-attach by
// name, not by object identity. The plugin-swap half lives in app/plugin.test.ts "swap (hot reload)".
describe("stable component ids", () => {
    beforeEach(() => {
        clear();
    });

    test("re-registering a name reuses the id and the storage", () => {
        const A = { value: sparse(u32) };
        register("widget", A);
        const id = idOf(A);

        // reload: a fresh object under the same name
        const B = { value: sparse(u32) };
        register("widget", B);

        expect(idOf(B)).toBe(id); // same id, minted by name
        expect(B.value).toBe(A.value); // B adopts A's store — runtime data survives
    });

    test("distinct names get distinct ids; an unregistered component auto-mints a stable one", () => {
        const A = { value: sparse(u32) };
        const B = { value: sparse(u32) };
        register("alpha", A);
        register("beta", B);
        expect(idOf(A)).not.toBe(idOf(B));

        const bare = { tag: sparse(u32) }; // never registered
        expect(idOf(bare)).toBe(idOf(bare)); // stable across calls
        expect(idOf(bare)).not.toBe(idOf(A));
    });

    test("a reloaded handle resolves the same membership, query, and data", () => {
        const A = { value: sparse(u32) };
        register("gadget", A);

        const state = new State();
        const eid = state.create();
        state.add(eid, A);
        A.value.set(eid, 7);
        expect([...state.query([A])]).toContain(eid);

        // reload: fresh object, same name, against the live State
        const B = { value: sparse(u32) };
        register("gadget", B);

        expect(state.has(eid, B)).toBe(true); // membership keyed by id
        expect(B.value.get(eid)).toBe(7); // storage reused
        expect([...state.query([B])]).toContain(eid); // resolves the populated query
    });

    test("a held stale handle resolves the re-registered traits", () => {
        const A = { value: sparse(u32) };
        register("relic", A, { defaults: () => ({ value: 1 }) });

        // reload: fresh object, new defaults under the same name
        const B = { value: sparse(u32) };
        register("relic", B, { defaults: () => ({ value: 2 }) });

        const state = new State();
        const eid = state.create();
        state.add(eid, A); // a stale handle held across the reload
        expect(B.value.get(eid)).toBe(2); // current defaults apply, not the pre-reload registration's

        // a second reload: the same held handle still resolves the newest registration
        const C = { value: sparse(u32) };
        register("relic", C, { defaults: () => ({ value: 3 }) });
        const second = state.create();
        state.add(second, A);
        expect(C.value.get(second)).toBe(3);
    });

    test("the stamped id is invisible to a field walk", () => {
        const A = { value: sparse(u32) };
        register("hidden", A);
        idOf(A);
        expect(Object.keys(A)).toEqual(["value"]); // Symbol-keyed id excluded
    });
});

describe("trait excludes", () => {
    const Slab = { value: [] as number[] };
    const Body = { value: [] as number[] };
    const Aux = { value: [] as number[] };

    const ExclusionPlugin: Plugin = {
        name: "Exclusion",
        components: { Slab, Body, Aux },
        traits: { Body: { excludes: [Slab] } },
    };

    beforeEach(() => {
        clear();
    });

    test("state.add throws when adding a component excluded by an existing one", async () => {
        const { state } = await build({ plugins: [ExclusionPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Body);

        expect(() => state.add(eid, Slab)).toThrow(/excluded by "body"/);
        expect(state.has(eid, Slab)).toBe(false);
    });

    test("exclusion is symmetric — declaring one direction enforces both", async () => {
        const { state } = await build({ plugins: [ExclusionPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Slab);

        expect(() => state.add(eid, Body)).toThrow(/excluded by "slab"/);
        expect(state.has(eid, Body)).toBe(false);
    });

    test("unrelated components attach freely", async () => {
        const { state } = await build({ plugins: [ExclusionPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Body);
        state.add(eid, Aux);

        expect(state.has(eid, Body)).toBe(true);
        expect(state.has(eid, Aux)).toBe(true);
    });

    test("after removing the excluder, the excluded component can attach", async () => {
        const { state } = await build({ plugins: [ExclusionPlugin], defaults: false });

        const eid = state.create();
        state.add(eid, Body);
        state.remove(eid, Body);
        state.add(eid, Slab);

        expect(state.has(eid, Slab)).toBe(true);
    });
});
