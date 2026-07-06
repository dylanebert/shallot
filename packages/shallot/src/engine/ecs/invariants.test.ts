import { beforeEach, describe, expect, test } from "bun:test";
import { State } from "../..";
import { clear, register } from "./core";
import { not } from "./query";

describe("Entity lifecycle invariants", () => {
    const A = { x: [] as number[] };
    const B = { y: [] as number[] };

    beforeEach(() => {
        clear();
        register("inv-a", A);
        register("inv-b", B);
    });

    test("destroyed entity absent from all queries", () => {
        const state = new State();
        const eid = state.create();
        state.add(eid, A);
        state.add(eid, B);
        expect([...state.query([A])]).toContain(eid);
        expect([...state.query([B])]).toContain(eid);

        state.destroy(eid);
        expect([...state.query([A])]).not.toContain(eid);
        expect([...state.query([B])]).not.toContain(eid);
    });

    test("removed component excludes entity from matching queries", () => {
        const state = new State();
        const eid = state.create();
        state.add(eid, A);
        state.add(eid, B);
        expect([...state.query([A, B])]).toContain(eid);

        state.remove(eid, B);
        expect([...state.query([A, B])]).not.toContain(eid);
        expect([...state.query([A])]).toContain(eid);
    });

    test("entity ID reuse does not leak components", () => {
        const state = new State();
        const eid1 = state.create();
        state.add(eid1, A);
        A.x[eid1] = 999;

        state.destroy(eid1);
        const eid2 = state.create();

        expect(state.has(eid2, A)).toBe(false);
        expect([...state.query([A])]).not.toContain(eid2);
    });

    test("query results update immediately after mutation", () => {
        const state = new State();
        const eid = state.create();

        state.add(eid, A);
        expect([...state.query([A])]).toContain(eid);

        state.remove(eid, A);
        expect([...state.query([A])]).not.toContain(eid);

        state.add(eid, A);
        expect([...state.query([A])]).toContain(eid);

        state.destroy(eid);
        expect([...state.query([A])]).not.toContain(eid);
    });

    // The iteration-during-mutation contract orbit/gltf rely on: adding a marker to the current eid
    // removes it from the [A, not(B)] query mid-loop (swap-remove + decremented count), yet every
    // original member is still visited exactly once. The iterator snapshots count at start and the
    // swap only overwrites already-visited slots, so the tail values are read at their original
    // indices. Pins the visited-eid multiset so a zero-alloc iterator rewrite can't change it.
    test("marker added during iteration visits every original member exactly once", () => {
        const state = new State();
        const eids: number[] = [];
        for (let i = 0; i < 8; i++) {
            const eid = state.create();
            state.add(eid, A);
            eids.push(eid);
        }

        const visited: number[] = [];
        for (const eid of state.query([A, not(B)])) {
            visited.push(eid);
            state.add(eid, B); // removes eid from this query (now has B)
        }

        expect([...visited].sort((a, b) => a - b)).toEqual([...eids].sort((a, b) => a - b));
        for (const eid of eids) expect(state.has(eid, B)).toBe(true);
        expect([...state.query([A, not(B)])]).toEqual([]);
    });

    // The pooled iterator returns its state object to the query's free-list on loop completion, so a
    // second loop over the same query reuses the same iterator object (proves zero per-loop alloc).
    test("sequential loops over one query reuse the iterator object", () => {
        const state = new State();
        for (let i = 0; i < 4; i++) state.add(state.create(), A);
        const q = state.query([A]);

        const it1 = q[Symbol.iterator]();
        while (!it1.next().done) {} // drive to completion → reclaimed to the pool
        const it2 = q[Symbol.iterator]();
        expect(it2).toBe(it1); // popped the same pooled state
    });

    // Nested iteration of the same query must borrow distinct states (the inner loop pops a second
    // one while the outer's is checked out), or the shared index would corrupt the cross-product.
    test("nested iteration over one query borrows distinct iterators and is correct", () => {
        const state = new State();
        const eids: number[] = [];
        for (let i = 0; i < 3; i++) {
            const eid = state.create();
            state.add(eid, A);
            eids.push(eid);
        }
        const q = state.query([A]);

        const pairs: Array<[number, number]> = [];
        for (const a of q) for (const b of q) pairs.push([a, b]);

        expect(pairs.length).toBe(eids.length * eids.length);
        for (const a of eids) for (const b of eids) expect(pairs).toContainEqual([a, b]);
    });

    // An early break must still reclaim the iterator (for…of calls return()), so a later loop reuses it.
    test("breaking out of a loop reclaims the iterator", () => {
        const state = new State();
        for (let i = 0; i < 5; i++) state.add(state.create(), A);
        const q = state.query([A]);

        const first = q[Symbol.iterator]();
        let n = 0;
        for (const _ of { [Symbol.iterator]: () => first }) if (++n === 2) break;
        const second = q[Symbol.iterator]();
        expect(second).toBe(first); // return() pushed it back on break
    });
});
