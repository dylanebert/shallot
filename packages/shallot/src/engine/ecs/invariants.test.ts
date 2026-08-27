import { beforeEach, describe, expect, test } from "bun:test";
import { build, type Plugin, State } from "../..";
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

    // ecs.md: the membership generation count is fixed for a State's life — build assigns every
    // component its bit up front. A post-build component that would require a new generation
    // (the 32nd, filling generation 0's 31 bits) must be refused rather than silently outsizing
    // the fixed GPU mirror SlabPlugin allocates at warm. Bare-marker adds that fit in an existing
    // generation are sanctioned; only growing the generation count past build is refused.
    test("post-freeze component add that grows membership generation past build-fixed count is refused", () => {
        const state = new State();
        const eid = state.create();
        // fill generation 0 (31 bits)
        const comps = Array.from({ length: 31 }, () => ({ x: [] as number[] }));
        for (const c of comps) state.add(eid, c);
        expect(state.membership.generations).toBe(1);
        // freeze the generation count (as warm does via allocMembership)
        state.membership.freeze();
        // a 32nd component would require generation 1 — refused
        const comp32 = { x: [] as number[] };
        expect(() => state.add(eid, comp32)).toThrow();
        expect(state.membership.generations).toBe(1);
    });

    // The class doc on RegisteredQuery narrows safe mutation during iteration to the current-eid case:
    // a swap-remove of the current eid overwrites an already-visited slot, so every original member is
    // still visited exactly once. Removing a *not-yet-visited* eid mid-iteration is not safe — the
    // swap-remove moves the tail into the unvisited slot, so the tail member is visited twice and the
    // removed one skipped. This arm pins that narrowed contract: the double-visit is the expected
    // behavior, not a bug to fix (fixing it would break the zero-alloc iterator or the current-eid case).
    test("removing a not-yet-visited eid mid-iteration double-visits the tail and skips the removed", () => {
        const state = new State();
        const eids: number[] = [];
        for (let i = 0; i < 5; i++) {
            const eid = state.create();
            state.add(eid, A);
            eids.push(eid);
        }

        const visited: number[] = [];
        for (const eid of state.query([A])) {
            visited.push(eid);
            // remove a not-yet-visited eid (eids[2]) on the first iteration
            if (eid === eids[0]) state.remove(eids[2], A);
        }

        // the removed eid is skipped (swap-remove overwrites its slot with the tail)
        expect(visited).not.toContain(eids[2]);
        // the tail member (eids[4]) is visited twice: once at the swapped-in slot, once at its stale copy
        const counts = new Map<number, number>();
        for (const v of visited) counts.set(v, (counts.get(v) ?? 0) + 1);
        expect(counts.get(eids[4])).toBe(2);
        // every other visited member appears exactly once
        for (const [eid, c] of counts) if (eid !== eids[4]) expect(c).toBe(1);
    });

    // The class doc on RegisteredQuery covers the add-during-iteration case: adding a *new* matching
    // entity mid-iteration appends it at `_dense[_count++]` past the snapshot, so it is not visited
    // this loop. Every original member is still visited exactly once — the new member simply appears
    // on the next query call. This arm pins that characterization: the new entity is absent from the
    // visited list, and no original member is skipped or double-visited.
    test("adding a new matching entity mid-iteration does not visit it this loop", () => {
        const state = new State();
        const eids: number[] = [];
        for (let i = 0; i < 4; i++) {
            const eid = state.create();
            state.add(eid, A);
            eids.push(eid);
        }

        const visited: number[] = [];
        let newEid = -1;
        for (const eid of state.query([A])) {
            visited.push(eid);
            if (eid === eids[0]) {
                newEid = state.create();
                state.add(newEid, A);
            }
        }

        // the new entity is appended past the snapshot count — not visited this loop
        expect(visited).not.toContain(newEid);
        // every original member is visited exactly once
        expect(visited).toEqual(eids);
        // the new entity appears on the next query call
        expect([...state.query([A])]).toContain(newEid);
    });

    // The membership generation freeze is an engine-owned session invariant: `app/index.ts` calls
    // `state.membership.freeze()` before `warmPlugins`, not a standard plugin's `warm`. This arm
    // exercises the invariant through the real build path — after `build()` returns, the membership
    // is frozen, so a component that would require a new generation is refused.
    test("build freezes membership generations, refusing a post-build 32nd component", async () => {
        clear();
        const comps = Array.from({ length: 31 }, (_, i) => ({ [`c${i}`]: [] as number[] }));
        const components: Record<string, any> = {};
        for (let i = 0; i < 31; i++) components[`c${i}`] = comps[i];
        const P: Plugin = { name: "P31", components };
        const { state } = await build({ plugins: [P], defaults: false });

        const eid = state.create();
        // add all 31 registered components (generation 0, 31 bits)
        for (let i = 0; i < 31; i++) state.add(eid, comps[i]);
        expect(state.membership.generations).toBe(1);
        // a 32nd component would require generation 1 — refused because build froze the count
        const comp32 = { z: [] as number[] };
        expect(() => state.add(eid, comp32)).toThrow();
        expect(state.membership.generations).toBe(1);
    });
});
