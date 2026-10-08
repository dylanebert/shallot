import { expect, spyOn, test } from "bun:test";
import { component } from "./component";
import { RegisteredQuery } from "./query";
import { World } from "./world";

function subject() {
    const query = new RegisteredQuery([]);
    for (const eid of [4, 1, 3, 2]) query.add(eid);
    return query;
}

test("query iteration sorts ascending only after membership changes, in the same array", () => {
    const query = subject();
    const sort = spyOn(Array.prototype, "sort");
    try {
        expect([...query]).toEqual([1, 2, 3, 4]);
        expect(sort).toHaveBeenCalledTimes(1);
        const array = sort.mock.contexts[0];
        expect([...query]).toEqual([1, 2, 3, 4]);
        query.add(2);
        query.remove(9);
        expect([...query]).toEqual([1, 2, 3, 4]);
        expect(sort).toHaveBeenCalledTimes(1);
        query.remove(2);
        expect([...query]).toEqual([1, 3, 4]);
        expect(sort).toHaveBeenCalledTimes(2);
        expect(sort.mock.contexts[1]).toBe(array);
    } finally {
        sort.mockRestore();
    }
});

for (const removal of ["current", "visited", "unvisited"] as const) {
    test(`removing a ${removal} query member never moves another visit`, () => {
        const query = subject();
        const visits: number[] = [];
        for (const eid of query) {
            visits.push(eid);
            if (removal === "current") query.remove(eid);
            if (removal === "visited" && eid === 2) query.remove(1);
            if (removal === "unvisited" && eid === 1) query.remove(3);
        }
        expect(visits).toEqual(removal === "unvisited" ? [1, 2, 4] : [1, 2, 3, 4]);
        expect([...query]).toEqual(
            removal === "current" ? [] : removal === "visited" ? [2, 3, 4] : [1, 2, 4],
        );
    });
}

for (const replaced of [1, 3]) {
    test(`new memberships and removed/re-added eid ${replaced} wait for the next iteration`, () => {
        const query = subject();
        const visits: number[] = [];
        for (const eid of query) {
            visits.push(eid);
            if (eid !== 1) continue;
            query.remove(replaced);
            query.add(replaced);
            query.add(5);
        }
        expect(visits).toEqual(replaced === 1 ? [1, 2, 3, 4] : [1, 2, 4]);
        expect([...query]).toEqual([1, 2, 3, 4, 5]);
    });
}

test("an eid destroyed and reused before its visit waits for the next query iteration", () => {
    const world = new World();
    const C = component("query-reuse", {});
    const a = world.create();
    const b = world.create();
    world.add(a, C);
    world.add(b, C);
    const query = world.query([C]);
    const visits: number[] = [];
    for (const eid of query) {
        visits.push(eid);
        if (eid !== a) continue;
        world.destroy(b);
        expect(world.create()).toBe(b);
        world.add(b, C);
    }
    expect(visits).toEqual([a]);
    expect([...query]).toEqual([a, b]);
});

test("a nested iteration after membership changes preserves the outer boundary and ascending visits", () => {
    const query = subject();
    const outer: number[] = [];
    for (const eid of query) {
        outer.push(eid);
        if (eid !== 1) continue;
        query.remove(1);
        query.add(1);
        query.remove(3);
        query.add(3);
        query.add(5);
        expect([...query]).toEqual([1, 2, 3, 4, 5]);
    }
    expect(outer).toEqual([1, 2, 4]);
});

test("borrowed query iterators and results are reused after completion or early return", () => {
    const query = subject();
    const iterator = query[Symbol.iterator]();
    const result = iterator.next();
    expect(result.value).toBe(1);
    expect(iterator.next()).toBe(result);
    expect(result.value).toBe(2);
    iterator.return!();
    expect(iterator.next().done).toBe(true);
    expect(query[Symbol.iterator]()).toBe(iterator);
    while (!iterator.next().done) {}
    expect(query[Symbol.iterator]()).toBe(iterator);
    iterator.return!();
});
