import { expect, test } from "bun:test";
import { BodyDependencies } from "./body-dependencies";

function dependents(index: BodyDependencies, body: number): number[] {
    const result: number[] = [];
    for (let eid = index.first(body); eid; eid = index.following(eid)) result.push(eid);
    return result;
}

test("body dependency lists append joins through growth, retargeting, removal and reusable recovery", () => {
    const index = new BodyDependencies();
    index.set(20, 5);
    index.set(2, 5);
    index.set(300, 5);
    index.set(4, 800);
    expect(dependents(index, 5)).toEqual([20, 2, 300]);
    const image = index.snapshot();
    index.set(2, 800);
    expect(dependents(index, 5)).toEqual([20, 300]);
    expect(dependents(index, 800)).toEqual([4, 2]);
    index.set(300, 0);
    index.set(20, 0);
    expect(dependents(index, 5)).toEqual([]);
    index.set(300, 800);
    expect(dependents(index, 800)).toEqual([4, 2, 300]);
    index.set(20, 800);
    expect(dependents(index, 800)).toEqual([4, 2, 300, 20]);
    for (let i = 0; i < 2; i++) {
        index.restore(image);
        expect(dependents(index, 5)).toEqual([20, 2, 300]);
        expect(dependents(index, 800)).toEqual([4]);
        index.set(2, 0);
        index.set(2, 5);
        expect(dependents(index, 5)).toEqual([20, 300, 2]);
        index.set(2, 5);
        expect(dependents(index, 5)).toEqual([20, 300, 2]);
        index.clear();
        expect(dependents(index, 5)).toEqual([]);
    }
});

test("retargeting reads a constant number of links in a one-member or 1000-member destination list", () => {
    function readsFor(size: number): { reads: number; list: number[] } {
        const index = new BodyDependencies();
        index.set(1, 10);
        for (let i = 2; i < size + 2; i++) index.set(i, 2000);
        const names = ["_heads", "_tails", "_next", "_previous"] as const;
        const columns = index as unknown as Record<(typeof names)[number], Uint32Array>;
        const saved = names.map((name) => columns[name]);
        let reads = 0;
        for (const name of names) {
            columns[name] = new Proxy(columns[name], {
                get(target, key) {
                    if (typeof key === "string" && /^\d+$/.test(key)) reads++;
                    return Reflect.get(target, key, target);
                },
                set(target, key, value) {
                    return Reflect.set(target, key, value, target);
                },
            });
        }
        try {
            index.set(1, 2000);
        } finally {
            for (let i = 0; i < names.length; i++) columns[names[i]] = saved[i];
        }
        return { reads, list: dependents(index, 2000) };
    }
    const small = readsFor(1),
        large = readsFor(1000);
    expect(large.reads).toBe(small.reads);
    expect(large.reads).toBeLessThanOrEqual(6);
    expect(small.list).toEqual([2, 1]);
    expect(large.list.length).toBe(1001);
    expect(large.list.at(-1)).toBe(1);
});
