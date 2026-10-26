import { expect, test } from "bun:test";
import { BodyDependencies } from "./body-dependencies";

function dependents(index: BodyDependencies, body: number): number[] {
    const result: number[] = [];
    for (let eid = index.first(body); eid; eid = index.following(eid)) result.push(eid);
    return result;
}

test("body dependency lists retain links through growth, retargeting, removal and reusable recovery", () => {
    const index = new BodyDependencies();
    index.set(1, 5);
    index.set(2, 5);
    index.set(300, 5);
    index.set(4, 800);
    expect(dependents(index, 5)).toEqual([300, 2, 1]);
    const image = index.snapshot();
    index.set(2, 800);
    expect(dependents(index, 5)).toEqual([300, 1]);
    expect(dependents(index, 800)).toEqual([2, 4]);
    index.set(300, 0);
    index.set(1, 0);
    expect(dependents(index, 5)).toEqual([]);
    index.restore(image);
    expect(dependents(index, 5)).toEqual([300, 2, 1]);
    expect(dependents(index, 800)).toEqual([4]);
    index.clear();
    expect(dependents(index, 5)).toEqual([]);
    index.restore(image);
    expect(dependents(index, 5)).toEqual([300, 2, 1]);
    index.set(2, 5);
    expect(dependents(index, 5)).toEqual([300, 2, 1]);
});
