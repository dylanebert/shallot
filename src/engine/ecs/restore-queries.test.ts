import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { not, or } from "./query";
import { World } from "./world";

const X = component("restore-query-not", { value: f32 });
const C = component("restore-query-observer", { value: f32 });
const U = component("restore-query-unrelated", { value: f32 });

test("a not-only query refilters newly created entities on restore", () => {
    const world = new World();
    const withoutX = world.query([not(X)]);
    const eid = world.create();
    const image = world.snapshot();
    world.restore(image);
    expect([...withoutX]).toEqual([eid]);
});

test("a not-only query includes entities created after registration, including after an unrelated add", () => {
    const world = new World();
    const excluded = world.create();
    world.add(excluded, X);
    const withoutX = world.query([not(X)]);
    world.create();
    const unrelated = world.create();
    world.add(unrelated, C);
    expect([...withoutX]).toEqual(world.entities().filter((eid) => !world.has(eid, X)));
});

test("a query with two excluded terms includes entities created after registration, including after an unrelated add", () => {
    const world = new World();
    const withX = world.create();
    world.add(withX, X);
    const withC = world.create();
    world.add(withC, C);
    const withoutXOrC = world.query([not(X), not(C)]);
    world.create();
    const unrelated = world.create();
    world.add(unrelated, U);
    expect([...withoutXOrC]).toEqual(
        world.entities().filter((eid) => !world.has(eid, X) && !world.has(eid, C)),
    );
});

test("an empty query includes entities created after registration, including after an unrelated add", () => {
    const world = new World();
    world.create();
    const all = world.query([]);
    world.create();
    const unrelated = world.create();
    world.add(unrelated, C);
    expect([...all]).toEqual([...world.entities()]);
});

test("an or query does not include a newly created entity", () => {
    const world = new World();
    const either = world.query([or(X, C)]);
    const eid = world.create();
    expect([...either]).toEqual(
        world.entities().filter((entity) => world.has(entity, X) || world.has(entity, C)),
    );
    expect([...either]).not.toContain(eid);
});

test("restore preserves query order for nested and active iterations", () => {
    const world = new World();
    const eids = Array.from({ length: 3 }, () => world.create());
    for (const eid of eids) world.add(eid, C);
    const withC = world.query([C]);
    const image = world.snapshot();
    world.remove(eids[0], C);
    const outer = withC[Symbol.iterator]();
    expect(outer.next().value).toBe(eids[1]);
    world.restore(image);
    const nested = [...withC];
    const rest: number[] = [];
    for (let next = outer.next(); !next.done; next = outer.next()) rest.push(next.value);
    expect(nested).toEqual(eids);
    expect(rest).toEqual([eids[1]]);
});

test("restore observers see changed query membership already reconciled", () => {
    const world = new World();
    const eid = world.create();
    world.add(eid, C);
    const withC = world.query([C]);
    const image = world.snapshot();
    world.remove(eid, C);
    let restoring = false;
    const observations: [boolean, number[]][] = [];
    world.observeMembership(C, (_eid, present) => {
        if (restoring) observations.push([present, [...withC]]);
    });
    restoring = true;
    world.restore(image);
    restoring = false;
    expect(observations).toEqual([[true, [eid]]]);
});
