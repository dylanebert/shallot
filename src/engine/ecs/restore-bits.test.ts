import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { World } from "./world";

const P = component("restore-bits-P", { v: f32 });
const Q = component("restore-bits-Q", { v: f32 });

test("restoring an abandoned branch reconciles each component's own membership", () => {
    const world = new World();
    world.registerRecovery("RestoreBits", "stateless");
    world.storage(P);
    world.storage(Q);
    const notified: string[] = [];
    world.observeMembership(P, (eid, present) => notified.push(`P ${eid} ${present}`));
    world.observeMembership(Q, (eid, present) => notified.push(`Q ${eid} ${present}`));
    const holdsP = world.query([P]);
    const holdsQ = world.query([Q]);
    const eid = world.create();
    const before = world.snapshot();
    world.add(eid, Q);
    const branch = world.snapshot();
    world.restore(before);
    world.add(eid, P);
    notified.length = 0;
    world.restore(branch);
    expect([world.has(eid, P), world.has(eid, Q)]).toEqual([false, true]);
    expect([[...holdsP], [...holdsQ]]).toEqual([[], [eid]]);
    expect(notified.sort()).toEqual([`P ${eid} false`, `Q ${eid} true`]);
});

test("a component first added after a snapshot past 31 stored components restores", () => {
    const world = new World();
    world.registerRecovery("RestoreBits", "stateless");
    const many = Array.from({ length: 32 }, (_, i) => component(`restore-bits-${i}`, { v: f32 }));
    for (const c of many) world.storage(c);
    const eid = world.create();
    for (const c of many.slice(0, 31)) world.add(eid, c);
    const image = world.snapshot();
    world.add(eid, many[31]);
    world.restore(image);
    expect(many.map((c) => world.has(eid, c))).toEqual([...Array(31).fill(true), false]);
    world.add(eid, many[31]);
    expect(world.has(eid, many[31])).toBe(true);
});
