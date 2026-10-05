import { expect, test } from "bun:test";
import { entity, World } from "./index";

test("a kept reference resolves while alive and reads missing after destruction and eid reuse", () => {
    const world = new World();
    try {
        const eid = world.create();
        const ref = world.ref(eid);
        expect(Number.isSafeInteger(ref)).toBe(true);
        expect(world.resolve(ref)).toBe(eid);
        world.destroy(eid);
        expect(Number(world.ref(eid))).toBe(0);
        expect(world.resolve(ref)).toBe(0);
        expect(world.create()).toBe(eid);
        expect(world.resolve(ref)).toBe(0);
        expect(world.resolve(world.ref(eid))).toBe(eid);
        expect(world.resolve(world.ref(0))).toBe(0);
    } finally {
        world.dispose();
    }
});

test("entity fields resolve missing after target reuse without changing their raw references", () => {
    const Link = { target: entity };
    const world = new World();
    try {
        const target = world.create();
        const owner = world.create();
        world.add(owner, Link, { target });
        const field = world.storage(Link).target;
        const ref = world.ref(target);
        expect(field.get(owner)).toBe(target);
        expect(field.column[owner]).toBe(ref);
        world.destroy(target);
        expect(field.get(owner)).toBe(0);
        expect(world.create()).toBe(target);
        expect(field.get(owner)).toBe(0);
        expect(field.column[owner]).toBe(ref);
        field.set(owner, target);
        expect(field.get(owner)).toBe(target);
        field.writeEncoded(new Uint32Array([owner]), new Float64Array([ref]));
        expect(field.get(owner)).toBe(0);
        expect(field.column[owner]).toBe(ref);
    } finally {
        world.dispose();
    }
});
