import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { World } from "./world";

const C = component("snapshot-query-order", { value: f32 });

test("snapshot restores retained query iteration order so order-sensitive fixed work replays", () => {
    const world = new World();
    const a = world.create();
    const b = world.create();
    world.add(a, C);
    world.add(b, C);
    const query = world.query([C]);
    world.remove(a, C);
    world.add(a, C);
    expect([...query]).toEqual([b, a]);
    const snapshot = world.snapshot();
    world.addSystem({
        group: "fixed",
        update: (w) => {
            let value = 0;
            for (const eid of query) w.storage(C).value.set(eid, ++value);
        },
    });
    world.tick();
    const values = world.storage(C).value;
    expect([values.get(a), values.get(b)]).toEqual([2, 1]);
    world.restore(snapshot);
    expect([...query]).toEqual([b, a]);
    world.tick();
    expect([values.get(a), values.get(b)]).toEqual([2, 1]);
    world.restore(snapshot);
    world.tick();
    expect([values.get(a), values.get(b)]).toEqual([2, 1]);
});

test("queries first registered after capture rebuild from restored membership and allocator order", () => {
    const world = new World();
    const a = world.create();
    const b = world.create();
    world.add(a, C);
    world.add(b, C);
    const snapshot = world.snapshot();
    world.destroy(a);
    const later = world.query([C]);
    expect([...later]).toEqual([b]);
    world.restore(snapshot);
    expect([...later]).toEqual([a, b]);
    expect(world.query([C])).toBe(later);
    world.remove(a, C);
    world.add(a, C);
    expect([...later]).toEqual([b, a]);
    world.restore(snapshot);
    expect([...later]).toEqual([a, b]);
});
