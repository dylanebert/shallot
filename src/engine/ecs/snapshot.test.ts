import { expect, test } from "bun:test";
import { component, entity, f32 } from "./component";
import { World } from "./world";

const State = component("snapshot-state", { value: f32, target: entity });
const Tag = component("snapshot-tag", {});

function subject() {
    const world = new World();
    world.registry.register(State);
    world.registry.register(Tag);
    const a = world.create();
    const b = world.create();
    world.add(a, State, { value: 3, target: b });
    world.add(b, Tag);
    world.addSystem({
        group: "fixed",
        update: (w) => {
            const storage = w.storage(State);
            for (const eid of w.query([State]))
                storage.value.set(eid, storage.value.get(eid) + w.time.elapsed);
            if (w.time.fixedTick % 2 === 0) {
                const eid = w.only([Tag]);
                if (eid !== -1) w.destroy(eid);
                const next = w.create();
                w.add(next, State, { value: w.time.fixedTick, target: a });
                w.add(next, Tag);
            }
        },
    });
    return { world, a, b };
}

function image(world: World) {
    return {
        tick: world.time.fixedTick,
        entities: world.entities(),
        members: [...world.query([State])].sort(),
        fields: [...world.storage(State).value.column],
        targets: [...world.storage(State).target.column],
        refs: world.entities().map((eid) => world.ref(eid)),
    };
}

test("snapshot restores identity, membership, columns and future allocation after despawn and reuse", () => {
    const { world, a, b } = subject();
    const sibling = subject().world;
    const untouched = image(sibling);
    world.tick();
    const saved = image(world);
    const accessor = world.storage(State).value;
    const ref = world.ref(b);
    const query = world.query([State]);
    const snapshot = world.snapshot();
    for (let i = 0; i < 4; i++) world.tick();
    const continued = image(world);
    const next = world.create();
    const nextRef = world.ref(next);
    for (let i = 0; i < 40; i++) world.create();
    expect(world.resolve(ref)).toBe(0);
    const events: [number, boolean][] = [];
    world.observeMembership(State, (eid, present) => events.push([eid, present]));
    world.restore(snapshot);
    expect(world.resolve(ref)).toBe(b);
    expect(accessor.get(a)).toBe(saved.fields[a]);
    expect([...query]).toEqual(saved.members);
    expect(events).toContainEqual([b, false]);
    expect(world.fieldStorage(State, "value").dirty[a >>> 5] & (1 << (a & 31))).not.toBe(0);
    expect(image(sibling)).toEqual(untouched);
    for (let i = 0; i < 4; i++) world.tick();
    const replay = image(world);
    expect(replay.fields.slice(0, continued.fields.length)).toEqual(continued.fields);
    expect(replay.targets.slice(0, continued.targets.length)).toEqual(continued.targets);
    expect(replay.entities).toEqual(continued.entities);
    expect(replay.members).toEqual(continued.members);
    expect(replay.refs).toEqual(continued.refs);
    expect(replay.tick).toBe(continued.tick);
    expect(world.create()).toBe(next);
    expect(world.ref(next)).toBe(nextRef);
});

test("snapshot preserves free-list order, inactive fields and registered but unattached columns", () => {
    const world = new World();
    world.registry.register(State);
    const ids = Array.from({ length: 6 }, () => world.create());
    world.destroy(ids[1]);
    world.destroy(ids[4]);
    const storage = world.storage(State);
    storage.value.set(ids[1], 91);
    const column = storage.value.column;
    const snapshot = world.snapshot();
    const allocated = Array.from({ length: 4 }, () => {
        const eid = world.create();
        return [eid, world.ref(eid)];
    });
    storage.value.set(ids[1], 0);
    world.restore(snapshot);
    expect(storage.value.column).toBe(column);
    expect(storage.value.get(ids[1])).toBe(91);
    expect([...world.query([State])]).toEqual([]);
    expect(
        Array.from({ length: 4 }, () => {
            const eid = world.create();
            return [eid, world.ref(eid)];
        }),
    ).toEqual(allocated);
});

test("restore refuses execution boundaries, foreign worlds and changed component registries by cause", () => {
    const { world } = subject();
    const snapshot = world.snapshot();
    expect(() => new World().restore(snapshot)).toThrow("another world");
    world.addSystem({
        group: "fixed",
        update: () => {
            expect(() => world.restore(snapshot)).toThrow("inside a step or tick");
            expect(() => world.snapshot()).toThrow("inside a step or tick");
        },
    });
    world.tick();
    world.step();
    world.registry.register(component("late-snapshot-component", {}));
    expect(() => world.restore(snapshot)).toThrow("component registry");
});
