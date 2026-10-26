import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { World } from "./world";

const Companion = component("membership-companion", { v: f32 }, { defaults: () => ({ v: 2 }) });
const Held = component(
    "membership-held",
    { v: f32, w: f32 },
    { defaults: () => ({ v: 7, w: 8 }), requires: [Companion] },
);
const Other = component("membership-other", { v: f32 });

function registered(): World {
    const world = new World();
    world.registry.register(Companion);
    world.registry.register(Held);
    world.registry.register(Other);
    return world;
}

test("an attachment observer reads the added values, defaults and required companions", () => {
    const world = registered();
    const held = world.storage(Held);
    const companion = world.storage(Companion);
    const seen: number[][] = [];
    world.observeMembership(Held, (eid, present) => {
        if (present)
            seen.push([
                held.v.get(eid),
                held.w.get(eid),
                Number(world.has(eid, Companion)),
                companion.v.get(eid),
            ]);
    });
    const order: string[] = [];
    world.observeMembership(Companion, (eid) => order.push(`Companion ${held.v.get(eid)}`));
    world.observeMembership(Held, () => order.push("Held"));

    world.add(world.create(), Held, { v: 3 });
    world.add(world.create(), Held);

    expect(seen).toEqual([
        [3, 8, 1, 2],
        [7, 8, 1, 2],
    ]);
    expect(order).toEqual(["Companion 3", "Held", "Companion 7", "Held"]);
});

test("restore notifies only changed memberships, after the restored values are in place", () => {
    const world = registered();
    world.registerRecovery("Membership", "stateless");
    const lost = world.create();
    const kept = world.create();
    world.add(lost, Other, { v: 5 });
    world.add(kept, Other, { v: 6 });
    const image = world.snapshot();
    world.remove(lost, Other);
    world.storage(Other).v.set(kept, 9);
    const seen: [number, boolean, number][] = [];
    world.observeMembership(Other, (eid, present) =>
        seen.push([eid, present, world.storage(Other).v.get(eid)]),
    );

    world.restore(image);

    expect(seen).toEqual([[lost, true, 5]]);
    expect(world.storage(Other).v.get(kept)).toBe(6);
});

test("an observer changes other components and entities immediately", () => {
    const world = registered();
    const others = world.query([Other]);
    let spawned = 0;
    world.observeMembership(Held, (eid, present) => {
        if (present) {
            world.add(eid, Other, { v: 4 });
            world.remove(eid, Companion);
            spawned = world.create();
            world.add(spawned, Other, { v: 1 });
        } else {
            world.remove(eid, Other);
            world.destroy(spawned);
        }
    });
    const eid = world.create();

    world.add(eid, Held);
    expect([world.has(eid, Other), world.has(eid, Companion), [...others]]).toEqual([
        true,
        false,
        [eid, spawned],
    ]);
    expect(world.storage(Other).v.get(eid)).toBe(4);

    world.remove(eid, Held);
    expect([world.has(eid, Other), world.exists(spawned), [...others]]).toEqual([false, false, []]);
});

const Failing = component(
    "membership-failing",
    { v: f32 },
    {
        defaults: () => {
            throw new Error("failing defaults");
        },
    },
);
const Parent = component("membership-parent", { v: f32 }, { requires: [Failing, Companion] });

test("an attachment that throws is undone without notifying", () => {
    const world = registered();
    world.registry.register(Failing);
    world.registry.register(Parent);
    const parents = world.query([Parent]);
    const others = world.query([Other]);
    const notified: string[] = [];
    for (const [name, c] of [
        ["Parent", Parent],
        ["Failing", Failing],
        ["Companion", Companion],
        ["Other", Other],
    ] as const)
        world.observeMembership(c, (_eid, present) => notified.push(`${name} ${present}`));
    const eid = world.create();

    expect(() => world.add(eid, Parent, { v: 3 })).toThrow("failing defaults");
    const values = {
        v: 4,
        get w(): number {
            throw new Error("failing value");
        },
    };
    expect(() => world.add(eid, Other, values as { v: number })).toThrow("failing value");

    expect(notified).toEqual([]);
    expect([world.has(eid, Parent), world.has(eid, Failing), world.has(eid, Companion)]).toEqual([
        false,
        false,
        false,
    ]);
    expect([world.has(eid, Other), [...parents], [...others]]).toEqual([false, [], []]);
    expect([world.storage(Parent).v.get(eid), world.storage(Other).v.get(eid)]).toEqual([0, 0]);
    world.remove(eid, Parent);
    world.remove(eid, Other);
    expect(notified).toEqual([]);
});

test("a change completes for every observer when an observer's error propagates", () => {
    for (const end of ["remove", "destroy"] as const) {
        const world = registered();
        const holders = world.query([Other]);
        const eid = world.create();
        world.add(eid, Other, { v: 42 });
        world.observeMembership(Other, () => {
            throw new Error("observer fails");
        });
        const later: boolean[] = [];
        world.observeMembership(Other, (_eid, present) => later.push(present));

        expect(() => (end === "remove" ? world.remove(eid, Other) : world.destroy(eid))).toThrow(
            "observer fails",
        );

        expect(later).toEqual([false]);
        expect([world.has(eid, Other), [...holders], world.storage(Other).v.get(eid)]).toEqual([
            false,
            [],
            0,
        ]);
        expect(world.exists(eid)).toBe(end === "remove");
    }
});

test("an attachment observer's error propagates after every later observer has run", () => {
    const world = registered();
    const eid = world.create();
    world.observeMembership(Other, () => {
        throw new Error("observer fails");
    });
    const later: [boolean, number][] = [];
    world.observeMembership(Other, (changed, present) =>
        later.push([present, world.storage(Other).v.get(changed)]),
    );

    expect(() => world.add(eid, Other, { v: 4 })).toThrow("observer fails");

    expect(later).toEqual([[true, 4]]);
    expect([world.has(eid, Other), [...world.query([Other])]]).toEqual([true, [eid]]);
});

test("a destroy inside a destroy's notification leaves the outer entity's fields cleared", () => {
    const world = registered();
    const outer = world.create();
    const inner = world.create();
    world.add(outer, Other, { v: 5 });
    world.add(inner, Companion, { v: 6 });
    world.observeMembership(Other, (eid, present) => {
        if (!present && eid === outer) world.destroy(inner);
    });

    world.destroy(outer);

    expect([world.exists(outer), world.exists(inner)]).toEqual([false, false]);
    expect([world.storage(Other).v.get(outer), world.storage(Companion).v.get(inner)]).toEqual([
        0, 0,
    ]);
});

test("a restore observer's error propagates after every change and participant is restored", () => {
    const world = registered();
    const recovered: string[] = [];
    world.registerRecovery("Membership", {
        snapshot: () => "image",
        restore: (state) => recovered.push(state),
    });
    const first = world.create();
    const second = world.create();
    world.add(first, Other);
    world.add(second, Other);
    const image = world.snapshot();
    world.remove(first, Other);
    world.remove(second, Other);
    world.observeMembership(Other, (eid) => {
        if (eid === first) throw new Error("first restored eid fails");
    });
    const seen: number[] = [];
    world.observeMembership(Other, (eid) => seen.push(eid));

    expect(() => world.restore(image)).toThrow("first restored eid fails");

    expect(seen).toEqual([first, second]);
    expect([world.has(first, Other), world.has(second, Other)]).toEqual([true, true]);
    expect(recovered).toEqual(["image"]);
});

test("destroying an entity inside its own destroy does nothing", () => {
    const world = registered();
    const eid = world.create();
    world.add(eid, Other, { v: 42 });
    world.add(eid, Companion, { v: 7 });
    let created = 0;
    world.observeMembership(Other, (changed) => {
        world.destroy(changed);
        created = world.create();
    });
    const seen: boolean[] = [];
    world.observeMembership(Other, (changed) => seen.push(world.exists(changed)));

    world.destroy(eid);

    expect(created).not.toBe(eid);
    expect([world.exists(eid), world.exists(created), seen]).toEqual([false, true, [true]]);
    expect(world.create()).toBe(eid);
});
