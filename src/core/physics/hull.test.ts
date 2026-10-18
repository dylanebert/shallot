import { expect, test } from "bun:test";
import { World } from "../../engine";
import { Hulls } from "./hull";

test("cylinder authoring refuses invalid dimensions and tessellation before allocating a handle", () => {
    const world = new World();
    const hulls = world.resource(Hulls);
    const cylinder = { height: 1, radius: 0.4, yOffset: -1, sides: 12 };
    for (const invalid of [
        { height: 0 },
        { height: Infinity },
        { radius: -1 },
        { radius: NaN },
        { yOffset: Infinity },
        { sides: 2 },
        { sides: 33 },
        { sides: 3.5 },
    ]) {
        expect(() =>
            hulls.add({ name: "invalid cylinder", cylinder: { ...cylinder, ...invalid } }),
        ).toThrow("cylinder");
    }
    expect(hulls.size).toBe(1);
    hulls.add({ name: "cylinder", cylinder });
    expect(hulls.size).toBe(2);
    world.dispose();
});

test("transformed boxes require finite positions and normalized finite quaternions", () => {
    const world = new World();
    const hulls = world.resource(Hulls);
    for (const transform of [
        { position: [Infinity, 0, 0], rotation: [0, 0, 0, 1] },
        { position: [0, 0, 0], rotation: [0, 0, 0, 0] },
        { position: [0, 0, 0], rotation: [NaN, 0, 0, 1] },
    ] as const)
        expect(() => hulls.add({ name: "invalid box", halfExtents: [1, 2, 3], transform })).toThrow(
            "box transform",
        );
    const box = hulls.add({
        name: "box",
        halfExtents: [1, 2, 3],
        transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1] },
    });
    const cylinder = hulls.add({
        name: "cylinder",
        cylinder: { height: 1, radius: 1, yOffset: 0, sides: 12 },
    });
    const saved = structuredClone(hulls.snapshot());
    hulls.clear();
    hulls.restore(saved);
    expect(hulls.get(box)).toEqual(saved.items[box]);
    expect(hulls.get(cylinder)).toEqual(saved.items[cylinder]);
    world.dispose();
});
