import { expect, test } from "bun:test";
import { recoverResource } from "./snapshot";
import { type Resource, World } from "./world";

type Sample = { step: number; point: number[]; refs: Map<number, { row: number }>; late?: number };
const Sample: Resource<Sample> = {
    create: () => ({ step: 0, point: [0, 0, 0], refs: new Map() }),
};

test("an image is independent of later writes and restores twice into the retained resource", () => {
    const world = new World();
    const recovery = recoverResource(Sample)(world);
    const live = world.resource(Sample);
    live.step = 3;
    live.refs.set(7, { row: 1 });
    const image = recovery.snapshot();
    const point = live.point;
    for (let run = 0; run < 2; run++) {
        live.step = 9;
        live.point[0] = 5;
        live.refs.get(7)!.row = 4;
        live.refs.set(8, { row: 2 });
        recovery.restore(image);
        expect(world.resource(Sample)).toBe(live);
        expect(live.step).toBe(3);
        expect(live.point).toEqual([0, 0, 0]);
        expect(live.point).not.toBe(point);
        expect([...live.refs]).toEqual([[7, { row: 1 }]]);
    }
    world.dispose();
});

test("restore removes a key the image lacks", () => {
    const world = new World();
    const recovery = recoverResource(Sample)(world);
    const image = recovery.snapshot();
    world.resource(Sample).late = 1;
    recovery.restore(image);
    expect("late" in world.resource(Sample)).toBe(false);
    world.dispose();
});

test("a function-valued resource throws at snapshot", () => {
    const world = new World();
    const Callback: Resource<{ run: () => void }> = { create: () => ({ run() {} }) };
    expect(() => recoverResource(Callback)(world).snapshot()).toThrow(
        expect.objectContaining({ name: "DataCloneError" }),
    );
    world.dispose();
});
