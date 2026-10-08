import { expect, test } from "bun:test";
import { component, f32 } from "./component";
import { World } from "./world";

const C = component("snapshot-query-order", { value: f32 });

for (const retained of [true, false]) {
    test(`${retained ? "retained" : "late"} query replay ignores remove/re-add history and assigns in ascending eid order`, () => {
        const world = new World();
        world.registerRecovery("SnapshotFixture", "stateless");
        const a = world.create();
        const b = world.create();
        world.add(a, C);
        world.add(b, C);
        if (retained) {
            world.query([C]);
            world.remove(a, C);
            world.add(a, C);
        }
        const snapshot = world.snapshot();
        world.addSystem(
            {
                group: "fixed",
                update: (w) => {
                    if (!retained && w.time.fixedTick === 1) {
                        w.remove(a, C);
                        w.add(a, C);
                    } else {
                        let value = 0;
                        for (const eid of w.query([C])) w.storage(C).value.set(eid, ++value);
                    }
                },
            },
            "SnapshotFixture",
        );
        const run = () => {
            world.tick();
            if (!retained) world.tick();
            const values = world.storage(C).value;
            return [values.get(a), values.get(b)];
        };
        expect(run()).toEqual([1, 2]);
        world.restore(snapshot);
        expect(run()).toEqual([1, 2]);
        world.restore(snapshot);
        expect(run()).toEqual([1, 2]);
    });
}
