import { expect, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    resetInterpolation,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";

test("a fixed target advances its follow camera 0.05 every 120 Hz frame, not alternating 0 and 0.10", async () => {
    let target = 0;
    const app = await createApp({
        defaults: false,
        plugins: [
            OrbitPlugin,
            {
                name: "Move",
                systems: [
                    {
                        group: "fixed",
                        update(w) {
                            const p = w.storage(Transform).translation;
                            p.x.set(target, p.x.get(target) + 0.1);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const w = app.world;
        target = w.create();
        w.add(target, Transform);
        const camera = w.create();
        w.add(camera, Transform);
        w.add(camera, Camera);
        w.add(camera, Orbit, { target, yaw: 0, pitch: 0, distance: 6 });
        w.step(0);
        const positions: number[] = [];
        for (let i = 0; i < 12; i++) {
            w.step(Time.FIXED_DT / 2);
            positions.push(w.get(camera, GlobalTransform)!.translation[0]);
        }
        for (let i = 2; i < positions.length; i++)
            expect(positions[i]! - positions[i - 1]!).toBeCloseTo(0.05, 6);
        w.storage(Transform).translation.x.set(target, 20);
        resetInterpolation(w, target);
        w.step(Time.FIXED_DT);
        expect(w.get(camera, Transform)!.translation[0]).toBeCloseTo(20.1, 5);
    } finally {
        app.dispose();
    }
});
