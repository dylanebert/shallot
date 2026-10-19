import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    InputPlugin,
    pressKey,
    releaseKey,
    StandardPhysicsPlugin,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { contact, EventSample, EventSamplePlugin, SampleInputPlugin } from "../src/events";

setDefaultTimeout(TIMEOUT.node);

test("Contact records focused keys and camera-relative axes before fixed gameplay without a GPU", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin, InputPlugin, EventSamplePlugin, SampleInputPlugin],
        setup(world) {
            contact(world);
            const camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform, { rotation: [0, Math.SQRT1_2, 0, Math.SQRT1_2] });
        },
    });
    try {
        const world = app.world;
        world.step(Time.FIXED_DT);
        pressKey(world, "KeyW");
        pressKey(world, "KeyA");
        world.step(Time.FIXED_DT);
        const sample = world.resource(EventSample);
        expect(sample.keys).toEqual({ W: 1, S: 0, A: 1, D: 0 });
        expect(sample.forward[0]).toBeCloseTo(-1, 5);
        expect(sample.forward[2]).toBeCloseTo(0, 5);
        expect(sample.right[0]).toBeCloseTo(0, 5);
        expect(sample.right[2]).toBeCloseTo(-1, 5);
        releaseKey(world, "KeyW");
        releaseKey(world, "KeyA");
        world.step(Time.FIXED_DT);
        expect(sample.keys).toEqual({ W: 0, S: 0, A: 0, D: 0 });
    } finally {
        app.dispose();
    }
});
