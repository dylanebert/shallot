import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    createApp,
    GlobalTransform,
    pressButton,
    pressKey,
    qRotate,
    releaseKey,
    Time,
    Transform,
} from "@dylanebert/shallot";
import { Orbit, OrbitPlugin } from "@dylanebert/shallot/extras";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { DrivingPlugin, DrivingState } from "../src/driving";
import { createScene } from "../src/scenes";
import { setSceneCamera } from "../src/view";

setDefaultTimeout(TIMEOUT.node);

test("Driving's camera target follows its chassis while WASD is live without a GPU", async () => {
    let camera = 0;
    const app = await createApp({
        defaults: false,
        plugins: [DrivingPlugin, OrbitPlugin],
        setup(world) {
            createScene(world, "driving");
            camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            world.add(camera, Orbit);
            setSceneCamera(world, "driving", camera);
        },
    });
    try {
        const world = app.world;
        world.step(Time.FIXED_DT);
        const chassis = world.resource(DrivingState).chassis;
        const start = world.get(chassis, GlobalTransform)!.translation;
        pressButton(world, "right");
        for (const key of ["W", "A", "S", "D"] as const) {
            pressKey(world, `Key${key}`);
            for (let i = 0; i < 60; i++) {
                world.step(Time.FIXED_DT);
                expect(world.resource(DrivingState).keys[key]).toBe(1);
                expect(world.get(camera, Orbit)!.target).toBe(chassis);
                const pose = world.get(camera, GlobalTransform)!;
                const forward = qRotate(...pose.rotation, 0, 0, -1);
                const target = pose.translation.map((value, i) => value + 12 * forward[i]!);
                const position = world.get(chassis, GlobalTransform)!.translation;
                for (const [i, value] of position.entries())
                    expect(target[i]).toBeCloseTo(value, 4);
            }
            releaseKey(world, `Key${key}`);
            world.step(Time.FIXED_DT);
            expect(world.resource(DrivingState).keys[key]).toBe(0);
        }
        expect(world.get(chassis, GlobalTransform)!.translation).not.toEqual(start);
    } finally {
        app.dispose();
    }
});
