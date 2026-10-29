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
import { createScene, scenes } from "../src/scenes";
import { setSceneCamera } from "../src/view";
import { VillagePlugin, VillageState } from "../src/village";

setDefaultTimeout(TIMEOUT.node);
test("Village keeps every movement key live with its follow camera and has no T toggle or Surface row", async () => {
    let camera = 0;
    const app = await createApp({
        defaults: false,
        plugins: [VillagePlugin, OrbitPlugin],
        setup(world) {
            createScene(world, "village");
            camera = world.create();
            world.add(camera, Camera);
            world.add(camera, Transform);
            world.add(camera, Orbit);
            setSceneCamera(world, "village", camera);
        },
    });
    try {
        const world = app.world,
            s = world.resource(VillageState);
        expect(scenes.village.controls).toBeUndefined();
        world.step(Time.FIXED_DT);
        pressButton(world, "right");
        for (const key of ["W", "A", "S", "D", "Space", "Shift"] as const) {
            const code = key === "Shift" ? "ShiftLeft" : key === "Space" ? "Space" : `Key${key}`;
            pressKey(world, code);
            const previous = { ...s.mover.position };
            world.step(Time.FIXED_DT);
            expect(s.keys[key]).toBe(1);
            expect(world.get(camera, Orbit)!.target).toBe(s.entity);
            const pose = world.get(camera, GlobalTransform)!;
            const forward = qRotate(...pose.rotation, 0, 0, -1);
            for (const [axis, value] of [previous.x, previous.y, previous.z].entries())
                expect(
                    pose.translation[axis]! + scenes.village.camera.distance * forward[axis]!,
                ).toBeCloseTo(value, 4);
            releaseKey(world, code);
            world.step(Time.FIXED_DT);
            expect(s.keys[key]).toBe(0);
        }
        pressKey(world, "KeyT");
        world.step(Time.FIXED_DT);
        expect(world.get(camera, Orbit)!.target).toBe(s.entity);
    } finally {
        app.dispose();
    }
});
