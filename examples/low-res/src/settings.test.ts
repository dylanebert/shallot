import { expect, test } from "bun:test";
import { createApp, Resolution } from "@dylanebert/shallot";
import { Posterize } from "./posterize";
import { Scene } from "./scene";
import { setDither, setSetting } from "./settings";

test("settings write resolution height, colour levels and dither on the camera", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "Settings",
                components: [Resolution, Posterize],
                initialize(world) {
                    const camera = world.create();
                    world.resource(Scene).camera = camera;
                    world.add(camera, Resolution, { height: 360 });
                    world.add(camera, Posterize);
                },
            },
        ],
    });
    try {
        const { world } = app;
        const camera = world.resource(Scene).camera;
        setSetting(world, 0, 240);
        setSetting(world, 1, 4);
        setDither(world, false);
        expect(world.storage(Resolution).height.get(camera)).toBe(240);
        expect(world.storage(Posterize).levels.get(camera)).toBe(4);
        expect(world.storage(Posterize).dither.get(camera)).toBe(0);
        setDither(world, true);
        expect(world.storage(Posterize).dither.get(camera)).toBe(1);
    } finally {
        app.dispose();
    }
});
