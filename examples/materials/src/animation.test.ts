import { expect, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { AnimationPlugin, Gallery } from "./gallery";

test("the gallery animation clock accumulates stepped time", async () => {
    const app = await createApp({ defaults: false, plugins: [AnimationPlugin] });
    try {
        const world = app.world;
        world.step(1 / 60);
        world.step(1 / 60);
        expect(world.resource(Gallery).time).toBeCloseTo(2 / 60, 6);
    } finally {
        app.dispose();
    }
});
