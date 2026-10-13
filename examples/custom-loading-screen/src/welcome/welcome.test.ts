import { expect, test } from "bun:test";
import { createApp, Transform } from "@dylanebert/shallot";
import { Welcome, WelcomeCube } from "./index";

test("the welcome cube turns when the world steps", async () => {
    const app = await createApp({ defaults: false, plugins: [Welcome] });
    try {
        const cube = [...app.world.query([WelcomeCube])][0];
        expect(cube).toBeDefined();
        const rotation = app.world.storage(Transform).rotation;
        const before = [rotation.y.get(cube!), rotation.w.get(cube!)];

        app.world.step(1 / 60);

        expect([rotation.y.get(cube!), rotation.w.get(cube!)]).not.toEqual(before);
    } finally {
        app.dispose();
    }
});
