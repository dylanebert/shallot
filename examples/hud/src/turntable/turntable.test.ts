import { expect, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { Turn, TurntablePlugin } from "./index";

test("the turntable's angle advances when the world steps", async () => {
    const app = await createApp({ defaults: false, plugins: [TurntablePlugin] });
    try {
        const turn = app.world.resource(Turn);
        const before = turn.angle;
        app.world.step(1 / 60);
        expect(turn.angle).toBeGreaterThan(before);
    } finally {
        app.dispose();
    }
});
