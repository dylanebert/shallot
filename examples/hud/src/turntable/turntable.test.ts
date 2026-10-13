import { expect, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { Turn, TurntablePlugin } from "./index";

test("a set speed changes the angle advanced per step", async () => {
    const app = await createApp({ defaults: false, plugins: [TurntablePlugin] });
    try {
        const turn = app.world.resource(Turn);
        Object.assign(turn, { speed: 2.4 });
        app.world.step(1 / 60);
        expect(turn.angle).toBeCloseTo(2.4 / 60, 8);
    } finally {
        app.dispose();
    }
});

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
