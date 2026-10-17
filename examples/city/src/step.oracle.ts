import { test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { City, CityPlugin } from "./city";
import { POPULATION } from "./layout";

test("report shadowed city world.step CPU time on a real adapter", async () => {
    await setupGlobals();
    const app = await createApp({ plugins: [CityPlugin] });
    try {
        const { world } = app;
        if (world.gpu.adapter.class !== "real")
            throw new Error("city timing requires a real adapter");
        attachTexture(world, world.resource(City).camera, { width: 1280, height: 720 });
        const samples: number[] = [];
        for (let frame = 0; frame < 150; frame++) {
            const start = performance.now();
            world.step(1 / 60);
            const elapsed = performance.now() - start;
            if (frame >= 30) samples.push(elapsed);
            await world.gpu.sync();
        }
        samples.sort((a, b) => a - b);
        console.info({
            adapter: world.gpu.adapter,
            population: POPULATION,
            resolution: [1280, 720],
            shadows: true,
            warmupFrames: 30,
            measuredFrames: samples.length,
            medianWorldStepMs: (samples[59] + samples[60]) / 2,
            scope: "CPU world.step only; GPU completion awaited outside the timer; not a refresh or GPU frame measurement",
        });
    } finally {
        app.dispose();
    }
});
