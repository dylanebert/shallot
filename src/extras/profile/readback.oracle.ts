import { expect, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../engine";
import { Profile, ProfilePlugin } from "./index";

await setupGlobals();

test("profiler timestamps arrive through the world's one-shot pool and staging is reused", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [
            {
                name: "TimestampWitness",
                dependencies: [ProfilePlugin],
                systems: [
                    {
                        group: "draw",
                        update(world) {
                            const timestampWrites = world.gpu.span?.("readback-witness");
                            if (!timestampWrites)
                                throw new Error("profiler did not supply timestamp writes");
                            const encoder = world.gpu.device.createCommandEncoder({
                                label: "readback-witness",
                            });
                            encoder.beginComputePass({ timestampWrites }).end();
                            world.gpu.device.queue.submit([encoder.finish()]);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const stats = app.world.resource(Profile);
        if (stats.gpuTiming !== "available") throw new Error("requires timestamp-query");
        const device = app.world.gpu.device;
        const lost = device.lost.then((info) => {
            throw new Error(`profiler device lost: ${info.message}`);
        });
        while (!stats.gpuTime.has("readback-witness")) {
            app.world.step(0);
            await Promise.race([device.queue.onSubmittedWorkDone(), lost]);
            await new Promise((resolve) => setTimeout(resolve, 1));
        }
        expect(stats.gpuFires.get("readback-witness")).toBeGreaterThan(0);
        expect(Number.isFinite(stats.gpuTime.get("readback-witness"))).toBe(true);
        expect(app.world.readback.allocated).toBeGreaterThan(0);
        expect(app.world.readback.allocated).toBeLessThanOrEqual(4);
    } finally {
        app.dispose();
    }
});
