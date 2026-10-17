import { expect, test } from "bun:test";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp, DiagnosticsStore } from "../../engine";
import { ProfilePlugin } from "./index";

await setupGlobals();
const spanLabels = ["readback-witness", "example//compute", "", "%2F"];

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
                            const encoder = world.gpu.device.createCommandEncoder({
                                label: "readback-witness",
                            });
                            for (const label of spanLabels) {
                                const timestampWrites = world.gpu.span?.(label);
                                if (!timestampWrites)
                                    throw new Error("profiler did not supply timestamp writes");
                                encoder.beginComputePass({ label, timestampWrites }).end();
                            }
                            world.gpu.device.queue.submit([encoder.finish()]);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const stats = app.world.resource(DiagnosticsStore);
        if (!app.world.gpu.device.features.has("timestamp-query"))
            throw new Error("requires timestamp-query");
        const device = app.world.gpu.device;
        const lost = device.lost.then((info) => {
            throw new Error(`profiler device lost: ${info.message}`);
        });
        while ((stats.get("render/readback-witness/elapsed_gpu")?.historyLength ?? 0) < 3) {
            app.world.step(0);
            await Promise.race([device.queue.onSubmittedWorkDone(), lost]);
            await new Promise((resolve) => setTimeout(resolve, 1));
        }
        const diagnostic = stats.get("render/readback-witness/elapsed_gpu")!;
        const history = [...diagnostic.measurements()];
        expect(history.length).toBeGreaterThanOrEqual(3);
        let smoothed = history[0].value;
        for (let i = 1; i < history.length; i++) {
            const alpha = Math.min(
                1,
                Math.max(0, (history[i].time - history[i - 1].time) / (2 / 21)),
            );
            smoothed += alpha * (history[i].value - smoothed);
        }
        expect(diagnostic.smoothed()).toBeCloseTo(smoothed, 10);
        expect(Number.isFinite(stats.get("render/readback-witness/elapsed_gpu")!.value())).toBe(
            true,
        );
        for (const label of ["example%2F%2Fcompute", "%", "%252F"]) {
            expect(stats.get(`render/${label}/elapsed_gpu`)!.historyLength).toBeGreaterThanOrEqual(
                3,
            );
        }
        expect(app.world.readback.allocated).toBeGreaterThan(0);
        expect(app.world.readback.allocated).toBeLessThanOrEqual(4);

        // A rejected timestamp copy must fail the same frame wait that drains it, not log and leave
        // the timestamp loop spinning until its process ceiling.
        const pool = app.world.readback;
        const request = pool.request.bind(pool);
        let injected = false;
        pool.request = (size, label, encode) => {
            if (label === "profile-timestamps" && !injected) {
                injected = true;
                return Promise.reject(new Error("injected timestamp validation failure"));
            }
            return request(size, label, encode);
        };
        app.world.step(0);
        await Promise.resolve();
        expect(injected).toBe(true);
        expect(() => app.world.step(0)).toThrow("profile timestamp readback failed");
        pool.request = request;
    } finally {
        app.dispose();
    }
});
