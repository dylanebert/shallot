import { expect, setDefaultTimeout, test } from "bun:test";
import { build } from "../../engine";
import { ProfilePlugin, profile } from "./index";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

async function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
        return await Promise.race([
            promise,
            new Promise<never>((_, reject) => {
                timer = setTimeout(() => reject(new Error(`${label} exceeded 750 ms`)), 750);
            }),
        ]);
    } finally {
        clearTimeout(timer);
    }
}

test("profiler timestamps arrive through the world's one-shot pool and staging is reused", async () => {
    const app = await build({
        defaults: false,
        plugins: [
            {
                name: "TimestampWitness",
                dependencies: [ProfilePlugin],
                systems: [
                    {
                        group: "draw",
                        update(state) {
                            const timestampWrites = state.gpu.span?.("readback-witness");
                            if (!timestampWrites)
                                throw new Error("profiler did not supply timestamp writes");
                            const encoder = state.gpu.device.createCommandEncoder({
                                label: "readback-witness",
                            });
                            encoder.beginComputePass({ timestampWrites }).end();
                            state.gpu.device.queue.submit([encoder.finish()]);
                        },
                    },
                ],
            },
        ],
    });
    try {
        const stats = profile(app.state);
        const deadline = performance.now() + 750;
        while (!stats.gpuTime.has("readback-witness")) {
            if (performance.now() >= deadline)
                throw new Error("profiler timestamp delivery exceeded 750 ms");
            app.state.step(0);
            await bounded(
                "timestamp witness submissions",
                app.state.gpu.device.queue.onSubmittedWorkDone(),
            );
            await new Promise((resolve) => setTimeout(resolve, 1));
        }
        expect(stats.gpuFires.get("readback-witness")).toBeGreaterThan(0);
        expect(Number.isFinite(stats.gpuTime.get("readback-witness"))).toBe(true);
        expect(app.state.readback.allocated).toBeGreaterThan(0);
        expect(app.state.readback.allocated).toBeLessThanOrEqual(4);
    } finally {
        app.dispose();
    }
});
