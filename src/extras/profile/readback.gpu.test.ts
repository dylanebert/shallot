import { expect, setDefaultTimeout, test } from "bun:test";
import { build } from "../../engine";
import { ProfilePlugin, profile } from "./index";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();

test("a profiler without timestamp-query runs and distinguishes missing GPU timings from zero", async () => {
    const owner = await build({ defaults: false, plugins: [] });
    let app: Awaited<ReturnType<typeof build>> | undefined;
    try {
        expect(owner.state.gpu.device.features.has("timestamp-query")).toBe(false);
        app = await build({
            defaults: false,
            plugins: [ProfilePlugin],
            device: owner.state.gpu.device,
        });
        const stats = profile(app.state);
        app.state.step(0);
        expect(stats.gpuTiming).toBe("requires timestamp-query");
        expect(app.state.gpu.span?.("untimed")).toBeUndefined();
        expect(stats.gpu.size).toBe(0);
        expect(stats.gpuTime.size).toBe(0);
        expect(stats.gpuFires.size).toBe(0);
        expect(app.state.readback.allocated).toBe(0);
        const before = stats.bufferBytes;
        const buffer = app.state.gpu.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
        });
        expect(stats.bufferBytes).toBe(before + 16);
        buffer.destroy();
        expect(stats.bufferBytes).toBe(before);
    } finally {
        app?.dispose();
        owner.dispose();
    }
});
