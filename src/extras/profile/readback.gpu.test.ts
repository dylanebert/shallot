import { expect, setDefaultTimeout, test } from "bun:test";
import { compileGpuFile, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../../engine";
import { Profile, ProfilePlugin } from "./index";

setDefaultTimeout(CEILING.gpu);
const subject = compileGpuFile(import.meta.path, async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("profiler adapter unavailable");
    const requiredFeatures = ["indirect-first-instance", "rg11b10ufloat-renderable"] as const;
    const device = await adapter.requestDevice({ requiredFeatures: [...requiredFeatures] });
    expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    expect(
        [...device.features].filter((feature) => feature !== "core-features-and-limits").sort(),
    ).toEqual([...requiredFeatures].sort());
    const app = await createApp({ defaults: false, plugins: [ProfilePlugin], device, adapter });
    return { app, device };
});

test("a profiler without timestamp-query runs and distinguishes missing GPU timings from zero", async () => {
    const { app, device } = subject();
    try {
        expect(app.world.gpu.device.features.has("timestamp-query")).toBe(false);
        const stats = app.world.resource(Profile);
        app.world.step(0);
        expect(stats.gpuTiming).toBe("requires timestamp-query");
        expect(app.world.gpu.span?.("untimed")).toBeUndefined();
        expect(stats.gpu.size).toBe(0);
        expect(stats.gpuTime.size).toBe(0);
        expect(stats.gpuFires.size).toBe(0);
        expect(app.world.readback.allocated).toBe(0);
        const before = stats.bufferBytes;
        const buffer = app.world.gpu.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
        });
        expect(stats.bufferBytes).toBe(before + 16);
        buffer.destroy();
        expect(stats.bufferBytes).toBe(before);
    } finally {
        await disposeGpuApps([app]);
        device.destroy();
    }
});
