import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { compileGpuFile, createTestGpuApp, disposeGpuApps } from "../../../scripts/gpu.fixture";
import { expectError } from "../../../scripts/test-console";
import { DiagnosticsStore } from "../../engine";
import { ProfilePlugin } from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const subject = compileGpuFile(import.meta.path, async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("profiler adapter unavailable");
    const requiredFeatures = ["indirect-first-instance"] as const;
    const device = await adapter.requestDevice({ requiredFeatures: [...requiredFeatures] });
    expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
    expect(
        [...device.features].filter((feature) => feature !== "core-features-and-limits").sort(),
    ).toEqual([...requiredFeatures].sort());
    const app = await createTestGpuApp({
        defaults: false,
        plugins: [ProfilePlugin],
        device,
        adapter,
    });
    return { app, device };
});

test("a profiler without timestamp-query runs and distinguishes missing GPU timings from zero", async () => {
    const { app, device } = subject();
    try {
        expect(app.world.gpu.device.features.has("timestamp-query")).toBe(false);
        const stats = app.world.resource(DiagnosticsStore);
        app.world.step(0);
        expect(app.world.gpu.span?.("untimed")).toBeUndefined();
        expect(stats.get("render/untimed/elapsed_gpu")).toBeUndefined();
        expect(app.world.readback.allocated).toBe(0);
        const before = stats.get("memory/bufferBytes")?.value() ?? 0;
        const buffer = app.world.gpu.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.COPY_DST,
        });
        expect(stats.get("memory/bufferBytes")?.value()).toBe(before + 16);
        buffer.destroy();
        expect(stats.get("memory/bufferBytes")?.value()).toBe(before);
    } finally {
        await disposeGpuApps([app]);
        expectError(/GPU device lost destroyed: device was destroyed/);
        device.destroy();
    }
});
