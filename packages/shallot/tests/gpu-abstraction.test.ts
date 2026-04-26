import { initGPU, shouldSkipGPU } from "./helpers/gpu";
import { describe, expect, test, beforeAll } from "bun:test";
import { trackDevice, type GpuRegistry } from "../src/standard/compute/profile";

describe("GPU registry", () => {
    let device: GPUDevice | null = null;
    let registry: GpuRegistry;

    beforeAll(async () => {
        if (shouldSkipGPU()) return;
        const ctx = await initGPU();
        device = ctx.device;
        registry = trackDevice(device);
    });

    test("tracked device creates and destroys buffers", () => {
        if (!device) return;

        const buffer = device.createBuffer({
            label: "test",
            size: 64,
            usage: GPUBufferUsage.STORAGE,
        });
        expect(registry.buffers.size).toBe(1);

        buffer.destroy();
        expect(registry.buffers.size).toBe(0);
    });

    test("tracked device delegates feature queries", () => {
        if (!device) return;

        expect(typeof device.features.has).toBe("function");
        expect(typeof device.limits.maxComputeWorkgroupsPerDimension).toBe("number");
    });

    test("tracked device readback works end-to-end", async () => {
        if (!device) return;

        const buffer = device.createBuffer({
            label: "readback-test",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        device.queue.writeBuffer(buffer, 0, new Float32Array([1, 2, 3, 4]));

        const staging = device.createBuffer({
            label: "staging",
            size: 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(buffer, 0, staging, 0, 16);
        device.queue.submit([encoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();

        expect(Array.from(data)).toEqual([1, 2, 3, 4]);

        buffer.destroy();
        staging.destroy();
        expect(registry.buffers.size).toBe(0);
    });
});
