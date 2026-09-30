import { expect, setDefaultTimeout, test } from "bun:test";

setDefaultTimeout(1000);

import { build, globalTransformTable, Transform } from "../../engine";
import { Part, partTable } from "./part";
import "../../standard";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 1000 ms`)), 1000);
        promise.then(
            (value) => {
                clearTimeout(timer);
                resolve(value);
            },
            (error) => {
                clearTimeout(timer);
                reject(error);
            },
        );
    });
}

test("Part compaction carries independent dense Transform and Part slots with each logical eid", async () => {
    const app = await build({ plugins: [] });
    const state = app.state;
    const device = state.gpu.device;
    const readback = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        for (let i = 0; i < 1000; i++) state.create();
        const extra = state.create();
        state.add(extra, Transform);
        const a = state.create();
        const b = state.create();
        state.add(a, Transform);
        state.add(b, Transform);
        // Different membership order forces unrelated row slots.
        state.add(b, Part);
        state.add(a, Part);
        const transforms = globalTransformTable(state);
        const parts = partTable(state);
        expect(transforms.rowIndex(b)).not.toBe(parts.rowIndex(b));
        device.pushErrorScope("validation");
        state.step();
        const instances = state.gpu.buffers.get("eids");
        if (!instances) throw new Error("Part did not publish its instance list");
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(instances, 0, readback, 0, 32);
        device.queue.submit([encoder.finish()]);
        const error = await bounded("Part payload validation", device.popErrorScope());
        if (error) throw new Error(error.message);
        await bounded("Part payload readback", readback.mapAsync(GPUMapMode.READ));
        const words = new Uint32Array(readback.getMappedRange());
        const records = [Array.from(words.subarray(0, 4)), Array.from(words.subarray(4, 8))].sort(
            (x, y) => x[0]! - y[0]!,
        );
        expect(records).toEqual(
            [a, b].map((eid) => [eid, transforms.rowIndex(eid), parts.rowIndex(eid) + 1, 0]),
        );
        readback.unmap();
    } finally {
        readback.destroy();
        app.dispose();
    }
}, 200);
