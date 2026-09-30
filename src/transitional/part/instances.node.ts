import { expect, setDefaultTimeout, test } from "bun:test";

import { CEILING } from "../../../scripts/test-tiers";

setDefaultTimeout(CEILING.node);

import { createApp, globalTransformTable, Transform } from "../../engine";
import { MeshInstance, partTable } from "./part";
import "../../standard";

const peerModule = "bun-webgpu";
const { setupGlobals } = (await import(peerModule)) as { setupGlobals(): Promise<void> };
await setupGlobals();

function bounded<T>(label: string, promise: PromiseLike<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`${label} timed out after 5000 ms`)), 5000);
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

test("MeshInstance compaction carries independent dense GlobalTransform and MeshInstance slots with each logical eid", async () => {
    const app = await createApp({ plugins: [] });
    const world = app.world;
    const device = world.gpu.device;
    const readback = device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        for (let i = 0; i < 1000; i++) world.create();
        const extra = world.create();
        world.add(extra, Transform);
        const a = world.create();
        const b = world.create();
        world.add(a, Transform);
        world.add(b, Transform);
        // Different membership order forces unrelated row slots.
        world.add(b, MeshInstance);
        world.add(a, MeshInstance);
        const globalTransforms = globalTransformTable(world);
        const parts = partTable(world);
        expect(globalTransforms.rowIndex(b)).not.toBe(parts.rowIndex(b));
        device.pushErrorScope("validation");
        world.step();
        const instances = world.gpu.buffers.get("eids");
        if (!instances) throw new Error("MeshInstance did not publish its instance list");
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(instances, 0, readback, 0, 32);
        device.queue.submit([encoder.finish()]);
        const error = await bounded("MeshInstance payload validation", device.popErrorScope());
        if (error) throw new Error(error.message);
        await bounded("MeshInstance payload readback", readback.mapAsync(GPUMapMode.READ));
        const words = new Uint32Array(readback.getMappedRange());
        const records = [Array.from(words.subarray(0, 4)), Array.from(words.subarray(4, 8))].sort(
            (x, y) => x[0]! - y[0]!,
        );
        expect(records).toEqual(
            [a, b].map((eid) => [eid, globalTransforms.rowIndex(eid), parts.rowIndex(eid) + 1, 0]),
        );
        readback.unmap();
    } finally {
        readback.destroy();
        app.dispose();
    }
});
