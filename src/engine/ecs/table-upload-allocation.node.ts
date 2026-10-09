import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../app";
import { setupGlobals } from "../runtime/webgpu";

await setupGlobals();

setDefaultTimeout(CEILING.node);
test("whole steady table frames reuse copy records and staging with one completion reaction", async () => {
    const app = await createApp({
        defaults: false,
        plugins: [{ name: "TableUploadAllocation", gpu: {} }],
    });
    const { world } = app;
    const device = world.gpu.device;
    const table = world.table("whole-frame-uploads", d.struct({ value: d.vec4f }));
    table.acquire(world.create());
    const data = new Float32Array(4);
    world.addSystem({
        group: "draw",
        update(world) {
            world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
            world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
        },
    });
    const frame = () => {
        world.step(0);
        return world.frameFence!;
    };
    for (let i = 0; i < 3; i++) await frame();
    let promises = 0,
        reactions = 0,
        buffers = 0;
    const queue = device.queue;
    const fence = queue.onSubmittedWorkDone.bind(queue);
    const own = world.own;
    const ownBound = own.bind(world);
    const fenceDescriptor = Object.getOwnPropertyDescriptor(queue, "onSubmittedWorkDone");
    // Pool growth and identity expose steady per-upload copy-record allocations directly.
    const copyPool = (world as unknown as { _frameCopies: object[] })._frameCopies;
    const pooledRecords = copyPool.slice();
    let copyRecordsAdded = 0;
    const copyPush = copyPool.push.bind(copyPool);
    const copyPushDescriptor = Object.getOwnPropertyDescriptor(copyPool, "push");
    Object.defineProperty(copyPool, "push", {
        configurable: true,
        value: (...records: object[]) => {
            copyRecordsAdded += records.length;
            return copyPush(...records);
        },
    });
    const then = Promise.prototype.then;
    Object.defineProperty(queue, "onSubmittedWorkDone", {
        configurable: true,
        value: () => {
            promises++;
            return fence();
        },
    });
    world.own = (resource) => {
        buffers++;
        ownBound(resource);
    };
    const thenDescriptor = Object.getOwnPropertyDescriptor(Promise.prototype, "then")!;
    // biome-ignore lint/suspicious/noThenProperty: count native promise reactions, not a new thenable.
    Object.defineProperty(Promise.prototype, "then", {
        value: function (this: Promise<unknown>, ...args: Parameters<typeof then>) {
            reactions++;
            return Reflect.apply(then, this, args);
        },
    });
    try {
        for (let i = 0; i < 8; i++) {
            const beforePromises = promises,
                beforeReactions = reactions;
            await frame();
            expect(promises - beforePromises).toBe(1);
            expect(reactions - beforeReactions).toBe(1);
        }
        expect(buffers).toBe(0);
        expect(copyRecordsAdded).toBe(0);
        expect(copyPool).toHaveLength(2);
        expect(copyPool[0]).toBe(pooledRecords[0]);
        expect(copyPool[1]).toBe(pooledRecords[1]);
    } finally {
        // biome-ignore lint/suspicious/noThenProperty: restore the native method after counting its reactions.
        Object.defineProperty(Promise.prototype, "then", thenDescriptor);
        if (fenceDescriptor) Object.defineProperty(queue, "onSubmittedWorkDone", fenceDescriptor);
        else Reflect.deleteProperty(queue, "onSubmittedWorkDone");
        if (copyPushDescriptor) Object.defineProperty(copyPool, "push", copyPushDescriptor);
        else Reflect.deleteProperty(copyPool, "push");
        world.own = own;
        app.dispose();
    }
});
