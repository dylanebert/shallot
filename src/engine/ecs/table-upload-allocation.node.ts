import { expect, setDefaultTimeout, test } from "bun:test";
import * as d from "typegpu/data";
import { CEILING } from "../../../scripts/test-tiers";
import { createApp } from "../app";
import { setupGlobals } from "../runtime/webgpu";

await setupGlobals();

setDefaultTimeout(CEILING.node);
test("whole steady table frames reuse staging and only the existing completion promise and reaction", async () => {
    const app = await createApp({ defaults: false, plugins: [] });
    const { world } = app;
    const device = world.gpu.device;
    const table = world.table("whole-frame-uploads", d.struct({ value: d.vec4f }));
    table.acquire(world.create());
    const data = new Float32Array(4);
    const commands: GPUCommandBuffer[] = [];
    const frame = () => {
        const encoder = device.createCommandEncoder();
        world.beginGpuFrame(encoder);
        world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
        world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
        commands[0] = encoder.finish();
        device.queue.submit(commands);
        world.endGpuFrame();
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
    } finally {
        // biome-ignore lint/suspicious/noThenProperty: restore the native method after counting its reactions.
        Object.defineProperty(Promise.prototype, "then", thenDescriptor);
        if (fenceDescriptor) Object.defineProperty(queue, "onSubmittedWorkDone", fenceDescriptor);
        else Reflect.deleteProperty(queue, "onSubmittedWorkDone");
        world.own = own;
        app.dispose();
    }
});
