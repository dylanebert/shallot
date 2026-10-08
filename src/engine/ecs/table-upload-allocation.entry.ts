import * as d from "typegpu/data";
import { createApp } from "../../engine";

export let controlSink: object;
export function control() {
    controlSink = { upload: 0 };
}

export default async function create(_input = "", device?: GPUDevice) {
    const app = await createApp({ defaults: false, plugins: [], device });
    const { world } = app;
    const table = world.table("allocation-uploads", d.struct({ value: d.vec4f }));
    table.acquire(world.create());
    const data = new Float32Array(4);
    const commands: GPUCommandBuffer[] = [];
    let encoder = world.gpu.device.createCommandEncoder();
    world.beginGpuFrame(encoder);
    return {
        step() {
            data[0]++;
            world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
            data[0]++;
            world.uploadGpuTable(table.buffer, 0, data.buffer, data.byteLength);
        },
        async wait() {
            commands[0] = encoder.finish();
            world.gpu.device.queue.submit(commands);
            world.endGpuFrame();
            await world.gpu.device.queue.onSubmittedWorkDone();
            encoder = world.gpu.device.createCommandEncoder();
            world.beginGpuFrame(encoder);
        },
        dispose() {
            app.dispose();
        },
    };
}
