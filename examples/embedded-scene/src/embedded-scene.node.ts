import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp as createApp } from "@dylanebert/shallot/testing/gpu";
import { LoadingWorld } from "./scene";

setDefaultTimeout(TIMEOUT.node);

test("embedded-scene draws with default WebGPU limits", async () => {
    const app = await createApp({ plugins: [LoadingWorld] });
    try {
        const { world } = app;
        const device = world.gpu.device;
        if (!device) throw new Error("embedded-scene renderer did not acquire a device");
        const adapter = world.gpu.adapter;
        console.info("embedded-scene floor", {
            adapter,
            features: [...device.features],
            maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
            maxUniformBufferBindingSize: device.limits.maxUniformBufferBindingSize,
        });
        expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
        expect(device.features.has("indirect-first-instance")).toBe(true);
        const camera = [...world.query([Camera])][0];
        if (camera === undefined) throw new Error("embedded-scene scene has no camera");
        attachTexture(world, camera, { width: 64, height: 64 });
        world.step(0);
        const frame = await captureTexture(world, camera);
        console.info("rendered clear", Array.from(frame.rgba.slice(0, 4)));
        expect(frame.rgba.some((value, index) => value !== frame.rgba[index % 4])).toBe(true);
        await device.queue.onSubmittedWorkDone();
    } finally {
        app.dispose();
    }
});
