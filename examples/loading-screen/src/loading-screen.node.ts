import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, createApp } from "@dylanebert/shallot";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { CEILING } from "../../../scripts/test-tiers";
import { LoadingWorld } from "./scene";

setDefaultTimeout(CEILING.node);

import { setupGlobals } from "@dylanebert/shallot/webgpu";

await setupGlobals();

test("loading-screen draws on the real default eight-buffer device", async () => {
    const app = await createApp({ plugins: [LoadingWorld] });
    try {
        const { world } = app;
        const device = world.gpu.device;
        if (!device) throw new Error("loading-screen renderer did not acquire a device");
        const adapter = world.gpu.adapter;
        console.info("loading-screen floor", {
            adapter,
            features: [...device.features],
            maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
            maxUniformBufferBindingSize: device.limits.maxUniformBufferBindingSize,
        });
        expect(adapter.class).toBe("real");
        expect(device.limits.maxStorageBuffersPerShaderStage).toBe(8);
        expect(
            [...device.features].filter((feature) => feature !== "core-features-and-limits").sort(),
        ).toEqual(["indirect-first-instance", "rg11b10ufloat-renderable"]);
        const camera = [...world.query([Camera])][0];
        if (camera === undefined) throw new Error("loading-screen scene has no camera");
        attachTexture(world, camera, { width: 64, height: 64 });
        world.step(0);
        const frame = await captureTexture(world, camera);
        expect(frame.rgba.some((value, index) => value !== frame.rgba[index % 4])).toBe(true);
        await device.queue.onSubmittedWorkDone();
    } finally {
        app.dispose();
    }
});
