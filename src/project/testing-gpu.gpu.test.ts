import { expect, setDefaultTimeout, test } from "bun:test";
import { create } from "webgpu";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import { setupGlobals } from "@dylanebert/shallot/webgpu";

setDefaultTimeout(TIMEOUT.gpu);

for (const missing of [false, true]) {
    test(`the GPU opener refuses ${missing ? "a missing adapter" : "Dawn's null backend"} before requesting a device`, async () => {
        await setupGlobals();
        const gpu = create(["backend=null"]);
        const adapter = await gpu.requestAdapter();
        expect(adapter).not.toBeNull();
        expect(adapter!.info.device).toBe("null-backend");
        let requests = 0;
        const requestDevice = adapter!.requestDevice;
        adapter!.requestDevice = async (descriptor) => {
            requests++;
            return requestDevice.call(adapter, descriptor);
        };
        const requestAdapter = navigator.gpu.requestAdapter;
        navigator.gpu.requestAdapter = async () => (missing ? null : adapter);
        try {
            await expect(
                openGpuApp({ defaults: false, plugins: [{ name: "GpuCheck", gpu: {} }] }),
            ).rejects.toThrow(missing ? "No WebGPU adapter" : "null-backend");
            expect(requests).toBe(0);
        } finally {
            navigator.gpu.requestAdapter = requestAdapter;
        }
    });
}
