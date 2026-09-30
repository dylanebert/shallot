import { expect, setDefaultTimeout, test } from "bun:test";
import { d } from "typegpu";
import { sharedGpuBuild } from "../app/gpu.fixture";
import { rawDevice } from "./gpu";
import { probeBuffer, probeTexture } from "./probe";

setDefaultTimeout(1000);
const peer = "bun-webgpu";
await (await import(peer)).setupGlobals();
const build = await sharedGpuBuild();

test("TypeGPU native buffers and textures belong to their world, and external allocations require explicit ownership", async () => {
    const app = await build({ defaults: false, plugins: [] });
    const state = app.state;
    try {
        const typed = state.gpu.root.createBuffer(d.arrayOf(d.u32, 1), [29]).$usage("storage");
        const buffer = state.gpu.root.unwrap(typed);
        const texture = state.gpu.root.unwrap(
            state.gpu.root.createTexture({ size: [1, 1], format: "rgba8unorm" }).$usage("sampled"),
        );
        expect(state.owns(buffer)).toBe(true);
        expect(state.owns(texture)).toBe(true);
        expect(new Uint32Array((await probeBuffer(state, buffer)).bytes)[0]).toBe(29);
        expect((await probeTexture(state, texture)).bytes.byteLength).toBe(4);
        const external = rawDevice(state.gpu.device).createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
        });
        try {
            await expect(probeBuffer(state, external)).rejects.toThrow("not owned by this world");
            state.own(external);
            await probeBuffer(state, external);
            external.destroy();
            expect(state.owns(external)).toBe(false);
            await expect(probeBuffer(state, external)).rejects.toThrow("not owned by this world");
        } finally {
            external.destroy();
        }
    } finally {
        app.dispose();
    }
});

for (const kind of ["buffer", "range", "texture"] as const) {
    test(`a ${kind} request refuses another world's resource on the same device, naming it before encoding`, async () => {
        const owner = await build({ defaults: false, plugins: [] });
        const reader = await build({
            defaults: false,
            plugins: [],
            device: owner.state.gpu.device,
        });
        const buffer = owner.state.gpu.device.createBuffer({
            label: "other-world-buffer",
            size: 16,
            usage: GPUBufferUsage.COPY_SRC,
        });
        const texture = owner.state.gpu.device.createTexture({
            label: "other-world-texture",
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.COPY_SRC,
        });
        let encoded = 0;
        const encode = () => {
            encoded++;
        };
        try {
            const request =
                kind === "texture"
                    ? probeTexture(reader.state, texture, { encode })
                    : probeBuffer(reader.state, buffer, {
                          offset: kind === "range" ? 4 : 0,
                          size: 4,
                          encode,
                      });
            await expect(request).rejects.toThrow(
                `other-world-${kind === "texture" ? "texture" : "buffer"}`,
            );
            expect(encoded).toBe(0);
            expect(reader.state.readback.allocated).toBe(0);
            await expect(
                reader.state.readback.request(
                    kind === "texture" ? 256 : 4,
                    "foreign copy",
                    (encoder, staging) => {
                        if (kind === "texture")
                            encoder.copyTextureToBuffer(
                                { texture },
                                { buffer: staging, bytesPerRow: 256 },
                                [1, 1],
                            );
                        else
                            encoder.copyBufferToBuffer(
                                buffer,
                                kind === "range" ? 4 : 0,
                                staging,
                                0,
                                4,
                            );
                    },
                ),
            ).rejects.toThrow(`other-world-${kind === "texture" ? "texture" : "buffer"}`);
            expect(reader.state.readback.allocated).toBe(0);
        } finally {
            reader.dispose();
            owner.dispose();
        }
    });
}
