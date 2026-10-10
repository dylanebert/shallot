import { expect, setDefaultTimeout, test } from "bun:test";
import { d } from "typegpu";
import { disposeGpuApps, gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { rawDevice } from "./gpu";
import { probeBuffer, probeTexture } from "./probe";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(
    import.meta.path,
    Array.from({ length: 7 }, () => ({ defaults: false, plugins: [] })),
);

test("TypeGPU native buffers and textures belong to their world, and external allocations require explicit ownership", async () => {
    const app = subjects()[0];
    const world = app.world;
    try {
        const typed = world.gpu.root.createBuffer(d.arrayOf(d.u32, 1), [29]).$usage("storage");
        const buffer = world.gpu.root.unwrap(typed);
        const texture = world.gpu.root.unwrap(
            world.gpu.root.createTexture({ size: [1, 1], format: "rgba8unorm" }).$usage("sampled"),
        );
        expect(world.owns(buffer)).toBe(true);
        expect(world.owns(texture)).toBe(true);
        expect(new Uint32Array((await probeBuffer(world, buffer)).bytes)[0]).toBe(29);
        expect((await probeTexture(world, texture)).bytes.byteLength).toBe(4);
        const external = rawDevice(world.gpu.device).createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_SRC,
        });
        try {
            await expect(probeBuffer(world, external)).rejects.toThrow("not owned by this world");
            world.own(external);
            await probeBuffer(world, external);
            external.destroy();
            expect(world.owns(external)).toBe(false);
            await expect(probeBuffer(world, external)).rejects.toThrow("not owned by this world");
        } finally {
            external.destroy();
        }
    } finally {
        await disposeGpuApps([app]);
    }
});

for (const [index, kind] of (["buffer", "range", "texture"] as const).entries()) {
    test(`a ${kind} request refuses another world's resource on the same device, naming it before encoding`, async () => {
        const owner = subjects()[1 + index * 2];
        const reader = subjects()[2 + index * 2];
        const buffer = owner.world.gpu.device.createBuffer({
            label: "other-world-buffer",
            size: 16,
            usage: GPUBufferUsage.COPY_SRC,
        });
        const texture = owner.world.gpu.device.createTexture({
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
                    ? probeTexture(reader.world, texture, { encode })
                    : probeBuffer(reader.world, buffer, {
                          offset: kind === "range" ? 4 : 0,
                          size: 4,
                          encode,
                      });
            await expect(request).rejects.toThrow(
                `other-world-${kind === "texture" ? "texture" : "buffer"}`,
            );
            expect(encoded).toBe(0);
            expect(reader.world.readback.allocated).toBe(0);
            await expect(
                reader.world.readback.request(
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
            expect(reader.world.readback.allocated).toBe(0);
        } finally {
            await disposeGpuApps([owner, reader]);
        }
    });
}
