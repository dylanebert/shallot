import { describe, expect, test } from "bun:test";
import { probeBuffer, probeTexture } from "./probe";

interface FakeBuffer extends GPUBuffer {
    backing: ArrayBuffer;
    destroyed: boolean;
}

interface FakeTexture extends GPUTexture {
    pixels: Uint8Array;
}

function fakeDevice(events: string[], maxBufferSize = 1 << 30): GPUDevice {
    const commands: (() => void)[] = [];
    const device = {
        createBuffer(descriptor: GPUBufferDescriptor) {
            events.push("create-buffer");
            let state: GPUBufferMapState = "unmapped";
            const buffer = {
                backing: new ArrayBuffer(Number(descriptor.size)),
                destroyed: false,
                label: descriptor.label ?? "",
                size: Number(descriptor.size),
                usage: descriptor.usage,
                get mapState() {
                    return state;
                },
                async mapAsync() {
                    events.push("map");
                    state = "mapped";
                },
                getMappedRange(offset = 0, size = buffer.size - offset) {
                    return buffer.backing.slice(offset, offset + size);
                },
                unmap() {
                    events.push("unmap");
                    state = "unmapped";
                },
                destroy() {
                    events.push("destroy");
                    buffer.destroyed = true;
                },
            } as unknown as FakeBuffer;
            return buffer;
        },
        createCommandEncoder() {
            return {
                copyBufferToBuffer(
                    source: FakeBuffer,
                    sourceOffset: number,
                    destination: FakeBuffer,
                    destinationOffset: number,
                    size: number,
                ) {
                    events.push("copy-buffer");
                    commands.push(() => {
                        new Uint8Array(destination.backing, destinationOffset, size).set(
                            new Uint8Array(source.backing, sourceOffset, size),
                        );
                    });
                },
                copyTextureToBuffer(
                    source: GPUImageCopyTexture,
                    destination: GPUImageCopyBuffer,
                    size: Required<GPUExtent3DDict>,
                ) {
                    events.push("copy-texture");
                    commands.push(() => {
                        const texture = source.texture as FakeTexture;
                        const buffer = destination.buffer as FakeBuffer;
                        const rowBytes =
                            texture.pixels.byteLength / (size.height * size.depthOrArrayLayers);
                        const out = new Uint8Array(buffer.backing);
                        for (let row = 0; row < size.height * size.depthOrArrayLayers; row++) {
                            out.set(
                                texture.pixels.subarray(row * rowBytes, (row + 1) * rowBytes),
                                row * (destination.bytesPerRow ?? rowBytes),
                            );
                        }
                    });
                },
                finish() {
                    return {} as GPUCommandBuffer;
                },
            } as GPUCommandEncoder;
        },
        queue: {
            submit() {
                events.push("submit");
                for (const command of commands.splice(0)) command();
            },
            async onSubmittedWorkDone() {
                events.push("fence");
            },
        },
        limits: { maxBufferSize },
    };
    return device as unknown as GPUDevice;
}

function buffer(bytes: number[], usage = GPUBufferUsage.COPY_SRC): FakeBuffer {
    const backing = Uint8Array.from(bytes).buffer;
    return {
        backing,
        destroyed: false,
        label: "source",
        size: backing.byteLength,
        usage,
        mapState: "unmapped",
    } as FakeBuffer;
}

function texture(
    format: GPUTextureFormat,
    width: number,
    height: number,
    pixels: Uint8Array,
    usage = GPUTextureUsage.COPY_SRC,
): FakeTexture {
    return {
        label: "source-texture",
        width,
        height,
        depthOrArrayLayers: 1,
        mipLevelCount: 1,
        sampleCount: 1,
        dimension: "2d",
        format,
        usage,
        pixels,
    } as FakeTexture;
}

describe("one-shot GPU probes", () => {
    test("buffer snapshots align the copy but own only the requested bytes", async () => {
        const events: string[] = [];
        const device = fakeDevice(events);
        const source = buffer([0, 1, 2, 3, 4, 5, 6, 7]);
        const first = await probeBuffer(device, source, {
            offset: 1,
            size: 5,
            encode() {
                events.push("encode");
            },
        });

        new Uint8Array(source.backing).fill(9);
        const second = await probeBuffer(device, source, { offset: 1, size: 5 });

        expect([...new Uint8Array(first.bytes)]).toEqual([1, 2, 3, 4, 5]);
        expect([...new Uint8Array(second.bytes)]).toEqual([9, 9, 9, 9, 9]);
        expect(first).toEqual({ bytes: first.bytes, offset: 1, size: 5 });
        expect(Object.isFrozen(first)).toBe(true);
        expect(events.slice(0, 8)).toEqual([
            "create-buffer",
            "encode",
            "copy-buffer",
            "submit",
            "fence",
            "map",
            "unmap",
            "destroy",
        ]);
    });

    test("color texture snapshots strip 256-byte row padding", async () => {
        const events: string[] = [];
        const device = fakeDevice(events);
        const pixels = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        const result = await probeTexture(device, texture("rgba8unorm", 2, 2, pixels));

        expect([...new Uint8Array(result.bytes)]).toEqual([...pixels]);
        expect(result).toMatchObject({
            format: "rgba8unorm",
            aspect: "all",
            width: 2,
            height: 2,
            depthOrArrayLayers: 1,
            bytesPerRow: 8,
        });
        expect(events).toEqual([
            "create-buffer",
            "copy-texture",
            "submit",
            "fence",
            "map",
            "unmap",
            "destroy",
        ]);
    });

    test("all uncompressed color format classes use their portable texel byte size", async () => {
        const formats: [GPUTextureFormat, number][] = [
            ["r8unorm", 1],
            ["r8snorm", 1],
            ["r8uint", 1],
            ["r8sint", 1],
            ["r16unorm", 2],
            ["r16snorm", 2],
            ["r16uint", 2],
            ["r16sint", 2],
            ["r16float", 2],
            ["rg8unorm", 2],
            ["rg8snorm", 2],
            ["rg8uint", 2],
            ["rg8sint", 2],
            ["r32uint", 4],
            ["r32sint", 4],
            ["r32float", 4],
            ["rg16unorm", 4],
            ["rg16snorm", 4],
            ["rg16uint", 4],
            ["rg16sint", 4],
            ["rg16float", 4],
            ["rgba8unorm", 4],
            ["rgba8unorm-srgb", 4],
            ["rgba8snorm", 4],
            ["rgba8uint", 4],
            ["rgba8sint", 4],
            ["bgra8unorm", 4],
            ["bgra8unorm-srgb", 4],
            ["rgb9e5ufloat", 4],
            ["rgb10a2uint", 4],
            ["rgb10a2unorm", 4],
            ["rg11b10ufloat", 4],
            ["rg32uint", 8],
            ["rg32sint", 8],
            ["rg32float", 8],
            ["rgba16unorm", 8],
            ["rgba16snorm", 8],
            ["rgba16uint", 8],
            ["rgba16sint", 8],
            ["rgba16float", 8],
            ["rgba32uint", 16],
            ["rgba32sint", 16],
            ["rgba32float", 16],
        ];
        for (const [format, bytes] of formats) {
            const result = await probeTexture(
                fakeDevice([]),
                texture(format, 1, 1, new Uint8Array(bytes)),
            );
            expect(result.bytesPerRow).toBe(bytes);
            expect(result.aspect).toBe("all");
        }
    });

    // The five rejects.toThrow calls were un-awaited while every sibling in the file awaits
    // them. Awaiting makes the arm's failure independent of the matcher's settle-time behavior
    // rather than newly possible — bun 1.4.0 reds either way, so this is hygiene and a
    // version-scoped guarantee, not a repair of an unfailable arm.
    test("depth/stencil defaults are unambiguous and combined formats require one aspect", async () => {
        const device = fakeDevice([]);
        const values = new Float32Array([1, 0.25]);
        const result = await probeTexture(
            device,
            texture("depth32float", 2, 1, new Uint8Array(values.buffer)),
        );
        expect([...new Float32Array(result.bytes)]).toEqual([1, 0.25]);
        expect(result.bytesPerRow).toBe(8);
        expect(result.aspect).toBe("depth-only");

        const combinedDepth = await probeTexture(
            device,
            texture("depth32float-stencil8", 1, 1, new Uint8Array(4)),
            { aspect: "depth-only" },
        );
        const combinedStencil = await probeTexture(
            device,
            texture("depth24plus-stencil8", 1, 1, new Uint8Array(1)),
            { aspect: "stencil-only" },
        );
        expect(combinedDepth.bytesPerRow).toBe(4);
        expect(combinedStencil.bytesPerRow).toBe(1);

        await expect(
            probeTexture(device, texture("depth32float-stencil8", 1, 1, new Uint8Array(4))),
        ).rejects.toThrow("requires an explicit");
        await expect(
            probeTexture(device, texture("depth24plus", 1, 1, new Uint8Array(4))),
        ).rejects.toThrow("no portable buffer-copy representation");
        await expect(
            probeTexture(device, texture("depth24plus-stencil8", 1, 1, new Uint8Array(4)), {
                aspect: "depth-only",
            }),
        ).rejects.toThrow("depth has no portable");
        await expect(
            probeTexture(device, texture("rgba8unorm", 1, 1, new Uint8Array(4)), {
                aspect: "depth-only",
            }),
        ).rejects.toThrow('requires aspect "all"');
        await expect(
            probeTexture(device, texture("bc1-rgba-unorm", 4, 4, new Uint8Array(8))),
        ).rejects.toThrow("accepts uncompressed copyable textures only");
    });

    test("invalid source usages and copy descriptors fail before staging allocation", async () => {
        const bufferEvents: string[] = [];
        await expect(
            probeBuffer(fakeDevice(bufferEvents), buffer([0, 1, 2, 3], GPUBufferUsage.COPY_DST)),
        ).rejects.toThrow("missing GPUBufferUsage.COPY_SRC");
        expect(bufferEvents).toEqual([]);

        const mapped = buffer([0, 1, 2, 3]);
        Object.defineProperty(mapped, "mapState", { value: "mapped" });
        await expect(probeBuffer(fakeDevice(bufferEvents), mapped)).rejects.toThrow(
            "source must be unmapped",
        );
        await expect(
            probeBuffer(fakeDevice(bufferEvents, 4), buffer([0, 1, 2, 3, 4, 5, 6, 7])),
        ).rejects.toThrow("exceeds device.limits.maxBufferSize");
        expect(bufferEvents).toEqual([]);

        const textureEvents: string[] = [];
        await expect(
            probeTexture(
                fakeDevice(textureEvents),
                texture("rgba8unorm", 1, 1, new Uint8Array(4), GPUTextureUsage.COPY_DST),
            ),
        ).rejects.toThrow("missing GPUTextureUsage.COPY_SRC");
        await expect(
            probeTexture(
                fakeDevice(textureEvents),
                texture("rgba8unorm", 1, 1, new Uint8Array(4)),
                { origin: { x: -1 } },
            ),
        ).rejects.toThrow("origin must contain non-negative integer coordinates");
        await expect(
            probeTexture(fakeDevice(textureEvents), {
                ...texture("rgba8unorm", 1, 1, new Uint8Array(4)),
                sampleCount: 4,
            } as FakeTexture),
        ).rejects.toThrow("multisampled textures cannot be copied");
        await expect(
            probeTexture(
                fakeDevice(textureEvents, 128),
                texture("rgba8unorm", 1, 1, new Uint8Array(4)),
            ),
        ).rejects.toThrow("exceeds device.limits.maxBufferSize");
        expect(textureEvents).toEqual([]);
    });
});
