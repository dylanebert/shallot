import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import { createApp } from "../../engine";
import {
    CompressedImageFormats,
    type ImageFormat,
    imageTexture,
    imageTextures,
    prepareImages,
    registerImage,
} from "./image";
import { RenderingPlugin } from "./index";

setDefaultTimeout(TIMEOUT.node);
await setupGlobals();
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("Headless Dawn adapter required");
console.log("Compressed images adapter", {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    fallback: adapter.info.isFallbackAdapter,
});
const families = [
    [
        "texture-compression-etc2",
        CompressedImageFormats.ETC2,
        ["etc2-rgb8unorm", "etc2-rgba8unorm"],
    ],
    ["texture-compression-bc", CompressedImageFormats.BC, ["bc7-rgba-unorm"]],
    ["texture-compression-astc", CompressedImageFormats.ASTC_LDR, ["astc-4x4-unorm"]],
] as const;

async function read(
    device: GPUDevice,
    texture: GPUTexture,
    level: number,
    rows: number,
    bytes: number,
): Promise<Uint8Array> {
    const buffer = device.createBuffer({
        size: rows * 256,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    const compressed = texture.format !== "rgba8unorm" && texture.format !== "rgba8unorm-srgb";
    const size = Math.max(1, texture.width >> level);
    encoder.copyTextureToBuffer(
        { texture, mipLevel: level },
        { buffer, bytesPerRow: 256, rowsPerImage: rows },
        [compressed ? Math.ceil(size / 4) * 4 : size, compressed ? Math.ceil(size / 4) * 4 : size],
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const result = new Uint8Array(rows * bytes);
    for (let row = 0; row < rows; row++)
        result.set(mapped.subarray(row * 256, row * 256 + bytes), row * bytes);
    buffer.unmap();
    buffer.destroy();
    return result;
}

for (const [feature, flag, formats] of families) {
    const offered = adapter.features.has(feature);
    test(`${feature}: ${offered ? "linear and sRGB levels round trip" : "an adapter without it leaves the family unsupported"} and disabled formats reject by name`, async () => {
        if (!offered) {
            // RenderingPlugin prefers the feature; on this adapter its device lacks it, so the family is refused.
            const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
            try {
                expect(app.world.gpu.device?.features.has(feature)).toBe(false);
                registerImage(app.world, (supported) => {
                    expect(supported & flag).toBe(0);
                    return {
                        format: formats[0] as ImageFormat,
                        width: 4,
                        height: 4,
                        levels: [new Uint8Array(16)],
                    };
                });
                await expect(prepareImages(app.world)).rejects.toThrow(formats[0]);
            } finally {
                app.dispose();
            }
        } else {
            const device = await (await navigator.gpu.requestAdapter())!.requestDevice({
                requiredFeatures: [feature],
            });
            device.pushErrorScope("validation");
            const app = await createApp({
                defaults: false,
                plugins: [RenderingPlugin],
                device,
                adapter,
            });
            try {
                for (const format of formats)
                    for (const isSrgb of [false, true]) {
                        const bytes = format === "etc2-rgb8unorm" ? 8 : 16;
                        const levels = [8, 4, 2, 1].map((size, level) =>
                            Uint8Array.from(
                                { length: Math.ceil(size / 4) ** 2 * bytes },
                                (_, i) => (i * 13 + level * 37) & 255,
                            ),
                        );
                        let calls = 0;
                        const id = registerImage(
                            app.world,
                            (supported) => {
                                expect(supported).toBe(flag);
                                calls++;
                                return { format, width: 8, height: 8, levels };
                            },
                            { isSrgb },
                        );
                        await prepareImages(app.world);
                        expect(calls).toBe(1);
                        const texture = imageTexture(app.world, id, { isSrgb });
                        expect(texture.format).toBe(isSrgb ? `${format}-srgb` : format);
                        for (let level = 0; level < levels.length; level++) {
                            const rows = Math.ceil(Math.max(1, 8 >> level) / 4);
                            expect(await read(device, texture, level, rows, rows * bytes)).toEqual(
                                levels[level],
                            );
                        }
                    }
                expect(await device.popErrorScope()).toBeNull();
            } finally {
                app.dispose();
                device.destroy();
            }
        }
        const bare = await (await navigator.gpu.requestAdapter())!.requestDevice();
        const other = await createApp({
            defaults: false,
            plugins: [RenderingPlugin],
            device: bare,
            adapter,
        });
        try {
            registerImage(other.world, (supported) => {
                expect(supported).toBe(0);
                return {
                    format: formats[0] as ImageFormat,
                    width: 4,
                    height: 4,
                    levels: [new Uint8Array(16)],
                };
            });
            await expect(prepareImages(other.world)).rejects.toThrow(formats[0]);
        } finally {
            other.dispose();
            bare.destroy();
        }
    });
}

test("one RGBA8 level generates its full mip chain", async () => {
    const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin], device, adapter });
    try {
        const id = registerImage(
            app.world,
            () => ({
                format: "rgba8unorm",
                width: 4,
                height: 4,
                levels: [new Uint8Array(64).fill(128)],
            }),
            { isSrgb: false },
        );
        await prepareImages(app.world);
        const texture = imageTexture(app.world, id, { isSrgb: false });
        expect(texture.mipLevelCount).toBe(3);
        expect(await read(device, texture, 1, 2, 8)).toEqual(new Uint8Array(16).fill(128));
        expect(await read(device, texture, 2, 1, 4)).toEqual(new Uint8Array(4).fill(128));
    } finally {
        app.dispose();
        device.destroy();
    }
});

test("a failed level source rejects the preparations that began before it was reported, then binds the fallback with one warning", async () => {
    const warn = spyOn(console, "warn").mockImplementation(() => {});
    const app = await createApp({ defaults: false, plugins: [RenderingPlugin] });
    try {
        const pixel = () => ({
            format: "rgba8unorm" as const,
            width: 1,
            height: 1,
            levels: [new Uint8Array(4)],
        });
        // Two preparations pending when the failure settles both reject; a later one resolves.
        const first = registerImage(app.world, () => {
            throw new Error("first source failed");
        });
        const turn = () => new Promise((resolve) => setImmediate(resolve));
        // A handler attached in the same turn: the rejection can settle before expect's would.
        const outcome = (preparation: Promise<void>) =>
            preparation.then(
                () => "resolved",
                (error: Error) => error.message,
            );
        const pending = [outcome(prepareImages(app.world)), outcome(prepareImages(app.world))];
        expect(await Promise.all(pending)).toEqual(["first source failed", "first source failed"]);
        expect(await outcome(prepareImages(app.world))).toBe("resolved");
        // A failure with no preparation pending is kept until one reports it.
        let fail!: (error: Error) => void;
        const second = registerImage(app.world, () => new Promise((_, reject) => (fail = reject)));
        fail(new Error("second source failed"));
        await turn();
        expect(await outcome(prepareImages(app.world))).toBe("second source failed");
        expect(await outcome(prepareImages(app.world))).toBe("resolved");
        // Only the failure that rejects a preparation is reported; a later one in it stays stored.
        const fails: ((error: Error) => void)[] = [];
        const [third, fourth] = [0, 1].map(() =>
            registerImage(app.world, () => new Promise((_, reject) => fails.push(reject))),
        );
        const both = outcome(prepareImages(app.world));
        fails[0](new Error("third source failed"));
        expect(await both).toBe("third source failed");
        fails[1](new Error("fourth source failed"));
        await turn();
        expect(await outcome(prepareImages(app.world))).toBe("fourth source failed");
        expect(await outcome(prepareImages(app.world))).toBe("resolved");
        const later = registerImage(app.world, pixel);
        await prepareImages(app.world);
        const { fallback } = app.world.resource(imageTextures);
        expect(imageTexture(app.world, later)).not.toBe(fallback[0]);
        // Bound twice: the fallback warning does not repeat.
        for (const id of [first, second, third, fourth, first, second, third, fourth])
            expect(imageTexture(app.world, id)).toBe(fallback[0]);
        // A failure no preparation ever awaits: its rejection settles in microtasks, and Bun ends
        // each event loop turn by handling rejected promises (src/event_loop/README.md at
        // bun-v1.4.3), failing this test if one is unhandled, before the next turn's setImmediate.
        registerImage(app.world, () => Promise.reject(new Error("unawaited source failed")));
        await turn();
        const warnings = warn.mock.calls.map((call) => call.map(String).join(" "));
        // The rejection carries the cause; binding the fallback names each id once.
        for (const id of [first, second, third, fourth])
            expect(warnings.filter((warning) => warning.includes(`image ${id} `))).toEqual([
                `[Rendering] color image ${id} has no native texture; binding fallback`,
            ]);
    } finally {
        app.dispose();
        warn.mockRestore();
    }
});
