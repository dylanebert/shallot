import { expect, setDefaultTimeout, test } from "bun:test";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import inputs from "../../../crates/basisu/tests/records/inputs.json";
import { syncFiles } from "../../../examples/gltf/sync";
import { bunChild } from "../../../scripts/bun-child";
import { imageTexture, prepareImages, RenderingPlugin, registerImage } from "../../core/rendering";
import { createApp } from "../../engine";
import { basisImageSource } from "./image";

setDefaultTimeout(TIMEOUT.node);
test("a scene without KTX2 imports no transcoder", async () => {
    const entry = new URL("./index.ts", import.meta.url).href;
    const engine = new URL("../../engine/index.ts", import.meta.url).href;
    const child = Bun.spawn({
        ...bunChild([
            "-e",
            `const {loadGltfScene} = await import(${JSON.stringify(entry)});
             const {World} = await import(${JSON.stringify(engine)});
             const world = new World();
             const json = {materials: [{pbrMetallicRoughness: {baseColorTexture: {index: 0}}}], textures: [{source: 0}], images: [{uri: 'https://example.test/color.png'}]};
             await loadGltfScene(world, 'data:model/gltf+json,' + encodeURIComponent(JSON.stringify(json)));
             world.dispose();
             if (Object.keys(require.cache).some(p => /basisu(?:\\.wasm)?\\.ts$/.test(p))) process.exit(1);`,
        ]),
        stdout: "pipe",
        stderr: "pipe",
    });
    expect(await child.exited, await new Response(child.stderr).text()).toBe(0);
});
await syncFiles(
    inputs.files,
    new URL(
        `https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/${inputs.commit}/Models/`,
    ),
    fileURLToPath(new URL("../../../test-results/basisu-assets/", import.meta.url)),
);
await setupGlobals();
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error("Headless Dawn adapter required");
console.log("BasisU importer adapter", {
    vendor: adapter.info.vendor,
    architecture: adapter.info.architecture,
    device: adapter.info.device,
    description: adapter.info.description,
    fallback: adapter.info.isFallbackAdapter,
});
const records = new Map<
    string,
    { codec: string; levels: Map<string, { length: number; hash: string }[]> }
>();
for (const codec of ["etc1s", "uastc"]) {
    const text = readFileSync(
        new URL(`../../../crates/basisu/tests/records/${codec}.tsv`, import.meta.url),
        "utf8",
    );
    for (const line of text.trim().split("\n").slice(1)) {
        const [asset, level, target, length, hash] = line.split("\t");
        if (!records.has(asset)) records.set(asset, { codec, levels: new Map() });
        const levels = records.get(asset)!.levels;
        if (!levels.has(target)) levels.set(target, []);
        levels.get(target)![Number(level)] = { length: Number(length), hash };
    }
}
const formats = {
    etc1: "etc2-rgb8unorm",
    "etc2-rgba": "etc2-rgba8unorm",
    bc7: "bc7-rgba-unorm",
    astc: "astc-4x4-unorm",
    rgba8: "rgba8unorm",
} as const;
async function readLevel(
    device: GPUDevice,
    texture: GPUTexture,
    level: number,
    target: keyof typeof formats,
) {
    const compressed = target !== "rgba8";
    const width = Math.max(1, texture.width >> level),
        height = Math.max(1, texture.height >> level);
    const rows = compressed ? Math.ceil(height / 4) : height;
    const rowBytes = compressed ? Math.ceil(width / 4) * (target === "etc1" ? 8 : 16) : width * 4;
    const stride = Math.ceil(rowBytes / 256) * 256;
    const buffer = device.createBuffer({
        size: rows * stride,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
        const encoder = device.createCommandEncoder();
        encoder.copyTextureToBuffer(
            { texture, mipLevel: level },
            { buffer, bytesPerRow: stride, rowsPerImage: rows },
            [
                compressed ? Math.ceil(width / 4) * 4 : width,
                compressed ? Math.ceil(height / 4) * 4 : height,
            ],
        );
        device.queue.submit([encoder.finish()]);
        await buffer.mapAsync(GPUMapMode.READ);
        const bytes = new Uint8Array(buffer.getMappedRange());
        const hash = createHash("sha256");
        for (let row = 0; row < rows; row++)
            hash.update(bytes.subarray(row * stride, row * stride + rowBytes));
        return { length: rows * rowBytes, hash: hash.digest("hex") };
    } finally {
        buffer.destroy();
    }
}
for (const feature of [
    "texture-compression-astc",
    "texture-compression-etc2",
    "texture-compression-bc",
    undefined,
] as const) {
    // An adapter without the feature holds the target the transcoder falls back to on a device lacking it.
    const granted = feature && adapter.features.has(feature) ? feature : undefined;
    test(`${feature ?? "no compression"}: every BasisU fixture uploads the selected reference target at every level${feature && !granted ? ", falling back where the adapter lacks it" : ""}`, async () => {
        const device = await (await navigator.gpu.requestAdapter())!.requestDevice({
            requiredFeatures: granted ? [granted] : [],
        });
        device.pushErrorScope("validation");
        const app = await createApp({
            defaults: false,
            plugins: [RenderingPlugin],
            device,
            adapter,
        });
        try {
            for (const [asset, record] of records) {
                const [model, file] = asset.split("/");
                const path =
                    model === "fixtures"
                        ? `../../../crates/basisu/tests/fixtures/${file}.ktx2`
                        : `../../../test-results/basisu-assets/${model}/glTF-KTX-BasisU/${file}.ktx2`;
                const bytes = readFileSync(new URL(path, import.meta.url));
                const isSrgb = /basecolor|emissive/.test(file);
                for (const mipmaps of model === "fixtures" ? [false, true] : [true]) {
                    const single = record.levels.get("rgba8")!.length === 1;
                    const target =
                        single && mipmaps
                            ? "rgba8"
                            : record.codec === "uastc" && granted === "texture-compression-astc"
                              ? "astc"
                              : record.codec === "etc1s" && granted === "texture-compression-etc2"
                                ? file.includes("-alpha")
                                    ? "etc2-rgba"
                                    : "etc1"
                                : granted === "texture-compression-bc"
                                  ? "bc7"
                                  : "rgba8";
                    const id = registerImage(
                        app.world,
                        basisImageSource({ bytes }, isSrgb, mipmaps),
                        { isSrgb },
                    );
                    await prepareImages(app.world);
                    const texture = imageTexture(app.world, id, { isSrgb });
                    expect(texture.format, asset).toBe(
                        `${formats[target]}${isSrgb ? "-srgb" : ""}`,
                    );
                    const levels = record.levels.get(target)!;
                    expect(texture.mipLevelCount, asset).toBe(
                        single && target === "rgba8"
                            ? Math.floor(Math.log2(Math.max(texture.width, texture.height))) + 1
                            : levels.length,
                    );
                    for (let level = 0; level < levels.length; level++)
                        expect(
                            await readLevel(device, texture, level, target),
                            `${asset} ${target} ${level}`,
                        ).toEqual(levels[level]);
                }
            }
            expect(await device.popErrorScope()).toBeNull();
        } finally {
            app.dispose();
            device.destroy();
        }
    });
}
