import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { MeshInstance } from "../../core/mesh";
import {
    attachTexture,
    Camera,
    captureTexture,
    Tonemapping,
    TonemappingMethod,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { World } from "../../engine";
import {
    extendedMaterial,
    type MaterialBinding,
    MaterialPlugin,
    Materials,
    type MaterialType,
    MeshMaterial,
    MeshRenderPlugin,
    materialLayout,
    materialType,
    StandardMaterial,
    StandardRenderer,
    StandardRenderingPlugin,
} from "./index";

setDefaultTimeout(TIMEOUT.gpu);
const PixelMaterial = materialType({
    name: "FramebufferPosition",
    parameters: d.struct({ unused: d.f32 }),
    defaults: { unused: 0 },
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.fragCoord.x / 8, ctx.fragCoord.y / 4, 0, 1);
    },
});
const ExtendedPixelMaterial = extendedMaterial(Materials, {
    name: "ExtendedFramebufferPosition",
    fragment: (ctx) => {
        "use gpu";
        return d.vec4f(ctx.fragCoord.x / 8, ctx.fragCoord.y / 4, 0, 1);
    },
});
// One owned resource per binding kind, each holding a value only a correct binding reads back.
const Owned = {
    create: (world: World) => {
        const device = world.gpu.device;
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST;
        // texel 0 red, texel 1 green
        const pair = device.createTexture({ size: [2, 1], format: "rgba8unorm", usage });
        device.queue.writeTexture(
            { texture: pair },
            new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]),
            { bytesPerRow: 8 },
            [2, 1],
        );
        // layer 0 red, layer 1 green
        const layers = device.createTexture({ size: [1, 1, 2], format: "rgba8unorm", usage });
        device.queue.writeTexture(
            { texture: layers },
            new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255]),
            { bytesPerRow: 4, rowsPerImage: 1 },
            [1, 1, 2],
        );
        const depth = device.createTexture({ size: [1, 1], format: "depth16unorm", usage });
        device.queue.writeTexture(
            { texture: depth },
            new Uint16Array([0xc000]),
            { bytesPerRow: 2 },
            [1, 1],
        );
        for (const texture of [pair, layers, depth]) world.own(texture);
        return {
            pair,
            layers,
            depth,
            linear: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
            less: device.createSampler({ compare: "less" }),
        };
    },
};
const DEPTH = 0xc000 / 0xffff;
const Unused = d.struct({ unused: d.f32 });
const pairLayout = materialLayout(Unused, { pair: { type: "texture-2d" } });
const layersLayout = materialLayout(Unused, { layers: { type: "texture-2d-array" } });
const depthLayout = materialLayout(Unused, { depth: { type: "texture-depth-2d" } });
const samplerLayout = materialLayout(Unused, {
    pair: { type: "texture-2d" },
    linear: { type: "sampler" },
});
const comparisonLayout = materialLayout(Unused, {
    depth: { type: "texture-depth-2d" },
    less: { type: "sampler-comparison" },
});
const bindingKinds: Record<
    string,
    { expected: number[]; type: MaterialType<typeof Unused, Record<string, MaterialBinding>> }
> = {
    "texture-2d": {
        expected: [0, 255, 0, 255],
        type: materialType({
            name: "OwnedTexture2d",
            layout: pairLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureLoad(pairLayout.$.pair, d.vec2i(1, 0), 0);
            },
        }),
    },
    "texture-2d-array": {
        expected: [0, 255, 0, 255],
        type: materialType({
            name: "OwnedTexture2dArray",
            layout: layersLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureLoad(layersLayout.$.layers, d.vec2i(0, 0), 1, 0);
            },
        }),
    },
    "texture-depth-2d": {
        expected: [srgbByte(DEPTH), 0, 0, 255],
        type: materialType({
            name: "OwnedTextureDepth2d",
            layout: depthLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return d.vec4f(std.textureLoad(depthLayout.$.depth, d.vec2i(0, 0), 0), 0, 0, 1);
            },
        }),
    },
    // the midpoint of the red and green texels filters to half of each
    sampler: {
        expected: [srgbByte(0.5), srgbByte(0.5), 0, 255],
        type: materialType({
            name: "OwnedSampler",
            layout: samplerLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureSample(
                    samplerLayout.$.pair,
                    samplerLayout.$.linear,
                    d.vec2f(0.5),
                );
            },
        }),
    },
    // "less" passes a reference below the stored depth and fails one above it
    "sampler-comparison": {
        expected: [255, 0, 0, 255],
        type: materialType({
            name: "OwnedSamplerComparison",
            layout: comparisonLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const uv = d.vec2f(0.5);
                const below = std.textureSampleCompare(
                    comparisonLayout.$.depth,
                    comparisonLayout.$.less,
                    uv,
                    0.5,
                );
                const above = std.textureSampleCompare(
                    comparisonLayout.$.depth,
                    comparisonLayout.$.less,
                    uv,
                    0.9,
                );
                return d.vec4f(below, above, 0, 1);
            },
        }),
    },
};
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            MaterialPlugin(PixelMaterial),
            MaterialPlugin(ExtendedPixelMaterial),
            ...Object.values(bindingKinds).map(({ type }) => MaterialPlugin(type)),
        ],
    },
]);
let camera: number;
let mesh: number;
beforeAll(() => {
    const { world } = subjects()[0];
    camera = world.create();
    world.add(camera, Transform, { translation: [0, 0, 5, 0] });
    world.add(camera, Camera);
    world.add(camera, StandardRenderer);
    world.add(camera, Tonemapping, { method: TonemappingMethod.None });
    attachTexture(world, camera, { width: 8, height: 4 });
    mesh = world.create();
    world.add(mesh, Transform, { scale: [20, 20, 1, 0] });
    world.add(mesh, MeshInstance);
    world.add(mesh, MeshMaterial, world.resource(PixelMaterial).add({ unused: 0 }));
    world.step(0);
    world.step(0);
}, TIMEOUT.startup);

function srgbByte(linear: number) {
    return Math.round((1.055 * linear ** (1 / 2.4) - 0.055) * 255);
}

for (const extended of [false, true]) {
    test(`${extended ? "extended" : "custom"} material reads top-left framebuffer pixel centres at target resolution`, async () => {
        const { world } = subjects()[0];
        world.remove(mesh, MeshMaterial);
        world.add(
            mesh,
            MeshMaterial,
            extended
                ? world.resource(ExtendedPixelMaterial).add(StandardMaterial())
                : world.resource(PixelMaterial).add({ unused: 0 }),
        );
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        for (let y = 0; y < 4; y++) {
            for (let x = 0; x < 8; x++) {
                const offset = (y * 8 + x) * 4;
                expect(Math.abs(rgba[offset] - srgbByte((x + 0.5) / 8))).toBeLessThanOrEqual(1);
                expect(Math.abs(rgba[offset + 1] - srgbByte((y + 0.5) / 4))).toBeLessThanOrEqual(1);
                expect(Array.from(rgba.subarray(offset + 2, offset + 4))).toEqual([0, 255]);
            }
        }
    });
}

for (const [kind, { type, expected }] of Object.entries(bindingKinds)) {
    test(`custom material reads its owned ${kind} binding`, async () => {
        const { world } = subjects()[0];
        world.remove(mesh, MeshMaterial);
        world.add(mesh, MeshMaterial, world.resource(type).add({ unused: 0 }));
        world.step(0);
        world.step(0);
        const { rgba } = await captureTexture(world, camera);
        const center = (2 * 8 + 4) * 4;
        const read = Array.from(rgba.subarray(center, center + 4));
        for (let i = 0; i < 4; i++) expect(Math.abs(read[i] - expected[i])).toBeLessThanOrEqual(1);
    });
}
