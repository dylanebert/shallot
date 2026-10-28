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
import { type Plugin, probeTexture, type World } from "../../engine";
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
// cube faces in +X, -X, +Y, -Y, +Z, -Z order
const FACES = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
];
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
        // r32float is unfilterable without float32-filterable: texel 0 a quarter, texel 1 three quarters
        const scalars = device.createTexture({ size: [2, 1], format: "r32float", usage });
        device.queue.writeTexture(
            { texture: scalars },
            new Float32Array([0.25, 0.75]),
            { bytesPerRow: 8 },
            [2, 1],
        );
        const signed = device.createTexture({ size: [1, 1], format: "r32sint", usage });
        device.queue.writeTexture(
            { texture: signed },
            new Int32Array([-7]),
            { bytesPerRow: 4 },
            [1, 1],
        );
        const unsigned = device.createTexture({ size: [1, 1], format: "r32uint", usage });
        device.queue.writeTexture(
            { texture: unsigned },
            new Uint32Array([7]),
            { bytesPerRow: 4 },
            [1, 1],
        );
        const upload = (
            size: [number, number, number],
            format: GPUTextureFormat,
            data: Uint8Array | Uint16Array,
            dimension: GPUTextureDimension = "2d",
        ) => {
            const texture = device.createTexture({ size, format, usage, dimension });
            const bytesPerRow = (data.byteLength / (size[1] * size[2])) | 0;
            device.queue.writeTexture(
                { texture },
                data,
                { bytesPerRow, rowsPerImage: size[1] },
                size,
            );
            return texture;
        };
        const rgba = (colors: number[][]) => new Uint8Array(colors.flatMap((c) => [...c, 255]));
        const pairColors = [
            [255, 0, 0],
            [0, 255, 0],
        ];
        // texel or slice 0 red, 1 green
        const line = upload([2, 1, 1], "rgba8unorm", rgba(pairColors), "1d");
        const volume = upload([1, 1, 2], "rgba8unorm", rgba(pairColors), "3d");
        // face i of cube c holds FACES[(i + c) % 6]
        const cube = upload([1, 1, 6], "rgba8unorm", rgba(FACES));
        const cubes = upload(
            [1, 1, 12],
            "rgba8unorm",
            rgba([...FACES, ...FACES.slice(1), FACES[0]]),
        );
        const depthLayers = upload([1, 1, 2], "depth16unorm", new Uint16Array([0x4000, 0xc000]));
        // face i holds (i + 1) / 8 of the depth range
        const depthCube = upload(
            [1, 1, 6],
            "depth16unorm",
            new Uint16Array(FACES.map((_, i) => (i + 1) * 0x2000)),
        );
        for (const texture of [pair, layers, depth, scalars, signed, unsigned]) world.own(texture);
        for (const texture of [line, volume, cube, cubes, depthLayers, depthCube])
            world.own(texture);
        const storage = (texels: number[], mipLevelCount = 1) => {
            const texture = device.createTexture({
                size: [texels.length, 1],
                mipLevelCount,
                format: "r32uint",
                usage:
                    GPUTextureUsage.STORAGE_BINDING |
                    GPUTextureUsage.COPY_SRC |
                    GPUTextureUsage.COPY_DST,
            });
            device.queue.writeTexture(
                { texture },
                new Uint32Array(texels),
                { bytesPerRow: 4 * texels.length },
                [texels.length, 1],
            );
            world.own(texture);
            return texture;
        };
        // cleared red, then sample 2 alone drawn green
        const samples = device.createTexture({
            size: [1, 1],
            format: "rgba8unorm",
            sampleCount: 4,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
        });
        world.own(samples);
        const module = device.createShaderModule({
            code: `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    return vec4f(f32(i / 2u) * 4 - 1, f32(i % 2u) * 4 - 1, 0, 1);
}
@fragment fn fs() -> @location(0) vec4f { return vec4f(0, 1, 0, 1); }`,
        });
        const pipeline = device.createRenderPipeline({
            label: "owned-sample-2",
            layout: "auto",
            vertex: { module },
            fragment: { module, targets: [{ format: "rgba8unorm" }] },
            multisample: { count: 4, mask: 1 << 2 },
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: samples.createView(),
                    clearValue: [1, 0, 0, 1],
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(pipeline);
        pass.draw(3);
        pass.end();
        device.queue.submit([encoder.finish()]);
        return {
            pair,
            layers,
            depth,
            scalars,
            signed,
            unsigned,
            line,
            volume,
            cube,
            cubes,
            depthLayers,
            depthCube,
            samples,
            readable: storage([7]),
            // a storage binding takes one mip, so a second one shows which is bound
            written: storage([0, 0], 2),
            // texel 1 is written from texel 0
            updated: storage([3, 0]),
            linear: device.createSampler({ magFilter: "linear", minFilter: "linear" }),
            less: device.createSampler({ compare: "less" }),
            nearest: device.createSampler(),
        };
    },
};
const DEPTH = 0xc000 / 0xffff;
const Unused = d.struct({ unused: d.f32 });
const pairLayout = materialLayout(Unused, { pair: { type: "texture" } });
const layersLayout = materialLayout(Unused, { layers: { type: "texture", dimension: "2d-array" } });
const depthLayout = materialLayout(Unused, { depth: { type: "texture", sampleType: "depth" } });
const samplerLayout = materialLayout(Unused, {
    pair: { type: "texture" },
    linear: { type: "sampler" },
});
const comparisonLayout = materialLayout(Unused, {
    depth: { type: "texture", sampleType: "depth" },
    less: { type: "sampler", samplerType: "comparison" },
});
const unfilterableLayout = materialLayout(Unused, {
    scalars: { type: "texture", filterable: false },
    nearest: { type: "sampler", samplerType: "non-filtering" },
});
const signedLayout = materialLayout(Unused, { signed: { type: "texture", sampleType: "sint" } });
const unsignedLayout = materialLayout(Unused, {
    unsigned: { type: "texture", sampleType: "uint" },
});
const lineLayout = materialLayout(Unused, { line: { type: "texture", dimension: "1d" } });
const volumeLayout = materialLayout(Unused, { volume: { type: "texture", dimension: "3d" } });
const cubeLayout = materialLayout(Unused, {
    cube: { type: "texture", dimension: "cube" },
    linear: { type: "sampler" },
});
const cubesLayout = materialLayout(Unused, {
    cubes: { type: "texture", dimension: "cube-array" },
    linear: { type: "sampler" },
});
const depthLayersLayout = materialLayout(Unused, {
    depthLayers: { type: "texture", dimension: "2d-array", sampleType: "depth" },
});
const depthCubeLayout = materialLayout(Unused, {
    depthCube: { type: "texture", dimension: "cube", sampleType: "depth" },
    nearest: { type: "sampler", samplerType: "non-filtering" },
});
const samplesLayout = materialLayout(Unused, {
    samples: { type: "texture", multisampled: true },
});
const readableLayout = materialLayout(Unused, {
    readable: { type: "storage-texture", format: "r32uint", access: "read" },
});
const writtenLayout = materialLayout(Unused, {
    written: { type: "storage-texture", format: "r32uint", access: "write" },
});
const updatedLayout = materialLayout(Unused, {
    updated: { type: "storage-texture", format: "r32uint", access: "read_write" },
});
const bindingKinds: Record<
    string,
    { expected: number[]; type: MaterialType<typeof Unused, Record<string, MaterialBinding>> }
> = {
    texture: {
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
    "2d-array texture": {
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
    "depth texture": {
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
    "comparison sampler": {
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
    // -7 reads back negated, so the sign survives
    "sint texture": {
        expected: [srgbByte(7 / 255), 0, 0, 255],
        type: materialType({
            name: "OwnedTextureSint",
            layout: signedLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const texel = std.textureLoad(signedLayout.$.signed, d.vec2i(0, 0), 0);
                return d.vec4f(d.f32(texel.x) / -255, 0, 0, 1);
            },
        }),
    },
    "uint texture": {
        expected: [srgbByte(7 / 255), 0, 0, 255],
        type: materialType({
            name: "OwnedTextureUint",
            layout: unsignedLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const texel = std.textureLoad(unsignedLayout.$.unsigned, d.vec2i(0, 0), 0);
                return d.vec4f(d.f32(texel.x) / 255, 0, 0, 1);
            },
        }),
    },
    "1d texture": {
        expected: [0, 255, 0, 255],
        type: materialType({
            name: "OwnedTexture1d",
            layout: lineLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureLoad(lineLayout.$.line, d.i32(1), 0);
            },
        }),
    },
    "3d texture": {
        expected: [0, 255, 0, 255],
        type: materialType({
            name: "OwnedTexture3d",
            layout: volumeLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureLoad(volumeLayout.$.volume, d.vec3i(0, 0, 1), 0);
            },
        }),
    },
    // +Y selects face 2
    "cube texture": {
        expected: [...FACES[2], 255],
        type: materialType({
            name: "OwnedTextureCube",
            layout: cubeLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureSampleLevel(
                    cubeLayout.$.cube,
                    cubeLayout.$.linear,
                    d.vec3f(0, 1, 0),
                    0,
                );
            },
        }),
    },
    "cube-array texture": {
        expected: [...FACES[3], 255],
        type: materialType({
            name: "OwnedTextureCubeArray",
            layout: cubesLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureSampleLevel(
                    cubesLayout.$.cubes,
                    cubesLayout.$.linear,
                    d.vec3f(0, 1, 0),
                    1,
                    0,
                );
            },
        }),
    },
    "depth 2d-array texture": {
        expected: [srgbByte(DEPTH), 0, 0, 255],
        type: materialType({
            name: "OwnedTextureDepth2dArray",
            layout: depthLayersLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const depth = std.textureLoad(depthLayersLayout.$.depthLayers, d.vec2i(0, 0), 1, 0);
                return d.vec4f(depth, 0, 0, 1);
            },
        }),
    },
    "depth cube texture": {
        expected: [srgbByte(0x6000 / 0xffff), 0, 0, 255],
        type: materialType({
            name: "OwnedTextureDepthCube",
            layout: depthCubeLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const depth = std.textureSampleLevel(
                    depthCubeLayout.$.depthCube,
                    depthCubeLayout.$.nearest,
                    d.vec3f(0, 1, 0),
                    0,
                );
                return d.vec4f(depth, 0, 0, 1);
            },
        }),
    },
    "read storage texture": {
        expected: [srgbByte(7 / 255), 0, 0, 255],
        type: materialType({
            name: "OwnedStorageTextureRead",
            layout: readableLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const texel = std.textureLoad(readableLayout.$.readable, d.vec2i(0, 0));
                return d.vec4f(d.f32(texel.x) / 255, 0, 0, 1);
            },
        }),
    },
    // red from sample 0, green from sample 2
    "multisampled texture": {
        expected: [255, 255, 0, 255],
        type: materialType({
            name: "OwnedTextureMultisampled2d",
            layout: samplesLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const first = std.textureLoad(samplesLayout.$.samples, d.vec2i(0, 0), 0);
                const third = std.textureLoad(samplesLayout.$.samples, d.vec2i(0, 0), 2);
                return d.vec4f(first.x, third.y, 0, 1);
            },
        }),
    },
    // the right half of the texture samples texel 1 alone
    "unfilterable texture and non-filtering sampler": {
        expected: [srgbByte(0.75), 0, 0, 255],
        type: materialType({
            name: "OwnedUnfilterableTexture",
            layout: unfilterableLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                return std.textureSample(
                    unfilterableLayout.$.scalars,
                    unfilterableLayout.$.nearest,
                    d.vec2f(0.75, 0.5),
                );
            },
        }),
    },
};
// Owned's sample pattern renders through a pipeline, so it is created while plugins initialize.
const OwnedPlugin: Plugin = {
    name: "OwnedBindings",
    gpu: {},
    initialize(world) {
        world.resource(Owned);
    },
};
// Each material writes its storage texture; the texels after a frame show the write.
const storageWrites: Record<
    string,
    {
        texture: "written" | "updated";
        texels: number[];
        type: MaterialType<typeof Unused, Record<string, MaterialBinding>>;
    }
> = {
    "write storage texture": {
        texture: "written",
        texels: [7, 0],
        type: materialType({
            name: "OwnedStorageTextureWrite",
            layout: writtenLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                std.textureStore(writtenLayout.$.written, d.vec2i(0, 0), d.vec4u(7, 0, 0, 0));
                return d.vec4f(0, 0, 0, 1);
            },
        }),
    },
    "read_write storage texture": {
        texture: "updated",
        texels: [3, 7],
        type: materialType({
            name: "OwnedStorageTextureReadWrite",
            layout: updatedLayout,
            defaults: { unused: 0 },
            bindings: (world) => world.resource(Owned),
            fragment: () => {
                "use gpu";
                const texel = std.textureLoad(updatedLayout.$.updated, d.vec2i(0, 0));
                std.textureStore(
                    updatedLayout.$.updated,
                    d.vec2i(1, 0),
                    std.add(texel, d.vec4u(4)),
                );
                return d.vec4f(0, 0, 0, 1);
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
            OwnedPlugin,
            MaterialPlugin(PixelMaterial),
            MaterialPlugin(ExtendedPixelMaterial),
            ...Object.values(bindingKinds).map(({ type }) => MaterialPlugin(type)),
            ...Object.values(storageWrites).map(({ type }) => MaterialPlugin(type)),
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
    test(`${extended ? "extended" : "custom"} material reads top-left framebuffer pixel centers at target resolution`, async () => {
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

for (const [kind, { type, texture, texels }] of Object.entries(storageWrites)) {
    test(`custom material writes its owned ${kind} binding`, async () => {
        const { world } = subjects()[0];
        world.remove(mesh, MeshMaterial);
        world.add(mesh, MeshMaterial, world.resource(type).add({ unused: 0 }));
        world.step(0);
        world.step(0);
        const { bytes } = await probeTexture(world, world.resource(Owned)[texture]);
        expect(Array.from(new Uint32Array(bytes, 0, texels.length))).toEqual(texels);
    });
}
