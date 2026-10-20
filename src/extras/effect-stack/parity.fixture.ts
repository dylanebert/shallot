import {
    attachTexture,
    Camera,
    EffectPasses,
    ImageBindings,
    TonemappingSystem,
    Views,
} from "../../core/rendering";
import { Transform } from "../../core/transform";
import type { Plugin, World } from "../../engine";
import { probeTexture } from "../../engine/runtime";
import { ChromaticAberration, EffectStackPlugin, LensDistortion, Vignette } from ".";

export const width = 192;
export const height = 96;
const lens = {
    intensity: 0.5,
    scale: 1,
    multiplier: [1, 1] as [number, number],
    center: [0.5, 0.5] as [number, number],
    edgeCurvature: 0,
};
const aberration = { intensity: 0.02, maxSamples: 8 };
const vignette = {
    intensity: 1,
    radius: 0.75,
    smoothness: 5,
    roundness: 1,
    center: [0.5, 0.5] as [number, number],
    edgeCompensation: 1,
    color: 0,
};
export const cases = [
    { name: "lens-default", lens },
    { name: "aberration-default", aberration },
    { name: "vignette-default", vignette },
    {
        name: "lens-fields",
        lens: {
            intensity: 0.8,
            scale: 1.2,
            multiplier: [0.6, 1.3] as [number, number],
            center: [0.42, 0.57] as [number, number],
            edgeCurvature: 0.7,
        },
    },
    {
        name: "lens-negative",
        lens: {
            intensity: -0.7,
            scale: 0.93,
            multiplier: [1.2, 0.8] as [number, number],
            center: [0.56, 0.44] as [number, number],
            edgeCurvature: -0.4,
        },
    },
    {
        name: "aberration-fields-custom-lut",
        aberration: { intensity: 0.13, maxSamples: 5 },
        custom: true,
    },
    {
        name: "vignette-fields",
        vignette: {
            intensity: 0.67,
            radius: 0.91,
            smoothness: 2.3,
            roundness: 1.4,
            center: [0.43, 0.61] as [number, number],
            edgeCompensation: 0.35,
            color: 0x4080c0,
        },
    },
    { name: "all-default", lens, aberration, vignette },
];
export type Case = (typeof cases)[number];
export const defaultLut = new Uint8Array([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255]);
export const customLut = new Uint8Array([64, 220, 255, 255, 255, 80, 128, 255, 180, 255, 48, 255]);

export function texture(world: World, format: GPUTextureFormat, w = width, h = height): GPUTexture {
    const t = world.gpu.device.createTexture({
        size: [w, h],
        format,
        usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    world.own(t);
    return t;
}
export function inputTexture(world: World): GPUTexture {
    const t = texture(world, "rgba16float");
    const data = new Uint16Array(width * height * 4);
    // Sharp HDR bars and a near-floor checkerboard, including distinct last-row texels.
    for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
            const at = (y * width + x) * 4;
            data[at] = (x >> 3) % 2 ? 0x4400 : 0x3400;
            data[at + 1] = (y >> 2) % 2 ? 0x4000 : 0x3800;
            data[at + 2] = ((x + y) >> 3) % 2 ? 0x4200 : 0x3000;
            data[at + 3] = 0x3c00;
        }
    world.gpu.device.queue.writeTexture({ texture: t }, data, { bytesPerRow: width * 8 }, [
        width,
        height,
    ]);
    return t;
}
export function lookup(world: World, custom = false): GPUTexture {
    const t = texture(world, "rgba8unorm-srgb", 3, 1);
    world.gpu.device.queue.writeTexture(
        { texture: t },
        custom ? customLut : defaultLut,
        { bytesPerRow: 12 },
        [3, 1],
    );
    return t;
}
export function parityCapture(): {
    plugin: Plugin;
    render: (world: World, c: Case) => Promise<Uint16Array>;
} {
    let camera = -1;
    let input: GPUTexture;
    let output: GPUTexture;
    return {
        plugin: {
            name: "ParityCapture",
            systems: [
                {
                    name: "parity-capture",
                    group: "draw",
                    after: [EffectStackPlugin.systems![0]!],
                    before: [TonemappingSystem],
                    update(world) {
                        if (camera < 0) return;
                        world.resource(EffectPasses).get(camera)!.before[0]!(
                            world,
                            camera,
                            world.resource(Views).get(camera)!,
                            input.createView(),
                            output.createView(),
                        );
                    },
                },
            ],
        },
        async render(world, c) {
            if (camera < 0) {
                camera = world.create();
                world.add(camera, Transform);
                world.add(camera, Camera);
                attachTexture(world, camera, { width, height });
                input = inputTexture(world);
                output = texture(world, "rgba16float");
            }
            for (const effect of [LensDistortion, ChromaticAberration, Vignette])
                world.remove(camera, effect);
            if (c.lens)
                world.add(camera, LensDistortion, c.name.endsWith("default") ? undefined : c.lens);
            if (c.vignette)
                world.add(camera, Vignette, c.name.endsWith("default") ? undefined : c.vignette);
            const bindings = world.resource(ImageBindings);
            const original = bindings.images;
            if (c.aberration)
                world.add(
                    camera,
                    ChromaticAberration,
                    c.name.endsWith("default")
                        ? undefined
                        : { ...c.aberration, colorLut: "custom" in c ? 0 : -1 },
                );
            if ("custom" in c) bindings.images = lookup(world, true);
            try {
                world.step(0);
                return new Uint16Array((await probeTexture(world, output)).bytes as ArrayBuffer);
            } finally {
                bindings.images = original;
            }
        },
    };
}
export function compare(actual: Uint16Array, expected: Uint16Array, name: string): void {
    if (actual.length !== expected.length) throw new Error(`${name}: length mismatch`);
    let max = 0;
    let first = "";
    for (let i = 0; i < actual.length; i++) {
        const distance = Math.abs(actual[i]! - expected[i]!);
        max = Math.max(max, distance);
        if (distance > 1 && !first)
            first = `pixel ${Math.floor(i / 4)} channel ${i % 4}: ${actual[i]} != ${expected[i]} (${distance} half steps)`;
    }
    console.info(`${name}: maximum ${max} half-float steps`);
    if (first) throw new Error(`${name}: ${first}`);
}
