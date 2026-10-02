import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { type Plugin, Transform } from "../../engine";
import { precompile } from "../../engine/runtime";
import { composite as baseline, GlazeConfig } from "./fixtures/glaze-compute";
import {
    attachTexture,
    Camera,
    CorePipelinePlugin,
    captureTexture,
    detachCanvas,
    RenderPhases,
    Views,
} from "./index";
import { Tonemap } from "./tonemap";
import { GlazeSystem } from "./tonemapping-state";

const sources = { create: () => new Map<number, GPUTextureView>() };
const comparison = {
    create: () => ({ config: null as unknown as ReturnType<typeof configBuffer> }),
};
function configBuffer(world: import("../../engine").World) {
    return world.gpu.root.createBuffer(GlazeConfig).$usage("uniform");
}

setDefaultTimeout(CEILING.gpu);
const Scene: Plugin = {
    name: "PresentationComparisonScene",
    dependencies: [CorePipelinePlugin],
    systems: [
        {
            group: "draw",
            after: [GlazeSystem],
            update(world) {
                for (const [eid, view] of world.resource(Views))
                    if (view.framebuffer) world.resource(sources).set(eid, view.framebuffer);
            },
        },
    ],
    async warm(world) {
        const device = world.gpu.device;
        const built = baseline(world, navigator.gpu.getPreferredCanvasFormat());
        world.resource(comparison).config = configBuffer(world);
        precompile(world, "base-composite-fixture", () => {
            const input = device.createTexture({
                size: [1, 1],
                format: "rg11b10ufloat",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const output = device.createTexture({
                size: [1, 1],
                format: navigator.gpu.getPreferredCanvasFormat(),
                usage: GPUTextureUsage.STORAGE_BINDING,
            });
            const bound = built.pipeline.with(
                world.gpu.root.createBindGroup(built.layout, {
                    input: input.createView(),
                    output: output.createView(),
                    glaze: world.resource(comparison).config,
                }),
            );
            input.destroy();
            output.destroy();
            return bound;
        });
    },
    initialize(world) {
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
}
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    return vec4f(p.x / 7.0, p.y / 9.0, f32((u32(p.x) + u32(p.y)) % 11u) / 3.0, 0.37);
}`,
        });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vertex" },
            fragment: { module, entryPoint: "fragment", targets: [{ format: "rg11b10ufloat" }] },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
        });
        world.resource(RenderPhases).push({
            opaque(_world, _eid, _view, pass) {
                pass.setPipeline(pipeline);
                pass.draw(3);
            },
        });
    },
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [Scene] }]);

import { Glaze } from "./tonemapping-state";

for (const [width, height] of [
    [16, 12],
    [23, 17],
]) {
    for (const mode of [null, ...Object.values(Tonemap)]) {
        test(`fragment matches base compute: ${width}x${height}, ${mode === null ? "no Glaze" : `Spindle/operator ${mode}`}`, async () => {
            const { world } = subjects()[0];
            const camera = world.create();
            world.add(camera, Transform);
            world.add(camera, Camera);
            world.storage(Camera).antialias.set(camera, 0);
            if (mode !== null) {
                world.add(camera, Glaze, {
                    tonemap: mode,
                    vignette: 0.1,
                    posterize: 8,
                    dither: 0.125,
                    slope: [1.06, 1, 0.92, 0],
                    offset: [0.006, 0.005, 0.003, 0],
                    power: [1, 1, 1, 0],
                    saturation: 1.05,
                });
            }
            attachTexture(world, camera, { width, height });
            const device = world.gpu.device;
            device.pushErrorScope("validation");
            world.step(0);
            const presented = await captureTexture(world, camera);
            const view = world.resource(Views).get(camera)!;
            const built = baseline(world, navigator.gpu.getPreferredCanvasFormat());
            const config = world.resource(comparison).config;
            config.write({
                exposure: 1,
                tonemapMode: mode ?? 0,
                saturation: mode === null ? 1 : 1.05,
                vignetteStrength: mode === null ? 0 : 0.1,
                vignetteInner: 0,
                vignetteOuter: mode === null ? 0 : 1,
                posterizeBands: mode === null ? 0 : 8,
                ditherStrength: mode === null ? 0 : 0.125,
                slope: mode === null ? [1, 1, 1, 0] : [1.06, 1, 0.92, 0],
                offset: mode === null ? [0, 0, 0, 0] : [0.006, 0.005, 0.003, 0],
                power: [1, 1, 1, 0],
            });
            const group = world.gpu.root.createBindGroup(built.layout, {
                input: world.resource(sources).get(camera)!,
                glaze: config,
                output: view.texture!.createView(),
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            built.pipeline
                .with(group)
                .with(pass)
                .dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
            pass.end();
            device.queue.submit([encoder.finish()]);
            const reference = await captureTexture(world, camera);
            expect(await device.popErrorScope()).toBeNull();
            expect([presented.width, presented.height]).toEqual([
                reference.width,
                reference.height,
            ]);
            expect(presented.rgba.some((value, i) => i % 4 !== 3 && value > 0)).toBe(true);
            const maximum = [0, 0, 0, 0];
            for (let y = 0; y < height; y++)
                for (let x = 0; x < width; x++) {
                    for (let channel = 0; channel < 4; channel++) {
                        const i = (y * width + x) * 4 + channel;
                        maximum[channel] = Math.max(
                            maximum[channel],
                            Math.abs(presented.rgba[i] - reference.rgba[i]),
                        );
                    }
                }
            console.log(
                `${width}x${height} ${mode === null ? "no Glaze" : `Spindle/operator ${mode}`}: max RGBA ${maximum.join(",")}`,
            );
            expect(maximum.every((difference) => difference <= 1)).toBe(true);
            detachCanvas(world, camera);
            world.destroy(camera);
        });
    }
}
