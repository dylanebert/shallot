import { expect, setDefaultTimeout, test } from "bun:test";
import {
    Camera,
    Exposure,
    probeTexture,
    StandardRenderer,
    Transform,
    VolumetricLight,
} from "@dylanebert/shallot";
import {
    FogPlugin,
    fogReadBindGroup,
    fogReadLayout,
    readFogInScatter,
} from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture, EffectPasses } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import tgpu from "typegpu";
import { room } from "./room.fixture";

setDefaultTimeout(TIMEOUT.node);
const lin = (x: number) => {
    const v = x / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const lum = (a: Uint8ClampedArray, i: number) =>
    0.2126 * lin(a[i]!) + 0.7152 * lin(a[i + 1]!) + 0.0722 * lin(a[i + 2]!);
test("a still edge mote's estimated display swing is under 0.005 over 16 frames", async () => {
    const app = await openGpuApp({ plugins: [FogPlugin] });
    try {
        const { world } = app;
        const sun = room(world);
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 1.7, 2, 0] });
        world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        world.add(camera, Exposure, { ev100: 5 });
        attachTexture(world, camera, { width: 160, height: 90 });
        world.storage(Transform).translation.set(camera, 0, 1.7, 2, 0);
        world.storage(Exposure).ev100.set(camera, 5);
        world.tick();
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: tgpu.resolve({
                externals: { readFogInScatter },
                template: `
 @vertex fn fullscreenVertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let x=f32((i<<1u)&2u); let y=f32(i&2u); return vec4f(x*2-1,y*2-1,0,1); }
 @fragment fn sample(@builtin(position) p:vec4f)->@location(0) vec4f {
 if (p.x < 1) { return vec4f(readFogInScatter(vec3f(1,1.05,-1.5)),1); }
 return vec4f(readFogInScatter(vec3f(0,1.05,-1.5)),1); }
 `,
            }),
        });
        const empty = device.createBindGroupLayout({ entries: [] });
        const pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [empty, empty, empty, world.gpu.root.unwrap(fogReadLayout)],
            }),
            vertex: { module, entryPoint: "fullscreenVertex" },
            fragment: { module, entryPoint: "sample", targets: [{ format: "rgba32float" }] },
        });
        const output = device.createTexture({
            size: [2, 1],
            format: "rgba32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        world.own(output);
        const sample = async () => {
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    { view: output.createView(), loadOp: "clear", storeOp: "store" },
                ],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(3, fogReadBindGroup(world, camera));
            pass.draw(3);
            pass.end();
            device.queue.submit([encoder.finish()]);
            return new Float32Array((await probeTexture(world, output)).bytes);
        };
        world.step(0);
        const values: number[] = [];
        const rgb = [0, 0, 0];
        for (let f = 0; f < 16; f++) {
            world.step(0);
            const a = await sample();
            for (let c = 0; c < 3; c++) rgb[c]! += a[c]! / 16;
            values.push(0.2126 * a[0]! + 0.7152 * a[1]! + 0.0722 * a[2]!);
        }
        const mean = values.reduce((s, v) => s + v, 0) / 16;
        const swing = Math.max(...values) - Math.min(...values);
        const result = { point: "edge", mean, swing, relative: swing / mean, values };
        const converged = async () => {
            for (let f = 0; f < 64; f++) world.step(0);
            const means = [0, 0];
            for (let f = 0; f < 16; f++) {
                world.step(0);
                const a = await sample();
                for (let p = 0; p < 2; p++)
                    means[p]! +=
                        (0.2126 * a[p * 4]! + 0.7152 * a[p * 4 + 1]! + 0.0722 * a[p * 4 + 2]!) / 16;
            }
            return means;
        };
        const [edge, inside] = await converged();
        world.remove(sun, VolumetricLight);
        const [ambient] = await converged();
        console.log("shaft edge premise", {
            edge,
            inside,
            ambient,
            overAmbient: edge! / ambient!,
            fractionOfInside: edge! / inside!,
        });
        // Require direct sunlight at the sample, but at least a quarter less light than inside the shaft.
        expect(edge!).toBeGreaterThan(10 * ambient!);
        expect(edge!).toBeLessThan(0.75 * inside!);
        // The mote has its game's 0.98 albedo, full pixel coverage against black and unit transmittance.
        // Probe the camera's actual tone curve at that mean RGB radiance; the central difference
        // assumes a locally linear response to the read's relative luminance swing, with fixed hue.
        const displayMean = async (scale: number) => {
            world.resource(EffectPasses).set(camera, {
                before: [
                    (world, _eid, _view, _input, output) => {
                        const pass = world.frameEncoder()!.beginRenderPass({
                            colorAttachments: [
                                {
                                    view: output,
                                    loadOp: "clear",
                                    storeOp: "store",
                                    clearValue: [
                                        rgb[0]! * 0.98 * scale,
                                        rgb[1]! * 0.98 * scale,
                                        rgb[2]! * 0.98 * scale,
                                        1,
                                    ],
                                },
                            ],
                        });
                        pass.end();
                    },
                ],
                after: [],
            });
            world.step(0);
            const pixels = (await captureTexture(world, camera)).rgba;
            let mean = 0;
            for (let i = 0; i < pixels.length; i += 4) mean += lum(pixels, i);
            return mean / (pixels.length / 4);
        };
        const dimmer = await displayMean(2 ** -0.25);
        const brighter = await displayMean(2 ** 0.25);
        const slope = (brighter - dimmer) / Math.log(2 ** 0.5);
        const estimatedDisplaySwing = result.relative * slope;
        console.log(
            JSON.stringify(
                {
                    result,
                    rgb,
                    slope,
                    estimatedDisplaySwing,
                },
                null,
                2,
            ),
        );
        expect(slope).toBeGreaterThan(0);
        expect(estimatedDisplaySwing).toBeLessThan(0.005);
    } finally {
        app.dispose();
    }
});
