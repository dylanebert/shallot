import { expect, setDefaultTimeout, test } from "bun:test";
import { Exposure, probeTexture, Transform } from "@dylanebert/shallot";
import { fogReadBindGroup, fogReadLayout, readFogInScatter } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture, EffectPasses } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import tgpu from "typegpu";
import { Motes, MotesPlugin } from "./motes";
import { Player } from "./player";
import { Room } from "./room";

setDefaultTimeout(TIMEOUT.node);
const lin = (x: number) => {
    const v = x / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const lum = (a: Uint8ClampedArray, i: number) =>
    0.2126 * lin(a[i]!) + 0.7152 * lin(a[i + 1]!) + 0.0722 * lin(a[i + 2]!);
test("a still edge mote's estimated display swing is under 0.005 over 16 frames", async () => {
    const app = await openGpuApp({ plugins: [MotesPlugin] });
    try {
        const { world } = app;
        const camera = world.resource(Room).camera;
        world.remove(world.resource(Room).player, Player);
        attachTexture(world, camera, { width: 160, height: 90 });
        world.storage(Transform).translation.set(camera, 0, 1.7, 2, 0);
        world.storage(Exposure).ev100.set(camera, 5);
        world.tick();
        const motes = world.resource(Motes);
        motes.visible = false;
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: tgpu.resolve({
                externals: { readFogInScatter },
                template: `
 @vertex fn fullscreenVertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
 let x=f32((i<<1u)&2u); let y=f32(i&2u); return vec4f(x*2-1,y*2-1,0,1); }
 @fragment fn sample()->@location(0) vec4f {
 return vec4f(readFogInScatter(vec3f(1,1.05,-1.5)),1); }
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
            size: [1, 1],
            format: "rgba32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        world.own(output);
        world.step(0);
        const values: number[] = [];
        const rgb = [0, 0, 0];
        for (let f = 0; f < 16; f++) {
            world.step(0);
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
            const a = new Float32Array((await probeTexture(world, output)).bytes);
            for (let c = 0; c < 3; c++) rgb[c]! += a[c]! / 16;
            values.push(0.2126 * a[0]! + 0.7152 * a[1]! + 0.0722 * a[2]!);
        }
        const mean = values.reduce((s, v) => s + v, 0) / 16;
        const swing = Math.max(...values) - Math.min(...values);
        const result = { point: "edge", mean, swing, relative: swing / mean, values };
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
