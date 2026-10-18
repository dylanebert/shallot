import { expect, setDefaultTimeout, test } from "bun:test";
import { Exposure, probeTexture, Transform } from "@dylanebert/shallot";
import { fogReadBindGroup, fogReadLayout, readFogInScatter } from "@dylanebert/shallot/extras";
import { attachTexture, captureTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import tgpu from "typegpu";
import { MOTE_COUNT, Motes, MotesPlugin } from "./motes";
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
        attachTexture(world, camera, { width: 1280, height: 720 });
        world.storage(Transform).translation.set(camera, 0, 1.7, 2, 0);
        world.storage(Exposure).ev100.set(camera, 5);
        world.tick();
        const motes = world.resource(Motes);
        const positions = new Float32Array(MOTE_COUNT * 4);
        for (let i = 0; i < MOTE_COUNT; i++) positions[i * 4] = 100;
        positions.set([1, 1.05, -1.5, 0]);
        world.gpu.device.queue.writeBuffer(motes.positions!, 0, positions);
        motes.visible = true;
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
        for (let i = 0; i < 64; i++) world.step(0);
        const values: number[] = [];
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
            values.push(0.2126 * a[0]! + 0.7152 * a[1]! + 0.0722 * a[2]!);
        }
        const mean = values.reduce((s, v) => s + v, 0) / 16;
        const swing = Math.max(...values) - Math.min(...values);
        const result = { point: "edge", mean, swing, relative: swing / mean, values };
        world.step(0);
        const on = (await captureTexture(world, camera)).rgba;
        let at = 0,
            peak = -Infinity;
        for (let y = 450; y <= 460; y++)
            for (let x = 782; x <= 792; x++) {
                const i = (y * 1280 + x) * 4;
                const value = lum(on, i);
                if (value > peak) {
                    peak = value;
                    at = i;
                }
            }
        const displayMean = async (ev: number) => {
            world.storage(Exposure).ev100.set(camera, ev);
            for (let i = 0; i < 64; i++) world.step(0);
            let mean = 0;
            for (let frame = 0; frame < 16; frame++) {
                world.step(0);
                mean += lum((await captureTexture(world, camera)).rgba, at) / 16;
            }
            return mean;
        };
        const dimmer = await displayMean(5.25);
        const brighter = await displayMean(4.75);
        // Scaling the background along with the mote overstates its display response conservatively.
        const slope = (brighter - dimmer) / Math.log(2 ** 0.5);
        const estimatedDisplaySwing = result.relative * slope;
        console.log(
            JSON.stringify(
                {
                    result,
                    edgePixel: [(at / 4) % 1280, Math.floor(at / 4 / 1280)],
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
