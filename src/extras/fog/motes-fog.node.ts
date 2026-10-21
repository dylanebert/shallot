import { expect, setDefaultTimeout, test } from "bun:test";
import { Camera, Exposure, probeTexture, StandardRenderer, Transform } from "@dylanebert/shallot";
import {
    Fog,
    FogPlugin,
    fogReadBindGroup,
    fogReadLayout,
    readFog,
    readFogInScatter,
} from "@dylanebert/shallot/extras";
import { attachTexture } from "@dylanebert/shallot/rendering";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { openGpuApp } from "@dylanebert/shallot/testing/gpu";
import tgpu from "typegpu";
import { room } from "./room.fixture";

setDefaultTimeout(TIMEOUT.node);

// in-scatter.node.ts holds shadowed local light, exposure exactly once and NoFog's zero.
// This claim retains the drawn point's own transmittance, not the wall's behind it.
test("a drawn point takes its own transmittance rather than the wall's", async () => {
    const app = await openGpuApp({ plugins: [FogPlugin] });
    try {
        const { world } = app;
        room(world);
        const camera = world.create();
        world.add(camera, Transform, { translation: [0, 1.7, 2, 0] });
        world.add(camera, Camera, { fov: 70, clearColor: 0x191816 });
        world.add(camera, StandardRenderer);
        world.add(camera, Exposure, { ev100: 5 });
        world.storage(Transform).translation.set(camera, 0, 0.7, 2, 0);
        world.storage(Exposure).ev100.set(camera, 12);
        attachTexture(world, camera, { width: 512, height: 512 });
        const fog = world.only([Fog]);
        world.storage(Fog).color.set(fog, 0);
        world.storage(Fog).absorption.set(fog, 1);
        world.storage(Fog).scattering.set(fog, 0);
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: tgpu.resolve({
                externals: { readFog, readFogInScatter },
                template: `
            @vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {
                let x=f32((i<<1u)&2u); let y=f32(i&2u); return vec4f(x*2-1,y*2-1,0,1);
            }
            @fragment fn sample()->@location(0) vec4f {
                let p=vec3f(0,0.7,0);
                return vec4f(0.98 * readFogInScatter(p) * readFog(p).a, 1);
            }`,
            }),
        });
        const empty = device.createBindGroupLayout({ entries: [] });
        const pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [empty, empty, empty, world.gpu.root.unwrap(fogReadLayout)],
            }),
            vertex: { module, entryPoint: "vertex" },
            fragment: { module, entryPoint: "sample", targets: [{ format: "rgba32float" }] },
        });
        const output = device.createTexture({
            size: [1, 1],
            format: "rgba32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        world.own(output);
        world.tick();
        const sample = async (density: number) => {
            world.storage(Fog).density.set(fog, density);
            world.step(0);
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
            return new Float32Array((await probeTexture(world, output)).bytes)[0]!;
        };
        const clear = await sample(0.00001);
        expect(clear).toBeGreaterThan(0.01);
        const ratio = (await sample(0.2)) / clear;
        const nearT = Math.exp(-0.2 * (2 - 0.1));
        const wallT = Math.exp(-0.2 * (5.3 - 0.1));
        console.log("point transmittance", { ratio, nearT, wallT });
        expect(Math.abs(ratio - nearT)).toBeLessThan(0.04);
        expect(Math.abs(ratio - wallT)).toBeGreaterThan(0.2);
    } finally {
        app.dispose();
    }
});
