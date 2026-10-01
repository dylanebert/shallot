import { expect, setDefaultTimeout, spyOn, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { Transform } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import {
    attachTexture,
    BeginFrameSystem,
    Camera,
    captureTexture,
    detachCanvas,
    Render,
    RenderPlugin,
    Resolution,
    Views,
} from "./index";

setDefaultTimeout(CEILING.gpu);
const subjects = gpuApps(import.meta.path, [
    {
        defaults: false,
        plugins: [
            RenderPlugin,
            {
                name: "Pattern",
                warm(world) {
                    const device = rawDevice(world.gpu.device);
                    const module = device.createShaderModule({
                        code: `
            @group(0) @binding(0) var outputImage: texture_storage_2d<${navigator.gpu.getPreferredCanvasFormat()}, write>;
            @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) p: vec3u) {
                textureStore(outputImage, p.xy, vec4f(f32(p.x + 17u), f32(p.y + 31u), 193.0, 255.0) / 255.0);
            }`,
                    });
                    world.resource(pipelineKey).pipeline = device.createComputePipeline({
                        layout: "auto",
                        compute: { module },
                    });
                },
                systems: [
                    {
                        group: "draw",
                        after: [BeginFrameSystem],
                        update(world) {
                            const pipeline = world.resource(pipelineKey).pipeline!;
                            for (const view of world.resource(Views).values()) {
                                if (!view.present) continue;
                                const device = rawDevice(world.gpu.device);
                                const pass = world.resource(Render).encoder!.beginComputePass();
                                pass.setPipeline(pipeline);
                                pass.setBindGroup(
                                    0,
                                    device.createBindGroup({
                                        layout: pipeline.getBindGroupLayout(0),
                                        entries: [{ binding: 0, resource: view.present }],
                                    }),
                                );
                                pass.dispatchWorkgroups(view.width, view.height);
                                pass.end();
                            }
                        },
                    },
                ],
            },
        ],
    },
]);
const pipelineKey = { create: () => ({ pipeline: null as GPUComputePipeline | null }) };

test("texture final frames capture tight RGBA and refuse missing presentation; owned targets release", async () => {
    const app = subjects()[0];
    const { world } = app;
    const eid = world.create();
    world.add(eid, Camera);
    world.add(eid, Transform);
    await expect(captureTexture(world, eid)).rejects.toThrow("no texture target");
    attachTexture(world, eid, { width: 2, height: 2 });
    const oldTexture = world.resource(Views).get(eid)!.texture!;
    const recycledTarget = spyOn(oldTexture, "destroy");
    world.destroy(eid);
    const recycled = world.create();
    expect(recycled).toBe(eid);
    world.add(recycled, Camera);
    world.add(recycled, Transform);
    world.step(0);
    expect(recycledTarget).toHaveBeenCalledTimes(1);
    await expect(captureTexture(world, recycled)).rejects.toThrow("no texture target");
    attachTexture(world, eid, { width: 7, height: 3 });
    world.add(eid, Resolution);
    world.storage(Resolution).width.set(eid, 1);
    await expect(captureTexture(world, eid)).rejects.toThrow("no frame");
    const texture = world.resource(Views).get(eid)!.texture!;
    console.log(
        `texture capture format: ${texture.format}; adapter classification reported at build`,
    );
    const destroyed = spyOn(texture, "destroy");
    world.gpu.device.pushErrorScope("validation");
    world.step(0);
    const capture = await captureTexture(world, eid);
    expect(await world.gpu.device.popErrorScope()).toBeNull();
    const expected = new Uint8ClampedArray(7 * 3 * 4);
    for (let y = 0; y < 3; y++)
        for (let x = 0; x < 7; x++) expected.set([x + 17, y + 31, 193, 255], (y * 7 + x) * 4);
    expect(capture.rgba).toEqual(expected);
    expect(capture.identity).toEqual({
        width: 7,
        height: 3,
        deviceScale: 1,
        surface: "final-texture",
        encoding: "rgba8-tight",
    });
    detachCanvas(world, eid);
    expect(destroyed).toHaveBeenCalledTimes(1);
    attachTexture(world, eid, { width: 2, height: 2 });
    const disposed = spyOn(world.resource(Views).get(eid)!.texture!, "destroy");
    world.remove(eid, Camera);
    world.step(0);
    expect(disposed).toHaveBeenCalledTimes(1);
    world.add(eid, Camera);
    attachTexture(world, eid, { width: 2, height: 2 });
    const finalTarget = spyOn(world.resource(Views).get(eid)!.texture!, "destroy");
    app.dispose();
    expect(finalTarget).toHaveBeenCalledTimes(1);
});
