import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { probeTexture } from "../../engine/runtime";
import { Transform } from "../transform";
import { gradingDefaults } from "./color-grading";
import {
    attachTexture,
    Camera,
    ColorGrading,
    CorePipelinePlugin,
    captureTexture,
    DebandDither,
    EffectPasses,
    Tonemapping,
    TonemappingMethod,
} from "./index";
import {
    bevyLut,
    bevyPipeline,
    bevyUniform,
    type Grade,
    methods,
} from "./tonemapping-bevy.fixture";

setDefaultTimeout(TIMEOUT.node);
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [CorePipelinePlugin] }]);
const width = 128,
    height = 32;
let eid: number;
let input: GPUTexture;
let output: GPUTexture;
const pipelines = new Map<string, GPURenderPipeline>();
const vertex = `@vertex fn vertex(@builtin(vertex_index) i:u32)->@builtin(position) vec4f {let uv=vec2f(f32((i<<1u)&2u),f32(i&2u));return vec4f(uv*2.0-1.0,0.0,1.0);}`;
beforeAll(async () => {
    const { world } = subjects()[0];
    const device = world.gpu.device;
    const info = device.adapterInfo;
    console.log(
        JSON.stringify({
            adapter: {
                vendor: info.vendor,
                architecture: info.architecture,
                device: info.device,
                description: info.description,
                isFallbackAdapter: info.isFallbackAdapter,
            },
        }),
    );
    for (const method of methods)
        for (const dither of [false, true])
            for (const graded of [false, true])
                pipelines.set(
                    `${method[0]}:${dither}:${graded}`,
                    await bevyPipeline(world, method, dither, graded),
                );
    input = device.createTexture({
        size: [width, height],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    world.own(input);
    output = device.createTexture({
        size: [width, height],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const module = device.createShaderModule({
        code: `${vertex}
@fragment fn fragment(@builtin(position) p:vec4f)->@location(0) vec4f {
    if(p.y<1.0){return vec4f(0.0,0.0,0.0,1.0);}
    if(p.y<8.0){return vec4f(vec3f(exp2(p.x/4.0-20.0)),1.0);}
    let phase=p.x/32.0;
    let hue=abs(fract(vec3f(phase)+vec3f(0.0,0.333333,0.666667))*6.0-3.0);
    return vec4f(clamp(hue-1.0,vec3f(0.0),vec3f(1.0))*exp2(p.y/4.0-4.0),1.0);
}`,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: { module, entryPoint: "fragment", targets: [{ format: "rgba16float" }] },
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: input.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const copyModule = device.createShaderModule({
        code: `${vertex}
@group(0) @binding(0) var image:texture_2d<f32>;
@fragment fn fragment(@builtin(position) p:vec4f)->@location(0) vec4f {return textureLoad(image,vec2u(p.xy),0);}`,
    });
    const copy = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: copyModule, entryPoint: "vertex" },
        fragment: {
            module: copyModule,
            entryPoint: "fragment",
            targets: [{ format: "rgba16float" }],
        },
    });
    const group = device.createBindGroup({
        layout: copy.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: input.createView() }],
    });
    eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera, { antialias: 0 });
    world.add(eid, Tonemapping);
    world.add(eid, DebandDither);
    attachTexture(world, eid, { width, height });
    world.resource(EffectPasses).set(eid, {
        before: [
            (w, _eid, _view, _input, destination) => {
                const pass = w.frameEncoder()!.beginRenderPass({
                    colorAttachments: [{ view: destination, loadOp: "clear", storeOp: "store" }],
                });
                pass.setPipeline(copy);
                pass.setBindGroup(0, group);
                pass.draw(3);
                pass.end();
            },
        ],
        after: [],
    });
}, TIMEOUT.startup);

test("the Bevy gate covers every TonemappingMethod", () => {
    expect(methods.map((method) => method[1]).sort()).toEqual(
        Object.values(TonemappingMethod).sort(),
    );
});

const grading: Grade = {
    exposure: -1.3,
    temperature: 0.015,
    tint: -0.007,
    hue: (24 * Math.PI) / 180,
    postSaturation: 0.85,
    midtonesRange: [0.28, 0.76],
    saturation: [0.6, 1.1, 0.8],
    contrast: [0.9, 1.2, 0.8],
    gamma: [0.8, 1.3, 1.1],
    gain: [1.1, 0.9, 1.2],
    lift: [-0.005, 0.012, 0.02],
};
for (const method of methods)
    for (const dither of [false, true])
        for (const graded of [false, true]) {
            test(`${method[0]}, debanding ${dither ? "on" : "off"}, grading ${graded ? "all fields (Blender crossover)" : "identity"} matches live Bevy within one display code`, async () => {
                const { world } = subjects()[0];
                const device = world.gpu.device;
                world.storage(Tonemapping).method.set(eid, method[1]);
                world.storage(DebandDither).enabled.set(eid, dither ? 1 : 0);
                if (graded) world.add(eid, ColorGrading, grading);
                else world.remove(eid, ColorGrading);
                world.step(0);
                const actual = (await captureTexture(world, eid)).rgba;
                const pipeline = pipelines.get(`${method[0]}:${dither}:${graded}`)!;
                const lut = bevyLut(world, method);
                const uniform = bevyUniform(world, graded ? grading : gradingDefaults);
                const group = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: { buffer: uniform } },
                        { binding: 1, resource: input.createView() },
                        { binding: 2, resource: device.createSampler() },
                        { binding: 3, resource: lut.createView() },
                        {
                            binding: 4,
                            resource: device.createSampler({
                                minFilter: "linear",
                                magFilter: "linear",
                                mipmapFilter: "linear",
                            }),
                        },
                    ],
                });
                const encoder = device.createCommandEncoder();
                const pass = encoder.beginRenderPass({
                    colorAttachments: [
                        { view: output.createView(), loadOp: "clear", storeOp: "store" },
                    ],
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.draw(3);
                pass.end();
                device.queue.submit([encoder.finish()]);
                const expected = new Uint8Array(
                    (await probeTexture(world, output)).bytes as ArrayBuffer,
                );
                let max = 0;
                let worst = 0;
                for (let i = 0; i < actual.length; i++) {
                    const delta = Math.abs(actual[i]! - expected[i]!);
                    if (delta > max) {
                        max = delta;
                        worst = i;
                    }
                }
                console.log(
                    JSON.stringify({
                        method: method[0],
                        dither,
                        graded,
                        max,
                        pixel: [Math.floor(worst / 4) % width, Math.floor(worst / 4 / width)],
                        channel: worst % 4,
                        actual: actual[worst],
                        bevy: expected[worst],
                    }),
                );
                expect(max).toBeLessThanOrEqual(1);
            });
        }
