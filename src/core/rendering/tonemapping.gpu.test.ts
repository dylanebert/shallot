import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { type Plugin, precompile } from "../../engine";
import { EffectStackPlugin, Vignette } from "../../extras/effect-stack";
import { Transform } from "../transform";
import {
    attachTexture,
    Camera,
    ColorGrading,
    CorePipelinePlugin,
    captureTexture,
    DebandDither,
    detachView,
    EffectPasses,
    RenderContext,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "./index";
import { tonemappingStateKey } from "./tonemapping-state";

setDefaultTimeout(TIMEOUT.gpu);
const vertex = `@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(uv * 2.0 - 1.0, 0.0, 1.0); }`;
let inversion: GPURenderPipeline;
const Scene: Plugin = {
    name: "PresentationComparisonScene",
    dependencies: [CorePipelinePlugin, EffectStackPlugin],
    initialize(world) {
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: `${vertex}
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    if (p.y < 8.0) { return vec4f(vec3f(exp2(p.x / 8.0 - 8.0)), 1.0); }
    let phase = p.x / 32.0;
    let hue = abs(fract(vec3f(phase) + vec3f(0.0, 0.333333, 0.666667)) * 6.0 - 3.0);
    return vec4f(clamp(hue - 1.0, vec3f(0.0), vec3f(1.0)) * exp2(p.y / 4.0 - 4.0), 1.0);
}`,
        });
        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module, entryPoint: "vertex" },
            fragment: {
                module,
                entryPoint: "fragment",
                targets: [{ format: world.resource(RenderContext).format }],
            },
            depthStencil: {
                format: "depth32float",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
        });
        const invertModule = device.createShaderModule({
            code: `${vertex}
@group(0) @binding(0) var input: texture_2d<f32>;
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f { return vec4f(1.0 - textureLoad(input, vec2u(p.xy), 0).rgb, 1.0); }`,
        });
        inversion = device.createRenderPipeline({
            label: "after-tonemap-invert",
            layout: "auto",
            vertex: { module: invertModule, entryPoint: "vertex" },
            fragment: {
                module: invertModule,
                entryPoint: "fragment",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
        });
        precompile(world, "presentation-comparison", () => [pipeline, inversion]);
        world.resource(RenderPhases).push({
            opaque(_world, _eid, _view, pass) {
                pass.setPipeline(pipeline);
                pass.draw(3);
            },
        });
    },
};
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [Scene] }]);

function camera(width = 128, height = 32) {
    const { world } = subjects()[0];
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).antialias.set(eid, 0);
    attachTexture(world, eid, { width, height });
    return { world, eid };
}
function maximum(a: Uint8ClampedArray, b: Uint8ClampedArray) {
    const result = [0, 0, 0, 0];
    for (let i = 0; i < a.length; i++)
        result[i % 4] = Math.max(result[i % 4], Math.abs(a[i] - b[i]));
    return result;
}

test("default tonemapping equals explicit TonyMcMapface", async () => {
    const { world, eid } = camera();
    world.step(0);
    const implicit = (await captureTexture(world, eid)).rgba;
    world.add(eid, Tonemapping, { method: TonemappingMethod.TonyMcMapface });
    world.step(0);
    expect((await captureTexture(world, eid)).rgba).toEqual(implicit);
    detachView(world, eid);
    world.destroy(eid);
});

test("flat mid-grey debanding varies codes by at most one and can be disabled per camera", async () => {
    const { world, eid } = camera(16, 16);
    const phases = world.resource(RenderPhases);
    const saved = phases.splice(0);
    world.resource(EffectPasses).set(eid, {
        before: [
            (_world, _eid, _view, _input, output) => {
                world
                    .frameEncoder()!
                    .beginRenderPass({
                        colorAttachments: [
                            {
                                view: output,
                                loadOp: "clear",
                                storeOp: "store",
                                clearValue: [0.25, 0.25, 0.25, 1],
                            },
                        ],
                    })
                    .end();
            },
        ],
        after: [],
    });
    try {
        const read = async () => {
            world.step(0);
            return (await captureTexture(world, eid)).rgba;
        };
        const uniform = (pixels: Uint8ClampedArray, count: number) => {
            for (let channel = 0; channel < 3; channel++)
                expect(new Set(Array.from(pixels).filter((_, i) => i % 4 === channel)).size).toBe(
                    count,
                );
        };
        const defaultOn = await read();
        uniform(defaultOn, 2);
        world.add(eid, DebandDither, { enabled: 0 });
        const defaultOff = await read();
        uniform(defaultOff, 1);
        expect(maximum(defaultOn, defaultOff).every((v) => v <= 1)).toBe(true);
        world.add(eid, Tonemapping, { method: TonemappingMethod.Linear });
        const linearOff = await read();
        uniform(linearOff, 1);
        world.storage(DebandDither).enabled.set(eid, 1);
        const linearOn = await read();
        uniform(linearOn, 2);
        expect(maximum(linearOn, linearOff).every((v) => v <= 1)).toBe(true);
        world.storage(Tonemapping).method.set(eid, TonemappingMethod.None);
        expect(await read()).toEqual(linearOff);
        world.add(eid, ColorGrading, { exposure: 2, postSaturation: 0, temperature: 1 });
        expect(await read()).toEqual(linearOff);
        world.storage(Tonemapping).method.set(eid, TonemappingMethod.Linear);
        expect(await read()).not.toEqual(linearOn);
    } finally {
        world.resource(EffectPasses).delete(eid);
        phases.push(...saved);
        detachView(world, eid);
        world.destroy(eid);
    }
});

test("after-tonemapping pass transforms the presented image; no registration adds no pass or target; steady play creates no groups", async () => {
    const { world, eid } = camera();
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    world.step(0);
    const baseline = await captureTexture(world, eid);
    const pipeline = inversion;
    const groups = new WeakMap<GPUTextureView, GPUBindGroup>();
    const stack = {
        before: [],
        after: [
            (
                _w: typeof world,
                _eid: number,
                _view: unknown,
                input: GPUTextureView,
                output: GPUTextureView,
            ) => {
                let group = groups.get(input);
                if (!group) {
                    group = device.createBindGroup({
                        layout: pipeline.getBindGroupLayout(0),
                        entries: [{ binding: 0, resource: input }],
                    });
                    groups.set(input, group);
                }
                const pass = world.frameEncoder()!.beginRenderPass({
                    colorAttachments: [{ view: output, loadOp: "clear", storeOp: "store" }],
                });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.draw(3);
                pass.end();
            },
        ],
    };
    const view = world.resource(Views).get(eid)!;
    expect(world.resource(tonemappingStateKey).targets.get(view)!.display).toHaveLength(0);
    const createEncoder = device.createCommandEncoder.bind(device);
    let count = 0;
    device.createCommandEncoder = (descriptor) => {
        const encoder = createEncoder(descriptor);
        const begin = encoder.beginRenderPass.bind(encoder);
        encoder.beginRenderPass = (pass) => {
            count++;
            return begin(pass);
        };
        return encoder;
    };
    try {
        world.step(0);
        const directCount = count;
        expect(directCount).toBe(2);
        count = 0;
        world.resource(EffectPasses).set(eid, stack);
        world.step(0);
        expect(count).toBe(directCount + 1);
        const effected = await captureTexture(world, eid);
        for (let i = 0; i < baseline.rgba.length; i++)
            expect(
                Math.abs(effected.rgba[i] - (i % 4 === 3 ? 255 : 255 - baseline.rgba[i])),
            ).toBeLessThanOrEqual(1);
        world.add(eid, Vignette, { intensity: 0.1 });
        count = 0;
        world.step(0);
        expect(count).toBe(directCount + 2);
        const vignetted = await captureTexture(world, eid);
        expect(vignetted.rgba).not.toEqual(effected.rgba);
        const original = device.createBindGroup.bind(device);
        let created = 0;
        device.createBindGroup = (descriptor) => {
            created++;
            return original(descriptor);
        };
        try {
            for (let i = 0; i < 4; i++) world.step(0);
            expect(created).toBe(0);
        } finally {
            device.createBindGroup = original;
        }
        world.resource(EffectPasses).delete(eid);
        world.remove(eid, Vignette);
        count = 0;
        world.step(0);
        expect(count).toBe(directCount);
        expect(world.resource(tonemappingStateKey).targets.get(view)!.display).toHaveLength(0);
    } finally {
        device.createCommandEncoder = createEncoder;
    }
    expect(await device.popErrorScope()).toBeNull();
    detachView(world, eid);
    world.destroy(eid);
});
