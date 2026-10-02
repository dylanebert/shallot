import { expect, setDefaultTimeout, test } from "bun:test";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { CEILING } from "../../../scripts/test-tiers";
import { type Plugin, Transform } from "../../engine";
import { Vignette, VignettePlugin } from "../../extras/vignette";
import { bevyGrading } from "./fixtures/bevy-grading";
import {
    attachTexture,
    Camera,
    ColorGrading,
    CorePipelinePlugin,
    captureTexture,
    detachCanvas,
    EffectPasses,
    Render,
    RenderPhases,
    Tonemapping,
    TonemappingMethod,
    Views,
} from "./index";
import { TonemappingSystem, tonemappingStateKey } from "./tonemapping-state";

setDefaultTimeout(CEILING.gpu);
const sources = { create: () => new Map<number, GPUTextureView>() };
const Scene: Plugin = {
    name: "PresentationComparisonScene",
    dependencies: [CorePipelinePlugin, VignettePlugin],
    systems: [
        {
            group: "draw",
            after: [TonemappingSystem],
            update(world) {
                for (const [eid, view] of world.resource(Views))
                    if (view.framebuffer) world.resource(sources).set(eid, view.framebuffer);
            },
        },
    ],
    initialize(world) {
        const device = world.gpu.device;
        const module = device.createShaderModule({
            code: `
@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
}
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
const vertex = `@vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(uv * 2.0 - 1.0, 0.0, 1.0); }`;
const encode = `fn encode(c: vec3f) -> vec3f { return select(1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308)); }`;
const config = `struct Config { exposure: f32, temperature: f32, tint: f32, hue: f32, postSaturation: f32, mode: u32, range: vec2f, saturation: vec4f, contrast: vec4f, gamma: vec4f, gain: vec4f, lift: vec4f }; @group(0) @binding(1) var<uniform> g: Config;`;
const reference = `
struct ColorGrading { balance: mat3x3f, saturation: vec3f, contrast: vec3f, gamma: vec3f, gain: vec3f, lift: vec3f, midtone_range: vec2f, exposure: f32, hue: f32, post_saturation: f32 };
${bevyGrading}
fn reference_grade(input: vec3f) -> vec3f {
    var color = max(input, vec3f(0.0));
    if (g.hue != 0.0) { var hsv = rgb_to_hsv(color); hsv.r = (hsv.r + g.hue) % 6.283185307179586; color = hsv_to_rgb(hsv); }
    if (g.temperature != 0.0 || g.tint != 0.0) {
        let xy = vec2f(0.31272 - g.temperature, 0.32903 + g.tint);
        let white = vec3f(0.701634, 1.15856, -0.904175) + (vec3f(-0.051461, 0.045854, 0.953127) + vec3f(0.452749, -0.296122, -0.955206) * xy.x) / xy.y;
        let adjust = vec3f(0.975538, 1.01648, 1.08475) / white;
        let to_lms = mat3x3f(vec3f(0.311692, 0.0905138, 0.00764433), vec3f(0.652085, 0.901341, 0.0486554), vec3f(0.0362225, 0.00814478, 0.9437));
        let to_rgb = mat3x3f(vec3f(4.06305, -0.40791, -0.0118812), vec3f(-2.93241, 1.40437, -0.0486532), vec3f(-0.130646, 0.0035363, 1.0605344));
        let balance = to_rgb * mat3x3f(vec3f(adjust.x, 0.0, 0.0), vec3f(0.0, adjust.y, 0.0), vec3f(0.0, 0.0, adjust.z)) * to_lms;
        color = max(balance * color, vec3f(0.0));
    }
    var grading: ColorGrading;
    grading.saturation = g.saturation.xyz; grading.contrast = g.contrast.xyz;
    grading.gamma = g.gamma.xyz; grading.gain = g.gain.xyz; grading.lift = g.lift.xyz;
    grading.midtone_range = g.range; grading.exposure = g.exposure;
    return sectional_color_grading(color, &grading);
}`;

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

test("TonyMcMapface HDR ramp and saturated sweep match the pinned HLSL sampling; default equals explicit Tony", async () => {
    const { world, eid } = camera();
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    world.step(0);
    const actual = await captureTexture(world, eid);
    world.add(eid, Tonemapping, { method: TonemappingMethod.TonyMcMapface });
    world.step(0);
    expect((await captureTexture(world, eid)).rgba).toEqual(actual.rgba);
    const view = world.resource(Views).get(eid)!;
    const state = world.resource(tonemappingStateKey);
    // tony-mc-mapface 0f249d366c9e960aa9828818786a6d5900fd85d9, shader/tony_mc_mapface.hlsl.
    const module = device.createShaderModule({
        code: `${vertex} ${encode}
@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var lut: texture_3d<f32>;
@group(0) @binding(2) var linear_clamp: sampler;
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let stimulus = textureLoad(input, vec2u(p.xy), 0).rgb;
    let encoded = stimulus / (stimulus + 1.0);
    let LUT_DIMS = 48.0;
    let uv = encoded * ((LUT_DIMS - 1.0) / LUT_DIMS) + 0.5 / LUT_DIMS;
    return vec4f(encode(textureSampleLevel(lut, linear_clamp, uv, 0.0).rgb), 1.0);
}`,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
            module,
            entryPoint: "fragment",
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
    });
    const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: world.resource(sources).get(eid)! },
            { binding: 1, resource: state.built!.lut },
            { binding: 2, resource: state.built!.sampler },
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: view.texture!.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const expected = await captureTexture(world, eid);
    const diff = maximum(actual.rgba, expected.rgba);
    console.log(`Tony ramp/sweep max RGBA ${diff}`);
    expect(diff.every((v) => v <= 1)).toBe(true);
    expect(await device.popErrorScope()).toBeNull();
    detachCanvas(world, eid);
    world.destroy(eid);
});

test("ColorGrading matches extracted corrected Bevy WGSL on the same HDR inputs", async () => {
    const { world, eid } = camera();
    const device = world.gpu.device;
    world.add(eid, Tonemapping, { method: TonemappingMethod.None });
    world.add(eid, ColorGrading, {
        exposure: -1.3,
        temperature: 0.015,
        tint: -0.007,
        hue: 0.4,
        postSaturation: 0.85,
        saturation: [0.6, 1.1, 0.8, 0],
        contrast: [0.9, 1.2, 0.8, 0],
        gamma: [0.8, 1.3, 1.1, 0],
        gain: [1.1, 0.9, 1.2, 0],
        lift: [-0.005, 0.012, 0.02, 0],
    });
    device.pushErrorScope("validation");
    world.step(0);
    const actual = await captureTexture(world, eid);
    const view = world.resource(Views).get(eid)!;
    const state = world.resource(tonemappingStateKey);
    const module = device.createShaderModule({
        code: `${vertex} ${encode} ${config} ${reference}
@group(0) @binding(0) var input: texture_2d<f32>;
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f {
    var color = reference_grade(textureLoad(input, vec2u(p.xy), 0).rgb);
    color = mix(vec3f(tonemapping_luminance(color)), color, g.postSaturation);
    return vec4f(encode(max(color, vec3f(0.0))), 1.0);
}`,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
            module,
            entryPoint: "fragment",
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
    });
    const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: world.resource(sources).get(eid)! },
            { binding: 1, resource: { buffer: state.buffers[view.slot] } },
        ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: view.texture!.createView(), loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.draw(3);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const diff = maximum(actual.rgba, (await captureTexture(world, eid)).rgba);
    console.log(`Bevy grading max RGBA ${diff}`);
    expect(diff.every((v) => v <= 1)).toBe(true);
    expect(await device.popErrorScope()).toBeNull();
    detachCanvas(world, eid);
    world.destroy(eid);
});

test("after-tonemapping pass transforms the presented image; no registration adds no pass or target; steady play creates no groups", async () => {
    const { world, eid } = camera();
    const device = world.gpu.device;
    device.pushErrorScope("validation");
    world.step(0);
    const baseline = await captureTexture(world, eid);
    const module = device.createShaderModule({
        code: `${vertex}
@group(0) @binding(0) var input: texture_2d<f32>;
@fragment fn fragment(@builtin(position) p: vec4f) -> @location(0) vec4f { return vec4f(1.0 - textureLoad(input, vec2u(p.xy), 0).rgb, 1.0); }`,
    });
    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module, entryPoint: "vertex" },
        fragment: {
            module,
            entryPoint: "fragment",
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
        },
    });
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
                const pass = world.resource(Render).encoder!.beginRenderPass({
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
        device.createBindGroup = (desc) => {
            created++;
            return original(desc);
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
    detachCanvas(world, eid);
    world.destroy(eid);
});
