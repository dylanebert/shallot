import type { ComputeNode, ExecutionContext } from "../compute";
import { SCENE_STRUCT_WGSL } from "./surface/structs";

export function compilePresentShader(): string {
    return /* wgsl */ `
${SCENE_STRUCT_WGSL}

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var maskTexture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> scene: Scene;

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn linearToOKLab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2220049174 * c.g + 0.6896926207 * c.b;
    let l_ = pow(max(l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(s, 0.0), 1.0 / 3.0);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn OKLabToLinear(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

fn applyPosterize(color: vec3<f32>) -> vec3<f32> {
    if (scene.posterizeBands <= 0.0) { return color; }
    var lab = linearToOKLab(color);
    let L = clamp(lab.x, 0.0, 1.0);
    lab.x = floor(L * scene.posterizeBands + 0.5) / scene.posterizeBands;
    lab.z += (lab.x - 0.5) * 0.05;
    return max(OKLabToLinear(lab), vec3<f32>(0.0));
}

fn bayer4(pos: vec2<f32>) -> f32 {
    let x = u32(pos.x) % 4u;
    let y = u32(pos.y) % 4u;
    let m = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0,
    );
    return m[x + y * 4u] / 16.0 - 0.5;
}

fn applyDither(color: vec3<f32>, pos: vec2<f32>) -> vec3<f32> {
    if (scene.ditherStrength <= 0.0) { return color; }
    let d = bayer4(pos) * scene.ditherStrength;
    return color + vec3<f32>(d);
}

fn applyVignette(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (scene.vignetteStrength <= 0.0) { return color; }
    let d = distance(uv, vec2<f32>(0.5, 0.5));
    let v = 1.0 - smoothstep(scene.vignetteInner, scene.vignetteOuter, d) * scene.vignetteStrength;
    return color * v;
}

fn applyTonemap(color: vec3<f32>) -> vec3<f32> {
    if (scene.tonemapMode == 0u) { return color; }
    return aces(color * scene.exposure);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn loadInput(coord: vec2<i32>, dims: vec2<i32>) -> vec3<f32> {
    return textureLoad(inputTexture, clamp(coord, vec2<i32>(0), dims - 1), 0).rgb;
}

const FXAA_REDUCE_MIN: f32 = 1.0 / 128.0;
const FXAA_REDUCE_MUL: f32 = 1.0 / 8.0;
const FXAA_SPAN_MAX: f32 = 8.0;

fn applyFXAA(coord: vec2<i32>, colorM: vec3<f32>, dims: vec2<i32>) -> vec3<f32> {
    let colorNW = loadInput(coord + vec2<i32>(-1, -1), dims);
    let colorNE = loadInput(coord + vec2<i32>(1, -1), dims);
    let colorSW = loadInput(coord + vec2<i32>(-1, 1), dims);
    let colorSE = loadInput(coord + vec2<i32>(1, 1), dims);

    let lumaM = luma(colorM);
    let lumaNW = luma(colorNW);
    let lumaNE = luma(colorNE);
    let lumaSW = luma(colorSW);
    let lumaSE = luma(colorSE);

    let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    var dir: vec2<f32>;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    let dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * FXAA_REDUCE_MUL,
        FXAA_REDUCE_MIN,
    );
    let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    let dirPixels = clamp(
        dir * rcpDirMin,
        vec2<f32>(-FXAA_SPAN_MAX),
        vec2<f32>(FXAA_SPAN_MAX),
    );

    let fc = vec2<f32>(f32(coord.x), f32(coord.y));
    let colorA = 0.5 * (
        loadInput(vec2<i32>(round(fc + dirPixels * (1.0 / 3.0 - 0.5))), dims) +
        loadInput(vec2<i32>(round(fc + dirPixels * (2.0 / 3.0 - 0.5))), dims)
    );

    let colorB = colorA * 0.5 + 0.25 * (
        loadInput(vec2<i32>(round(fc + dirPixels * -0.5)), dims) +
        loadInput(vec2<i32>(round(fc + dirPixels * 0.5)), dims)
    );

    let lumaB = luma(colorB);
    if (lumaB < lumaMin || lumaB > lumaMax) { return colorA; }
    return colorB;
}

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VOut {
    let xy = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    var out: VOut;
    out.pos = vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(xy.x, 1.0 - xy.y);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let inDims = vec2<i32>(textureDimensions(inputTexture));
    let inCoord = clamp(
        vec2<i32>(in.uv * vec2<f32>(inDims)),
        vec2<i32>(0),
        inDims - 1,
    );

    var color = textureLoad(inputTexture, inCoord, 0).rgb;
    let inPos = vec2<f32>(f32(inCoord.x), f32(inCoord.y));

    if (scene.fxaaEnabled != 0u) {
        let mask = textureLoad(maskTexture, inCoord, 0).r;
        let fxaaColor = applyFXAA(inCoord, color, inDims);
        color = select(fxaaColor, color, mask >= 0.5);
    }

    color = applyTonemap(color);
    color = linearToSrgb(saturate(color));
    color = applyDither(color, inPos);
    color = applyPosterize(color);
    color = applyVignette(color, in.uv);

    return vec4<f32>(saturate(color), 1.0);
}
`;
}

interface PresentCache {
    bindGroup: GPUBindGroup | null;
    cachedInputView: GPUTextureView | null;
    cachedMaskView: GPUTextureView | null;
}

export function createPresentNode(scene: GPUBuffer): ComputeNode {
    let pipeline: GPURenderPipeline | null = null;
    const cache: PresentCache = {
        bindGroup: null,
        cachedInputView: null,
        cachedMaskView: null,
    };

    return {
        name: "present",
        inputs: ["color", "mask"],
        outputs: ["framebuffer"],

        async prepare(device: GPUDevice) {
            const format = navigator.gpu.getPreferredCanvasFormat();
            const code = compilePresentShader();
            const module = device.createShaderModule({ code });
            pipeline = await device.createRenderPipelineAsync({
                label: "present",
                layout: "auto",
                vertex: { module, entryPoint: "vs" },
                fragment: { module, entryPoint: "fs", targets: [{ format }] },
                primitive: { topology: "triangle-list" },
            });
        },

        execute(ctx: ExecutionContext) {
            if (!pipeline) return;
            const { device, encoder, canvasView } = ctx;
            const inputView = ctx.getTextureView("color");
            const maskView = ctx.getTextureView("mask");
            if (!inputView || !maskView) return;

            if (inputView !== cache.cachedInputView || maskView !== cache.cachedMaskView) {
                cache.bindGroup = device.createBindGroup({
                    layout: pipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: inputView },
                        { binding: 1, resource: maskView },
                        { binding: 2, resource: { buffer: scene } },
                    ],
                });
                cache.cachedInputView = inputView;
                cache.cachedMaskView = maskView;
            }

            const ts = ctx.timestampWrites?.("present");
            const pass = encoder.beginRenderPass({
                label: "present",
                colorAttachments: [
                    {
                        view: canvasView,
                        loadOp: "clear",
                        storeOp: "store",
                        clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    },
                ],
                timestampWrites: ts,
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, cache.bindGroup!);
            pass.draw(3);
            pass.end();
        },
    };
}
