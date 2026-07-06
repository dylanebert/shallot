// Glaze — the default postfx composite + the postfx chain. A renderer draws into each camera's offscreen
// scene-color target (`view.framebuffer`); glaze runs one compute dispatch per camera that reads it and
// writes the swapchain (`view.present`), applying the per-camera postfx chain on the way. The swapchain
// is a base-format storage texture (not sRGB), so glaze encodes linear→sRGB itself (`linearToSrgb`) — the
// same path a consumer's own fused composite takes. Compute, not a render pass, so the present costs no
// tile load/store on TBDR (gpu.md "Render passes on TBDR"); WebGPU exposes no programmable blending, so a
// compute dispatch reading the offscreen and writing the swapchain once is the portable fused-postfx
// substitute. Renderer-agnostic: it imports only `render/core` and reads `view.framebuffer` / `view.present`,
// so sear (MSAA-resolved) and a custom renderer (single-sample) both composite through it. A renderer
// orders itself ahead with `before: [GlazeSystem]`; glaze never imports a renderer.
//
// The chain: a scene-referred grade (ASC CDL + exposure) on the HDR radiance, then a tonemap operator
// (`Tonemap`, default Khronos Neutral), then a post-tonemap saturation, then OkLab-L posterize/dither
// (dither before posterize so noise breaks bands), then γ-2.2 vignette. The grade straddles the tonemap
// on purpose: CDL + exposure are scene-referred so they sit before the operator (a pushed highlight
// rolls off its shoulder), saturation is perceptual so it sits after (display-referred values read
// predictably). No post-process AA — MSAA in sear is the geometric AA baseline. A camera with no `Glaze`
// still tonemaps Neutral (the zero-config default display transform); the grade defaults to a no-op and
// posterize / dither / vignette gate off, so only the tonemap + linear→sRGB encode run. The
// rg11b10ufloat HDR offscreen is what lets the tonemap roll off highlights >1 (they'd clamp at store on
// an LDR offscreen).
import type { Plugin, State, System } from "../../engine";
import { Compute, f32, sparse, u32, vec4 } from "../../engine";
import { Camera, RenderPlugin } from "../render";
import {
    BeginFrameSystem,
    LINEAR_TO_SRGB_WGSL,
    MAX_VIEWS,
    Render,
    VIEW_STRIDE,
    Views,
} from "../render/core";
import { TONEMAP_WGSL } from "./tonemap";

export { TONEMAP_WGSL, Tonemap } from "./tonemap";

/**
 * per-camera postfx tuning. A camera tonemaps Neutral by default (no `Glaze` needed); add `Glaze` to
 * pick a different {@link Tonemap} operator, dial a color grade, or enable vignette / posterize / dither.
 * `tonemap` is a {@link Tonemap} index (0 = Neutral default, 1 = None); `exposure` scales the scene
 * pre-tonemap; the grade is ASC CDL `slope`/`offset`/`power` (per-channel rgb, scene-referred, pre-tonemap)
 * plus a post-tonemap `saturation`; `vignette` is corner darkness in [0,1] between `vignetteInner` and
 * `vignetteOuter` screen radii; `posterize` is the band count (0 = off) and `dither` the OkLab-L dither
 * amplitude that breaks bands. The grade defaults to a no-op (slope/power 1, offset 0, saturation 1).
 *
 * @example
 * ```
 * // warm, crushed, slightly desaturated
 * <a camera sear glaze="slope: 1.05 1 0.9; offset: -0.02 -0.02 -0.02; power: 1.2 1.2 1.2; saturation: 0.85" />
 * ```
 */
export const Glaze = {
    exposure: sparse(f32),
    tonemap: sparse(u32),
    slope: sparse(vec4),
    offset: sparse(vec4),
    power: sparse(vec4),
    saturation: sparse(f32),
    vignette: sparse(f32),
    vignetteInner: sparse(f32),
    vignetteOuter: sparse(f32),
    posterize: sparse(f32),
    dither: sparse(f32),
};

const WORKGROUP = 8;

// dynamic-offset uniform stride — `minUniformBufferOffsetAlignment` ≥ 256 forces it even though the
// struct is 32 bytes. One slot per view, indexed by `view.slot`, so each camera's config is its own
// region: `writeBuffer` is queue-ordered against the submit, so a single rewritten uniform would
// clobber every camera's composite with the last camera's config. Slot-major writes never collide
const GLAZE_STRIDE = VIEW_STRIDE;
const GLAZE_STRUCT_BYTES = 80;
const GLAZE_UNIFORM_SIZE = GLAZE_STRIDE * MAX_VIEWS;

const GLAZE_STRUCT_WGSL = /* wgsl */ `
struct Glaze {
    exposure: f32,
    vignetteStrength: f32,
    vignetteInner: f32,
    vignetteOuter: f32,
    posterizeBands: f32,
    ditherStrength: f32,
    tonemapMode: u32,
    saturation: f32,
    slope: vec4<f32>,
    offset: vec4<f32>,
    power: vec4<f32>,
}`;

function code(format: GPUTextureFormat): string {
    return /* wgsl */ `
${GLAZE_STRUCT_WGSL}

@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var<uniform> glaze: Glaze;
@group(0) @binding(2) var output: texture_storage_2d<${format}, write>;

${LINEAR_TO_SRGB_WGSL}
${TONEMAP_WGSL}

fn oklabL(c: vec3<f32>) -> f32 {
    let lms_l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let lms_m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let lms_s = 0.0883024619 * c.r + 0.2220049174 * c.g + 0.6896926207 * c.b;
    let l_ = pow(max(lms_l, 0.0), 1.0 / 3.0);
    let m_ = pow(max(lms_m, 0.0), 1.0 / 3.0);
    let s_ = pow(max(lms_s, 0.0), 1.0 / 3.0);
    return 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
}

// scene-referred grade: ASC CDL (out = (in*slope + offset)^power) then exposure, on linear radiance.
// Pre-tonemap, so the operator rolls any pushed highlight off its shoulder; the HDR offscreen is what
// holds radiance >1 for it to roll off.
fn applyGrade(c: vec3<f32>) -> vec3<f32> {
    let cdl = pow(max(c * glaze.slope.rgb + glaze.offset.rgb, vec3<f32>(0.0)), glaze.power.rgb);
    return cdl * glaze.exposure;
}

// perceptual saturation, post-tonemap (Bevy's post_saturation): display-referred values read predictably
fn applySaturation(c: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(tmLuma(c)), c, glaze.saturation);
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

// dither runs before posterize so noise pushes adjacent pixels across band boundaries. Caller scales
// color by newL/oldL to preserve hue.
fn ditherPosterizeL(L: f32, pos: vec2<f32>) -> f32 {
    var out = L;
    if (glaze.ditherStrength > 0.0) {
        out = out + bayer4(pos) * glaze.ditherStrength;
    }
    if (glaze.posterizeBands > 0.0) {
        out = floor(saturate(out) * glaze.posterizeBands + 0.5) / glaze.posterizeBands;
    }
    return out;
}

// strength means perceived corner darkness; pow(v, 2.2) compensates for the sRGB encode that follows so
// a 0.5-strength vignette renders ~0.5 perceived (not ~0.73).
fn applyVignette(color: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (glaze.vignetteStrength <= 0.0) { return color; }
    let d = distance(uv, vec2<f32>(0.5, 0.5));
    let v = 1.0 - smoothstep(glaze.vignetteInner, glaze.vignetteOuter, d) * glaze.vignetteStrength;
    return color * pow(v, 2.2);
}

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(output);
    if (gid.x >= dim.x || gid.y >= dim.y) { return; }
    let p = vec2<i32>(gid.xy);
    // the offscreen is Render.format (sRGB), so textureLoad returns linear
    var color = textureLoad(input, p, 0).rgb;

    color = tonemap(glaze.tonemapMode, applyGrade(color));
    color = applySaturation(color);
    color = saturate(color);

    if (glaze.posterizeBands > 0.0 || glaze.ditherStrength > 0.0) {
        let oldL = max(oklabL(color), 1e-4);
        let newL = ditherPosterizeL(oldL, vec2<f32>(gid.xy));
        color = max(color * (newL / oldL), vec3<f32>(0.0));
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dim);
    color = applyVignette(color, uv);

    textureStore(output, p, vec4<f32>(linearToSrgb(max(color, vec3<f32>(0.0))), 1.0));
}
`;
}

const _glaze = {
    pipeline: null as GPUComputePipeline | null,
    layout: null as GPUBindGroupLayout | null,
    buffer: null as GPUBuffer | null,
};

const _scratch = new ArrayBuffer(GLAZE_STRUCT_BYTES);
const _scratchF32 = new Float32Array(_scratch);
const _scratchU32 = new Uint32Array(_scratch);

// pack a camera's postfx config into its uniform slot. A zeroed config tonemaps Neutral (mode 0) at
// exposure 1 — vignette / posterize / dither each gate on their own zero default, so a camera without
// `Glaze` gets the default Neutral display transform and nothing else. Writes only this slot — distinct
// offsets never collide
function uploadConfig(state: State, eid: number, slot: number): void {
    _scratchF32.fill(0);
    // grade identities so the no-`Glaze` zeroed path is a no-op: exposure 1, slope/power 1, saturation 1
    // (offset 0 is already the fill). The no-`Glaze` path is then Neutral at unit exposure, no grade.
    _scratchF32[0] = 1; // exposure
    _scratchF32[7] = 1; // saturation
    _scratchF32[8] = 1; // slope.r
    _scratchF32[9] = 1; // slope.g
    _scratchF32[10] = 1; // slope.b
    _scratchF32[16] = 1; // power.r
    _scratchF32[17] = 1; // power.g
    _scratchF32[18] = 1; // power.b
    if (state.has(eid, Glaze)) {
        _scratchF32[0] = Glaze.exposure.get(eid);
        _scratchF32[1] = Glaze.vignette.get(eid);
        _scratchF32[2] = Glaze.vignetteInner.get(eid);
        _scratchF32[3] = Glaze.vignetteOuter.get(eid);
        _scratchF32[4] = Glaze.posterize.get(eid);
        _scratchF32[5] = Glaze.dither.get(eid);
        _scratchU32[6] = Glaze.tonemap.get(eid);
        _scratchF32[7] = Glaze.saturation.get(eid);
        _scratchF32[8] = Glaze.slope.x.get(eid);
        _scratchF32[9] = Glaze.slope.y.get(eid);
        _scratchF32[10] = Glaze.slope.z.get(eid);
        _scratchF32[12] = Glaze.offset.x.get(eid);
        _scratchF32[13] = Glaze.offset.y.get(eid);
        _scratchF32[14] = Glaze.offset.z.get(eid);
        _scratchF32[16] = Glaze.power.x.get(eid);
        _scratchF32[17] = Glaze.power.y.get(eid);
        _scratchF32[18] = Glaze.power.z.get(eid);
    }
    Compute.device.queue.writeBuffer(_glaze.buffer!, slot * GLAZE_STRIDE, _scratch);
}

/**
 * the postfx composite, per camera: reads the camera's offscreen scene color (`view.framebuffer`) and
 * writes the swapchain (`view.present`) through one compute dispatch, applying its {@link Glaze} chain
 * and the linear→sRGB encode. Renderer-agnostic: it queries every camera with both targets, so sear and
 * custom renderers compose the same way. Runs after every renderer (each declares `before: [GlazeSystem]`);
 * a canvas-less view (a shadow light) has no `present` and is skipped. The bind group is rebuilt per frame
 * because the swapchain view changes each frame (`getCurrentTexture`).
 */
export const GlazeSystem: System = {
    name: "glaze",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    update(state) {
        const encoder = Render.encoder;
        const device = Compute.device;
        if (!encoder || !device || !_glaze.pipeline || !_glaze.layout) return;
        for (const eid of state.query([Camera])) {
            const view = Views.get(eid);
            if (!view?.present || !view.framebuffer) continue;
            uploadConfig(state, eid, view.slot);
            const group = device.createBindGroup({
                label: `glaze/${eid}`,
                layout: _glaze.layout,
                entries: [
                    { binding: 0, resource: view.framebuffer },
                    { binding: 1, resource: { buffer: _glaze.buffer!, size: GLAZE_STRUCT_BYTES } },
                    { binding: 2, resource: view.present },
                ],
            });
            const pass = encoder.beginComputePass({
                label: `glaze/${eid}`,
                timestampWrites: Compute.span?.("glaze"),
            });
            pass.setPipeline(_glaze.pipeline);
            pass.setBindGroup(0, group, [view.slot * GLAZE_STRIDE]);
            pass.dispatchWorkgroups(
                Math.ceil(view.width / WORKGROUP),
                Math.ceil(view.height / WORKGROUP),
            );
            pass.end();
        }
    },
};

/**
 * the default postfx composite. A renderer draws into `view.framebuffer` and glaze composites it to the
 * swapchain (`view.present`) via one compute dispatch per camera. Presenting is a composite the consumer
 * picks ({@link SearPlugin} depends only on `RenderPlugin`): register `GlazePlugin` for the zero-config
 * postfx chain, or ship a custom composite instead. Add a {@link Glaze} component to a camera to pick a
 * tonemap, dial a color grade, or enable vignette / posterize / dither.
 */
export const GlazePlugin: Plugin = {
    name: "Glaze",
    components: { Glaze },
    traits: {
        Glaze: {
            requires: [Camera],
            defaults: () => ({
                exposure: 1,
                tonemap: 0,
                slope: [1, 1, 1, 0],
                offset: [0, 0, 0, 0],
                power: [1, 1, 1, 0],
                saturation: 1,
                vignette: 0,
                vignetteInner: 0,
                vignetteOuter: 1,
                posterize: 0,
                dither: 0,
            }),
        },
    },
    systems: [GlazeSystem],
    dependencies: [RenderPlugin],

    async warm() {
        const { device } = Compute;
        if (!device) return;
        const format = navigator.gpu.getPreferredCanvasFormat();
        _glaze.buffer?.destroy();
        _glaze.buffer = device.createBuffer({
            label: "glaze-config",
            size: GLAZE_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        _glaze.layout = device.createBindGroupLayout({
            label: "glaze",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    texture: { sampleType: "float" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                        hasDynamicOffset: true,
                        minBindingSize: GLAZE_STRUCT_BYTES,
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: { access: "write-only", format },
                },
            ],
        });
        const module = device.createShaderModule({ label: "glaze", code: code(format) });
        _glaze.pipeline = await device.createComputePipelineAsync({
            label: "glaze",
            layout: device.createPipelineLayout({ bindGroupLayouts: [_glaze.layout] }),
            compute: { module, entryPoint: "main" },
        });
    },

    dispose() {
        _glaze.buffer?.destroy();
        _glaze.buffer = null;
        _glaze.pipeline = null;
        _glaze.layout = null;
    },
};
