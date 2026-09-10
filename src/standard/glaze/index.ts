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
// The chain itself — the operators, the grade, the OkLab quantize, the vignette, the kernel — is
// `composite.ts` beside this file; here live the component, the system, the plugin, and the per-camera
// uniform write. A camera with no `Glaze` still tonemaps Neutral (the zero-config default display
// transform); the grade defaults to a no-op and posterize / dither / vignette gate off, so only the
// tonemap + linear→sRGB encode run. The rg11b10ufloat HDR offscreen is what lets the tonemap roll off
// highlights >1 (they'd clamp at store on an LDR offscreen).
import type { Plugin, State, System } from "../../engine";
import { Compute, f32, sparse, u32, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { Camera, RenderPlugin } from "../render";
import { BeginFrameSystem, MAX_VIEWS, Render, Views } from "../render/core";
import { composite, GlazeConfig, WORKGROUP } from "./composite";

export { Tonemap, tonemapWgsl } from "./tonemap";

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

// One uniform buffer per view slot, not one strided buffer indexed by a dynamic offset: a typegpu bind
// group binds a whole buffer (no offset/size, no `hasDynamicOffset`). It keeps the property the stride
// existed for — `writeBuffer` is queue-ordered against the submit, so a single rewritten uniform would
// clobber every camera's composite with the last camera's config, while distinct buffers never collide
function configBuffer(slot: number) {
    return Compute.root.createBuffer(GlazeConfig).$usage("uniform").$name(`glaze-config-${slot}`);
}

let _configs: ReturnType<typeof configBuffer>[] = [];
// the format-keyed composite, resolved once at warm: the swapchain format can't change without a new
// canvas configure, and a rebuild re-reads it
let _composite: ReturnType<typeof composite> | null = null;

// the grade identities, so a camera without `Glaze` composites a no-op grade at unit exposure and mode 0
// (Neutral). The vec4 `w` lanes are unread — only `.xyz` reaches the shader
const DEFAULT = {
    exposure: 1,
    vignetteStrength: 0,
    vignetteInner: 0,
    vignetteOuter: 0,
    posterizeBands: 0,
    ditherStrength: 0,
    tonemapMode: 0,
    saturation: 1,
    slope: [1, 1, 1, 0],
    offset: [0, 0, 0, 0],
    power: [1, 1, 1, 0],
} as const;

// write a camera's postfx config into its own slot buffer. Vignette / posterize / dither each gate on
// their own zero default in the kernel, so a camera without `Glaze` gets the default Neutral display
// transform and nothing else
// a shading view's slot is always < MAX_VIEWS (`render/view.ts` gates the assignment), so the slot buffer
// `warm` allocated always exists — no guard, since the bind group that follows would throw on a missing one
// anyway rather than skip the camera
function uploadConfig(state: State, eid: number, slot: number): void {
    const buffer = _configs[slot];
    if (!state.has(eid, Glaze)) {
        buffer.write(DEFAULT);
        return;
    }
    buffer.write({
        exposure: Glaze.exposure.get(eid),
        vignetteStrength: Glaze.vignette.get(eid),
        vignetteInner: Glaze.vignetteInner.get(eid),
        vignetteOuter: Glaze.vignetteOuter.get(eid),
        posterizeBands: Glaze.posterize.get(eid),
        ditherStrength: Glaze.dither.get(eid),
        tonemapMode: Glaze.tonemap.get(eid),
        saturation: Glaze.saturation.get(eid),
        slope: [Glaze.slope.x.get(eid), Glaze.slope.y.get(eid), Glaze.slope.z.get(eid), 0],
        offset: [Glaze.offset.x.get(eid), Glaze.offset.y.get(eid), Glaze.offset.z.get(eid), 0],
        power: [Glaze.power.x.get(eid), Glaze.power.y.get(eid), Glaze.power.z.get(eid), 0],
    });
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
    after: [BeginFrameSystem],
    update(state) {
        const encoder = Render.encoder;
        if (!encoder || !Compute.device || !_composite) return;
        const { layout, pipeline } = _composite;
        for (const eid of state.query([Camera])) {
            const view = Views.get(eid);
            if (!view?.present || !view.framebuffer) continue;
            uploadConfig(state, eid, view.slot);
            const group = Compute.root.createBindGroup(layout, {
                input: view.framebuffer,
                glaze: _configs[view.slot],
                output: view.present,
            });
            const pass = encoder.beginComputePass({
                label: `glaze/${eid}`,
                timestampWrites: Compute.span?.("glaze"),
            });
            pipeline
                .with(group)
                .with(pass)
                .dispatchWorkgroups(
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
        const device = Compute.device;
        if (!device) return;
        const format = navigator.gpu.getPreferredCanvasFormat();
        for (const buffer of _configs) buffer.destroy();
        _configs = [];
        for (let slot = 0; slot < MAX_VIEWS; slot++) _configs.push(configBuffer(slot));
        _composite = composite(format);
        const { layout, pipeline } = _composite;

        // typegpu pipelines are created synchronously and Dawn defers the real shader compile, so without
        // this the composite compiles inside frame 1 — glaze runs every frame, so that
        // is the whole first-frame stall. The real bind needs the per-frame swapchain view, which does not
        // exist at warm, so the forcer binds 1×1 throwaways of the same formats
        precompile("glaze", () => {
            const src = device.createTexture({
                label: "glaze-precompile-src",
                size: { width: 1, height: 1 },
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            const dst = device.createTexture({
                label: "glaze-precompile-dst",
                size: { width: 1, height: 1 },
                format,
                usage: GPUTextureUsage.STORAGE_BINDING,
            });
            const bound = pipeline.with(
                Compute.root.createBindGroup(layout, {
                    input: src.createView(),
                    glaze: _configs[0],
                    output: dst.createView(),
                }),
            );
            src.destroy();
            dst.destroy();
            return bound;
        });
    },

    dispose() {
        for (const buffer of _configs) buffer.destroy();
        _configs = [];
        // the pipeline memo is device-scoped and outlives a build (`composite`), like every other typed
        // pipeline cache — only this build's per-slot uniforms are ours to free
        _composite = null;
    },
};
