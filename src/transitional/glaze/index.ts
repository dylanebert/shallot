// Destination: core/rendering; owner: presentation.md.
/// <reference types="@webgpu/types" />
// Glaze — the default postfx composite + the postfx chain. A renderer draws into each camera's offscreen
// scene-color target (`view.framebuffer`); glaze runs one compute dispatch per camera that reads it and
// writes the swapchain (`view.present`), applying the per-camera postfx chain on the way. The swapchain
// is a base-format storage texture (not sRGB), so glaze encodes linear→sRGB itself (`linearToSrgb`) — the
// same path a consumer's own fused composite takes. state.gpu, not a render pass, so the present costs no
// tile load/store on TBDR; WebGPU exposes no programmable blending, so a
// compute dispatch reading the offscreen and writing the swapchain once is the portable fused-postfx
// substitute. Renderer-agnostic: it imports only `render` and reads `view.framebuffer` / `view.present`,
// so sear (MSAA-resolved) and a custom renderer (single-sample) both composite through it. A renderer
// orders itself ahead with `before: [GlazeSystem]`; glaze never imports a renderer.
//
// The chain itself — the operators, the grade, the OkLab quantize, the vignette, the kernel — is
// `composite.ts` beside this file; here live the component, the system, the plugin, and the per-camera
// uniform write. A camera with no `Glaze` still tonemaps Neutral (the zero-config default display
// transform); the grade defaults to a no-op and posterize / dither / vignette gate off, so only the
// tonemap + linear→sRGB encode run. The rg11b10ufloat HDR offscreen is what lets the tonemap roll off
// highlights >1 (they'd clamp at store on an LDR offscreen).

import {
    BeginFrameSystem,
    Camera,
    MAX_VIEWS,
    Render,
    RenderPlugin,
    Views,
} from "../../core/rendering";
import type { Plugin, World, System } from "../../engine";
import { f32, u32, vec4 } from "../../engine";
import { precompile } from "../../engine/runtime";
import { composite, GlazeConfig, initializeCompositeState, WORKGROUP } from "./composite";

export { Tonemap, tonemapWgsl } from "./tonemap";

interface GlazeState {
    configs: ReturnType<typeof configBuffer>[];
    composite: ReturnType<typeof composite> | null;
    pass: GPUComputePassDescriptor;
    raw: {
        pipeline: GPUComputePipeline;
        layout: GPUBindGroupLayout;
        configs: GPUBuffer[];
        composite: ReturnType<typeof composite>;
    } | null;
    inputEntry: GPUBindGroupEntry;
    glazeBinding: GPUBufferBinding;
    glazeEntry: GPUBindGroupEntry;
    outputEntry: GPUBindGroupEntry;
    groupEntries: GPUBindGroupEntry[];
    groupDesc: GPUBindGroupDescriptor;
    labels: Map<number, string>;
}

const glazeStateKey = { create: createGlazeState };

function createGlazeState(): GlazeState {
    const inputEntry: GPUBindGroupEntry = { binding: 0, resource: null! };
    const glazeBinding: GPUBufferBinding = { buffer: null! };
    const glazeEntry: GPUBindGroupEntry = { binding: 1, resource: glazeBinding };
    const outputEntry: GPUBindGroupEntry = { binding: 2, resource: null! };
    const groupEntries = [inputEntry, glazeEntry, outputEntry];
    return {
        configs: [],
        composite: null,
        pass: { label: "" },
        raw: null,
        inputEntry,
        glazeBinding,
        glazeEntry,
        outputEntry,
        groupEntries,
        groupDesc: { label: "glaze", layout: null!, entries: groupEntries },
        labels: new Map(),
    };
}

function _glazeState(state: World): GlazeState {
    return state.resource(glazeStateKey);
}

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
    exposure: f32,
    tonemap: u32,
    slope: vec4,
    offset: vec4,
    power: vec4,
    saturation: f32,
    vignette: f32,
    vignetteInner: f32,
    vignetteOuter: f32,
    posterize: f32,
    dither: f32,
};

// One uniform buffer per view slot, not one strided buffer indexed by a dynamic offset: a typegpu bind
// group binds a whole buffer (no offset/size, no `hasDynamicOffset`). It keeps the property the stride
// existed for — `writeBuffer` is queue-ordered against the submit, so a single rewritten uniform would
// clobber every camera's composite with the last camera's config, while distinct buffers never collide
function configBuffer(state: World, slot: number) {
    return state.gpu.root.createBuffer(GlazeConfig).$usage("uniform").$name(`glaze-config-${slot}`);
}

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
// the camera query terms, the composite pass descriptor, and each camera's pass label, held so the
// per-frame composite mints only its bind group and WebGPU objects
const CAMERAS = [Camera];
// The raw group descriptors and handles live in the world's Glaze resource.

// the composite's raw handles, resolved once per build from the typegpu pipeline, layout and uniforms
function rawComposite(
    state: World,
    built: ReturnType<typeof composite>,
): {
    pipeline: GPUComputePipeline;
    layout: GPUBindGroupLayout;
    configs: GPUBuffer[];
} {
    const _glazeState = state.resource(glazeStateKey);

    if (_glazeState.raw && _glazeState.raw.composite === built) return _glazeState.raw;
    _glazeState.raw = {
        composite: built,
        pipeline: state.gpu.root.unwrap(built.pipeline),
        layout: state.gpu.root.unwrap(built.layout),
        configs: _glazeState.configs.map((buffer) => state.gpu.root.unwrap(buffer)),
    };
    return _glazeState.raw;
}
function uploadConfig(state: World, eid: number, slot: number): void {
    const buffer = state.resource(glazeStateKey).configs[slot];
    if (!state.has(eid, Glaze)) {
        buffer.write(DEFAULT);
        return;
    }
    buffer.write({
        exposure: state.of(Glaze).exposure.get(eid),
        vignetteStrength: state.of(Glaze).vignette.get(eid),
        vignetteInner: state.of(Glaze).vignetteInner.get(eid),
        vignetteOuter: state.of(Glaze).vignetteOuter.get(eid),
        posterizeBands: state.of(Glaze).posterize.get(eid),
        ditherStrength: state.of(Glaze).dither.get(eid),
        tonemapMode: state.of(Glaze).tonemap.get(eid),
        saturation: state.of(Glaze).saturation.get(eid),
        slope: [
            state.of(Glaze).slope.x.get(eid),
            state.of(Glaze).slope.y.get(eid),
            state.of(Glaze).slope.z.get(eid),
            0,
        ],
        offset: [
            state.of(Glaze).offset.x.get(eid),
            state.of(Glaze).offset.y.get(eid),
            state.of(Glaze).offset.z.get(eid),
            0,
        ],
        power: [
            state.of(Glaze).power.x.get(eid),
            state.of(Glaze).power.y.get(eid),
            state.of(Glaze).power.z.get(eid),
            0,
        ],
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
        const _glazeState = state.resource(glazeStateKey);

        const encoder = state.resource(Render).encoder;
        if (!encoder || !state.gpu.device || !_glazeState.composite) return;
        const raw = rawComposite(state, _glazeState.composite);
        _glazeState.groupDesc.layout = raw.layout;
        for (const eid of state.query(CAMERAS)) {
            const view = state.resource(Views).get(eid);
            if (!view?.present || !view.framebuffer) continue;
            uploadConfig(state, eid, view.slot);
            _glazeState.inputEntry.resource = view.framebuffer;
            _glazeState.glazeBinding.buffer = raw.configs[view.slot];
            _glazeState.outputEntry.resource = view.present;
            const group = state.gpu.device.createBindGroup(_glazeState.groupDesc);
            let label = _glazeState.labels.get(eid);
            if (label === undefined) {
                label = `glaze/${eid}`;
                _glazeState.labels.set(eid, label);
            }
            _glazeState.pass.label = label;
            _glazeState.pass.timestampWrites = state.gpu.span?.("glaze");
            const pass = encoder.beginComputePass(_glazeState.pass);
            pass.setPipeline(raw.pipeline);
            pass.setBindGroup(0, group);
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

    initialize(state) {
        state.resource(glazeStateKey);
        initializeCompositeState(state);
    },

    async warm(state: World) {
        const _glazeState = state.resource(glazeStateKey);

        const device = state.gpu.device;
        if (!device) return;
        const format = navigator.gpu.getPreferredCanvasFormat();
        for (const buffer of _glazeState.configs) buffer.destroy();
        _glazeState.configs = [];
        for (let slot = 0; slot < MAX_VIEWS; slot++)
            _glazeState.configs.push(configBuffer(state, slot));
        _glazeState.composite = composite(state, format);
        const { layout, pipeline } = _glazeState.composite;

        // typegpu pipelines are created synchronously and Dawn defers the real shader compile, so without
        // this the composite compiles inside frame 1 — glaze runs every frame, so that
        // is the whole first-frame stall. The real bind needs the per-frame swapchain view, which does not
        // exist at warm, so the forcer binds 1×1 throwaways of the same formats
        precompile(state, "glaze", () => {
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
                state.gpu.root.createBindGroup(layout, {
                    input: src.createView(),
                    glaze: state.resource(glazeStateKey).configs[0],
                    output: dst.createView(),
                }),
            );
            src.destroy();
            dst.destroy();
            return bound;
        });
    },

    dispose(state: World) {
        const _glazeState = state.resource(glazeStateKey);

        for (const buffer of _glazeState.configs) buffer.destroy();
        _glazeState.configs = [];
        // the pipeline memo is device-scoped and outlives a build (`composite`), like every other typed
        // pipeline cache — only this build's per-slot uniforms are ours to free
        _glazeState.composite = null;
        _glazeState.raw = null;
    },
};
