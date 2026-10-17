/// <reference types="@webgpu/types" />

import * as d from "typegpu/data";
import type { Plugin, System, World } from "../../engine";
import { ClearChangeMarksSystem, invertMat4, unpackColor } from "../../engine";
import {
    beginComputePass,
    dispatchWorkgroups,
    endComputePass,
    setComputeBindGroup,
    setComputePipeline,
} from "../../engine/runtime";
import {
    composeGlobalTransform,
    PrepareGlobalTransformSystem,
    TransformPlugin,
} from "../transform";
import { Anchor } from "./anchor";

import {
    Camera,
    CameraMode,
    computeViewProj,
    EXPOSURE_EV100_BLENDER,
    Exposure,
    Resolution,
} from "./camera";
import { FRAME_UNIFORM_SIZE, Frame, initializeFrameState, writeFrame } from "./frame";
import {
    EndFrameSystem,
    initializeRenderFrameState,
    renderFrameKey,
    VIEW_KEY_FLOATS,
} from "./frame-state";
import { CULL_VOLUME_FLOATS, frustumVolume } from "./frustum";
import {
    GlobalTransformHistory,
    GlobalTransformHistoryEndSystem,
    GlobalTransformHistoryStartSystem,
    globalTransformTable,
    PrepareGlobalTransformHistorySystem,
    prepareGlobalTransformFrame,
    recoverGlobalTransformHistory,
} from "./global-transform";
import { initializeImageState, warmImages } from "./image";
import {
    AmbientLight,
    DirectionalLight,
    GlobalAmbientLight,
    NotShadowCaster,
    PointLight,
    SpotLight,
    SunDisk,
    VolumetricLight,
} from "./lighting";
import { initializeRenderState, RenderContext } from "./render";
import { ShadowFilteringMethod, ShadowFilteringMethodMode } from "./shadow-filtering";

import {
    bindCamera,
    clearOffscreens,
    clearScratch,
    INITIAL_VIEW_CAPACITY,
    initializeViewState,
    MAX_SLOTS,
    offscreen,
    presentingViewRefusal,
    presentingViewSlot,
    pruneViews,
    sizeView,
    VIEW_BYTES,
    VIEW_STRIDE,
    type View as ViewSlot,
    Views,
    ViewUniforms,
} from "./view";

/** Fixed read-only cull table, shared by camera and shadow slots. */
export const CullVolumes = d.arrayOf(d.vec4f, MAX_SLOTS * (CULL_VOLUME_FLOATS / 4));
const AMBIENT_COLOR_F32 = d.memoryLayoutOf(ViewUniforms, (view) => view.ambientColor).offset / 4;
const EXPOSURE_F32 = d.memoryLayoutOf(ViewUniforms, (view) => view.exposure).offset / 4;
const SHADOW_FILTERING_U32 =
    d.memoryLayoutOf(ViewUniforms, (view) => view.shadowFilteringMethod).offset / 4;

export { Camera, CameraMode, Exposure, Resolution } from "./camera";
export { CAPTURE_CONTRACT, type Capture, captureFrame, captureTexture } from "./capture";
export {
    AmbientLight,
    DirectionalLight,
    GlobalAmbientLight,
    NotShadowCaster,
    PointLight,
    SpotLight,
    SUN_DISK_EARTH_ANGULAR_SIZE,
    SunDisk,
    VolumetricLight,
} from "./lighting";
export { ShadowFilteringMethod, ShadowFilteringMethodMode } from "./shadow-filtering";
export {
    DEPTH_FORMAT,
    DepthPrepass,
    type DepthPrepassRequest,
    DepthPrepassRequests,
    SAMPLE_COUNT,
} from "./targets";

const SLOT_FLOATS = VIEW_STRIDE / 4;
const CAMERAS = [Camera];
const GLOBAL_TRANSFORM_PASS: GPUComputePassDescriptor = {};
function basisColumn(world: Float32Array, base: number, out: Float32Array, at: number): void {
    const x = world[base];
    const y = world[base + 1];
    const z = world[base + 2];
    const inv = 1 / (Math.sqrt(x * x + y * y + z * z) || 1);
    out[at] = x * inv;
    out[at + 1] = y * inv;
    out[at + 2] = z * inv;
    out[at + 3] = 0;
}

// Shared per-slot camera state. Presenting views additionally carry projection parameters
// and the inverse view-projection matrix; depth-only views use the cheaper MAX_SLOTS budget.
function packView(world: World, eid: number, view: ViewSlot, shading: boolean, slot: number): void {
    const _renderFrame = world.resource(renderFrameKey);
    const _render = world.resource(RenderContext);

    view.slot = slot;
    const camera = world.storage(Camera);
    const isPerspective = camera.mode.get(eid) !== CameraMode.Orthographic;
    // the camera basis (floats 20-27) and the eye (32-35) come from the world matrix, which is also what
    // the viewProj is composed from, so it is read before the unchanged-slot test below
    composeGlobalTransform(world, eid, _renderFrame.camWorld);
    if (!slotInputsChanged(world, eid, view, shading, slot)) return;
    const offset = slot * SLOT_FLOATS;
    const viewProj = _renderFrame.viewProjs[slot];
    const ambientPacked = world.has(eid, AmbientLight)
        ? world.storage(AmbientLight).color.get(eid)
        : world.resource(GlobalAmbientLight).color;
    const ambientBrightness = world.has(eid, AmbientLight)
        ? world.storage(AmbientLight).brightness.get(eid)
        : world.resource(GlobalAmbientLight).brightness;
    const ambientRgb = unpackColor(ambientPacked);
    const ambientAt = offset + AMBIENT_COLOR_F32;
    _render.viewStaging[ambientAt] = ambientRgb.r * ambientBrightness;
    _render.viewStaging[ambientAt + 1] = ambientRgb.g * ambientBrightness;
    _render.viewStaging[ambientAt + 2] = ambientRgb.b * ambientBrightness;
    _render.viewStaging[ambientAt + 3] = 0;
    const ev100 = world.has(eid, Exposure)
        ? world.storage(Exposure).ev100.get(eid)
        : EXPOSURE_EV100_BLENDER;
    _render.viewStaging[offset + EXPOSURE_F32] = 2 ** -ev100 / 1.2;
    new Uint32Array(_render.viewStaging.buffer)[offset + SHADOW_FILTERING_U32] = world.has(
        eid,
        ShadowFilteringMethod,
    )
        ? world.storage(ShadowFilteringMethod).method.get(eid)
        : ShadowFilteringMethodMode.Gaussian;
    computeViewProj(world, eid, view.width / view.height, viewProj, view.viewFromWorld);
    // resolution (pixels) follows viewProj in the ViewUniforms struct — a screen-space
    // producer (lines) reads it to size constant-pixel-width geometry
    _render.viewStaging[offset + 16] = view.width;
    _render.viewStaging[offset + 17] = view.height;
    // camera basis (right at floats 20-23, up at 24-27; 18-19 pad before the vec4) —
    // billboard surfaces orient quads from it (in a shadow pass, the light camera's, so
    // billboards face the light). Normalized: the camera Transform may scale
    basisColumn(_renderFrame.camWorld, 0, _render.viewStaging, offset + 20);
    basisColumn(_renderFrame.camWorld, 4, _render.viewStaging, offset + 24);
    frustumVolume(
        _render.cullVolumeStaging,
        slot,
        viewProj,
        isPerspective ? _renderFrame.camWorld : undefined,
        isPerspective ? camera.far.get(eid) : undefined,
    );
    _render.cullVolumeStaging[slot * CULL_VOLUME_FLOATS + 1] = shading ? 0 : 1;
    // ViewUniforms.projection packs near, far, perspective flag and cull slot in xyzw.
    if (shading) {
        _render.viewStaging[offset + 28] = camera.near.get(eid);
        _render.viewStaging[offset + 29] = camera.far.get(eid);
        _render.viewStaging[offset + 30] = isPerspective ? 1 : 0;
    } else {
        _render.viewStaging[offset + 28] = 0;
        _render.viewStaging[offset + 29] = 0;
        _render.viewStaging[offset + 30] = 0;
    }
    _render.viewStaging[offset + 31] = slot;
    // Specular shading needs the camera's world-space eye position.
    _render.viewStaging[offset + 32] = _renderFrame.camWorld[12];
    _render.viewStaging[offset + 33] = _renderFrame.camWorld[13];
    _render.viewStaging[offset + 34] = _renderFrame.camWorld[14];
    _render.viewStaging[offset + 35] = 1;
    // Only shading views reconstruct screen-space positions; depth-only views need no inverse.
    if (shading) invertMat4(viewProj, _renderFrame.invViewProjs[slot]);
    else _render.viewStaging.fill(0, offset + 36, offset + 52);
}

// whether this slot's pack inputs differ from the ones it was last packed with; records them when they do.
// `_frame.camWorld` holds the camera's world matrix, composed by the caller.
function slotInputsChanged(
    world: World,
    eid: number,
    view: ViewSlot,
    shading: boolean,
    slot: number,
): boolean {
    const _renderFrame = world.resource(renderFrameKey);

    _renderFrame.viewKeyNext[0] = eid;
    _renderFrame.viewKeyNext[1] = shading ? 1 : 0;
    _renderFrame.viewKeyNext[2] = view.width;
    _renderFrame.viewKeyNext[3] = view.height;
    _renderFrame.viewKeyNext[4] = world.storage(Camera).mode.get(eid);
    _renderFrame.viewKeyNext[5] = world.storage(Camera).fov.get(eid);
    _renderFrame.viewKeyNext[6] = world.storage(Camera).size.get(eid);
    _renderFrame.viewKeyNext[7] = world.storage(Camera).near.get(eid);
    _renderFrame.viewKeyNext[8] = world.storage(Camera).far.get(eid);
    _renderFrame.viewKeyNext.set(_renderFrame.camWorld, 9);
    const ev100 = world.has(eid, Exposure)
        ? world.storage(Exposure).ev100.get(eid)
        : EXPOSURE_EV100_BLENDER;
    const ambientColor = world.has(eid, AmbientLight)
        ? world.storage(AmbientLight).color.get(eid)
        : world.resource(GlobalAmbientLight).color;
    const ambientBrightness = world.has(eid, AmbientLight)
        ? world.storage(AmbientLight).brightness.get(eid)
        : world.resource(GlobalAmbientLight).brightness;
    _renderFrame.viewKeyNext[25] = ev100;
    _renderFrame.viewKeyNext[26] = ambientColor;
    _renderFrame.viewKeyNext[27] = ambientBrightness;
    _renderFrame.viewKeyNext[28] = world.has(eid, ShadowFilteringMethod)
        ? world.storage(ShadowFilteringMethod).method.get(eid)
        : ShadowFilteringMethodMode.Gaussian;
    const at = slot * VIEW_KEY_FLOATS;
    let changed = _renderFrame.slotViews[slot] !== view;
    for (let i = 0; i < VIEW_KEY_FLOATS; i++) {
        if (_renderFrame.viewKeys[at + i] !== _renderFrame.viewKeyNext[i]) {
            changed = true;
            break;
        }
    }
    if (changed) {
        _renderFrame.viewKeys.set(_renderFrame.viewKeyNext, at);
        _renderFrame.slotViews[slot] = view;
    }
    return changed;
}

function ensureRenderSlotCapacity(world: World, requiredSlots: number, shadeCount: number): void {
    const render = world.resource(RenderContext);
    const frame = world.resource(renderFrameKey);
    const capacity = render.viewStaging.byteLength / (SLOT_FLOATS * Float32Array.BYTES_PER_ELEMENT);
    if (requiredSlots > capacity) {
        const nextCapacity = requiredSlots;
        const staging = new Float32Array(nextCapacity * SLOT_FLOATS);
        staging.set(render.viewStaging);
        render.viewStaging = staging;
        const keys = new Float64Array(nextCapacity * VIEW_KEY_FLOATS).fill(Number.NaN);
        keys.set(frame.viewKeys);
        frame.viewKeys = keys;
        frame.viewProjs = Array.from({ length: nextCapacity }, (_, slot) =>
            staging.subarray(slot * SLOT_FLOATS, slot * SLOT_FLOATS + 16),
        );
        frame.invViewProjs = Array.from({ length: nextCapacity }, (_, slot) =>
            staging.subarray(slot * SLOT_FLOATS + 36, slot * SLOT_FLOATS + 52),
        );
    }
    while (render.viewBuffers.length < shadeCount) {
        const slot = render.viewBuffers.length;
        render.viewBuffers.push(
            world.gpu.device.createBuffer({
                label: `shallot-view-${slot}`,
                size: VIEW_BYTES,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }),
        );
    }
}

/**
 * prepares per-frame render state: writes the Frame UBO, records history work
 * on the World's lazy frame encoder, acquires each view's swapchain backbuffer
 * (`view.present`) + offscreen scene-color target (`view.framebuffer`), and
 * packs the ViewUniforms UBO. Producer and renderer systems run after this anchor;
 * the engine submits any recorded encoder after the draw group.
 */
export const BeginFrameSystem: System = {
    group: "draw",
    first: true,
    after: [PrepareGlobalTransformSystem],
    before: [ClearChangeMarksSystem],
    update(world) {
        const _render = world.resource(RenderContext);
        const _renderFrame = world.resource(renderFrameKey);

        const device = world.gpu.device;
        if (!device) return;

        // Prune before binding so recycled camera eids cannot inherit an old surface.
        pruneViews(world);

        prepareGlobalTransformFrame(world);
        writeFrame(world);

        let shadeCount = 0;
        let depthOnly = 0;
        const presentingEids = _renderFrame.presentingEids;
        const presentingViews = _renderFrame.presentingViews;
        // Presenting views own the low uniform slots; depth-only views stack above them in the
        // fixed cull-table tail. These scratch arrays are retained by renderFrameKey across frames.
        for (const eid of world.query(CAMERAS)) {
            const view = bindCamera(eid, world);
            if (!view) continue;
            view.framebuffer = null;
            view.framebufferFormat = undefined;
            view.present = null;
            view.canvasTexture = undefined;
            sizeView(world, eid, view);
            if (view.width === 0 || view.height === 0) continue;
            if (!view.context && !view.texture) {
                view.framebuffer = null;
                view.present = null;
                _renderFrame.depthOnlyEids[depthOnly] = eid;
                _renderFrame.depthOnlyViews[depthOnly] = view;
                depthOnly++;
                continue;
            }
            const slot = presentingViewSlot(shadeCount);
            if (slot === undefined) {
                console.warn(presentingViewRefusal(eid));
                continue;
            }
            const texture = view.texture ?? view.context!.getCurrentTexture();
            if (view.context) view.canvasTexture = texture;
            if (!texture) continue;
            if (view.surfaceTexture !== texture) {
                view.surfaceTexture = texture;
                view.surfaceView = texture.createView();
            }
            view.present = view.surfaceView!;
            view.framebuffer = offscreen(world, eid, view.width, view.height);
            view.framebufferFormat = _render.format;
            presentingEids[slot] = eid;
            presentingViews[slot] = view;
            shadeCount = slot + 1;
        }

        _renderFrame.presentingEids.length = shadeCount;
        _renderFrame.presentingViews.length = shadeCount;
        _render.shadeCount = shadeCount;
        const depthCount = Math.min(depthOnly, MAX_SLOTS - shadeCount);
        ensureRenderSlotCapacity(world, shadeCount + depthCount, shadeCount);
        for (let slot = 0; slot < shadeCount; slot++)
            packView(world, presentingEids[slot]!, presentingViews[slot]!, true, slot);
        for (let i = 0; i < depthCount; i++)
            packView(
                world,
                _renderFrame.depthOnlyEids[i]!,
                _renderFrame.depthOnlyViews[i]!,
                false,
                shadeCount + i,
            );
        if (depthOnly > depthCount) {
            console.warn(
                `shallot: ${MAX_SLOTS} cull-view slots exhausted; ${depthOnly - depthCount} depth-only view(s) skipped`,
            );
        }
        const count = shadeCount + depthCount;
        _renderFrame.slotViews.length = count;
        _render.viewCount = count;
        // Depth-only slots bind no ViewUniforms buffer; atlas pipelines use an unread slot-zero placeholder.
        const viewFloats = VIEW_BYTES / 4;
        for (let slot = 0; slot < _render.shadeCount; slot++) {
            device.queue.writeBuffer(
                _render.viewBuffers[slot],
                0,
                _render.viewStaging as Float32Array<ArrayBuffer>,
                slot * SLOT_FLOATS,
                viewFloats,
            );
        }
        if (count > 0) {
            device.queue.writeBuffer(
                _render.cullVolumes,
                0,
                _render.cullVolumeStaging as Float32Array<ArrayBuffer>,
                0,
                count * CULL_VOLUME_FLOATS,
            );
        }

        // Every renderer reads interpolated GlobalTransforms, independently of clustered lighting.
        const globalTransformRuntime = world.resource(GlobalTransformHistory);
        const globalTransformCount =
            _render.viewCount > 0 && globalTransformRuntime?.enabled
                ? (globalTransformRuntime.current?.count ?? 0)
                : 0;
        if (globalTransformRuntime && globalTransformCount > 0) {
            const encoder = world.frameEncoder()!;
            const pass = beginComputePass(encoder, GLOBAL_TRANSFORM_PASS);
            setComputePipeline(pass, globalTransformRuntime.pipeline!);
            setComputeBindGroup(pass, 0, globalTransformRuntime.group!);
            dispatchWorkgroups(pass, Math.ceil(globalTransformCount / 64));
            endComputePass(pass);
        }
    },
};

/** Draw ordering anchor: scene effects run before it, overlays after it and before PresentationSystem. */
export const OverlaySystem: System = {
    name: "overlay",
    group: "draw",
};

/**
 * Closes the overlay seam: scene color is complete before OverlaySystem, overlays run between
 * the two anchors, and presentation runs after this anchor. No update work is performed.
 */
export const PresentationSystem: System = {
    name: "presentation",
    group: "draw",
    after: [OverlaySystem],
};

async function initRender(world: World): Promise<void> {
    const _render = world.resource(RenderContext);
    const _renderFrame = world.resource(renderFrameKey);

    if (!world.gpu.device) return;
    const { device } = world.gpu;

    // Half-float radiance preserves smooth gradients through tonemapping; see Bevy v0.20.0
    // (b3bdd75b), ViewTarget::main_texture_format and view/composition.rs. The swapchain
    // keeps the base canvas format and presentation encodes linear to sRGB.
    _render.format = "rgba16float";

    const uniform = (label: string, size: number) =>
        device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

    for (const b of _render.viewBuffers) b.destroy();
    _render.viewBuffers = Array.from({ length: INITIAL_VIEW_CAPACITY }, (_, slot) =>
        uniform(`shallot-view-${slot}`, VIEW_BYTES),
    );
    _render.viewStaging = new Float32Array(INITIAL_VIEW_CAPACITY * SLOT_FLOATS);
    // Fresh staging: each active slot repacks against its own buffer on its first frame.
    _renderFrame.viewKeys.fill(Number.NaN);
    const staging = _render.viewStaging;
    _renderFrame.viewProjs = Array.from({ length: INITIAL_VIEW_CAPACITY }, (_, slot) =>
        staging.subarray(slot * SLOT_FLOATS, slot * SLOT_FLOATS + 16),
    );
    _renderFrame.invViewProjs = Array.from({ length: INITIAL_VIEW_CAPACITY }, (_, slot) =>
        staging.subarray(slot * SLOT_FLOATS + 36, slot * SLOT_FLOATS + 52),
    );
    world.resource(Frame).buffer = uniform("shallot-frame", FRAME_UNIFORM_SIZE);

    // one tagged cull volume per view, packed for the GPU cull pass and published
    // by name so any producer's cull resolves it the same way it resolves slabs
    _render.cullVolumes = device.createBuffer({
        label: "shallot-cull-volumes",
        size: MAX_SLOTS * CULL_VOLUME_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _render.cullVolumeStaging = new Float32Array(MAX_SLOTS * CULL_VOLUME_FLOATS);
    _render.viewCount = 0;
    _render.shadeCount = 0;
    world.resource(Views).clear();
    clearOffscreens(world);
    clearScratch(world);
}

/** Camera/view state and frame scheduling without a renderer; included by MeshRenderPlugin. */
export const RenderingPlugin: Plugin = {
    gpu: {},
    dependencies: [TransformPlugin],
    name: "Rendering",
    systems: [
        GlobalTransformHistoryStartSystem,
        GlobalTransformHistoryEndSystem,
        PrepareGlobalTransformHistorySystem,
        BeginFrameSystem,
        OverlaySystem,
        PresentationSystem,
        EndFrameSystem,
    ],
    recovery: recoverGlobalTransformHistory,
    components: [
        Anchor,
        Camera,
        Exposure,
        Resolution,
        AmbientLight,
        DirectionalLight,
        PointLight,
        SpotLight,
        SunDisk,
        VolumetricLight,
        NotShadowCaster,
        ShadowFilteringMethod,
    ],

    warm: warmImages,

    async initialize(world) {
        initializeRenderState(world);
        initializeViewState(world);
        initializeFrameState(world);
        initializeImageState(world);
        initializeRenderFrameState(world);
        world.resource(GlobalAmbientLight);
        await initRender(world);
        const globalTransformRuntime = world.resource(GlobalTransformHistory);
        // Its uniform binding reuses the leading vec4 in the Frame buffer written each frame.
        globalTransformRuntime.params = world.resource(Frame).buffer;
        globalTransformTable(world);
    },
};

// Extension consumers read the packed cull table; they must not repack it independently.

export { computeViewFromWorld, computeViewProj } from "./camera";
export { Frame, FrameGpu } from "./frame";
export { CULL_FRUSTUM, CULL_VOLUME_FLOATS } from "./frustum";
export { Images, imageArray, registerImage } from "./image";

export { RenderContext } from "./render";
export {
    attachCanvas,
    attachTexture,
    attachView,
    detachView,
    INITIAL_VIEW_CAPACITY,
    linearToSrgb3,
    MAX_SLOTS,
    MAX_VIEWS,
    sceneTransform,
    sizeView,
    type View,
    Views,
    ViewUniforms,
} from "./view";
