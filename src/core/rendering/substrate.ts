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
    CameraViewport,
    computeClipFromWorld,
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
    RenderLayers,
    renderLayers,
    SpotLight,
    SunDisk,
    VolumetricLight,
} from "./lighting";
import { initializeRenderState, RenderContext } from "./render";
import { ShadowFilteringMethod, ShadowFilteringMethodMode } from "./shadow-filtering";
import { SAMPLE_COUNT } from "./targets";
import { warmUpscaling } from "./upscaling";

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
    releaseStacked,
    sizeCameraViewport,
    sizeView,
    stacked,
    VIEW_BYTES,
    type View as ViewSlot,
    Views,
    ViewUniforms,
} from "./view";

/** @internal This frame's presenting views indexed by shading slot, filled by BeginFrameSystem;
 * the array is retained and refilled every frame. */
export function presentingViews(world: World): readonly ViewSlot[] {
    return world.resource(renderFrameKey).presentingViews;
}

/** Fixed read-only cull table, shared by camera and shadow slots. */
export const CullVolumes = d.arrayOf(d.vec4f, MAX_SLOTS * (CULL_VOLUME_FLOATS / 4));
const AMBIENT_COLOR_F32 = d.memoryLayoutOf(ViewUniforms, (view) => view.ambientColor).offset / 4;
const EXPOSURE_F32 = d.memoryLayoutOf(ViewUniforms, (view) => view.exposure).offset / 4;
const SHADOW_FILTERING_U32 =
    d.memoryLayoutOf(ViewUniforms, (view) => view.shadowFilteringMethod).offset / 4;
const CLIP_FROM_WORLD_F32 = d.memoryLayoutOf(ViewUniforms, (v) => v.clipFromWorld).offset / 4;
const WORLD_FROM_CLIP_F32 = d.memoryLayoutOf(ViewUniforms, (v) => v.worldFromClip).offset / 4;
const VIEW_FROM_WORLD_F32 = d.memoryLayoutOf(ViewUniforms, (v) => v.viewFromWorld).offset / 4;
const CLIP_FROM_VIEW_F32 = d.memoryLayoutOf(ViewUniforms, (v) => v.clipFromView).offset / 4;

// One ViewUniforms matrix of every slot, as views into `staging`.
function slotMatrices(staging: Float32Array, capacity: number, at: number): Float32Array[] {
    return Array.from({ length: capacity }, (_, slot) =>
        staging.subarray(slot * SLOT_FLOATS + at, slot * SLOT_FLOATS + at + 16),
    );
}

function bindSlotMatrices(world: World, capacity: number): void {
    const frame = world.resource(renderFrameKey);
    const staging = world.resource(RenderContext).viewStaging;
    frame.clipFromWorlds = slotMatrices(staging, capacity, CLIP_FROM_WORLD_F32);
    frame.worldFromClips = slotMatrices(staging, capacity, WORLD_FROM_CLIP_F32);
    frame.viewFromWorlds = slotMatrices(staging, capacity, VIEW_FROM_WORLD_F32);
    frame.clipFromViews = slotMatrices(staging, capacity, CLIP_FROM_VIEW_F32);
}

export { Camera, CameraMode, CameraViewport, Exposure, Resolution } from "./camera";
export { CAPTURE_CONTRACT, type Capture, captureFrame, captureTexture } from "./capture";
export {
    AmbientLight,
    DirectionalLight,
    GlobalAmbientLight,
    NotShadowCaster,
    PointLight,
    RenderLayers,
    renderLayers,
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
    NORMAL_FORMAT,
    NormalPrepass,
    NormalPrepassRequests,
    SAMPLE_COUNT,
} from "./targets";

const SLOT_FLOATS = VIEW_BYTES / 4;
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
    // clipFromWorld is composed from, so it is read before the unchanged-slot test below
    composeGlobalTransform(world, eid, _renderFrame.camWorld);
    const mask = renderLayers(world, eid);
    _render.cullVolumeWords[slot * CULL_VOLUME_FLOATS + 2] = mask;
    if (!slotInputsChanged(world, eid, view, shading, slot)) return;
    const offset = slot * SLOT_FLOATS;
    const clipFromWorld = _renderFrame.clipFromWorlds[slot];
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
    const viewFromWorld = _renderFrame.viewFromWorlds[slot]!;
    computeClipFromWorld(
        world,
        eid,
        view.projectionAspect ?? view.width / view.height,
        clipFromWorld,
        viewFromWorld,
        _renderFrame.clipFromViews[slot],
    );
    view.viewFromWorld?.set(viewFromWorld);
    // resolution (pixels) follows clipFromWorld in the ViewUniforms struct — a screen-space
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
        clipFromWorld,
        isPerspective ? _renderFrame.camWorld : undefined,
        isPerspective ? camera.far.get(eid) : undefined,
    );
    _render.cullVolumeStaging[slot * CULL_VOLUME_FLOATS + 1] = shading ? 0 : 1;
    _render.cullVolumeWords[slot * CULL_VOLUME_FLOATS + 2] = mask;
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
    if (shading) invertMat4(clipFromWorld, _renderFrame.worldFromClips[slot]);
    else _renderFrame.worldFromClips[slot]!.fill(0);
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
        bindSlotMatrices(world, nextCapacity);
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

function targetBelow(
    world: World,
    a: number,
    viewA: ViewSlot,
    b: number,
    viewB: ViewSlot,
): boolean {
    const targetA = viewA.target ?? 0;
    const targetB = viewB.target ?? 0;
    if (targetA !== targetB) return targetA < targetB;
    const order = world.storage(Camera).order;
    const orderA = order.get(a);
    const orderB = order.get(b);
    return orderA !== orderB ? orderA < orderB : a < b;
}

// Sorts canvas and texture views by (target, order, eid) and indexes each target's cameras bottom to
// top, as Bevy's sort_cameras does by (order, target). Insertion sort: the order is stable across frames.
function sortTargets(world: World, eids: number[], views: ViewSlot[]): void {
    for (let i = 1; i < eids.length; i++) {
        const eid = eids[i]!;
        const view = views[i]!;
        let j = i - 1;
        while (j >= 0 && targetBelow(world, eid, view, eids[j]!, views[j]!)) {
            eids[j + 1] = eids[j]!;
            views[j + 1] = views[j]!;
            j--;
        }
        eids[j + 1] = eid;
        views[j + 1] = view;
    }
    const order = world.storage(Camera).order;
    for (let i = 0; i < eids.length; i++) {
        const view = views[i]!;
        const below = i > 0 ? views[i - 1] : undefined;
        // A view without a target shares nothing, so it is the bottom of its own.
        if (!view.target || !below || below.target !== view.target) {
            view.sortedIndexForTarget = 0;
            continue;
        }
        view.sortedIndexForTarget = (below.sortedIndexForTarget ?? 0) + 1;
        if (!view.orderTieWarned && order.get(eids[i - 1]!) === order.get(eids[i]!)) {
            view.orderTieWarned = true;
            console.warn(
                `shallot: cameras ${eids[i - 1]} and ${eids[i]} share Camera.order ${order.get(eids[i]!)} on one target; ${eids[i]} draws on top. Give each camera on a target its own order.`,
            );
        }
    }
}

/** Sample counts to prepare for the current cameras, including views that auto-bind a canvas. */
export function initialViewSampleCounts(world: World): Set<number> {
    const counts = new Set<number>();
    for (const eid of world.query([Camera]))
        counts.add(world.storage(Camera).antialias.get(eid) ? SAMPLE_COUNT : 1);
    return counts;
}

/** Initial explicitly attached camera stacks, sorted by the same target order as draw. @internal */
export function initialStackedViews(world: World): ViewSlot[] {
    const eids: number[] = [];
    const views: ViewSlot[] = [];
    for (const [eid, view] of world.resource(Views)) {
        if (!world.has(eid, Camera) || !view.target) continue;
        eids.push(eid);
        views.push(view);
    }
    sortTargets(world, eids, views);
    return views.filter((view) => view.sortedIndexForTarget);
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
        let targetCount = 0;
        const presentingEids = _renderFrame.presentingEids;
        const presentingViews = _renderFrame.presentingViews;
        const targetEids = _renderFrame.targetEids;
        const targetViews = _renderFrame.targetViews;
        // Presenting views own the low uniform slots; depth-only views stack above them in the
        // fixed cull-table tail. These scratch arrays are retained by renderFrameKey across frames.
        for (const eid of world.query(CAMERAS)) {
            const view = bindCamera(eid, world);
            if (!view) continue;
            view.framebuffer = null;
            view.framebufferFormat = undefined;
            view.present = null;
            view.canvasTexture = undefined;
            if (view.context || view.texture) {
                targetEids[targetCount] = eid;
                targetViews[targetCount] = view;
                targetCount++;
                continue;
            }
            sizeView(world, eid, view);
            if (view.width === 0 || view.height === 0) continue;
            _renderFrame.depthOnlyEids[depthOnly] = eid;
            _renderFrame.depthOnlyViews[depthOnly] = view;
            depthOnly++;
        }
        targetEids.length = targetCount;
        targetViews.length = targetCount;
        sortTargets(world, targetEids, targetViews);
        // Bottom to top, so a camera above the bottom of a canvas sizes after the canvas does.
        for (let i = 0; i < targetCount; i++) {
            const eid = targetEids[i]!;
            const view = targetViews[i]!;
            sizeView(world, eid, view);
            sizeCameraViewport(world, eid, view);
            if (
                view.width === 0 ||
                view.height === 0 ||
                view.viewport?.width === 0 ||
                view.viewport?.height === 0
            )
                continue;
            const slot = presentingViewSlot(shadeCount);
            if (slot === undefined) {
                console.warn(presentingViewRefusal(eid));
                continue;
            }
            const texture = view.texture ?? view.context!.getCurrentTexture();
            if (view.context) {
                view.canvasTexture = texture;
                view.canvasTextureReused = view.surfaceTexture === texture;
            }
            if (!texture) continue;
            if (view.surfaceTexture !== texture) {
                view.surfaceTexture = texture;
                view.surfaceView = texture.createView();
            }
            if (!view.sortedIndexForTarget && !view.viewport) {
                releaseStacked(world, eid);
                view.present = view.surfaceView!;
            } else view.present = stacked(world, eid, view.width, view.height);
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
        // Depth-only slots bind no ViewUniforms buffer; shadow atlases bind the camera the shadows
        // are fit to when it presents, otherwise slot 0.
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
    bindSlotMatrices(world, INITIAL_VIEW_CAPACITY);
    world.resource(Frame).buffer = uniform("shallot-frame", FRAME_UNIFORM_SIZE);

    // one tagged cull volume per view, packed for the GPU cull pass and published
    // by name so any producer's cull resolves it the same way it resolves slabs
    _render.cullVolumes = device.createBuffer({
        label: "shallot-cull-volumes",
        size: MAX_SLOTS * CULL_VOLUME_FLOATS * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _render.cullVolumeStaging = new Float32Array(MAX_SLOTS * CULL_VOLUME_FLOATS);
    _render.cullVolumeWords = new Uint32Array(_render.cullVolumeStaging.buffer);
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
        CameraViewport,
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
        RenderLayers,
        ShadowFilteringMethod,
    ],

    warm(world) {
        warmUpscaling(world);
        return warmImages(world);
    },

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

export { computeClipFromWorld, computeViewFromWorld } from "./camera";
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
    VIEW_BYTES,
    type View,
    Views,
    ViewUniforms,
} from "./view";
