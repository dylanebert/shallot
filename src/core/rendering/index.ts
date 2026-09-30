/// <reference types="@webgpu/types" />

import * as d from "typegpu/data";
import type { Plugin, State, System } from "../../engine";
import {
    composeTransform,
    formatHex,
    GlobalTransform,
    globalTransformTable,
    invert,
} from "../../engine";

import { Camera, CameraMode, computeViewProj, Resolution } from "./camera";
import {
    ClusterSystem,
    initializeClusterState,
    LightCull,
    LightCullSystem,
    packClusterView,
    warmClusters,
    warmLightCull,
} from "./cluster";
import { initializeSurfaceState } from "./contract";
import { FRAME_UNIFORM_SIZE, Frame, initializeFrameState, writeFrame } from "./frame";
import { CULL_VOLUME_FLOATS, frustumVolume } from "./frustum";
import { initializeImageState } from "./image";
import {
    AmbientLight,
    DirectionalLight,
    initializeLightingState,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    PointLight,
    Spot,
    Volumetric,
    writeLighting,
} from "./lighting";
import { clearMeshes, flushMeshes, initializeMeshState } from "./mesh";
import { Draws, initializeDrawState, Surfaces } from "./registry";
import { initializeRenderState, Render } from "./render";
import {
    bindCamera,
    clearOffscreens,
    clearScratch,
    initializeViewState,
    MAX_SLOTS,
    MAX_VIEWS,
    offscreen,
    pruneViews,
    sizeView,
    VIEW_BYTES,
    VIEW_STRIDE,
    VIEW_UNIFORM_SIZE,
    type View as ViewSlot,
    Views,
} from "./view";

// the public happy path: the component contract (camera + lights) and meshes.
// Everything else a renderer or producer touches — the Render singleton, the
// View/Surface/Draw contract, canvas binding, the Lighting uniform, the frame
// loop — is the extension API, exported below. A producer (Part) and a renderer
// (Sear) meet only through that contract and neither imports the other, so a
// custom producer is a peer of Part rather than a fork of it.
export { Camera, CameraMode, Resolution } from "./camera";
export { CAPTURE_CONTRACT, type Capture, captureFrame } from "./capture";
export { requestLightOverflow } from "./cluster";
export { AmbientLight, DirectionalLight, PointLight, Spot, Volumetric } from "./lighting";
export type { Mesh } from "./mesh";
export { mesh } from "./mesh";

const SLOT_FLOATS = VIEW_STRIDE / 4;
const CAMERAS = [Camera];
const FRAME_ENCODER: GPUCommandEncoderDescriptor = { label: "shallot-frame" };
const GLOBAL_TRANSFORM_PASS: GPUComputePassDescriptor = {};
const VIEW_KEY_FLOATS = 26;

interface RenderFrameState {
    camWorld: Float32Array;
    submit: GPUCommandBuffer[];
    depthOnlyEids: number[];
    depthOnlyViews: ViewSlot[];
    viewProjs: Float32Array[];
    invViewProjs: Float32Array[];
    lightViews: Float32Array[];
    viewKeys: Float64Array;
    viewKeyNext: Float64Array;
}

const renderFrameKey = { create: createRenderFrameState };

function createRenderFrameState(): RenderFrameState {
    return {
        camWorld: new Float32Array(16),
        submit: [],
        depthOnlyEids: [],
        depthOnlyViews: [],
        viewProjs: [],
        invViewProjs: [],
        lightViews: [],
        viewKeys: new Float64Array(MAX_SLOTS * VIEW_KEY_FLOATS).fill(Number.NaN),
        viewKeyNext: new Float64Array(VIEW_KEY_FLOATS),
    };
}

function _renderFrameState(state: State): RenderFrameState {
    return state.resource(renderFrameKey);
}

function initializeRenderFrameState(state: State): void {
    state.resource(renderFrameKey);
}

// write a world-matrix column (base = column index * 4), normalized, into `out` at `at`
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

// viewProj + resolution + basis + frustum, the per-slot state every view carries. A shading
// view (presenting camera) additionally packs the clustered-light state — its cluster params
// and world→view matrix — into the same slot index; a depth-only view (a shadow light's
// off-screen camera) never does, so the cluster substrate is sized by MAX_VIEWS while the
// cheap slots run to MAX_SLOTS
function packView(state: State, eid: number, view: ViewSlot, shading: boolean, slot: number): void {
    const _renderFrame = state.resource(renderFrameKey);
    const _render = state.resource(Render);

    // record the live camera's create-stamp so next frame's pruneViews detects a realias
    view.stamp = state.stamp(eid);
    view.slot = slot;
    // the camera basis (floats 20-27) and the eye (32-35) come from the world matrix, which is also what
    // the viewProj is composed from, so it is read before the unchanged-slot test below
    composeTransform(state, eid, _renderFrame.camWorld);
    if (!slotInputsChanged(state, eid, view, shading, slot)) return;
    const offset = slot * SLOT_FLOATS;
    const viewProj = _renderFrame.viewProjs[slot];
    // the light cull reads each shading slot's world→view matrix to bring
    // world-space lights into cluster space
    computeViewProj(
        state,
        eid,
        view.width / view.height,
        viewProj,
        shading ? _renderFrame.lightViews[slot] : undefined,
    );
    // resolution (pixels) follows viewProj in the View struct — a screen-space
    // producer (lines) reads it to size constant-pixel-width geometry
    _render.viewStaging[offset + 16] = view.width;
    _render.viewStaging[offset + 17] = view.height;
    // camera basis (right at floats 20-23, up at 24-27; 18-19 pad before the vec4) —
    // billboard surfaces orient quads from it (in a shadow pass, the light camera's, so
    // billboards face the light). Normalized: the camera Transform may scale
    basisColumn(_renderFrame.camWorld, 0, _render.viewStaging, offset + 20);
    basisColumn(_renderFrame.camWorld, 4, _render.viewStaging, offset + 24);
    // pack this view's frustum cull volume — the pack tests each instance's bound against
    // cullVolumes[slot]'s 6 planes. Every view culls by frustum: cameras, the sun, and each
    // point/spot shadow combo (its own frustum-culled depth view)
    frustumVolume(_render.cullVolumeStaging, slot, viewProj);
    // pack the view's cluster params from the same camera fields —
    // ClusterSystem rebuilds the AABB grid only when they change.
    // View.cluster: (near, far, perspective, slot) — sear's FS maps a
    // fragment to its froxel and indexes the slot-major light grid
    if (shading) {
        const cv = packClusterView(state, eid, view.width / view.height, slot);
        _render.viewStaging[offset + 28] = cv.near;
        _render.viewStaging[offset + 29] = cv.far;
        _render.viewStaging[offset + 30] = cv.perspective ? 1 : 0;
    } else {
        _render.viewStaging[offset + 28] = 0;
        _render.viewStaging[offset + 29] = 0;
        _render.viewStaging[offset + 30] = 0;
    }
    _render.viewStaging[offset + 31] = slot;
    // eye (floats 32-35): the camera's world-space position — viewProj's translation column —
    // for view-dependent shading (specular V = normalize(eye - world))
    _render.viewStaging[offset + 32] = _renderFrame.camWorld[12];
    _render.viewStaging[offset + 33] = _renderFrame.camWorld[13];
    _render.viewStaging[offset + 34] = _renderFrame.camWorld[14];
    _render.viewStaging[offset + 35] = 1;
    // invViewProj (floats 36-51): a screen-space pass (fog) reconstructs world position from depth
    // via ndc → invViewProj. Only a shading view (a presenting camera) runs such a pass, so a
    // depth-only shadow view skips the 4×4 inverse — the costliest op in the pack — and zeroes the
    // slot. invert reads viewProj fully into locals before writing, so inverting into a sibling
    // view of the same staging never aliases
    if (shading) invert(viewProj, _renderFrame.invViewProjs[slot]);
    else _render.viewStaging.fill(0, offset + 36, offset + 52);
}

// whether this slot's pack inputs differ from the ones it was last packed with; records them when they do.
// `_frame.camWorld` holds the camera's world matrix, composed by the caller.
function slotInputsChanged(
    state: State,
    eid: number,
    view: ViewSlot,
    shading: boolean,
    slot: number,
): boolean {
    const _renderFrame = state.resource(renderFrameKey);

    _renderFrame.viewKeyNext[0] = eid;
    _renderFrame.viewKeyNext[1] = state.stamp(eid);
    _renderFrame.viewKeyNext[2] = shading ? 1 : 0;
    _renderFrame.viewKeyNext[3] = view.width;
    _renderFrame.viewKeyNext[4] = view.height;
    _renderFrame.viewKeyNext[5] = state.of(Camera).mode.get(eid);
    _renderFrame.viewKeyNext[6] = state.of(Camera).fov.get(eid);
    _renderFrame.viewKeyNext[7] = state.of(Camera).size.get(eid);
    _renderFrame.viewKeyNext[8] = state.of(Camera).near.get(eid);
    _renderFrame.viewKeyNext[9] = state.of(Camera).far.get(eid);
    _renderFrame.viewKeyNext.set(_renderFrame.camWorld, 10);
    const at = slot * VIEW_KEY_FLOATS;
    let changed = false;
    for (let i = 0; i < VIEW_KEY_FLOATS; i++) {
        if (_renderFrame.viewKeys[at + i] !== _renderFrame.viewKeyNext[i]) {
            changed = true;
            break;
        }
    }
    if (changed) _renderFrame.viewKeys.set(_renderFrame.viewKeyNext, at);
    return changed;
}

function clearTargets(view: ViewSlot): void {
    view.framebuffer = null;
    view.framebufferFormat = undefined;
    view.present = null;
}

/**
 * opens the frame: creates the encoder, writes the Frame UBO, records the
 * world-matrix compose dispatch, acquires each view's swapchain backbuffer
 * (`view.present`) + offscreen scene-color target (`view.framebuffer`), and
 * packs the View UBO. Producer and renderer systems both run
 * `after: [BeginFrameSystem]`; the terminal submission system closes the frame
 * after every producer and renderer in the draw group.
 */
export const BeginFrameSystem: System = {
    group: "draw",
    first: true,
    update(state) {
        const _render = state.resource(Render);
        const _renderFrame = state.resource(renderFrameKey);

        _render.encoder = null;
        const device = state.gpu.device;
        if (!device) return;

        // auto-bind's inverse. A destroyed camera leaves a stale View whose ResizeObserver leaks
        // and whose eid, once recycled, re-binds to the wrong canvas. Membership is the liveness
        // signal (re-derived each frame, the gate Part's pack also applies) and the create-stamp
        // catches a same-update realias membership misses, so a View lacking a live camera — or bound
        // to a recycled eid — is dropped here.
        pruneViews(state);

        const encoder = device.createCommandEncoder(FRAME_ENCODER);
        _render.encoder = encoder;
        state.beginGpuFrame(encoder);
        writeFrame(state);
        writeLighting(state);

        let count = 0;
        let depthOnly = 0;
        // shading views first, so they own the low slots the cluster + light-cull substrate is
        // sized for; depth-only views stack above them out of the cheap MAX_SLOTS budget
        for (const eid of state.query(CAMERAS)) {
            // auto-bind to the first <canvas> the frame it exists; an explicitly attachCanvas'd
            // camera is already in Views, so this is a no-op for it. Retried each frame until mount
            const view = bindCamera(eid, state);
            if (!view) continue;
            view.framebuffer = null;
            view.framebufferFormat = undefined;
            view.present = null;
            // derive the backing store from the display size + the camera's `Resolution` pin before any
            // consumer reads view.width/height (the offscreen + present below, the cluster pack above)
            sizeView(state, eid, view);
            if (view.width === 0 || view.height === 0) {
                view.framebuffer = null;
                view.present = null;
                continue;
            }
            if (!view.context) {
                // a canvas-less view (a shadow light's off-screen camera): it takes a cull slot and
                // packs its viewProj, but draws no framebuffer — its owner renders it to its own target
                view.framebuffer = null;
                view.present = null;
                _renderFrame.depthOnlyEids[depthOnly] = eid;
                _renderFrame.depthOnlyViews[depthOnly] = view;
                depthOnly++;
                continue;
            }
            if (count >= MAX_VIEWS) {
                console.warn(`shallot: ${MAX_VIEWS} camera cap reached; entity ${eid} skipped`);
                continue;
            }
            const texture = view.context.getCurrentTexture();
            if (!texture) {
                view.framebuffer = null;
                view.present = null;
                continue;
            }
            // present = the swapchain as a base-format storage view a compute composite writes via
            // textureStore (it encodes linear→sRGB itself); framebuffer = the offscreen the renderer
            // draws into and the composite reads (Render.format / sRGB, decoded to linear on load).
            view.present = texture.createView();
            view.framebuffer = offscreen(state, eid, view.width, view.height);
            view.framebufferFormat = _render.format;
            packView(state, eid, view, true, count);
            count++;
        }
        _render.shadeCount = count;
        for (let i = 0; i < depthOnly; i++) {
            if (count >= MAX_SLOTS) {
                console.warn(
                    `shallot: ${MAX_SLOTS} view-slot cap reached; entity ${_renderFrame.depthOnlyEids[i]} skipped`,
                );
                break;
            }
            packView(
                state,
                _renderFrame.depthOnlyEids[i],
                _renderFrame.depthOnlyViews[i],
                false,
                count,
            );
            count++;
        }

        _render.viewCount = count;
        // per-slot writer: only shading slots ([0, shadeCount)) ever bind a real View buffer — the
        // point/cascade atlas passes bind slot 0's buffer as an unread placeholder — so a depth-only slot
        // gets no write at all (design lock). Each write sources VIEW_BYTES from the same
        // per-slot viewStaging subrange the pack loop above always wrote
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
        const globalTransformRuntime = state.globalTransformRuntime;
        const globalTransformCount =
            _render.viewCount > 0 && globalTransformRuntime?.enabled
                ? (globalTransformRuntime.current?.count ?? 0)
                : 0;
        if (globalTransformRuntime && globalTransformCount > 0) {
            const pass = encoder.beginComputePass(GLOBAL_TRANSFORM_PASS);
            pass.setPipeline(globalTransformRuntime.pipeline!);
            pass.setBindGroup(0, globalTransformRuntime.group!);
            pass.dispatchWorkgroups(Math.ceil(globalTransformCount / 64));
            pass.end();
        }
    },
};

/** closes the frame: submits the encoder, advances `state.gpu.frame` */
const EndFrameSystem: System = {
    group: "draw",
    terminal: true,
    update(state) {
        const _render = state.resource(Render);
        const _renderFrame = state.resource(renderFrameKey);

        const device = state.gpu.device;
        if (!device) return;
        const encoder = _render.encoder;
        if (!encoder)
            throw new Error("render submission requires BeginFrameSystem to open an encoder");
        _renderFrame.submit[0] = encoder.finish();
        device.queue.submit(_renderFrame.submit);
        state.endGpuFrame();
        _render.encoder = null;
        state.resource(Views).forEach(clearTargets);
    },
};

/**
 * a no-op ordering anchor splitting the post-color seam: scene-space effects (fog) run `before` it,
 * screen-space overlays (outline) run `after` it, so an overlay composites on top of the transformed
 * scene. Both reference it by name, so neither imports the other (the scene-transform / overlay pair
 * stays decoupled). It carries no `update`: pure scheduling, invisible to the profiler. Sits in `draw`
 * with the rest of the seam; `BeginFrameSystem`/`EndFrameSystem` and the per-effect Color/Glaze edges
 * still bound it, so it needs no Color/Glaze edge of its own (render must not import sear/glaze).
 */
export const OverlaySystem: System = {
    name: "overlay",
    group: "draw",
};

/** allocates the device-shared substrate: format, view UBO, frame UBO */
async function initRender(state: State): Promise<void> {
    const _render = state.resource(Render);
    const _renderFrame = state.resource(renderFrameKey);

    if (!state.gpu.device) return;
    const { device } = state.gpu;

    // clear the render registries so each build re-registers from a clean slate (clear then
    // rebuild). This runs in RenderPlugin.initialize, before any producer / sear re-registers (they
    // depend on RenderPlugin), so a producer toggled off leaves no stale surface / draw /
    // mesh behind to be drawn against its torn-down buffers. A same-set rebuild is unchanged (every
    // plugin re-registers); a first build clears empty registries (a no-op).
    state.resource(Surfaces).clear();
    state.resource(Draws).clear();
    clearMeshes(state);

    // the scene renders into an rg11b10ufloat HDR offscreen so a tonemap (glaze, default Khronos Neutral)
    // rolls off radiance >1 rather than clamping it to white at store. rg11b10 (4B) halves the MSAA
    // color-target + resolve bandwidth vs rgba16float (8B), the dominant sear:color cost at 4× MSAA, for
    // ~3% relative precision (no alpha; over-blending doesn't need dst alpha). Single path, no flag — the
    // swapchain stays the base canvas format (glaze encodes linear→sRGB into it); this is the offscreen +
    // sear color-target format only
    _render.format = "rg11b10ufloat";

    const uniform = (label: string, size: number) =>
        device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

    _render.encoder = null;
    for (const b of _render.viewBuffers) b.destroy();
    _render.viewBuffers = Array.from({ length: MAX_VIEWS }, (_, slot) =>
        uniform(`shallot-view-${slot}`, VIEW_BYTES),
    );
    _render.viewStaging = new Float32Array(VIEW_UNIFORM_SIZE / 4);
    // fresh staging: every slot repacks on its first frame against the new buffers
    _renderFrame.viewKeys.fill(Number.NaN);
    const staging = _render.viewStaging;
    _renderFrame.viewProjs = Array.from({ length: MAX_SLOTS }, (_, slot) =>
        staging.subarray(slot * SLOT_FLOATS, slot * SLOT_FLOATS + 16),
    );
    _renderFrame.invViewProjs = Array.from({ length: MAX_VIEWS }, (_, slot) =>
        staging.subarray(slot * SLOT_FLOATS + 36, slot * SLOT_FLOATS + 52),
    );
    _renderFrame.lightViews = Array.from({ length: MAX_VIEWS }, (_, slot) =>
        state.resource(LightCull).viewStaging.subarray(slot * 16, slot * 16 + 16),
    );
    state.resource(Frame).buffer = uniform("shallot-frame", FRAME_UNIFORM_SIZE);
    state.resource(Lighting).buffer = uniform("shallot-lighting", LIGHTING_UNIFORM_SIZE);

    // one tagged cull volume per view, packed for the GPU cull pass and published
    // by name so any producer's cull resolves it the same way it resolves slabs
    _render.cullVolumes = device.createBuffer({
        label: "shallot-cull-volumes",
        size: MAX_SLOTS * CULL_VOLUME_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    _render.cullVolumeStaging = new Float32Array(MAX_SLOTS * CULL_VOLUME_FLOATS);
    _render.viewCount = 0;
    _render.shadeCount = 0;
    state.gpu.buffers.set("cullVolumes", _render.cullVolumes);
    state.gpu.typed.set(
        "cullVolumes",
        state.gpu.root
            .createBuffer(
                d.arrayOf(d.vec4f, MAX_SLOTS * (CULL_VOLUME_FLOATS / 4)),
                _render.cullVolumes,
            )
            .$usage("storage"),
    );
    state.resource(Views).clear();
    clearOffscreens(state);
    clearScratch(state);
}

/**
 * the renderer-agnostic substrate: frame loop, camera, Frame/View UBOs, and
 * the `Surfaces` / `Meshes` / `Draws` registries. Producer and consumer
 * plugins (Part, Sear, custom producers) depend on this. Users
 * typically don't list it directly: `PartPlugin` pulls it transitively,
 * and either can become a default plugin
 */
export const RenderPlugin: Plugin = {
    name: "Render",
    systems: [BeginFrameSystem, ClusterSystem, LightCullSystem, OverlaySystem, EndFrameSystem],
    components: {
        Camera,
        Resolution,
        AmbientLight,
        DirectionalLight,
        PointLight,
        Spot,
        Volumetric,
    },
    traits: {
        Camera: {
            requires: [GlobalTransform],
            defaults: () => ({
                mode: CameraMode.Perspective,
                fov: 60,
                near: 0.1,
                far: 1000,
                size: 5,
                clearColor: 0x2e2b28,
                antialias: 1,
            }),
            format: { clearColor: formatHex },
            enums: { mode: CameraMode },
        },
        Resolution: {
            requires: [Camera],
            defaults: () => ({ width: 0, height: 0 }),
        },
        AmbientLight: {
            singleton: true,
            defaults: () => ({ color: 0xffffff, intensity: 0.5 }),
            format: { color: formatHex },
        },
        DirectionalLight: {
            singleton: true,
            defaults: () => ({
                color: 0xffffff,
                intensity: 1.5,
                direction: [-0.6, -1.0, -0.8, 0],
            }),
            format: { color: formatHex },
        },
        PointLight: {
            requires: [GlobalTransform],
            defaults: () => ({ color: 0xffffff, intensity: 1, range: 10, radius: 0.1 }),
            format: { color: formatHex },
        },
        Spot: {
            requires: [PointLight],
            defaults: () => ({ inner: 20, outer: 30 }),
        },
        Volumetric: {
            defaults: () => ({}),
        },
    },

    async initialize(state) {
        initializeRenderState(state);
        initializeViewState(state);
        initializeClusterState(state);
        initializeFrameState(state);
        initializeLightingState(state);
        initializeMeshState(state);
        initializeImageState(state);
        initializeRenderFrameState(state);
        initializeDrawState(state);
        initializeSurfaceState(state);
        await initRender(state);
        const globalTransformRuntime = state.globalTransformRuntime;
        if (!globalTransformRuntime)
            throw new Error("GlobalTransform is unavailable before RenderPlugin initialization");
        // Its uniform binding reuses the leading vec4 in the Frame buffer written each frame.
        globalTransformRuntime.params = state.resource(Frame).buffer;
        globalTransformTable(state);
    },

    // pack the static meshes staged by `mesh()` during initialize into the
    // shared family buffer (runs after every initialize, before first render)
    warm(state) {
        flushMeshes(state);
        warmClusters(state);
        warmLightCull(state);
    },
};

// extension API for renderer + producer authors: the contract registries, the
// per-frame uniform singletons + their WGSL structs, the vertex-pull contract,
// canvas binding, and the frame-loop ordering anchor. The typical-user surface
// (components, plugin, public types, mesh()) lives in the index barrel. `VIEW_STRIDE`
// + `MAX_VIEWS` size a per-view uniform a consumer packs slot-major (glaze's postfx
// config); the buffer sizes and the cull-volume packer stay internal — a consumer reads
// the packed `Render.cullVolumes` buffer, never re-packs it. A producer that runs its own
// cull (Part's pack) reads the per-slot layout constants below to index + dispatch on the tag.

export { computeViewProj } from "./camera";
export type { ClusterView } from "./cluster";
export {
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    Clusters,
    clusterAabb,
    clusterCell,
    clusterCoord,
    clusterIndex,
    clusterView,
    LIGHT_POOL,
    LightCull,
    lightClusters,
    sliceDepth,
    zSlice,
} from "./cluster";
export * from "./contract";
export { Frame, FrameGpu, frameWgsl } from "./frame";
export { CULL_FRUSTUM, CULL_VOLUME_FLOATS, FRUSTUM_FLOATS, frustumPlanes } from "./frustum";
// the shared image→`texture_2d_array` upload path — the producer substrate glTF baseColor + the sprite atlas
// both sample, inward of both extras so neither reaches sideways into the other
export {
    allocArray,
    arrayFromBitmaps,
    commonSize,
    imageArray,
    mipLevels,
    uploadLayer,
} from "./image";

export {
    distanceAttenuation,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    LightingGpu,
    lightingWgsl,
    MAX_POINT_LIGHTS,
    PointLightGpu,
    PointLights,
    pointLightsWgsl,
    spotFactor,
    spotParams,
} from "./lighting";
export type { MeshBinding, MeshIndex, MeshStorage, QuantStreams } from "./mesh";
export {
    Meshes,
    meshBounds,
    packMeshes,
    quantizeMeshes,
    VERTEX_FLOATS,
    VERTEX_STRIDE,
} from "./mesh";
export type { Draw, DrawIndirectBuffer } from "./registry";
export { DrawIndexedIndirect, Draws } from "./registry";
export { Render } from "./render";
export {
    attachCanvas,
    attachView,
    backingSize,
    detachCanvas,
    linearToSrgb,
    linearToSrgbWgsl,
    MAX_SLOTS,
    MAX_VIEWS,
    sceneTransform,
    sizeView,
    VIEW_BYTES,
    VIEW_STRIDE,
    View,
    Views,
    viewWgsl,
} from "./view";
