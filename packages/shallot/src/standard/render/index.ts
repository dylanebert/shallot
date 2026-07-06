import type { Plugin, System } from "../../engine";
import { Compute, formatHex, invert } from "../../engine";
import { SlabPlugin } from "../slab";
import { composeTransform, composeTransforms, Transform, TransformsPlugin } from "../transforms";
import { Camera, CameraMode, computeViewProj, Resolution } from "./camera";
import {
    ClusterSystem,
    LightCull,
    LightCullSystem,
    packClusterView,
    warmClusters,
    warmLightCull,
} from "./cluster";
import { FRAME_UNIFORM_SIZE, Frame, writeFrame } from "./frame";
import { CULL_VOLUME_FLOATS, frustumVolume } from "./frustum";

import {
    AmbientLight,
    DirectionalLight,
    LIGHTING_UNIFORM_SIZE,
    Lighting,
    PointLight,
    Spot,
    Volumetric,
    writeLighting,
} from "./lighting";
import { clearMeshes, flushMeshes } from "./mesh";
import { Draws, Surfaces } from "./registry";
import { Render } from "./render";
import {
    bindCamera,
    clearOffscreens,
    clearScratch,
    detachCanvas,
    MAX_SLOTS,
    MAX_VIEWS,
    offscreen,
    sizeView,
    VIEW_STRIDE,
    VIEW_UNIFORM_SIZE,
    type View,
    Views,
} from "./view";

// the public happy path: the component contract (camera + lights) and meshes.
// Everything else a renderer or producer touches — the Render singleton, the
// View/Surface/Draw contract, canvas binding, the Lighting uniform, the frame
// loop — is the extension API in `render/core`.
export { Camera, CameraMode, Resolution } from "./camera";
export { AmbientLight, DirectionalLight, PointLight, Spot, Volumetric } from "./lighting";
export type { Mesh } from "./mesh";
export { mesh } from "./mesh";

const _camWorld = new Float32Array(16);

// write a world-matrix column (base = column index * 4), normalized, into `out` at `at`
function basisColumn(world: Float32Array, base: number, out: Float32Array, at: number): void {
    const x = world[base];
    const y = world[base + 1];
    const z = world[base + 2];
    const inv = 1 / (Math.hypot(x, y, z) || 1);
    out[at] = x * inv;
    out[at + 1] = y * inv;
    out[at + 2] = z * inv;
    out[at + 3] = 0;
}

/**
 * opens the frame: creates the encoder, writes the Frame UBO, records the
 * world-matrix compose dispatch, acquires each view's swapchain backbuffer
 * (`view.present`) + offscreen scene-color target (`view.framebuffer`), and
 * packs the View UBO. Producer and renderer systems both run
 * `after: [BeginFrameSystem]`; `EndFrameSystem` (the sole `last: true` system)
 * submits the encoder.
 */
export const BeginFrameSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    first: true,
    update(state) {
        Render.encoder = null;
        const device = Compute.device;
        if (!device) return;

        // auto-bind's inverse. A destroyed camera leaves a stale View whose ResizeObserver leaks
        // and whose eid, once recycled, re-binds to the wrong canvas. Membership is the liveness
        // signal (re-derived each frame, the gate Part's pack also applies), so a View lacking a
        // live Camera is dropped here.
        for (const eid of Views.keys()) if (!state.has(eid, Camera)) detachCanvas(eid);

        Render.encoder = device.createCommandEncoder({ label: "kitchen-frame" });
        writeFrame(state);
        writeLighting(state);
        composeTransforms(Render.encoder);

        let count = 0;
        const slotFloats = VIEW_STRIDE / 4;
        // viewProj + resolution + basis + frustum, the per-slot state every view carries. A shading
        // view (presenting camera) additionally packs the clustered-light state — its cluster params
        // and world→view matrix — into the same slot index; a depth-only view (a shadow light's
        // off-screen camera) never does, so the cluster substrate is sized by MAX_VIEWS while the
        // cheap slots run to MAX_SLOTS
        const pack = (eid: number, view: View, shading: boolean) => {
            view.slot = count;
            const offset = count * slotFloats;
            const viewProj = Render.viewStaging.subarray(offset, offset + 16);
            // the light cull reads each shading slot's world→view matrix to bring
            // world-space lights into cluster space
            computeViewProj(
                eid,
                view.width / view.height,
                viewProj,
                shading ? LightCull.viewStaging.subarray(count * 16, count * 16 + 16) : undefined,
            );
            // resolution (pixels) follows viewProj in the View struct — a screen-space
            // producer (lines) reads it to size constant-pixel-width geometry
            Render.viewStaging[offset + 16] = view.width;
            Render.viewStaging[offset + 17] = view.height;
            // camera basis (right at floats 20-23, up at 24-27; 18-19 pad before the vec4) —
            // billboard surfaces orient quads from it (in a shadow pass, the light camera's, so
            // billboards face the light). Normalized: the camera Transform may scale
            composeTransform(eid, _camWorld);
            basisColumn(_camWorld, 0, Render.viewStaging, offset + 20);
            basisColumn(_camWorld, 4, Render.viewStaging, offset + 24);
            // pack this view's frustum cull volume — the pack tests each instance's bound against
            // cullVolumes[slot]'s 6 planes. Every view culls by frustum: cameras, the sun, and each
            // point/spot shadow combo (its own frustum-culled depth view)
            frustumVolume(Render.cullVolumeStaging, count, viewProj);
            // pack the view's cluster params from the same camera fields —
            // ClusterSystem rebuilds the AABB grid only when they change.
            // View.cluster: (near, far, perspective, slot) — sear's FS maps a
            // fragment to its froxel and indexes the slot-major light grid
            if (shading) {
                const cv = packClusterView(eid, view.width / view.height, count);
                Render.viewStaging[offset + 28] = cv.near;
                Render.viewStaging[offset + 29] = cv.far;
                Render.viewStaging[offset + 30] = cv.perspective ? 1 : 0;
            } else {
                Render.viewStaging[offset + 28] = 0;
                Render.viewStaging[offset + 29] = 0;
                Render.viewStaging[offset + 30] = 0;
            }
            Render.viewStaging[offset + 31] = count;
            // eye (floats 32-35): the camera's world-space position — viewProj's translation column —
            // for view-dependent shading (specular V = normalize(eye - world))
            Render.viewStaging[offset + 32] = _camWorld[12];
            Render.viewStaging[offset + 33] = _camWorld[13];
            Render.viewStaging[offset + 34] = _camWorld[14];
            Render.viewStaging[offset + 35] = 1;
            // invViewProj (floats 36-51): a screen-space pass (fog) reconstructs world position from depth
            // via ndc → invViewProj. Only a shading view (a presenting camera) runs such a pass, so a
            // depth-only shadow view skips the 4×4 inverse — the costliest op in the pack — and zeroes the
            // slot. invert reads viewProj fully into locals before writing, so inverting into a sibling
            // subarray of the same staging never aliases
            if (shading) invert(viewProj, Render.viewStaging.subarray(offset + 36, offset + 52));
            else Render.viewStaging.fill(0, offset + 36, offset + 52);
            count++;
        };

        // shading views first, so they own the low slots the cluster + light-cull substrate is
        // sized for; depth-only views stack above them out of the cheap MAX_SLOTS budget
        const depthOnly: [number, View][] = [];
        for (const eid of state.query([Camera])) {
            // auto-bind to the first <canvas> the frame it exists; an explicitly attachCanvas'd
            // camera is already in Views, so this is a no-op for it. Retried each frame until mount
            const view = bindCamera(eid);
            if (!view) continue;
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
                depthOnly.push([eid, view]);
                continue;
            }
            if (count >= MAX_VIEWS) {
                console.warn(`kitchen: ${MAX_VIEWS} camera cap reached; entity ${eid} skipped`);
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
            view.framebuffer = offscreen(eid, view.width, view.height);
            pack(eid, view, true);
        }
        Render.shadeCount = count;
        for (const [eid, view] of depthOnly) {
            if (count >= MAX_SLOTS) {
                console.warn(`kitchen: ${MAX_SLOTS} view-slot cap reached; entity ${eid} skipped`);
                break;
            }
            pack(eid, view, false);
        }

        Render.viewCount = count;
        if (count > 0) {
            device.queue.writeBuffer(
                Render.viewBuffer,
                0,
                Render.viewStaging as Float32Array<ArrayBuffer>,
                0,
                count * slotFloats,
            );
            device.queue.writeBuffer(
                Render.cullVolumes,
                0,
                Render.cullVolumeStaging as Float32Array<ArrayBuffer>,
                0,
                count * CULL_VOLUME_FLOATS,
            );
        }
    },
};

/** closes the frame: submits the encoder, advances `Compute.frame` */
const EndFrameSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    last: true,
    update() {
        const encoder = Render.encoder;
        if (!Compute.device || !encoder) return;
        Compute.device.queue.submit([encoder.finish()]);
        Compute.frame++;
        Render.encoder = null;
        for (const view of Views.values()) {
            view.framebuffer = null;
            view.present = null;
        }
    },
    dispose() {
        for (const view of Views.values()) view.observer?.disconnect();
        Views.clear();
        clearOffscreens();
        clearScratch();
    },
};

/**
 * a no-op ordering anchor splitting the post-color seam: scene-space transforms (fog) run `before` it,
 * screen-space overlays (outline) run `after` it, so an overlay composites on top of the transformed
 * scene. Both reference it by name, so neither imports the other (the scene-transform / overlay pair
 * stays decoupled). It carries no `update`: pure scheduling, invisible to the profiler. Sits in `draw`
 * with the rest of the seam; `BeginFrameSystem`/`EndFrameSystem` and the per-effect Color/Glaze edges
 * still bound it, so it needs no Color/Glaze edge of its own (render must not import sear/glaze).
 */
export const OverlaySystem: System = {
    name: "overlay",
    group: "draw",
    annotations: { mode: "always" },
};

/** allocates the device-shared substrate: format, view UBO, frame UBO */
async function initRender(): Promise<void> {
    if (!Compute.device) return;
    const { device } = Compute;

    // clear the render registries so each build re-registers from a clean slate (ecs.md "clear then
    // rebuild"). This runs in RenderPlugin.initialize, before any producer / sear re-registers (they
    // depend on RenderPlugin), so a producer toggled off in the editor leaves no stale surface / draw /
    // mesh behind to be drawn against its torn-down buffers. A same-set rebuild is unchanged (every
    // plugin re-registers); a first build clears empty registries (a no-op).
    Surfaces.clear();
    Draws.clear();
    clearMeshes();

    // the scene renders into an rg11b10ufloat HDR offscreen so a tonemap (glaze, default Khronos Neutral)
    // rolls off radiance >1 rather than clamping it to white at store. rg11b10 (4B) halves the MSAA
    // color-target + resolve bandwidth vs rgba16float (8B), the dominant sear:color cost at 4× MSAA, for
    // ~3% relative precision (no alpha; over-blending doesn't need dst alpha). Single path, no flag — the
    // swapchain stays the base canvas format (glaze encodes linear→sRGB into it); this is the offscreen +
    // sear color-target format only
    Render.format = "rg11b10ufloat";

    const uniform = (label: string, size: number) =>
        device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

    Render.encoder = null;
    Render.viewBuffer = uniform("kitchen-view", VIEW_UNIFORM_SIZE);
    Render.viewStaging = new Float32Array(VIEW_UNIFORM_SIZE / 4);
    Frame.buffer = uniform("kitchen-frame", FRAME_UNIFORM_SIZE);
    Lighting.buffer = uniform("kitchen-lighting", LIGHTING_UNIFORM_SIZE);

    // one tagged cull volume per view, packed for the GPU cull pass and published
    // by name so any producer's cull resolves it the same way it resolves slabs
    Render.cullVolumes = device.createBuffer({
        label: "kitchen-cull-volumes",
        size: MAX_SLOTS * CULL_VOLUME_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    Render.cullVolumeStaging = new Float32Array(MAX_SLOTS * CULL_VOLUME_FLOATS);
    Render.viewCount = 0;
    Render.shadeCount = 0;
    Compute.buffers.set("cullVolumes", Render.cullVolumes);
    Views.clear();
    clearOffscreens();
    clearScratch();
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
            requires: [Transform],
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
            requires: [Transform],
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
    dependencies: [SlabPlugin, TransformsPlugin],

    async initialize() {
        await initRender();
    },

    // pack the static meshes staged by `mesh()` during initialize into the
    // shared family buffer (runs after every initialize, before first render)
    async warm(state) {
        flushMeshes();
        await Promise.all([warmClusters(), warmLightCull(state)]);
    },
};
