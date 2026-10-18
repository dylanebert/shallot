import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { EntityRef, World } from "../../engine";
import { resizeViewport, Viewports } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { Camera, Resolution } from "./camera";
import { CULL_VOLUME_FLOATS } from "./frustum";
import { RenderContext } from "./render";

/**
 * Per-shading-slot uniform schema shared by GPU bindings and CPU staging offsets.
 * Each shading slot owns a separate static buffer in RenderContext.viewBuffers.
 */
export const ViewUniforms = d
    .struct({
        viewProj: d.mat4x4f,
        resolution: d.vec2f,
        right: d.vec4f,
        up: d.vec4f,
        projection: d.vec4f,
        eye: d.vec4f,
        invViewProj: d.mat4x4f,
        /** The selected global or camera ambient source, already multiplied by its cd/m² brightness. */
        ambientColor: d.vec4f,
        /** EV100 exposure factor, `2^-EV100 / 1.2`. */
        exposure: d.f32,
        /** Per-camera shadow filter enum; absent ShadowFilteringMethod packs Gaussian. */
        shadowFilteringMethod: d.u32,
    })
    .$name("ViewUniforms");

/**
 * dynamic-offset uniform stride. WebGPU `minUniformBufferOffsetAlignment` ≥ 256
 * forces this even though only the leading bytes carry data
 */
export const VIEW_STRIDE = 256;

/** One initial slot keeps the common single-view path small; all per-view state grows on demand. */
export const INITIAL_VIEW_CAPACITY = 1;

// WebGPU's default 64 KiB uniform binding fits 585 cull slots of 28 floats each.
export const MAX_SLOTS = Math.floor(
    (64 * 1024) / (CULL_VOLUME_FLOATS * Float32Array.BYTES_PER_ELEMENT),
);

/** Reserve the maximum 48 point-shadow faces and four directional cascades in the cull table. */
export const MAX_SHADOW_SLOTS = 52;

/** Hardware-derived presenting ceiling after reserving every built-in shadow slot; buffers remain lazy-grown. */
export const MAX_VIEWS = MAX_SLOTS - MAX_SHADOW_SLOTS;

/** Assign a presenting camera's packed slot while keeping overflow out of the fixed tables. */
export function presentingViewSlot(slot: number): number | undefined {
    return slot < MAX_VIEWS ? slot : undefined;
}

/** The warning BeginFrameSystem emits when it refuses a presenting camera past the limit. */
export function presentingViewRefusal(eid: number): string {
    return `shallot: ${MAX_VIEWS} presenting-view cull limit reached; entity ${eid} skipped`;
}

/** Byte size of each shading slot's whole ViewUniforms buffer. */
export const VIEW_BYTES = d.sizeOf(ViewUniforms);

/** Linear→sRGB (IEC 61966-2-1); presentation into a non-sRGB target must encode exactly once. */
export const linearToSrgb3 = tgpu.fn(
    [d.vec3f],
    d.vec3f,
)((c) => {
    "use gpu";
    const lo = std.mul(c, 12.92);
    const hi = std.sub(std.mul(1.055, std.pow(std.max(c, d.vec3f(0)), d.vec3f(1 / 2.4))), 0.055);
    return std.select(hi, lo, std.le(c, d.vec3f(0.0031308)));
});

/**
 * BeginFrameSystem sets framebuffer, present and slot each frame. For presenting views, slot indexes
 * RenderContext.viewBuffers. framebuffer starts as the view-sized, sampleable linear scene target in
 * RenderContext.format: resolve MSAA into it, or draw into it single-sample. Effects may replace it.
 * present is the final attachment in the non-sRGB base canvas format; presentation encodes linear→sRGB
 * exactly once. Scene color stays offscreen so effects can read it instead of writing the swapchain in place.
 * depth matches the main pass sample count and is null unless DepthPrepass or a plugin requests it.
 * A depth-only attachView registration takes a frustum-cull slot but has no canvas, context or color targets;
 * its owner renders to its own target.
 */
export interface View {
    canvas: HTMLCanvasElement | null;
    context: GPUCanvasContext | null;
    /** Current canvas swapchain texture, retained through the frame for pre-presentation readback. */
    canvasTexture?: GPUTexture;
    /** Last surface texture identity and its cached view. */
    surfaceTexture?: GPUTexture;
    surfaceView?: GPUTextureView;
    canvasTextureReused?: boolean;
    /** External captures copied by EndFrameSystem into the same submission as the presenting pass. */
    frameCaptures?: CanvasFrameCaptureRequest[];
    /** A weak device-loss watcher serves this view's captures without retaining detached canvases. */
    frameCaptureLossWatched?: boolean;
    /** Device loss is permanent for this view; later capture requests fail immediately. */
    frameCaptureLost?: Error;
    /** World-owned fixed-size final surface, absent on canvas and depth-only views. */
    texture?: GPUTexture;
    /** True after a frame acquired this surface and submitted its encoder. */
    presented?: boolean;
    /** Backing-store dimensions in device pixels, including Resolution or pixelRatio. */
    width: number;
    height: number;
    /** Canvas display dimensions in CSS pixels, mirrored from the world-owned viewport row. */
    clientWidth: number;
    clientHeight: number;
    /** canvas.clientWidth / clientHeight and window.devicePixelRatio at the last display-size sample */
    sampledClientWidth: number;
    sampledClientHeight: number;
    sampledDpr: number;
    /** index of this canvas's World-scoped viewport row */
    viewportIndex: number;
    framebuffer: GPUTextureView | null;
    /** current `framebuffer` format; scene-transform effects may redirect it to a scratch format */
    framebufferFormat?: GPUTextureFormat;
    present: GPUTextureView | null;
    depth: GPUTextureView | null;
    /** CPU-only world-to-view matrix shared by camera sorting and clustered-light packing. */
    viewFromWorld?: Float32Array;
    slot: number;
    camera: EntityRef;
}

export interface CanvasFrameCaptureRequest {
    buffer: GPUBuffer;
    width: number;
    height: number;
    bytesPerRow: number;
    resolve: (format: GPUTextureFormat) => void;
    reject: (error: Error) => void;
}

interface SceneTransformResult {
    read: GPUTextureView;
    write: GPUTextureView;
}

interface ScratchPair {
    a: Scratch | null;
    b: Scratch | null;
    w: number;
    h: number;
    result: SceneTransformResult;
}

interface ViewResources {
    views: Map<number, View>;
    offscreen: Map<number, { texture: GPUTexture; view: GPUTextureView; w: number; h: number }>;
    scratch: Map<number, ScratchPair>;
}

export const viewResourcesKey = { create: createViewResources };

function createViewResources(world: World): ViewResources {
    const resources: ViewResources = {
        views: new Map(),
        offscreen: new Map(),
        scratch: new Map(),
    };
    world.onDispose(() => {
        for (const view of resources.views.values()) {
            if (view.canvas && _canvasViews.get(view.canvas)?.view === view)
                _canvasViews.delete(view.canvas);
            rejectFrameCaptures(
                view,
                new Error("captureFrame refused: world disposed before presentation"),
            );
            view.surfaceTexture = undefined;
            view.surfaceView = undefined;
            view.texture?.destroy();
            view.context?.unconfigure();
        }
        resources.views.clear();
        for (const target of resources.offscreen.values()) target.texture.destroy();
        resources.offscreen.clear();
        for (const pair of resources.scratch.values()) {
            pair.a?.texture.destroy();
            pair.b?.texture.destroy();
        }
        resources.scratch.clear();
    });
    return resources;
}

export function initializeViewState(world: World): void {
    world.resource(viewResourcesKey);
}

/** every camera with a view, keyed by eid: canvas-bound ({@link attachCanvas}) or off-screen ({@link attachView}) */
export const Views: import("../../engine").Resource<Map<number, View>> = {
    create: (world) => world.resource(viewResourcesKey).views,
};

// canvas → the World that last bound it, for the dev-only rebuild guard below. WeakMap so a collected
// canvas drops its entry; never populated in production (the guard is dev-gated).
const _canvasOwners: WeakMap<HTMLCanvasElement, World> = new WeakMap();
const CANVAS_VIEWS = Symbol.for("@dylanebert/shallot/canvas-views");
const _canvasViews = (() => {
    const registry = globalThis as unknown as Record<symbol, unknown>;
    let views = registry[CANVAS_VIEWS] as
        | WeakMap<HTMLCanvasElement, { world: World; view: View }>
        | undefined;
    if (!views) {
        views = new WeakMap();
        registry[CANVAS_VIEWS] = views;
    }
    return views;
})();

/** Live canvas binding used by the frame-capture path; no source exists before a camera binds. */
export function canvasFrameBinding(
    canvas: HTMLCanvasElement,
): { world: World; view: View; device: GPUDevice } | undefined {
    const binding = _canvasViews.get(canvas);
    if (!binding || binding.world.disposed) return undefined;
    const device = binding.world.gpu.device;
    if (!device) return undefined;
    return { ...binding, device: rawDevice(device) };
}

/** Reject readbacks queued for a canvas view that cannot reach its next presentation. */
function rejectFrameCaptures(view: View, error: Error): void {
    for (const capture of view.frameCaptures ?? []) capture.reject(error);
    view.frameCaptures = undefined;
}

// Consumer bundlers may omit import.meta.env; missing or inaccessible DEV means production.
export function devEnabled(): boolean {
    try {
        const env = import.meta.env as Record<string, unknown> | undefined;
        return env?.DEV === true;
    } catch {
        return false;
    }
}

/** Records canvas ownership; warns if a different live World still owns it. Call only in development. */
export function trackCanvasOwner(canvas: HTMLCanvasElement, world: World): void {
    const prior = _canvasOwners.get(canvas);
    if (prior && prior !== world && !prior.disposed) {
        console.warn(
            "attachCanvas: canvas already bound to a live World — did an app rebuild without disposing the previous one? dispose it first (app.dispose() / world.dispose())",
        );
    }
    _canvasOwners.set(canvas, world);
}

/**
 * bind a canvas to a camera entity, 1:1: each camera owns one canvas. In dev builds it arms the
 * rebuild guard ({@link trackCanvasOwner}), which catches a rebuild that skipped `dispose`.
 */
export function attachCanvas(world: World, eid: number, canvas: HTMLCanvasElement): void {
    const _views = world.resource(Views);

    if (!world.gpu.device) throw new Error("attachCanvas: RenderingPlugin not initialized");
    if (!world.resource(RenderContext).format)
        throw new Error("attachCanvas: RenderContext.format not set");
    if (cameraView(world, eid)) throw new Error(`attachCanvas: eid ${eid} already bound`);

    const context = canvas.getContext("webgpu") as unknown as GPUCanvasContext | null;
    if (!context) throw new Error("attachCanvas: WebGPU canvas context unavailable");

    // record ownership only after the attach is validated — a failed attach must not claim the canvas, or a
    // later legitimate attach from a different live World warns spuriously.
    if (devEnabled()) trackCanvasOwner(canvas, world);

    const linearFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: rawDevice(world.gpu.device),
        format: linearFormat,
        alphaMode: "premultiplied",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });

    const rect = canvas.getBoundingClientRect();
    const viewportIndex =
        typeof document === "undefined"
            ? 0
            : Math.max(0, Array.from(document.querySelectorAll("canvas")).indexOf(canvas));
    const dpr = (typeof window === "undefined" ? 1 : window.devicePixelRatio) || 1;
    resizeViewport(world, viewportIndex, rect.width, rect.height, dpr);
    const view: View = {
        canvas,
        context,
        width: 0,
        height: 0,
        clientWidth: rect.width,
        clientHeight: rect.height,
        sampledClientWidth: canvas.clientWidth,
        sampledClientHeight: canvas.clientHeight,
        sampledDpr: dpr,
        viewportIndex,
        framebuffer: null,
        present: null,
        depth: null,
        viewFromWorld: new Float32Array(16),
        slot: 0,
        camera: world.ref(eid),
    };
    _views.set(eid, view);
    _canvasViews.set(canvas, { world, view });
}

/**
 * Backing dimensions in device pixels from CSS display dimensions. Zero resW/resH leaves an axis unset:
 * both unset use display×ratio; a set axis is exact and one unset follows display aspect.
 * pixelated marks backing below display size, including ratio < 1, for nearest-neighbour upscaling.
 */
export function backingSize(
    resW: number,
    resH: number,
    clientW: number,
    clientH: number,
    ratio: number,
): { w: number; h: number; pixelated: boolean } {
    let w: number;
    let h: number;
    if (resW > 0 && resH > 0) {
        w = resW;
        h = resH;
    } else if (resH > 0) {
        h = resH;
        w = Math.max(1, Math.round(resH * (clientW / clientH)));
    } else if (resW > 0) {
        w = resW;
        h = Math.max(1, Math.round(resW * (clientH / clientW)));
    } else {
        w = Math.max(1, Math.floor(clientW * ratio));
        h = Math.max(1, Math.floor(clientH * ratio));
    }
    return { w, h, pixelated: w < clientW || h < clientH };
}

// Cache by View identity so reattached cameras resolve size even when display inputs are unchanged.
const _sizeInputs = new WeakMap<View, Float64Array>();

/**
 * Refreshes canvas display geometry and resolves backing size from Resolution or world pixelRatio.
 * Called after binding each frame; sets nearest-neighbour upscaling when undersized. Resolution edits
 * apply without waiting for a resize event. Reads Resolution only when the entity carries it, so stale
 * storage from a recycled eid cannot act as a pin. No-op without a canvas; offscreen owners size their targets.
 */
export function sizeView(world: World, eid: number, view: View): void {
    const canvas = view.canvas;
    if (!canvas) return;
    // This read forces current layout before swapchain acquisition can use a stale display size.
    const clientWidth = canvas.clientWidth;
    const clientHeight = canvas.clientHeight;
    const dpr = (typeof window === "undefined" ? 1 : window.devicePixelRatio) || 1;
    if (
        clientWidth !== view.sampledClientWidth ||
        clientHeight !== view.sampledClientHeight ||
        dpr !== view.sampledDpr
    ) {
        const rect = canvas.getBoundingClientRect();
        view.sampledClientWidth = clientWidth;
        view.sampledClientHeight = clientHeight;
        view.sampledDpr = dpr;
        resizeViewport(world, view.viewportIndex, rect.width, rect.height, dpr);
    }
    const viewport = world.resource(Viewports).get(view.viewportIndex);
    if (!viewport || viewport.cssWidth <= 0 || viewport.cssHeight <= 0) return;
    view.clientWidth = viewport.cssWidth;
    view.clientHeight = viewport.cssHeight;
    const ratio =
        world.pixelRatio === "auto" ? Math.min(Math.max(viewport.dpr, 1), 2) : world.pixelRatio;
    const pinned = world.has(eid, Resolution);
    const resW = pinned ? world.storage(Resolution).width.get(eid) | 0 : 0;
    const resH = pinned ? world.storage(Resolution).height.get(eid) | 0 : 0;
    let inputs = _sizeInputs.get(view);
    if (
        inputs &&
        inputs[0] === resW &&
        inputs[1] === resH &&
        inputs[2] === view.clientWidth &&
        inputs[3] === view.clientHeight &&
        inputs[4] === ratio
    )
        return;
    if (!inputs) {
        inputs = new Float64Array(5);
        _sizeInputs.set(view, inputs);
    }
    inputs[0] = resW;
    inputs[1] = resH;
    inputs[2] = view.clientWidth;
    inputs[3] = view.clientHeight;
    inputs[4] = ratio;
    const { w, h, pixelated } = backingSize(resW, resH, view.clientWidth, view.clientHeight, ratio);
    const ir = pixelated ? "pixelated" : "auto";
    if (canvas.style.imageRendering !== ir) canvas.style.imageRendering = ir;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    view.width = w;
    view.height = h;
}

/**
 * Bind a camera to a world-owned final texture in device pixels. Dimensions stay fixed even with
 * Resolution. Detachment, camera pruning or world disposal destroys it. Refuses non-cameras,
 * duplicate bindings, uninitialized rendering and dimensions outside the device's texture limit.
 * BeginFrameSystem acquires it on the next draw; the final pass writes the same base format as a canvas.
 */
export function attachTexture(
    world: World,
    eid: number,
    size: { width: number; height: number },
): void {
    const device = world.gpu.device;
    if (!device || !world.resource(RenderContext).format)
        throw new Error("attachTexture: RenderingPlugin not initialized");
    if (!world.has(eid, Camera)) throw new Error("attachTexture: eid is not a camera");
    for (const value of [size.width, size.height]) {
        if (!Number.isInteger(value) || value <= 0 || value > device.limits.maxTextureDimension2D)
            throw new RangeError(
                "attachTexture: dimensions must be positive integers within maxTextureDimension2D",
            );
    }
    attachView(world, eid);
    const view = world.resource(Views).get(eid)!;
    const texture = rawDevice(device).createTexture({
        label: "camera final texture",
        size: [size.width, size.height],
        format: navigator.gpu.getPreferredCanvasFormat(),
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(texture);
    view.texture = texture;
    view.width = size.width;
    view.height = size.height;
}

/**
 * Register a depth-only view: it takes a cull slot and packs Camera + Transform, but has no
 * framebuffer or presenting surface. Its caller renders to its own target. Refuses duplicate
 * bindings; detachView, camera pruning and disposal release the view registration.
 */
export function attachView(world: World, eid: number): void {
    const _views = world.resource(Views);

    if (cameraView(world, eid)) throw new Error(`attachView: eid ${eid} already has a view`);
    _views.set(eid, {
        canvas: null,
        context: null,
        // square (aspect 1) — an off-screen view's own target sets the resolution; only the aspect
        // feeds `computeViewProj`, so a shadow's ortho box stays square
        width: 1,
        height: 1,
        clientWidth: 0,
        clientHeight: 0,
        sampledClientWidth: 0,
        sampledClientHeight: 0,
        sampledDpr: 0,
        viewportIndex: -1,
        framebuffer: null,
        present: null,
        depth: null,
        viewFromWorld: new Float32Array(16),
        slot: 0,
        camera: world.ref(eid),
    });
}

/** Release a camera's view and owned targets (canvas, texture or depth-only). Safe on unbound eids. */
export function detachView(world: World, eid: number): void {
    const _views = world.resource(Views);
    const view = _views.get(eid);

    if (view) {
        view.texture?.destroy();
        view.canvasTexture = undefined;
        view.surfaceTexture = undefined;
        view.surfaceView = undefined;
        rejectFrameCaptures(
            view,
            new Error("captureFrame refused: canvas detached before presentation"),
        );
        if (view.canvas && _canvasViews.get(view.canvas)?.view === view)
            _canvasViews.delete(view.canvas);
    }
    _views.delete(eid);
    releaseOffscreen(world, eid);
    releaseScratch(world, eid);
}

/** The eid's View while the entity it was attached to is live. A View left by a destroyed
 * entity, including one whose eid was recycled, is released here rather than at the next frame. */
export function cameraView(world: World, eid: number): View | undefined {
    const view = world.resource(Views).get(eid);
    if (!view || world.resolve(view.camera)) return view;
    detachView(world, eid);
    return undefined;
}

/**
 * Releases views whose camera despawned or lost Camera, including recycled eids before their first frame.
 * BeginFrameSystem calls this before binding; retained references prevent a replacement camera from
 * inheriting the previous camera's canvas or texture.
 */
export function pruneViews(world: World): void {
    world.resource(Views).forEach(pruneView, world);
}

function pruneView(this: World, view: View, eid: number): void {
    if (!this.resolve(view.camera) || !this.has(eid, Camera)) detachView(this, eid);
}

/** the camera's offscreen color target, (re)allocated to the view size. Renderer-agnostic: the main pass's
 * MSAA resolve and the `Custom` single-sample draw both target it; {@link BeginFrameSystem} sets it on
 * `view.framebuffer` each frame. Stored per camera so views never overwrite one shared target.
 * Uses RenderContext.format for linear HDR color, preserving values above one until tonemapping. */
export function offscreen(world: World, eid: number, w: number, h: number): GPUTextureView {
    const _viewResources = world.resource(viewResourcesKey);

    const cached = _viewResources.offscreen.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = world.gpu.device.createTexture({
        label: `shallot-offscreen-${eid}`,
        size: { width: w, height: h },
        format: world.resource(RenderContext).format,
        // Keep the actual scene target readable by `probeTexture` without inserting a render pass.
        usage:
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
    });
    const view = texture.createView();
    _viewResources.offscreen.set(eid, { texture, view, w, h });
    return view;
}

/** @internal the World-owned texture behind a camera's rendered offscreen view. */
export function offscreenTexture(world: World, eid: number): GPUTexture | undefined {
    return world.resource(viewResourcesKey).offscreen.get(eid)?.texture;
}

function releaseOffscreen(world: World, eid: number): void {
    const _viewResources = world.resource(viewResourcesKey);

    _viewResources.offscreen.get(eid)?.texture.destroy();
    _viewResources.offscreen.delete(eid);
}

// Scene-transform scratches support storage writes as well as render attachments.
interface Scratch {
    texture: GPUTexture;
    view: GPUTextureView;
}

const SCENE_SCRATCH_FORMAT: GPUTextureFormat = "rgba16float";

function scratchTexture(world: World, eid: number, slot: "a" | "b", w: number, h: number): Scratch {
    const texture = world.gpu.device.createTexture({
        label: `scene-scratch-${eid}-${slot}`,
        size: { width: w, height: h },
        format: SCENE_SCRATCH_FORMAT,
        usage:
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT,
    });
    return { texture, view: texture.createView() };
}

/**
 * Redirects scene color through a compute effect: read is the current framebuffer; write is the other
 * half of a lazily allocated, per-camera rgba16float scratch pair and becomes the next framebuffer.
 * Bind read as input and write as storage output, then dispatch before the next effect reads it.
 * Call after MainPassSystem, before OverlaySystem for scene effects or after it for overlays,
 * and before presentation. read and write never alias within a call.
 * BeginFrameSystem resets the chain to the offscreen each frame. The result object is reused on this
 * camera's next call; later calls also reuse earlier scratch outputs, so do not retain those outputs
 * as independent images. A single effect allocates only the first scratch.
 */
export function sceneTransform(world: World, view: View, eid: number): SceneTransformResult {
    const _viewResources = world.resource(viewResourcesKey);

    const read = view.framebuffer;
    if (!read) throw new Error("sceneTransform: view has no framebuffer");
    let pair = _viewResources.scratch.get(eid);
    if (!pair || pair.w !== view.width || pair.h !== view.height) {
        pair?.a?.texture.destroy();
        pair?.b?.texture.destroy();
        pair = {
            a: null,
            b: null,
            w: view.width,
            h: view.height,
            result: { read, write: read },
        };
        _viewResources.scratch.set(eid, pair);
    }
    const slot: "a" | "b" = read === pair.a?.view ? "b" : "a";
    const scratch = (pair[slot] ??= scratchTexture(world, eid, slot, view.width, view.height));
    view.framebuffer = scratch.view;
    view.framebufferFormat = SCENE_SCRATCH_FORMAT;
    pair.result.read = read;
    pair.result.write = scratch.view;
    return pair.result;
}

function releaseScratch(world: World, eid: number): void {
    const _viewResources = world.resource(viewResourcesKey);

    const pair = _viewResources.scratch.get(eid);
    pair?.a?.texture.destroy();
    pair?.b?.texture.destroy();
    _viewResources.scratch.delete(eid);
}

export function clearOffscreens(world: World): void {
    const _viewResources = world.resource(viewResourcesKey);

    for (const o of _viewResources.offscreen.values()) o.texture.destroy();
    _viewResources.offscreen.clear();
}

export function clearScratch(world: World): void {
    const _viewResources = world.resource(viewResourcesKey);

    for (const p of _viewResources.scratch.values()) {
        p.a?.texture.destroy();
        p.b?.texture.destroy();
    }
    _viewResources.scratch.clear();
}

/**
 * auto-bind a camera to the first `<canvas>` in the document, idempotently. The zero-config
 * single-view path. A no-op (returns undefined) headless or until a canvas mounts;
 * `BeginFrameSystem` retries each frame, so a late-mounted canvas binds when it appears.
 * Multi-view binds each camera explicitly via {@link attachCanvas} before its first frame.
 */
export function bindCamera(eid: number, world: World): View | undefined {
    const _views = world.resource(Views);

    const existing = _views.get(eid);
    if (existing) return existing;
    if (typeof document === "undefined") return undefined;
    const canvas = document.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return undefined;
    // Two cameras acquiring one canvas would invalidate the first camera's swapchain texture.
    for (const view of _views.values()) if (view.canvas === canvas) return undefined;
    attachCanvas(world, eid, canvas);
    return _views.get(eid);
}
