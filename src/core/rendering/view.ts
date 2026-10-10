import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import type { EntityRef, World } from "../../engine";
import { resizeViewport, Viewports } from "../../engine";
import { rawDevice } from "../../engine/runtime";
import { Camera, Resolution } from "./camera";
import { RenderContext } from "./render";

/**
 * the per-camera `ViewUniforms` UBO schema — the single source of truth for both sides of the layout
 * (`d.sizeOf` / `d.memoryLayoutOf` size {@link VIEW_BYTES} and every CPU staging write, `view.test.ts`
 * red-proven against a field reorder, the `Step` precedent). One instance per shading slot lives in its
 * own static uniform buffer ({@link RenderContext.viewBuffers}).
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
    })
    .$name("ViewUniforms");

/**
 * dynamic-offset uniform stride. WebGPU `minUniformBufferOffsetAlignment` ≥ 256
 * forces this even though only the leading bytes carry data
 */
export const VIEW_STRIDE = 256;

/**
 * Maximum presenting views per frame. `BeginFrameSystem` assigns them the low slots
 * `[0, MAX_VIEWS)`; depth-only views get the slots above, out of {@link MAX_SLOTS}.
 */
export const MAX_VIEWS = 8;

/** total view slots per frame: shading cameras + depth-only views (the sun's light camera + the
 * point-shadow member-compaction "union" camera). Sizes only the cheap per-slot state
 * (`RenderContext.viewStaging`, `RenderContext.cullVolumes` — depth-only slots allocate no {@link RenderContext.viewBuffers}
 * entry, only the shading prefix does), so it's generous */
export const MAX_SLOTS = 64;

export const VIEW_UNIFORM_SIZE = VIEW_STRIDE * MAX_SLOTS;

/**
 * the byte size of the {@link ViewUniforms} uniform a surface statically reads, from the schema: projection,
 * camera basis, eye and inverse view-projection, followed by per-camera ambient radiance and exposure. Fog
 * reconstructs surface positions and backgrounds derive camera rays through `invViewProj`; each shading slot also
 * carries the camera's effective ambient override and EV100 factor. Each shading slot binds its own whole
 * {@link RenderContext.viewBuffers} buffer of exactly this size.
 */
export const VIEW_BYTES = d.sizeOf(ViewUniforms);

/**
 * Linear→sRGB encode (IEC 61966-2-1) for presentation into a non-sRGB target.
 * Tonemapping and replacement passes encode exactly once. The per-channel scalar twin is
 * `linearToSrgb1` (`utils`), which the LDR color codec packs through.
 */
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
 * a camera's per-frame view state. `framebuffer` + `present` + `slot` are set by `BeginFrameSystem`
 * each frame and read by the renderers. `slot` is the camera's index into the packed ViewUniforms UBO; use it
 * to index {@link RenderContext.viewBuffers} (`RenderContext.viewBuffers[slot]`) when binding. `framebuffer` is the
 * per-camera **offscreen**
 * scene-color target the renderer draws into (core resolves its MSAA color into it; a custom renderer
 * draws straight into it single-sample): sampleable (`TEXTURE_BINDING`), in `RenderContext.format`, sized to
 * the view; a composite `textureLoad`s it and writes the result into `present`. `present` is the swapchain
 * backbuffer, as a render attachment in the base canvas format (not sRGB). The final pass
 * writes it, encoding linear→sRGB once. The split from `framebuffer` exists so postfx
 * has a rendered color to read: writing the swapchain in place leaves nothing to read back.
 * `depth` is the camera's stored depth lane, matching the main pass's sample count. Core publishes it only when
 * the camera carries `DepthPrepass` or a registered plugin requests it; otherwise it is null. A
 * canvas-bound view (`attachCanvas`) renders to that canvas; a canvas-less view (`attachView`) has
 * no `canvas` / `context` / `observer` and a null `framebuffer` / `present`. It still takes a cull slot
 * (a shadow light's off-screen camera renders to its own target, not the screen). Every view
 * frustum-culls from its viewProj: cameras, the sun, and each point/spot shadow combo's depth view.
 */
export interface View {
    canvas: HTMLCanvasElement | null;
    context: GPUCanvasContext | null;
    /** Current canvas swapchain texture, retained through the frame for pre-presentation readback. */
    canvasTexture?: GPUTexture;
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
    // the render backing-store size (device px). Derived each frame by `sizeView` from the display size
    // below + the camera's `Resolution` pin (or the world's pixelRatio). Every consumer — offscreen,
    // Consumers read this view size, so a low-res pin flows through by sizing it here.
    width: number;
    height: number;
    // the canvas CSS display size (px), mirrored from the World-scoped viewport row for compatibility.
    // The backing above reads the row directly, so a runtime `Resolution` edit re-sizes the view without
    // waiting on a resize event.
    clientWidth: number;
    clientHeight: number;
    /** index of this canvas's World-scoped viewport row */
    viewportIndex: number;
    framebuffer: GPUTextureView | null;
    /** current `framebuffer` format; scene-transform effects may redirect it to a scratch format */
    framebufferFormat?: GPUTextureFormat;
    present: GPUTextureView | null;
    depth: GPUTextureView | null;
    slot: number;
    observer: ResizeObserver | null;
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

interface ViewResources {
    views: Map<number, View>;
    offscreen: Map<number, { texture: GPUTexture; view: GPUTextureView; w: number; h: number }>;
    scratch: Map<number, { a: Scratch | null; b: Scratch | null; w: number; h: number }>;
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
            view.texture?.destroy();
            view.observer?.disconnect();
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

/** Create this world's view and target registries during RenderingPlugin initialization. */
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

// read `import.meta.env.DEV` typeof-safely: the engine is bundled by arbitrary consumer bundlers, and a
// bare `import.meta.env.DEV` throws where `import.meta.env` is undefined (non-vite). Optional-chained,
// wrapped so an exotic `import.meta` shape can't take down attachCanvas. Exported as a test seam (pins the
// false-not-throw contract off a vite build) — not on the rendering barrel.
export function devEnabled(): boolean {
    try {
        const env = import.meta.env as Record<string, unknown> | undefined;
        return env?.DEV === true;
    } catch {
        return false;
    }
}

/**
 * dev-only rebuild guard, canvas-keyed: warn when `canvas` is still held by a live, undisposed *different*
 * World — an app rebuilt without disposing the prior one (the leak class `World.onDispose` closes), then
 * record the new owner. Two apps on distinct canvases stay silent; a proper dispose flips the prior owner's
 * `disposed`, so a later rebind is silent too. Internal + a test seam — not on the rendering barrel.
 */
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
export function attachCanvas(eid: number, canvas: HTMLCanvasElement, world: World): void {
    const _views = world.resource(Views);

    if (!world.gpu.device) throw new Error("attachCanvas: RenderingPlugin not initialized");
    if (!world.resource(RenderContext).format)
        throw new Error("attachCanvas: RenderContext.format not set");
    if (_views.has(eid)) throw new Error(`attachCanvas: eid ${eid} already bound`);

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
        viewportIndex,
        framebuffer: null,
        present: null,
        depth: null,
        slot: 0,
        observer: null!,
        camera: world.ref(eid),
    };
    // the observer is the DOM producer for the World-scoped viewport row. `sizeView` derives the backing
    // from that row each frame, so a runtime `Resolution` edit re-sizes (the observer never fires for that)
    // and the backing write stays at frame start, off the async resize callback.
    view.observer = new ResizeObserver(() => {
        const r = canvas.getBoundingClientRect();
        view.clientWidth = r.width;
        view.clientHeight = r.height;
        const nextDpr = (typeof window === "undefined" ? 1 : window.devicePixelRatio) || 1;
        resizeViewport(world, viewportIndex, r.width, r.height, nextDpr);
    });
    view.observer.observe(canvas);
    _views.set(eid, view);
    _canvasViews.set(canvas, { world, view });
}

/**
 * resolve a view's backing-store size (device px) from its CSS display size. `resW`/`resH` are a
 * {@link Resolution} pin (0 = that axis unset); both unset → the display size × `ratio` (the pixelRatio
 * policy). A set axis is exact and the unset one follows the display aspect, so `resH = 360` alone renders
 * 360 lines tall. `pixelated` is true whenever the backing is below the display: the nearest-neighbor
 * upscale that keeps a pinned low resolution crisp (and the `ratio < 1` pixel-art case, unchanged).
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

// the inputs each view's backing size was last resolved from: `[resW, resH, clientW, clientH, ratio]`. The
// size is a pure function of them, so a frame that changes none keeps the sizes already on the view and the
// canvas and resolves nothing. Keyed by the View object, so a re-attached camera's fresh view sizes on its
// first frame.
const _sizeInputs = new WeakMap<View, Float64Array>();

/**
 * size a canvas-bound view's backing store from its cached display size + its {@link Resolution} pin (the
 * world's pixelRatio when absent), and set the nearest-neighbor upscale. {@link BeginFrameSystem} calls it
 * per camera each frame after binding, so a `Resolution` edit and a canvas resize both re-size here. A
 * no-op for a canvas-less (off-screen) view, which sizes its own target. The Resolution read is membership-
 * gated: a recycled eid's stale field value never leaks into a camera that carries no pin.
 */
export function sizeView(world: World, eid: number, view: View): void {
    const canvas = view.canvas;
    if (!canvas) return;
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
 * bindings; detachCanvas, camera pruning and disposal release the view registration.
 */
export function attachView(world: World, eid: number): void {
    const _views = world.resource(Views);

    if (_views.has(eid)) throw new Error(`attachView: eid ${eid} already has a view`);
    _views.set(eid, {
        canvas: null,
        context: null,
        // square (aspect 1) — an off-screen view's own target sets the resolution; only the aspect
        // feeds `computeViewProj`, so a shadow's ortho box stays square
        width: 1,
        height: 1,
        clientWidth: 0,
        clientHeight: 0,
        viewportIndex: -1,
        framebuffer: null,
        present: null,
        depth: null,
        slot: 0,
        observer: null,
        camera: world.ref(eid),
    });
}

/** Release a camera's view and owned targets (canvas, texture or depth-only). Safe on unbound eids. */
export function detachCanvas(world: World, eid: number): void {
    const _views = world.resource(Views);
    const view = _views.get(eid);

    if (view) {
        view.observer?.disconnect();
        view.texture?.destroy();
        view.canvasTexture = undefined;
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

/**
 * drop the auto-bind's inverse: a View whose camera despawned, or whose eid was recycled to a new camera.
 * The kept camera reference prevents a recycled eid inheriting its canvas or texture,
 * including recycling before the first frame.
 * {@link BeginFrameSystem} calls it at frame start, before binding.
 */
export function pruneViews(world: World): void {
    world.resource(Views).forEach(pruneView, world);
}

// one View's liveness check for the `pruneViews` walk; the walk passes the World as `this`
function pruneView(this: World, view: View, eid: number): void {
    if (!this.resolve(view.camera) || !this.has(eid, Camera)) detachCanvas(this, eid);
}

// per-camera offscreen scene-color target — the `view.framebuffer` a renderer draws (or resolves)
// into and tonemapping presents. `RenderContext.format` is rg11b10ufloat (HDR): a renderer writes
// linear and tonemapping reads it linear, keeping radiance >1 alive for the operator. Sized to
// the view, recreated on resize; one per camera so multi-view never last-camera-wins a single shared texture

/** the camera's offscreen color target, (re)allocated to the view size. Renderer-agnostic: the main pass's
 * MSAA resolve and the `Custom` single-sample draw both target it; {@link BeginFrameSystem} sets it on
 * `view.framebuffer` each frame */
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

// free one camera's offscreen target (on detach). Safe on cameras that never allocated one
function releaseOffscreen(world: World, eid: number): void {
    const _viewResources = world.resource(viewResourcesKey);

    _viewResources.offscreen.get(eid)?.texture.destroy();
    _viewResources.offscreen.delete(eid);
}

// the write half of a scene-transform postfx effect: a per-view **ping-pong pair** of scratches the
// chained effects bounce the transformed scene between. Always rgba16float (a storage-capable, sampleable
// HDR format): the framebuffer offscreen (rg11b10ufloat) isn't storage-capable, so a scratch must be its
// own storage-writable format. One effect uses `a` alone; a second (fog → outline) writes `b`, reading
// `a`; `b` allocates lazily so a fog-only scene keeps one scratch. **Ceiling: two effects per frame** — a
// third in-frame consumer would alias `a` (read=b → write=a, overwriting the first effect's output mid-frame);
// that's the trigger to grow a real ring buffer here, not a silent corruption to leave in place
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
 * the scene-transform seam: redirect a camera's scene color through a postfx compute effect. Returns
 * `read` (the current `view.framebuffer`: the renderer's resolved scene, or the prior effect's output) and
 * `write` (a lazily-allocated scratch, the *other* half of the ping-pong pair from `read`), and repoints
 * `view.framebuffer` at `write` so the next effect, or tonemapping, reads this
 * one's output. Call from a compute system in the post-color seam (`after: [MainPassSystem]`, scene effects
 * `before: [OverlaySystem]`, overlays `after: [OverlaySystem]`): bind `read` as input, `write` as the
 * storage output, dispatch once. `write` is always the pair slot `read` isn't, so two effects chain
 * (fog reads the offscreen → writes `a`; outline reads `a` → writes `b`) and `read` is never `write`. The
 * renderer resets `view.framebuffer` to the offscreen each frame, so the chain restarts every frame.
 */
export function sceneTransform(
    world: World,
    view: View,
    eid: number,
): { read: GPUTextureView; write: GPUTextureView } {
    const _viewResources = world.resource(viewResourcesKey);

    const read = view.framebuffer;
    if (!read) throw new Error("sceneTransform: view has no framebuffer");
    let pair = _viewResources.scratch.get(eid);
    if (!pair || pair.w !== view.width || pair.h !== view.height) {
        pair?.a?.texture.destroy();
        pair?.b?.texture.destroy();
        pair = { a: null, b: null, w: view.width, h: view.height };
        _viewResources.scratch.set(eid, pair);
    }
    // write to whichever slot isn't the current read (first call read=offscreen → `a`; second read=`a` → `b`)
    const slot: "a" | "b" = read === pair.a?.view ? "b" : "a";
    const scratch = (pair[slot] ??= scratchTexture(world, eid, slot, view.width, view.height));
    view.framebuffer = scratch.view;
    view.framebufferFormat = SCENE_SCRATCH_FORMAT;
    return { read, write: scratch.view };
}

// free one camera's scene-transform scratch pair (on detach). Safe on cameras that never allocated one
function releaseScratch(world: World, eid: number): void {
    const _viewResources = world.resource(viewResourcesKey);

    const pair = _viewResources.scratch.get(eid);
    pair?.a?.texture.destroy();
    pair?.b?.texture.destroy();
    _viewResources.scratch.delete(eid);
}

/** free every offscreen target (on render teardown / HMR re-init) */
export function clearOffscreens(world: World): void {
    const _viewResources = world.resource(viewResourcesKey);

    for (const o of _viewResources.offscreen.values()) o.texture.destroy();
    _viewResources.offscreen.clear();
}

/** free every scene-transform scratch pair (on render teardown / HMR re-init) */
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
    // claim the canvas for exactly one camera. A second unbound camera (a multi-view scene's extra camera
    // alongside an explicitly-attached viewport camera) must not also grab it — two cameras on one
    // context each call getCurrentTexture per frame, and the second destroys the first's swapchain texture.
    for (const view of _views.values()) if (view.canvas === canvas) return undefined;
    attachCanvas(eid, canvas, world);
    return _views.get(eid);
}
