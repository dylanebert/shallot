import { Compute, pixelRatio, type State } from "../../engine";
import { Camera, Resolution } from "./camera";
import { Render } from "./render";

/**
 * dynamic-offset uniform stride. WebGPU `minUniformBufferOffsetAlignment` ≥ 256
 * forces this even though only the leading bytes carry data
 */
export const VIEW_STRIDE = 256;

/**
 * max **shading** views (presenting cameras) per frame. A shading slot carries the clustered-light
 * substrate (its froxel AABB grid + light grid are ~140 KB each), so this cap stays small.
 * `BeginFrameSystem` assigns shading views the low slots `[0, MAX_VIEWS)`; depth-only views (a
 * shadow light's off-screen camera) get the slots above, out of {@link MAX_SLOTS}, and never
 * allocate cluster state
 */
export const MAX_VIEWS = 8;

/** total view slots per frame: shading cameras + depth-only views (the sun's light camera + the
 * point-shadow member-compaction "union" camera). Sizes only the cheap per-slot state
 * ({@link Render.viewBuffer}, `Render.cullVolumes`), so it's generous */
export const MAX_SLOTS = 64;

export const VIEW_UNIFORM_SIZE = VIEW_STRIDE * MAX_SLOTS;

/** the per-camera `View` UBO's WGSL struct, spliced by sear for every surface and by any relocatable
 * screen-space consumer (the fog march) that binds `view`; layout mirrors {@link VIEW_BYTES}. */
export const VIEW_STRUCT_WGSL = /* wgsl */ `
struct View {
    viewProj: mat4x4<f32>,
    resolution: vec2<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    cluster: vec4<f32>,
    eye: vec4<f32>,
    invViewProj: mat4x4<f32>,
}`;

/**
 * the byte size of the {@link View} uniform a surface statically reads: `mat4` (64) + `vec2`
 * resolution (8, padded to 16 by the vec4 that follows) + two `vec4` camera-basis columns (right at
 * byte 80, up at 96: the camera's normalized world-space right/up, packed by `BeginFrameSystem`;
 * forward derives as `-cross(right, up)`) + the `cluster` vec4 at 112 (near, far, perspective flag,
 * view slot: what sear's FS needs to map a fragment to its froxel cluster and index the slot-major
 * light grid) + the `eye` vec4 at 128 (the camera's world-space position, for view-dependent shading
 * (specular, fresnel, fog)). Billboard-shaped surfaces orient quads from `right`/`up`. Note
 * the shadow light camera packs through the same path, so a billboard in the shadow pass faces the
 * light (Godot-consistent). Then `invViewProj` at byte 144 (the inverse of `viewProj`). A screen-space
 * pass (fog / volumetrics) reconstructs a fragment's world position from its depth: `ndc(uv, depth)`
 * → `invViewProj` → world. The renderer's bind-group layout declares this as the View binding's
 * `minBindingSize`; the bound range is the full {@link VIEW_STRIDE} slot. Bump this in lockstep
 * with {@link VIEW_STRUCT_WGSL} when the struct grows.
 */
export const VIEW_BYTES = 208;

/**
 * linear→sRGB encode for a compute composite writing `view.present`. The swapchain is a base-format
 * storage view (a storage view can't be sRGB), so the composite encodes the transfer the hardware would
 * apply on a render-attachment write. One source of truth so every composite (glaze + consumer-fused)
 * agrees, and the present gamma can't drift between them.
 */
export const LINEAR_TO_SRGB_WGSL = /* wgsl */ `
fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}`;

/**
 * a camera's per-frame view state. `framebuffer` + `present` + `slot` are set by `BeginFrameSystem`
 * each frame and read by the renderers. `slot` is the camera's index into the packed View UBO; use it
 * as the dynamic offset (`slot * VIEW_STRIDE`) when binding. `framebuffer` is the per-camera **offscreen**
 * scene-color target the renderer draws into (sear resolves its MSAA color into it; the `Custom` renderer
 * draws straight into it single-sample): sampleable (`TEXTURE_BINDING`), in `Render.format`, sized to
 * the view; a composite `textureLoad`s it and writes the result into `present`. `present` is the swapchain
 * backbuffer, as a **storage** view in the base canvas format (not sRGB). The only target the user ever
 * sees. A compute composite (glaze, or a consumer's own fused pass) `textureStore`s into it, encoding
 * linear→sRGB itself since a storage view can't be sRGB. The split from `framebuffer` exists so postfx
 * has a rendered color to read: writing the swapchain in place leaves nothing to read back. `depth` + `tag`
 * are the renderer's opt-in **prepass lanes**, each gated by a per-camera marker (sear's `Depth` / `Tag`).
 * `depth` is the camera's single-sample depth, *stored* + published by the prepass only when the camera
 * carries `Depth`, read by screen-space consumers (AO, fog). `null` otherwise (a tag-only camera tests
 * depth but discards it). `tag` is the screen-space surface-tag (object-id) target, written by the same
 * prepass for a camera carrying `Tag`: the front-most opaque fragment's surface-authored tag per pixel
 * (`TAG_NONE` where no surface owns it, defaulting to the entity's eid for an instanced surface). It's
 * the `GPUTexture` (not a view, unlike `depth` / `framebuffer`) because its consumers need the texture:
 * a hover readback `copyTextureToBuffer`s the cursor pixel, and a view can't be turned back into a
 * texture; an outline pass `createView`s it to sample. `null` until the prepass has drawn the camera. A
 * canvas-bound view (`attachCanvas`) renders to that canvas; a canvas-less view (`attachView`) has
 * no `canvas` / `context` / `observer` and a null `framebuffer` / `present`. It still takes a cull slot
 * (a shadow light's off-screen camera renders to its own target, not the screen). Every view
 * frustum-culls from its viewProj: cameras, the sun, and each point/spot shadow combo's depth view.
 * @expand
 */
export interface View {
    canvas: HTMLCanvasElement | null;
    context: GPUCanvasContext | null;
    // the render backing-store size (device px). Derived each frame by `sizeView` from the display size
    // below + the camera's `Resolution` pin (or the global pixelRatio). Every consumer — offscreen,
    // present, glaze, the cluster grid — reads these, so a low-res pin flows through by sizing them alone
    width: number;
    height: number;
    // the canvas CSS display size (px), cached by the ResizeObserver. The backing above derives from it,
    // so a runtime `Resolution` edit re-sizes the view without waiting on a resize event
    clientWidth: number;
    clientHeight: number;
    framebuffer: GPUTextureView | null;
    present: GPUTextureView | null;
    depth: GPUTextureView | null;
    tag: GPUTexture | null;
    slot: number;
    observer: ResizeObserver | null;
    // the create-stamp of the camera this view was last packed for (`state.stamp`, `0` until first packed).
    // Membership catches a plain despawn; the stamp catches a same-update destroy+create realias that keeps
    // Camera membership, so a recycled eid doesn't inherit the dead camera's canvas. See {@link pruneViews}
    stamp: number;
}

/** every camera with a view, keyed by eid: canvas-bound ({@link attachCanvas}) or off-screen ({@link attachView}) */
export const Views: Map<number, View> = new Map();

// canvas → the State that last bound it, for the dev-only rebuild guard below. WeakMap so a collected
// canvas drops its entry; never populated in production (the guard is dev-gated).
const _canvasOwners: WeakMap<HTMLCanvasElement, State> = new WeakMap();

// read `import.meta.env.DEV` typeof-safely: the engine is bundled by arbitrary consumer bundlers, and a
// bare `import.meta.env.DEV` throws where `import.meta.env` is undefined (non-vite). Optional-chained,
// wrapped so an exotic `import.meta` shape can't take down attachCanvas. Exported as a test seam (pins the
// false-not-throw contract off a vite build) — not on the `render/core` barrel.
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
 * State — an app rebuilt without disposing the prior one (the leak class `State.onDispose` closes), then
 * record the new owner. Two apps on distinct canvases stay silent; a proper dispose flips the prior owner's
 * `disposed`, so a later rebind is silent too. Internal + a test seam — not on the `render/core` barrel.
 */
export function trackCanvasOwner(canvas: HTMLCanvasElement, state: State): void {
    const prior = _canvasOwners.get(canvas);
    if (prior && prior !== state && !prior.disposed) {
        console.warn(
            "attachCanvas: canvas already bound to a live State — did an app rebuild without disposing the previous one? dispose it first (app.dispose() / state.dispose())",
        );
    }
    _canvasOwners.set(canvas, state);
}

/**
 * bind a canvas to a camera entity, 1:1: each camera owns one canvas. Pass `state` to arm the dev-only
 * rebuild guard ({@link trackCanvasOwner}) — it only warns for callers that pass it, so a multi-view app
 * binding its cameras directly should pass `state` to catch a rebuild that skipped `dispose`.
 */
export function attachCanvas(eid: number, canvas: HTMLCanvasElement, state?: State): void {
    if (!Compute.device) throw new Error("attachCanvas: RenderPlugin not initialized");
    if (!Render.format) throw new Error("attachCanvas: Render.format not set");
    if (Views.has(eid)) throw new Error(`attachCanvas: eid ${eid} already bound`);

    const context = canvas.getContext("webgpu") as unknown as GPUCanvasContext | null;
    if (!context) throw new Error("attachCanvas: WebGPU canvas context unavailable");

    // record ownership only after the attach is validated — a failed attach must not claim the canvas, or a
    // later legitimate attach from a different live State warns spuriously.
    if (state && devEnabled()) trackCanvasOwner(canvas, state);

    const linearFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: Compute.device,
        format: linearFormat,
        alphaMode: "premultiplied",
        // the present path is a compute composite writing the swapchain via textureStore, so it needs
        // STORAGE_BINDING and the base (non-srgb) format — a storage view can't be sRGB. No sRGB
        // viewFormat: the composite encodes sRGB itself. RENDER_ATTACHMENT keeps it a presentable surface.
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
    });

    const rect = canvas.getBoundingClientRect();
    const view: View = {
        canvas,
        context,
        width: 0,
        height: 0,
        clientWidth: rect.width,
        clientHeight: rect.height,
        framebuffer: null,
        present: null,
        depth: null,
        tag: null,
        slot: 0,
        observer: null!,
        stamp: 0,
    };
    // the observer is a pure sensor — it caches the display size and `sizeView` derives the backing from
    // it each frame. Splitting it this way lets a runtime `Resolution` edit re-size (the observer never
    // fires for that) and moves the backing write to frame start, off the async resize callback.
    view.observer = new ResizeObserver(() => {
        const r = canvas.getBoundingClientRect();
        view.clientWidth = r.width;
        view.clientHeight = r.height;
    });
    view.observer.observe(canvas);
    Views.set(eid, view);
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

/**
 * size a canvas-bound view's backing store from its cached display size + its {@link Resolution} pin (the
 * global pixelRatio when absent), and set the nearest-neighbor upscale. {@link BeginFrameSystem} calls it
 * per camera each frame after binding, so a `Resolution` edit and a canvas resize both re-size here. A
 * no-op for a canvas-less (off-screen) view, which sizes its own target. The Resolution read is membership-
 * gated: a recycled eid's stale field value never leaks into a camera that carries no pin.
 */
export function sizeView(state: State, eid: number, view: View): void {
    const canvas = view.canvas;
    if (!canvas || view.clientWidth <= 0 || view.clientHeight <= 0) return;
    const dpr = (typeof window === "undefined" ? 1 : window.devicePixelRatio) || 1;
    const ratio = pixelRatio === "auto" ? Math.min(Math.max(dpr, 1), 2) : pixelRatio;
    const pinned = state.has(eid, Resolution);
    const resW = pinned ? Resolution.width.get(eid) | 0 : 0;
    const resH = pinned ? Resolution.height.get(eid) | 0 : 0;
    const { w, h, pixelated } = backingSize(resW, resH, view.clientWidth, view.clientHeight, ratio);
    const ir = pixelated ? "pixelated" : "auto";
    if (canvas.style.imageRendering !== ir) canvas.style.imageRendering = ir;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    view.width = w;
    view.height = h;
}

/**
 * register a camera entity as a canvas-less, off-screen view: it takes a cull slot and its viewProj
 * is packed from its `Camera` + `Transform` like any camera, but it has no canvas and `framebuffer`
 * stays null; the caller renders it to its own target. A directional shadow's light-space camera is
 * one (so is each point/spot shadow combo's depth view). 1:1 per eid, like {@link attachCanvas}; the
 * caller draws to the `slot * VIEW_STRIDE` offset. Frustum-culls from its viewProj like any camera.
 */
export function attachView(eid: number): void {
    if (Views.has(eid)) throw new Error(`attachView: eid ${eid} already has a view`);
    Views.set(eid, {
        canvas: null,
        context: null,
        // square (aspect 1) — an off-screen view's own target sets the resolution; only the aspect
        // feeds `computeViewProj`, so a shadow's ortho box stays square
        width: 1,
        height: 1,
        clientWidth: 0,
        clientHeight: 0,
        framebuffer: null,
        present: null,
        depth: null,
        tag: null,
        slot: 0,
        observer: null,
        stamp: 0,
    });
}

/** release a camera's view (canvas-bound or off-screen). Safe to call on unbound eids */
export function detachCanvas(eid: number): void {
    Views.get(eid)?.observer?.disconnect();
    Views.delete(eid);
    releaseOffscreen(eid);
    releaseScratch(eid);
}

/**
 * drop the auto-bind's inverse: a View whose camera despawned, or whose eid was recycled to a new camera.
 * `state.has(eid, Camera)` catches a plain despawn (the destroy dropped Camera); the create-stamp catches a
 * same-update destroy+create realias that *keeps* Camera membership — without it the recycled eid keeps the
 * dead camera's View (its canvas + leaked ResizeObserver, re-binding to the wrong canvas). A View not yet
 * packed (`stamp` 0) skips the stamp arm for that frame; the next pack records its camera's live stamp.
 * {@link BeginFrameSystem} calls it at frame start, before binding.
 */
export function pruneViews(state: State): void {
    for (const [eid, view] of Views) {
        if (!state.has(eid, Camera) || (view.stamp !== 0 && state.stamp(eid) !== view.stamp))
            detachCanvas(eid);
    }
}

// per-camera offscreen scene-color target — the `view.framebuffer` a renderer draws (or resolves)
// into and glaze composites to the swapchain. `Render.format` is rg11b10ufloat (HDR): a renderer writes
// linear and glaze's `textureLoad` reads it linear, keeping radiance >1 alive for the tonemap. Sized to
// the view, recreated on resize; one per camera so multi-view never last-camera-wins a single shared texture
const _offscreen = new Map<
    number,
    { texture: GPUTexture; view: GPUTextureView; w: number; h: number }
>();

/** the camera's offscreen color target, (re)allocated to the view size. Renderer-agnostic: sear's
 * MSAA resolve and the `Custom` single-sample draw both target it; {@link BeginFrameSystem} sets it on
 * `view.framebuffer` each frame */
export function offscreen(eid: number, w: number, h: number): GPUTextureView {
    const cached = _offscreen.get(eid);
    if (cached && cached.w === w && cached.h === h) return cached.view;
    cached?.texture.destroy();
    const texture = Compute.device.createTexture({
        label: `kitchen-offscreen-${eid}`,
        size: { width: w, height: h },
        format: Render.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const view = texture.createView();
    _offscreen.set(eid, { texture, view, w, h });
    return view;
}

// free one camera's offscreen target (on detach). Safe on cameras that never allocated one
function releaseOffscreen(eid: number): void {
    _offscreen.get(eid)?.texture.destroy();
    _offscreen.delete(eid);
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
const _scratch = new Map<number, { a: Scratch | null; b: Scratch | null; w: number; h: number }>();

function scratchTexture(eid: number, slot: "a" | "b", w: number, h: number): Scratch {
    const texture = Compute.device.createTexture({
        label: `scene-scratch-${eid}-${slot}`,
        size: { width: w, height: h },
        format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    return { texture, view: texture.createView() };
}

/**
 * the scene-transform seam: redirect a camera's scene color through a postfx compute effect. Returns
 * `read` (the current `view.framebuffer`: the renderer's resolved scene, or the prior effect's output) and
 * `write` (a lazily-allocated scratch, the *other* half of the ping-pong pair from `read`), and repoints
 * `view.framebuffer` at `write` so the next effect, or the compositor ({@link GlazeSystem}), reads this
 * one's output. Call from a compute system in the post-color seam (`after: [ColorSystem]`, scene-transforms
 * `before: [OverlaySystem]`, overlays `after: [OverlaySystem]`): bind `read` as input, `write` as the
 * storage output, dispatch once. `write` is always the pair slot `read` isn't, so two effects chain
 * (fog reads the offscreen → writes `a`; outline reads `a` → writes `b`) and `read` is never `write`. The
 * renderer resets `view.framebuffer` to the offscreen each frame, so the chain restarts every frame.
 */
export function sceneTransform(
    view: View,
    eid: number,
): { read: GPUTextureView; write: GPUTextureView } {
    const read = view.framebuffer;
    if (!read) throw new Error("sceneTransform: view has no framebuffer");
    let pair = _scratch.get(eid);
    if (!pair || pair.w !== view.width || pair.h !== view.height) {
        pair?.a?.texture.destroy();
        pair?.b?.texture.destroy();
        pair = { a: null, b: null, w: view.width, h: view.height };
        _scratch.set(eid, pair);
    }
    // write to whichever slot isn't the current read (first call read=offscreen → `a`; second read=`a` → `b`)
    const slot: "a" | "b" = read === pair.a?.view ? "b" : "a";
    const scratch = (pair[slot] ??= scratchTexture(eid, slot, view.width, view.height));
    view.framebuffer = scratch.view;
    return { read, write: scratch.view };
}

// free one camera's scene-transform scratch pair (on detach). Safe on cameras that never allocated one
function releaseScratch(eid: number): void {
    const pair = _scratch.get(eid);
    pair?.a?.texture.destroy();
    pair?.b?.texture.destroy();
    _scratch.delete(eid);
}

/** free every offscreen target (on render teardown / HMR re-init) */
export function clearOffscreens(): void {
    for (const o of _offscreen.values()) o.texture.destroy();
    _offscreen.clear();
}

/** free every scene-transform scratch pair (on render teardown / HMR re-init) */
export function clearScratch(): void {
    for (const p of _scratch.values()) {
        p.a?.texture.destroy();
        p.b?.texture.destroy();
    }
    _scratch.clear();
}

/**
 * auto-bind a camera to the first `<canvas>` in the document, idempotently. The zero-config
 * single-view path. A no-op (returns undefined) headless or until a canvas mounts;
 * `BeginFrameSystem` retries each frame, so a late-mounted canvas binds when it appears.
 * Multi-view binds each camera explicitly via {@link attachCanvas} before its first frame.
 */
export function bindCamera(eid: number, state?: State): View | undefined {
    const existing = Views.get(eid);
    if (existing) return existing;
    if (typeof document === "undefined") return undefined;
    const canvas = document.querySelector("canvas");
    if (!(canvas instanceof HTMLCanvasElement)) return undefined;
    // claim the canvas for exactly one camera. A second unbound camera (a multi-view scene's extra camera
    // alongside an explicitly-attached viewport camera) must not also grab it — two cameras on one
    // context each call getCurrentTexture per frame, and the second destroys the first's swapchain texture.
    for (const view of Views.values()) if (view.canvas === canvas) return undefined;
    attachCanvas(eid, canvas, state);
    return Views.get(eid);
}
