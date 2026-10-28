import type { World } from "../../engine";
import {
    component,
    f32,
    i32,
    invertMat4,
    multiplyMat4,
    orthographic,
    perspective,
    u32,
} from "../../engine";
import { composeGlobalTransform } from "../transform";

/** Camera projection mode; defaults to Perspective. */
export const CameraMode = {
    Perspective: 0,
    Orthographic: 1,
} as const;

/** Blender-calibrated default exposure in EV100. */
export const EXPOSURE_EV100_BLENDER = 9.7;

/** Camera exposure in EV100; defaults to 9.7. */
export const Exposure = component(
    "Exposure",
    { ev100: f32 },
    { defaults: () => ({ ev100: EXPOSURE_EV100_BLENDER }) },
);

/**
 * Camera looking down TransformPlugin's fixed-tick GlobalTransform local -Z. A camera with no target
 * auto-binds the document's first canvas, a late-mounted canvas retried each frame; bind other targets
 * explicitly with attachCanvas or attachTexture before the camera's first frame.
 *
 * Cameras sharing a canvas or texture draw bottom to top by `order`, a tie warned once and broken by
 * eid, the lower eid below. Each camera renders its own image with its own depth, antialiasing, effects and
 * presentation. The bottom camera clears the target with its `clearColor`; every camera above it
 * starts from transparent black, its `clearColor` unused, and its presented image is blended over the
 * target by its premultiplied alpha, so pixels it draws nothing on keep what the cameras below left.
 * That alpha is coverage: a camera above another cannot add light to or multiply the cameras below,
 * so standard's AlphaMode.Add light over pixels it left empty is dropped, and AlphaMode.Multiply
 * darkens the cameras below toward black by its alpha.
 * `clearColor` is sRGB hex, decoded to linear when recording the render pass.
 */
export const Camera = component(
    "Camera",
    {
        /** a {@link CameraMode}: perspective (0) or orthographic (1) projection */
        mode: u32,
        /** field of view in degrees (perspective mode) */
        fov: f32,
        /** Near-plane distance in world units; may be negative for an orthographic camera. */
        near: f32,
        /** perspective frustum cull bound and clustered-light range; does not limit perspective depth precision */
        far: f32,
        /** view size in world units (orthographic mode) */
        size: f32,
        /** render target color as sRGB-encoded hex (e.g. 0x5cbfbf) */
        clearColor: u32,
        /** antialiasing: 1 = 4× MSAA (default), 0 = off (single-sample, crisp, for a pixel-art look) */
        antialias: u32,
        /** draw order among cameras sharing a target: higher draws later, on top (default 0) */
        order: i32,
    },
    {
        defaults: () => ({
            mode: CameraMode.Perspective,
            fov: 60,
            near: 0.1,
            far: 1000,
            size: 5,
            clearColor: 0x2e2b28,
            antialias: 1,
            order: 0,
        }),
    },
);

/**
 * Optional rectangle within a Camera's target, with a top-left origin. Position and size use whole CSS
 * pixels of a bound canvas, or pixels of an attached texture. Unlike Bevy's Camera::viewport these
 * are not physical canvas pixels, so pixelRatio does not move the rectangle. Clamped to the target
 * as Bevy's Viewport::clamp_to_size: an outside origin moves to the last pixel with size 1; a zero
 * target dimension gives zero. Each frame writes the clamped position and size back to this component.
 * Projection, culling, effects and presentation use the clamped size.
 * The bottom camera still clears the whole target to its clearColor.
 */
export const CameraViewport = component("CameraViewport", {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
});

/**
 * render this camera at a fixed low resolution and scale it up to fill the canvas, crisp not blurred.
 * Without it the view renders at the canvas backing size (the world's pixelRatio policy). `0` on an axis
 * derives it from the other to keep the canvas aspect, so `height: 360` alone renders 360 lines tall and
 * as wide as the canvas shape needs; set both for an exact (possibly aspect-distorting) target. Per camera,
 * so each canvas in a multi-view scene pins its own. On a shared canvas the bottom camera's Resolution
 * sizes the canvas; a camera above it renders at its own Resolution, or at the canvas size without one,
 * and is scaled nearest-neighbour onto the canvas. A texture target keeps its size for every camera.
 * Pairs with {@link Camera} `antialias` off.
 */
export const Resolution = component(
    "Resolution",
    {
        /** render width in pixels; 0 = derive from height to keep the canvas aspect */
        width: u32,
        /** render height in pixels; 0 = derive from width to keep the canvas aspect */
        height: u32,
    },
    {
        defaults: () => ({ width: 0, height: 0 }),
    },
);

const _proj = new Float32Array(16);
const _world = new Float32Array(16);
const _view = new Float32Array(16);

/** Writes the world→view matrix into caller-owned `out`. */
export function computeViewFromWorld(world: World, eid: number, out: Float32Array): void {
    composeGlobalTransform(world, eid, _world);
    out.set(invertMat4(_world, _view));
}

/**
 * compute the world→clip matrix for a camera entity. `aspect` is the bound surface's width / height.
 * `viewOut` optionally receives the world→view matrix and `projectionOut` the view→clip matrix.
 */
export function computeClipFromWorld(
    world: World,
    eid: number,
    aspect: number,
    out: Float32Array,
    viewOut?: Float32Array,
    projectionOut?: Float32Array,
): void {
    const near = world.storage(Camera).near.get(eid);
    const far = world.storage(Camera).far.get(eid);
    const proj =
        world.storage(Camera).mode.get(eid) === CameraMode.Orthographic
            ? orthographic(world.storage(Camera).size.get(eid), aspect, near, far, _proj)
            : perspective(world.storage(Camera).fov.get(eid), aspect, near, _proj);
    composeGlobalTransform(world, eid, _world);
    const view = invertMat4(_world, _view);
    viewOut?.set(view);
    projectionOut?.set(proj);
    multiplyMat4(proj, view, out);
}
