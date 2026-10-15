import type { World } from "../../engine";
import {
    component,
    f32,
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
 * Camera looking down TransformPlugin's fixed-tick GlobalTransform local -Z. A lone camera auto-binds
 * the first canvas. Bind multi-view cameras explicitly with attachCanvas; a late-mounted canvas is
 * retried each frame. `clearColor` is sRGB hex, decoded to linear when recording the render pass.
 */
export const Camera = component(
    "Camera",
    {
        /** a {@link CameraMode}: perspective (0) or orthographic (1) projection */
        mode: u32,
        /** field of view in degrees (perspective mode) */
        fov: f32,
        /** Near-plane distance in world units. */
        near: f32,
        /** perspective frustum cull bound and clustered-light range; does not limit perspective depth precision */
        far: f32,
        /** view size in world units (orthographic mode) */
        size: f32,
        /** render target color as sRGB-encoded hex (e.g. 0x5cbfbf) */
        clearColor: u32,
        /** antialiasing: 1 = 4× MSAA (default), 0 = off (single-sample, crisp, for a pixel-art look) */
        antialias: u32,
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
        }),
    },
);

/**
 * render this camera at a fixed low resolution and scale it up to fill the canvas, crisp not blurred.
 * Without it the view renders at the canvas backing size (the world's pixelRatio policy). `0` on an axis
 * derives it from the other to keep the canvas aspect, so `height: 360` alone renders 360 lines tall and
 * as wide as the canvas shape needs; set both for an exact (possibly aspect-distorting) target. Per camera,
 * so each canvas in a multi-view scene pins its own. Pairs with {@link Camera} `antialias` off.
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
 * compute viewProj for a camera entity. `aspect` is the bound surface's width / height.
 * `viewOut` optionally receives the world→view matrix alone (the light cull
 * transforms world-space lights into view space with it)
 */
export function computeViewProj(
    world: World,
    eid: number,
    aspect: number,
    out: Float32Array,
    viewOut?: Float32Array,
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
    multiplyMat4(proj, view, out);
}
