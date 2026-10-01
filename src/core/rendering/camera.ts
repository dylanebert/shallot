import type { World } from "../../engine";
import {
    composeGlobalTransform,
    f32,
    invertMat4,
    multiplyMat4,
    orthographic,
    perspective,
    u32,
} from "../../engine";

/**
 * a camera's projection model: `Perspective` (fov-based, the default) or `Orthographic` (size-based).
 * Stored in {@link Camera} `mode`: `world.add(eid, Camera, { mode: CameraMode.Orthographic })`.
 */
export const CameraMode = {
    Perspective: 0,
    Orthographic: 1,
} as const;

/**
 * camera component. Placement comes from the engine's fixed-tick GlobalTransform (looks down its local -Z). A lone camera
 * auto-binds to the first `<canvas>` in the document, so the single-view case needs no wiring;
 * multi-view (or a dynamically-created canvas) binds each camera explicitly via `attachCanvas`
 * from `render`. `clearColor` is hex sRGB-encoded (e.g. `0x5cbfbf`); sear unpacks to linear
 * when recording the camera's render pass
 *
 * @example
 * ```
 * const camera = world.create();
 * world.add(camera, Camera, { mode: CameraMode.Perspective, fov: 60, clearColor: 0x5cbfbf });
 * world.add(camera, Transform, { translation: [4, 3, 4, 0] });
 * ```
 */
export const Camera = {
    /** a {@link CameraMode}: perspective (0) or orthographic (1) projection */
    mode: u32,
    /** field of view in degrees (perspective mode) */
    fov: f32,
    /** near plane distance */
    near: f32,
    /** far plane distance */
    far: f32,
    /** view size in world units (orthographic mode) */
    size: f32,
    /** render target color as sRGB-encoded hex (e.g. 0x5cbfbf) */
    clearColor: u32,
    /** antialiasing: 1 = 4× MSAA (default), 0 = off (single-sample, crisp, for a pixel-art look) */
    antialias: u32,
};

/**
 * render this camera at a fixed low resolution and scale it up to fill the canvas, crisp not blurred.
 * Without it the view renders at the canvas backing size (the world's pixelRatio policy). `0` on an axis
 * derives it from the other to keep the canvas aspect, so `height: 360` alone renders 360 lines tall and
 * as wide as the canvas shape needs; set both for an exact (possibly aspect-distorting) target. Per camera,
 * so each canvas in a multi-view scene pins its own. Pairs with {@link Camera} `antialias` off.
 *
 * @example
 * ```
 * const camera = world.create();
 * world.add(camera, Camera, { antialias: 0 });
 * world.add(camera, Resolution, { height: 360 });
 * ```
 */
export const Resolution = {
    /** render width in pixels; 0 = derive from height to keep the canvas aspect */
    width: u32,
    /** render height in pixels; 0 = derive from width to keep the canvas aspect */
    height: u32,
};

const _proj = new Float32Array(16);
const _world = new Float32Array(16);
const _view = new Float32Array(16);

/**
 * compute viewProj for a camera entity. `aspect` is the bound surface's width / height.
 * `viewOut` optionally receives the world→view matrix alone (the light cull
 * transforms world-space lights into cluster space with it)
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
            : perspective(world.storage(Camera).fov.get(eid), aspect, near, far, _proj);
    composeGlobalTransform(world, eid, _world);
    const view = invertMat4(_world, _view);
    viewOut?.set(view);
    multiplyMat4(proj, view, out);
}
