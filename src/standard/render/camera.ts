import { f32, invert, multiply, orthographic, perspective, sparse, u32 } from "../../engine";
import { composeTransform, Transform } from "../transforms";

/**
 * a camera's projection model: `Perspective` (fov-based, the default) or `Orthographic` (size-based).
 * Set via the `mode` field on the `camera` attribute (`camera="mode: perspective"` / `camera="mode: orthographic"`).
 */
export const CameraMode = {
    Perspective: 0,
    Orthographic: 1,
} as const;

/**
 * camera component. Pose comes from {@link Transform} (looks down its local -Z). A lone camera
 * auto-binds to the first `<canvas>` in the document, so the single-view case needs no wiring;
 * multi-view (or a dynamically-created canvas) binds each camera explicitly via `attachCanvas`
 * from `render`. `clearColor` is hex sRGB-encoded (e.g. `0x5cbfbf`); sear unpacks to linear
 * when recording the camera's render pass
 *
 * @example
 * ```
 * <a camera="mode: perspective; fov: 60; clear-color: 0x5cbfbf" transform="pos: 4 3 4" />
 * ```
 */
export const Camera = {
    /** perspective (0) or orthographic (1) projection, set via the `mode` scene attribute */
    mode: sparse(u32),
    /** field of view in degrees (perspective mode) */
    fov: sparse(f32),
    /** near plane distance */
    near: sparse(f32),
    /** far plane distance */
    far: sparse(f32),
    /** view size in world units (orthographic mode) */
    size: sparse(f32),
    /** render target color as sRGB-encoded hex (e.g. 0x5cbfbf) */
    clearColor: sparse(u32),
    /** antialiasing: 1 = 4× MSAA (default), 0 = off (single-sample, crisp, for a pixel-art look) */
    antialias: sparse(u32),
};

/**
 * render this camera at a fixed low resolution and scale it up to fill the canvas, crisp not blurred.
 * Without it the view renders at the canvas backing size (the global pixelRatio policy). `0` on an axis
 * derives it from the other to keep the canvas aspect, so `height: 360` alone renders 360 lines tall and
 * as wide as the canvas shape needs; set both for an exact (possibly aspect-distorting) target. Per camera,
 * so each canvas in a multi-view scene pins its own. Pairs with {@link Camera} `antialias` off.
 *
 * @example
 * ```
 * <a camera resolution="height: 360" />
 * ```
 */
export const Resolution = {
    /** render width in pixels; 0 = derive from height to keep the canvas aspect */
    width: sparse(u32),
    /** render height in pixels; 0 = derive from width to keep the canvas aspect */
    height: sparse(u32),
};

const _proj = new Float32Array(16);
const _world = new Float32Array(16);
const _view = new Float32Array(16);
// the projection's scalar inputs, `[fov or size, aspect, near, far]`
const _lens = new Float64Array(4);

/**
 * compute viewProj for a camera entity bound to a `width` × `height` surface (the aspect is their ratio).
 * `viewOut` optionally receives the world→view matrix alone (the light cull
 * transforms world-space lights into cluster space with it)
 */
export function computeViewProj(
    eid: number,
    width: number,
    height: number,
    out: Float32Array,
    viewOut?: Float32Array,
): void {
    const ortho = Camera.mode.get(eid) === CameraMode.Orthographic;
    _lens[0] = ortho ? Camera.size.get(eid) : Camera.fov.get(eid);
    _lens[1] = width / height;
    _lens[2] = Camera.near.get(eid);
    _lens[3] = Camera.far.get(eid);
    const proj = ortho ? orthographic(_lens, _proj) : perspective(_lens, _proj);
    composeTransform(eid, _world);
    const view = invert(_world, _view);
    viewOut?.set(view);
    multiply(proj, view, out);
}
