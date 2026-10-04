// Shadow cameras share core's view/projection and frustum packing. They are created lazily for lights
// with shadowMapsEnabled; the mesh producer culls into their depth-only view slots.

import {
    attachView,
    Camera,
    CameraMode,
    DirectionalLight,
    detachCanvas,
    MAX_SLOTS,
    MAX_VIEWS,
    PointLight,
    SpotLight,
    Views,
} from "../../core/rendering";
import {
    composeGlobalTransform,
    GlobalTransform,
    lookAt,
    lookAtRotation,
    multiplyMat4,
    orthographic,
    perspective,
    type Resource,
    Transform,
    type World,
} from "../../engine";

/** the directional shadow's cascade ceiling: each cascade takes one of the depth view slots reserved out
 * of the point-shadow combo budget, so the count can't exceed it. Four is the three.js / Bevy default. */
export const MAX_CASCADES = 4;

/**
 * the directional shadow map size, Bevy's `DirectionalLightShadowMap`: `size` is the side in pixels of each
 * cascade's square tile (default 2048), clamped to [256, 4096] and snapped to a power of two. The tiles share
 * one depth32float atlas `ceil(√numCascades)` tiles on a side, so 4096 with more than one cascade allocates
 * an 8192² atlas (256 MiB). It is read when the app warms and when the sun first casts, and is not changed
 * after; set it in `AppConfig.setup` or a plugin's `initialize`, e.g.
 * `world.resource(DirectionalLightShadowMap).size = 1024`. Each world owns its own. The cascade split itself
 * lives on the light ({@link DirectionalLight}'s `numCascades`, `firstCascadeFarBound`, `overlapProportion`
 * and `maximumDistance`); point and spot tiles are sized from {@link PointShadows}.
 */
export interface DirectionalLightShadowMap {
    size: number;
}

export const DirectionalLightShadowMap: Resource<DirectionalLightShadowMap> = {
    create: () => ({ size: 2048 }),
};

/** a directional light's resolved cascade count: its `numCascades` clamped to [1, {@link MAX_CASCADES}] */
export function lightCascades(world: World, light: number): number {
    const n = world.storage(DirectionalLight).numCascades.get(light);
    return Math.min(Math.max(Math.round(n), 1), MAX_CASCADES);
}

/** the world's resolved per-cascade shadow-map size: {@link DirectionalLightShadowMap} `size` clamped to
 * [256, 4096] and snapped to a power of two (so the atlas + the per-cascade tile size + the texel-snap grid
 * all stay aligned). */
export function sunResolution(world: World): number {
    const s = Math.min(
        Math.max(Math.round(world.resource(DirectionalLightShadowMap).size), 256),
        4096,
    );
    return Math.min(4096, 1 << Math.round(Math.log2(s)));
}

// a light box placement: the light eye, its look target and up hint, the ortho half-extent (`cover`) and
// the near-extended box depth. The fits write one in place.
type LightFit = {
    eye: Float64Array;
    focus: Float64Array;
    up: Float64Array;
    /** `[cover, depth]`: the ortho half-extent and the near-extended box depth */
    extent: Float64Array;
};

// from a fit center + ortho half-extent, place the light eye back toward the sun and snap it onto the light's
// texel grid (the plane ⊥ the sun) so the shadow doesn't crawl as the box moves. `margin` extends the box's
// near plane *toward the light* past the fit (Bevy pushes the directional near plane to ∞, three.js's finite
// `lightMargin`): a tight cascade's near plane would otherwise clip a caster's own occluder that sits between
// the light and the slice, so its depth map never records it and the receiver reads "lit" (the boundary
// bleed). The eye moves back by `cover + margin`, the far stays at `center + dir·cover`, so the box depth is
// `2·cover + margin`. `margin` is along the sun, ⊥ the snap plane, so the texel snap is unaffected. Shared by
// the per-cascade slice fit ({@link cascadeFit}) and the ortho footprint fit ({@link orthoFootprintFit}).
// It writes the placement into `out`.
function placeFromCenter(
    cenX: number,
    cenY: number,
    cenZ: number,
    cover: number,
    dir: ArrayLike<number>,
    resolution: number,
    margin: number,
    out: LightFit,
): LightFit {
    let dx = dir[0];
    let dy = dir[1];
    let dz = dir[2];
    const dl = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    dx /= dl;
    dy /= dl;
    dz /= dl;

    // an orthonormal basis in the plane perpendicular to the sun, to snap the eye onto. up0 avoids the
    // degenerate parallel case when the sun points near-straight-down
    const upY = Math.abs(dy) > 0.99 ? 0 : 1;
    const upZ = Math.abs(dy) > 0.99 ? 1 : 0;
    let rx = upY * dz - upZ * dy;
    let ry = upZ * dx;
    let rz = -upY * dx;
    const rl = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
    rx /= rl;
    ry /= rl;
    rz /= rl;
    const ux = dy * rz - dz * ry;
    const uy = dz * rx - dx * rz;
    const uz = dx * ry - dy * rx;

    const back = cover + margin;
    let ex = cenX - dx * back;
    let ey = cenY - dy * back;
    let ez = cenZ - dz * back;

    const texel = (2 * cover) / resolution;
    const er = ex * rx + ey * ry + ez * rz;
    const eu = ex * ux + ey * uy + ez * uz;
    const sr = Math.round(er / texel) * texel - er;
    const su = Math.round(eu / texel) * texel - eu;
    ex += sr * rx + su * ux;
    ey += sr * ry + su * uy;
    ez += sr * rz + su * uz;

    out.eye[0] = ex;
    out.eye[1] = ey;
    out.eye[2] = ez;
    out.focus[0] = ex + dx * back;
    out.focus[1] = ey + dy * back;
    out.focus[2] = ez + dz * back;
    out.up[0] = 0;
    out.up[1] = upY;
    out.up[2] = upZ;
    out.extent[0] = cover;
    out.extent[1] = 2 * cover + margin;
    return out;
}

/**
 * the N cascade far-bounds out to `far`, Bevy's `calculate_cascade_bounds`: the first ends at `first` and the
 * rest are spaced exponentially to `far` (one cascade ends at `far`). `first` must be below `far`
 * ({@link updateCascades} refuses one that is not), and is floored just above 0.
 * Cascade `i` covers `[splits[i-1], splits[i]]` (`splits[-1]` = the camera near implicitly); the last bound is
 * `far` exactly. Writes the `n` bounds into `out`; otherwise pure. The receiver selects a cascade by these
 * bounds (Bevy `get_cascade_index`), so they're the same numbers the fit and the FS read.
 */
export function cascadeSplits(
    far: number,
    n: number,
    first: number,
    out: Float64Array,
): Float64Array {
    if (n === 1) {
        out[0] = far;
        return out;
    }
    const nearest = Math.max(first, 1e-6);
    const base = (far / nearest) ** (1 / (n - 1));
    for (let i = 0; i < n - 1; i++) out[i] = nearest * base ** i;
    out[n - 1] = far;
    return out;
}

/**
 * the snapped light-camera placement for one CSM cascade: fit the main camera's frustum **slice** between
 * `nearSplit` and `farSplit` (not the ground footprint {@link orthoFootprintFit} uses) and place the ortho
 * light box around it, eye back toward the sun and texel-snapped per cascade. The box is a **bounding
 * sphere** of the symmetric slice (its two extreme corners are the far-plane diagonal or the whole-slice
 * diagonal, whichever is longer, three.js CSM), so the `cover` is rotation-stable as the camera turns
 * (no per-frame size pumping). `margin` extends the box's near plane toward the light ({@link placeFromCenter})
 * so a tight cascade still captures occluders above its slice. Reads only the camera GlobalTransform + projection (the
 * perspective `fov` or the ortho `size`), never its own near/far. Writes the placement into `out`,
 * otherwise pure.
 */
export function cascadeFit(
    camWorld: Float32Array,
    mode: number,
    fov: number,
    size: number,
    aspect: number,
    nearSplit: number,
    farSplit: number,
    dir: ArrayLike<number>,
    resolution: number,
    margin: number,
    out: LightFit,
): LightFit {
    const px = camWorld[12];
    const py = camWorld[13];
    const pz = camWorld[14];
    // forward (local -Z), right (col 0), up (col 1), each normalized
    let fx = -camWorld[8];
    let fy = -camWorld[9];
    let fz = -camWorld[10];
    const fl = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
    fx /= fl;
    fy /= fl;
    fz /= fl;
    let rx = camWorld[0];
    let ry = camWorld[1];
    let rz = camWorld[2];
    const rl = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
    rx /= rl;
    ry /= rl;
    rz /= rl;
    let ux = camWorld[4];
    let uy = camWorld[5];
    let uz = camWorld[6];
    const ul = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1;
    ux /= ul;
    uy /= ul;
    uz /= ul;

    // the slice half-extents at the near and far split: a perspective camera's grow linearly (tan(fov/2)),
    // an ortho camera's are constant (its `size`); aspect widens the horizontal
    const tanH = mode === CameraMode.Orthographic ? 0 : Math.tan((fov * Math.PI) / 360);
    const farH = mode === CameraMode.Orthographic ? size : farSplit * tanH;
    const farW = farH * aspect;
    const nearH = mode === CameraMode.Orthographic ? size : nearSplit * tanH;
    const nearW = nearH * aspect;
    // the symmetric slice's bounding sphere: one extreme is the far (+right, +up) corner, the other the
    // opposite corner of whichever diagonal is longer (the far plane's, or the full slice's) — both lie on
    // the view axis
    const ax = px + fx * farSplit + rx * farW + ux * farH;
    const ay = py + fy * farSplit + ry * farW + uy * farH;
    const az = pz + fz * farSplit + rz * farW + uz * farH;
    const farX = px + fx * farSplit - rx * farW - ux * farH;
    const farY = py + fy * farSplit - ry * farW - uy * farH;
    const farZ = pz + fz * farSplit - rz * farW - uz * farH;
    const nearX = px + fx * nearSplit - rx * nearW - ux * nearH;
    const nearY = py + fy * nearSplit - ry * nearW - uy * nearH;
    const nearZ = pz + fz * nearSplit - rz * nearW - uz * nearH;
    const nearDist = Math.sqrt(
        (ax - nearX) * (ax - nearX) + (ay - nearY) * (ay - nearY) + (az - nearZ) * (az - nearZ),
    );
    const farDist = Math.sqrt(
        (ax - farX) * (ax - farX) + (ay - farY) * (ay - farY) + (az - farZ) * (az - farZ),
    );
    const nearWins = nearDist > farDist;
    const bx = nearWins ? nearX : farX;
    const by = nearWins ? nearY : farY;
    const bz = nearWins ? nearZ : farZ;
    return placeFromCenter(
        (ax + bx) / 2,
        (ay + by) / 2,
        (az + bz) / 2,
        (nearWins ? nearDist : farDist) / 2,
        dir,
        resolution,
        margin,
        out,
    );
}

/**
 * the single-box light placement for an **orthographic** main camera, fit to its visible ground (y=0)
 * footprint. An ortho camera has uniform texel density, so depth cascades buy nothing (three.js / standard
 * practice special-cases ortho): one box covering the whole view beats N frustum slices, which for a camera
 * posed far from the scene don't even reach the ground its forward range targets. The center ray meets y=0 at
 * the look point; a screen-corner offset `v` slides its parallel ray's ground hit by `v − (v.y/fwd.y)·fwd`
 * (the extra travel as `v` changes the height it must fall), so the footprint corners are `center ± Kr ± Ku`
 * and the bounding radius is the longer half-diagonal. A camera not angled at the ground (`fwd.y ≥ −1e-3`)
 * has no convergent footprint, so it falls back to a forward-distance box (`center = pos + fwd·distance`,
 * `cover = distance`). `margin` extends the near plane toward the light ({@link placeFromCenter}). Writes
 * the placement into `out`, otherwise pure.
 */
export function orthoFootprintFit(
    camWorld: Float32Array,
    size: number,
    aspect: number,
    dir: ArrayLike<number>,
    distance: number,
    resolution: number,
    margin: number,
    out: LightFit,
): LightFit {
    const px = camWorld[12];
    const py = camWorld[13];
    const pz = camWorld[14];
    // forward (local -Z), right (col 0), up (col 1), each normalized
    let fx = -camWorld[8];
    let fy = -camWorld[9];
    let fz = -camWorld[10];
    const fl = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
    fx /= fl;
    fy /= fl;
    fz /= fl;

    let cenX: number;
    let cenY: number;
    let cenZ: number;
    let cover: number;
    if (fy < -1e-3) {
        let crx = camWorld[0];
        let cry = camWorld[1];
        let crz = camWorld[2];
        const crl = Math.sqrt(crx * crx + cry * cry + crz * crz) || 1;
        crx /= crl;
        cry /= crl;
        crz /= crl;
        let cux = camWorld[4];
        let cuy = camWorld[5];
        let cuz = camWorld[6];
        const cul = Math.sqrt(cux * cux + cuy * cuy + cuz * cuz) || 1;
        cux /= cul;
        cuy /= cul;
        cuz /= cul;

        const t0 = -py / fy;
        cenX = px + t0 * fx;
        cenY = 0;
        cenZ = pz + t0 * fz;
        const hw = size * aspect;
        const hh = size;
        const krx = hw * (crx - (cry / fy) * fx);
        const krz = hw * (crz - (cry / fy) * fz);
        const kux = hh * (cux - (cuy / fy) * fx);
        const kuz = hh * (cuz - (cuy / fy) * fz);
        const diagA = Math.sqrt((krx + kux) * (krx + kux) + (krz + kuz) * (krz + kuz));
        const diagB = Math.sqrt((krx - kux) * (krx - kux) + (krz - kuz) * (krz - kuz));
        cover = Math.max(diagA, diagB);
    } else {
        cenX = px + fx * distance;
        cenY = py + fy * distance;
        cenZ = pz + fz * distance;
        cover = distance;
    }

    return placeFromCenter(cenX, cenY, cenZ, cover, dir, resolution, margin, out);
}

// ---- CSM: the cascade combo-camera pool (the sun's analogue of the point combo pool) ----
//
// The single sun box is split into N depth slices along the camera's view range, each its own
// frustum-culled ortho depth view — a pooled off-screen camera the MeshInstance pack culls casters into through the
// same `cull → count → scan → scatter` spine every view uses (the sun joining the unified culled-combo
// path). Each cascade renders into a tile of a dedicated atlas (the fixed grid below — cascades are
// equal-resolution, so no importance sizing), the tile placement folded into its viewProj (`tileTransform`).
// standard re-gathers the per-cascade culled members into one indirect draw per casting mesh (the point path's
// shared `Regather`), and the receiver selects a cascade by view-depth + blends across the overlap band.

/** the atlas-UV tile rect `[u0, v0, du, dv]` for cascade `k` of `n` in the fixed cascade grid, written into
 * `out` at `at`. Cascades are equal-resolution, so a deterministic `ceil(√n)`-per-side grid packs them (n=1 →
 * the whole atlas; n=2 → side-by-side; n∈{3,4} → 2×2): no importance allocator. Otherwise pure. */
export function cascadeTileRect(k: number, n: number, out: Float32Array, at: number): void {
    const cols = Math.ceil(Math.sqrt(n));
    const d = 1 / cols;
    out[at] = (k % cols) * d;
    out[at + 1] = Math.floor(k / cols) * d;
    out[at + 2] = d;
    out[at + 3] = d;
}

/** the cascade atlas side in pixels: `ceil(√n) · resolution`, each cascade a `resolution`-square tile in the
 * fixed grid. With {@link sunResolution}'s power of two in [256, 4096] and {@link MAX_CASCADES} 4, it is a
 * power of two of at most 8192, WebGPU's default `maxTextureDimension2D`. */
export function cascadeAtlasSize(resolution: number, n: number): number {
    return Math.ceil(Math.sqrt(n)) * resolution;
}

// shadow cameras and their retained atlas inputs belong to the World that authors the lights. Frame-local
// matrix scratch below is reused synchronously; no cached eid, signature, output buffer or warning latch crosses Worlds.
const CASC_KEY_FLOATS = 29;
interface ShadowRuntime {
    cascadeEids: number[];
    cascadeCount: number;
    cascadeVP: Float32Array;
    cascadeRecv: Float32Array;
    cascadeMeta: Uint32Array;
    cascadeRects: Float32Array;
    cascadeFar: Float32Array;
    cascadeCover: Float32Array;
    cascadeDepth: Float32Array;
    sunBias: Float64Array;
    sunCascades: number;
    sunOverlap: number;
    cascKey: Float64Array;
    comboEids: number[];
    comboCount: number;
    faceVP: Float32Array;
    comboMeta: Uint32Array;
    tileRects: Float32Array;
    lastCasters: Set<number>;
    capWarned: boolean;
    overflowWarned: boolean;
    slotWarned: boolean;
}
const shadowRuntimeKey = { create: () => createShadowRuntime() };
const createShadowRuntime = (): ShadowRuntime => ({
    cascadeEids: [],
    cascadeCount: 0,
    cascadeVP: new Float32Array(MAX_CASCADES * 16),
    cascadeRecv: new Float32Array(MAX_CASCADES * 16),
    cascadeMeta: new Uint32Array(MAX_CASCADES * 4),
    cascadeRects: new Float32Array(MAX_CASCADES * 4),
    cascadeFar: new Float32Array(MAX_CASCADES),
    cascadeCover: new Float32Array(MAX_CASCADES),
    cascadeDepth: new Float32Array(MAX_CASCADES),
    sunBias: new Float64Array(2),
    sunCascades: 0,
    sunOverlap: 0,
    cascKey: new Float64Array(CASC_KEY_FLOATS).fill(Number.NaN),
    comboEids: [],
    comboCount: 0,
    faceVP: new Float32Array(0),
    comboMeta: new Uint32Array(0),
    tileRects: new Float32Array(0),
    lastCasters: new Set(),
    capWarned: false,
    overflowWarned: false,
    slotWarned: false,
});
const shadows = (world: World): ShadowRuntime => world.resource(shadowRuntimeKey);
const _cascNext = new Float64Array(CASC_KEY_FLOATS);
const _cascWorld = new Float32Array(16);
const _cascView = new Float32Array(16);
const _cascProj = new Float32Array(16);
const _cascTileMat = new Float32Array(16);
const SUN_TERMS = [DirectionalLight];
const _sunDir = new Float64Array(3);
const _splits = new Float64Array(MAX_CASCADES);
const _fit: LightFit = {
    eye: new Float64Array(3),
    focus: new Float64Array(3),
    up: new Float64Array(3),
    extent: new Float64Array(2),
};
const _cascRecv = new Float32Array(16);
const _cascFolded = new Float32Array(16);

/** the pooled cascade cameras' eids, one per active cascade (the first {@link cascadeCount} valid). Each is a
 * depth-only frustum-culled view slot: the per-cascade cull. An oracle reads each one's
 * `Views.get(eid).slot` + `computeViewProj(eid, 1)` to pin the pack's per-cascade survivor counts to a CPU
 * frustum test, the {@link pointComboEids} shape over cascade slots. */
export function cascadeComboEids(world: World): number[] {
    return shadows(world).cascadeEids;
}

/** the number of active cascades this frame (the light's {@link lightCascades}, or 1 for an orthographic
 * main camera, when the sun casts; else 0). */
export function cascadeCount(world: World): number {
    return shadows(world).cascadeCount;
}

/** the dense per-cascade **folded** tile viewProjs the atlas VS projects by (tile placement folded in via
 * {@link tileTransform}); the first {@link cascadeCount} mat4 valid. Filled by {@link updateCascades}. */
export function cascadeFaceVP(world: World): Float32Array {
    return shadows(world).cascadeVP;
}

/** the dense per-cascade **unfolded** receiver viewProjs (ortho × lookAt, no tile) the receiver projects a
 * fragment by before remapping into its tile rect; the first {@link cascadeCount} mat4 valid. */
export function cascadeRecvVP(world: World): Float32Array {
    return shadows(world).cascadeRecv;
}

/** the per-cascade `(tileIndex, …)` meta the atlas VS reads to index its tile rect; one `vec4<u32>` per
 * cascade, the first {@link cascadeCount} valid. */
export function cascadeMeta(world: World): Uint32Array {
    return shadows(world).cascadeMeta;
}

/** the per-cascade atlas-UV tile rects (`[u0, v0, du, dv]`), the first {@link cascadeCount} valid: what the
 * atlas VS discards by and the receiver remaps into. */
export function cascadeTileRects(world: World): Float32Array {
    return shadows(world).cascadeRects;
}

/** the per-cascade far-bound splits in linear view-z (the receiver's `get_cascade_index` selects by these,
 * Bevy's shape), the first {@link cascadeCount} valid. */
export function cascadeFars(world: World): Float32Array {
    return shadows(world).cascadeFar;
}

/** the per-cascade box half-extents (`cover`), the first {@link cascadeCount} valid: the receiver derives a
 * per-cascade shadow texel world size `2·cover/resolution` for its normal-offset bias. */
export function cascadeCovers(world: World): Float32Array {
    return shadows(world).cascadeCover;
}

/** the casting sun's resolved `numCascades` this frame ({@link lightCascades}): what sizes the cascade atlas
 * when it first casts, whatever the active {@link cascadeCount}. */
export function sunCascades(world: World): number {
    return shadows(world).sunCascades;
}

/** the casting sun's `overlapProportion` this frame, floored at 0: the receiver's cascade blend band. */
export function sunOverlap(world: World): number {
    return shadows(world).sunOverlap;
}

/** the casting sun's bias knobs this frame, `[depthBias, normalBias]` (the residual clip-space lift, the
 * receiver normal-offset multiplier): the renderer writes them into the receiver's params. */
export function sunBias(world: World): Float64Array {
    return shadows(world).sunBias;
}

// a pooled cascade camera: an off-screen ortho Camera (no canvas, `attachView`) posed per frame by
// `poseCascade` — the single sun camera's shape, one per cascade. It takes a cull slot, so the pack culls
// casters into it as one more depth-only view
function createCascadeCamera(world: World): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).mode.set(eid, CameraMode.Orthographic);
    attachView(world, eid);
    return eid;
}

// grow/shrink the cascade-camera pool to exactly `n` (the active cascade count). The count is hysteresis-free
// but `numCascades` is fixed once the light casts, so this is effectively a one-time create
function syncCascadePool(world: World, shadow: ShadowRuntime, n: number): void {
    if (shadow.cascadeEids.length !== n) shadow.cascKey.fill(Number.NaN);
    while (shadow.cascadeEids.length < n) shadow.cascadeEids.push(createCascadeCamera(world));
    while (shadow.cascadeEids.length > n) {
        const eid = shadow.cascadeEids.pop()!;
        detachCanvas(world, eid);
        world.destroy(eid);
    }
}

// pose a pooled cascade camera so render's `computeViewProj` reproduces this cascade's ortho projection — the
// frustum the pack culls against. `lookAtRotation` returns the lookAt orientation as a quaternion, so
// `invertMat4(composeMat4(pos, rot))` equals the `lookAt(eye, eye→focus, up)` the atlas render's `_cascadeRecv` folds
// the tile onto (the cull frustum and the render projection agree to f32 — the sun camera's guarantee)
function poseCascade(world: World, eid: number, fit: LightFit): void {
    const { eye, focus, up } = fit;
    const q = lookAtRotation(
        eye[0],
        eye[1],
        eye[2],
        focus[0],
        focus[1],
        focus[2],
        up[0],
        up[1],
        up[2],
    );
    world.storage(Transform).translation.set(eid, eye[0], eye[1], eye[2], 1);
    world.storage(Transform).rotation.set(eid, q.x, q.y, q.z, q.w);
    world.storage(Camera).mode.set(eid, CameraMode.Orthographic);
    world.storage(Camera).near.set(eid, 0);
    // size = the cover, far = the near-extended box depth (2·cover + margin), so the cull frustum matches the
    // render box and the toward-light occluder margin is culled in, not clipped out
    world.storage(Camera).size.set(eid, fit.extent[0]);
    world.storage(Camera).far.set(eid, fit.extent[1]);
}

/** destroy the pooled cascade cameras + their views (at plugin dispose). */
export function destroyCascades(world: World): void {
    const shadow = shadows(world);
    for (const eid of shadow.cascadeEids) {
        detachCanvas(world, eid);
        world.destroy(eid);
    }
    shadow.cascadeEids.length = 0;
    shadow.cascKey.fill(Number.NaN);
}

/** forget the cached cascade camera eids on a (re)build: the prior World owns its own teardown, a fresh one
 * recreates lazily (the same lifecycle-reset as {@link resetPointShadows}). */
export function resetCascades(world: World): void {
    const shadow = shadows(world);
    shadow.cascadeEids.length = 0;
    shadow.cascKey.fill(Number.NaN);
    shadow.cascadeCount = 0;
}

/**
 * pose the sun's cascade light cameras + fill the dense per-cascade viewProjs + meta + rects + far-bounds +
 * covers the atlas render + the receiver read. Runs in the `simulation` group, before the draw frame opens.
 * Casts nothing (sets `_cascadeCount = 0`) when the directional light has shadowMapsEnabled off or there's no
 * main camera. Two paths by main-camera projection:
 *
 * - **perspective** — split `[near, DirectionalLight.maximumDistance]` into the light's `numCascades` depth slices, fit one ortho box
 *   per slice ({@link cascadeFit}); the receiver selects a cascade by view-z and blends across the overlap band.
 * - **orthographic** — uniform texel density means depth cascades buy nothing, so a **single** box fit to the
 *   visible ground footprint ({@link orthoFootprintFit}); its far-bound is a sentinel so the receiver always
 *   selects it (count = 1, no blend). The frustum-slice fit doesn't reach an ortho camera's visible ground.
 *
 * `DirectionalLight.maximumDistance` is the CSM max shadow distance and doubles as the near-plane
 * margin every box extends toward the light ({@link placeFromCenter}), so a caster within shadow range above a
 * slice is captured, not clipped. The boxes texel-snap per cascade so the edges don't crawl.
 */
export function updateCascades(world: World, main: number): void {
    const shadow = shadows(world);
    const light = world.only(SUN_TERMS);
    if (light < 0 || !world.storage(DirectionalLight).shadowMapsEnabled.get(light) || main < 0) {
        shadow.cascadeCount = 0;
        return;
    }
    const resolution = sunResolution(world);
    const maxDist = Math.max(1e-3, world.storage(DirectionalLight).maximumDistance.get(light));
    shadow.sunBias[0] = world.storage(DirectionalLight).shadowDepthBias.get(light);
    shadow.sunBias[1] = world.storage(DirectionalLight).shadowNormalBias.get(light);
    _sunDir[0] = world.storage(DirectionalLight).direction.x.get(light);
    _sunDir[1] = world.storage(DirectionalLight).direction.y.get(light);
    _sunDir[2] = world.storage(DirectionalLight).direction.z.get(light);
    const view = world.resource(Views).get(main);
    const aspect = view && view.height > 0 ? view.width / view.height : 1;
    composeGlobalTransform(world, main, _cascWorld);
    const mode = world.storage(Camera).mode.get(main);
    const fov = world.storage(Camera).fov.get(main);
    const size = world.storage(Camera).size.get(main);
    const near = Math.max(1e-3, world.storage(Camera).near.get(main));
    // ortho cameras get one footprint box; perspective gets N depth slices
    const ortho = mode === CameraMode.Orthographic;
    const cascades = lightCascades(world, light);
    const n = ortho ? 1 : cascades;
    const overlap = Math.max(0, world.storage(DirectionalLight).overlapProportion.get(light));
    const first = world.storage(DirectionalLight).firstCascadeFarBound.get(light);
    // refused rather than split into degenerate cascades; the config lives in columns, so it is checked
    // where it is read, as Bevy's builder checks its own
    if (n > 1 && !(first < maxDist))
        throw new Error(
            `standard: DirectionalLight ${light} firstCascadeFarBound (${first}) must be less than its maximumDistance (${maxDist}) when numCascades is above 1`,
        );
    shadow.sunCascades = cascades;
    shadow.sunOverlap = overlap;

    syncCascadePool(world, shadow, n);

    _cascNext.set(_cascWorld, 0);
    _cascNext[16] = mode;
    _cascNext[17] = fov;
    _cascNext[18] = size;
    _cascNext[19] = aspect;
    _cascNext[20] = near;
    _cascNext[21] = maxDist;
    _cascNext[22] = first;
    _cascNext[23] = overlap;
    _cascNext[24] = resolution;
    _cascNext[25] = n;
    _cascNext[26] = _sunDir[0];
    _cascNext[27] = _sunDir[1];
    _cascNext[28] = _sunDir[2];
    let changed = false;
    for (let i = 0; i < CASC_KEY_FLOATS; i++) {
        if (shadow.cascKey[i] !== _cascNext[i]) {
            changed = true;
            break;
        }
    }
    // a pooled camera whose size or far no longer reads back what this pass wrote was posed from outside
    // it (or its eid was recycled under the pool), so the boxes are rebuilt even when the inputs agree
    if (!changed) {
        for (let i = 0; i < n; i++) {
            const cam = shadow.cascadeEids[i];
            if (cam === undefined) continue;
            if (
                world.storage(Camera).size.get(cam) !== shadow.cascadeCover[i] ||
                world.storage(Camera).far.get(cam) !== shadow.cascadeDepth[i]
            ) {
                changed = true;
                break;
            }
        }
    }
    if (!changed) {
        shadow.cascadeCount = n;
        return;
    }
    shadow.cascKey.set(_cascNext);

    if (!ortho) cascadeSplits(maxDist, n, first, _splits);
    for (let i = 0; i < n; i++) {
        let farBound: number;
        if (ortho) {
            orthoFootprintFit(
                _cascWorld,
                size,
                aspect,
                _sunDir,
                maxDist,
                resolution,
                maxDist,
                _fit,
            );
            // a sentinel beyond any visible fragment's view-z, so the receiver's get_cascade_index always
            // picks this single box (no blend, count = 1)
            farBound = 1e9;
        } else {
            const farSplit = _splits[i];
            // widen the near edge back over the blend band (Bevy's `next_near = (1−overlap)·this_far`), so the
            // band the receiver blends across is covered by both this cascade and its predecessor
            const nearSplit = i === 0 ? near : (1 - overlap) * _splits[i - 1];
            cascadeFit(
                _cascWorld,
                mode,
                fov,
                size,
                aspect,
                nearSplit,
                farSplit,
                _sunDir,
                resolution,
                maxDist,
                _fit,
            );
            farBound = farSplit;
        }
        cascadeTileRect(i, n, shadow.cascadeRects, i * 4);
        // unfolded receiver viewProj (ortho × lookAt) — matches `computeViewProj` of this cascade's camera
        // (aspect 1), so the cull frustum and the render projection agree; the folded VP adds the tile placement
        orthographic(_fit.extent[0], 1, 0, _fit.extent[1], _cascProj);
        lookAt(
            _fit.eye[0],
            _fit.eye[1],
            _fit.eye[2],
            _fit.focus[0],
            _fit.focus[1],
            _fit.focus[2],
            _fit.up[0],
            _fit.up[1],
            _fit.up[2],
            _cascView,
        );
        multiplyMat4(_cascProj, _cascView, _cascRecv);
        shadow.cascadeRecv.set(_cascRecv, i * 16);
        multiplyMat4(
            tileTransform(shadow.cascadeRects, _cascTileMat, i * 4),
            _cascRecv,
            _cascFolded,
        );
        shadow.cascadeVP.set(_cascFolded, i * 16);
        shadow.cascadeMeta[i * 4] = i; // tile index = cascade index (the VS reads cascadeRects[meta.x])
        shadow.cascadeFar[i] = farBound;
        shadow.cascadeCover[i] = _fit.extent[0];
        shadow.cascadeDepth[i] = _fit.extent[1];
        const cam = shadow.cascadeEids[i];
        if (cam !== undefined) poseCascade(world, cam, _fit);
    }
    shadow.cascadeCount = n;
}

// ---- point-light shadows: the CPU half of the importance-sized depth atlas ----
//
// Storage is one fixed **square** depth atlas sub-allocated by importance (PlayCanvas's
// light-texture-atlas.js model) — the same `texture_depth_2d` + comparison-sampler binding shape the sun
// shadow uses, so it needs no cube-array support and fits the integrated/WebGL floor. Each shadowed light
// claims power-of-two square tiles sized from its apparent contribution (`intensity·range²/dist²`): a
// point caster six face tiles, a `SpotLight` caster one cone tile. Tile *area* tracks the score (the hero light
// large, distant lights small), and the tiles are buddy/quadtree-packed so power-of-two squares pack with
// no fragmentation. Over-budget (the smallest uniform tiling still overflows the square) drops the least
// important with a non-silent warn.
//
// Each combo (a point caster's cube face, a spot's cone) is its own **frustum-culled depth view** — a
// pooled off-screen camera the MeshInstance pack culls casters into, the same `cull → count → scan → scatter` spine
// every camera uses. So a member rasterizes only the faces it actually hits, not all six (no
// over-amplification). The viewProjs are computed here CPU-side (one per combo, the tile placement folded
// in — {@link tileTransform}); standard re-gathers the per-combo culled members into one contiguous run per
// casting mesh + a per-instance combo index, so the atlas still renders in **one indirect draw per casting
// mesh** (the Dawn ~1µs/indirect-draw floor), now reading per-combo
// *culled* counts. The face/cone frustum is widened by a constant texel margin (the PlayCanvas seam fix)
// and the receiver clamps its 3×3 PCF taps to the tile interior, so a sample never bleeds into a neighbour.

/** the shadowed-caster ceiling: the per-frame caster array + the combo buffers size to it; the combo
 * view-slot budget (each caster claims up to 6 depth slots) is {@link MAX_COMBO_SLOTS} below */
export const MAX_POINT_CASTERS = 8;

// view slots reserved for the future CSM sun cascades, so building them never re-opens the combo budget.
// Each combo (a point caster's cube face / a spot's cone) is a depth-only cull view sharing the MAX_SLOTS
// pool with the shading cameras (MAX_VIEWS) and the cascades — so the combos a frame may pose are capped to
// what's left. At the default cap (8 point casters → 48 combos ≤ 52) this never bites; it's the guard if
// MAX_POINT_CASTERS is raised.
const CASCADE_RESERVE = MAX_CASCADES;
const MAX_COMBO_SLOTS = MAX_SLOTS - MAX_VIEWS - CASCADE_RESERVE;

// the active combo count for the first `count` casters: a point spans six cube faces, a spot one cone
function comboSlots(frames: PointShadowFrame[], count: number): number {
    let n = 0;
    for (let i = 0; i < count; i++) n += frames[i].spot ? 1 : 6;
    return n;
}

/**
 * the point-shadow budget. `atlas` + `casters` are read when the app warms and not changed on a live app:
 * standard compiles its shadow shaders against `atlas` and sizes the atlas texture from it when a light
 * first casts, and `casters` caps the lights ranked each frame. Set them in `AppConfig.setup` or a plugin's
 * `initialize`, e.g. `world.resource(PointShadows).atlas = 1024`; each world owns its own settings. `atlas`
 * is the square depth atlas's side in pixels (snapped to a power of two in [256, 4096], default 2048 ≈ 16 MB
 * of depth), sub-allocated by importance. `casters` is how many shadowed point/spot
 * lights compete for the atlas (clamped to [1, {@link MAX_POINT_CASTERS}]); lights beyond it stay lit but
 * cast nothing, with a non-silent warn. A caster that won't fit the atlas budget is dropped the same
 * way. A point caster claims six power-of-two face tiles, a spot one, each tile sized so its **area** tracks
 * the light's apparent contribution (`intensity·range²/dist²`): the hero light renders large, distant lights
 * small, and a spot costs one tile rather than six.
 *
 * `hysteresis` is the over-cap incumbent margin and IS live-tunable (pure CPU ranking, not baked into a
 * shader): when more shadowed lights exist than `casters`, a light that cast last frame keeps its slot
 * unless a challenger's importance beats it by this fraction. It stops a light's shadow flickering on/off
 * as the camera moves and re-ranks the winners by distance (set 0 for the raw nearest-wins behavior).
 */
export interface PointShadows {
    atlas: number;
    casters: number;
    hysteresis: number;
}

export const PointShadows: Resource<PointShadows> = {
    create: () => ({ atlas: 2048, casters: 8, hysteresis: 0.25 }),
};

/** the world's resolved caster cap: {@link PointShadows} `casters` clamped to [1, {@link MAX_POINT_CASTERS}] */
export function pointCasters(world: World): number {
    return Math.min(
        Math.max(Math.round(world.resource(PointShadows).casters), 1),
        MAX_POINT_CASTERS,
    );
}

/** the world's resolved atlas side in pixels: {@link PointShadows} `atlas` clamped to [256, 4096] and snapped
 * to a power of two (the buddy packer needs a power-of-two square) */
export function pointAtlasSize(world: World): number {
    const s = Math.min(Math.max(Math.round(world.resource(PointShadows).atlas), 256), 4096);
    return Math.min(4096, 1 << Math.round(Math.log2(s)));
}

// the smallest face tile (matches the prior fixed-tile clamp floor) and the PCF seam margin in face texels
// (how far past the 90° face / spot cone the projection must extend so the 3×3 footprint stays in the tile —
// PlayCanvas shadow-renderer-local.js). EDGE_TEXELS is constant in *texels*, so a tile's widened tangent
// scales with its own pixel size to keep the world margin the same fraction of every tile.
const MIN_TILE = 64;
/** the PCF seam margin in face texels: standard's FS recomputes the widened tangent (`1 + 2·EDGE/tilePx`)
 * per matched tile, so this is the one source for both the projection ({@link pointTanHalf}) and the receiver */
export const EDGE_TEXELS = 3;

/** the widened face frustum's tangent half-angle for a tile of `tilePx` pixels: 90° is tan = 1, plus
 * EDGE_TEXELS of the tile's 2/T texel size. The receiver divides by the same constant (derived from the
 * matched tile's pixel size), so the projection matches the render exactly and a face-boundary direction
 * lands EDGE_TEXELS inside the tile edge */
export function pointTanHalf(tilePx: number): number {
    return 1 + (2 * EDGE_TEXELS) / tilePx;
}

/** the face frustum's vertical FOV in degrees (the fov `perspective()` takes for each face viewProj) for a
 * tile of `tilePx` pixels */
export function pointFov(tilePx: number): number {
    return (Math.atan(pointTanHalf(tilePx)) * 360) / Math.PI;
}

/**
 * a buddy quadtree packer over a square atlas of side `side` pixels (a power of two). `alloc(size)` returns
 * the pixel origin `[x, y]` of a free `size × size` tile (`size` a power of two ≤ `side`), or `null` when
 * the atlas can't fit it; `reset()` reclaims the whole atlas: the per-frame reuse path, so a dropped
 * caster's space is free again next frame. Power-of-two square tiles pack with no fragmentation when
 * allocated largest-first. Pure; the allocator the importance sizing builds on (exported for the pack tests).
 */
export function createPacker(side: number): {
    reset: () => void;
    alloc: (size: number) => [number, number] | null;
} {
    // free tile origins per size, encoded `y*side + x`; a node splits into four children on demand
    const free = new Map<number, number[]>();
    const reset = (): void => {
        free.clear();
        free.set(side, [0]);
    };
    const alloc = (size: number): [number, number] | null => {
        let code: number;
        const list = free.get(size);
        if (list && list.length > 0) {
            code = list.pop()!;
        } else {
            if (size >= side) return null;
            const parent = alloc(size * 2);
            if (!parent) return null;
            const [px, py] = parent;
            const kids = free.get(size) ?? [];
            // keep the parent's own corner; free the other three quadrants for later (same-size) allocs
            kids.push(
                py * side + px + size,
                (py + size) * side + px,
                (py + size) * side + px + size,
            );
            free.set(size, kids);
            code = py * side + px;
        }
        const x = code % side;
        return [x, (code - x) / side];
    };
    reset();
    return { reset, alloc };
}

type Vec3 = [number, number, number];
const cross = (a: Vec3, b: Vec3): Vec3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];
const norm = (v: Vec3): Vec3 => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
};

/** a spot caster's cone basis for the atlas render + the FS reconstruct, from the light's Transform
 * rotation. `fwd` = the cone axis (the entity's local -Z, the same forward the compact pass oct-packs);
 * `right`/`up` = the `lookAt` basis (`right = normalize(fwd × up0)`, `up = right × fwd`) so the FS's
 * analytic receiver matches the rendered viewProj exactly. `coneTanHalf` is `tan(outer)` widened by the PCF
 * margin of the caster's allocated tile (`tilePx` pixels, the spot analogue of {@link pointTanHalf}); the
 * perspective FOV derives from it. Pure; the oracle the spot's WGSL receiver reconstruct is pinned to. */
export function spotBasis(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    outerDeg: number,
    tilePx: number,
) {
    const fwd = norm([
        -2 * (qx * qz + qw * qy),
        -2 * (qy * qz - qw * qx),
        -(1 - 2 * (qx * qx + qy * qy)),
    ]);
    // up0 dodges the cone-axis-parallel degenerate; the orthonormalized `up` carries it past here
    const up0: Vec3 = Math.abs(fwd[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
    const right = norm(cross(fwd, up0));
    const up = cross(right, fwd);
    // clamp the outer half-angle so the perspective frustum stays under 180° (tan finite); widen by the
    // PCF seam margin like pointTanHalf, so a cone-edge fragment's 3×3 footprint stays inside its tile
    const clamped = Math.min(Math.max(outerDeg, 1), 80);
    const coneTanHalf = Math.tan((clamped * Math.PI) / 180) + (2 * EDGE_TEXELS) / tilePx;
    const coneFov = (Math.atan(coneTanHalf) * 360) / Math.PI;
    return { fwd, right, up, coneTanHalf, coneFov };
}

/** one cube face's render basis. `right` is derived (fwd × up, the camera basis `lookAtRotation` produces), so the
 * WGSL face selection generated from this table and the camera orientation can never disagree */
export interface PointFaceBasis {
    fwd: [number, number, number];
    up: [number, number, number];
    right: [number, number, number];
}

function basis(fwd: [number, number, number], up: [number, number, number]): PointFaceBasis {
    const right: [number, number, number] = [
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    ];
    return { fwd, up, right };
}

/** the six faces, ±X ±Y ±Z. The ±Y faces take a Z up (the look direction is parallel to world up, the
 * degenerate case) */
export const POINT_FACES: PointFaceBasis[] = [
    basis([1, 0, 0], [0, 1, 0]),
    basis([-1, 0, 0], [0, 1, 0]),
    basis([0, 1, 0], [0, 0, 1]),
    basis([0, -1, 0], [0, 0, -1]),
    basis([0, 0, 1], [0, 1, 0]),
    basis([0, 0, -1], [0, 1, 0]),
];

/**
 * the clip-space tile-placement matrix `D` for an atlas-UV rect `[u0, v0, du, dv]`: left-multiplying a
 * face's viewProj by it lands the face's projection inside that atlas tile, with the divide and the y-flip
 * baked into clip space so the **hardware** does the perspective divide + near-plane clip (a manual
 * `fc.xy/fc.w` in the VS can't: a vertex behind the face near plane, `fc.w ≤ 0`, divides to garbage).
 * The tile remap is a viewport transform, affine in clip space: `clip.x = du·fc.x + (2u0+du−1)·fc.w`,
 * `clip.y = dv·fc.y + (1−2v0−dv)·fc.w`, z/w untouched, so `tileVP = D · faceVP` and the VS is one matrix
 * multiply. The receiver (`pointShadowOf`) reconstructs the same tile uv analytically from its rect, so it
 * reads identical depth at identical pixels: `D` changes only what the render writes, not where it samples.
 * Column-major, the layout `multiplyMat4`/the shader expect. Reads the rect at `at` in `rect`. Writes into `out`
 * when given (so the per-frame loop reuses a scratch matrix, like `perspective`/`lookAt`/`multiplyMat4`), else
 * allocates. Pure; pinned to the receiver's uv by unit test.
 */
export function tileTransform(
    rect: ArrayLike<number>,
    out = new Float32Array(16),
    at = 0,
): Float32Array {
    const u0 = rect[at];
    const v0 = rect[at + 1];
    const du = rect[at + 2];
    const dv = rect[at + 3];
    out.fill(0);
    out[0] = du;
    out[5] = dv;
    out[10] = 1;
    out[12] = 2 * u0 + du - 1; // ax
    out[13] = 1 - 2 * v0 - dv; // ay
    out[15] = 1;
    return out;
}

/** one shadowed caster's per-frame placement the renderer needs: the caster slot, the light's GlobalTransform +
 * range-derived clip planes, the bias knobs, and the importance fields the atlas allocator reads (`score`
 * sizes its tile; `tilePx` is the resolved face-tile pixel size). A point caster spans six cube-face combos;
 * a `spot` caster a single cone combo (the `cone*` basis the FS reconstructs the receiver from: `right`/
 * `up`/`fwd` the lookAt basis, `coneTanHalf` the widened cone tangent, 0 for a point). The combos' viewProjs
 * are computed CPU-side into {@link pointFaceVP} (no per-face cameras), one per entry of {@link pointComboMeta} */
export interface PointShadowFrame {
    light: number;
    slot: number;
    score: number;
    tilePx: number;
    pos: [number, number, number];
    near: number;
    far: number;
    depthBias: number;
    normalBias: number;
    spot: boolean;
    fwd: [number, number, number];
    right: [number, number, number]; // the lookAt-derived basis the FS projects fragments onto
    up: [number, number, number];
    coneTanHalf: number; // tan(outer half-angle) widened for the PCF seam, 0 for a point caster
    coneFov: number; // the perspective FOV (degrees) the cone viewProj uses
}

// the shadowed point-light query terms and the ranked candidates, a capacity pool reused in place
const POINT_CASTER_TERMS = [PointLight, GlobalTransform];
const SPOT_CASTER_TERMS = [SpotLight, GlobalTransform];
const LIGHT_CASTER_TERMS = [POINT_CASTER_TERMS, SPOT_CASTER_TERMS];
const _cands: { light: number; range: number; score: number; rank: number }[] = [];

// one caster frame record for a caller's pool, written in place each frame by `updatePointShadows`
function newPointFrame(): PointShadowFrame {
    return {
        light: 0,
        slot: 0,
        score: 0,
        tilePx: MIN_TILE,
        pos: [0, 0, 0],
        near: 0,
        far: 0,
        depthBias: 0,
        normalBias: 0,
        spot: false,
        fwd: [0, 0, -1],
        right: [1, 0, 0],
        up: [0, 1, 0],
        coneTanHalf: 0,
        coneFov: 0,
    };
}

// per-combo view scratch stays process-local; retained outputs and pooled eids live in ShadowRuntime.
const _faceView = new Float32Array(16);
const _faceProj = new Float32Array(16);
// the proj·view product before the tile transform is folded on (the combo viewProj carries the tile
// placement, so the VS emits `tileVP·world` directly — see {@link tileTransform})
const _pv = new Float32Array(16);
// scratch for the per-combo tile-placement matrix, reused across combos (no per-frame matrix alloc)
const _tileMat = new Float32Array(16);

/** the per-combo **tile** viewProjs the renderer uploads each shadowed frame, dense (one per active
 * combo, the first active-combo-count entries valid). Each carries its atlas tile placement folded in
 * ({@link tileTransform}), so the atlas VS emits `tileVP·world` with no manual divide. Filled by
 * {@link updatePointShadows} */
export function pointFaceVP(world: World): Float32Array {
    return shadows(world).faceVP;
}

/** the per-combo `(casterSlot, face, _, _)` the atlas VS reads (keyed by the re-gathered instance's combo
 * index) to index its tile rect (`slot·6 + face`). One `vec4<u32>` per combo, the first
 * active-combo-count valid */
export function pointComboMeta(world: World): Uint32Array {
    return shadows(world).comboMeta;
}

/** the per-(caster, face) allocated atlas-UV rects (`[u0, v0, du, dv]`, square), sparse and indexed
 * `slot·6 + face`: what the receiver samples and the atlas VS reads for its tile-discard bounds. Sized to
 * `cap·6` vec4. Filled by {@link updatePointShadows}, uploaded by standard as the `"pointTileRects"` uniform */
export function pointTileRects(world: World): Float32Array {
    return shadows(world).tileRects;
}

/** the pooled combo cameras' eids, one per active combo (combo-major: each caster's faces/cone in turn,
 * the first active-combo-count valid). Each is a depth-only frustum-culled view slot: the per-combo
 * cull. An oracle reads each combo's `Views.get(eid).slot` + `computeViewProj(eid, 1)` to pin the
 * pack's per-combo survivor counts to a CPU frustum test (the combo's frustum is what the pack culls
 * against, == the pre-fold proj·view the atlas VS folds the tile into). */
export function pointComboEids(world: World): number[] {
    return shadows(world).comboEids;
}

// a pooled combo camera: an off-screen perspective Camera (no canvas, `attachView`) whose pose + fov are
// set per frame by `poseCombo`. The sun camera's shape — it takes a cull slot and packs its viewProj
// through the same `computeViewProj` as any camera, so the pack culls casters into it as one more view
function createComboCamera(world: World): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).mode.set(eid, CameraMode.Perspective);
    attachView(world, eid);
    return eid;
}

// grow/shrink the combo-camera pool to exactly `n` (the active combo count), creating depth-only cameras
// as casters appear and tearing the excess down as they leave. Only a count change drives create/destroy —
// the per-frame path just reposes the live ones — and the caster set is hysteresis-stable, so this is
// cold
function syncComboPool(world: World, shadows: ShadowRuntime, n: number): void {
    while (shadows.comboEids.length < n) shadows.comboEids.push(createComboCamera(world));
    while (shadows.comboEids.length > n) {
        const eid = shadows.comboEids.pop()!;
        detachCanvas(world, eid);
        world.destroy(eid);
    }
}

// pose a pooled combo camera so render's `computeViewProj` reproduces this combo's face/cone projection —
// the frustum the pack culls against. `lookAtRotation` returns the lookAt orientation as a quaternion (the sun
// camera's path), so `invertMat4(composeMat4(pos, rot))` equals the `lookAt(eye, eye+fwd, up)` the atlas render's
// `_faceVP` folds the tile placement onto — the cull frustum and the render projection agree (to f32).
function poseCombo(
    world: World,
    eid: number,
    px: number,
    py: number,
    pz: number,
    fwd: readonly [number, number, number],
    up: readonly [number, number, number],
    fov: number,
    near: number,
    far: number,
): void {
    const q = lookAtRotation(
        px,
        py,
        pz,
        px + fwd[0],
        py + fwd[1],
        pz + fwd[2],
        up[0],
        up[1],
        up[2],
    );
    world.storage(Transform).translation.set(eid, px, py, pz, 1);
    world.storage(Transform).rotation.set(eid, q.x, q.y, q.z, q.w);
    world.storage(Camera).fov.set(eid, fov);
    world.storage(Camera).near.set(eid, near);
    world.storage(Camera).far.set(eid, far);
}

/** destroy the pooled combo cameras + their views (at plugin dispose) */
export function destroyPointShadows(world: World): void {
    const shadow = shadows(world);
    for (const eid of shadow.comboEids) {
        detachCanvas(world, eid);
        world.destroy(eid);
    }
    shadow.comboEids.length = 0;
}

/** forget the cached combo camera eids on a (re)build: the prior World owns its own teardown, a fresh
 * one recreates lazily */
export function resetPointShadows(world: World): void {
    const shadow = shadows(world);
    shadow.comboEids.length = 0;
    shadow.capWarned = false;
    shadow.overflowWarned = false;
    shadow.slotWarned = false;
    shadow.lastCasters.clear();
}

// rect = atlas-UV [u0, v0, du, dv] (square, du == dv)
type Rect = [number, number, number, number];

/** size + place the first `count` casters' face tiles by importance: tile **area ∝ score** (side ∝ √score),
 * the most important the largest tile that still lets the whole set pack into the square atlas. A point
 * requests 6 same-size face tiles, a spot 1. Returns the per-(caster, face) atlas-UV rects (indexed
 * `[frame][face]`), or `null` when even the smallest uniform tiling (every face MIN_TILE) overflows: the
 * caller drops the least-important caster and retries. Pure (reads only its args). */
export function packCasters(
    frames: PointShadowFrame[],
    count: number,
    side: number,
): Rect[][] | null {
    let maxScore = 1e-9;
    for (let i = 0; i < count; i++) maxScore = Math.max(maxScore, frames[i].score);
    // halve the hero's "base" tile until the whole set packs; area ∝ score so side drops one power of two
    // per 4× score drop (0.5·log2). MIN_TILE is the floor; an over-budget set fails every base and returns null
    for (let base = side; base >= MIN_TILE; base >>= 1) {
        const packer = createPacker(side);
        const reqs: { frame: number; face: number; size: number }[] = [];
        for (let i = 0; i < count; i++) {
            const drop = Math.max(
                0,
                Math.round(0.5 * Math.log2(maxScore / Math.max(frames[i].score, 1e-9))),
            );
            const size = Math.max(MIN_TILE, base / 2 ** Math.min(drop, 20));
            const faces = frames[i].spot ? 1 : 6;
            for (let face = 0; face < faces; face++) reqs.push({ frame: i, face, size });
        }
        reqs.sort((a, b) => b.size - a.size); // largest first — buddy packs with no fragmentation
        const rects: Rect[][] = [];
        for (let i = 0; i < count; i++) rects.push([]);
        let ok = true;
        for (const r of reqs) {
            const o = packer.alloc(r.size);
            if (!o) {
                ok = false;
                break;
            }
            rects[r.frame][r.face] = [o[0] / side, o[1] / side, r.size / side, r.size / side];
        }
        if (ok) return rects;
    }
    return null;
}

/**
 * rank the shadowed point/spot lights, size + pack their atlas tiles by importance, and compute the
 * per-combo tile viewProjs + rects the atlas render projects by. Runs in the `simulation` group. Casters
 * are the PointLight and SpotLight entities with shadowMapsEnabled, capped at {@link pointCasters} with a non-silent
 * warn. Over the cap the **highest-importance** lights win (apparent contribution at the `main` camera,
 * `intensity · range² / dist²`, scale-invariant), so a far dim light never steals a slot from the hero by
 * query order; a hysteresis margin keeps an incumbent its slot so the set doesn't flicker. {@link packCasters}
 * then sizes each caster's tiles (area ∝ score) and buddy-packs them into the square atlas; a caster that
 * won't fit even at the smallest tiling is dropped (warn). Each combo's viewProj is `tileTransform(rect) ×
 * perspective([pointFov(tilePx), 1, near, far]) × lookAt(light, light+fwd, up)` (near/far = `[range/1000,
 * range]`), written into the shared {@link pointFaceVP} buffer combo-major, with its rect in {@link pointTileRects}.
 * Each combo also gets a pooled depth-only camera ({@link pointComboEids}) the pack frustum-culls casters
 * into (the per-combo cull), spawned lazily at the first casting frame. The frames are written into the
 * caller's `frames` pool (grown by one record per new high-water caster, reused in place after) and the
 * live caster count is returned.
 */
export function updatePointShadows(world: World, main: number, frames: PointShadowFrame[]): number {
    const shadow = shadows(world);
    const cap = pointCasters(world);
    const atlas = pointAtlasSize(world);
    const hysteresis = Math.max(0, world.resource(PointShadows).hysteresis);
    const cx = main >= 0 ? world.storage(GlobalTransform).translation.x.get(main) : 0;
    const cy = main >= 0 ? world.storage(GlobalTransform).translation.y.get(main) : 0;
    const cz = main >= 0 ? world.storage(GlobalTransform).translation.z.get(main) : 0;
    let candCount = 0;
    for (const terms of LIGHT_CASTER_TERMS)
        for (const light of world.query(terms)) {
            if (terms === POINT_CASTER_TERMS && world.has(light, SpotLight)) continue;
            const source = world.has(light, SpotLight)
                ? world.storage(SpotLight)
                : world.storage(PointLight);
            if (!source.shadowMapsEnabled.get(light)) continue;
            const range = source.range.get(light);
            if (range <= 0) continue;
            const dx = world.storage(GlobalTransform).translation.x.get(light) - cx;
            const dy = world.storage(GlobalTransform).translation.y.get(light) - cy;
            const dz = world.storage(GlobalTransform).translation.z.get(light) - cz;
            const distSq = main >= 0 ? Math.max(dx * dx + dy * dy + dz * dz, 1) : 1;
            const score = (source.intensity.get(light) * range * range) / distSq;
            // an incumbent (cast last frame) ranks with the hysteresis margin so a sub-margin challenger can't
            // evict it — the set stays put under small camera moves, killing the shadow flicker
            const rank = shadow.lastCasters.has(light) ? score * (1 + hysteresis) : score;
            let cand = _cands[candCount];
            if (!cand) {
                cand = { light: 0, range: 0, score: 0, rank: 0 };
                _cands[candCount] = cand;
            }
            cand.light = light;
            cand.range = range;
            cand.score = score;
            cand.rank = rank;
            candCount++;
        }
    // highest rank first, ties by light eid: an insertion sort over the pooled records
    for (let i = 1; i < candCount; i++) {
        const cand = _cands[i];
        let j = i;
        while (
            j > 0 &&
            (cand.rank > _cands[j - 1].rank ||
                (cand.rank === _cands[j - 1].rank && cand.light < _cands[j - 1].light))
        ) {
            _cands[j] = _cands[j - 1];
            j--;
        }
        _cands[j] = cand;
    }
    const extra = candCount - cap;
    let count = candCount;
    if (extra > 0) {
        count = cap;
        if (!shadow.capWarned) {
            shadow.capWarned = true;
            console.warn(
                `standard: ${cap + extra} shadowed point lights exceed the ${cap} caster cap; the ${extra} least important cast no shadow (raise PointShadows.casters, max ${MAX_POINT_CASTERS})`,
            );
        }
    } else {
        shadow.capWarned = false;
    }

    for (let slot = 0; slot < count; slot++) {
        const c = _cands[slot];
        let f = frames[slot];
        if (!f) {
            f = newPointFrame();
            frames[slot] = f;
        }
        f.light = c.light;
        f.slot = slot;
        f.score = c.score;
        f.tilePx = MIN_TILE;
        f.pos[0] = world.storage(GlobalTransform).translation.x.get(c.light);
        f.pos[1] = world.storage(GlobalTransform).translation.y.get(c.light);
        f.pos[2] = world.storage(GlobalTransform).translation.z.get(c.light);
        f.near = c.range / 1000;
        f.far = c.range;
        const source = world.has(c.light, SpotLight)
            ? world.storage(SpotLight)
            : world.storage(PointLight);
        f.depthBias = source.shadowDepthBias.get(c.light);
        f.normalBias = source.shadowNormalBias.get(c.light);
        f.spot = world.has(c.light, SpotLight);
        f.fwd[0] = 0;
        f.fwd[1] = 0;
        f.fwd[2] = -1;
        f.right[0] = 1;
        f.right[1] = 0;
        f.right[2] = 0;
        f.up[0] = 0;
        f.up[1] = 1;
        f.up[2] = 0;
        f.coneTanHalf = 0;
        f.coneFov = 0;
    }

    // size + pack by importance; on atlas overflow drop the least important (the tail, frames are rank-sorted)
    // and retry, warning once per episode. No caster packs nothing
    let rects = count > 0 ? packCasters(frames, count, atlas) : null;
    let dropped = 0;
    while (!rects && count > 0) {
        count--;
        dropped++;
        rects = count > 0 ? packCasters(frames, count, atlas) : null;
    }
    if (dropped > 0) {
        if (!shadow.overflowWarned) {
            shadow.overflowWarned = true;
            console.warn(
                `standard: ${dropped} shadowed point light(s) dropped — the ${atlas}×${atlas} shadow atlas is full (raise PointShadows.atlas or lower PointShadows.casters)`,
            );
        }
    } else {
        shadow.overflowWarned = false;
    }
    // slot-budget: each combo is a depth-only cull view sharing the MAX_SLOTS pool with the shading
    // cameras + the reserved cascades. Drop the least-important caster (the rank-sorted tail) while the
    // active combos would overflow the pool — a loud warn, the shape of the atlas drop above. Unreachable
    // at the default cap (8 point casters → 48 combos ≤ MAX_COMBO_SLOTS); the guard if the cap is raised.
    let slotDropped = 0;
    while (comboSlots(frames, count) > MAX_COMBO_SLOTS && count > 0) {
        count--;
        slotDropped++;
    }
    if (slotDropped > 0) {
        if (!shadow.slotWarned) {
            shadow.slotWarned = true;
            console.warn(
                `standard: ${slotDropped} shadowed light(s) dropped — combo views exceed the ${MAX_COMBO_SLOTS}-slot budget (lower PointShadows.casters)`,
            );
        }
    } else {
        shadow.slotWarned = false;
    }
    // record the surviving winners as next frame's incumbents (the hysteresis basis)
    shadow.lastCasters.clear();
    for (let i = 0; i < count; i++) shadow.lastCasters.add(frames[i].light);

    // resolve each caster's tile pixel size + (for spots) the cone basis, now that sizes are known
    for (let i = 0; i < count; i++) {
        const f = frames[i];
        const rect = rects ? rects[f.slot][0] : undefined;
        f.tilePx = rect ? rect[2] * atlas : MIN_TILE;
        if (f.spot) {
            const b = spotBasis(
                world.storage(GlobalTransform).rotation.x.get(f.light),
                world.storage(GlobalTransform).rotation.y.get(f.light),
                world.storage(GlobalTransform).rotation.z.get(f.light),
                world.storage(GlobalTransform).rotation.w.get(f.light),
                world.storage(SpotLight).outerAngle.get(f.light),
                f.tilePx,
            );
            f.fwd = b.fwd;
            f.right = b.right;
            f.up = b.up;
            f.coneTanHalf = b.coneTanHalf;
            f.coneFov = b.coneFov;
        }
    }

    // size the combo-camera pool to the active combo count, so each cube face / spot cone is its own
    // depth-only frustum-culled view this frame (the per-combo cull) — the pack culls casters into each
    // independently. The loop below poses each from its face/cone basis, and an empty caster set tears
    // the pool down
    syncComboPool(world, shadow, comboSlots(frames, count));

    // fill the combo tile-viewProjs densely (a point caster's 6 cube faces, a spot's 1 cone), the per-combo
    // (caster slot, face), and the per-(caster, face) rects. Each viewProj has its allocated atlas-UV rect
    // folded in (tileTransform), so the VS projects straight into the tile. ci runs over combos, ≤ 6·cap.
    if (shadow.faceVP.length < cap * 6 * 16) shadow.faceVP = new Float32Array(cap * 6 * 16);
    if (shadow.comboMeta.length < cap * 6 * 4) shadow.comboMeta = new Uint32Array(cap * 6 * 4);
    if (shadow.tileRects.length < cap * 6 * 4) shadow.tileRects = new Float32Array(cap * 6 * 4);
    shadow.tileRects.fill(0);
    let ci = 0;
    for (let k = 0; k < count; k++) {
        const frame = frames[k];
        const px = frame.pos[0];
        const py = frame.pos[1];
        const pz = frame.pos[2];
        const faceRects = rects![frame.slot];
        if (frame.spot) {
            const rect = faceRects[0];
            perspective(frame.coneFov, 1, frame.near, frame.far, _faceProj);
            // `up` is orthonormal to `fwd`, so it serves as the lookAt up directly (same basis as up0)
            lookAt(
                px,
                py,
                pz,
                px + frame.fwd[0],
                py + frame.fwd[1],
                pz + frame.fwd[2],
                frame.up[0],
                frame.up[1],
                frame.up[2],
                _faceView,
            );
            // fold the tile placement into the viewProj, so the VS projects straight into the atlas tile
            // with the hardware doing the divide + near-plane clip
            multiplyMat4(_faceProj, _faceView, _pv);
            multiplyMat4(tileTransform(rect, _tileMat), _pv, shadow.faceVP.subarray(ci * 16));
            shadow.comboMeta[ci * 4] = frame.slot;
            shadow.comboMeta[ci * 4 + 1] = 0; // a spot's lone combo is face 0 of its slot
            shadow.tileRects.set(rect, frame.slot * 6 * 4);
            // pose this combo's cull camera (its frustum == the pre-fold `_pv` above) so the pack culls
            // casters into its slot independently
            const comboCam = shadow.comboEids[ci];
            if (comboCam !== undefined)
                poseCombo(
                    world,
                    comboCam,
                    px,
                    py,
                    pz,
                    frame.fwd,
                    frame.up,
                    frame.coneFov,
                    frame.near,
                    frame.far,
                );
            ci++;
        } else {
            const fov = pointFov(frame.tilePx);
            perspective(fov, 1, frame.near, frame.far, _faceProj);
            for (let f = 0; f < 6; f++) {
                const { fwd, up } = POINT_FACES[f];
                const rect = faceRects[f];
                lookAt(
                    px,
                    py,
                    pz,
                    px + fwd[0],
                    py + fwd[1],
                    pz + fwd[2],
                    up[0],
                    up[1],
                    up[2],
                    _faceView,
                );
                // fold the face's atlas tile placement into its viewProj — the VS emits `tileVP·world`
                // with no manual divide, so the hardware clips a triangle behind the face near plane
                multiplyMat4(_faceProj, _faceView, _pv);
                multiplyMat4(tileTransform(rect, _tileMat), _pv, shadow.faceVP.subarray(ci * 16));
                shadow.comboMeta[ci * 4] = frame.slot;
                shadow.comboMeta[ci * 4 + 1] = f;
                shadow.tileRects.set(rect, (frame.slot * 6 + f) * 4);
                // pose this face's cull camera (its frustum == the pre-fold `_pv` above)
                const comboCam = shadow.comboEids[ci];
                if (comboCam !== undefined)
                    poseCombo(world, comboCam, px, py, pz, fwd, up, fov, frame.near, frame.far);
                ci++;
            }
        }
    }
    shadow.comboCount = ci;
    return count;
}
