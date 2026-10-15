// Shadow cameras share core's view/projection and frustum packing. They are created lazily for lights
// with shadowMapsEnabled; the mesh producer culls into their depth-only view slots.

import {
    attachView,
    Camera,
    CameraMode,
    DirectionalLight,
    detachView,
    MAX_SLOTS,
    MAX_VIEWS,
    PointLight,
    SpotLight,
    Views,
} from "../../core/rendering";
import { composeGlobalTransform, GlobalTransform, Transform } from "../../core/transform";
import {
    lookAt,
    lookAtRotation,
    multiplyMat4,
    orthographic,
    perspective,
    type Resource,
    type World,
} from "../../engine";
import { directionalTravelDirection, shadowDirectionalLight } from "./lighting";

/** the directional shadow's cascade ceiling: each cascade takes one of the depth view slots reserved out
 * of the point-shadow combo budget, so the count can't exceed it. */
export const MAX_CASCADES = 4;

/**
 * World-owned directional tile size in pixels (default 2048), clamped to [256, 4096] and snapped to a power of two.
 * Changes reallocate before the next casting frame. Tiles share a depth32float atlas ceil(√numCascades) across;
 * 4096-pixel tiles with multiple cascades require an 8192² atlas (256 MiB).
 * Only the brightest enabled directional casts (ties by lower eid); conflicting requests warn once.
 * One caster bounds depth passes and atlas area. Cascade settings live on {@link DirectionalLight}.
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

// Texel snapping prevents shadow crawl. Extending the near plane toward the light captures occluders
// above the fitted slice without changing the snap plane.
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
 * rest are spaced exponentially to `far` (one cascade ends at `far`). A `first` at or beyond `far` gives
 * equal or shrinking bounds; `first` is floored just above 0.
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
 * Fits the camera slice `[nearSplit, farSplit]` into a texel-snapped light box.
 * Its bounding-sphere half-extent stays constant as the camera rotates; `margin` captures occluders above it.
 * Reads camera placement and projection (`fov` or `size`), not camera near/far. Writes `out`.
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
 * Fits an orthographic camera's visible y=0 footprint into one texel-snapped light box.
 * Uniform texel density needs no depth cascades. When `fwd.y ≥ −1e-3`, uses a forward-distance box
 * with half-extent `distance`. `margin` extends the near plane toward the light. Writes `out`.
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
    sunTowardLight: Float64Array;
    sunCascades: number;
    sunOverlap: number;
    cascKey: Float64Array;
    comboEids: number[];
    comboCount: number;
    faceVP: Float32Array;
    comboMeta: Uint32Array;
    tileRects: Float32Array;
    casterProj: Float32Array[];
    casterProjKey: Float64Array;
    packer: Packer | undefined;
    packSizes: Int32Array;
    packRects: Float64Array;
    lastCasters: Set<number>;
    pendingCasters: Set<number>;
    casterSelectionPending: boolean;
    commitCasters: () => void;
    capWarned: boolean;
    overflowWarned: boolean;
    slotWarned: boolean;
}
const shadowRuntimeKey = { create: () => createShadowRuntime() };
const createShadowRuntime = (): ShadowRuntime => {
    let runtime: ShadowRuntime;
    runtime = {
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
        sunTowardLight: new Float64Array(3),
        sunCascades: 0,
        sunOverlap: 0,
        cascKey: new Float64Array(CASC_KEY_FLOATS).fill(Number.NaN),
        comboEids: [],
        comboCount: 0,
        faceVP: new Float32Array(0),
        comboMeta: new Uint32Array(0),
        tileRects: new Float32Array(0),
        casterProj: Array.from({ length: MAX_POINT_CASTERS }, () => new Float32Array(16)),
        casterProjKey: new Float64Array(MAX_POINT_CASTERS * 2).fill(Number.NaN),
        packer: undefined,
        packSizes: new Int32Array(MAX_POINT_CASTERS),
        packRects: new Float64Array(MAX_POINT_CASTERS * 6 * 4),
        lastCasters: new Set(),
        pendingCasters: new Set(),
        casterSelectionPending: false,
        commitCasters: () => {
            if (!runtime.casterSelectionPending) return;
            const previous = runtime.lastCasters;
            runtime.lastCasters = runtime.pendingCasters;
            runtime.pendingCasters = previous;
            runtime.casterSelectionPending = false;
        },
        capWarned: false,
        overflowWarned: false,
        slotWarned: false,
    };
    return runtime;
};
const shadows = (world: World): ShadowRuntime => world.resource(shadowRuntimeKey);
const _cascNext = new Float64Array(CASC_KEY_FLOATS);
const _cascWorld = new Float32Array(16);
const _cascView = new Float32Array(16);
const _cascProj = new Float32Array(16);
const _cascTileMat = new Float32Array(16);
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

/** Borrowed depth-only camera eids; the first {@link cascadeCount} entries are active. */
export function cascadeComboEids(world: World): number[] {
    return shadows(world).cascadeEids;
}

/** the number of active cascades this frame (the light's {@link lightCascades}, or 1 for an orthographic
 * main camera, when the sun casts; else 0). */
export function cascadeCount(world: World): number {
    return shadows(world).cascadeCount;
}

/** Borrowed tile-folded viewProjs; the first cascadeCount matrices are valid after updateCascades. */
export function cascadeFaceVP(world: World): Float32Array {
    return shadows(world).cascadeVP;
}

/** Borrowed receiver viewProjs without tile placement; the first cascadeCount matrices are valid. */
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

/** the per-cascade far-bound splits in linear view-z, the first {@link cascadeCount} valid. */
export function cascadeFars(world: World): Float32Array {
    return shadows(world).cascadeFar;
}

/** the per-cascade box half-extents (`cover`), the first {@link cascadeCount} valid: the receiver derives a
 * per-cascade shadow texel world size `2·cover/resolution` for its normal-offset bias. */
export function cascadeCovers(world: World): Float32Array {
    return shadows(world).cascadeCover;
}

/** the casting sun's resolved `numCascades` this frame ({@link lightCascades}): what sizes the cascade atlas,
 * whatever the active {@link cascadeCount}. */
export function sunCascades(world: World): number {
    return shadows(world).sunCascades;
}

/** the casting sun's `overlapProportion` this frame, floored at 0: the receiver's cascade blend band. */
export function sunOverlap(world: World): number {
    return shadows(world).sunOverlap;
}

/** The selected directional caster's `[depthBias, normalBias]` for this frame. */
export function sunBias(world: World): Float64Array {
    return shadows(world).sunBias;
}

/** The selected caster's normalized direction from the receiver toward the light. */
export function sunTowardLight(world: World): Float64Array {
    return shadows(world).sunTowardLight;
}

function createCascadeCamera(world: World): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).mode.set(eid, CameraMode.Orthographic);
    attachView(world, eid);
    return eid;
}

function syncCascadePool(world: World, shadow: ShadowRuntime, n: number): void {
    if (shadow.cascadeEids.length !== n) shadow.cascKey.fill(Number.NaN);
    while (shadow.cascadeEids.length < n) shadow.cascadeEids.push(createCascadeCamera(world));
    while (shadow.cascadeEids.length > n) {
        const eid = shadow.cascadeEids.pop()!;
        detachView(world, eid);
        world.destroy(eid);
    }
}

// Culling must use the same light-space projection as atlas rasterization.
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
    world.storage(Transform).translation.set(eid, eye[0], eye[1], eye[2]);
    world.storage(Transform).rotation.set(eid, ...q);
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
        detachView(world, eid);
        world.destroy(eid);
    }
    shadow.cascadeEids.length = 0;
    shadow.cascKey.fill(Number.NaN);
}

/** Clears cached cascade state without destroying cameras; teardown belongs to destroyCascades. */
export function resetCascades(world: World): void {
    const shadow = shadows(world);
    shadow.cascadeEids.length = 0;
    shadow.cascKey.fill(Number.NaN);
    shadow.cascadeCount = 0;
}

/**
 * Updates cascade cameras and borrowed atlas/receiver arrays before draw. Without a casting light or
 * main camera, publishes zero cascades. Perspective uses view-depth slices; orthographic uses one ground fit.
 * maximumDistance also extends the near plane toward the light to retain occluders above each slice.
 */
export function updateCascades(world: World, main: number): void {
    const shadow = shadows(world);
    const light = shadowDirectionalLight(world);
    if (light < 0 || main < 0) {
        shadow.cascadeCount = 0;
        return;
    }
    const resolution = sunResolution(world);
    const maxDist = Math.max(1e-3, world.storage(DirectionalLight).maximumDistance.get(light));
    shadow.sunBias[0] = world.storage(DirectionalLight).depthBias.get(light);
    shadow.sunBias[1] = world.storage(DirectionalLight).shadowNormalBias.get(light);
    directionalTravelDirection(world, light, _sunDir);
    shadow.sunTowardLight[0] = -_sunDir[0];
    shadow.sunTowardLight[1] = -_sunDir[1];
    shadow.sunTowardLight[2] = -_sunDir[2];
    const view = world.resource(Views).get(main);
    const aspect = view && view.height > 0 ? view.width / view.height : 1;
    composeGlobalTransform(world, main, _cascWorld);
    const mode = world.storage(Camera).mode.get(main);
    const fov = world.storage(Camera).fov.get(main);
    const size = world.storage(Camera).size.get(main);
    const near = Math.max(1e-3, world.storage(Camera).near.get(main));
    const ortho = mode === CameraMode.Orthographic;
    const cascades = lightCascades(world, light);
    const n = ortho ? 1 : cascades;
    const overlap = Math.max(0, world.storage(DirectionalLight).overlapProportion.get(light));
    const first = world.storage(DirectionalLight).firstCascadeFarBound.get(light);
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
            // widen the near edge back over the blend band, so the
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

// A 2D depth atlas avoids requiring cube-array support. Tile area tracks apparent light contribution.
// Per-face culling limits rasterization; gathering faces per mesh limits indirect draws.
// Widened frusta and tile-clamped PCF taps prevent seams and sampling neighbouring tiles.

/** the shadowed-caster ceiling: the per-frame caster array + the combo buffers size to it; the combo
 * view-slot budget (each caster claims up to 6 depth slots) is {@link MAX_COMBO_SLOTS} below */
export const MAX_POINT_CASTERS = 8;

// Reserve four cascade slots and up to 48 point-shadow face slots from the 585-slot cull uniform.
// The derived presenting-view ceiling leaves exactly these 52 depth-only slots available.
const CASCADE_RESERVE = MAX_CASCADES;
const MAX_COMBO_SLOTS = MAX_SLOTS - MAX_VIEWS - CASCADE_RESERVE;

function comboSlots(frames: PointShadowFrame[], count: number): number {
    let n = 0;
    for (let i = 0; i < count; i++) n += frames[i].spot ? 1 : 6;
    return n;
}

/**
 * World-owned point/spot shadow budget; changes apply on the next drawn frame.
 * `atlas` is the square atlas side in pixels, snapped to a power of two in [256, 4096] (default 2048).
 * `casters` is clamped to [1, {@link MAX_POINT_CASTERS}]. Excess or unpackable casters warn and cast no shadow.
 * Tile area follows `intensity·range²/dist²`; a point light uses six tiles, a spot one.
 * `hysteresis` is the fractional importance advantage required to displace an incumbent when over cap;
 * zero disables the incumbent margin.
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

// A texel-sized projection margin keeps the 3×3 PCF footprint inside each face or cone tile.
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

/** a buddy quadtree packer over a square atlas; {@link createPacker} builds it */
export interface Packer {
    /** the atlas side in pixels */
    readonly side: number;
    /** reclaim the whole atlas: the per-frame reuse path, so a dropped caster's space is free again next frame */
    reset(): void;
    /** the origin code `y·side + x` of a free `size × size` tile, or -1 when the atlas can't fit it */
    alloc(size: number): number;
}

/**
 * a buddy quadtree packer over a square atlas of side `side` pixels (a power of two), for tiles from `min` to
 * `side` pixels (powers of two). Power-of-two square tiles pack with no fragmentation when allocated
 * largest-first. Each size's free list is preallocated to every tile of that size, so `alloc` and `reset`
 * allocate nothing. Pure; the allocator the importance sizing builds on.
 */
export function createPacker(side: number, min: number): Packer {
    const levels = 32 - Math.clz32(side / min);
    // free tile origins per size level (level 0 is the whole atlas), a stack each; a tile splits into four
    // children on demand
    const free: Int32Array[] = [];
    for (let level = 0; level < levels; level++) free.push(new Int32Array(4 ** level));
    const top = new Int32Array(levels);
    const reset = (): void => {
        top.fill(0);
        free[0][0] = 0;
        top[0] = 1;
    };
    const alloc = (size: number): number => {
        const level = Math.clz32(size) - Math.clz32(side);
        if (level < 0 || level >= levels) return -1;
        if (top[level] > 0) return free[level][--top[level]];
        if (level === 0) return -1;
        const parent = alloc(size * 2);
        if (parent < 0) return -1;
        const px = parent % side;
        const py = (parent - px) / side;
        // keep the parent's own corner; free the other three quadrants for later (same-size) allocs
        const list = free[level];
        const t = top[level];
        list[t] = py * side + px + size;
        list[t + 1] = (py + size) * side + px;
        list[t + 2] = (py + size) * side + px + size;
        top[level] = t + 3;
        return parent;
    };
    reset();
    return { side, reset, alloc };
}

/**
 * Writes the spot's local -Z cone axis and lookAt basis into `out` for matching render/receiver projection.
 * `outer` is a half-angle in radians, clamped to [1°,80°]; `tilePx` sets the PCF projection margin.
 */
export function spotBasis(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    outer: number,
    tilePx: number,
    out: Pick<PointShadowFrame, "fwd" | "right" | "up" | "coneTanHalf" | "coneFov">,
): void {
    const ax = -2 * (qx * qz + qw * qy);
    const ay = -2 * (qy * qz - qw * qx);
    const az = -(1 - 2 * (qx * qx + qy * qy));
    const al = Math.sqrt(ax * ax + ay * ay + az * az) || 1;
    const fx = ax / al;
    const fy = ay / al;
    const fz = az / al;
    // up0 dodges the cone-axis-parallel degenerate; the orthonormalized `up` carries it past here
    const parallel = Math.abs(fy) > 0.99;
    const ux = 0;
    const uy = parallel ? 0 : 1;
    const uz = parallel ? 1 : 0;
    const cx = fy * uz - fz * uy;
    const cy = fz * ux - fx * uz;
    const cz = fx * uy - fy * ux;
    const cl = Math.sqrt(cx * cx + cy * cy + cz * cz) || 1;
    const rx = cx / cl;
    const ry = cy / cl;
    const rz = cz / cl;
    out.fwd[0] = fx;
    out.fwd[1] = fy;
    out.fwd[2] = fz;
    out.right[0] = rx;
    out.right[1] = ry;
    out.right[2] = rz;
    out.up[0] = ry * fz - rz * fy;
    out.up[1] = rz * fx - rx * fz;
    out.up[2] = rx * fy - ry * fx;
    // clamp the outer half-angle to [1°, 80°] so the perspective frustum stays under 180° (tan finite); widen
    // by the PCF seam margin like pointTanHalf, so a cone-edge fragment's 3×3 footprint stays inside its tile
    const clamped = Math.min(Math.max(outer, Math.PI / 180), (80 * Math.PI) / 180);
    out.coneTanHalf = Math.tan(clamped) + (2 * EDGE_TEXELS) / tilePx;
    out.coneFov = (Math.atan(out.coneTanHalf) * 360) / Math.PI;
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
 * Column-major clip-space transform for the atlas-UV rect `[u0, v0, du, dv]` at `at`.
 * Left-multiply the face viewProj to place it in the tile without changing z/w;
 * homogeneous coordinates preserve hardware near-plane clipping.
 * Writes `out`, allocating only when omitted.
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

/**
 * Reused caster placement. `tilePx` is the resolved tile side in pixels; cone basis vectors match lookAt.
 * A point uses six combos and coneTanHalf zero; a spot uses one cone combo.
 */
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
// the proj·view product before the tile transform is folded on (the combo viewProj carries the tile
// placement, so the VS emits `tileVP·world` directly — see {@link tileTransform})
const _pv = new Float32Array(16);
const _tileMat = new Float32Array(16);
const _tileVP = new Float32Array(16);

/** Borrowed tile-folded viewProjs, one matrix per active combo, filled by updatePointShadows. */
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

/** Borrowed depth-only camera eids in caster/face order; only the active combo prefix is valid. */
export function pointComboEids(world: World): number[] {
    return shadows(world).comboEids;
}

function createComboCamera(world: World): number {
    const eid = world.create();
    world.add(eid, Transform);
    world.add(eid, Camera);
    world.storage(Camera).mode.set(eid, CameraMode.Perspective);
    attachView(world, eid);
    return eid;
}

function syncComboPool(world: World, shadows: ShadowRuntime, n: number): void {
    while (shadows.comboEids.length < n) shadows.comboEids.push(createComboCamera(world));
    while (shadows.comboEids.length > n) {
        const eid = shadows.comboEids.pop()!;
        detachView(world, eid);
        world.destroy(eid);
    }
}

// a combo's look basis and vertical FOV: a spot's cone, else the point caster's cube face `face`. A spot's
// `up` is orthonormal to its `fwd`, so it serves as the lookAt up directly (same basis as up0)
const comboFwd = (frame: PointShadowFrame, face: number) =>
    frame.spot ? frame.fwd : POINT_FACES[face].fwd;
const comboUp = (frame: PointShadowFrame, face: number) =>
    frame.spot ? frame.up : POINT_FACES[face].up;
const comboFov = (frame: PointShadowFrame) => (frame.spot ? frame.coneFov : pointFov(frame.tilePx));

// write combo `ci`'s tile viewProj into `faceVP`: `tileTransform(rect at `at`) × perspective × lookAt`. The
// tile placement is folded in, so the VS emits `tileVP·world` with no manual divide and the hardware clips a
// triangle behind the face near plane.
function writeComboViewProj(
    shadow: ShadowRuntime,
    ci: number,
    frame: PointShadowFrame,
    face: number,
    rects: Float64Array,
    at: number,
): void {
    const fwd = comboFwd(frame, face);
    const up = comboUp(frame, face);
    const px = frame.pos[0];
    const py = frame.pos[1];
    const pz = frame.pos[2];
    // the caster's projection depends only on its fov and near, so it is rebuilt only when packing resizes the
    // tile or the cone or range changes: steady play, a moving light included, makes no perspective call
    const proj = shadow.casterProj[frame.slot];
    const key = frame.slot * 2;
    const fov = comboFov(frame);
    if (shadow.casterProjKey[key] !== fov || shadow.casterProjKey[key + 1] !== frame.near) {
        perspective(fov, 1, frame.near, proj);
        shadow.casterProjKey[key] = fov;
        shadow.casterProjKey[key + 1] = frame.near;
    }
    lookAt(px, py, pz, px + fwd[0], py + fwd[1], pz + fwd[2], up[0], up[1], up[2], _faceView);
    multiplyMat4(proj, _faceView, _pv);
    multiplyMat4(tileTransform(rects, _tileMat, at), _pv, _tileVP);
    shadow.faceVP.set(_tileVP, ci * 16);
}

// each cube face's camera rotation (x, y, z, w). A lookAt orientation depends only on its direction and up,
// not the eye, so the six are constants
const POINT_FACE_ROTATIONS = new Float64Array(6 * 4);
for (let face = 0; face < 6; face++) {
    const { fwd, up } = POINT_FACES[face];
    const q = lookAtRotation(0, 0, 0, fwd[0], fwd[1], fwd[2], up[0], up[1], up[2]);
    POINT_FACE_ROTATIONS.set(q, face * 4);
}

// Camera basis is right/up/-fwd; the receiver uses this same cone basis.
function poseSpotRotation(world: World, eid: number, frame: PointShadowFrame): void {
    const xx = frame.right[0];
    const xy = frame.right[1];
    const xz = frame.right[2];
    const yx = frame.up[0];
    const yy = frame.up[1];
    const yz = frame.up[2];
    const zx = -frame.fwd[0];
    const zy = -frame.fwd[1];
    const zz = -frame.fwd[2];
    const trace = xx + yy + zz;
    let qw: number;
    let qx: number;
    let qy: number;
    let qz: number;
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        qw = 0.25 / s;
        qx = (yz - zy) * s;
        qy = (zx - xz) * s;
        qz = (xy - yx) * s;
    } else if (xx > yy && xx > zz) {
        const s = 2 * Math.sqrt(1 + xx - yy - zz);
        qw = (yz - zy) / s;
        qx = 0.25 * s;
        qy = (yx + xy) / s;
        qz = (zx + xz) / s;
    } else if (yy > zz) {
        const s = 2 * Math.sqrt(1 + yy - xx - zz);
        qw = (zx - xz) / s;
        qx = (yx + xy) / s;
        qy = 0.25 * s;
        qz = (yz + zy) / s;
    } else {
        const s = 2 * Math.sqrt(1 + zz - xx - yy);
        qw = (xy - yx) / s;
        qx = (zx + xz) / s;
        qy = (yz + zy) / s;
        qz = 0.25 * s;
    }
    world.storage(Transform).rotation.set(eid, qx, qy, qz, qw);
}

// Combo culling must match the face/cone projection before atlas tile placement.
function poseCombo(world: World, eid: number, frame: PointShadowFrame, face: number): void {
    world.storage(Transform).translation.set(eid, frame.pos[0], frame.pos[1], frame.pos[2]);
    if (frame.spot) poseSpotRotation(world, eid, frame);
    else {
        const r = face * 4;
        world
            .storage(Transform)
            .rotation.set(
                eid,
                POINT_FACE_ROTATIONS[r],
                POINT_FACE_ROTATIONS[r + 1],
                POINT_FACE_ROTATIONS[r + 2],
                POINT_FACE_ROTATIONS[r + 3],
            );
    }
    world.storage(Camera).fov.set(eid, comboFov(frame));
    world.storage(Camera).near.set(eid, frame.near);
    world.storage(Camera).far.set(eid, frame.far);
}

/** destroy the pooled combo cameras + their views (at plugin dispose) */
export function destroyPointShadows(world: World): void {
    const shadow = shadows(world);
    for (const eid of shadow.comboEids) {
        detachView(world, eid);
        world.destroy(eid);
    }
    shadow.comboEids.length = 0;
}

/** Clears cached point-shadow state without destroying cameras; teardown belongs to destroyPointShadows. */
export function resetPointShadows(world: World): void {
    const shadow = shadows(world);
    shadow.comboEids.length = 0;
    shadow.lastCasters.clear();
    shadow.pendingCasters.clear();
    shadow.casterSelectionPending = false;
    shadow.capWarned = false;
    shadow.overflowWarned = false;
    shadow.slotWarned = false;
}

/** @internal Queue the selected caster set when the point atlas has encoded its draw. */
export function commitPointCastersOnSubmit(world: World): void {
    const shadow = shadows(world);
    if (shadow.casterSelectionPending) world.onFrameSubmit(shadow.commitCasters);
}

/**
 * Packs the first `count` casters into a `side`-pixel atlas, with tile area proportional to score.
 * Writes pixel sides to `sizes` and square atlas-UV rects to `rects` at `(caster·6+face)·4`.
 * Mutates `packer`; returns false if even MIN_TILE faces cannot fit. Points use six tiles, spots one.
 */
export function packCasters(
    frames: PointShadowFrame[],
    count: number,
    side: number,
    packer: Packer,
    sizes: Int32Array,
    rects: Float64Array,
): boolean {
    let maxScore = 1e-9;
    for (let i = 0; i < count; i++) maxScore = Math.max(maxScore, frames[i].score);
    // A fourfold score decrease halves tile side, preserving area proportionality.
    for (let base = side; base >= MIN_TILE; base >>= 1) {
        packer.reset();
        for (let i = 0; i < count; i++) {
            const drop = Math.max(
                0,
                Math.round(0.5 * Math.log2(maxScore / Math.max(frames[i].score, 1e-9))),
            );
            sizes[i] = Math.max(MIN_TILE, base / 2 ** Math.min(drop, 20));
        }
        // largest first, then caster and face order — buddy packs with no fragmentation
        let ok = true;
        for (let size = base; ok && size >= MIN_TILE; size >>= 1)
            for (let i = 0; ok && i < count; i++) {
                if (sizes[i] !== size) continue;
                const faces = frames[i].spot ? 1 : 6;
                for (let face = 0; face < faces; face++) {
                    const code = packer.alloc(size);
                    if (code < 0) {
                        ok = false;
                        break;
                    }
                    const x = code % side;
                    const at = (i * 6 + face) * 4;
                    rects[at] = x / side;
                    rects[at + 1] = (code - x) / side / side;
                    rects[at + 2] = size / side;
                    rects[at + 3] = size / side;
                }
            }
        if (ok) return true;
    }
    return false;
}

/**
 * Updates point/spot shadow selection and placement before draw. Writes the caller's reusable `frames`
 * pool and returns its live prefix length. Excess or unpackable casters warn and are dropped by rank.
 * Importance is `intensity·range²/dist²` at `main`; hysteresis favours the last submitted caster set.
 * Publishes combo-major viewProjs, sparse tile rects and lazily pooled depth-only cameras.
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
            // The incumbent margin prevents small camera moves from flickering the caster set.
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
        f.depthBias = source.depthBias.get(c.light);
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

    if (shadow.packer?.side !== atlas) shadow.packer = createPacker(atlas, MIN_TILE);
    const { packer, packSizes, packRects: rects } = shadow;
    let dropped = 0;
    while (count > 0 && !packCasters(frames, count, atlas, packer, packSizes, rects)) {
        count--;
        dropped++;
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
    // Combo views share the fixed cull table with presenting cameras and reserved cascade slots.
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
    // Keep the hysteresis basis at the last submitted caster set; an aborted draw must not make its
    // CPU-selected winners incumbents before their atlas reaches the frame.
    let sameCasters = shadow.lastCasters.size === count;
    for (let i = 0; sameCasters && i < count; i++)
        sameCasters = shadow.lastCasters.has(frames[i].light);
    if (sameCasters) {
        shadow.casterSelectionPending = false;
    } else {
        let samePending = shadow.pendingCasters.size === count;
        for (let i = 0; samePending && i < count; i++)
            samePending = shadow.pendingCasters.has(frames[i].light);
        if (!samePending) {
            shadow.pendingCasters.clear();
            for (let i = 0; i < count; i++) shadow.pendingCasters.add(frames[i].light);
        }
        shadow.casterSelectionPending = true;
    }

    for (let i = 0; i < count; i++) {
        const f = frames[i];
        f.tilePx = rects[f.slot * 6 * 4 + 2] * atlas;
        if (f.spot)
            spotBasis(
                world.storage(GlobalTransform).rotation.x.get(f.light),
                world.storage(GlobalTransform).rotation.y.get(f.light),
                world.storage(GlobalTransform).rotation.z.get(f.light),
                world.storage(GlobalTransform).rotation.w.get(f.light),
                world.storage(SpotLight).outerAngle.get(f.light),
                f.tilePx,
                f,
            );
    }

    syncComboPool(world, shadow, comboSlots(frames, count));

    if (shadow.faceVP.length < cap * 6 * 16) shadow.faceVP = new Float32Array(cap * 6 * 16);
    if (shadow.comboMeta.length < cap * 6 * 4) shadow.comboMeta = new Uint32Array(cap * 6 * 4);
    if (shadow.tileRects.length < cap * 6 * 4) shadow.tileRects = new Float32Array(cap * 6 * 4);
    shadow.tileRects.fill(0);
    let ci = 0;
    for (let k = 0; k < count; k++) {
        const frame = frames[k];
        const faces = frame.spot ? 1 : 6;
        for (let face = 0; face < faces; face++) {
            const at = (frame.slot * 6 + face) * 4;
            writeComboViewProj(shadow, ci, frame, face, rects, at);
            // a spot's lone combo is face 0 of its slot
            shadow.comboMeta[ci * 4] = frame.slot;
            shadow.comboMeta[ci * 4 + 1] = face;
            for (let j = 0; j < 4; j++) shadow.tileRects[at + j] = rects[at + j];
            // pose this combo's cull camera (its frustum is the pre-fold `_pv`) so the pack culls casters
            // into its slot independently
            const comboCam = shadow.comboEids[ci];
            if (comboCam !== undefined) poseCombo(world, comboCam, frame, face);
            ci++;
        }
    }
    shadow.comboCount = ci;
    return count;
}
