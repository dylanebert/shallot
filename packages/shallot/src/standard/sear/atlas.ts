// Sear's shadow-atlas GPU state: the sun's CSM cascade atlas, the point/spot importance-packed atlas, and
// the 1×1 fallback + comparison sampler bound when nothing casts. Owns every buffer/texture/bind-group
// these need, the two atlas render passes (`renderPointShadows` / `renderCascades`), the color pass's
// group-1 bind group (`shadowGroup`), and the getters a screen-space consumer (the fog march) binds to
// sample the same shadows sear's color pass does. `forward.ts` resolves the frame's draw list and passes
// it in; this module never reaches back into `forward.ts` at runtime (only for the `Recorded` type).

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Compute } from "../../engine";
import type { Draw } from "../render/core";
import { Render, Views } from "../render/core";
import { DEPTH_FORMAT, VS_FS } from "./codegen";
import { engineLayout } from "./engine";
import type { Recorded } from "./forward";
import { createRegather, SHADOW_ARG_STRIDE } from "./regather";
import {
    CASCADE_FLOATS,
    comboMetaSchema,
    faceVPsSchema,
    POINT_CASTER_FLOATS,
    pointCastersSchema,
    SHADOW_PARAMS_BYTES,
    SUN_PARAMS,
    SunShadow,
    tileRectsSchema,
} from "./shade";
import {
    cascadeAtlasSize,
    cascadeComboEids,
    cascadeCount,
    cascadeCovers,
    cascadeFaceVP,
    cascadeFars,
    cascadeMeta,
    cascadeRecvVP,
    cascadeTileRects,
    MAX_CASCADES,
    type PointShadowFrame,
    pointAtlasSize,
    pointCasters,
    pointComboCount,
    pointComboEids,
    pointComboMeta,
    pointFaceVP,
    pointTileRects,
    SunShadows,
    sunBias,
    sunCascades,
    sunResolution,
} from "./shadows";

// ---- sun shadows: the GPU half — the CSM cascade atlas (the CPU/ECS half — cascade cameras + fit — is in
// ./shadows). The single directional map is gone: the sun renders through the cascade atlas like the point
// atlas, and the receiver selects a cascade by view-z ----

// the sun-shadow seam, sear-internal: the cascade atlas view + the per-cascade SunShadow params the color
// pass's opaque + transparent draws sample inline via group 1. Set by `renderCascades` after it renders the
// caster depth, or `null` when no sun casts (fallback → fully lit). Sear owns the atlas and reads its own
// state directly — no cross-module seam
let _sun: { map: GPUTextureView; params: GPUBuffer } | null = null;

// the no-shadow fallback bound when no light casts: a 1×1 depth texture (never sampled — `enabled: 0`
// in the all-zero params short-circuits `sampleSunShadow`) + that params buffer. Sear owns the
// comparison sampler too — one config, shared by the fallback and the real map
let _shadowSampler: GPUSampler | null = null;
let _fallbackDepth: GPUTexture | null = null;
let _fallbackView: GPUTextureView | null = null;
let _fallbackParams: GPUBuffer | null = null;

// group 1 (sun shadow) — one layout shared across surfaces: the map depth texture + the comparison
// sampler + the params uniform. The color + transparent pipelines bind it; the tag + depth pipelines
// omit it. One global bind group (one sun), cached on the bound map + params identity
let _shadowBgl: GPUBindGroupLayout | null = null;
let _shadowGroup: {
    map: GPUTextureView;
    params: GPUBuffer;
    atlas: GPUTextureView;
    group: GPUBindGroup;
} | null = null;

// the real SunShadow params sear writes each casting frame (created at warm); `shadowReady()` gates the
// render until warm has run
let _shadowReady = false;
let _sunParams: GPUBuffer | null = null;

// the SunShadow params staging: MAX_CASCADES Cascade rows (CASCADE_FLOATS each) then the globals tail,
// every field index derived from the schemas (SUN_PARAMS, ./shade)
const _paramsBuf = new ArrayBuffer(SHADOW_PARAMS_BYTES);
const _paramsF32 = new Float32Array(_paramsBuf);

// ---- point-light shadows: the GPU half (face viewProjs + tile math in ./shadows) ----
//
// One fixed-size depth atlas shared by every shadowed point light (cube faces as tiles, the
// PlayCanvas model), allocated lazily on the first casting frame. `_pointParams` is the PointCaster
// uniform array the FS matches compacted lights against — always bound on group 1 (an empty slot's
// pos.w = -1 never matches a real eid, so the no-caster path reads the fallback atlas never).
// Published as "pointShadows" so the gym Mirror can pin the metadata to the TS oracle.
// `_pointAtlasView` doubles as the seam: non-null once the atlas exists.
//
// The atlas renders in one pass, one indirect draw per casting mesh (the re-gather concatenates each mesh's
// per-combo culled members into one run — gpu.md "WebGPU-specific traps"). `_faceVP` is the combo-major
// face viewProj uniform the VS projects by; the re-gather state (`pointRegather` etc) is below
let _pointAtlas: GPUTexture | null = null;
let _pointAtlasView: GPUTextureView | null = null;
let _pointParams: GPUBuffer | null = null;
// the per-(caster, face) allocated atlas-UV rects, indexed slot·6 + face — the receiver samples it (color
// group 1) and the atlas VS reads it for the tile-discard bounds (point group 1). Published "pointTileRects"
// so the gym Mirror can pin the allocation; (re)sized at warm when the PointShadows config is final
let _pointTileRects: GPUBuffer | null = null;
let _pointFrames: PointShadowFrame[] = [];
// pos + nf + spotA/B/C vec4s per caster — (re)sized at warm, when the PointShadows config is final
let _pointBuf = new ArrayBuffer(0);
let _pointF32 = new Float32Array(_pointBuf);

/** the point-shadow atlas depth view a screen-space consumer (the fog volumetric march) binds to sample
 * the casters' shadows: the real atlas once a point/spot light casts, else the 1×1 fallback (whose empty
 * caster slots never match a light, so the march reads it as fully lit). Pairs with {@link shadowSampler}
 * + the published `"pointShadows"` caster uniform. */
export function pointAtlasView(): GPUTextureView | null {
    return _pointAtlasView ?? _fallbackView;
}

/** the shared shadow comparison sampler (less-equal + linear PCF): a screen-space consumer binds it to
 * comparison-sample {@link pointAtlasView} or {@link sunShadowView}. */
export function shadowSampler(): GPUSampler | null {
    return _shadowSampler;
}

/** the sun (directional) shadow map depth view a screen-space consumer (the fog volumetric march) binds
 * to sample shadowed sun shafts: the real map once the sun casts (a `Shadow` on the directional light),
 * else the 1×1 fallback (whose `enabled: 0` params make {@link sunShadowWgsl} return 1.0, so the
 * march scatters the sun unshadowed). Pairs with {@link shadowSampler} + {@link sunShadowParams}. */
export function sunShadowView(): GPUTextureView | null {
    return _sun?.map ?? _fallbackView;
}

/** the {@link sunStructWgsl} params uniform a screen-space consumer binds: the real
 * light viewProj + bias when the sun casts, else the all-zero `enabled: 0` fallback. Pairs with
 * {@link sunShadowView}. */
export function sunShadowParams(): GPUBuffer | null {
    return _sun?.params ?? _fallbackParams;
}

/** whether `resetShadowAtlas` has run (gates a lazy pipeline compile that references the shadow/point
 * group-1 layouts before they exist). */
export function shadowReady(): boolean {
    return _shadowReady;
}

/** the sun-shadow group-1 layout (map + comparison sampler + params uniform) every color/background
 * pipeline binds group 1 against. `null` until {@link resetShadowAtlas} runs. */
export function shadowLayout(): GPUBindGroupLayout | null {
    return _shadowBgl;
}

// the typed twin of `shadowLayout()`/`_shadowBgl` (4a-ii-c-2): the same six bindings, same order, so the
// WGSL-bodied real references `sampleSunShadow` / `pointShadowOf` (shade.ts) — which read them as free
// names, the relocatable-global law — resolve against it once a typed color pipeline references it (the
// fog `fogLayout1` precedent, `standard/fog/pipeline.ts`). Visibility mirrors the raw `_shadowBgl`
// exactly (c-3b): the sampler + point-shadow bindings are vertex-visible too, because a per-vertex
// surface's vs chunk (`vertex`) calls `litPbr` → `pointShadowOf`, statically reaching them from the
// vertex stage (it early-outs on `pointScale == 0` at runtime, and `textureSampleCompareLevel` is
// vertex-legal); the sun map + params stay fragment-only — `sampleSunShadow` is scaffold-called, never
// reachable from a vs chunk.
// One module-scope instance, built once (the fog `_pointCastersGpu`/`_tileRectsGpu` shape) — a fresh
// `pointCastersSchema()`/`tileRectsSchema()` call mints a new struct object each time, and this typed
// pipeline's own resolve must stay pinned to one. This is the THIRD independent instance in the codebase
// (fog's own, sear's raw `casterWgsl()` chunk) — each one's own standalone resolve, matching the raw
// path's existing precedent (`casterWgsl`/`pointShadowCode` mint their own too) rather than a new
// anti-pattern. Safe only because none of the three ever resolves alongside another in one `tgpu.resolve`
// call (a same-resolve collision would suffix the second `PointCasters`/`TileRects` name) — that
// invariant isn't machine-checked (not in `splice.test.ts`'s pairwise matrix), so a future consumer that
// tries to combine two of these resolves in one shader module must give one a fresh, deliberately-shared
// instance instead.
const _shadowTypedCasters = pointCastersSchema();
const _shadowTypedRects = tileRectsSchema(pointCasters() * 6);

/**
 * the typed group-1 shadow layout a typed color pipeline references to force `shadowMap` / `shadowSamp` /
 * `sunShadow` / `pointAtlas` / `pointShadows` / `tileRects` into scope (the forcing-touch law — the free
 * names inside `sampleSunShadow`/`pointShadowOf`'s WGSL bodies are invisible to `tgpu.resolve`'s call-graph
 * walk otherwise). The runtime bind group is {@link shadowGroupTyped} — the same underlying
 * textures/buffers `shadowGroup()` binds.
 */
export const shadowLayoutTyped = tgpu
    .bindGroupLayout({
        shadowMap: { texture: d.textureDepth2d(), visibility: ["fragment"] },
        shadowSamp: { sampler: "comparison", visibility: ["vertex", "fragment"] },
        sunShadow: { uniform: SunShadow, visibility: ["fragment"] },
        pointAtlas: { texture: d.textureDepth2d(), visibility: ["vertex", "fragment"] },
        pointShadows: { uniform: _shadowTypedCasters, visibility: ["vertex", "fragment"] },
        tileRects: { uniform: _shadowTypedRects, visibility: ["vertex", "fragment"] },
    })
    .$idx(1);

/** the point/cascade atlas pipelines' group-1 layout (face viewProjs + combo meta + tile rects). `null`
 * until {@link resetShadowAtlas} runs. */
export function pointLayout(): GPUBindGroupLayout | null {
    return _pointBgl;
}

// the typed twins of `pointLayout()`/`_pointBgl` (4a-ii-c-3a-3): the same three uniforms, same order, so a
// typed point/cascade VS (`pipelines.ts`'s `typedShadowVs`) reads `faceVP`/`comboMeta`/`tileRects` off a
// real bind-group layout rather than a free name. Unlike `shadowLayoutTyped` (one shared color-pass shadow
// group), the point and cascade atlases each need their OWN instance: `faceVPsSchema`/`comboMetaSchema`/
// `tileRectsSchema` fold the **slot count** into the schema type (6·casters for the point atlas, exactly
// `MAX_CASCADES` for the cascade atlas — `pointShadowCode`'s own `slots` split, codegen.ts), and a schema is
// sized once at first resolve (the `casterWgsl`/`_folded` config-lock precedent above) — a single shared
// layout could only carry one of the two slot counts. Fragment-only would be wrong here (the raw `_pointBgl`
// is vertex-only — the point/cascade VS is the sole reader, no fs samples this group), so both typed twins
// stay `["vertex"]`, matching `_pointBgl`'s raw entries exactly.
//
// Each is its own self-contained schema instance (the `_shadowTypedCasters`/`_shadowTypedRects` discipline
// above): a `FaceVPs`/`ComboMeta`/`TileRects`-named struct can't be resolved twice under the same name in one
// `tgpu.resolve` call, so the point layout's instances and the cascade layout's instances must never land in
// the same pipeline's resolve — true here, since `compileTypedVariant`'s point pipeline and cascade pipeline
// are two independent `Compute.root.createRenderPipeline` calls (pipelines.ts), never combined.
const _pointTypedFaceVP = faceVPsSchema(pointCasters() * 6);
const _pointTypedCombo = comboMetaSchema(pointCasters() * 6);
const _pointTypedRects = tileRectsSchema(pointCasters() * 6);

/** the typed point-atlas pipeline's group-1 layout: the combo-major face viewProjs, the per-combo (caster
 * slot, face) meta, and the per-(caster, face) tile rects — all vertex-only uniforms, the typed twin of
 * {@link pointLayout}. Config-folded to `6 · pointCasters()` slots at module load (the caster cap is fixed
 * before `build()`, like `capacity` — `checkShadowConfig`'s law). */
export const pointLayoutTyped = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _pointTypedFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _pointTypedCombo, visibility: ["vertex"] },
        tileRects: { uniform: _pointTypedRects, visibility: ["vertex"] },
    })
    .$idx(1);

const _cascadeTypedFaceVP = faceVPsSchema(MAX_CASCADES);
const _cascadeTypedCombo = comboMetaSchema(MAX_CASCADES);
const _cascadeTypedRects = tileRectsSchema(MAX_CASCADES);

/** the typed cascade-atlas pipeline's group-1 layout — {@link pointLayoutTyped}'s twin, config-folded to
 * `MAX_CASCADES` slots (fixed, unlike the point atlas's live caster count). */
export const cascadeLayoutTyped = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _cascadeTypedFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _cascadeTypedCombo, visibility: ["vertex"] },
        tileRects: { uniform: _cascadeTypedRects, visibility: ["vertex"] },
    })
    .$idx(1);

/** publish this frame's ranked point/spot casters (`ShadowCameraSystem`, from `updatePointShadows`) —
 * read by {@link renderPointShadows}. */
export function setPointFrames(frames: PointShadowFrame[]): void {
    _pointFrames = frames;
}

// the point pipeline's group 1: the combo tile-viewProjs + the per-combo (caster, face) meta + the
// per-(caster, face) tile rects (shared with the color group, read for the VS's tile-discard bounds), all
// uniforms. The tile placement is folded into the viewProjs, so the VS's rect read is only for the seam
// discard; the per-instance (eid, combo) rides the re-gathered list at the surface's `eids` lane
let _pointBgl: GPUBindGroupLayout | null = null;
let _faceVP: GPUBuffer | null = null; // dense per-combo viewProjs (≤ 6 · casters mat4), uploaded per frame
let _comboMeta: GPUBuffer | null = null; // dense per-combo (caster slot, face) the VS tile-decodes
let _pointGroup1: {
    faceVP: GPUBuffer;
    combo: GPUBuffer;
    rects: GPUBuffer;
    group: GPUBindGroup;
} | null = null;

// the point atlas's re-gather instance: concatenates each casting mesh's per-combo culled members (the Part
// pack output) into one contiguous run + a per-instance combo index, so the atlas renders in one indirect
// draw per mesh. Its packed list (`pointRegather.eids()`) binds at the point pass's `eids` lane. The CSM
// cascade atlas owns a second instance (`regather.ts`); both share the singleton A/B pipelines.
/** the point/spot atlas's re-gather instance ({@link createRegather} "point") — `forward.ts`'s
 * `record`/`ShadowCameraSystem` reach it directly for the `eids` lane swap + the alloc trigger. */
export const pointRegather = createRegather("point");
// the casting draws this frame (those whose surface compiled a point pipeline), filled in renderPointShadows
const _castDraws: { draw: Draw; r: Recorded }[] = [];
// per-frame re-gather meta scratch (no per-frame alloc): the view slot each dense combo culled into, and the
// (surface,mesh) pair each casting draw owns — `Regather.run` reads these. Shared across the point + cascade
// renders (each fills then consumes them in turn within ShadowMapSystem)
const _comboSlots: number[] = [];
const _drawPairs: number[] = [];

// ---- CSM cascade atlas: the GPU half (the cascade combo cameras + fit are in ./shadows) ----
//
// A dedicated depth atlas (separate from the point atlas — Bevy's directional/point split), N cascade tiles
// in a fixed grid. Each cascade is its own frustum-culled depth view (the per-cascade cull, ./shadows poses
// the cameras); the cascade `Regather` instance concatenates each casting mesh's per-cascade culled members
// into one indirect draw per mesh, the cascade pipeline's VS projecting each into its tile. Mirrors the point
// atlas exactly, the per-cascade vs per-(caster, face) tile index the only difference.
let _cascadeAtlas: GPUTexture | null = null;
let _cascadeAtlasView: GPUTextureView | null = null;
// the cascade pipeline's group 1 (same 3-uniform shape as the point group 1, reuses `_pointBgl`): the dense
// per-cascade folded tile viewProjs the VS projects by, the per-cascade meta (tile index), and the tile rects
let _cascadeVPBuf: GPUBuffer | null = null;
let _cascadeMetaBuf: GPUBuffer | null = null;
let _cascadeRectsBuf: GPUBuffer | null = null;
let _cascadeGroup1: {
    faceVP: GPUBuffer;
    combo: GPUBuffer;
    rects: GPUBuffer;
    group: GPUBindGroup;
} | null = null;
/** the CSM cascade atlas's re-gather instance ({@link createRegather} "cascade") — the point atlas's twin. */
export const cascadeRegather = createRegather("cascade");
// the casting draws this frame whose surface compiled a cascade pipeline, filled in renderCascades
const _cascadeCastDraws: { draw: Draw; r: Recorded }[] = [];

// every slot empty: pos.w = -1 (eids are non-negative, so nothing matches)
function clearPointParams(): void {
    _pointF32.fill(0);
    for (let k = 0; k < pointCasters(); k++) _pointF32[k * POINT_CASTER_FLOATS + 3] = -1;
}

/**
 * the color pass's group-1 bind group: the sun map + comparison sampler + params, plus the point atlas +
 * its params + tile rects, cached on the bound identities (real once something casts, else the shared 1×1
 * fallback for the sun map/params — the point atlas fields are always bound, an empty caster slot matching
 * no eid).
 */
export function shadowGroup(): GPUBindGroup {
    const map = _sun?.map ?? _fallbackView!;
    const params = _sun?.params ?? _fallbackParams!;
    const atlas = _pointAtlasView ?? _fallbackView!;
    if (
        _shadowGroup &&
        _shadowGroup.map === map &&
        _shadowGroup.params === params &&
        _shadowGroup.atlas === atlas
    ) {
        return _shadowGroup.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-shadow",
        layout: _shadowBgl!,
        entries: [
            { binding: 0, resource: map },
            { binding: 1, resource: _shadowSampler! },
            { binding: 2, resource: { buffer: params } },
            { binding: 3, resource: atlas },
            { binding: 4, resource: { buffer: _pointParams! } },
            { binding: 5, resource: { buffer: _pointTileRects! } },
        ],
    });
    _shadowGroup = { map, params, atlas, group };
    return group;
}

// the typed twin of `_shadowGroup` (4a-ii-c-3b): the same six resources against `shadowLayoutTyped`,
// unwrapped raw so a typed draw registers it with `.with(shadowLayoutTyped, group)` — cached on the same
// bound identities
let _shadowGroupTyped: {
    map: GPUTextureView;
    params: GPUBuffer;
    atlas: GPUTextureView;
    group: GPUBindGroup;
} | null = null;

/**
 * the typed color pass's group-1 bind group ({@link shadowGroup}'s typed twin, against
 * {@link shadowLayoutTyped}): the same underlying sun map / comparison sampler / params + point atlas /
 * caster params / tile rects, cached on the bound identities.
 */
export function shadowGroupTyped(): GPUBindGroup {
    const map = _sun?.map ?? _fallbackView!;
    const params = _sun?.params ?? _fallbackParams!;
    const atlas = _pointAtlasView ?? _fallbackView!;
    if (
        _shadowGroupTyped &&
        _shadowGroupTyped.map === map &&
        _shadowGroupTyped.params === params &&
        _shadowGroupTyped.atlas === atlas
    ) {
        return _shadowGroupTyped.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(shadowLayoutTyped, {
            shadowMap: map,
            shadowSamp: _shadowSampler!,
            sunShadow: params,
            pointAtlas: atlas,
            pointShadows: _pointParams!,
            tileRects: _pointTileRects!,
        }),
    );
    _shadowGroupTyped = { map, params, atlas, group };
    return group;
}

// the typed twins of `_pointGroup1` / `_cascadeGroup1` (built below): the same three uniforms against
// `pointLayoutTyped` / `cascadeLayoutTyped`, cached on the bound identities like their raw twins
let _pointGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null = null;
let _cascadeGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null = null;

function pointGroup1Typed(): GPUBindGroup {
    if (_pointGroup1Typed?.faceVP === _faceVP && _pointGroup1Typed?.combo === _comboMeta) {
        return _pointGroup1Typed!.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(pointLayoutTyped, {
            faceVP: _faceVP!,
            comboMeta: _comboMeta!,
            tileRects: _pointTileRects!,
        }),
    );
    _pointGroup1Typed = { faceVP: _faceVP!, combo: _comboMeta!, group };
    return group;
}

function cascadeGroup1Typed(): GPUBindGroup {
    if (
        _cascadeGroup1Typed?.faceVP === _cascadeVPBuf &&
        _cascadeGroup1Typed?.combo === _cascadeMetaBuf
    ) {
        return _cascadeGroup1Typed!.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(cascadeLayoutTyped, {
            faceVP: _cascadeVPBuf!,
            comboMeta: _cascadeMetaBuf!,
            tileRects: _cascadeRectsBuf!,
        }),
    );
    _cascadeGroup1Typed = { faceVP: _cascadeVPBuf!, combo: _cascadeMetaBuf!, group };
    return group;
}

/**
 * (re)allocate every shadow-atlas GPU resource sear owns — the sun-shadow seam (comparison sampler, 1×1
 * fallback, the real params buffer), the point atlas's params/tile-rects/group-1 layout+buffers, and the
 * cascade atlas's group-1 buffers — the atlas half of `prepareSear` (the pipeline-compilation half is
 * `pipelines.ts`'s `preparePipelines`). Surviving HMR re-warms; called once per `prepareSear`, before the
 * pipeline compiles that reference {@link shadowLayout} / {@link pointLayout}.
 */
export function resetShadowAtlas(device: GPUDevice): void {
    _shadowGroup = null;
    _shadowGroupTyped = null;
    _pointGroup1Typed = null;
    _cascadeGroup1Typed = null;
    // drop any seam a prior State left behind (module-level survives HMR)
    _sun = null;
    // the comparison sampler — `greater-equal` (reverse-Z: a lit receiver is at or in front of the
    // stored occluder, i.e. ≥ its depth) + linear filtering, so each `textureSampleCompareLevel` tap is
    // a 2×2 hardware PCF. Shared by the fallback and a real shadow map
    _shadowSampler = device.createSampler({
        label: "sear-shadow-cmp",
        compare: "greater-equal",
        magFilter: "linear",
        minFilter: "linear",
    });
    // the 1×1 fallback depth + all-zero params (enabled: 0) bound when no light casts. The map is never
    // sampled (the enabled gate short-circuits), so its undefined contents don't matter
    _fallbackDepth?.destroy();
    _fallbackDepth = device.createTexture({
        label: "sear-shadow-fallback",
        size: { width: 1, height: 1 },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _fallbackView = _fallbackDepth.createView();
    _fallbackParams?.destroy();
    _fallbackParams = device.createBuffer({
        label: "sear-shadow-fallback-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(_fallbackParams, 0, new Float32Array(SHADOW_PARAMS_BYTES / 4));
    // the real params buffer sear writes each shadowed frame (viewProj + texel + depth/normal bias)
    _sunParams?.destroy();
    _sunParams = device.createBuffer({
        label: "sear-shadow-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // the point-shadow atlas also allocates lazily; the params buffer always exists (always bound on
    // group 1, cleared to empty slots). COPY_SRC + published by name for the gym's metadata Mirror
    _pointAtlas?.destroy();
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointFrames = [];
    _pointParams?.destroy();
    // both uniforms are sized from the schemas the shadow WGSL emits, so the binding and the struct the
    // receiver reads can't drift apart (checkShadowConfig catches a config change after that resolve)
    _pointBuf = new ArrayBuffer(d.sizeOf(pointCastersSchema()));
    _pointF32 = new Float32Array(_pointBuf);
    _pointParams = device.createBuffer({
        label: "sear-point-shadow-params",
        size: _pointBuf.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    clearPointParams();
    device.queue.writeBuffer(_pointParams, 0, _pointBuf);
    Compute.buffers.set("pointShadows", _pointParams);
    // the per-(caster, face) tile rects — bound on both the color shadow group (the receiver) and the point
    // group (the atlas VS's discard bounds). Always exists (cleared to zero), COPY_SRC + published for the
    // gym Mirror. 6 vec4 per caster
    _pointTileRects?.destroy();
    _pointTileRects = device.createBuffer({
        label: "sear-point-tilerects",
        size: d.sizeOf(tileRectsSchema(pointCasters() * 6)),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(_pointTileRects, 0, new Float32Array(pointCasters() * 6 * 4));
    Compute.buffers.set("pointTileRects", _pointTileRects);
    _shadowBgl = device.createBindGroupLayout({
        label: "sear-shadow",
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
            // the sampler + point-shadow bindings are vertex-visible too: a per-vertex surface's vs
            // chunk calls lightFactor → pointFactor → pointShadowOf, so they're statically reachable
            // from the vertex stage (it early-outs on pointScale == 0 at runtime, and
            // textureSampleCompareLevel is vertex-legal — the reason the FS uses the Level variant).
            // The sun map + params (0, 2) stay fragment-only: sampleSunShadow is scaffold-called,
            // never reachable from a vs chunk. The tile rects (5) are part of pointShadowOf, so vertex-visible
            { binding: 1, visibility: VS_FS, sampler: { type: "comparison" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            { binding: 3, visibility: VS_FS, texture: { sampleType: "depth" } },
            { binding: 4, visibility: VS_FS, buffer: { type: "uniform" } },
            { binding: 5, visibility: VS_FS, buffer: { type: "uniform" } },
        ],
    });

    // the point pipeline's group 1 buffers: the combo-major face viewProjs + the per-combo meta (the tile
    // rects bind alongside, all uniforms the point VS reads)
    _faceVP?.destroy();
    _faceVP = device.createBuffer({
        label: "sear-point-facevp",
        size: pointCasters() * 6 * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _comboMeta?.destroy();
    _comboMeta = device.createBuffer({
        label: "sear-point-combometa",
        size: pointCasters() * 6 * 16, // vec4<u32> per combo
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _pointBgl = device.createBindGroupLayout({
        label: "sear-point",
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        ],
    });
    _pointGroup1 = null;
    // the cascade pipeline's group 1 buffers (same shape as the point group 1, reuses `_pointBgl`): the dense
    // per-cascade folded tile viewProjs, the per-cascade meta, and the tile rects — all MAX_CASCADES-sized
    _cascadeVPBuf?.destroy();
    _cascadeVPBuf = device.createBuffer({
        label: "sear-cascade-vp",
        size: MAX_CASCADES * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeMetaBuf?.destroy();
    _cascadeMetaBuf = device.createBuffer({
        label: "sear-cascade-meta",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeRectsBuf?.destroy();
    _cascadeRectsBuf = device.createBuffer({
        label: "sear-cascade-rects",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _cascadeGroup1 = null;
    _shadowReady = true;
}

/** free every shadow-atlas GPU resource sear owns (at plugin dispose): both atlases + their params/buffers,
 * the fallback + comparison sampler, and both re-gather instances. The per-camera prepass/color targets are
 * `forward.ts`'s own (`disposeSear`). */
export function disposeShadowAtlas(): void {
    _fallbackDepth?.destroy();
    _fallbackParams?.destroy();
    _sunParams?.destroy();
    _pointAtlas?.destroy();
    _pointParams?.destroy();
    _pointTileRects?.destroy();
    _faceVP?.destroy();
    _comboMeta?.destroy();
    pointRegather.dispose();
    _cascadeAtlas?.destroy();
    _cascadeVPBuf?.destroy();
    _cascadeMetaBuf?.destroy();
    _cascadeRectsBuf?.destroy();
    cascadeRegather.dispose();
    _cascadeAtlas = null;
    _cascadeAtlasView = null;
    _cascadeVPBuf = null;
    _cascadeMetaBuf = null;
    _cascadeRectsBuf = null;
    _cascadeGroup1 = null;
    _shadowGroupTyped = null;
    _pointGroup1Typed = null;
    _cascadeGroup1Typed = null;
    _faceVP = null;
    _comboMeta = null;
    _pointGroup1 = null;
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointParams = null;
    _pointTileRects = null;
    _pointFrames = [];
    _fallbackDepth = null;
    _fallbackView = null;
    _fallbackParams = null;
    _sunParams = null;
    _sun = null;
    _shadowReady = false;
}

// the point-shadow atlas, fixed-size, allocated on the first casting frame (the bare path — no
// `Shadow` on any point light — never allocates it)
function ensureAtlas(): void {
    if (_pointAtlas) return;
    const side = pointAtlasSize();
    _pointAtlas = Compute.device.createTexture({
        label: "sear-point-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _pointAtlasView = _pointAtlas.createView();
}

// the cascade atlas, fixed-size (the per-cascade resolution × the grid), allocated on the first casting frame
// — the bare path (no `Shadow` on the sun) never allocates it
function ensureCascadeAtlas(): void {
    if (_cascadeAtlas) return;
    const side = cascadeAtlasSize(sunResolution(), sunCascades());
    _cascadeAtlas = Compute.device.createTexture({
        label: "sear-cascade-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _cascadeAtlasView = _cascadeAtlas.createView();
}

// the cascade pipeline's group 1 — per-cascade tile viewProjs + meta + rects, the point group 1's twin
// (reuses `_pointBgl`). Cached on the bound identities (the buffers are stable post-warm, so this hits once)
function cascadeGroup1(): GPUBindGroup {
    if (
        _cascadeGroup1 &&
        _cascadeGroup1.faceVP === _cascadeVPBuf &&
        _cascadeGroup1.combo === _cascadeMetaBuf &&
        _cascadeGroup1.rects === _cascadeRectsBuf
    ) {
        return _cascadeGroup1.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-cascade-group1",
        layout: _pointBgl!,
        entries: [
            { binding: 0, resource: { buffer: _cascadeVPBuf! } },
            { binding: 1, resource: { buffer: _cascadeMetaBuf! } },
            { binding: 2, resource: { buffer: _cascadeRectsBuf! } },
        ],
    });
    _cascadeGroup1 = {
        faceVP: _cascadeVPBuf!,
        combo: _cascadeMetaBuf!,
        rects: _cascadeRectsBuf!,
        group,
    };
    return group;
}

// the point pipeline's group 1 — combo tile-viewProjs + combo meta + the tile rects, all uniforms (the
// per-instance (eid, combo) rides the re-gathered list at the eids lane). Cached on the bound identities
function pointGroup1(): GPUBindGroup {
    if (
        _pointGroup1 &&
        _pointGroup1.faceVP === _faceVP &&
        _pointGroup1.combo === _comboMeta &&
        _pointGroup1.rects === _pointTileRects
    ) {
        return _pointGroup1.group;
    }
    const group = Compute.device.createBindGroup({
        label: "sear-point-group1",
        layout: _pointBgl!,
        entries: [
            { binding: 0, resource: { buffer: _faceVP! } },
            { binding: 1, resource: { buffer: _comboMeta! } },
            { binding: 2, resource: { buffer: _pointTileRects! } },
        ],
    });
    _pointGroup1 = { faceVP: _faceVP!, combo: _comboMeta!, rects: _pointTileRects!, group };
    return group;
}

/**
 * render every shadowed caster's depth into the atlas in **one pass, one indirect draw per casting mesh**.
 * Each combo (cube face / spot cone) culled independently through the Part pack into its own depth-only
 * view slot (the per-combo cull, `updatePointShadows` poses the cameras), then a two-pass **re-gather**
 * concatenates each casting mesh's per-combo culled members into one contiguous mesh-major run + a
 * per-instance combo index: so one indirect draw per mesh covers all its combos (the property the deleted
 * amplify trick bought, now reading per-combo *culled* counts, no over-amplification). The VS reads the
 * re-gathered packed list at the eids lane. Writes the PointCaster params the FS matches lights against,
 * uploads the CPU face viewProjs. No casters → params cleared, no pass, no atlas allocated. `frameDraws` is
 * `forward.ts`'s resolved draw list (`PrepassSystem`), shared with the color pass.
 */
export function renderPointShadows(frameDraws: { draw: Draw; r: Recorded }[]): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    if (_pointFrames.length === 0) {
        if (_pointF32[3] !== -1) {
            clearPointParams();
            Compute.device.queue.writeBuffer(_pointParams!, 0, _pointBuf);
        }
        return;
    }
    ensureAtlas();
    pointRegather.ensure(pointCasters() * 6);

    // the caster params the FS samples (pos + source eid, clip planes + bias, + the spot basis —
    // right.xyz/coneTanHalf, up.xyz, fwd.xyz; coneTanHalf 0 routes the FS to the cube-face path). The tile
    // rects ride a separate uniform (uploaded below), indexed slot·6 + face
    clearPointParams();
    for (const frame of _pointFrames) {
        const o = frame.slot * POINT_CASTER_FLOATS;
        _pointF32[o] = frame.pos[0];
        _pointF32[o + 1] = frame.pos[1];
        _pointF32[o + 2] = frame.pos[2];
        _pointF32[o + 3] = frame.light;
        _pointF32[o + 4] = frame.near;
        _pointF32[o + 5] = frame.far;
        _pointF32[o + 6] = frame.depthBias;
        _pointF32[o + 7] = frame.normalBias;
        _pointF32[o + 8] = frame.right[0];
        _pointF32[o + 9] = frame.right[1];
        _pointF32[o + 10] = frame.right[2];
        _pointF32[o + 11] = frame.coneTanHalf;
        _pointF32[o + 12] = frame.up[0];
        _pointF32[o + 13] = frame.up[1];
        _pointF32[o + 14] = frame.up[2];
        _pointF32[o + 16] = frame.fwd[0];
        _pointF32[o + 17] = frame.fwd[1];
        _pointF32[o + 18] = frame.fwd[2];
    }
    Compute.device.queue.writeBuffer(_pointParams!, 0, _pointBuf);
    // the per-(caster, face) tile rects (sparse, slot·6 + face) the receiver samples + the VS discards by
    const tileRects = pointTileRects();
    Compute.device.queue.writeBuffer(
        _pointTileRects!,
        0,
        tileRects as Float32Array<ArrayBuffer>,
        0,
        tileRects.length,
    );
    // the combo viewProjs the VS projects by + their (caster, face) meta (dense, CPU-side in updatePointShadows)
    const faceVP = pointFaceVP();
    Compute.device.queue.writeBuffer(
        _faceVP!,
        0,
        faceVP as Float32Array<ArrayBuffer>,
        0,
        faceVP.length,
    );
    const comboMeta = pointComboMeta();
    Compute.device.queue.writeBuffer(
        _comboMeta!,
        0,
        comboMeta as Uint32Array<ArrayBuffer>,
        0,
        comboMeta.length,
    );

    // the casting draws (a compiled point pipeline + its point bind group) sharing the Part pack's one
    // indirect buffer — read from the Draws, not Part (sear stays part-agnostic). A producer owning its own
    // indirect buffer can't ride the shared-buffer re-gather, so it's skipped (a non-Part caster is unusual)
    _castDraws.length = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    for (const item of frameDraws) {
        const casts =
            "c" in item.r
                ? item.r.c.point && item.r.entry.pointGroup
                : item.r.t.point && item.r.g.point;
        if (!casts) continue;
        const buf = item.draw.args.indirect;
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            continue;
        }
        _castDraws.push(item);
    }
    const D = _castDraws.length;
    const C = pointComboCount();
    const packed = Compute.buffers.get("eids");
    if (D === 0 || C === 0 || !drawArgs || !packed || pairCount === 0) return;

    // the re-gather inputs: the view slot each dense combo culled into, and the (surface,mesh) pair each
    // casting draw owns. `Regather.run` concatenates each mesh's per-combo culled members into one run +
    // a per-instance combo index (Pass A per-mesh args → Pass B scatter), in one compute pass
    const combos = pointComboEids();
    _comboSlots.length = 0;
    for (let c = 0; c < C; c++) _comboSlots.push(Views.get(combos[c])?.slot ?? 0);
    _drawPairs.length = 0;
    for (let i = 0; i < D; i++)
        _drawPairs.push(Math.floor((_castDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE));
    const cpass = encoder.beginComputePass({
        label: "sear:pointregather",
        timestampWrites: Compute.span?.("sear:pointregather"),
    });
    pointRegather.run(cpass, drawArgs, packed, _comboSlots, _drawPairs, pairCount);
    cpass.end();

    // one pass into the whole atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its combo's tile. The point VS projects by faceVP (not view), so the bound View buffer
    // (slot 0's, `record`'s eidsSwap) is an unread placeholder
    const pass = encoder.beginRenderPass({
        label: "sear-pointshadow",
        timestampWrites: Compute.span?.("sear:pointshadow"),
        colorAttachments: [],
        depthStencilAttachment: {
            view: _pointAtlasView!,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 0,
        },
    });
    const group1 = pointGroup1();
    for (let i = 0; i < D; i++) {
        const { r } = _castDraws[i];
        if ("c" in r) {
            pass.setPipeline(r.c.point!);
            pass.setBindGroup(0, r.entry.pointGroup!);
            pass.setBindGroup(1, group1);
            pass.setIndexBuffer(r.index, "uint32");
            pass.drawIndexedIndirect(pointRegather.args()!, i * SHADOW_ARG_STRIDE);
        } else {
            // a typed draw: engine group 0 (slot 0's unread-View instance), the typed atlas group 1,
            // and the surface's own depth-shape group 2 with the eids lane swapped to the re-gathered
            // list. The group registers under BOTH group-2 layout objects — a vs-chunk surface's own
            // reads resolve against `layout` while the depth vertex pull resolves against
            // `layout.depthVariant`, and whichever object the resolution recorded at idx 2 must find
            // the (structurally identical, WebGPU-group-equivalent) bind group
            r.t
                .point!.with(pass)
                .with(engineLayout, r.g.atlasG0)
                .with(pointLayoutTyped, pointGroup1Typed())
                .with(r.g.layout, r.g.point!)
                .with(r.g.layout.depthVariant, r.g.point!)
                .withIndexBuffer(r.index, "uint32")
                .drawIndexedIndirect(pointRegather.args()!, i * SHADOW_ARG_STRIDE);
        }
    }
    pass.end();
    // one indirect draw per casting mesh — the Dawn indirect-validation floor (gpu.md); the per-combo
    // fan-out is collapsed by the re-gather, not amplified
    Compute.indirect?.("sear:pointshadow", D);
}

/**
 * render the CSM cascades into the dedicated cascade atlas, then publish the sun seam (`_sun` → the cascade
 * atlas + the per-cascade {@link SunShadow} params) for the color pass to sample inline: the sun's twin of
 * {@link renderPointShadows}. Each cascade is its own frustum-culled depth view (`updateCascades` poses the
 * cameras); the cascade {@link Regather} concatenates each casting mesh's per-cascade culled members into one
 * indirect draw per mesh, the cascade VS projecting each into its atlas tile. No casting sun
 * ({@link cascadeCount} 0) or no casting geometry → `_sun = null` (the fully-lit fallback), no atlas allocated.
 * `frameDraws` is `forward.ts`'s resolved draw list (`PrepassSystem`), shared with the color pass.
 */
export function renderCascades(frameDraws: { draw: Draw; r: Recorded }[]): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    const C = cascadeCount();
    if (C === 0) {
        _sun = null;
        return;
    }
    ensureCascadeAtlas();
    cascadeRegather.ensure(MAX_CASCADES);

    // upload the per-cascade folded tile viewProjs + meta + rects (CPU-computed in updateCascades)
    const vp = cascadeFaceVP();
    Compute.device.queue.writeBuffer(_cascadeVPBuf!, 0, vp as Float32Array<ArrayBuffer>, 0, C * 16);
    const meta = cascadeMeta();
    Compute.device.queue.writeBuffer(
        _cascadeMetaBuf!,
        0,
        meta as Uint32Array<ArrayBuffer>,
        0,
        C * 4,
    );
    const rects = cascadeTileRects();
    Compute.device.queue.writeBuffer(
        _cascadeRectsBuf!,
        0,
        rects as Float32Array<ArrayBuffer>,
        0,
        C * 4,
    );

    // the casting draws (a compiled cascade pipeline + its cascade bind group) sharing the Part pack's one
    // indirect buffer — same part-agnostic gather as the point path
    _cascadeCastDraws.length = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    for (const item of frameDraws) {
        const casts =
            "c" in item.r
                ? item.r.c.cascade && item.r.entry.cascadeGroup
                : item.r.t.cascade && item.r.g.cascade;
        if (!casts) continue;
        const buf = item.draw.args.indirect;
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            continue;
        }
        _cascadeCastDraws.push(item);
    }
    const D = _cascadeCastDraws.length;
    const packed = Compute.buffers.get("eids");
    if (D === 0 || !drawArgs || !packed || pairCount === 0) {
        _sun = null; // a casting sun with no casting geometry — fully lit, like the no-cast path
        return;
    }

    // the re-gather inputs: the view slot each cascade culled into, the (surface,mesh) pair each casting draw owns
    const combos = cascadeComboEids();
    _comboSlots.length = 0;
    for (let c = 0; c < C; c++) _comboSlots.push(Views.get(combos[c])?.slot ?? 0);
    _drawPairs.length = 0;
    for (let i = 0; i < D; i++)
        _drawPairs.push(
            Math.floor((_cascadeCastDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE),
        );
    const cpass = encoder.beginComputePass({
        label: "sear:cascaderegather",
        timestampWrites: Compute.span?.("sear:cascaderegather"),
    });
    cascadeRegather.run(cpass, drawArgs, packed, _comboSlots, _drawPairs, pairCount);
    cpass.end();

    // one pass into the cascade atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its cascade's tile. The cascade VS projects by the folded tile viewProj (not view), so
    // the bound View buffer (slot 0's, `record`'s eidsSwap) is an unread placeholder
    const pass = encoder.beginRenderPass({
        label: "sear-cascadeshadow",
        timestampWrites: Compute.span?.("sear:cascadeshadow"),
        colorAttachments: [],
        depthStencilAttachment: {
            view: _cascadeAtlasView!,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            depthClearValue: 0,
        },
    });
    const group1 = cascadeGroup1();
    for (let i = 0; i < D; i++) {
        const { r } = _cascadeCastDraws[i];
        if ("c" in r) {
            pass.setPipeline(r.c.cascade!);
            pass.setBindGroup(0, r.entry.cascadeGroup!);
            pass.setBindGroup(1, group1);
            pass.setIndexBuffer(r.index, "uint32");
            pass.drawIndexedIndirect(cascadeRegather.args()!, i * SHADOW_ARG_STRIDE);
        } else {
            // the point pass's typed shape, cascade-flavored (see renderPointShadows for the dual
            // group-2 registration why)
            r.t
                .cascade!.with(pass)
                .with(engineLayout, r.g.atlasG0)
                .with(cascadeLayoutTyped, cascadeGroup1Typed())
                .with(r.g.layout, r.g.cascade!)
                .with(r.g.layout.depthVariant, r.g.cascade!)
                .withIndexBuffer(r.index, "uint32")
                .drawIndexedIndirect(cascadeRegather.args()!, i * SHADOW_ARG_STRIDE);
        }
    }
    pass.end();
    Compute.indirect?.("sear:cascadeshadow", D);

    // write the per-cascade SunShadow params + publish the seam: the receiver selects a cascade by view-z and
    // samples the cascade atlas (`sampleSunShadow`). One atlas pixel in uv (`texel`) is the PCF tap step; each
    // cascade carries its own world texel size (2·cover/resolution) for the normal-offset bias
    const recv = cascadeRecvVP();
    const tileRects = cascadeTileRects();
    const fars = cascadeFars();
    const covers = cascadeCovers();
    const res = sunResolution();
    _paramsF32.fill(0);
    for (let i = 0; i < C; i++) {
        const base = i * CASCADE_FLOATS;
        _paramsF32.set(recv.subarray(i * 16, i * 16 + 16), base + SUN_PARAMS.cascade.viewProj);
        _paramsF32.set(tileRects.subarray(i * 4, i * 4 + 4), base + SUN_PARAMS.cascade.rect);
        _paramsF32[base + SUN_PARAMS.cascade.far] = fars[i];
        _paramsF32[base + SUN_PARAMS.cascade.texelWorld] = (2 * covers[i]) / res;
    }
    const bias = sunBias();
    _paramsF32[SUN_PARAMS.globals.count] = C;
    _paramsF32[SUN_PARAMS.globals.overlap] = SunShadows.overlap;
    _paramsF32[SUN_PARAMS.globals.depthBias] = bias.depthBias;
    _paramsF32[SUN_PARAMS.globals.enabled] = 1;
    _paramsF32[SUN_PARAMS.globals.normalBias] = bias.normalBias;
    // one atlas pixel in uv — the actual texture side (allocated for the fixed sunCascades()), not the live
    // count: an ortho main camera runs C = 1 into the whole atlas, so its PCF tap step is still 1 physical pixel
    _paramsF32[SUN_PARAMS.globals.texel] = 1 / cascadeAtlasSize(res, sunCascades());
    Compute.device.queue.writeBuffer(_sunParams!, 0, _paramsBuf, 0, SHADOW_PARAMS_BYTES);
    _sun = { map: _cascadeAtlasView!, params: _sunParams! };
}
