// StandardRenderer's shadow-atlas GPU state: the sun's CSM cascade atlas, the point/spot importance-packed atlas, and
// the 1×1 fallback + comparison sampler bound when nothing casts. Owns every buffer/texture/bind-group
// these need, the two atlas render passes (`renderPointShadows` / `renderCascades`), the color pass's
// group-1 bind group, and the getters a screen-space consumer (the fog march) binds to
// sample the same shadows standard's color pass does. `forward.ts` resolves the frame's draw list and passes
// it in; this module never reaches back into `forward.ts` at runtime (only for the `Recorded` type).

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { DEPTH_FORMAT, RenderContext, Views } from "../../core/rendering";
import type { World } from "../../engine";
import { boundPipeline } from "./bound";
import type { BundleDraw, PassBundle } from "./bundle";
import { bundleChanged, bundleDraw, newPassBundle, recordBundle } from "./bundle";
import { engineLayout } from "./engine";
import type { Recorded } from "./forward";
import { engineGroup } from "./pipelines";
import { createRegather, type Regather, SHADOW_ARG_STRIDE } from "./regather";
import type { Draw } from "./registry";
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
    pointComboEids,
    pointComboMeta,
    pointFaceVP,
    pointTileRects,
    SunShadows,
    sunBias,
    sunCascades,
    sunResolution,
} from "./shadows";

interface AtlasState {
    sunCasting: boolean;
    shadowSampler: GPUSampler | null;
    fallbackDepth: GPUTexture | null;
    fallbackView: GPUTextureView | null;
    fallbackParams: GPUBuffer | null;
    shadowReady: boolean;
    sunParams: GPUBuffer | null;
    paramsBuf: ArrayBuffer;
    paramsF32: Float32Array;
    pointAtlas: GPUTexture | null;
    pointAtlasView: GPUTextureView | null;
    pointParams: GPUBuffer | null;
    pointTileRects: GPUBuffer | null;
    pointFrames: PointShadowFrame[];
    pointFrameCount: number;
    pointBuf: ArrayBuffer;
    pointF32: Float32Array;
    pointCleared: boolean;
    faceVP: GPUBuffer | null;
    comboMeta: GPUBuffer | null;
    castDraws: { draw: Draw; r: Recorded }[];
    batchDropWarned: boolean;
    comboSlots: number[];
    comboIndices: number[];
    drawPairs: number[];
    pointRegatherPass: GPUComputePassDescriptor;
    pointShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & { view: GPUTextureView };
    pointShadowPass: GPURenderPassDescriptor;
    cascadeRegatherPass: GPUComputePassDescriptor;
    shadowBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] };
    compactVP: Float32Array;
    compactMeta: Uint32Array;
    pointBundle: PassBundle;
    pointProgram: BundleDraw[];
    cascadeBundles: PassBundle[];
    cascadeProgram: BundleDraw[];
    cascadeShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & {
        view: GPUTextureView;
    };
    cascadeShadowPass: GPURenderPassDescriptor;
    comboMissWarned: boolean;
    cascadeAtlas: GPUTexture | null;
    cascadeAtlasView: GPUTextureView | null;
    cascadeVPBuf: GPUBuffer | null;
    cascadeMetaBuf: GPUBuffer | null;
    cascadeRectsBuf: GPUBuffer | null;
    cascadeBatches: CascadeBatch[];
    shadowGroup: {
        map: GPUTextureView;
        params: GPUBuffer;
        atlas: GPUTextureView;
        group: GPUBindGroup;
    } | null;
    pointGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null;
    cascadeGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null;
    pointRegather: Regather;
    cascadeRegather: Regather;
}

const atlasStateKey = { create: createAtlasState };

function createAtlasState(world: World): AtlasState {
    const paramsBuf = new ArrayBuffer(SHADOW_PARAMS_BYTES);
    const pointBuf = new ArrayBuffer(0);
    const pointShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & {
        view: GPUTextureView;
    } = {
        view: null!,
        depthLoadOp: "clear",
        depthStoreOp: "store",
        depthClearValue: 0,
    };
    const cascadeShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & {
        view: GPUTextureView;
    } = {
        view: null!,
        depthLoadOp: "clear",
        depthStoreOp: "store",
        depthClearValue: 0,
    };
    return {
        sunCasting: false,
        shadowSampler: null,
        fallbackDepth: null,
        fallbackView: null,
        fallbackParams: null,
        shadowReady: false,
        sunParams: null,
        paramsBuf,
        paramsF32: new Float32Array(paramsBuf),
        pointAtlas: null,
        pointAtlasView: null,
        pointParams: null,
        pointTileRects: null,
        pointFrames: [],
        pointFrameCount: 0,
        pointBuf,
        pointF32: new Float32Array(pointBuf),
        pointCleared: false,
        faceVP: null,
        comboMeta: null,
        castDraws: [],
        batchDropWarned: false,
        comboSlots: [],
        comboIndices: [],
        drawPairs: [],
        pointRegatherPass: { label: "standard:pointregather" },
        pointShadowDepth,
        pointShadowPass: {
            label: "standard-pointshadow",
            colorAttachments: [],
            depthStencilAttachment: pointShadowDepth,
        },
        cascadeRegatherPass: { label: "standard:cascaderegather" },
        shadowBundleDesc: {
            label: "standard-shadow",
            colorFormats: [],
            depthStencilFormat: DEPTH_FORMAT,
            sampleCount: 1,
        },
        compactVP: new Float32Array(0),
        compactMeta: new Uint32Array(0),
        pointBundle: newPassBundle(),
        pointProgram: [],
        cascadeBundles: [],
        cascadeProgram: [],
        cascadeShadowDepth,
        cascadeShadowPass: {
            label: "standard-cascadeshadow",
            colorAttachments: [],
            depthStencilAttachment: cascadeShadowDepth,
        },
        comboMissWarned: false,
        cascadeAtlas: null,
        cascadeAtlasView: null,
        cascadeVPBuf: null,
        cascadeMetaBuf: null,
        cascadeRectsBuf: null,
        cascadeBatches: [],
        shadowGroup: null,
        pointGroup1Typed: null,
        cascadeGroup1Typed: null,
        pointRegather: createRegather(world, "point"),
        cascadeRegather: createRegather(world, "cascade"),
    };
}

/** Create this world's shadow-atlas resources during StandardRenderer initialization. */
export function initializeShadowAtlasState(world: World): void {
    world.resource(atlasStateKey);
}

// ---- sun shadows: the GPU half — the CSM cascade atlas (the CPU/ECS half — cascade cameras + fit — is in
// ./shadows). The single directional map is gone: the sun renders through the cascade atlas like the point
// atlas, and the receiver selects a cascade by view-z ----

// the sun-shadow seam, standard-internal: while live, the cascade atlas view + the per-cascade SunShadow params
// (`_atlas.sunParams`) the color pass's opaque + transparent draws sample inline via group 1. Set by
// `renderCascades` after it renders the caster depth, cleared when no sun casts (fallback → fully lit). StandardRenderer
// owns the atlas and reads its own state directly — no cross-module seam

// the no-shadow fallback bound when no light casts: a 1×1 depth texture (never sampled — `enabled: 0`
// in the all-zero params short-circuits `sampleSunShadow`) + that params buffer. StandardRenderer owns the
// comparison sampler too — one config, shared by the fallback and the real map

// the real SunShadow params standard writes each casting frame (created at warm); `shadowReady()` gates the
// render until warm has run

// the SunShadow params staging: MAX_CASCADES Cascade rows (CASCADE_FLOATS each) then the globals tail,
// every field index derived from the schemas (SUN_PARAMS, ./shade)

// ---- point-light shadows: the GPU half (face viewProjs + tile math in ./shadows) ----
//
// One fixed-size depth atlas shared by every shadowed point light (cube faces as tiles, the
// PlayCanvas model), allocated lazily on the first casting frame. `_atlas.pointParams` is the PointCaster
// uniform array the FS matches compacted lights against — always bound on group 1 (an empty slot's
// pos.w = -1 never matches a real eid, so the no-caster path reads the fallback atlas never).
// Published as "pointShadows" so a one-shot probe can pin the metadata to the TS oracle.
// `_atlas.pointAtlasView` doubles as the seam: non-null once the atlas exists.
//
// The atlas renders in one pass, one indirect draw per casting mesh (the re-gather concatenates each mesh's
// per-combo culled members into one run). `_atlas.faceVP` is the combo-major
// face viewProj uniform the VS projects by; the re-gather state (`pointRegather` etc) is below

// the per-(caster, face) allocated atlas-UV rects, indexed slot·6 + face — the receiver samples it (color
// group 1) and the atlas VS reads it for the tile-discard bounds (point group 1). Published "pointTileRects"
// so a one-shot probe can pin the allocation; (re)sized at warm when the PointShadows config is final

// this frame's ranked casters: the first `_atlas.pointFrameCount` of `_atlas.pointFrames`

// pos + nf + spotA/B/C vec4s per caster — (re)sized at warm, when the PointShadows config is final

// whether the params buffer on the GPU already holds the cleared set. This replaces reading slot 0's
// `light` lane back as a sentinel: that lane is data, not a flag, so a caster whose `light` lane happened
// to read exactly -1 would have satisfied the sentinel and skipped a clear the scene needed. The flag is
// written on the same path as every write to the buffer, so it cannot disagree with it.

/** the point-shadow atlas depth view a screen-space consumer (the fog volumetric march) binds to sample
 * the casters' shadows: the real atlas once a point/spot light casts, else the 1×1 fallback (whose empty
 * caster slots never match a light, so the march reads it as fully lit). Pairs with {@link shadowSampler}
 * + the published `"pointShadows"` caster uniform. */
export function pointAtlasView(world: World): GPUTextureView | null {
    const _atlasState = world.resource(atlasStateKey);

    return _atlasState.pointAtlasView ?? _atlasState.fallbackView;
}

/** the shared shadow comparison sampler (less-equal + linear PCF): a screen-space consumer binds it to
 * comparison-sample {@link pointAtlasView} or {@link sunShadowView}. */
export function shadowSampler(world: World): GPUSampler | null {
    return world.resource(atlasStateKey).shadowSampler;
}

/** the sun (directional) shadow map depth view a screen-space consumer (the fog volumetric march) binds
 * to sample shadowed sun shafts: the real map once the directional light's shadowMapsEnabled is on,
 * else the 1×1 fallback (whose `enabled: 0` params make `sampleSunShadow` return 1.0, so the
 * march scatters the sun unshadowed). Pairs with {@link shadowSampler} + {@link sunShadowParams}. */
export function sunShadowView(world: World): GPUTextureView | null {
    const _atlasState = world.resource(atlasStateKey);

    return _atlasState.sunCasting ? _atlasState.cascadeAtlasView : _atlasState.fallbackView;
}

/** the {@link SunShadow} params uniform a screen-space consumer binds: the real
 * light viewProj + bias when the sun casts, else the all-zero `enabled: 0` fallback. Pairs with
 * {@link sunShadowView}. */
export function sunShadowParams(world: World): GPUBuffer | null {
    const _atlasState = world.resource(atlasStateKey);

    return _atlasState.sunCasting ? _atlasState.sunParams : _atlasState.fallbackParams;
}

/** whether `resetShadowAtlas` has run (gates a lazy pipeline compile that references the shadow/point
 * group-1 layouts before they exist). */
export function shadowReady(world: World): boolean {
    return world.resource(atlasStateKey).shadowReady;
}

// The shared shadow layout exposes the same resources to every color/background pipeline, so the
// WGSL-bodied real references `sampleSunShadow` / `pointShadowOf` (shade.ts) — which read them as free
// names, the relocatable-global law — resolve against it once a typed color pipeline references it (the
// fog `fogLayout1` precedent, `extras/fog/pipeline.ts`). The sampler + point-shadow bindings are
// vertex-visible too, because a per-vertex
// surface's vs chunk (`vertex`) calls `litPbr` → `pointShadowOf`, statically reaching them from the
// vertex stage (it early-outs on `pointScale == 0` at runtime, and `textureSampleCompareLevel` is
// vertex-legal); the sun map + params stay fragment-only — `sampleSunShadow` is scaffold-called, never
// reachable from a vs chunk.
// One module-scope instance, built once (the fog `_pointCastersGpu`/`_tileRectsGpu` shape) — a fresh
// `pointCastersSchema()`/`tileRectsSchema()` call mints a new struct object each time, and this typed
// pipeline's own resolve must stay pinned to one. It remains independent from fog's schema instance and
// safe because neither resolves alongside the other in one `tgpu.resolve`
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
 * walk otherwise). The runtime bind group is {@link shadowGroup}.
 */
export const shadowLayout = tgpu
    .bindGroupLayout({
        shadowMap: { texture: d.textureDepth2d(), visibility: ["fragment"] },
        shadowSamp: { sampler: "comparison", visibility: ["vertex", "fragment"] },
        sunShadow: { uniform: SunShadow, visibility: ["fragment"] },
        pointAtlas: { texture: d.textureDepth2d(), visibility: ["vertex", "fragment"] },
        pointShadows: { uniform: _shadowTypedCasters, visibility: ["vertex", "fragment"] },
        tileRects: { uniform: _shadowTypedRects, visibility: ["vertex", "fragment"] },
    })
    .$idx(1);

// The point and cascade layouts have the same three uniforms in the same order, but the point/cascade VS
// reads each from a real bind-group layout rather than a free name. Each atlas needs its own instance:
// `faceVPsSchema`/`comboMetaSchema`/
// `tileRectsSchema` fold the **slot count** into the schema type (6·casters for the point atlas, exactly
// `MAX_CASCADES` for the cascade atlas), and a schema is sized once at first resolve. A single shared layout
// could only carry one slot count. Both layouts are vertex-only because their VS is the sole reader.
//
// Each is its own self-contained schema instance (the `_shadowTypedCasters`/`_shadowTypedRects` discipline
// above): a `FaceVPs`/`ComboMeta`/`TileRects`-named struct can't be resolved twice under the same name in one
// `tgpu.resolve` call, so the point layout's instances and the cascade layout's instances must never land in
// the same pipeline's resolve — true here, since `compileSurface`'s point pipeline and cascade pipeline
// are two independent `world.gpu.root.createRenderPipeline` calls (pipelines.ts), never combined.
const _pointTypedFaceVP = faceVPsSchema(pointCasters() * 6);
const _pointTypedCombo = comboMetaSchema(pointCasters() * 6);
const _pointTypedRects = tileRectsSchema(pointCasters() * 6);

/** the point-atlas pipeline's group-1 layout: the combo-major face viewProjs, the per-combo (caster
 * slot, face) meta, and the per-(caster, face) tile rects — all vertex-only uniforms.
 * AppConfig-folded to `6 · pointCasters()` slots at module load (the caster cap is fixed
 * before `createApp()`, like `capacity` — `checkShadowConfig`'s law). */
export const pointLayout = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _pointTypedFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _pointTypedCombo, visibility: ["vertex"] },
        tileRects: { uniform: _pointTypedRects, visibility: ["vertex"] },
    })
    .$idx(1);

const _cascadeTypedFaceVP = faceVPsSchema(MAX_CASCADES);
const _cascadeTypedCombo = comboMetaSchema(MAX_CASCADES);
const _cascadeTypedRects = tileRectsSchema(MAX_CASCADES);

/** the typed cascade-atlas pipeline's group-1 layout — {@link pointLayout}'s twin, config-folded to
 * `MAX_CASCADES` slots (fixed, unlike the point atlas's live caster count). */
export const cascadeLayout = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _cascadeTypedFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _cascadeTypedCombo, visibility: ["vertex"] },
        tileRects: { uniform: _cascadeTypedRects, visibility: ["vertex"] },
    })
    .$idx(1);

/** publish this frame's ranked point/spot casters (`ShadowCameraSystem`, from `updatePointShadows`) —
 * read by {@link renderPointShadows}. */
export function setPointFrames(world: World, frames: PointShadowFrame[], count: number): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.pointFrames = frames;
    _atlasState.pointFrameCount = count;
}

// the point pipeline's group 1: the combo tile-viewProjs + the per-combo (caster, face) meta + the
// per-(caster, face) tile rects (shared with the color group, read for the VS's tile-discard bounds), all
// uniforms. The tile placement is folded into the viewProjs, so the VS's rect read is only for the seam
// discard; the per-instance (eid, combo) rides the re-gathered list at the surface's `eids` lane

// the point atlas's re-gather instance: concatenates each casting mesh's per-combo culled members (the MeshInstance
// pack output) into one contiguous run + a per-instance combo index, so the atlas renders in one indirect
// draw per mesh. Its packed list (`pointRegather.eids()`) binds at the point pass's `eids` lane. The CSM
// cascade atlas owns a second instance (`regather.ts`); both share the singleton A/B pipelines.
/** the point/spot atlas's re-gather instance ({@link createRegather} "point") — `forward.ts`'s
 * `record`/`ShadowCameraSystem` reach it directly for the `eids` lane swap + the alloc trigger. */
export const pointRegather: import("../../engine").Resource<Regather> = {
    create: (world) => world.resource(atlasStateKey).pointRegather,
};
// the casting draws this frame (those whose surface compiled a point pipeline), filled in renderPointShadows
// with an explicit count

// warn-once (per episode — resets once every point caster shares one indirect buffer again) for a caster
// dropped because its producer owns a second indirect buffer: renderPointShadows re-gathers only the
// FIRST-seen buffer (the cascade twin's `_atlas.cascadeBatches` batches every distinct source instead — batching
// the point atlas the same way is unbuilt; this is the loud floor every drop path
// in this file owes, until a real second-indirect-buffer caster earns the batching rewrite)

// per-frame re-gather meta scratch (no per-frame alloc, explicit counts): the view slot each dense combo
// culled into with its original combo index, and the (surface,mesh) pair each casting draw owns —
// `Regather.run` reads these. Shared across the point + cascade renders (each fills then consumes them in
// turn within ShadowMapSystem)

// the atlas passes' descriptors, rewritten per frame so opening a pass mints only its WebGPU objects

// both atlas passes are depth-only, single-sample, into the same depth format, so they share one bundle
// encoder shape. The point atlas is one pass; the cascade atlas is one pass per indirect-source batch

// the compaction staging the survivor path writes when fewer cascades carry a view than the sun declares:
// a capacity pool over the cascade count, grown geometrically and rewritten in place

function compactVPScratch(world: World, count: number): Float32Array {
    const _atlasState = world.resource(atlasStateKey);

    if (_atlasState.compactVP.length < count * 16)
        _atlasState.compactVP = new Float32Array(count * 16);
    return _atlasState.compactVP;
}

function compactMetaScratch(world: World, count: number): Uint32Array {
    const _atlasState = world.resource(atlasStateKey);

    if (_atlasState.compactMeta.length < count * 4)
        _atlasState.compactMeta = new Uint32Array(count * 4);
    return _atlasState.compactMeta;
}

// warn-once (per episode — resets once a frame has zero missing combo views) for a combo view missing
// from the pool. The combo camera pool (`createComboCamera`) attaches a view per combo, so a missing
// view is a wiring bug — the camera was detached or recycled. The combo is skipped (not marshaled as
// slot 0) so the shadow pass never binds another camera's culled set into the missing combo's tile.
// Latched while misses persist, reset on the first clean frame — the `_atlas.batchDropWarned` idiom (above),
// so a persistent wiring bug warns once per episode, not once per frame (a render-loop log flood).

/**
 * filter combo camera eids to their view slots, skipping any whose View is missing (a wiring bug —
 * `createComboCamera` attaches a view per combo, so a miss means the camera was detached or recycled).
 * Writes the dense combo slots into `slots` and the original combo indices of the survivors into `indices`,
 * and returns the survivor count, so the caller can compact the `faceVP`/`comboMeta` arrays to match the new
 * dense index space. Simply not writing would shift every later combo's dense index, misaligning the atlas
 * VS's `faceVP.m[combo]` / `comboMeta.m[combo]` lookups — `indices` lets the caller compact those arrays in
 * lockstep.
 */
export function comboViewSlots(
    world: World,
    combos: number[],
    slots: number[],
    indices: number[],
): number {
    const _atlasState = world.resource(atlasStateKey);

    let count = 0;
    let missed = 0;
    for (let c = 0; c < combos.length; c++) {
        const view = world.resource(Views).get(combos[c]);
        if (view) {
            slots[count] = view.slot;
            indices[count] = c;
            count++;
        } else {
            missed++;
        }
    }
    if (missed > 0) {
        if (!_atlasState.comboMissWarned) {
            _atlasState.comboMissWarned = true;
            console.warn(
                `standard: ${missed} combo view(s) missing — skipping combo(s) (wiring bug: the combo camera pool should have attached a view per combo)`,
            );
        }
    } else {
        _atlasState.comboMissWarned = false;
    }
    return count;
}

// ---- CSM cascade atlas: the GPU half (the cascade combo cameras + fit are in ./shadows) ----
//
// A dedicated depth atlas (separate from the point atlas — Bevy's directional/point split), N cascade tiles
// in a fixed grid. Each cascade is its own frustum-culled depth view (the per-cascade cull, ./shadows poses
// the cameras); the cascade `Regather` instance concatenates each casting mesh's per-cascade culled members
// into one indirect draw per mesh, the cascade pipeline's VS projecting each into its tile. Mirrors the point
// atlas exactly, the per-cascade vs per-(caster, face) tile index the only difference.

// the cascade pipeline's group 1: the dense
// per-cascade folded tile viewProjs the VS projects by, the per-cascade meta (tile index), and the tile rects

/** the CSM cascade atlas's re-gather instance ({@link createRegather} "cascade") — the point atlas's twin. */
export const cascadeRegather: import("../../engine").Resource<Regather> = {
    create: (world) => world.resource(atlasStateKey).cascadeRegather,
};
interface CascadeBatch {
    drawArgs: GPUBuffer;
    packed: GPUBuffer;
    pairCount: number;
    /** this frame's draws: the first `count` */
    draws: { draw: Draw; r: Recorded }[];
    count: number;
}

// every slot empty: pos.w = -1 (eids are non-negative, so nothing matches)
function clearPointParams(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.pointF32.fill(0);
    for (let k = 0; k < pointCasters(); k++) _atlasState.pointF32[k * POINT_CASTER_FLOATS + 3] = -1;
}

// The shared color-pass shadow group, cached on resource identity.

/**
 * the color pass's group-1 bind group against {@link shadowLayout}: the sun map / comparison sampler /
 * params + point atlas /
 * caster params / tile rects, cached on the bound identities.
 */
export function shadowGroup(world: World): GPUBindGroup {
    const _atlasState = world.resource(atlasStateKey);

    const map = _atlasState.sunCasting ? _atlasState.cascadeAtlasView! : _atlasState.fallbackView!;
    const params = _atlasState.sunCasting ? _atlasState.sunParams! : _atlasState.fallbackParams!;
    const atlas = _atlasState.pointAtlasView ?? _atlasState.fallbackView!;
    if (
        _atlasState.shadowGroup &&
        _atlasState.shadowGroup.map === map &&
        _atlasState.shadowGroup.params === params &&
        _atlasState.shadowGroup.atlas === atlas
    ) {
        return _atlasState.shadowGroup.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(shadowLayout, {
            shadowMap: map,
            shadowSamp: _atlasState.shadowSampler!,
            sunShadow: params,
            pointAtlas: atlas,
            pointShadows: _atlasState.pointParams!,
            tileRects: _atlasState.pointTileRects!,
        }),
    );
    _atlasState.shadowGroup = { map, params, atlas, group };
    return group;
}

// Atlas bind groups cache the three uniform resources by identity.

function pointGroup1Typed(world: World): GPUBindGroup {
    const _atlasState = world.resource(atlasStateKey);

    if (
        _atlasState.pointGroup1Typed?.faceVP === _atlasState.faceVP &&
        _atlasState.pointGroup1Typed?.combo === _atlasState.comboMeta
    ) {
        return _atlasState.pointGroup1Typed!.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(pointLayout, {
            faceVP: _atlasState.faceVP!,
            comboMeta: _atlasState.comboMeta!,
            tileRects: _atlasState.pointTileRects!,
        }),
    );
    _atlasState.pointGroup1Typed = {
        faceVP: _atlasState.faceVP!,
        combo: _atlasState.comboMeta!,
        group,
    };
    return group;
}

function cascadeGroup1Typed(world: World): GPUBindGroup {
    const _atlasState = world.resource(atlasStateKey);

    if (
        _atlasState.cascadeGroup1Typed?.faceVP === _atlasState.cascadeVPBuf &&
        _atlasState.cascadeGroup1Typed?.combo === _atlasState.cascadeMetaBuf
    ) {
        return _atlasState.cascadeGroup1Typed!.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(cascadeLayout, {
            faceVP: _atlasState.cascadeVPBuf!,
            comboMeta: _atlasState.cascadeMetaBuf!,
            tileRects: _atlasState.cascadeRectsBuf!,
        }),
    );
    _atlasState.cascadeGroup1Typed = {
        faceVP: _atlasState.cascadeVPBuf!,
        combo: _atlasState.cascadeMetaBuf!,
        group,
    };
    return group;
}

/**
 * (re)allocate every shadow-atlas GPU resource standard owns — the sun-shadow seam (comparison sampler, 1×1
 * fallback, the real params buffer), the point atlas's params/tile-rects/group-1 layout+buffers, and the
 * cascade atlas's group-1 buffers — the atlas half of `prepareStandardRenderer` (the pipeline-compilation half is
 * `pipelines.ts`'s `preparePipelines`). Surviving HMR re-warms; called once per `prepareStandardRenderer`, before the
 * pipeline compiles that reference the TypeGPU layouts above.
 */
export function resetShadowAtlas(world: World, device: GPUDevice): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.shadowGroup = null;
    _atlasState.pointGroup1Typed = null;
    _atlasState.cascadeGroup1Typed = null;
    // drop any seam a prior World left behind (module-level survives HMR)
    _atlasState.sunCasting = false;
    // the comparison sampler — `greater-equal` (reverse-Z: a lit receiver is at or in front of the
    // stored occluder, i.e. ≥ its depth) + linear filtering, so each `textureSampleCompareLevel` tap is
    // a 2×2 hardware PCF. Shared by the fallback and a real shadow map
    _atlasState.shadowSampler = device.createSampler({
        label: "standard-shadow-cmp",
        compare: "greater-equal",
        magFilter: "linear",
        minFilter: "linear",
    });
    // the 1×1 fallback depth + all-zero params (enabled: 0) bound when no light casts. The map is never
    // sampled (the enabled gate short-circuits), so its undefined contents don't matter
    _atlasState.fallbackDepth?.destroy();
    _atlasState.fallbackDepth = device.createTexture({
        label: "standard-shadow-fallback",
        size: { width: 1, height: 1 },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _atlasState.fallbackView = _atlasState.fallbackDepth.createView();
    _atlasState.fallbackParams?.destroy();
    _atlasState.fallbackParams = device.createBuffer({
        label: "standard-shadow-fallback-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(
        _atlasState.fallbackParams,
        0,
        new Float32Array(SHADOW_PARAMS_BYTES / 4),
    );
    // the real params buffer standard writes each shadowed frame (viewProj + texel + depth/normal bias)
    _atlasState.sunParams?.destroy();
    _atlasState.sunParams = device.createBuffer({
        label: "standard-shadow-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // the point-shadow atlas also allocates lazily; the params buffer always exists (always bound on
    // group 1, cleared to empty slots). COPY_SRC + published by name for a metadata probe
    _atlasState.pointAtlas?.destroy();
    _atlasState.pointAtlas = null;
    _atlasState.pointAtlasView = null;
    _atlasState.pointFrames = [];
    _atlasState.pointFrameCount = 0;
    _atlasState.pointParams?.destroy();
    // both uniforms are sized from the schemas the shadow WGSL emits, so the binding and the struct the
    // receiver reads can't drift apart (checkShadowConfig catches a config change after that resolve)
    _atlasState.pointBuf = new ArrayBuffer(d.sizeOf(pointCastersSchema()));
    _atlasState.pointF32 = new Float32Array(_atlasState.pointBuf);
    _atlasState.pointParams = device.createBuffer({
        label: "standard-point-shadow-params",
        size: _atlasState.pointBuf.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    clearPointParams(world);
    device.queue.writeBuffer(_atlasState.pointParams, 0, _atlasState.pointBuf);
    _atlasState.pointCleared = true;
    world.gpu.buffers.set("pointShadows", _atlasState.pointParams);
    world.gpu.typed.set(
        "pointShadows",
        world.gpu.root
            .createBuffer(_shadowTypedCasters, _atlasState.pointParams)
            .$usage("uniform")
            .$name("standard-point-shadow-params"),
    );
    // the per-(caster, face) tile rects — bound on both the color shadow group (the receiver) and the point
    // group (the atlas VS's discard bounds). Always exists (cleared to zero), COPY_SRC + published for the
    // probe. 6 vec4 per caster
    _atlasState.pointTileRects?.destroy();
    _atlasState.pointTileRects = device.createBuffer({
        label: "standard-point-tilerects",
        size: d.sizeOf(tileRectsSchema(pointCasters() * 6)),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(
        _atlasState.pointTileRects,
        0,
        new Float32Array(pointCasters() * 6 * 4),
    );
    world.gpu.buffers.set("pointTileRects", _atlasState.pointTileRects);
    world.gpu.typed.set(
        "pointTileRects",
        world.gpu.root
            .createBuffer(_shadowTypedRects, _atlasState.pointTileRects)
            .$usage("uniform")
            .$name("standard-point-tilerects"),
    );

    // the point pipeline's group 1 buffers: the combo-major face viewProjs + the per-combo meta (the tile
    // rects bind alongside, all uniforms the point VS reads)
    _atlasState.faceVP?.destroy();
    _atlasState.faceVP = device.createBuffer({
        label: "standard-point-facevp",
        size: pointCasters() * 6 * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _atlasState.comboMeta?.destroy();
    _atlasState.comboMeta = device.createBuffer({
        label: "standard-point-combometa",
        size: pointCasters() * 6 * 16, // vec4<u32> per combo
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // the cascade pipeline's group 1 buffers: the dense
    // per-cascade folded tile viewProjs, the per-cascade meta, and the tile rects — all MAX_CASCADES-sized
    _atlasState.cascadeVPBuf?.destroy();
    _atlasState.cascadeVPBuf = device.createBuffer({
        label: "standard-cascade-vp",
        size: MAX_CASCADES * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _atlasState.cascadeMetaBuf?.destroy();
    _atlasState.cascadeMetaBuf = device.createBuffer({
        label: "standard-cascade-meta",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _atlasState.cascadeRectsBuf?.destroy();
    _atlasState.cascadeRectsBuf = device.createBuffer({
        label: "standard-cascade-rects",
        size: MAX_CASCADES * 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _atlasState.shadowReady = true;
}

/** free every shadow-atlas GPU resource standard owns (at plugin dispose): both atlases + their params/buffers,
 * the fallback + comparison sampler, and both re-gather instances. The per-camera prepass/color targets are
 * `forward.ts`'s own (`disposeStandardRenderer`). */
export function disposeShadowAtlas(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.fallbackDepth?.destroy();
    _atlasState.fallbackParams?.destroy();
    _atlasState.sunParams?.destroy();
    _atlasState.pointAtlas?.destroy();
    _atlasState.pointParams?.destroy();
    _atlasState.pointTileRects?.destroy();
    _atlasState.faceVP?.destroy();
    _atlasState.comboMeta?.destroy();
    world.resource(pointRegather).dispose();
    _atlasState.cascadeAtlas?.destroy();
    _atlasState.cascadeVPBuf?.destroy();
    _atlasState.cascadeMetaBuf?.destroy();
    _atlasState.cascadeRectsBuf?.destroy();
    world.resource(cascadeRegather).dispose();
    _atlasState.cascadeAtlas = null;
    _atlasState.cascadeAtlasView = null;
    _atlasState.cascadeVPBuf = null;
    _atlasState.cascadeMetaBuf = null;
    _atlasState.cascadeRectsBuf = null;
    _atlasState.shadowGroup = null;
    _atlasState.pointGroup1Typed = null;
    _atlasState.cascadeGroup1Typed = null;
    _atlasState.faceVP = null;
    _atlasState.comboMeta = null;
    _atlasState.pointAtlas = null;
    _atlasState.pointAtlasView = null;
    _atlasState.pointParams = null;
    _atlasState.pointTileRects = null;
    _atlasState.pointFrames = [];
    _atlasState.pointFrameCount = 0;
    _atlasState.fallbackDepth = null;
    _atlasState.fallbackView = null;
    _atlasState.fallbackParams = null;
    _atlasState.sunParams = null;
    _atlasState.sunCasting = false;
    _atlasState.shadowReady = false;
}

// the point-shadow atlas, fixed-size, allocated on the first casting frame (the bare path — no
// shadowMapsEnabled on any point/spot light — never allocates it)
function ensureAtlas(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    if (_atlasState.pointAtlas) return;
    const side = pointAtlasSize();
    _atlasState.pointAtlas = world.gpu.device.createTexture({
        label: "standard-point-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _atlasState.pointAtlasView = _atlasState.pointAtlas.createView();
}

// the cascade atlas, fixed-size (the per-cascade resolution × the grid), allocated on the first casting frame
// — the bare path (shadowMapsEnabled off on the sun) never allocates it
function ensureCascadeAtlas(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    if (_atlasState.cascadeAtlas) return;
    const side = cascadeAtlasSize(sunResolution(), sunCascades());
    _atlasState.cascadeAtlas = world.gpu.device.createTexture({
        label: "standard-cascade-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _atlasState.cascadeAtlasView = _atlasState.cascadeAtlas.createView();
}

/**
 * render every shadowed caster's depth into the atlas in **one pass, one indirect draw per casting mesh**.
 * Each combo (cube face / spot cone) culled independently through the MeshInstance pack into its own depth-only
 * view slot (the per-combo cull, `updatePointShadows` poses the cameras), then a two-pass **re-gather**
 * concatenates each casting mesh's per-combo culled members into one contiguous mesh-major run + a
 * per-instance combo index: so one indirect draw per mesh covers all its combos (the property the deleted
 * amplify trick bought, now reading per-combo *culled* counts, no over-amplification). The VS reads the
 * re-gathered packed list at the eids lane. Writes the PointCaster params the FS matches lights against,
 * uploads the CPU face viewProjs. No casters → params cleared, no pass, no atlas allocated. `frameDraws` is
 * `forward.ts`'s resolved draw list (`PrepassSystem`), shared with the color pass.
 */
export function renderPointShadows(
    world: World,
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
    capacity: number,
): void {
    const _atlasState = world.resource(atlasStateKey);
    const _pointRegather = world.resource(pointRegather);

    const encoder = world.resource(RenderContext).encoder;
    if (!encoder || !_atlasState.shadowReady) return;
    if (_atlasState.pointFrameCount === 0) {
        if (!_atlasState.pointCleared) {
            clearPointParams(world);
            world.gpu.device.queue.writeBuffer(_atlasState.pointParams!, 0, _atlasState.pointBuf);
            _atlasState.pointCleared = true;
        }
        return;
    }
    ensureAtlas(world);
    _pointRegather.ensure(pointCasters() * 6, capacity);

    // the caster params the FS samples (pos + source eid, clip planes + bias, + the spot basis —
    // right.xyz/coneTanHalf, up.xyz, fwd.xyz; coneTanHalf 0 routes the FS to the cube-face path). The tile
    // rects ride a separate uniform (uploaded below), indexed slot·6 + face
    clearPointParams(world);
    for (let k = 0; k < _atlasState.pointFrameCount; k++) {
        const caster = _atlasState.pointFrames[k];
        const o = caster.slot * POINT_CASTER_FLOATS;
        _atlasState.pointF32[o] = caster.pos[0];
        _atlasState.pointF32[o + 1] = caster.pos[1];
        _atlasState.pointF32[o + 2] = caster.pos[2];
        _atlasState.pointF32[o + 3] = caster.light;
        _atlasState.pointF32[o + 4] = caster.near;
        _atlasState.pointF32[o + 5] = caster.far;
        _atlasState.pointF32[o + 6] = caster.depthBias;
        _atlasState.pointF32[o + 7] = caster.normalBias;
        _atlasState.pointF32[o + 8] = caster.right[0];
        _atlasState.pointF32[o + 9] = caster.right[1];
        _atlasState.pointF32[o + 10] = caster.right[2];
        _atlasState.pointF32[o + 11] = caster.coneTanHalf;
        _atlasState.pointF32[o + 12] = caster.up[0];
        _atlasState.pointF32[o + 13] = caster.up[1];
        _atlasState.pointF32[o + 14] = caster.up[2];
        _atlasState.pointF32[o + 16] = caster.fwd[0];
        _atlasState.pointF32[o + 17] = caster.fwd[1];
        _atlasState.pointF32[o + 18] = caster.fwd[2];
    }
    world.gpu.device.queue.writeBuffer(_atlasState.pointParams!, 0, _atlasState.pointBuf);
    _atlasState.pointCleared = false;
    // the per-(caster, face) tile rects (sparse, slot·6 + face) the receiver samples + the VS discards by
    const tileRects = pointTileRects(world);
    world.gpu.device.queue.writeBuffer(
        _atlasState.pointTileRects!,
        0,
        tileRects as Float32Array<ArrayBuffer>,
        0,
        tileRects.length,
    );
    // the combo viewProjs the VS projects by + their (caster, face) meta (dense, CPU-side in updatePointShadows).
    // A missing combo view (wiring bug) is skipped — comboViewSlots writes the survivors' slots + original
    // indices so we compact faceVP/comboMeta to the new dense index space the re-gather's combo index uses
    const combos = pointComboEids(world);
    const C = comboViewSlots(world, combos, _atlasState.comboSlots, _atlasState.comboIndices);
    const faceVP = pointFaceVP(world);
    const comboMeta = pointComboMeta(world);
    if (C === combos.length) {
        // no misses — upload the full arrays as before
        world.gpu.device.queue.writeBuffer(
            _atlasState.faceVP!,
            0,
            faceVP as Float32Array<ArrayBuffer>,
            0,
            faceVP.length,
        );
        world.gpu.device.queue.writeBuffer(
            _atlasState.comboMeta!,
            0,
            comboMeta as Uint32Array<ArrayBuffer>,
            0,
            comboMeta.length,
        );
    } else {
        // compact to the survivors' dense index space
        const compactedVP = compactVPScratch(world, C);
        const compactedMeta = compactMetaScratch(world, C);
        for (let i = 0; i < C; i++) {
            const src = _atlasState.comboIndices[i];
            for (let k = 0; k < 16; k++) compactedVP[i * 16 + k] = faceVP[src * 16 + k];
            for (let k = 0; k < 4; k++) compactedMeta[i * 4 + k] = comboMeta[src * 4 + k];
        }
        world.gpu.device.queue.writeBuffer(
            _atlasState.faceVP!,
            0,
            compactedVP as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        world.gpu.device.queue.writeBuffer(
            _atlasState.comboMeta!,
            0,
            compactedMeta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    }

    // the casting draws (a compiled point pipeline + its point bind group) sharing the MeshInstance pack's one
    // indirect buffer — read from the Draws, not MeshInstance (the atlas stays producer-agnostic). A producer owning its own
    // indirect buffer can't ride the shared-buffer re-gather, so it's skipped (a non-MeshInstance caster is unusual)
    let D = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    let batchDropped = 0;
    for (let k = 0; k < frameCount; k++) {
        const item = frameDraws[k];
        const casts = item.r.t.point && item.r.g.point;
        if (!casts) continue;
        const buf = world.gpu.root.unwrap(item.draw.args.indirect);
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            batchDropped++;
            continue;
        }
        _atlasState.castDraws[D++] = item;
    }
    if (batchDropped > 0) {
        if (!_atlasState.batchDropWarned) {
            _atlasState.batchDropWarned = true;
            console.warn(
                `standard: ${batchDropped} point-shadow caster(s) dropped — their producer's indirect buffer differs from the first casting draw's, and renderPointShadows batches only one indirect-buffer source per frame`,
            );
        }
    } else {
        _atlasState.batchDropWarned = false;
    }
    const packed = world.gpu.buffers.get("eids");
    if (D === 0 || C === 0 || !drawArgs || !packed || pairCount === 0) return;
    _pointRegather.reserve(D);

    // the re-gather inputs: the view slot each dense combo culled into, and the (surface,mesh) pair each
    // casting draw owns. `Regather.run` concatenates each mesh's per-combo culled members into one run +
    // a per-instance combo index (Pass A per-mesh args → Pass B scatter), in one compute pass
    for (let i = 0; i < D; i++)
        _atlasState.drawPairs[i] = Math.floor(
            (_atlasState.castDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE,
        );
    _atlasState.pointRegatherPass.timestampWrites = world.gpu.span?.("standard:pointregather");
    const cpass = encoder.beginComputePass(_atlasState.pointRegatherPass);
    _pointRegather.run(
        cpass,
        drawArgs,
        packed,
        _atlasState.comboSlots,
        C,
        _atlasState.drawPairs,
        D,
        pairCount,
    );
    cpass.end();

    // one pass into the whole atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its combo's tile. The point VS projects by faceVP (not view), so the ViewUniforms buffer bound
    // at slot 0 is an unread placeholder
    _atlasState.pointShadowDepth.view = _atlasState.pointAtlasView!;
    _atlasState.pointShadowPass.timestampWrites = world.gpu.span?.("standard:pointshadow");
    const group1 = pointGroup1Typed(world);
    const args = _pointRegather.args()!;
    for (let i = 0; i < D; i++) {
        const { r } = _atlasState.castDraws[i];
        const step = bundleDraw(_atlasState.pointProgram, i);
        step.pipeline = boundPipeline(r.g, r.t.point!, r.g.point!, true, r.index) as never;
        step.layout0 = engineLayout;
        step.group0 = engineGroup(world, r.g.engineCache, 0, r.g.quant);
        step.layout1 = pointLayout;
        step.group1 = group1;
        step.layout2 = null;
        step.group2 = null;
        step.indirect = args;
        step.offset = i * SHADOW_ARG_STRIDE;
    }
    if (
        bundleChanged(
            _atlasState.pointBundle,
            _atlasState.pointProgram,
            D,
            _atlasState.shadowBundleDesc,
        )
    ) {
        recordBundle(
            world,
            _atlasState.pointBundle,
            _atlasState.pointProgram,
            D,
            _atlasState.shadowBundleDesc,
        );
    }
    const pass = encoder.beginRenderPass(_atlasState.pointShadowPass);
    if (_atlasState.pointBundle.bundle) pass.executeBundles(_atlasState.pointBundle.replay);
    pass.end();
    // one indirect draw per casting mesh — the Dawn indirect-validation floor; the per-combo
    // fan-out is collapsed by the re-gather, not amplified
    world.gpu.indirect?.("standard:pointshadow", D);
}

/**
 * render the CSM cascades into the dedicated cascade atlas, then publish the sun seam (the cascade atlas
 * + the per-cascade {@link SunShadow} params) for the color pass to sample inline: the sun's twin of
 * {@link renderPointShadows}. Each cascade is its own frustum-culled depth view (`updateCascades` poses the
 * cameras); the cascade {@link Regather} concatenates each casting mesh's per-cascade culled members into one
 * indirect draw per mesh, the cascade VS projecting each into its atlas tile. No casting sun
 * ({@link cascadeCount} 0) or no casting geometry → the seam is cleared (the fully-lit fallback), no atlas
 * allocated. `frameDraws` is `forward.ts`'s resolved draw list (`PrepassSystem`, the first `frameCount`),
 * shared with the color pass.
 */
export function renderCascades(
    world: World,
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
    capacity: number,
): void {
    const _atlasState = world.resource(atlasStateKey);
    const _cascadeRegather = world.resource(cascadeRegather);

    const encoder = world.resource(RenderContext).encoder;
    if (!encoder || !_atlasState.shadowReady) return;
    const COriginal = cascadeCount(world);
    if (COriginal === 0) {
        _atlasState.sunCasting = false;
        return;
    }
    ensureCascadeAtlas(world);
    _cascadeRegather.ensure(MAX_CASCADES, capacity);

    // filter combo (cascade) cameras to those with an attached View — a missing view is a wiring bug
    // (the cascade pool attaches a view per cascade). The survivors' original indices compact the
    // faceVP/comboMeta arrays to the new dense index space the re-gather's combo index uses; the rects
    // stay at the original cascade indices (the VS reads `tileRects.rects[meta.x]` where meta.x is the
    // original cascade index, not the dense combo index)
    const combos = cascadeComboEids(world);
    const C = comboViewSlots(world, combos, _atlasState.comboSlots, _atlasState.comboIndices);
    if (C === 0) {
        _atlasState.sunCasting = false;
        return;
    }

    // upload the per-cascade folded tile viewProjs + meta (compacted to the survivors' dense index space)
    const vp = cascadeFaceVP(world);
    const meta = cascadeMeta(world);
    if (C === COriginal) {
        world.gpu.device.queue.writeBuffer(
            _atlasState.cascadeVPBuf!,
            0,
            vp as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        world.gpu.device.queue.writeBuffer(
            _atlasState.cascadeMetaBuf!,
            0,
            meta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    } else {
        const compactedVP = compactVPScratch(world, C);
        const compactedMeta = compactMetaScratch(world, C);
        for (let i = 0; i < C; i++) {
            const src = _atlasState.comboIndices[i];
            for (let k = 0; k < 16; k++) compactedVP[i * 16 + k] = vp[src * 16 + k];
            for (let k = 0; k < 4; k++) compactedMeta[i * 4 + k] = meta[src * 4 + k];
        }
        world.gpu.device.queue.writeBuffer(
            _atlasState.cascadeVPBuf!,
            0,
            compactedVP as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        world.gpu.device.queue.writeBuffer(
            _atlasState.cascadeMetaBuf!,
            0,
            compactedMeta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    }
    // the rects are indexed by the original cascade index (meta.x), not the dense combo index
    const rects = cascadeTileRects(world);
    world.gpu.device.queue.writeBuffer(
        _atlasState.cascadeRectsBuf!,
        0,
        rects as Float32Array<ArrayBuffer>,
        0,
        COriginal * 4,
    );

    // Group culled draws by the MeshInstance pack's slot-major source, and view-independent producer draws by
    // their own indirect/eids source. Regather's pairCount=0 arm duplicates the latter across cascades.
    for (let b = 0; b < _atlasState.cascadeBatches.length; b++)
        _atlasState.cascadeBatches[b].count = 0;
    let batchCount = 0;
    const culledEids = world.gpu.buffers.get("eids");
    for (let k = 0; k < frameCount; k++) {
        const item = frameDraws[k];
        const casts = item.r.t.cascade && item.r.g.cascade;
        if (!casts) continue;
        const drawArgs = world.gpu.root.unwrap(item.draw.args.indirect);
        const pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        const packed = pairCount > 0 ? culledEids : item.r.g.eids;
        if (!packed) continue;
        let batch: CascadeBatch | undefined;
        for (let b = 0; b < batchCount; b++) {
            const candidate = _atlasState.cascadeBatches[b];
            if (
                candidate.drawArgs === drawArgs &&
                candidate.packed === packed &&
                candidate.pairCount === pairCount
            ) {
                batch = candidate;
                break;
            }
        }
        if (!batch) {
            batch = _atlasState.cascadeBatches[batchCount] ?? {
                drawArgs,
                packed,
                pairCount,
                draws: [],
                count: 0,
            };
            batch.drawArgs = drawArgs;
            batch.packed = packed;
            batch.pairCount = pairCount;
            _atlasState.cascadeBatches[batchCount] = batch;
            batchCount++;
        }
        batch.draws[batch.count++] = item;
    }
    if (batchCount === 0) {
        _atlasState.sunCasting = false; // a casting sun with no casting geometry — fully lit, like the no-cast path
        return;
    }

    // the re-gather inputs: the view slot each cascade culled into, the (surface,mesh) pair each casting draw owns
    let maxBatchDraws = 0;
    for (let b = 0; b < batchCount; b++) {
        maxBatchDraws = Math.max(maxBatchDraws, _atlasState.cascadeBatches[b].count);
    }
    _cascadeRegather.reserve(maxBatchDraws);
    const group1 = cascadeGroup1Typed(world);
    const args = _cascadeRegather.args()!;
    let totalDraws = 0;
    for (let b = 0; b < batchCount; b++) {
        const batch = _atlasState.cascadeBatches[b];
        const D = batch.count;
        for (let i = 0; i < D; i++)
            _atlasState.drawPairs[i] = Math.floor(
                (batch.draws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE,
            );
        _atlasState.cascadeRegatherPass.timestampWrites = world.gpu.span?.(
            "standard:cascaderegather",
        );
        const cpass = encoder.beginComputePass(_atlasState.cascadeRegatherPass);
        _cascadeRegather.run(
            cpass,
            batch.drawArgs,
            batch.packed,
            _atlasState.comboSlots,
            C,
            _atlasState.drawPairs,
            D,
            batch.pairCount,
            b,
        );
        cpass.end();

        _atlasState.cascadeShadowDepth.view = _atlasState.cascadeAtlasView!;
        _atlasState.cascadeShadowDepth.depthLoadOp = b === 0 ? "clear" : "load";
        _atlasState.cascadeShadowPass.timestampWrites = world.gpu.span?.("standard:cascadeshadow");
        for (let i = 0; i < D; i++) {
            const { r } = batch.draws[i];
            const step = bundleDraw(_atlasState.cascadeProgram, i);
            step.pipeline = boundPipeline(r.g, r.t.cascade!, r.g.cascade!, true, r.index) as never;
            step.layout0 = engineLayout;
            step.group0 = engineGroup(world, r.g.engineCache, 0, r.g.quant);
            step.layout1 = cascadeLayout;
            step.group1 = group1;
            step.layout2 = null;
            step.group2 = null;
            step.indirect = args;
            step.offset = i * SHADOW_ARG_STRIDE;
        }
        // one recording per batch index: a batch is a distinct indirect source, and the batch count is a
        // capacity pool that only grows
        let bundle = _atlasState.cascadeBundles[b];
        if (!bundle) {
            bundle = newPassBundle();
            _atlasState.cascadeBundles[b] = bundle;
        }
        if (bundleChanged(bundle, _atlasState.cascadeProgram, D, _atlasState.shadowBundleDesc)) {
            recordBundle(
                world,
                bundle,
                _atlasState.cascadeProgram,
                D,
                _atlasState.shadowBundleDesc,
            );
        }
        const pass = encoder.beginRenderPass(_atlasState.cascadeShadowPass);
        if (bundle.bundle) pass.executeBundles(bundle.replay);
        pass.end();
        totalDraws += D;
    }
    world.gpu.indirect?.("standard:cascadeshadow", totalDraws);

    // write the per-cascade SunShadow params + publish the seam: the receiver selects a cascade by view-z and
    // samples the cascade atlas (`sampleSunShadow`). One atlas pixel in uv (`texel`) is the PCF tap step; each
    // cascade carries its own world texel size (2·cover/resolution) for the normal-offset bias. The params
    // are compacted to the survivors' dense index space (the receiver's cascade index matches the compacted
    // faceVP/comboMeta), reading the original arrays via `_atlas.comboIndices`
    const recv = cascadeRecvVP(world);
    const tileRects = cascadeTileRects(world);
    const fars = cascadeFars(world);
    const covers = cascadeCovers(world);
    const res = sunResolution();
    _atlasState.paramsF32.fill(0);
    for (let i = 0; i < C; i++) {
        const src = _atlasState.comboIndices[i];
        const base = i * CASCADE_FLOATS;
        const viewProj = base + SUN_PARAMS.cascade.viewProj;
        for (let k = 0; k < 16; k++) _atlasState.paramsF32[viewProj + k] = recv[src * 16 + k];
        const rect = base + SUN_PARAMS.cascade.rect;
        for (let k = 0; k < 4; k++) _atlasState.paramsF32[rect + k] = tileRects[src * 4 + k];
        _atlasState.paramsF32[base + SUN_PARAMS.cascade.far] = fars[src];
        _atlasState.paramsF32[base + SUN_PARAMS.cascade.texelWorld] = (2 * covers[src]) / res;
    }
    const bias = sunBias(world);
    _atlasState.paramsF32[SUN_PARAMS.globals.count] = C;
    _atlasState.paramsF32[SUN_PARAMS.globals.overlap] = SunShadows.overlap;
    _atlasState.paramsF32[SUN_PARAMS.globals.depthBias] = bias[0];
    _atlasState.paramsF32[SUN_PARAMS.globals.enabled] = 1;
    _atlasState.paramsF32[SUN_PARAMS.globals.normalBias] = bias[1];
    // one atlas pixel in uv — the actual texture side (allocated for the fixed sunCascades()), not the live
    // count: an ortho main camera runs C = 1 into the whole atlas, so its PCF tap step is still 1 physical pixel
    _atlasState.paramsF32[SUN_PARAMS.globals.texel] = 1 / cascadeAtlasSize(res, sunCascades());
    world.gpu.device.queue.writeBuffer(
        _atlasState.sunParams!,
        0,
        _atlasState.paramsBuf,
        0,
        SHADOW_PARAMS_BYTES,
    );
    _atlasState.sunCasting = true;
}
