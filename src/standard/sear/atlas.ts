// Sear's shadow-atlas GPU state: the sun's CSM cascade atlas, the point/spot importance-packed atlas, and
// the 1×1 fallback + comparison sampler bound when nothing casts. Owns every buffer/texture/bind-group
// these need, the two atlas render passes (`renderPointShadows` / `renderCascades`), the color pass's
// group-1 bind group, and the getters a screen-space consumer (the fog march) binds to
// sample the same shadows sear's color pass does. `forward.ts` resolves the frame's draw list and passes
// it in; this module never reaches back into `forward.ts` at runtime (only for the `Recorded` type).

import tgpu from "typegpu";
import * as d from "typegpu/data";
import { Compute } from "../../engine";
import type { Draw } from "../render";
import { Render, Views } from "../render";
import { boundPipeline } from "./bound";
import type { BundleDraw, PassBundle } from "./bundle";
import { bundleChanged, bundleDraw, newPassBundle, recordBundle } from "./bundle";
import { DEPTH_FORMAT } from "./codegen";
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

// the sun-shadow seam, sear-internal: while live, the cascade atlas view + the per-cascade SunShadow params
// (`_sunParams`) the color pass's opaque + transparent draws sample inline via group 1. Set by
// `renderCascades` after it renders the caster depth, cleared when no sun casts (fallback → fully lit). Sear
// owns the atlas and reads its own state directly — no cross-module seam
let _sunCasting = false;

// the no-shadow fallback bound when no light casts: a 1×1 depth texture (never sampled — `enabled: 0`
// in the all-zero params short-circuits `sampleSunShadow`) + that params buffer. Sear owns the
// comparison sampler too — one config, shared by the fallback and the real map
let _shadowSampler: GPUSampler | null = null;
let _fallbackDepth: GPUTexture | null = null;
let _fallbackView: GPUTextureView | null = null;
let _fallbackParams: GPUBuffer | null = null;

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
// Published as "pointShadows" so a Mirror can pin the metadata to the TS oracle.
// `_pointAtlasView` doubles as the seam: non-null once the atlas exists.
//
// The atlas renders in one pass, one indirect draw per casting mesh (the re-gather concatenates each mesh's
// per-combo culled members into one run). `_faceVP` is the combo-major
// face viewProj uniform the VS projects by; the re-gather state (`pointRegather` etc) is below
let _pointAtlas: GPUTexture | null = null;
let _pointAtlasView: GPUTextureView | null = null;
let _pointParams: GPUBuffer | null = null;
// the per-(caster, face) allocated atlas-UV rects, indexed slot·6 + face — the receiver samples it (color
// group 1) and the atlas VS reads it for the tile-discard bounds (point group 1). Published "pointTileRects"
// so a Mirror can pin the allocation; (re)sized at warm when the PointShadows config is final
let _pointTileRects: GPUBuffer | null = null;
// this frame's ranked casters: the first `_pointFrameCount` of `_pointFrames`
let _pointFrames: PointShadowFrame[] = [];
let _pointFrameCount = 0;
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
    return _sunCasting ? _cascadeAtlasView : _fallbackView;
}

/** the {@link sunStructWgsl} params uniform a screen-space consumer binds: the real
 * light viewProj + bias when the sun casts, else the all-zero `enabled: 0` fallback. Pairs with
 * {@link sunShadowView}. */
export function sunShadowParams(): GPUBuffer | null {
    return _sunCasting ? _sunParams : _fallbackParams;
}

/** whether `resetShadowAtlas` has run (gates a lazy pipeline compile that references the shadow/point
 * group-1 layouts before they exist). */
export function shadowReady(): boolean {
    return _shadowReady;
}

// The shared shadow layout exposes the same resources to every color/background pipeline, so the
// WGSL-bodied real references `sampleSunShadow` / `pointShadowOf` (shade.ts) — which read them as free
// names, the relocatable-global law — resolve against it once a typed color pipeline references it (the
// fog `fogLayout1` precedent, `standard/fog/pipeline.ts`). The sampler + point-shadow bindings are
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
// the same pipeline's resolve — true here, since `compileVariant`'s point pipeline and cascade pipeline
// are two independent `Compute.root.createRenderPipeline` calls (pipelines.ts), never combined.
const _pointTypedFaceVP = faceVPsSchema(pointCasters() * 6);
const _pointTypedCombo = comboMetaSchema(pointCasters() * 6);
const _pointTypedRects = tileRectsSchema(pointCasters() * 6);

/** the point-atlas pipeline's group-1 layout: the combo-major face viewProjs, the per-combo (caster
 * slot, face) meta, and the per-(caster, face) tile rects — all vertex-only uniforms.
 * Config-folded to `6 · pointCasters()` slots at module load (the caster cap is fixed
 * before `build()`, like `capacity` — `checkShadowConfig`'s law). */
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
export function setPointFrames(frames: PointShadowFrame[], count: number): void {
    _pointFrames = frames;
    _pointFrameCount = count;
}

// the point pipeline's group 1: the combo tile-viewProjs + the per-combo (caster, face) meta + the
// per-(caster, face) tile rects (shared with the color group, read for the VS's tile-discard bounds), all
// uniforms. The tile placement is folded into the viewProjs, so the VS's rect read is only for the seam
// discard; the per-instance (eid, combo) rides the re-gathered list at the surface's `eids` lane
let _faceVP: GPUBuffer | null = null; // dense per-combo viewProjs (≤ 6 · casters mat4), uploaded per frame
let _comboMeta: GPUBuffer | null = null; // dense per-combo (caster slot, face) the VS tile-decodes

// the point atlas's re-gather instance: concatenates each casting mesh's per-combo culled members (the Part
// pack output) into one contiguous run + a per-instance combo index, so the atlas renders in one indirect
// draw per mesh. Its packed list (`pointRegather.eids()`) binds at the point pass's `eids` lane. The CSM
// cascade atlas owns a second instance (`regather.ts`); both share the singleton A/B pipelines.
/** the point/spot atlas's re-gather instance ({@link createRegather} "point") — `forward.ts`'s
 * `record`/`ShadowCameraSystem` reach it directly for the `eids` lane swap + the alloc trigger. */
export const pointRegather = createRegather("point");
// the casting draws this frame (those whose surface compiled a point pipeline), filled in renderPointShadows
// with an explicit count
const _castDraws: { draw: Draw; r: Recorded }[] = [];
// warn-once (per episode — resets once every point caster shares one indirect buffer again) for a caster
// dropped because its producer owns a second indirect buffer: renderPointShadows re-gathers only the
// FIRST-seen buffer (the cascade twin's `_cascadeBatches` batches every distinct source instead — batching
// the point atlas the same way is unbuilt; this is the loud floor every drop path
// in this file owes, until a real second-indirect-buffer caster earns the batching rewrite)
let _batchDropWarned = false;
// per-frame re-gather meta scratch (no per-frame alloc, explicit counts): the view slot each dense combo
// culled into with its original combo index, and the (surface,mesh) pair each casting draw owns —
// `Regather.run` reads these. Shared across the point + cascade renders (each fills then consumes them in
// turn within ShadowMapSystem)
const _comboSlots: number[] = [];
const _comboIndices: number[] = [];
const _drawPairs: number[] = [];

// the atlas passes' descriptors, rewritten per frame so opening a pass mints only its WebGPU objects
const _pointRegatherPass: GPUComputePassDescriptor = { label: "sear:pointregather" };
const _pointShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & {
    view: GPUTextureView;
} = {
    view: null!,
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 0,
};
const _pointShadowPass: GPURenderPassDescriptor = {
    label: "sear-pointshadow",
    colorAttachments: [],
    depthStencilAttachment: _pointShadowDepth,
};
const _cascadeRegatherPass: GPUComputePassDescriptor = { label: "sear:cascaderegather" };

// both atlas passes are depth-only, single-sample, into the same depth format, so they share one bundle
// encoder shape. The point atlas is one pass; the cascade atlas is one pass per indirect-source batch
const _shadowBundleDesc: GPURenderBundleEncoderDescriptor & { colorFormats: GPUTextureFormat[] } = {
    label: "sear-shadow",
    colorFormats: [],
    depthStencilFormat: DEPTH_FORMAT,
    sampleCount: 1,
};
// the compaction staging the survivor path writes when fewer cascades carry a view than the sun declares:
// a capacity pool over the cascade count, grown geometrically and rewritten in place
let _compactVP = new Float32Array(0);
let _compactMeta = new Uint32Array(0);

function compactVPScratch(count: number): Float32Array {
    if (_compactVP.length < count * 16) _compactVP = new Float32Array(count * 16);
    return _compactVP;
}

function compactMetaScratch(count: number): Uint32Array {
    if (_compactMeta.length < count * 4) _compactMeta = new Uint32Array(count * 4);
    return _compactMeta;
}

const _pointBundle: PassBundle = newPassBundle();
const _pointProgram: BundleDraw[] = [];
const _cascadeBundles: PassBundle[] = [];
const _cascadeProgram: BundleDraw[] = [];
const _cascadeShadowDepth: Omit<GPURenderPassDepthStencilAttachment, "view"> & {
    view: GPUTextureView;
} = {
    view: null!,
    depthLoadOp: "clear",
    depthStoreOp: "store",
    depthClearValue: 0,
};
const _cascadeShadowPass: GPURenderPassDescriptor = {
    label: "sear-cascadeshadow",
    colorAttachments: [],
    depthStencilAttachment: _cascadeShadowDepth,
};

// warn-once (per episode — resets once a frame has zero missing combo views) for a combo view missing
// from the pool. The combo camera pool (`createComboCamera`) attaches a view per combo, so a missing
// view is a wiring bug — the camera was detached or recycled. The combo is skipped (not marshaled as
// slot 0) so the shadow pass never binds another camera's culled set into the missing combo's tile.
// Latched while misses persist, reset on the first clean frame — the `_batchDropWarned` idiom (above),
// so a persistent wiring bug warns once per episode, not once per frame (a render-loop log flood).
let _comboMissWarned = false;

/**
 * filter combo camera eids to their view slots, skipping any whose View is missing (a wiring bug —
 * `createComboCamera` attaches a view per combo, so a miss means the camera was detached or recycled).
 * Writes the dense combo slots into `slots` and the original combo indices of the survivors into `indices`,
 * and returns the survivor count, so the caller can compact the `faceVP`/`comboMeta` arrays to match the new
 * dense index space. Simply not writing would shift every later combo's dense index, misaligning the atlas
 * VS's `faceVP.m[combo]` / `comboMeta.m[combo]` lookups — `indices` lets the caller compact those arrays in
 * lockstep.
 */
export function comboViewSlots(combos: number[], slots: number[], indices: number[]): number {
    let count = 0;
    let missed = 0;
    for (let c = 0; c < combos.length; c++) {
        const view = Views.get(combos[c]);
        if (view) {
            slots[count] = view.slot;
            indices[count] = c;
            count++;
        } else {
            missed++;
        }
    }
    if (missed > 0) {
        if (!_comboMissWarned) {
            _comboMissWarned = true;
            console.warn(
                `sear: ${missed} combo view(s) missing — skipping combo(s) (wiring bug: the combo camera pool should have attached a view per combo)`,
            );
        }
    } else {
        _comboMissWarned = false;
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
let _cascadeAtlas: GPUTexture | null = null;
let _cascadeAtlasView: GPUTextureView | null = null;
// the cascade pipeline's group 1: the dense
// per-cascade folded tile viewProjs the VS projects by, the per-cascade meta (tile index), and the tile rects
let _cascadeVPBuf: GPUBuffer | null = null;
let _cascadeMetaBuf: GPUBuffer | null = null;
let _cascadeRectsBuf: GPUBuffer | null = null;
/** the CSM cascade atlas's re-gather instance ({@link createRegather} "cascade") — the point atlas's twin. */
export const cascadeRegather = createRegather("cascade");
interface CascadeBatch {
    drawArgs: GPUBuffer;
    packed: GPUBuffer;
    pairCount: number;
    /** this frame's draws: the first `count` */
    draws: { draw: Draw; r: Recorded }[];
    count: number;
}
const _cascadeBatches: CascadeBatch[] = [];

// every slot empty: pos.w = -1 (eids are non-negative, so nothing matches)
function clearPointParams(): void {
    _pointF32.fill(0);
    for (let k = 0; k < pointCasters(); k++) _pointF32[k * POINT_CASTER_FLOATS + 3] = -1;
}

// The shared color-pass shadow group, cached on resource identity.
let _shadowGroup: {
    map: GPUTextureView;
    params: GPUBuffer;
    atlas: GPUTextureView;
    group: GPUBindGroup;
} | null = null;

/**
 * the color pass's group-1 bind group against {@link shadowLayout}: the sun map / comparison sampler /
 * params + point atlas /
 * caster params / tile rects, cached on the bound identities.
 */
export function shadowGroup(): GPUBindGroup {
    const map = _sunCasting ? _cascadeAtlasView! : _fallbackView!;
    const params = _sunCasting ? _sunParams! : _fallbackParams!;
    const atlas = _pointAtlasView ?? _fallbackView!;
    if (
        _shadowGroup &&
        _shadowGroup.map === map &&
        _shadowGroup.params === params &&
        _shadowGroup.atlas === atlas
    ) {
        return _shadowGroup.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(shadowLayout, {
            shadowMap: map,
            shadowSamp: _shadowSampler!,
            sunShadow: params,
            pointAtlas: atlas,
            pointShadows: _pointParams!,
            tileRects: _pointTileRects!,
        }),
    );
    _shadowGroup = { map, params, atlas, group };
    return group;
}

// Atlas bind groups cache the three uniform resources by identity.
let _pointGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null = null;
let _cascadeGroup1Typed: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null = null;

function pointGroup1Typed(): GPUBindGroup {
    if (_pointGroup1Typed?.faceVP === _faceVP && _pointGroup1Typed?.combo === _comboMeta) {
        return _pointGroup1Typed!.group;
    }
    const group = Compute.root.unwrap(
        Compute.root.createBindGroup(pointLayout, {
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
        Compute.root.createBindGroup(cascadeLayout, {
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
 * pipeline compiles that reference the TypeGPU layouts above.
 */
export function resetShadowAtlas(device: GPUDevice): void {
    _shadowGroup = null;
    _pointGroup1Typed = null;
    _cascadeGroup1Typed = null;
    // drop any seam a prior State left behind (module-level survives HMR)
    _sunCasting = false;
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
    // group 1, cleared to empty slots). COPY_SRC + published by name for a metadata Mirror
    _pointAtlas?.destroy();
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointFrames = [];
    _pointFrameCount = 0;
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
    Compute.typed.set(
        "pointShadows",
        Compute.root
            .createBuffer(_shadowTypedCasters, _pointParams)
            .$usage("uniform")
            .$name("sear-point-shadow-params"),
    );
    // the per-(caster, face) tile rects — bound on both the color shadow group (the receiver) and the point
    // group (the atlas VS's discard bounds). Always exists (cleared to zero), COPY_SRC + published for the
    // Mirror. 6 vec4 per caster
    _pointTileRects?.destroy();
    _pointTileRects = device.createBuffer({
        label: "sear-point-tilerects",
        size: d.sizeOf(tileRectsSchema(pointCasters() * 6)),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(_pointTileRects, 0, new Float32Array(pointCasters() * 6 * 4));
    Compute.buffers.set("pointTileRects", _pointTileRects);
    Compute.typed.set(
        "pointTileRects",
        Compute.root
            .createBuffer(_shadowTypedRects, _pointTileRects)
            .$usage("uniform")
            .$name("sear-point-tilerects"),
    );

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
    // the cascade pipeline's group 1 buffers: the dense
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
    _shadowGroup = null;
    _pointGroup1Typed = null;
    _cascadeGroup1Typed = null;
    _faceVP = null;
    _comboMeta = null;
    _pointAtlas = null;
    _pointAtlasView = null;
    _pointParams = null;
    _pointTileRects = null;
    _pointFrames = [];
    _pointFrameCount = 0;
    _fallbackDepth = null;
    _fallbackView = null;
    _fallbackParams = null;
    _sunParams = null;
    _sunCasting = false;
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
export function renderPointShadows(
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    if (_pointFrameCount === 0) {
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
    for (let k = 0; k < _pointFrameCount; k++) {
        const caster = _pointFrames[k];
        const o = caster.slot * POINT_CASTER_FLOATS;
        _pointF32[o] = caster.pos[0];
        _pointF32[o + 1] = caster.pos[1];
        _pointF32[o + 2] = caster.pos[2];
        _pointF32[o + 3] = caster.light;
        _pointF32[o + 4] = caster.near;
        _pointF32[o + 5] = caster.far;
        _pointF32[o + 6] = caster.depthBias;
        _pointF32[o + 7] = caster.normalBias;
        _pointF32[o + 8] = caster.right[0];
        _pointF32[o + 9] = caster.right[1];
        _pointF32[o + 10] = caster.right[2];
        _pointF32[o + 11] = caster.coneTanHalf;
        _pointF32[o + 12] = caster.up[0];
        _pointF32[o + 13] = caster.up[1];
        _pointF32[o + 14] = caster.up[2];
        _pointF32[o + 16] = caster.fwd[0];
        _pointF32[o + 17] = caster.fwd[1];
        _pointF32[o + 18] = caster.fwd[2];
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
    // the combo viewProjs the VS projects by + their (caster, face) meta (dense, CPU-side in updatePointShadows).
    // A missing combo view (wiring bug) is skipped — comboViewSlots writes the survivors' slots + original
    // indices so we compact faceVP/comboMeta to the new dense index space the re-gather's combo index uses
    const combos = pointComboEids();
    const C = comboViewSlots(combos, _comboSlots, _comboIndices);
    const faceVP = pointFaceVP();
    const comboMeta = pointComboMeta();
    if (C === combos.length) {
        // no misses — upload the full arrays as before
        Compute.device.queue.writeBuffer(
            _faceVP!,
            0,
            faceVP as Float32Array<ArrayBuffer>,
            0,
            faceVP.length,
        );
        Compute.device.queue.writeBuffer(
            _comboMeta!,
            0,
            comboMeta as Uint32Array<ArrayBuffer>,
            0,
            comboMeta.length,
        );
    } else {
        // compact to the survivors' dense index space
        const compactedVP = compactVPScratch(C);
        const compactedMeta = compactMetaScratch(C);
        for (let i = 0; i < C; i++) {
            const src = _comboIndices[i];
            for (let k = 0; k < 16; k++) compactedVP[i * 16 + k] = faceVP[src * 16 + k];
            for (let k = 0; k < 4; k++) compactedMeta[i * 4 + k] = comboMeta[src * 4 + k];
        }
        Compute.device.queue.writeBuffer(
            _faceVP!,
            0,
            compactedVP as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        Compute.device.queue.writeBuffer(
            _comboMeta!,
            0,
            compactedMeta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    }

    // the casting draws (a compiled point pipeline + its point bind group) sharing the Part pack's one
    // indirect buffer — read from the Draws, not Part (sear stays part-agnostic). A producer owning its own
    // indirect buffer can't ride the shared-buffer re-gather, so it's skipped (a non-Part caster is unusual)
    let D = 0;
    let drawArgs: GPUBuffer | null = null;
    let pairCount = 0;
    let batchDropped = 0;
    for (let k = 0; k < frameCount; k++) {
        const item = frameDraws[k];
        const casts = item.r.t.point && item.r.g.point;
        if (!casts) continue;
        const buf = Compute.root.unwrap(item.draw.args.indirect);
        if (!drawArgs) {
            drawArgs = buf;
            pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        } else if (buf !== drawArgs) {
            batchDropped++;
            continue;
        }
        _castDraws[D++] = item;
    }
    if (batchDropped > 0) {
        if (!_batchDropWarned) {
            _batchDropWarned = true;
            console.warn(
                `sear: ${batchDropped} point-shadow caster(s) dropped — their producer's indirect buffer differs from the first casting draw's, and renderPointShadows batches only one indirect-buffer source per frame`,
            );
        }
    } else {
        _batchDropWarned = false;
    }
    const packed = Compute.buffers.get("eids");
    if (D === 0 || C === 0 || !drawArgs || !packed || pairCount === 0) return;
    pointRegather.reserve(D);

    // the re-gather inputs: the view slot each dense combo culled into, and the (surface,mesh) pair each
    // casting draw owns. `Regather.run` concatenates each mesh's per-combo culled members into one run +
    // a per-instance combo index (Pass A per-mesh args → Pass B scatter), in one compute pass
    for (let i = 0; i < D; i++)
        _drawPairs[i] = Math.floor((_castDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE);
    _pointRegatherPass.timestampWrites = Compute.span?.("sear:pointregather");
    const cpass = encoder.beginComputePass(_pointRegatherPass);
    pointRegather.run(cpass, drawArgs, packed, _comboSlots, C, _drawPairs, D, pairCount);
    cpass.end();

    // one pass into the whole atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its combo's tile. The point VS projects by faceVP (not view), so the View buffer bound
    // at slot 0 is an unread placeholder
    _pointShadowDepth.view = _pointAtlasView!;
    _pointShadowPass.timestampWrites = Compute.span?.("sear:pointshadow");
    const group1 = pointGroup1Typed();
    const args = pointRegather.args()!;
    for (let i = 0; i < D; i++) {
        const { r } = _castDraws[i];
        const step = bundleDraw(_pointProgram, i);
        step.pipeline = boundPipeline(r.g, r.t.point!, r.g.point!, true, r.index) as never;
        step.layout0 = engineLayout;
        step.group0 = r.g.atlasG0;
        step.layout1 = pointLayout;
        step.group1 = group1;
        step.layout2 = null;
        step.group2 = null;
        step.indirect = args;
        step.offset = i * SHADOW_ARG_STRIDE;
    }
    if (bundleChanged(_pointBundle, _pointProgram, D, _shadowBundleDesc)) {
        recordBundle(_pointBundle, _pointProgram, D, _shadowBundleDesc);
    }
    const pass = encoder.beginRenderPass(_pointShadowPass);
    if (_pointBundle.bundle) pass.executeBundles(_pointBundle.replay);
    pass.end();
    // one indirect draw per casting mesh — the Dawn indirect-validation floor; the per-combo
    // fan-out is collapsed by the re-gather, not amplified
    Compute.indirect?.("sear:pointshadow", D);
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
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
): void {
    const encoder = Render.encoder;
    if (!encoder || !_shadowReady) return;
    const COriginal = cascadeCount();
    if (COriginal === 0) {
        _sunCasting = false;
        return;
    }
    ensureCascadeAtlas();
    cascadeRegather.ensure(MAX_CASCADES);

    // filter combo (cascade) cameras to those with an attached View — a missing view is a wiring bug
    // (the cascade pool attaches a view per cascade). The survivors' original indices compact the
    // faceVP/comboMeta arrays to the new dense index space the re-gather's combo index uses; the rects
    // stay at the original cascade indices (the VS reads `tileRects.rects[meta.x]` where meta.x is the
    // original cascade index, not the dense combo index)
    const combos = cascadeComboEids();
    const C = comboViewSlots(combos, _comboSlots, _comboIndices);
    if (C === 0) {
        _sunCasting = false;
        return;
    }

    // upload the per-cascade folded tile viewProjs + meta (compacted to the survivors' dense index space)
    const vp = cascadeFaceVP();
    const meta = cascadeMeta();
    if (C === COriginal) {
        Compute.device.queue.writeBuffer(
            _cascadeVPBuf!,
            0,
            vp as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        Compute.device.queue.writeBuffer(
            _cascadeMetaBuf!,
            0,
            meta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    } else {
        const compactedVP = compactVPScratch(C);
        const compactedMeta = compactMetaScratch(C);
        for (let i = 0; i < C; i++) {
            const src = _comboIndices[i];
            for (let k = 0; k < 16; k++) compactedVP[i * 16 + k] = vp[src * 16 + k];
            for (let k = 0; k < 4; k++) compactedMeta[i * 4 + k] = meta[src * 4 + k];
        }
        Compute.device.queue.writeBuffer(
            _cascadeVPBuf!,
            0,
            compactedVP as Float32Array<ArrayBuffer>,
            0,
            C * 16,
        );
        Compute.device.queue.writeBuffer(
            _cascadeMetaBuf!,
            0,
            compactedMeta as Uint32Array<ArrayBuffer>,
            0,
            C * 4,
        );
    }
    // the rects are indexed by the original cascade index (meta.x), not the dense combo index
    const rects = cascadeTileRects();
    Compute.device.queue.writeBuffer(
        _cascadeRectsBuf!,
        0,
        rects as Float32Array<ArrayBuffer>,
        0,
        COriginal * 4,
    );

    // Group culled draws by the Part pack's slot-major source, and view-independent producer draws by
    // their own indirect/eids source. Regather's pairCount=0 arm duplicates the latter across cascades.
    for (let b = 0; b < _cascadeBatches.length; b++) _cascadeBatches[b].count = 0;
    let batchCount = 0;
    const culledEids = Compute.buffers.get("eids");
    for (let k = 0; k < frameCount; k++) {
        const item = frameDraws[k];
        const casts = item.r.t.cascade && item.r.g.cascade;
        if (!casts) continue;
        const drawArgs = Compute.root.unwrap(item.draw.args.indirect);
        const pairCount = Math.floor((item.draw.args.viewStride ?? 0) / SHADOW_ARG_STRIDE);
        const packed = pairCount > 0 ? culledEids : item.r.g.eids;
        if (!packed) continue;
        let batch: CascadeBatch | undefined;
        for (let b = 0; b < batchCount; b++) {
            const candidate = _cascadeBatches[b];
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
            batch = _cascadeBatches[batchCount] ?? {
                drawArgs,
                packed,
                pairCount,
                draws: [],
                count: 0,
            };
            batch.drawArgs = drawArgs;
            batch.packed = packed;
            batch.pairCount = pairCount;
            _cascadeBatches[batchCount] = batch;
            batchCount++;
        }
        batch.draws[batch.count++] = item;
    }
    if (batchCount === 0) {
        _sunCasting = false; // a casting sun with no casting geometry — fully lit, like the no-cast path
        return;
    }

    // the re-gather inputs: the view slot each cascade culled into, the (surface,mesh) pair each casting draw owns
    let maxBatchDraws = 0;
    for (let b = 0; b < batchCount; b++) {
        maxBatchDraws = Math.max(maxBatchDraws, _cascadeBatches[b].count);
    }
    cascadeRegather.reserve(maxBatchDraws);
    const group1 = cascadeGroup1Typed();
    const args = cascadeRegather.args()!;
    let totalDraws = 0;
    for (let b = 0; b < batchCount; b++) {
        const batch = _cascadeBatches[b];
        const D = batch.count;
        for (let i = 0; i < D; i++)
            _drawPairs[i] = Math.floor((batch.draws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE);
        _cascadeRegatherPass.timestampWrites = Compute.span?.("sear:cascaderegather");
        const cpass = encoder.beginComputePass(_cascadeRegatherPass);
        cascadeRegather.run(
            cpass,
            batch.drawArgs,
            batch.packed,
            _comboSlots,
            C,
            _drawPairs,
            D,
            batch.pairCount,
            b,
        );
        cpass.end();

        _cascadeShadowDepth.view = _cascadeAtlasView!;
        _cascadeShadowDepth.depthLoadOp = b === 0 ? "clear" : "load";
        _cascadeShadowPass.timestampWrites = Compute.span?.("sear:cascadeshadow");
        for (let i = 0; i < D; i++) {
            const { r } = batch.draws[i];
            const step = bundleDraw(_cascadeProgram, i);
            step.pipeline = boundPipeline(r.g, r.t.cascade!, r.g.cascade!, true, r.index) as never;
            step.layout0 = engineLayout;
            step.group0 = r.g.atlasG0;
            step.layout1 = cascadeLayout;
            step.group1 = group1;
            step.layout2 = null;
            step.group2 = null;
            step.indirect = args;
            step.offset = i * SHADOW_ARG_STRIDE;
        }
        // one recording per batch index: a batch is a distinct indirect source, and the batch count is a
        // capacity pool that only grows
        let bundle = _cascadeBundles[b];
        if (!bundle) {
            bundle = newPassBundle();
            _cascadeBundles[b] = bundle;
        }
        if (bundleChanged(bundle, _cascadeProgram, D, _shadowBundleDesc)) {
            recordBundle(bundle, _cascadeProgram, D, _shadowBundleDesc);
        }
        const pass = encoder.beginRenderPass(_cascadeShadowPass);
        if (bundle.bundle) pass.executeBundles(bundle.replay);
        pass.end();
        totalDraws += D;
    }
    Compute.indirect?.("sear:cascadeshadow", totalDraws);

    // write the per-cascade SunShadow params + publish the seam: the receiver selects a cascade by view-z and
    // samples the cascade atlas (`sampleSunShadow`). One atlas pixel in uv (`texel`) is the PCF tap step; each
    // cascade carries its own world texel size (2·cover/resolution) for the normal-offset bias. The params
    // are compacted to the survivors' dense index space (the receiver's cascade index matches the compacted
    // faceVP/comboMeta), reading the original arrays via `_comboIndices`
    const recv = cascadeRecvVP();
    const tileRects = cascadeTileRects();
    const fars = cascadeFars();
    const covers = cascadeCovers();
    const res = sunResolution();
    _paramsF32.fill(0);
    for (let i = 0; i < C; i++) {
        const src = _comboIndices[i];
        const base = i * CASCADE_FLOATS;
        const viewProj = base + SUN_PARAMS.cascade.viewProj;
        for (let k = 0; k < 16; k++) _paramsF32[viewProj + k] = recv[src * 16 + k];
        const rect = base + SUN_PARAMS.cascade.rect;
        for (let k = 0; k < 4; k++) _paramsF32[rect + k] = tileRects[src * 4 + k];
        _paramsF32[base + SUN_PARAMS.cascade.far] = fars[src];
        _paramsF32[base + SUN_PARAMS.cascade.texelWorld] = (2 * covers[src]) / res;
    }
    const bias = sunBias();
    _paramsF32[SUN_PARAMS.globals.count] = C;
    _paramsF32[SUN_PARAMS.globals.overlap] = SunShadows.overlap;
    _paramsF32[SUN_PARAMS.globals.depthBias] = bias[0];
    _paramsF32[SUN_PARAMS.globals.enabled] = 1;
    _paramsF32[SUN_PARAMS.globals.normalBias] = bias[1];
    // one atlas pixel in uv — the actual texture side (allocated for the fixed sunCascades()), not the live
    // count: an ortho main camera runs C = 1 into the whole atlas, so its PCF tap step is still 1 physical pixel
    _paramsF32[SUN_PARAMS.globals.texel] = 1 / cascadeAtlasSize(res, sunCascades());
    Compute.device.queue.writeBuffer(_sunParams!, 0, _paramsBuf, 0, SHADOW_PARAMS_BYTES);
    _sunCasting = true;
}
