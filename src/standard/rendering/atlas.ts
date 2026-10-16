import tgpu from "typegpu";
import * as d from "typegpu/data";
import { DEPTH_FORMAT, Views } from "../../core/rendering";
import type { World } from "../../engine";
import {
    beginComputePass,
    beginRenderPass,
    endComputePass,
    endRenderPass,
    executeRenderBundles,
} from "../../engine/runtime";
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
    commitPointCastersOnSubmit,
    MAX_CASCADES,
    MAX_POINT_CASTERS,
    type PointShadowFrame,
    pointAtlasSize,
    pointCasters,
    pointComboEids,
    pointComboMeta,
    pointFaceVP,
    pointTileRects,
    sunBias,
    sunCascades,
    sunOverlap,
    sunResolution,
    sunTowardLight,
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
    pointGroup1: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null;
    cascadeGroup1: { faceVP: GPUBuffer; combo: GPUBuffer; group: GPUBindGroup } | null;
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
        pointGroup1: null,
        cascadeGroup1: null,
        pointRegather: createRegather(world, "point"),
        cascadeRegather: createRegather(world, "cascade"),
    };
}

export function initializeShadowAtlasState(world: World): void {
    world.resource(atlasStateKey);
}

// Cascade cameras and fits live in ./shadows.

// Caster data is not a cleared-buffer flag. Update pointCleared on every params-buffer write
// so a data value cannot suppress a needed clear.

/** the point-shadow atlas depth view a screen-space consumer (the fog volumetric march) binds to sample
 * the casters' shadows: the real atlas once a point/spot light casts, else the 1×1 fallback (whose empty
 * caster slots never match a light, so the march reads it as fully lit). Pairs with {@link shadowSampler}
 * + {@link pointShadowCasters} and {@link pointShadowRects}. */
export function pointAtlasView(world: World): GPUTextureView | null {
    const _atlasState = world.resource(atlasStateKey);

    return _atlasState.pointAtlasView ?? _atlasState.fallbackView;
}

/** World-owned point/spot caster uniform, including empty slots when no light casts. Valid after renderer warm-up. */
export function pointShadowCasters(world: World): GPUBuffer | null {
    return world.resource(atlasStateKey).pointParams;
}

/** World-owned point/spot atlas tile-rectangle uniform. Resolve again after renderer warm-up. */
export function pointShadowRects(world: World): GPUBuffer | null {
    return world.resource(atlasStateKey).pointTileRects;
}

/** Shared reverse-Z greater-equal comparison sampler with linear PCF for pointAtlasView or sunShadowView. */
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

// Per-vertex litPbr statically reaches the point receiver even when pointScale is zero at runtime,
// so its sampler/point bindings need vertex visibility. The sun receiver is scaffold-called only
// from fragments, so sun bindings remain fragment-only.
// Each schema factory call creates a distinct struct object. Reuse these instances within a resolve;
// resolving a second same-named PointCasters/TileRects instance would suffix its WGSL name and break
// free-name references. Fog resolves separately; consumers combined in one module must share instances.
const _shadowCasters = pointCastersSchema();
const _shadowRects = tileRectsSchema(MAX_POINT_CASTERS * 6);

/**
 * the group-1 shadow layout a color pipeline references to force `shadowMap` / `shadowSamp` /
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
        pointShadows: { uniform: _shadowCasters, visibility: ["vertex", "fragment"] },
        tileRects: { uniform: _shadowRects, visibility: ["vertex", "fragment"] },
    })
    .$idx(1);

// A schema's slot count is fixed at first resolve: point arrays hold 6*MAX_POINT_CASTERS entries,
// cascade arrays MAX_CASCADES. One layout cannot serve both sizes. Keep the two schema sets in
// separate pipeline resolves as well, or their shared FaceVPs/ComboMeta/TileRects names collide.
const _pointFaceVP = faceVPsSchema(MAX_POINT_CASTERS * 6);
const _pointCombo = comboMetaSchema(MAX_POINT_CASTERS * 6);
const _pointRects = tileRectsSchema(MAX_POINT_CASTERS * 6);

/** the point-atlas pipeline's group-1 layout: the combo-major face viewProjs, the per-combo (caster
 * slot, face) meta, and the per-(caster, face) tile rects — all vertex-only uniforms.
 * Sized to `6 · MAX_POINT_CASTERS` slots; the `PointShadows.casters` limit only caps the ranked casters. */
export const pointLayout = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _pointFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _pointCombo, visibility: ["vertex"] },
        tileRects: { uniform: _pointRects, visibility: ["vertex"] },
    })
    .$idx(1);

const _cascadeFaceVP = faceVPsSchema(MAX_CASCADES);
const _cascadeCombo = comboMetaSchema(MAX_CASCADES);
const _cascadeRects = tileRectsSchema(MAX_CASCADES);

/** the cascade-atlas pipeline's group-1 layout — {@link pointLayout}'s twin, sized to
 * `MAX_CASCADES` slots. */
export const cascadeLayout = tgpu
    .bindGroupLayout({
        faceVP: { uniform: _cascadeFaceVP, visibility: ["vertex"] },
        comboMeta: { uniform: _cascadeCombo, visibility: ["vertex"] },
        tileRects: { uniform: _cascadeRects, visibility: ["vertex"] },
    })
    .$idx(1);

/** publish this frame's ranked point/spot casters (`ShadowCameraSystem`, from `updatePointShadows`) —
 * read by {@link renderPointShadows}. */
export function setPointFrames(world: World, frames: PointShadowFrame[], count: number): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.pointFrames = frames;
    _atlasState.pointFrameCount = count;
}

/** the point/spot atlas's re-gather instance ({@link createRegather} "point") — `forward.ts`'s
 * `record`/`ShadowCameraSystem` reach it directly for the `eids` lane swap + the alloc trigger. */
export const pointRegather: import("../../engine").Resource<Regather> = {
    create: (world) => world.resource(atlasStateKey).pointRegather,
};

// Point re-gather accepts only the first indirect source and warns once while other sources are
// dropped, resetting on a clean frame. Cascades instead batch every distinct indirect source.

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

// Missing combo views must not alias slot zero's culled set. Warn once until a clean frame.

/**
 * Writes attached combo view slots and their original indices into `slots` and `indices`.
 * Missing views indicate a detached/recycled combo camera and warn once per episode until a clean
 * frame; they are skipped rather than mapped to slot zero. Returns the valid prefix length; callers must compact
 * faceVP/comboMeta by the same indices to preserve dense combo addressing.
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
    for (let k = 0; k < MAX_POINT_CASTERS; k++)
        _atlasState.pointF32[k * POINT_CASTER_FLOATS + 3] = -1;
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

function pointGroup1(world: World): GPUBindGroup {
    const _atlasState = world.resource(atlasStateKey);

    if (
        _atlasState.pointGroup1?.faceVP === _atlasState.faceVP &&
        _atlasState.pointGroup1?.combo === _atlasState.comboMeta
    ) {
        return _atlasState.pointGroup1!.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(pointLayout, {
            faceVP: _atlasState.faceVP!,
            comboMeta: _atlasState.comboMeta!,
            tileRects: _atlasState.pointTileRects!,
        }),
    );
    _atlasState.pointGroup1 = {
        faceVP: _atlasState.faceVP!,
        combo: _atlasState.comboMeta!,
        group,
    };
    return group;
}

function cascadeGroup1(world: World): GPUBindGroup {
    const _atlasState = world.resource(atlasStateKey);

    if (
        _atlasState.cascadeGroup1?.faceVP === _atlasState.cascadeVPBuf &&
        _atlasState.cascadeGroup1?.combo === _atlasState.cascadeMetaBuf
    ) {
        return _atlasState.cascadeGroup1!.group;
    }
    const group = world.gpu.root.unwrap(
        world.gpu.root.createBindGroup(cascadeLayout, {
            faceVP: _atlasState.cascadeVPBuf!,
            comboMeta: _atlasState.cascadeMetaBuf!,
            tileRects: _atlasState.cascadeRectsBuf!,
        }),
    );
    _atlasState.cascadeGroup1 = {
        faceVP: _atlasState.cascadeVPBuf!,
        combo: _atlasState.cascadeMetaBuf!,
        group,
    };
    return group;
}

/** Recreates world-owned shadow resources before compiling pipelines that reference their layouts. */
export function resetShadowAtlas(world: World, device: GPUDevice): void {
    const _atlasState = world.resource(atlasStateKey);

    _atlasState.shadowGroup = null;
    _atlasState.pointGroup1 = null;
    _atlasState.cascadeGroup1 = null;
    _atlasState.sunCasting = false;
    // Reverse-Z receivers are lit when their depth is at least the stored occluder's. Linear
    // comparison filtering makes each textureSampleCompareLevel tap a 2x2 hardware PCF footprint;
    // both the fallback and real maps use this sampler.
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
    _atlasState.sunParams?.destroy();
    _atlasState.sunParams = device.createBuffer({
        label: "standard-shadow-params",
        size: SHADOW_PARAMS_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Params stay bound even before the lazy atlas exists; empty slots cannot match a light.
    _atlasState.pointAtlas?.destroy();
    _atlasState.pointAtlas = null;
    _atlasState.pointAtlasView = null;
    _atlasState.pointFrames = [];
    _atlasState.pointFrameCount = 0;
    _atlasState.pointParams?.destroy();
    // both uniforms are sized from the schemas the shadow WGSL emits, so the binding and the struct the
    // receiver reads can't drift apart
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
            .createBuffer(_shadowCasters, _atlasState.pointParams)
            .$usage("uniform")
            .$name("standard-point-shadow-params"),
    );
    // Both receiver sampling and atlas seam-discard must use the same tile rects.
    _atlasState.pointTileRects?.destroy();
    _atlasState.pointTileRects = device.createBuffer({
        label: "standard-point-tilerects",
        size: d.sizeOf(tileRectsSchema(MAX_POINT_CASTERS * 6)),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(
        _atlasState.pointTileRects,
        0,
        new Float32Array(MAX_POINT_CASTERS * 6 * 4),
    );
    world.gpu.buffers.set("pointTileRects", _atlasState.pointTileRects);
    world.gpu.typed.set(
        "pointTileRects",
        world.gpu.root
            .createBuffer(_shadowRects, _atlasState.pointTileRects)
            .$usage("uniform")
            .$name("standard-point-tilerects"),
    );

    _atlasState.faceVP?.destroy();
    _atlasState.faceVP = device.createBuffer({
        label: "standard-point-facevp",
        size: MAX_POINT_CASTERS * 6 * 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    _atlasState.comboMeta?.destroy();
    _atlasState.comboMeta = device.createBuffer({
        label: "standard-point-combometa",
        size: MAX_POINT_CASTERS * 6 * 16, // vec4<u32> per combo
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
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
    _atlasState.pointGroup1 = null;
    _atlasState.cascadeGroup1 = null;
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

/**
 * Allocates the point atlas at pointAtlasSize on the first casting frame and replaces it on resize.
 * Call only when casting: the no-caster path needs no atlas. ShadowCameraSystem calls before draw
 * recording because the prepass binds the shadow group before the shadow pass; replacing a texture
 * after that binding invalidates submission.
 */
export function ensurePointAtlas(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    const side = pointAtlasSize(world);
    if (_atlasState.pointAtlas?.width === side) return;
    _atlasState.pointAtlas?.destroy();
    _atlasState.pointAtlas = world.gpu.device.createTexture({
        label: "standard-point-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _atlasState.pointAtlasView = _atlasState.pointAtlas.createView();
}

/**
 * Allocates the cascade atlas on the first casting frame and replaces it when tile resolution or
 * cascade-grid size changes. Call only when casting so the disabled path allocates no atlas.
 * ShadowCameraSystem calls before draw recording: the prepass binds the shadow group before the
 * shadow pass, and replacing a texture after that binding invalidates submission.
 */
export function ensureCascadeAtlas(world: World): void {
    const _atlasState = world.resource(atlasStateKey);

    const side = cascadeAtlasSize(sunResolution(world), sunCascades(world));
    if (_atlasState.cascadeAtlas?.width === side) return;
    _atlasState.cascadeAtlas?.destroy();
    _atlasState.cascadeAtlas = world.gpu.device.createTexture({
        label: "standard-cascade-shadow-atlas",
        size: { width: side, height: side },
        format: DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    _atlasState.cascadeAtlasView = _atlasState.cascadeAtlas.createView();
}

/**
 * Records point/spot atlas depth and publishes caster parameters for the color/fog receivers.
 * Consumes the first frameCount resolved draws, shared with the prepass and color pass.
 * Each face/cone culls independently; re-gather concatenates those survivors with a combo index so
 * one indirect draw per casting mesh covers all its combos in one atlas pass. The atlas eids lane
 * must bind that gathered list, not the producer's original list.
 * Without casters, clears parameters and records no pass or atlas allocation.
 */
export function renderPointShadows(
    world: World,
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
    capacity: number,
): void {
    const _atlasState = world.resource(atlasStateKey);
    const _pointRegather = world.resource(pointRegather);

    if (!_atlasState.shadowReady) return;
    if (_atlasState.pointFrameCount === 0) {
        if (!_atlasState.pointCleared) {
            clearPointParams(world);
            world.gpu.device.queue.writeBuffer(_atlasState.pointParams!, 0, _atlasState.pointBuf);
            _atlasState.pointCleared = true;
        }
        return;
    }
    ensurePointAtlas(world);
    _pointRegather.ensure(pointCasters(world) * 6, capacity);

    // the caster params the FS samples (pos + source eid, near + depth/normal bias, + the spot basis —
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
        _atlasState.pointF32[o + 5] = 0;
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
    // meta.z carries the atlas side the point VS scales its tile box by, since PointShadows.atlas sizes it live
    for (let i = 0; i < combos.length; i++) comboMeta[i * 4 + 2] = _atlasState.pointAtlas!.width;
    if (C === combos.length) {
        // All combo views survived; dense indices already match.
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

    // Point re-gather can consume only one indirect source; draws from other sources must be skipped.
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
    const encoder = world.frameEncoder()!;

    for (let i = 0; i < D; i++)
        _atlasState.drawPairs[i] = Math.floor(
            (_atlasState.castDraws[i].draw.args.offset ?? 0) / SHADOW_ARG_STRIDE,
        );
    _atlasState.pointRegatherPass.timestampWrites = world.gpu.span?.("standard:pointregather");
    const cpass = beginComputePass(encoder, _atlasState.pointRegatherPass);
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
    endComputePass(cpass);

    // one pass into the whole atlas — one indirect draw per casting mesh, the VS placing each re-gathered
    // instance into its combo's tile. The point VS projects by faceVP (not view), so the ViewUniforms buffer bound
    // at slot 0 is an unread placeholder
    _atlasState.pointShadowDepth.view = _atlasState.pointAtlasView!;
    _atlasState.pointShadowPass.timestampWrites = world.gpu.span?.("standard:pointshadow");
    const group1 = pointGroup1(world);
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
    const pass = beginRenderPass(encoder, _atlasState.pointShadowPass);
    if (_atlasState.pointBundle.bundle) executeRenderBundles(pass, _atlasState.pointBundle.replay);
    endRenderPass(pass);
    world.gpu.indirect?.("standard:pointshadow", D);
    commitPointCastersOnSubmit(world);
}

/**
 * Records cascade atlas depth, then publishes SunShadow parameters and the atlas for color/fog receivers.
 * Consumes the first frameCount resolved draws, shared with the prepass and color pass. Each cascade
 * culls independently; re-gather joins its survivors into one indirect draw per mesh/source batch.
 * Without a casting sun, publishes the fully-lit fallback and allocates no atlas. Missing combo views
 * or no casting geometry also select the fallback, but atlas allocation may already have occurred.
 */
export function renderCascades(
    world: World,
    frameDraws: { draw: Draw; r: Recorded }[],
    frameCount: number,
    capacity: number,
): void {
    const _atlasState = world.resource(atlasStateKey);
    const _cascadeRegather = world.resource(cascadeRegather);

    if (!_atlasState.shadowReady) return;
    const COriginal = cascadeCount(world);
    if (COriginal === 0) {
        _atlasState.sunCasting = false;
        return;
    }
    ensureCascadeAtlas(world);
    _cascadeRegather.ensure(MAX_CASCADES, capacity);

    // Dense combo indices compact VP/meta, but meta.x must retain the original cascade rect index.
    const combos = cascadeComboEids(world);
    const C = comboViewSlots(world, combos, _atlasState.comboSlots, _atlasState.comboIndices);
    if (C === 0) {
        _atlasState.sunCasting = false;
        return;
    }

    // upload the per-cascade folded tile viewProjs + meta (compacted to the survivors' dense index space);
    // meta.z carries the atlas side the cascade VS scales its tile box by, since the light sizes the atlas
    const vp = cascadeFaceVP(world);
    const meta = cascadeMeta(world);
    for (let i = 0; i < COriginal; i++) meta[i * 4 + 2] = _atlasState.cascadeAtlas!.width;
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

    const encoder = world.frameEncoder()!;
    let maxBatchDraws = 0;
    for (let b = 0; b < batchCount; b++) {
        maxBatchDraws = Math.max(maxBatchDraws, _atlasState.cascadeBatches[b].count);
    }
    _cascadeRegather.reserve(maxBatchDraws);
    const group1 = cascadeGroup1(world);
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
        const cpass = beginComputePass(encoder, _atlasState.cascadeRegatherPass);
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
        endComputePass(cpass);

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
        const pass = beginRenderPass(encoder, _atlasState.cascadeShadowPass);
        if (bundle.bundle) executeRenderBundles(pass, bundle.replay);
        endRenderPass(pass);
        totalDraws += D;
    }
    world.gpu.indirect?.("standard:cascadeshadow", totalDraws);

    // Receiver params must use the same surviving cascade order as the atlas VP/meta arrays.
    // Read original arrays through comboIndices; each cascade's normal-bias scale is its own
    // world texel size (2*cover/resolution), not the atlas's combined pixel size.
    const recv = cascadeRecvVP(world);
    const tileRects = cascadeTileRects(world);
    const fars = cascadeFars(world);
    const covers = cascadeCovers(world);
    const res = sunResolution(world);
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
    _atlasState.paramsF32[SUN_PARAMS.globals.overlap] = sunOverlap(world);
    _atlasState.paramsF32[SUN_PARAMS.globals.depthBias] = bias[0];
    _atlasState.paramsF32[SUN_PARAMS.globals.enabled] = 1;
    _atlasState.paramsF32[SUN_PARAMS.globals.normalBias] = bias[1];
    const towardLight = sunTowardLight(world);
    const direction = SUN_PARAMS.globals.lightDirection;
    _atlasState.paramsF32[direction] = towardLight[0];
    _atlasState.paramsF32[direction + 1] = towardLight[1];
    _atlasState.paramsF32[direction + 2] = towardLight[2];
    world.gpu.device.queue.writeBuffer(
        _atlasState.sunParams!,
        0,
        _atlasState.paramsBuf,
        0,
        SHADOW_PARAMS_BYTES,
    );
    _atlasState.sunCasting = true;
}
