// The GPU rasterizer: a TGSL compute pass that writes one dirty tile's albedo + boundary-distance content
// directly from a `StrokeDocument` (`document.ts`) — the real producer `atlas.ts`'s `redraw` calls per
// drained tile, replacing stage 4's CPU `TilePacker` (`stroke.ts`'s `packStrokeTile`, now a CPU reference
// kept for its own tests and the cross-check in `document.test.ts`, no longer the live path).
//
// Why a storage *buffer*, not `texture_storage_2d<r8unorm, write>`: WebGPU's storage-texture write
// capability is restricted to a fixed format list (rgba8unorm, the r32/rg32/rgba32 family, rgba16float,
// …) that does not include r8unorm — `tiles.ts`'s DIST_FORMAT, chosen in stage 4 to spend every
// quantization step where the fwidth threshold samples. `textureStore` into it would fail pipeline
// validation. The fix keeps r8unorm as the atlas's sampled format and moves the packing into the kernel
// itself: one thread per 4-texel group computes all 4 texels' distance, quantizes each to a byte
// (`encodeDistGpu`, `tiles.ts`'s `encodeDist` reimplemented in TGSL — no shared source, but the same
// codec `terrain.ts`'s fs decodes), and packs them little-endian into one `u32` — no two threads ever
// write the same output word, so no atomics. Albedo is rgba8unorm, which *is* storage-writable, but the
// same one-thread-owns-one-word shape (one thread emits 4 texels' worth of `u32`s) keeps both channels on
// one dispatch. `encoder.copyBufferToTexture` then moves both packed buffers into the atlas array layer —
// a GPU-side command, not a CPU readback, so `redraw`'s per-frame throttle stays synchronous exactly as
// stage 4 left it (`queue.ts`'s `drain`/`allocate` are untouched).

import { Compute, type State } from "@dylanebert/shallot";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { flattenSegments, type StrokeDocument } from "./document";
import { ROAD_ALBEDO } from "./stroke";
import { DIST_RANGE, TEXEL_SIZE, TILE_RES, tileOrigin } from "./tiles";

const GROUP_TEXELS = 4; // texels packed per output distance/index word (4 × r8 = one u32, little-endian)
const GROUPS_PER_SIDE = TILE_RES / GROUP_TEXELS; // 128 — TILE_RES is a multiple of 4 by construction

const WG = 8;
const DISPATCH = { x: Math.ceil(GROUPS_PER_SIDE / WG), y: Math.ceil(TILE_RES / WG) };

/** one flattened polyline segment, GPU-side: two endpoints + half-width. */
const GpuSegment = d.struct({ a: d.vec2f, b: d.vec2f, halfWidth: d.f32 });

const RasterParams = d.struct({
    tileOrigin: d.vec2f,
    segmentCount: d.u32,
});

const rasterLayout = tgpu.bindGroupLayout({
    segments: { storage: d.arrayOf(GpuSegment), access: "readonly" },
    params: { uniform: RasterParams },
    albedoOut: { storage: d.arrayOf(d.u32), access: "mutable" },
    distOut: { storage: d.arrayOf(d.u32), access: "mutable" },
});

/**
 * point-segment signed distance (minus half-width), the compute kernel's own derivation: clamp the
 * projection parameter `t` onto [0, 1], measure straight-line distance to the clamped point. CPU-callable
 * so `bun test` exercises this exact math with no device — the GPU half of
 * stage 5's differential oracle against `document.ts`'s independently-derived `segmentDistance`.
 * @example segmentDistanceGpu(5, 0, -10, 0, 10, 0, 2) // -2 (on the centreline, inside a 2 m half-width)
 */
export const segmentDistanceGpu = tgpu.fn(
    [d.f32, d.f32, d.f32, d.f32, d.f32, d.f32, d.f32],
    d.f32,
)((px, pz, ax, az, bx, bz, halfWidth) => {
    "use gpu";
    const abx = bx - ax;
    const abz = bz - az;
    const apx = px - ax;
    const apz = pz - az;
    const dd = abx * abx + abz * abz;
    let t = d.f32(0);
    if (dd > 0) t = std.clamp((apx * abx + apz * abz) / dd, 0, 1);
    const cx = ax + t * abx;
    const cz = az + t * abz;
    return std.distance(d.vec2f(px, pz), d.vec2f(cx, cz)) - halfWidth;
});

/** the document's segment-only distance at (px, pz) — loops `rasterLayout.$.segments` up to
 *  `params.segmentCount`, the GPU-side twin of `document.ts`'s `flattenSegments` + minimum. */
const segmentsDistanceGpu = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((px, pz) => {
    "use gpu";
    let best = d.f32(3.402823e38); // f32 max — no segment yet
    let i = d.u32(0);
    for (; i < rasterLayout.$.params.segmentCount; i = i + d.u32(1)) {
        const seg = rasterLayout.$.segments[i];
        const dist = segmentDistanceGpu(px, pz, seg.a.x, seg.a.y, seg.b.x, seg.b.y, seg.halfWidth);
        if (dist < best) best = dist;
    }
    return best;
});

/** the document's full distance at (px, pz) — the minimum over every segment, the GPU twin of
 *  `document.ts`'s `documentDistance`. */
export const documentDistanceGpu = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((px, pz) => {
    "use gpu";
    return segmentsDistanceGpu(px, pz);
});

/** quantize a signed world-metre distance to its r8unorm byte — TGSL reimplementation of `tiles.ts`'s
 *  `encodeDist` (no shared source: this is what the kernel actually runs; `tiles.ts`'s stays the CPU
 *  codec the fs's decode and the differential oracle's tolerance both key off). CPU-callable, so the
 *  oracle can quantize the GPU derivation's raw distance the same way the real kernel would.
 * @example encodeDistGpu(0) // 128u, the zero crossing */
export const encodeDistGpu = tgpu.fn(
    [d.f32],
    d.u32,
)((metres) => {
    "use gpu";
    const clamped = std.clamp(metres, d.f32(-DIST_RANGE), d.f32(DIST_RANGE));
    const unit = clamped / d.f32(2 * DIST_RANGE) + d.f32(0.5);
    return d.u32(std.round(unit * d.f32(255)));
});

/** pack ROAD_ALBEDO's rgb + full alpha into rgba8unorm's little-endian byte order — every texel gets the
 *  same word (stage 4's `packStrokeTile` wrote albedo everywhere too: cheap, and the fs never reads it
 *  past the coverage the distance channel derives). Computed once at module load, not per-thread. */
function packAlbedoWord(rgb: readonly [number, number, number]): number {
    const r = Math.round(rgb[0] * 255);
    const g = Math.round(rgb[1] * 255);
    const b = Math.round(rgb[2] * 255);
    return (r | (g << 8) | (b << 16) | (255 << 24)) >>> 0;
}

const albedoWord = tgpu.const(d.u32, packAlbedoWord(ROAD_ALBEDO)).$name("roadAlbedoWord");

const rasterKernel = tgpu
    .computeFn({
        workgroupSize: [WG, WG, 1],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const gx = input.gid.x; // texel-group column, 0..GROUPS_PER_SIDE-1
        const gy = input.gid.y; // texel row, 0..TILE_RES-1
        if (gx >= d.u32(GROUPS_PER_SIDE) || gy >= d.u32(TILE_RES)) return;

        const origin = rasterLayout.$.params.tileOrigin;
        const worldZ = origin.y + (d.f32(gy) + d.f32(0.5)) * d.f32(TEXEL_SIZE);

        let word = d.u32(0);
        let k = d.u32(0);
        for (; k < d.u32(GROUP_TEXELS); k = k + d.u32(1)) {
            const texelX = gx * d.u32(GROUP_TEXELS) + k;
            const worldX = origin.x + (d.f32(texelX) + d.f32(0.5)) * d.f32(TEXEL_SIZE);
            const dist = documentDistanceGpu(worldX, worldZ);
            const byte = encodeDistGpu(dist);
            word = word | (byte << (k * d.u32(8)));

            const albedoIdx = gy * d.u32(TILE_RES) + texelX;
            rasterLayout.$.albedoOut[albedoIdx] = albedoWord.$;
        }
        const distIdx = gy * d.u32(GROUPS_PER_SIDE) + gx;
        rasterLayout.$.distOut[distIdx] = word;
    })
    .$name("roads-rasterize");

/** the emitted rasterizer WGSL — the device-free structural seam stage 5's tests resolve. */
export function rasterizeWgsl(): string {
    return tgpu.resolve(
        [segmentDistanceGpu, segmentsDistanceGpu, documentDistanceGpu, encodeDistGpu, rasterKernel],
        { names: "strict" },
    );
}

type RasterRoot = typeof Compute.root;
type RasterPipeline = ReturnType<RasterRoot["createComputePipeline"]>;

let pipeline: RasterPipeline | null = null;
let albedoRaw: GPUBuffer | null = null;
let distRaw: GPUBuffer | null = null;

const ALBEDO_WORDS = TILE_RES * TILE_RES;
const DIST_WORDS = TILE_RES * GROUPS_PER_SIDE;

function teardown(): void {
    albedoRaw?.destroy();
    distRaw?.destroy();
    pipeline = null;
    albedoRaw = null;
    distRaw = null;
}

/** allocate the rasterizer's persistent GPU scratch (compute pipeline + the two packed-output buffers,
 *  sized once for one tile's worth of texels — every tile is the same size, so these are reused across
 *  every `rasterizeTile` call rather than allocated per redraw). Ties cleanup to `state.onDispose`. */
export function warm(state: State): void {
    teardown();
    state.onDispose(teardown);
    const { device, root } = Compute;
    pipeline = root.createComputePipeline({ compute: rasterKernel }).$name("roads-rasterize");
    albedoRaw = device.createBuffer({
        label: "roads-raster-albedo",
        size: ALBEDO_WORDS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    distRaw = device.createBuffer({
        label: "roads-raster-dist",
        size: DIST_WORDS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
}

/**
 * rasterize `doc` into tile (tx, tz)'s content and copy it straight into `albedoTex`/`distTex`'s `layer`
 * — one compute dispatch + two `copyBufferToTexture` calls, all GPU-side (no CPU readback, so `atlas.ts`'s
 * `redraw` stays synchronous). The segment buffer is rebuilt each call, sized to `doc`'s actual
 * content (the same create-per-call shape `terrain/generate.ts`'s permutation buffer uses) — a redrawn
 * tile is throttled to a handful per frame (`tiles.ts`'s THROTTLE), so this isn't a hot per-frame path.
 */
export function rasterizeTile(
    doc: StrokeDocument,
    tx: number,
    tz: number,
    albedoTex: GPUTexture,
    distTex: GPUTexture,
    layer: number,
): void {
    if (!pipeline || !albedoRaw || !distRaw) {
        throw new Error("rasterize: rasterizeTile before warm()");
    }
    const { device, root } = Compute;

    const segments = flattenSegments(doc);
    const segmentsData =
        segments.length > 0 ? segments : [{ ax: 0, az: 0, bx: 0, bz: 0, halfWidth: 0 }];
    const segRaw = device.createBuffer({
        label: "roads-raster-segments",
        size: segmentsData.length * d.sizeOf(GpuSegment),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const segBuf = root
        .createBuffer(d.arrayOf(GpuSegment, segmentsData.length), segRaw)
        .$usage("storage");
    segBuf.write(
        segmentsData.map((s) => ({
            a: d.vec2f(s.ax, s.az),
            b: d.vec2f(s.bx, s.bz),
            halfWidth: s.halfWidth,
        })),
    );

    const [ox, oz] = tileOrigin(tx, tz);
    const paramsRaw = device.createBuffer({
        label: "roads-raster-params",
        size: d.sizeOf(RasterParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = root.createBuffer(RasterParams, paramsRaw).$usage("uniform");
    paramsBuf.write({ tileOrigin: d.vec2f(ox, oz), segmentCount: segments.length });

    const albedoOut = root
        .createBuffer(d.arrayOf(d.u32, ALBEDO_WORDS), albedoRaw)
        .$usage("storage");
    const distOut = root.createBuffer(d.arrayOf(d.u32, DIST_WORDS), distRaw).$usage("storage");

    const bindGroup = root.createBindGroup(rasterLayout, {
        segments: segBuf,
        params: paramsBuf,
        albedoOut,
        distOut,
    });

    try {
        const enc = device.createCommandEncoder({ label: "roads-rasterize" });
        const pass = enc.beginComputePass({ label: "roads-rasterize" });
        pipeline.with(bindGroup).with(pass).dispatchWorkgroups(DISPATCH.x, DISPATCH.y, 1);
        pass.end();
        enc.copyBufferToTexture(
            { buffer: albedoRaw, bytesPerRow: TILE_RES * 4, rowsPerImage: TILE_RES },
            { texture: albedoTex, origin: { x: 0, y: 0, z: layer } },
            { width: TILE_RES, height: TILE_RES, depthOrArrayLayers: 1 },
        );
        enc.copyBufferToTexture(
            { buffer: distRaw, bytesPerRow: TILE_RES, rowsPerImage: TILE_RES },
            { texture: distTex, origin: { x: 0, y: 0, z: layer } },
            { width: TILE_RES, height: TILE_RES, depthOrArrayLayers: 1 },
        );
        device.queue.submit([enc.finish()]);
    } finally {
        segRaw.destroy();
        paramsRaw.destroy();
    }
}
