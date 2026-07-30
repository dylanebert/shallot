// Morton-code assignment — primitive AABBs + scene AABB → one 30-bit spatial key
// per prim, plus an identity payload, a stage of the LBVH builder. The pass between
// scene-bounds (bounds.ts) and
// the sort (sort.ts): the key is what the radix sort orders, the payload rides along as
// the primitive index so the sorted order is a permutation the build reads back.
//
// Per prim: centroid = AABB midpoint, normalize by the scene extent into [0,1] per
// axis, quantize to 10 bits (×1023), interleave x | y<<1 | z<<2 into a 30-bit code.
// Bit-identical to NexusBVH's ComputeMortonCodesKernel + MortonCode<uint32_t>
// (reference/NexusBVH). A degenerate extent yields a defined code rather than a NaN one:
// the divisor is guarded as well as the result, so a coplanar axis (ext = 0) codes as 0
// and a NaN extent no longer reaches the quantizer at all. That matches the oracle's
// `if (ext <= 0) return 0` branch (tests/bvh/oracle.ts mortonCodes), and it is the
// deterministic-consumer exception gpu.md's NaN policy names for the BVH centroid — every
// other path here computes through.
//
// Embarrassingly parallel — no subgroup ops, no shared memory, one grid-stride loop.
// The one float subtlety is the normalize divide. The oracle models it in f32
// (Math.fround) so the 10-bit quantization lands in the same bin as this GPU math,
// and the `accel` gym scenario's build gate's bit-identical check holds exactly on the
// fixture set. It is NOT exact in bulk: WGSL allows f32 divide up to 2.5 ULP and real
// hardware (NVIDIA Ada, observed) isn't correctly-rounded, so a centroid within ~1 ULP
// of a bin boundary lands ±1 bin off (~one prim per ~12k at 1M uniform-random). That
// reorders two near-coincident prims, which the topology-agnostic build doesn't care
// about — so it's not a correctness concern, only why the exact gate stays on the
// fixtures (no boundary centroids) and bulk scale is timed, not bit-compared.
//
// 64-bit codes (21 bits/axis, NexusBVH's huge-scene path) are deferred: they need the
// 8-pass sort variant that sort.ts (4×8-bit, u32-only) does not provide. Adding an
// unsortable key here would be a half-built feature, so the seam waits for that sort.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { precompile } from "../../engine/runtime";
import { rootOf } from "./root";

const WG = 256; // workgroup size; the grid-stride loop folds any count past one dispatch
const MAX_DISPATCH = 65535; // maxComputeWorkgroupsPerDimension floor
const SENTINEL = 0xffffffff; // above any 30-bit code → sorts past the live prims

/** the pass's one bind group: prim AABBs + the scene AABB in, keys + payload out, the GPU count gating
 *  the live range. @internal */
export const mortonLayout = tgpu.bindGroupLayout({
    prims: { storage: d.arrayOf(d.vec4f), access: "readonly" }, // 2 vec4/prim: min.xyz+pad, max.xyz+pad
    bounds: { storage: d.arrayOf(d.vec4f, 2), access: "readonly" }, // scene AABB: min.xyz+pad, max.xyz+pad
    keys: { storage: d.arrayOf(d.u32), access: "mutable" }, // out: 30-bit Morton code / sentinel
    payload: { storage: d.arrayOf(d.u32), access: "mutable" }, // out: primIndex = identity
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" }, // [0] = GPU-driven prim count
});

/**
 * spread the low 10 bits of `x` to every third bit (two zeros between bits). Bit-identical to NexusBVH
 * `InterleaveBits32` and the oracle's `interleaveBits32`; every intermediate stays within 32 bits, and
 * under 2³¹ at that — so the CPU arm's JS bit ops (signed 32-bit) agree with the shader's.
 * @example const spread = interleaveBits32(0x3ff); // 0x9249249
 */
export const interleaveBits32 = tgpu.fn(
    [d.u32],
    d.u32,
)((x) => {
    "use gpu";
    let v = x & 0x3ff;
    v = (v | (v << 16)) & 0x30000ff;
    v = (v | (v << 8)) & 0x300f00f;
    v = (v | (v << 4)) & 0x30c30c3;
    v = (v | (v << 2)) & 0x9249249;
    return v;
});

/**
 * one primitive's 30-bit Morton code: normalize the centroid by the scene extent into [0,1] per axis,
 * quantize to 10 bits (×1023), interleave `x | y<<1 | z<<2`. A zero-extent axis (coplanar scene)
 * normalizes to 0 rather than 0/0 = NaN — the defined-code requirement, matching the oracle's
 * `if (ext <= 0) return 0` branch.
 * @example const key = mortonCode(centroid, sceneMin, sceneExtent);
 */
export const mortonCode = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f],
    d.u32,
)((centroid, bmin, ext) => {
    "use gpu";
    // the divisor is guarded as well as the result: a zero-extent axis would otherwise compute 0/0,
    // which the select discards on the GPU but which the CPU arm cannot even evaluate (typegpu holds
    // WGSL's finite-math assumption and throws on a non-finite intermediate). Guarding the divisor
    // changes no live axis — it substitutes 1 only where the result is thrown away
    const safe = std.select(d.vec3f(1), ext, std.gt(ext, d.vec3f()));
    const t = std.select(std.div(std.sub(centroid, bmin), safe), d.vec3f(), std.le(ext, d.vec3f()));
    const c = std.clamp(t, d.vec3f(), d.vec3f(1));
    const q = d.vec3u(std.floor(std.mul(c, 1023)));
    return interleaveBits32(q.x) | (interleaveBits32(q.y) << 1) | (interleaveBits32(q.z) << 2);
});

// `maxPrims` is baked as a captured constant (it folds to a literal): the pass fills the whole
// [0, maxPrims) key range every run — a real 30-bit code for a live prim (`i < count`), a max-Morton
// sentinel for the padding tail (`i >= count`). The sentinel sorts above every real code, so the radix
// pushes padding to the end with no CPU tail-pad — the count stays GPU-resident.
function mortonKernel(maxPrims: number) {
    const cap = Math.max(1, maxPrims);
    return tgpu
        .computeFn({
            workgroupSize: [WG],
            in: { gid: d.builtin.globalInvocationId, nwg: d.builtin.numWorkgroups },
        })((input) => {
            "use gpu";
            const bmin = mortonLayout.$.bounds[0].xyz;
            const ext = std.sub(mortonLayout.$.bounds[1].xyz, bmin);
            const stride = input.nwg.x * WG;
            const count = mortonLayout.$.countBuf[0];
            let i = input.gid.x;
            while (i < cap) {
                if (i < count) {
                    const mn = mortonLayout.$.prims[i * 2].xyz;
                    const mx = mortonLayout.$.prims[i * 2 + 1].xyz;
                    mortonLayout.$.keys[i] = mortonCode(std.mul(std.add(mn, mx), 0.5), bmin, ext);
                } else {
                    mortonLayout.$.keys[i] = SENTINEL;
                }
                mortonLayout.$.payload[i] = i;
                i = i + stride;
            }
        })
        .$name("morton");
}

/** the emitted Morton WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function mortonWgsl(maxPrims: number): string {
    return tgpu.resolve([mortonKernel(maxPrims)], { names: "strict" });
}

/**
 * a built Morton-code pass sized for `maxPrims`. Write primitive AABBs into {@link
 * Morton.prims} (2 × vec4<f32> per prim) and the scene AABB into {@link Morton.bounds}
 * (2 × vec4<f32>: `min.xyz+pad`, `max.xyz+pad`, the {@link SceneBounds} output),
 * record {@link Morton.compute}, submit, then read {@link Morton.keys} (one 30-bit
 * code per prim) and {@link Morton.payload} (primitive index, identity before sort).
 * The two output buffers are the radix sort's key / payload input.
 */
/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface MortonShared {
    /** input prim AABB buffer — 2 vec4<f32> per prim */
    prims?: GPUBuffer;
    /** input scene AABB (the bounds reduction's output) */
    bounds?: GPUBuffer;
    /** Morton code output (the sort's key input) */
    keys?: GPUBuffer;
    /** prim-index output (the sort's payload input) */
    payload?: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]); live prims are [0, count), the rest sentinel-padded */
    count?: GPUBuffer;
}

/**
 * a built Morton-code pass sized for `maxPrims`. Reads prim AABBs + the scene bounds ({@link
 * createSceneBounds}) and writes each prim's Morton code into {@link Morton.keys} with an identity
 * {@link Morton.payload}, sentinel-padding the [count, maxPrims) tail so the radix sort moves the live
 * range to the front. The sort's key/payload input.
 */
export interface Morton {
    /** input prim AABB buffer — fill [0, count) prims, 2 vec4 each */
    readonly prims: GPUBuffer;
    /** input scene AABB — 2 vec4<f32> (min then max), produced by the bounds reduction */
    readonly bounds: GPUBuffer;
    /** output Morton codes — real code for [0, count), max-Morton sentinel for [count, maxPrims) */
    readonly keys: GPUBuffer;
    /** output primitive indices — identity (payload[i] = i) over the full key range */
    readonly payload: GPUBuffer;
    /** GPU-driven prim count (one u32 at [0]) — write it (≤ `maxPrims`) before recording */
    readonly count: GPUBuffer;
    /** capacity the buffers are sized for */
    readonly maxPrims: number;
    /** record the Morton-code + sentinel-pad pass over [0, maxPrims) onto `encoder`, count read from {@link Morton.count} */
    compute(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

/**
 * build a Morton-code pass for up to `maxPrims` primitive AABBs. Compiles the kernel
 * and allocates the input (prims, bounds) and output (keys, payload) buffers up front;
 * {@link Morton.compute} then records with no further allocation.
 *
 * @example
 * const mc = await createMorton(device, 1 << 20);
 * device.queue.writeBuffer(mc.prims, 0, primAabbs);
 * device.queue.writeBuffer(mc.bounds, 0, sceneAabb); // from createSceneBounds
 * device.queue.writeBuffer(mc.count, 0, new Uint32Array([count]));
 * const enc = device.createCommandEncoder();
 * mc.compute(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createMorton(
    device: GPUDevice,
    maxPrims: number,
    shared: MortonShared = {},
): Promise<Morton> {
    // before any allocation: a wrong-device call would otherwise leak everything allocated below it
    const root = rootOf(device, "createMorton");
    const cap = Math.max(1, maxPrims);
    const owned: GPUBuffer[] = [];
    const own = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    const StoreSrc = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const StoreDst = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const prims = shared.prims ?? own("morton-prims", cap * 32, StoreDst);
    const bounds = shared.bounds ?? own("morton-bounds", 32, StoreDst);
    const keys = shared.keys ?? own("morton-keys", cap * 4, StoreSrc);
    const payload = shared.payload ?? own("morton-payload", cap * 4, StoreSrc);
    const count = shared.count ?? own("morton-count", 4, StoreDst);

    const bound = root
        .createComputePipeline({ compute: mortonKernel(cap) })
        .$name("morton")
        .with(
            root.createBindGroup(mortonLayout, { prims, bounds, keys, payload, countBuf: count }),
        );
    precompile("morton", () => {
        bound.dispatchWorkgroups(0);
        return bound;
    });

    // dispatch is a CPU constant — the kernel fills the whole [0, maxPrims) key range,
    // branching per-thread on the GPU-read count, so the count never crosses to the CPU
    const numWg = Math.min(Math.ceil(cap / WG), MAX_DISPATCH);

    return {
        prims,
        bounds,
        keys,
        payload,
        count,
        maxPrims,
        compute(encoder: GPUCommandEncoder): void {
            const pass = encoder.beginComputePass({
                timestampWrites: Compute.span?.("bvh:morton"),
            });
            bound.with(pass).dispatchWorkgroups(numWg);
            pass.end();
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
