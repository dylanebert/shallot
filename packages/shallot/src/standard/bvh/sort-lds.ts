// Subgroup-free radix sort — the LDS sibling of the Onesweep sort in sort.ts, selected
// when the device has no `subgroups` (WebKit: Safari / WKWebView). Same contract as
// {@link createRadixSort}: sorts (key, payload) u32 pairs ascending, stable, the sorted
// result left in `keys`/`payload`. The BVH builder picks it via the `subgroups` flag
// threaded through {@link createBvh}.
//
// 4-bit LSD radix (eight passes for a full u32 key), reduce-then-scan — a faithful port of
// the proven, wave-free PlayCanvas WebGPU sort (reference/playcanvas-engine radix-sort:
// compute-radix-sort-4bit + compute-prefix-sum + compute-radix-sort-reorder, MIT; the same
// algorithm as FidelityFX ParallelSort). Each pass:
//   1. histogram   — one workgroup per block, per-digit counts → blockSums, digit-major
//      (all of digit d's per-block counts contiguous), so a flat exclusive scan over it is
//      exactly the global digit base each block needs.
//   2. scan        — work-efficient Blelloch exclusive scan over blockSums (2 levels: per-
//      chunk scan, scan the chunk totals, add back), pure LDS, no subgroup ops.
//   3. reorder     — one workgroup per block re-ranks its keys by digit via 16 × 256-bit LDS
//      bitmasks + `countOneBits`, scatters to scanned-base + local rank (stable).
//
// vs Onesweep: more dispatches (5/pass × 8 = 40 vs 7) and no subgroup acceleration — the cost
// of portability, paid only where subgroups are absent. Block size LDS_EPW == sort.ts PART_SIZE,
// so the shared key/payload buffers + the Morton sentinel padding are unchanged; the live range
// is gated by the GPU count (`countBuf[0]`), masking out-of-range lanes rather than relying on
// the padding. Dispatch counts are CPU-known (from maxKeys), so the GPU count only masks — no
// indirect prepare. Scaling the dispatch down to the live count would cut the small-N cost but
// isn't built here (the reference masks at capacity too).

import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { precompile, precompileScope } from "../../engine/runtime";
import type { RadixSort, RadixSortShared } from "./sort";

const THREADS = 256; // workgroup size, every kernel
const EPT = 14; // elements per thread in histogram/reorder
const EPW = THREADS * EPT; // 3584 — keys per block, == sort.ts PART_SIZE
const RADIX = 16; // 4-bit digit
const DIGIT_MASK = RADIX - 1;
const MASK_WORDS = 8; // 256 bits of lane flags per digit
const PASSES = 8; // 8 × 4 bits = full u32
const SCAN_ITEMS = 2 * THREADS; // 512 — elements one scan workgroup folds (2 per thread)
const SCAN_HALF = SCAN_ITEMS >> 1; // the up-sweep's first stride
const MAX_DISPATCH = 65535;

/** the histogram + reorder uniform: `workgroupCount` sizes the digit-major stride, `shift` selects the
 *  4-bit window. One per pass (the shift differs); the count is constant. @internal */
export const PassParams = d.struct({ workgroupCount: d.u32, shift: d.u32 }).$name("PassParams");

/** the scan + add uniform: how many of the digit-major block sums are live. @internal */
export const ScanParams = d.struct({ elementCount: d.u32 }).$name("ScanParams");

/** per-block digit histogram: keys in, digit-major block sums out. @internal */
export const histLayout = tgpu.bindGroupLayout({
    keys: { storage: d.arrayOf(d.u32), access: "readonly" },
    blockSums: { storage: d.arrayOf(d.u32), access: "mutable" },
    params: { uniform: PassParams },
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" },
});

/** the Blelloch scan's I/O: the array it scans in place plus the per-chunk totals it emits. @internal */
export const scanLayout = tgpu.bindGroupLayout({
    items: { storage: d.arrayOf(d.u32), access: "mutable" },
    chunkSums: { storage: d.arrayOf(d.u32), access: "mutable" },
    params: { uniform: ScanParams },
});

/** the add-back pass: the scanned chunk totals folded into each chunk's elements. @internal */
export const addLayout = tgpu.bindGroupLayout({
    items: { storage: d.arrayOf(d.u32), access: "mutable" },
    chunkSums: { storage: d.arrayOf(d.u32), access: "readonly" },
    params: { uniform: ScanParams },
});

/** the ranked scatter's I/O: source keys/values, the scanned global prefix, destination pair.
 *  @internal */
export const reorderLayout = tgpu.bindGroupLayout({
    inKeys: { storage: d.arrayOf(d.u32), access: "readonly" },
    outKeys: { storage: d.arrayOf(d.u32), access: "mutable" },
    prefix: { storage: d.arrayOf(d.u32), access: "readonly" },
    inVals: { storage: d.arrayOf(d.u32), access: "readonly" },
    outVals: { storage: d.arrayOf(d.u32), access: "mutable" },
    params: { uniform: PassParams },
    countBuf: { storage: d.arrayOf(d.u32), access: "readonly" },
});

const hist = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), RADIX));

// digit-major: digit d's per-block counts occupy [d*workgroupCount, (d+1)*workgroupCount), so a flat
// exclusive scan over blockSums is exactly the global digit base each block needs.
const histKernel = tgpu
    .computeFn({
        workgroupSize: [THREADS],
        in: { wid: d.builtin.workgroupId, tid: d.builtin.localInvocationIndex },
    })((input) => {
        "use gpu";
        const lane = input.tid;
        const base = input.wid.x * EPW;
        if (lane < RADIX) std.atomicStore(hist.$[lane], 0);
        std.workgroupBarrier();
        const count = histLayout.$.countBuf[0];
        let r = d.u32(0);
        while (r < EPT) {
            const gid = base + r * THREADS + lane;
            if (gid < count) {
                std.atomicAdd(
                    hist.$[(histLayout.$.keys[gid] >>> histLayout.$.params.shift) & DIGIT_MASK],
                    1,
                );
            }
            r = r + 1;
        }
        std.workgroupBarrier();
        if (lane < RADIX)
            histLayout.$.blockSums[lane * histLayout.$.params.workgroupCount + input.wid.x] =
                std.atomicLoad(hist.$[lane]);
    })
    .$name("radixLdsHist");

const temp = tgpu.workgroupVar(d.arrayOf(d.u32, SCAN_ITEMS));

// Blelloch work-efficient exclusive scan over one chunk of SCAN_ITEMS, writing the chunk total to
// chunkSums[chunk]. Run over every chunk (level 0), then once over the chunk totals (level 1).
const scanKernel = tgpu
    .computeFn({
        workgroupSize: [THREADS],
        in: { wid: d.builtin.workgroupId, tid: d.builtin.localInvocationIndex },
    })((input) => {
        "use gpu";
        const lane = input.tid;
        const n = scanLayout.$.params.elementCount;
        const e0 = lane * 2;
        const g0 = input.wid.x * SCAN_ITEMS + e0;
        temp.$[e0] = std.select(d.u32(0), scanLayout.$.items[g0], g0 < n);
        temp.$[e0 + 1] = std.select(d.u32(0), scanLayout.$.items[g0 + 1], g0 + 1 < n);

        let offset = d.u32(1);
        let up = d.u32(SCAN_HALF);
        while (up > 0) {
            std.workgroupBarrier();
            if (lane < up) {
                const ai = offset * (e0 + 1) - 1;
                const bi = offset * (e0 + 2) - 1;
                temp.$[bi] = temp.$[bi] + temp.$[ai];
            }
            offset = offset * 2;
            up = up >>> 1;
        }
        if (lane === 0) {
            scanLayout.$.chunkSums[input.wid.x] = temp.$[SCAN_ITEMS - 1];
            temp.$[SCAN_ITEMS - 1] = 0;
        }
        let down = d.u32(1);
        while (down < SCAN_ITEMS) {
            offset = offset >>> 1;
            std.workgroupBarrier();
            if (lane < down) {
                const ai = offset * (e0 + 1) - 1;
                const bi = offset * (e0 + 2) - 1;
                const t = temp.$[ai];
                temp.$[ai] = temp.$[bi];
                temp.$[bi] = temp.$[bi] + t;
            }
            down = down * 2;
        }
        std.workgroupBarrier();
        if (g0 < n) scanLayout.$.items[g0] = temp.$[e0];
        if (g0 + 1 < n) scanLayout.$.items[g0 + 1] = temp.$[e0 + 1];
    })
    .$name("radixLdsScan");

// add each chunk's scanned total back into its elements → the global exclusive prefix.
const addKernel = tgpu
    .computeFn({
        workgroupSize: [THREADS],
        in: { wid: d.builtin.workgroupId, tid: d.builtin.localInvocationIndex },
    })((input) => {
        "use gpu";
        const n = addLayout.$.params.elementCount;
        const g0 = input.wid.x * SCAN_ITEMS + input.tid * 2;
        if (g0 >= n) return;
        const add = addLayout.$.chunkSums[input.wid.x];
        addLayout.$.items[g0] = addLayout.$.items[g0] + add;
        if (g0 + 1 >= n) return;
        addLayout.$.items[g0 + 1] = addLayout.$.items[g0 + 1] + add;
    })
    .$name("radixLdsAdd");

const masks = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), RADIX * MASK_WORDS)); // 16 digits × 256 bits
const offsets = tgpu.workgroupVar(d.arrayOf(d.u32, RADIX)); // cumulative per-digit rank across rounds

// rank each key within its block by digit (16 × 256-bit LDS bitmasks + popcount), scatter to the
// scanned global base + local rank. Stable: a lane's rank counts only earlier lanes.
const reorderKernel = tgpu
    .computeFn({
        workgroupSize: [THREADS],
        in: { wid: d.builtin.workgroupId, tid: d.builtin.localInvocationIndex },
    })((input) => {
        "use gpu";
        const lane = input.tid;
        const base = input.wid.x * EPW;
        const word = lane >>> 5;
        const bit = lane & 31;
        if (lane < RADIX) offsets.$[lane] = 0;
        if (lane < RADIX * MASK_WORDS) std.atomicStore(masks.$[lane], 0);
        std.workgroupBarrier();
        const count = reorderLayout.$.countBuf[0];

        let r = d.u32(0);
        while (r < EPT) {
            const gid = base + r * THREADS + lane;
            const valid = gid < count;
            const k = std.select(d.u32(0), reorderLayout.$.inKeys[gid], valid);
            // an invalid lane parks on the out-of-range digit RADIX, so it flags no real digit's mask
            const digit = std.select(
                d.u32(RADIX),
                (k >>> reorderLayout.$.params.shift) & DIGIT_MASK,
                valid,
            );
            const v = std.select(d.u32(0), reorderLayout.$.inVals[gid], valid);
            if (valid) std.atomicOr(masks.$[digit * MASK_WORDS + word], d.u32(1) << bit);
            std.workgroupBarrier();
            if (valid) {
                const mbase = digit * MASK_WORDS;
                let local = offsets.$[digit];
                let w = d.u32(0);
                while (w < word) {
                    local = local + std.countOneBits(std.atomicLoad(masks.$[mbase + w]));
                    w = w + 1;
                }
                local =
                    local +
                    std.countOneBits(
                        std.atomicLoad(masks.$[mbase + word]) & ((d.u32(1) << bit) - 1),
                    );
                const pos =
                    reorderLayout.$.prefix[
                        digit * reorderLayout.$.params.workgroupCount + input.wid.x
                    ] + local;
                reorderLayout.$.outKeys[pos] = k;
                reorderLayout.$.outVals[pos] = v;
            }
            // fold this round's per-digit counts into the cumulative offsets, then clear for the next
            if (r < EPT - 1) {
                std.workgroupBarrier();
                if (lane < RADIX) {
                    let c = d.u32(0);
                    let w = d.u32(0);
                    while (w < MASK_WORDS) {
                        const idx = lane * MASK_WORDS + w;
                        c = c + std.countOneBits(std.atomicLoad(masks.$[idx]));
                        std.atomicStore(masks.$[idx], 0);
                        w = w + 1;
                    }
                    offsets.$[lane] = offsets.$[lane] + c;
                }
                std.workgroupBarrier();
            }
            r = r + 1;
        }
    })
    .$name("radixLdsReorder");

/** the emitted subgroup-free sort WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function radixLdsWgsl(): {
    hist: string;
    scan: string;
    add: string;
    reorder: string;
} {
    const names = { names: "strict" } as const;
    return {
        hist: tgpu.resolve([histKernel], names),
        scan: tgpu.resolve([scanKernel], names),
        add: tgpu.resolve([addKernel], names),
        reorder: tgpu.resolve([reorderKernel], names),
    };
}

/**
 * build a subgroup-free radix sorter for up to `maxKeys` (key, payload) u32 pairs. The LDS
 * sibling {@link createRadixSort} delegates to when `subgroups` is false. Same {@link RadixSort}
 * contract; {@link createBvh} threads its shared buffers + count in.
 */
export async function createRadixSortLds(
    device: GPUDevice,
    maxKeys: number,
    shared: RadixSortShared = {},
): Promise<RadixSort> {
    const root = Compute.root;
    const maxBlocks = Math.max(1, Math.ceil(maxKeys / EPW));
    if (maxBlocks > MAX_DISPATCH) {
        throw new Error(
            `createRadixSortLds: maxKeys ${maxKeys} needs ${maxBlocks} blocks, over the ${MAX_DISPATCH} dispatch limit`,
        );
    }
    const sumsLen = RADIX * maxBlocks; // digit-major blockSums length
    const nChunks = Math.ceil(sumsLen / SCAN_ITEMS);
    if (nChunks > SCAN_ITEMS) {
        // level-1 scan folds the chunk totals in one workgroup; past SCAN_ITEMS chunks it can't
        throw new Error(
            `createRadixSortLds: maxKeys ${maxKeys} needs ${nChunks} scan chunks, over the ${SCAN_ITEMS} single-workgroup limit`,
        );
    }
    const paddedMax = maxBlocks * EPW;

    const owned: GPUBuffer[] = [];
    const own = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    const StoreCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const keys = shared.keys ?? own("radix-lds-keys", paddedMax * 4, StoreCopy);
    const payload = shared.payload ?? own("radix-lds-payload", paddedMax * 4, StoreCopy);
    const altKeys = own("radix-lds-alt-keys", paddedMax * 4, StoreCopy);
    const altPayload = own("radix-lds-alt-payload", paddedMax * 4, StoreCopy);
    const blockSums = own("radix-lds-block-sums", sumsLen * 4, GPUBufferUsage.STORAGE);
    const chunkSums = own("radix-lds-chunk-sums", nChunks * 4, GPUBufferUsage.STORAGE);
    // level-1 writes one total per (1) folding workgroup; a single slot suffices under the cap
    const chunkScratch = own("radix-lds-chunk-scratch", 4, GPUBufferUsage.STORAGE);

    if (!shared.count) {
        throw new Error("createRadixSortLds: needs a count buffer (shared.count)");
    }
    const count = shared.count;

    const u16 = (label: string): GPUBuffer =>
        own(label, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    // one {workgroupCount, shift} per pass (shift = pass*4); reused by histogram + reorder
    const passU = Array.from({ length: PASSES }, (_, p) => {
        const b = u16(`radix-lds-pass-${p}`);
        device.queue.writeBuffer(b, 0, new Uint32Array([maxBlocks, p * 4]));
        return b;
    });
    const sumsLenU = u16("radix-lds-sums-len");
    device.queue.writeBuffer(sumsLenU, 0, new Uint32Array([sumsLen]));
    const chunksLenU = u16("radix-lds-chunks-len");
    device.queue.writeBuffer(chunksLenU, 0, new Uint32Array([nChunks]));

    const histPipe = root.createComputePipeline({ compute: histKernel }).$name("radix-lds-hist");
    const scanPipe = root.createComputePipeline({ compute: scanKernel }).$name("radix-lds-scan");
    const addPipe = root.createComputePipeline({ compute: addKernel }).$name("radix-lds-add");
    const reorderPipe = root
        .createComputePipeline({ compute: reorderKernel })
        .$name("radix-lds-reorder");

    // pass i reads from keys when i is even, alt when odd (8 passes → result back in keys)
    const src = (i: number): [GPUBuffer, GPUBuffer] =>
        i % 2 === 0 ? [keys, payload] : [altKeys, altPayload];
    const dst = (i: number): [GPUBuffer, GPUBuffer] =>
        i % 2 === 0 ? [altKeys, altPayload] : [keys, payload];

    const histBound = Array.from({ length: PASSES }, (_, i) =>
        histPipe.with(
            root.createBindGroup(histLayout, {
                keys: src(i)[0],
                blockSums,
                params: passU[i],
                countBuf: count,
            }),
        ),
    );
    const reorderBound = Array.from({ length: PASSES }, (_, i) =>
        reorderPipe.with(
            root.createBindGroup(reorderLayout, {
                inKeys: src(i)[0],
                outKeys: dst(i)[0],
                prefix: blockSums,
                inVals: src(i)[1],
                outVals: dst(i)[1],
                params: passU[i],
                countBuf: count,
            }),
        ),
    );
    const scanL0 = scanPipe.with(
        root.createBindGroup(scanLayout, { items: blockSums, chunkSums, params: sumsLenU }),
    );
    // level 1 folds the chunk totals in one workgroup, so its own total lands in a scratch slot
    const scanL1 = scanPipe.with(
        root.createBindGroup(scanLayout, {
            items: chunkSums,
            chunkSums: chunkScratch,
            params: chunksLenU,
        }),
    );
    const addBound = addPipe.with(
        root.createBindGroup(addLayout, { items: blockSums, chunkSums, params: sumsLenU }),
    );

    // per-sorter labels — an app can hold several sorters, and the queue rejects a duplicate label
    const scope = precompileScope("radix-lds");
    for (const [label, bound] of [
        ["hist", histBound[0]],
        ["scan", scanL0],
        ["add", addBound],
        ["reorder", reorderBound[0]],
    ] as const) {
        await precompile(`${scope}-${label}`, () => {
            bound.dispatchWorkgroups(0);
            return bound;
        });
    }

    const span = (): GPUComputePassTimestampWrites | undefined => Compute.span?.("bvh:sort");
    const pass = (
        encoder: GPUCommandEncoder,
        bound: TgpuComputePipeline,
        workgroups: number,
    ): void => {
        const p = encoder.beginComputePass({ timestampWrites: span() });
        bound.with(p).dispatchWorkgroups(workgroups);
        p.end();
    };

    return {
        keys,
        payload,
        maxKeys,
        sortIndirect(encoder: GPUCommandEncoder): void {
            for (let i = 0; i < PASSES; i++) {
                pass(encoder, histBound[i], maxBlocks); // per-block digit histograms
                pass(encoder, scanL0, nChunks); // per-chunk exclusive scan
                pass(encoder, scanL1, 1); // scan the chunk totals
                pass(encoder, addBound, nChunks); // add back → global prefix
                pass(encoder, reorderBound[i], maxBlocks); // ranked scatter
            }
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
