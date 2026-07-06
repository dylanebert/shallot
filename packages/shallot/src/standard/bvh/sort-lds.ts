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

import { Compute } from "../../engine";
import type { RadixSort, RadixSortShared } from "./sort";

const THREADS = 256; // workgroup size, every kernel
const EPT = 14; // elements per thread in histogram/reorder
const EPW = THREADS * EPT; // 3584 — keys per block, == sort.ts PART_SIZE
const RADIX = 16; // 4-bit digit
const PASSES = 8; // 8 × 4 bits = full u32
const SCAN_ITEMS = 2 * THREADS; // 512 — elements one scan workgroup folds (2 per thread)
const MAX_DISPATCH = 65535;

// histogram + reorder uniform: workgroupCount sizes the digit-major stride, shift selects the
// 4-bit window. One per pass (shift differs); workgroupCount is constant.
const HIST_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;
struct U { workgroupCount: u32, shift: u32 };
@group(0) @binding(2) var<uniform> u: U;
@group(0) @binding(3) var<storage, read> countBuf: array<u32>;

var<workgroup> hist: array<atomic<u32>, ${RADIX}>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let block = wid.x;
    let base = block * ${EPW}u;
    if (tid < ${RADIX}u) { atomicStore(&hist[tid], 0u); }
    workgroupBarrier();
    let count = countBuf[0];
    for (var r = 0u; r < ${EPT}u; r++) {
        let gid = base + r * ${THREADS}u + tid;
        if (gid < count) {
            let digit = (keys[gid] >> u.shift) & ${RADIX - 1}u;
            atomicAdd(&hist[digit], 1u);
        }
    }
    workgroupBarrier();
    // digit-major: digit d's per-block counts occupy [d*workgroupCount, (d+1)*workgroupCount)
    if (tid < ${RADIX}u) { blockSums[tid * u.workgroupCount + block] = atomicLoad(&hist[tid]); }
}
`;

// Blelloch work-efficient exclusive scan over one chunk of SCAN_ITEMS, writing the chunk total
// to chunkSums[chunk]. Run over every chunk (level 0), then once over the chunk totals (level 1).
const SCAN_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> chunkSums: array<u32>;
struct U { elementCount: u32 };
@group(0) @binding(2) var<uniform> u: U;

var<workgroup> temp: array<u32, ${SCAN_ITEMS}>;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let chunk = wid.x;
    let e0 = tid * 2u;
    let g0 = chunk * ${SCAN_ITEMS}u + e0;
    temp[e0] = select(0u, items[g0], g0 < u.elementCount);
    temp[e0 + 1u] = select(0u, items[g0 + 1u], (g0 + 1u) < u.elementCount);

    var offset = 1u;
    for (var d = ${SCAN_ITEMS >> 1}u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (e0 + 1u) - 1u;
            let bi = offset * (e0 + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset *= 2u;
    }
    if (tid == 0u) {
        chunkSums[chunk] = temp[${SCAN_ITEMS - 1}u];
        temp[${SCAN_ITEMS - 1}u] = 0u;
    }
    for (var d = 1u; d < ${SCAN_ITEMS}u; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (e0 + 1u) - 1u;
            let bi = offset * (e0 + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();
    if (g0 < u.elementCount) { items[g0] = temp[e0]; }
    if ((g0 + 1u) < u.elementCount) { items[g0 + 1u] = temp[e0 + 1u]; }
}
`;

// add each chunk's scanned total back into its elements → the global exclusive prefix.
const ADD_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read> chunkSums: array<u32>;
struct U { elementCount: u32 };
@group(0) @binding(2) var<uniform> u: U;

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let chunk = wid.x;
    let g0 = chunk * ${SCAN_ITEMS}u + tid * 2u;
    if (g0 >= u.elementCount) { return; }
    let add = chunkSums[chunk];
    items[g0] += add;
    if ((g0 + 1u) >= u.elementCount) { return; }
    items[g0 + 1u] += add;
}
`;

// rank each key within its block by digit (16 × 256-bit LDS bitmasks + popcount), scatter to
// the scanned global base + local rank. Stable: a lane's rank counts only earlier lanes.
const REORDER_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> inKeys: array<u32>;
@group(0) @binding(1) var<storage, read_write> outKeys: array<u32>;
@group(0) @binding(2) var<storage, read> prefix: array<u32>;
@group(0) @binding(3) var<storage, read> inVals: array<u32>;
@group(0) @binding(4) var<storage, read_write> outVals: array<u32>;
struct U { workgroupCount: u32, shift: u32 };
@group(0) @binding(5) var<uniform> u: U;
@group(0) @binding(6) var<storage, read> countBuf: array<u32>;

var<workgroup> masks: array<atomic<u32>, ${RADIX * 8}>;  // 16 digits × 8 words = 256 bits each
var<workgroup> offsets: array<u32, ${RADIX}>;            // cumulative per-digit rank across rounds

@compute @workgroup_size(${THREADS})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let block = wid.x;
    let base = block * ${EPW}u;
    let word = tid >> 5u;
    let bit = tid & 31u;
    if (tid < ${RADIX}u) { offsets[tid] = 0u; }
    if (tid < ${RADIX * 8}u) { atomicStore(&masks[tid], 0u); }
    workgroupBarrier();
    let count = countBuf[0];

    for (var r = 0u; r < ${EPT}u; r++) {
        let gid = base + r * ${THREADS}u + tid;
        let valid = gid < count;
        let k = select(0u, inKeys[gid], valid);
        let digit = select(${RADIX}u, (k >> u.shift) & ${RADIX - 1}u, valid);
        let v = select(0u, inVals[gid], valid);
        if (valid) { atomicOr(&masks[digit * 8u + word], 1u << bit); }
        workgroupBarrier();
        if (valid) {
            let mbase = digit * 8u;
            var local = offsets[digit];
            for (var w = 0u; w < word; w++) { local += countOneBits(atomicLoad(&masks[mbase + w])); }
            local += countOneBits(atomicLoad(&masks[mbase + word]) & ((1u << bit) - 1u));
            let pos = prefix[digit * u.workgroupCount + block] + local;
            outKeys[pos] = k;
            outVals[pos] = v;
        }
        // fold this round's per-digit counts into the cumulative offsets, then clear for the next
        if (r < ${EPT - 1}u) {
            workgroupBarrier();
            if (tid < ${RADIX}u) {
                var c = 0u;
                for (var w = 0u; w < 8u; w++) {
                    let idx = tid * 8u + w;
                    c += countOneBits(atomicLoad(&masks[idx]));
                    atomicStore(&masks[idx], 0u);
                }
                offsets[tid] += c;
            }
            workgroupBarrier();
        }
    }
}
`;

function storageEntry(binding: number, readonly: boolean): GPUBindGroupLayoutEntry {
    return {
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: readonly ? "read-only-storage" : "storage" },
    };
}
function uniformEntry(binding: number): GPUBindGroupLayoutEntry {
    return { binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } };
}

/**
 * build a subgroup-free radix sorter for up to `maxKeys` (key, payload) u32 pairs — the LDS
 * sibling {@link createRadixSort} delegates to when `subgroups` is false. Same {@link RadixSort}
 * contract; {@link createBvh} threads its shared buffers + count in.
 */
export async function createRadixSortLds(
    device: GPUDevice,
    maxKeys: number,
    shared: RadixSortShared = {},
): Promise<RadixSort> {
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

    const histLayout = device.createBindGroupLayout({
        label: "radix-lds-hist",
        entries: [
            storageEntry(0, true),
            storageEntry(1, false),
            uniformEntry(2),
            storageEntry(3, true),
        ],
    });
    const scanLayout = device.createBindGroupLayout({
        label: "radix-lds-scan",
        entries: [storageEntry(0, false), storageEntry(1, false), uniformEntry(2)],
    });
    const addLayout = device.createBindGroupLayout({
        label: "radix-lds-add",
        entries: [storageEntry(0, false), storageEntry(1, true), uniformEntry(2)],
    });
    const reorderLayout = device.createBindGroupLayout({
        label: "radix-lds-reorder",
        entries: [
            storageEntry(0, true),
            storageEntry(1, false),
            storageEntry(2, true),
            storageEntry(3, true),
            storageEntry(4, false),
            uniformEntry(5),
            storageEntry(6, true),
        ],
    });

    const pipe = (
        label: string,
        code: string,
        layout: GPUBindGroupLayout,
    ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });

    const [histPipe, scanPipe, addPipe, reorderPipe] = await Promise.all([
        pipe("radix-lds-hist", HIST_WGSL, histLayout),
        pipe("radix-lds-scan", SCAN_WGSL, scanLayout),
        pipe("radix-lds-add", ADD_WGSL, addLayout),
        pipe("radix-lds-reorder", REORDER_WGSL, reorderLayout),
    ]);

    const bg = (layout: GPUBindGroupLayout, buffers: GPUBuffer[]): GPUBindGroup =>
        device.createBindGroup({
            layout,
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });

    // pass i reads from keys when i is even, alt when odd (8 passes → result back in keys)
    const src = (i: number): [GPUBuffer, GPUBuffer] =>
        i % 2 === 0 ? [keys, payload] : [altKeys, altPayload];
    const dst = (i: number): [GPUBuffer, GPUBuffer] =>
        i % 2 === 0 ? [altKeys, altPayload] : [keys, payload];

    const histBg = Array.from({ length: PASSES }, (_, i) =>
        bg(histLayout, [src(i)[0], blockSums, passU[i], count]),
    );
    const reorderBg = Array.from({ length: PASSES }, (_, i) =>
        bg(reorderLayout, [src(i)[0], dst(i)[0], blockSums, src(i)[1], dst(i)[1], passU[i], count]),
    );
    const scanL0Bg = bg(scanLayout, [blockSums, chunkSums, sumsLenU]);
    const scanL1Bg = bg(scanLayout, [chunkSums, chunkScratch, chunksLenU]);
    const addBg = bg(addLayout, [blockSums, chunkSums, sumsLenU]);

    const span = (): GPUComputePassTimestampWrites | undefined => Compute.span?.("bvh:sort");
    const pass = (
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        group: GPUBindGroup,
        workgroups: number,
    ): void => {
        const p = encoder.beginComputePass({ timestampWrites: span() });
        p.setPipeline(pipeline);
        p.setBindGroup(0, group);
        p.dispatchWorkgroups(workgroups);
        p.end();
    };

    return {
        keys,
        payload,
        maxKeys,
        sortIndirect(encoder: GPUCommandEncoder): void {
            for (let i = 0; i < PASSES; i++) {
                pass(encoder, histPipe, histBg[i], maxBlocks); // per-block digit histograms
                pass(encoder, scanPipe, scanL0Bg, nChunks); // per-chunk exclusive scan
                pass(encoder, scanPipe, scanL1Bg, 1); // scan the chunk totals
                pass(encoder, addPipe, addBg, nChunks); // add back → global prefix
                pass(encoder, reorderPipe, reorderBg[i], maxBlocks); // ranked scatter
            }
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
