// Radix sort — the Morton-ordering stage of the LBVH builder (standard/bvh). Sorts
// (key, payload) u32 pairs ascending; the builder feeds 30-bit Morton codes + prim indices.
//
// Onesweep / Decoupled-Fallback LSD radix (Merrill & Adinets 2022; Smith/Levien/Owens SPAA '25).
// One global histogram + one seed scan are computed ONCE for all four 8-bit passes — per-byte
// digit counts are order-invariant under the stable reorder — then one binning pass per digit
// recovers its per-partition prefix by a chained scan: 7 dispatches, vs the 16 a per-pass
// reduce-then-scan (DeviceRadixSort) needs, for all N with no count branch (gpu.md "Dispatch
// count is a first-class cost" — the dispatch floor is the dominant per-frame cost for the small,
// GPU-count producers the builder serves: caster TLAS, physics broadphase, terrain chunks).
//
// Decoupled-Fallback, not plain decoupled-lookback: lookback spins on a forward-progress
// guarantee WebGPU withholds for Metal/ARM at the spec level (gpu.md "Cross-workgroup ordering").
// The fallback is the work-stealing fix — a partition that would stall on an unpublished
// predecessor recomputes that predecessor's histogram itself from the (stable, prior-dispatch)
// input keys. The cross-workgroup channel is one atomic word packing value<<2 | flag, so the
// only coherence it needs is per-location — which WGSL relaxed atomics give, no acquire/release.
//
// Ported from b0nes164's references (MIT), HLSL/WGSL → WGSL: the radix ranking + global-hist +
// seed scan from reference/GPUSorting (OneSweep.hlsl + SweepCommon.hlsl + SortCommon.hlsl), and
// the WGSL lookback structure from reference/GPUPrefixSums csdldf.wgsl (the DF paper author's
// Dawn-validated WebGPU scan). WGE16 path only (subgroup >= 16, the platform floor); the < 16
// path is dropped (software is out of scope).
//
// WGSL idiom (load-bearing): the lookback loop early-exits the instant every subgroup resolves,
// gated by `workgroupUniformLoad(&wgDone)` — a control barrier whose result the uniformity
// analysis treats as uniform, so the in-loop barriers are legal. A plain atomicLoad gate is
// rejected by Tint; a fixed-count loop runs every block to full length (a large-N cliff). Thread
// 0 sets the non-atomic wgDone once the completed-subgroup count reaches numSub. See gpu.md
// "decoupled-scan exception".

import { Compute } from "../../engine";
import { createRadixSortLds } from "./sort-lds";

const RADIX = 256;
const RADIX_MASK = RADIX - 1;
const RADIX_PASSES = 4; // four 8-bit passes (full 32-bit key; a 30-bit Morton uses the low three)
const WG = 256; // binning / scan / init workgroup; == RADIX so digit-indexed work needs no guard
const KEYS_PER_THREAD = 14;
const PART_SIZE = WG * KEYS_PER_THREAD; // 3584; keys per binning partition
/** keys per block; sort capacity rounds up to a multiple of this */
export const KEYS_PER_BLOCK = PART_SIZE;
const G_D = PART_SIZE + RADIX; // binning shared: sorted partition + per-digit device base (3840)
const G_HIST_DIM = 128; // global-histogram workgroup
const G_HIST_PART_SIZE = 32768; // keys per global-histogram tile
const MAX_DISPATCH = 65535;

// Tile-descriptor flags, packed value<<2 | flag into one atomic word so a reader sees both
// consistently from per-location coherence alone (no cross-location acquire/release).
const FLAG_NOT_READY = 0;
const FLAG_REDUCTION = 1; // partition's own digit count is published
const FLAG_INCLUSIVE = 2; // partition's inclusive device prefix is published
const FLAG_MASK = 3;
// spins reading a predecessor's descriptor before the cooperative fallback recomputes it —
// at high concurrency the reduction is usually only a few cycles late, so a brief spin avoids
// the rescan (reference/GPUPrefixSums csdldf.wgsl uses 4).
const MAX_SPIN = 4;

/** buffers {@link createBvh} threads through so the stages share one set; an omitted field is allocated internally */
export interface RadixSortShared {
    /** key buffer — sorted in place; must allow `COPY_DST` (the sort pads the tail) */
    keys?: GPUBuffer;
    /** payload buffer — moves with its key */
    payload?: GPUBuffer;
    /** GPU prim-count buffer (one u32) — thread it in to enable {@link RadixSort.sortIndirect} */
    count?: GPUBuffer;
}

/**
 * a built stable LSB radix sort over u32 keys sized for `maxKeys`, each key carrying a u32 payload.
 * {@link RadixSort.sortIndirect} sorts [0, count) ascending off a GPU-driven count buffer, keeping equal
 * keys in input order. The Morton pass feeds it; {@link createBvh} consumes the sorted order.
 */
export interface RadixSort {
    /** key buffer — fill [0, count), holds the ascending-sorted keys after the sort */
    readonly keys: GPUBuffer;
    /** payload buffer — moves with its key; equal keys keep input order (stable) */
    readonly payload: GPUBuffer;
    /** capacity the buffers are sized for */
    readonly maxKeys: number;
    /**
     * the GPU-count path: dispatched indirect off the count buffer threaded in at construction,
     * so the work scales with the actual count and never crosses to the CPU. Requires
     * `shared.count`; the keys must be sentinel-padded past the count (the Morton pass does this).
     * Used by {@link createBvh}
     */
    sortIndirect(encoder: GPUCommandEncoder): void;
    destroy(): void;
}

const ENABLE_SUBGROUPS = "enable subgroups;\n";

// Per-pass params. Storage (not uniform) so the indirect prepare can write them from the GPU
// count. numKeys = padded (block-multiple)
// count; the tail past the real count is 0xffffffff, sorted to the end, never read back.
const PARAMS_WGSL = /* wgsl */ `
struct Params { numKeys: u32, binBlocks: u32, histBlocks: u32, shift: u32 };
@group(1) @binding(0) var<storage, read> P: Params;
`;

// Workgroup-wide exclusive prefix sum (WG = 256): subgroup scan + a scan of the subgroup
// totals. Used by the seed scan.
const SCAN_HELPER_WGSL = /* wgsl */ `
var<workgroup> wg_sgsums: array<u32, 64>;
fn wgExclusiveScan(tid: u32, sid: u32, sgsize: u32, val: u32) -> u32 {
    let sgid = tid / sgsize;
    let inc = subgroupInclusiveAdd(val);
    let excl = inc - val;
    if (sid == sgsize - 1u) { wg_sgsums[sgid] = inc; }
    workgroupBarrier();
    let numSub = ${WG}u / sgsize;
    let tv = select(0u, wg_sgsums[sid], sid < numSub);
    let prefix = subgroupExclusiveAdd(tv);
    let base = subgroupShuffle(prefix, sgid);
    workgroupBarrier();
    return base + excl;
}
`;

// INIT — zero the cross-workgroup state: passHist [0, binBlocks*RADIX*PASSES) (one descriptor
// per (pass, partition, digit)), globalHist [0, RADIX*PASSES), index [0, PASSES) (the per-pass
// partition-assignment counters). Dispatched binBlocks*PASSES workgroups, so it scales with N.
const INIT_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> passHist: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHist: array<u32>;
@group(0) @binding(2) var<storage, read_write> index: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    passHist[wid.x * ${WG}u + tid] = 0u;       // WG == RADIX, so this covers one (pass,partition) row
    if (wid.x == 0u) {
        for (var i = tid; i < ${RADIX * RADIX_PASSES}u; i += ${WG}u) { globalHist[i] = 0u; }
        if (tid < ${RADIX_PASSES}u) { index[tid] = 0u; }
    }
}
`;

// GLOBAL HISTOGRAM — all four per-byte digit counts in one pass over the original keys. Valid
// for every pass up front because a byte's digit-count distribution is invariant under the
// stable reorder the prior passes apply (only the per-partition split changes, recovered in
// the binning pass). Two sub-histograms (gtid/64) halve the shared-atomic contention, then
// reduce + atomicAdd to the device histogram.
const GLOBAL_HIST_WGSL =
    PARAMS_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> globalHist: array<atomic<u32>>;
const SUB_STRIDE = ${RADIX_PASSES * RADIX}u;
var<workgroup> g_hist: array<atomic<u32>, ${2 * RADIX_PASSES * RADIX}>;

@compute @workgroup_size(${G_HIST_DIM})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    for (var i = tid; i < ${2 * RADIX_PASSES * RADIX}u; i += ${G_HIST_DIM}u) { atomicStore(&g_hist[i], 0u); }
    workgroupBarrier();

    let subOff = (tid / 64u) * SUB_STRIDE;
    let tileEnd = select((wid.x + 1u) * ${G_HIST_PART_SIZE}u, P.numKeys, wid.x == P.histBlocks - 1u);
    for (var i = tid + wid.x * ${G_HIST_PART_SIZE}u; i < tileEnd; i += ${G_HIST_DIM}u) {
        let k = keys[i];
        atomicAdd(&g_hist[subOff + 0u * ${RADIX}u + ((k >> 0u) & ${RADIX_MASK}u)], 1u);
        atomicAdd(&g_hist[subOff + 1u * ${RADIX}u + ((k >> 8u) & ${RADIX_MASK}u)], 1u);
        atomicAdd(&g_hist[subOff + 2u * ${RADIX}u + ((k >> 16u) & ${RADIX_MASK}u)], 1u);
        atomicAdd(&g_hist[subOff + 3u * ${RADIX}u + ((k >> 24u) & ${RADIX_MASK}u)], 1u);
    }
    workgroupBarrier();

    for (var d = tid; d < ${RADIX}u; d += ${G_HIST_DIM}u) {
        for (var p = 0u; p < ${RADIX_PASSES}u; p++) {
            let c = atomicLoad(&g_hist[p * ${RADIX}u + d]) + atomicLoad(&g_hist[SUB_STRIDE + p * ${RADIX}u + d]);
            atomicAdd(&globalHist[p * ${RADIX}u + d], c);
        }
    }
}
`;

// SCAN — exclusive-scan each pass's 256-bin global histogram into the global digit base, and
// seed it as partition 0's descriptor (flag INCLUSIVE). That seed terminates every chained
// lookback: partition 0 has no predecessors, so its inclusive prefix IS the global base.
const SCAN_WGSL =
    ENABLE_SUBGROUPS +
    SCAN_HELPER_WGSL +
    PARAMS_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> globalHist: array<u32>;
@group(0) @binding(1) var<storage, read_write> passHist: array<u32>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) sgsize: u32) {
    let tid = lid.x;
    let passIdx = wid.x;
    let base = wgExclusiveScan(tid, sid, sgsize, globalHist[passIdx * ${RADIX}u + tid]);
    passHist[(passIdx * P.binBlocks) * ${RADIX}u + tid] = (base << 2u) | ${FLAG_INCLUSIVE}u;
}
`;

// BINNING PASS (one per digit) — the Onesweep downsweep's per-block subgroup multisplit ranking,
// then the chained lookback-with-fallback to recover the global digit offset (where a per-pass
// reduce-then-scan would precompute digitBase + blockHist instead). The CAS broadcast of this
// partition's digit count to its successor's descriptor is spliced where the across-wave
// reduction is live, so the successor's lookback can resolve against it.
const BINNING_WGSL =
    ENABLE_SUBGROUPS +
    "diagnostic(off, subgroup_uniformity);\n" + // lookback subgroupAny/All run in a workgroupUniformLoad-gated loop; the gate is uniform at runtime
    PARAMS_WGSL +
    /* wgsl */ `
@group(0) @binding(0) var<storage, read> srcKeys: array<u32>;
@group(0) @binding(1) var<storage, read> srcPayload: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstKeys: array<u32>;
@group(0) @binding(3) var<storage, read_write> dstPayload: array<u32>;
@group(0) @binding(4) var<storage, read_write> passHist: array<atomic<u32>>; // cross-workgroup tile descriptors
@group(0) @binding(5) var<storage, read_write> index: array<atomic<u32>>;    // per-pass partition counter

// Sorted partition slot [0, PART_SIZE) + per-digit device base [PART_SIZE, PART_SIZE+RADIX).
// During the lookback (before the key scatter) g_d[0] is the completed-subgroup counter, g_d[1]
// the fallback trigger, g_d[RADIX + d] the cooperative fallback histogram.
var<workgroup> g_d: array<atomic<u32>, ${G_D}>;
var<workgroup> wgPart: u32;
var<workgroup> wgDone: u32; // lookback early-exit gate, read via workgroupUniformLoad

@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(subgroup_invocation_id) sid: u32, @builtin(subgroup_size) sgsize: u32) {
    let tid = lid.x;
    let passIdx = P.shift >> 3u;
    let waveIndex = tid / sgsize;
    let waveHists = (${WG}u / sgsize) * ${RADIX}u;

    // 1. clear per-wave histograms, then claim a partition by atomic counter (the assignment
    //    order tracks execution order — what makes the decoupled lookback meaningful)
    for (var i = tid; i < waveHists; i += ${WG}u) { atomicStore(&g_d[i], 0u); }
    if (tid == 0u) { wgPart = atomicAdd(&index[passIdx], 1u); }
    workgroupBarrier();
    let part = wgPart;

    // 2. load this partition's keys (full; the tail is 0xffffffff-padded in the source buffer)
    let devBase = sid + waveIndex * (${KEYS_PER_THREAD}u * sgsize) + part * ${PART_SIZE}u;
    var k: array<u32, ${KEYS_PER_THREAD}>;
    for (var i = 0u; i < ${KEYS_PER_THREAD}u; i++) { k[i] = srcKeys[devBase + i * sgsize]; }

    // 3. rank within the wave by subgroup multisplit; accumulate the per-wave histogram
    var offsets: array<u32, ${KEYS_PER_THREAD}>;
    let waveParts = (sgsize + 31u) / 32u;
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) {
        let key = k[slot];
        var flags = array<u32, 4>(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
        for (var b = 0u; b < 8u; b++) {
            let bit = (key >> (b + P.shift)) & 1u;
            let ballot = subgroupBallot(bit == 1u);
            let bArr = array<u32, 4>(ballot.x, ballot.y, ballot.z, ballot.w);
            let m = select(0xffffffffu, 0u, bit == 1u);
            for (var wp = 0u; wp < waveParts; wp++) { flags[wp] = flags[wp] & (m ^ bArr[wp]); }
        }
        let digit = (key >> P.shift) & ${RADIX_MASK}u;
        let idx = digit + waveIndex * ${RADIX}u;
        var lowest = 0u;
        for (var wp = 0u; wp < waveParts; wp++) {
            let fbl = firstTrailingBit(flags[wp]);
            if (fbl == 0xffffffffu) { lowest += 32u; } else { lowest += fbl; break; }
        }
        var peerBits = 0u;
        var totalBits = 0u;
        for (var wp = 0u; wp < waveParts; wp++) {
            if (sid >= wp * 32u) {
                let ltMask = select((1u << (sid & 31u)) - 1u, 0xffffffffu, sid >= (wp + 1u) * 32u);
                peerBits += countOneBits(flags[wp] & ltMask);
            }
            totalBits += countOneBits(flags[wp]);
        }
        var pre = 0u;
        if (peerBits == 0u) { pre = atomicAdd(&g_d[idx], totalBits); }
        offsets[slot] = subgroupShuffle(pre, lowest) + peerBits;
    }
    workgroupBarrier();

    // 4a. per digit: inclusive scan across waves with a circular shift, leaving each wave's
    //     exclusive-across-waves prefix in place. histReduction = this partition's digit count.
    var histReduction = atomicLoad(&g_d[tid]);
    for (var i = tid + ${RADIX}u; i < waveHists; i += ${RADIX}u) {
        let c = atomicLoad(&g_d[i]);
        histReduction += c;
        atomicStore(&g_d[i], histReduction - c);
    }
    // CAS-broadcast this partition's digit count to its successor's descriptor (write-if-unset:
    // a fallback may have already published it). The one edit vs the shared DOWNSWEEP ranking.
    if (part < P.binBlocks - 1u) {
        let succ = ((passIdx * P.binBlocks) + part + 1u) * ${RADIX}u + tid;
        atomicCompareExchangeWeak(&passHist[succ], 0u, ${FLAG_REDUCTION}u | (histReduction << 2u));
    }
    // 4b. begin the cross-digit scan: inclusive within the subgroup
    histReduction += subgroupExclusiveAdd(histReduction);
    workgroupBarrier();
    // 4c. finish the cross-digit exclusive scan into g_d[digit] (block-local digit base)
    let laneMask = sgsize - 1u;
    atomicStore(&g_d[((sid + 1u) & laneMask) + (tid & ~laneMask)], histReduction);
    workgroupBarrier();
    let numSub = ${RADIX}u / sgsize;
    {
        let i = select(0u, tid * sgsize, tid < numSub);
        let s = subgroupExclusiveAdd(atomicLoad(&g_d[i]));
        if (tid < numSub) { atomicStore(&g_d[tid * sgsize], s); }
    }
    workgroupBarrier();
    {
        let prevIdx = select(tid - 1u, 0u, sid == 0u);
        let cbase = subgroupBroadcast(atomicLoad(&g_d[prevIdx]), 1u);
        if (sid != 0u) { atomicStore(&g_d[tid], atomicLoad(&g_d[tid]) + cbase); }
    }
    workgroupBarrier();

    // 5. block-local position = wave-local rank + cross-wave base + block-local digit base
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) {
        let digit = (k[slot] >> P.shift) & ${RADIX_MASK}u;
        if (tid >= sgsize) {
            offsets[slot] += atomicLoad(&g_d[digit + waveIndex * ${RADIX}u]) + atomicLoad(&g_d[digit]);
        } else {
            offsets[slot] += atomicLoad(&g_d[digit]);
        }
    }

    // 6. capture the block-local digit base, then repurpose g_d[0] as the completed-subgroup
    //    counter + g_d[1] as the fallback trigger; wgDone is the early-exit gate
    let exclusiveHistReduction = atomicLoad(&g_d[tid]);
    workgroupBarrier();
    if (tid == 0u) { atomicStore(&g_d[0], 0u); atomicStore(&g_d[1], 0u); wgDone = 0u; }
    workgroupBarrier();

    // 7. LOOKBACK WITH FALLBACK — a faithful port of OneSweep's LookbackWithFallback
    //    (reference/GPUPrefixSums csdldf.wgsl + GPUSorting SweepCommon.hlsl). Each digit-lane
    //    walks its descriptor chain back through partitions part, part-1, … accumulating each
    //    predecessor's digit count until an INCLUSIVE one (partition 0 is INCLUSIVE from the seed
    //    scan). A lane spins MAX_SPIN times for a not-yet-published reduction; if any lane still
    //    stalls, the whole workgroup cooperatively recomputes that predecessor's histogram from
    //    the (prior-dispatch, stable) input keys and publishes it (the 3584-key rescan amortizes
    //    across 256 lanes). The loop EARLY-EXITS the instant every subgroup resolves — gated by
    //    workgroupUniformLoad(&wgDone), the WGSL idiom that makes the exit analysis-uniform so the
    //    in-loop barriers are legal. (Don't replace it with a fixed binBlocks-count loop: a plain
    //    atomicLoad gate is rejected by Tint, and a fixed loop runs every block to full length — a
    //    large-N cliff.)
    var lookbackReduction = 0u;
    var lookbackComplete = false;
    var warpComplete = false;
    var lookbackPart = part;
    var spinCount = 0u;
    loop {
        if (workgroupUniformLoad(&wgDone) != 0u) { break; }
        let descIdx = ((passIdx * P.binBlocks) + lookbackPart) * ${RADIX}u + tid;
        var flagPayload = 0u;
        if (!warpComplete && !lookbackComplete) {
            loop {
                flagPayload = atomicLoad(&passHist[descIdx]);
                if ((flagPayload & ${FLAG_MASK}u) != ${FLAG_NOT_READY}u) { break; }
                spinCount += 1u;
                if (spinCount >= ${MAX_SPIN}u) { break; }
            }
        }
        let notReady =
            !warpComplete && !lookbackComplete && (flagPayload & ${FLAG_MASK}u) == ${FLAG_NOT_READY}u;
        if (!warpComplete) {
            if (subgroupAny(notReady) && sid == 0u) { atomicOr(&g_d[1], 1u); }
        }
        workgroupBarrier();

        let doFallback = atomicLoad(&g_d[1]) != 0u;
        atomicStore(&g_d[${RADIX}u + tid], 0u);
        workgroupBarrier();
        if (tid == 0u) { atomicStore(&g_d[1], 0u); }
        if (doFallback) {
            let fbBase = (lookbackPart - 1u) * ${PART_SIZE}u;
            for (var i = tid; i < ${PART_SIZE}u; i += ${WG}u) {
                atomicAdd(&g_d[${RADIX}u + ((srcKeys[fbBase + i] >> P.shift) & ${RADIX_MASK}u)], 1u);
            }
        }
        workgroupBarrier();

        if (!warpComplete && !lookbackComplete) {
            if (doFallback) {
                let recomputed = atomicLoad(&g_d[${RADIX}u + tid]);
                let res = atomicCompareExchangeWeak(&passHist[descIdx], 0u,
                    ${FLAG_REDUCTION}u | (recomputed << 2u));
                let old = res.old_value;
                if ((old & ${FLAG_MASK}u) == ${FLAG_INCLUSIVE}u) {
                    lookbackReduction += old >> 2u;
                    lookbackComplete = true;
                } else {
                    lookbackReduction += recomputed;
                }
            } else {
                lookbackReduction += flagPayload >> 2u;
                if ((flagPayload & ${FLAG_MASK}u) == ${FLAG_INCLUSIVE}u) { lookbackComplete = true; }
            }
            spinCount = 0u;
            // on completion, publish this partition's inclusive prefix to its successor so it
            // need not look back past us (bumps the successor's REDUCTION flag to INCLUSIVE)
            if (lookbackComplete && part < P.binBlocks - 1u) {
                let succ = ((passIdx * P.binBlocks) + part + 1u) * ${RADIX}u + tid;
                atomicAdd(&passHist[succ], 1u | (lookbackReduction << 2u));
            }
        }
        lookbackPart -= 1u;
        if (!warpComplete) {
            warpComplete = subgroupAll(lookbackComplete);
            if (warpComplete && sid == 0u) { atomicAdd(&g_d[0], 1u); }
        }
        workgroupBarrier();
        if (tid == 0u && atomicLoad(&g_d[0]) >= numSub) { wgDone = 1u; }
    }
    // device base for digit tid = inclusive device prefix - block-local digit base
    atomicStore(&g_d[tid + ${PART_SIZE}u], lookbackReduction - exclusiveHistReduction);
    workgroupBarrier();

    // 8. scatter keys into shared at their block-local sorted position
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) { atomicStore(&g_d[offsets[slot]], k[slot]); }
    workgroupBarrier();

    // 9. scatter sorted keys to the device; capture each slot's digit for the payload pass
    var digits: array<u32, ${KEYS_PER_THREAD}>;
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) {
        let t = tid + slot * ${WG}u;
        let key = atomicLoad(&g_d[t]);
        let d = (key >> P.shift) & ${RADIX_MASK}u;
        digits[slot] = d;
        dstKeys[atomicLoad(&g_d[d + ${PART_SIZE}u]) + t] = key;
    }
    workgroupBarrier();

    // 10. payloads: load in source order, scatter to shared at the key offsets, then to device
    var pl: array<u32, ${KEYS_PER_THREAD}>;
    for (var i = 0u; i < ${KEYS_PER_THREAD}u; i++) { pl[i] = srcPayload[devBase + i * sgsize]; }
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) { atomicStore(&g_d[offsets[slot]], pl[slot]); }
    workgroupBarrier();
    for (var slot = 0u; slot < ${KEYS_PER_THREAD}u; slot++) {
        let t = tid + slot * ${WG}u;
        dstPayload[atomicLoad(&g_d[digits[slot] + ${PART_SIZE}u]) + t] = atomicLoad(&g_d[t]);
    }
}
`;

// GPU-count prepare (indirect path): derive binBlocks/histBlocks from the count, write the per-
// pass params + the init/hist/binning indirect dispatch args. Scan stays direct (RADIX_PASSES).
const PREPARE_WGSL = /* wgsl */ `
struct Params { numKeys: u32, binBlocks: u32, histBlocks: u32, shift: u32 };
@group(0) @binding(0) var<storage, read> count: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>; // [init xyz][hist xyz][bin xyz]
@group(0) @binding(2) var<storage, read_write> p0: Params;
@group(0) @binding(3) var<storage, read_write> p1: Params;
@group(0) @binding(4) var<storage, read_write> p2: Params;
@group(0) @binding(5) var<storage, read_write> p3: Params;
@compute @workgroup_size(1) fn main() {
    let binBlocks = max(1u, (count[0] + ${PART_SIZE - 1}u) / ${PART_SIZE}u);
    let padded = binBlocks * ${PART_SIZE}u;
    let histBlocks = max(1u, (padded + ${G_HIST_PART_SIZE - 1}u) / ${G_HIST_PART_SIZE}u);
    args[0] = binBlocks * ${RADIX_PASSES}u; args[1] = 1u; args[2] = 1u;
    args[3] = histBlocks; args[4] = 1u; args[5] = 1u;
    args[6] = binBlocks; args[7] = 1u; args[8] = 1u;
    p0 = Params(padded, binBlocks, histBlocks, 0u);
    p1 = Params(padded, binBlocks, histBlocks, 8u);
    p2 = Params(padded, binBlocks, histBlocks, 16u);
    p3 = Params(padded, binBlocks, histBlocks, 24u);
}
`;

function storageEntry(binding: number, readonly: boolean): GPUBindGroupLayoutEntry {
    return {
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: readonly ? "read-only-storage" : "storage" },
    };
}

/**
 * build a radix sorter for up to `maxKeys` (key, payload) u32 pairs. Compiles the five kernels
 * (init, global-hist, scan, binning, + an indirect prepare when a count buffer is shared) and
 * allocates the ping-pong + descriptor buffers up front; {@link RadixSort.sortIndirect} then
 * records a sort with no further allocation. {@link createBvh} threads its shared buffers in.
 *
 * `subgroups` picks the implementation: the Onesweep sort here (default, when the device has the
 * feature) or the subgroup-free LDS sibling ({@link createRadixSortLds}) for WebKit. Both honor
 * this same contract; force `false` to exercise the LDS path on a subgroup-capable device.
 *
 * @example
 * const count = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
 * const rs = await createRadixSort(device, 1 << 20, { count });
 * device.queue.writeBuffer(rs.keys, 0, mortonCodes);
 * device.queue.writeBuffer(rs.payload, 0, primIndices);
 * device.queue.writeBuffer(count, 0, new Uint32Array([n]));
 * const enc = device.createCommandEncoder();
 * rs.sortIndirect(enc);
 * device.queue.submit([enc.finish()]);
 */
export async function createRadixSort(
    device: GPUDevice,
    maxKeys: number,
    shared: RadixSortShared = {},
    subgroups: boolean = device.features.has("subgroups"),
): Promise<RadixSort> {
    if (!subgroups) return createRadixSortLds(device, maxKeys, shared);
    const maxBlocks = Math.max(1, Math.ceil(maxKeys / PART_SIZE));
    if (maxBlocks * RADIX_PASSES > MAX_DISPATCH) {
        throw new Error(
            `createRadixSort: maxKeys ${maxKeys} needs ${maxBlocks * RADIX_PASSES} init workgroups, over the ${MAX_DISPATCH} dispatch limit`,
        );
    }
    const paddedMax = maxBlocks * PART_SIZE;

    const owned: GPUBuffer[] = [];
    const pair = (label: string): GPUBuffer => {
        const b = device.createBuffer({
            label: `radix-${label}`,
            size: paddedMax * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        owned.push(b);
        return b;
    };
    const keys = shared.keys ?? pair("keys");
    const payload = shared.payload ?? pair("payload");
    const altKeys = pair("alt-keys");
    const altPayload = pair("alt-payload");
    const buf = (label: string, size: number, usage: number): GPUBuffer => {
        const b = device.createBuffer({ label, size, usage });
        owned.push(b);
        return b;
    };
    // one descriptor per (pass, partition, digit). The single big cross-workgroup buffer.
    const passHist = buf(
        "radix-pass-hist",
        RADIX_PASSES * maxBlocks * RADIX * 4,
        GPUBufferUsage.STORAGE,
    );
    const globalHist = buf("radix-global-hist", RADIX_PASSES * RADIX * 4, GPUBufferUsage.STORAGE);
    const index = buf("radix-index", RADIX_PASSES * 4, GPUBufferUsage.STORAGE);
    const params = Array.from({ length: RADIX_PASSES }, (_, p) =>
        buf(`radix-params-${p}`, 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    );

    const paramLayout = device.createBindGroupLayout({
        label: "radix-params",
        entries: [storageEntry(0, true)],
    });
    const initLayout = device.createBindGroupLayout({
        label: "radix-init",
        entries: [storageEntry(0, false), storageEntry(1, false), storageEntry(2, false)],
    });
    const histLayout = device.createBindGroupLayout({
        label: "radix-global-hist",
        entries: [storageEntry(0, true), storageEntry(1, false)],
    });
    const scanLayout = device.createBindGroupLayout({
        label: "radix-scan",
        entries: [storageEntry(0, false), storageEntry(1, false)],
    });
    const binLayout = device.createBindGroupLayout({
        label: "radix-binning",
        entries: [0, 1, 2, 3, 4, 5].map((b) => storageEntry(b, b < 2)),
    });

    const pipe = (
        label: string,
        code: string,
        layouts: GPUBindGroupLayout[],
    ): Promise<GPUComputePipeline> =>
        device.createComputePipelineAsync({
            label,
            layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
            compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
        });

    const [init, globalHistPipe, scan, binning] = await Promise.all([
        pipe("radix-init", INIT_WGSL, [initLayout]),
        pipe("radix-global-hist", GLOBAL_HIST_WGSL, [histLayout, paramLayout]),
        pipe("radix-scan", SCAN_WGSL, [scanLayout, paramLayout]),
        pipe("radix-binning", BINNING_WGSL, [binLayout, paramLayout]),
    ]);

    const bg = (layout: GPUBindGroupLayout, buffers: GPUBuffer[]): GPUBindGroup =>
        device.createBindGroup({
            layout,
            entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        });

    const initBg = bg(initLayout, [passHist, globalHist, index]);
    const histBg = bg(histLayout, [keys, globalHist]);
    const scanBg = bg(scanLayout, [globalHist, passHist]);
    // binning group(0) indexed [parity]: even pass reads keys → alt, odd reads alt → keys
    const binBg = [
        bg(binLayout, [keys, payload, altKeys, altPayload, passHist, index]),
        bg(binLayout, [altKeys, altPayload, keys, payload, passHist, index]),
    ];
    const paramBg = params.map((p) => bg(paramLayout, [p]));

    const padding = new Uint32Array(PART_SIZE).fill(0xffffffff);

    let prepare: { pipeline: GPUComputePipeline; group: GPUBindGroup; args: GPUBuffer } | null =
        null;
    if (shared.count) {
        const args = buf(
            "radix-indirect-args",
            36,
            GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
        );
        const prepareLayout = device.createBindGroupLayout({
            label: "radix-prepare",
            entries: [0, 1, 2, 3, 4, 5].map((b) => storageEntry(b, b === 0)),
        });
        const pipeline = await pipe("radix-prepare", PREPARE_WGSL, [prepareLayout]);
        prepare = { pipeline, group: bg(prepareLayout, [shared.count, args, ...params]), args };
        // the constant [maxKeys, paddedMax) tail of the last block (count-independent; the
        // Morton pass sentinel-pads [count, maxKeys) on the GPU each build)
        if (paddedMax > maxKeys) {
            device.queue.writeBuffer(keys, maxKeys * 4, padding, 0, paddedMax - maxKeys);
        }
    }

    const span = (): GPUComputePassTimestampWrites | undefined => Compute.span?.("bvh:sort");
    const run = (
        encoder: GPUCommandEncoder,
        pipeline: GPUComputePipeline,
        group0: GPUBindGroup,
        group1: GPUBindGroup | null,
        body: (pass: GPUComputePassEncoder) => void,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: span() });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group0);
        if (group1) pass.setBindGroup(1, group1);
        body(pass);
        pass.end();
    };

    return {
        keys,
        payload,
        maxKeys,
        sortIndirect(encoder: GPUCommandEncoder): void {
            if (!prepare) {
                throw new Error(
                    "sortIndirect: createRadixSort needs a count buffer (shared.count)",
                );
            }
            const { args } = prepare;
            const pp = encoder.beginComputePass({ timestampWrites: span() });
            pp.setPipeline(prepare.pipeline);
            pp.setBindGroup(0, prepare.group);
            pp.dispatchWorkgroups(1);
            pp.end();
            run(encoder, init, initBg, null, (pass) => pass.dispatchWorkgroupsIndirect(args, 0));
            run(encoder, globalHistPipe, histBg, paramBg[0], (pass) =>
                pass.dispatchWorkgroupsIndirect(args, 12),
            );
            run(encoder, scan, scanBg, paramBg[0], (pass) => pass.dispatchWorkgroups(RADIX_PASSES));
            for (let p = 0; p < RADIX_PASSES; p++) {
                run(encoder, binning, binBg[p & 1], paramBg[p], (pass) =>
                    pass.dispatchWorkgroupsIndirect(args, 24),
                );
            }
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
