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

import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { precompile, precompileScope } from "../../engine/runtime";
import { compareExchange, idiv, subgroupUniformityOff, uniformLoad } from "../../engine/utils/core";
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
const SUB_STRIDE = RADIX_PASSES * RADIX; // one sub-histogram's span in the shared array
const MAX_SUB = 64; // max subgroup size on the floor — sizes the scan's per-subgroup partials
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

// Per-pass params. Storage (not uniform) so the indirect prepare can write them from the GPU count.
// `numKeys` is the padded (block-multiple) count; the tail past the real count is 0xffffffff, sorted to
// the end, never read back.
/** @internal */
export const Params = d
    .struct({ numKeys: d.u32, binBlocks: d.u32, histBlocks: d.u32, shift: d.u32 })
    .$name("Params");

/** the per-pass params, its own group so the four binning dispatches swap only it. @internal */
export const paramLayout = tgpu.bindGroupLayout({
    P: { storage: Params, access: "readonly" },
});

/** the cross-workgroup state the init pass zeroes. @internal */
export const initLayout = tgpu.bindGroupLayout({
    passHist: { storage: d.arrayOf(d.u32), access: "mutable" },
    globalHist: { storage: d.arrayOf(d.u32), access: "mutable" },
    index: { storage: d.arrayOf(d.u32), access: "mutable" },
});

/** the global histogram's I/O — the same `globalHist` buffer, bound atomic here. @internal */
export const histLayout = tgpu.bindGroupLayout({
    keys: { storage: d.arrayOf(d.u32), access: "readonly" },
    globalHist: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
});

/** the seed scan's I/O: the device histogram in, partition 0's descriptor out. @internal */
export const scanLayout = tgpu.bindGroupLayout({
    globalHist: { storage: d.arrayOf(d.u32), access: "mutable" },
    passHist: { storage: d.arrayOf(d.u32), access: "mutable" },
});

/** the binning pass's I/O: the ping-pong pair, the tile descriptors, and the partition counter.
 *  @internal */
export const binLayout = tgpu.bindGroupLayout({
    srcKeys: { storage: d.arrayOf(d.u32), access: "readonly" },
    srcPayload: { storage: d.arrayOf(d.u32), access: "readonly" },
    dstKeys: { storage: d.arrayOf(d.u32), access: "mutable" },
    dstPayload: { storage: d.arrayOf(d.u32), access: "mutable" },
    passHist: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" }, // cross-workgroup tile descriptors
    index: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" }, // per-pass partition counter
});

/** the GPU-count prepare's I/O: the count in, the indirect args + all four passes' params out.
 *  @internal */
export const prepareLayout = tgpu.bindGroupLayout({
    count: { storage: d.arrayOf(d.u32), access: "readonly" },
    args: { storage: d.arrayOf(d.u32), access: "mutable" }, // [init xyz][hist xyz][bin xyz]
    p0: { storage: Params, access: "mutable" },
    p1: { storage: Params, access: "mutable" },
    p2: { storage: Params, access: "mutable" },
    p3: { storage: Params, access: "mutable" },
});

// INIT — zero the cross-workgroup state: passHist [0, binBlocks*RADIX*PASSES) (one descriptor
// per (pass, partition, digit)), globalHist [0, RADIX*PASSES), index [0, PASSES) (the per-pass
// partition-assignment counters). Dispatched binBlocks*PASSES workgroups, so it scales with N.
const initKernel = tgpu
    .computeFn({
        workgroupSize: [WG],
        in: { tid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    })((input) => {
        "use gpu";
        // WG == RADIX, so this covers one (pass, partition) row
        initLayout.$.passHist[input.wid.x * WG + input.tid] = 0;
        if (input.wid.x === 0) {
            let i = input.tid;
            while (i < RADIX * RADIX_PASSES) {
                initLayout.$.globalHist[i] = 0;
                i = i + WG;
            }
            if (input.tid < RADIX_PASSES) initLayout.$.index[input.tid] = 0;
        }
    })
    .$name("radixInit");

// two sub-histograms (lane/64) halve the shared-atomic contention before the reduce + device atomicAdd
const gHist = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), 2 * RADIX_PASSES * RADIX));

// GLOBAL HISTOGRAM — all four per-byte digit counts in one pass over the original keys. Valid
// for every pass up front because a byte's digit-count distribution is invariant under the
// stable reorder the prior passes apply (only the per-partition split changes, recovered in
// the binning pass).
const globalHistKernel = tgpu
    .computeFn({
        workgroupSize: [G_HIST_DIM],
        in: { tid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    })((input) => {
        "use gpu";
        let i = input.tid;
        while (i < 2 * RADIX_PASSES * RADIX) {
            std.atomicStore(gHist.$[i], 0);
            i = i + G_HIST_DIM;
        }
        std.workgroupBarrier();

        const subOff = idiv(input.tid, 64) * SUB_STRIDE;
        const tileEnd = std.select(
            (input.wid.x + 1) * G_HIST_PART_SIZE,
            paramLayout.$.P.numKeys,
            input.wid.x === paramLayout.$.P.histBlocks - 1,
        );
        let j = input.tid + input.wid.x * G_HIST_PART_SIZE;
        while (j < tileEnd) {
            const k = histLayout.$.keys[j];
            std.atomicAdd(gHist.$[subOff + (k & RADIX_MASK)], 1);
            std.atomicAdd(gHist.$[subOff + RADIX + ((k >> 8) & RADIX_MASK)], 1);
            std.atomicAdd(gHist.$[subOff + 2 * RADIX + ((k >> 16) & RADIX_MASK)], 1);
            std.atomicAdd(gHist.$[subOff + 3 * RADIX + ((k >> 24) & RADIX_MASK)], 1);
            j = j + G_HIST_DIM;
        }
        std.workgroupBarrier();

        let digit = input.tid;
        while (digit < RADIX) {
            let p = d.u32(0);
            while (p < RADIX_PASSES) {
                const c =
                    std.atomicLoad(gHist.$[p * RADIX + digit]) +
                    std.atomicLoad(gHist.$[SUB_STRIDE + p * RADIX + digit]);
                std.atomicAdd(histLayout.$.globalHist[p * RADIX + digit], c);
                p = p + 1;
            }
            digit = digit + G_HIST_DIM;
        }
    })
    .$name("radixGlobalHist");

const sgSums = tgpu.workgroupVar(d.arrayOf(d.u32, MAX_SUB));

// workgroup-wide exclusive prefix sum (WG = 256): a subgroup scan plus a scan of the subgroup totals.
const wgExclusiveScan = tgpu
    .fn(
        [d.u32, d.u32, d.u32, d.u32],
        d.u32,
    )((tid, sid, sgsize, val) => {
        "use gpu";
        const sgid = idiv(tid, sgsize);
        const inc = std.subgroupInclusiveAdd(val);
        const excl = inc - val;
        if (sid === sgsize - 1) sgSums.$[sgid] = inc;
        std.workgroupBarrier();
        const numSub = idiv(WG, sgsize);
        const tv = std.select(d.u32(0), sgSums.$[sid], sid < numSub);
        const prefix = std.subgroupExclusiveAdd(tv);
        const base = std.subgroupShuffle(prefix, sgid);
        std.workgroupBarrier();
        return base + excl;
    })
    .$name("wgExclusiveScan");

// SCAN — exclusive-scan each pass's 256-bin global histogram into the global digit base, and
// seed it as partition 0's descriptor (flag INCLUSIVE). That seed terminates every chained
// lookback: partition 0 has no predecessors, so its inclusive prefix IS the global base.
const scanKernel = tgpu
    .computeFn({
        workgroupSize: [WG],
        in: {
            tid: d.builtin.localInvocationIndex,
            wid: d.builtin.workgroupId,
            sid: d.builtin.subgroupInvocationId,
            sgsize: d.builtin.subgroupSize,
        },
    })((input) => {
        "use gpu";
        const passIdx = input.wid.x;
        const base = wgExclusiveScan(
            input.tid,
            input.sid,
            input.sgsize,
            scanLayout.$.globalHist[passIdx * RADIX + input.tid],
        );
        scanLayout.$.passHist[passIdx * paramLayout.$.P.binBlocks * RADIX + input.tid] =
            (base << 2) | FLAG_INCLUSIVE;
    })
    .$name("radixScan");

// The lookback's `subgroupAny` / `subgroupAll` run inside a `workgroupUniformLoad`-gated loop, which
// WGSL's uniformity analysis rejects on principle: it treats every subgroup-reduction result as
// non-uniform for control flow, and the gate is uniform only at runtime (gpu.md "Subgroup ops in
// data-dependent loops"). The sanctioned opt-out is a module-scope diagnostic, and it can only ride a
// WGSL-bodied function's `$uses` — `$uses` throws on a body the transform produced. So these two
// wrappers exist to carry it, and the loop below is capped by construction (it walks strictly
// decreasing partition indices from `part` down to the seeded partition 0).
const lookbackAny = tgpu
    .fn(
        [d.bool],
        d.bool,
    )(/* wgsl */ `(x: bool) -> bool { return subgroupAny(x); }`)
    .$name("lookbackAny");

const lookbackAll = tgpu
    .fn(
        [d.bool],
        d.bool,
    )(/* wgsl */ `(x: bool) -> bool { return subgroupAll(x); }`)
    .$name("lookbackAll");

// The closest TGSL comes to writing `diagnostic(off, subgroup_uniformity);` at the top of the shader:
// call this first and the directive lands at module scope. Two typegpu facts force the shape — a
// directive can only ride a WGSL-bodied function's `$uses` (`$uses` throws on a body the transform
// produced), and declarations emit in *first-use* order while WGSL requires every directive ahead of
// every global declaration. So the host has to be the first thing the kernel resolves, and it must take
// no argument (an argument's own declarations would resolve ahead of it). Move the call off line one and
// the module stops parsing at pipeline creation — the `accel` bench is the gate.
const uniformityOptOut = tgpu
    .fn([])(/* wgsl */ `() {
    diagnosticOff
}`)
    // the directive resolves to nothing in place; naming it in the template is what keeps typegpu from
    // warning about an unused external
    .$uses({ diagnosticOff: subgroupUniformityOff })
    .$name("uniformityOptOut");

// Sorted partition slot [0, PART_SIZE) + per-digit device base [PART_SIZE, PART_SIZE+RADIX).
// During the lookback (before the key scatter) gD[0] is the completed-subgroup counter, gD[1]
// the fallback trigger, gD[RADIX + d] the cooperative fallback histogram.
const gD = tgpu.workgroupVar(d.arrayOf(d.atomic(d.u32), G_D));
const wgPart = tgpu.workgroupVar(d.u32);
const wgDone = tgpu.workgroupVar(d.u32); // lookback early-exit gate, read via workgroupUniformLoad

const Keys = d.arrayOf(d.u32, KEYS_PER_THREAD);
const Ballot = d.arrayOf(d.u32, 4);

// BINNING PASS (one per digit) — the Onesweep downsweep's per-block subgroup multisplit ranking,
// then the chained lookback-with-fallback to recover the global digit offset (where a per-pass
// reduce-then-scan would precompute digitBase + blockHist instead). The CAS broadcast of this
// partition's digit count to its successor's descriptor is spliced where the across-wave
// reduction is live, so the successor's lookback can resolve against it.
const binningKernel = tgpu
    .computeFn({
        workgroupSize: [WG],
        in: {
            tid: d.builtin.localInvocationIndex,
            sid: d.builtin.subgroupInvocationId,
            sgsize: d.builtin.subgroupSize,
        },
    })((input) => {
        "use gpu";
        uniformityOptOut();
        const shift = paramLayout.$.P.shift;
        const binBlocks = paramLayout.$.P.binBlocks;
        const passIdx = shift >> 3;
        const waveIndex = idiv(input.tid, input.sgsize);
        const waveHists = idiv(WG, input.sgsize) * RADIX;

        // 1. clear per-wave histograms, then claim a partition by atomic counter (the assignment
        //    order tracks execution order — what makes the decoupled lookback meaningful)
        let c = input.tid;
        while (c < waveHists) {
            std.atomicStore(gD.$[c], 0);
            c = c + WG;
        }
        if (input.tid === 0) wgPart.$ = std.atomicAdd(binLayout.$.index[passIdx], 1);
        std.workgroupBarrier();
        const part = wgPart.$;

        // 2. load this partition's keys (full; the tail is 0xffffffff-padded in the source buffer)
        const devBase = input.sid + waveIndex * (KEYS_PER_THREAD * input.sgsize) + part * PART_SIZE;
        const k = Keys();
        let i = d.u32(0);
        while (i < KEYS_PER_THREAD) {
            k[i] = d.u32(binLayout.$.srcKeys[devBase + i * input.sgsize]);
            i = i + 1;
        }

        // 3. rank within the wave by subgroup multisplit; accumulate the per-wave histogram
        const offsets = Keys();
        const waveParts = idiv(input.sgsize + 31, 32);
        let slot = d.u32(0);
        while (slot < KEYS_PER_THREAD) {
            const key = k[slot];
            const flags = Ballot([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
            let b = d.u32(0);
            while (b < 8) {
                const bit = (key >> (b + shift)) & 1;
                const ballot = std.subgroupBallot(bit === 1);
                const bArr = Ballot([ballot.x, ballot.y, ballot.z, ballot.w]);
                const m = std.select(d.u32(0xffffffff), d.u32(0), bit === 1);
                let wp = d.u32(0);
                while (wp < waveParts) {
                    flags[wp] = flags[wp] & (m ^ bArr[wp]);
                    wp = wp + 1;
                }
                b = b + 1;
            }
            const digit = (key >> shift) & RADIX_MASK;
            const idx = digit + waveIndex * RADIX;
            let lowest = d.u32(0);
            let scan = d.u32(0);
            while (scan < waveParts) {
                const fbl = std.firstTrailingBit(flags[scan]);
                if (fbl === 0xffffffff) {
                    lowest = lowest + 32;
                    scan = scan + 1;
                } else {
                    lowest = lowest + fbl;
                    break;
                }
            }
            let peerBits = d.u32(0);
            let totalBits = d.u32(0);
            let wp2 = d.u32(0);
            while (wp2 < waveParts) {
                if (input.sid >= wp2 * 32) {
                    const ltMask = std.select(
                        (d.u32(1) << (input.sid & 31)) - 1,
                        d.u32(0xffffffff),
                        input.sid >= (wp2 + 1) * 32,
                    );
                    peerBits = peerBits + std.countOneBits(flags[wp2] & ltMask);
                }
                totalBits = totalBits + std.countOneBits(flags[wp2]);
                wp2 = wp2 + 1;
            }
            let pre = d.u32(0);
            if (peerBits === 0) pre = std.atomicAdd(gD.$[idx], totalBits);
            offsets[slot] = std.subgroupShuffle(pre, lowest) + peerBits;
            slot = slot + 1;
        }
        std.workgroupBarrier();

        // 4a. per digit: inclusive scan across waves with a circular shift, leaving each wave's
        //     exclusive-across-waves prefix in place. histReduction = this partition's digit count.
        let histReduction = std.atomicLoad(gD.$[input.tid]);
        let h = input.tid + RADIX;
        while (h < waveHists) {
            const hc = std.atomicLoad(gD.$[h]);
            histReduction = histReduction + hc;
            std.atomicStore(gD.$[h], histReduction - hc);
            h = h + RADIX;
        }
        // CAS-broadcast this partition's digit count to its successor's descriptor (write-if-unset:
        // a fallback may have already published it). The one edit vs the shared DOWNSWEEP ranking.
        if (part < binBlocks - 1) {
            const succ = (passIdx * binBlocks + part + 1) * RADIX + input.tid;
            compareExchange(binLayout.$.passHist[succ], 0, FLAG_REDUCTION | (histReduction << 2));
        }
        // 4b. begin the cross-digit scan: inclusive within the subgroup
        histReduction = histReduction + std.subgroupExclusiveAdd(histReduction);
        std.workgroupBarrier();
        // 4c. finish the cross-digit exclusive scan into gD[digit] (block-local digit base)
        const laneMask = input.sgsize - 1;
        std.atomicStore(
            gD.$[((input.sid + 1) & laneMask) + (input.tid & ~laneMask)],
            histReduction,
        );
        std.workgroupBarrier();
        const numSub = idiv(RADIX, input.sgsize);
        const si = std.select(d.u32(0), input.tid * input.sgsize, input.tid < numSub);
        const ss = std.subgroupExclusiveAdd(std.atomicLoad(gD.$[si]));
        if (input.tid < numSub) std.atomicStore(gD.$[input.tid * input.sgsize], ss);
        std.workgroupBarrier();
        const prevIdx = std.select(input.tid - 1, d.u32(0), input.sid === 0);
        const cbase = std.subgroupBroadcast(std.atomicLoad(gD.$[prevIdx]), d.u32(1));
        if (input.sid !== 0)
            std.atomicStore(gD.$[input.tid], std.atomicLoad(gD.$[input.tid]) + cbase);
        std.workgroupBarrier();

        // 5. block-local position = wave-local rank + cross-wave base + block-local digit base
        let s5 = d.u32(0);
        while (s5 < KEYS_PER_THREAD) {
            const digit = (k[s5] >> shift) & RADIX_MASK;
            if (input.tid >= input.sgsize)
                offsets[s5] =
                    offsets[s5] +
                    std.atomicLoad(gD.$[digit + waveIndex * RADIX]) +
                    std.atomicLoad(gD.$[digit]);
            else offsets[s5] = offsets[s5] + std.atomicLoad(gD.$[digit]);
            s5 = s5 + 1;
        }

        // 6. capture the block-local digit base, then repurpose gD[0] as the completed-subgroup
        //    counter + gD[1] as the fallback trigger; wgDone is the early-exit gate
        const exclusiveHistReduction = std.atomicLoad(gD.$[input.tid]);
        std.workgroupBarrier();
        if (input.tid === 0) {
            std.atomicStore(gD.$[0], 0);
            std.atomicStore(gD.$[1], 0);
            wgDone.$ = 0;
        }
        std.workgroupBarrier();

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
        let lookbackReduction = d.u32(0);
        let lookbackComplete = false;
        let warpComplete = false;
        let lookbackPart = part;
        let spinCount = d.u32(0);
        while (true) {
            if (uniformLoad(wgDone.$) !== 0) break;
            const descIdx = (passIdx * binBlocks + lookbackPart) * RADIX + input.tid;
            let flagPayload = d.u32(0);
            if (!warpComplete && !lookbackComplete) {
                while (true) {
                    flagPayload = std.atomicLoad(binLayout.$.passHist[descIdx]);
                    if ((flagPayload & FLAG_MASK) !== FLAG_NOT_READY) break;
                    spinCount = spinCount + 1;
                    if (spinCount >= MAX_SPIN) break;
                }
            }
            const notReady =
                !warpComplete && !lookbackComplete && (flagPayload & FLAG_MASK) === FLAG_NOT_READY;
            if (!warpComplete) {
                if (lookbackAny(notReady) && input.sid === 0) std.atomicOr(gD.$[1], 1);
            }
            std.workgroupBarrier();

            const doFallback = std.atomicLoad(gD.$[1]) !== 0;
            std.atomicStore(gD.$[RADIX + input.tid], 0);
            std.workgroupBarrier();
            if (input.tid === 0) std.atomicStore(gD.$[1], 0);
            if (doFallback) {
                const fbBase = (lookbackPart - 1) * PART_SIZE;
                let f = input.tid;
                while (f < PART_SIZE) {
                    std.atomicAdd(
                        gD.$[RADIX + ((binLayout.$.srcKeys[fbBase + f] >> shift) & RADIX_MASK)],
                        1,
                    );
                    f = f + WG;
                }
            }
            std.workgroupBarrier();

            if (!warpComplete && !lookbackComplete) {
                if (doFallback) {
                    const recomputed = std.atomicLoad(gD.$[RADIX + input.tid]);
                    const old = compareExchange(
                        binLayout.$.passHist[descIdx],
                        0,
                        FLAG_REDUCTION | (recomputed << 2),
                    );
                    if ((old & FLAG_MASK) === FLAG_INCLUSIVE) {
                        lookbackReduction = lookbackReduction + (old >> 2);
                        lookbackComplete = true;
                    } else {
                        lookbackReduction = lookbackReduction + recomputed;
                    }
                } else {
                    lookbackReduction = lookbackReduction + (flagPayload >> 2);
                    if ((flagPayload & FLAG_MASK) === FLAG_INCLUSIVE) lookbackComplete = true;
                }
                spinCount = 0;
                // on completion, publish this partition's inclusive prefix to its successor so it
                // need not look back past us (bumps the successor's REDUCTION flag to INCLUSIVE)
                if (lookbackComplete && part < binBlocks - 1) {
                    const succ = (passIdx * binBlocks + part + 1) * RADIX + input.tid;
                    std.atomicAdd(binLayout.$.passHist[succ], 1 | (lookbackReduction << 2));
                }
            }
            lookbackPart = lookbackPart - 1;
            if (!warpComplete) {
                warpComplete = lookbackAll(lookbackComplete);
                if (warpComplete && input.sid === 0) std.atomicAdd(gD.$[0], 1);
            }
            std.workgroupBarrier();
            if (input.tid === 0 && std.atomicLoad(gD.$[0]) >= numSub) wgDone.$ = 1;
        }
        // device base for digit input.tid = inclusive device prefix - block-local digit base
        std.atomicStore(gD.$[input.tid + PART_SIZE], lookbackReduction - exclusiveHistReduction);
        std.workgroupBarrier();

        // 8. scatter keys into shared at their block-local sorted position
        let s8 = d.u32(0);
        while (s8 < KEYS_PER_THREAD) {
            std.atomicStore(gD.$[offsets[s8]], k[s8]);
            s8 = s8 + 1;
        }
        std.workgroupBarrier();

        // 9. scatter sorted keys to the device; capture each slot's digit for the payload pass
        const digits = Keys();
        let s9 = d.u32(0);
        while (s9 < KEYS_PER_THREAD) {
            const t = input.tid + s9 * WG;
            const key = std.atomicLoad(gD.$[t]);
            const dg = (key >> shift) & RADIX_MASK;
            digits[s9] = dg;
            binLayout.$.dstKeys[std.atomicLoad(gD.$[dg + PART_SIZE]) + t] = key;
            s9 = s9 + 1;
        }
        std.workgroupBarrier();

        // 10. payloads: load in source order, scatter to shared at the key offsets, then to device
        const pl = Keys();
        let s10 = d.u32(0);
        while (s10 < KEYS_PER_THREAD) {
            pl[s10] = d.u32(binLayout.$.srcPayload[devBase + s10 * input.sgsize]);
            s10 = s10 + 1;
        }
        let s11 = d.u32(0);
        while (s11 < KEYS_PER_THREAD) {
            std.atomicStore(gD.$[offsets[s11]], pl[s11]);
            s11 = s11 + 1;
        }
        std.workgroupBarrier();
        let s12 = d.u32(0);
        while (s12 < KEYS_PER_THREAD) {
            const t = input.tid + s12 * WG;
            binLayout.$.dstPayload[std.atomicLoad(gD.$[digits[s12] + PART_SIZE]) + t] =
                std.atomicLoad(gD.$[t]);
            s12 = s12 + 1;
        }
    })
    .$name("radixBinning");

// GPU-count prepare (indirect path): derive binBlocks/histBlocks from the count, write the per-
// pass params + the init/hist/binning indirect dispatch args. Scan stays direct (RADIX_PASSES).
const prepareKernel = tgpu
    .computeFn({ workgroupSize: [1] })(() => {
        "use gpu";
        const binBlocks = std.max(
            d.u32(1),
            idiv(prepareLayout.$.count[0] + (PART_SIZE - 1), PART_SIZE),
        );
        const padded = binBlocks * PART_SIZE;
        const histBlocks = std.max(
            d.u32(1),
            idiv(padded + (G_HIST_PART_SIZE - 1), G_HIST_PART_SIZE),
        );
        prepareLayout.$.args[0] = binBlocks * RADIX_PASSES;
        prepareLayout.$.args[1] = 1;
        prepareLayout.$.args[2] = 1;
        prepareLayout.$.args[3] = histBlocks;
        prepareLayout.$.args[4] = 1;
        prepareLayout.$.args[5] = 1;
        prepareLayout.$.args[6] = binBlocks;
        prepareLayout.$.args[7] = 1;
        prepareLayout.$.args[8] = 1;
        prepareLayout.$.p0 = Params({ numKeys: padded, binBlocks, histBlocks, shift: 0 });
        prepareLayout.$.p1 = Params({ numKeys: padded, binBlocks, histBlocks, shift: 8 });
        prepareLayout.$.p2 = Params({ numKeys: padded, binBlocks, histBlocks, shift: 16 });
        prepareLayout.$.p3 = Params({ numKeys: padded, binBlocks, histBlocks, shift: 24 });
    })
    .$name("radixPrepare");

/** the emitted Onesweep WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function radixWgsl(): {
    init: string;
    globalHist: string;
    scan: string;
    binning: string;
    prepare: string;
} {
    const names = { names: "strict" } as const;
    return {
        init: tgpu.resolve([initKernel], names),
        globalHist: tgpu.resolve([globalHistKernel], names),
        scan: tgpu.resolve([scanKernel], names),
        binning: tgpu.resolve([binningKernel], names),
        prepare: tgpu.resolve([prepareKernel], names),
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
    const root = Compute.root;
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

    const paramGroups = params.map((p) => root.createBindGroup(paramLayout, { P: p }));
    const initBound = root
        .createComputePipeline({ compute: initKernel })
        .$name("radix-init")
        .with(root.createBindGroup(initLayout, { passHist, globalHist, index }));
    const histBound = root
        .createComputePipeline({ compute: globalHistKernel })
        .$name("radix-global-hist")
        .with(root.createBindGroup(histLayout, { keys, globalHist }))
        .with(paramGroups[0]);
    const scanBound = root
        .createComputePipeline({ compute: scanKernel })
        .$name("radix-scan")
        .with(root.createBindGroup(scanLayout, { globalHist, passHist }))
        .with(paramGroups[0]);
    // the binning group is indexed [parity]: an even pass reads keys → alt, an odd one alt → keys
    const binningPipe = root
        .createComputePipeline({ compute: binningKernel })
        .$name("radix-binning");
    const binGroups = [
        root.createBindGroup(binLayout, {
            srcKeys: keys,
            srcPayload: payload,
            dstKeys: altKeys,
            dstPayload: altPayload,
            passHist,
            index,
        }),
        root.createBindGroup(binLayout, {
            srcKeys: altKeys,
            srcPayload: altPayload,
            dstKeys: keys,
            dstPayload: payload,
            passHist,
            index,
        }),
    ];
    const binBound = Array.from({ length: RADIX_PASSES }, (_, p) =>
        binningPipe.with(binGroups[p & 1]).with(paramGroups[p]),
    );

    const padding = new Uint32Array(PART_SIZE).fill(0xffffffff);

    let prepare: { bound: TgpuComputePipeline; args: GPUBuffer } | null = null;
    if (shared.count) {
        const args = buf(
            "radix-indirect-args",
            36,
            GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
        );
        const bound = root
            .createComputePipeline({ compute: prepareKernel })
            .$name("radix-prepare")
            .with(
                root.createBindGroup(prepareLayout, {
                    count: shared.count,
                    args,
                    p0: params[0],
                    p1: params[1],
                    p2: params[2],
                    p3: params[3],
                }),
            );
        prepare = { bound, args };
        // the constant [maxKeys, paddedMax) tail of the last block (count-independent; the
        // Morton pass sentinel-pads [count, maxKeys) on the GPU each build)
        if (paddedMax > maxKeys) {
            device.queue.writeBuffer(keys, maxKeys * 4, padding, 0, paddedMax - maxKeys);
        }
    }

    // labels are per-sorter: one app can stand up several (a BVH sorts internally, and a consumer may
    // hold its own beside it), and the precompile queue rejects a duplicate label
    const scope = precompileScope("radix");
    for (const [label, bound] of [
        ["init", initBound],
        ["global-hist", histBound],
        ["scan", scanBound],
        ["binning", binBound[0]],
        ...(prepare ? ([["prepare", prepare.bound]] as const) : []),
    ] as const) {
        precompile(`${scope}-${label}`, () => {
            bound.dispatchWorkgroups(0);
            return bound;
        });
    }

    const span = (): GPUComputePassTimestampWrites | undefined => Compute.span?.("bvh:sort");
    const run = (
        encoder: GPUCommandEncoder,
        bound: TgpuComputePipeline,
        body: (p: TgpuComputePipeline) => void,
    ): void => {
        const pass = encoder.beginComputePass({ timestampWrites: span() });
        body(bound.with(pass));
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
            run(encoder, prepare.bound, (p) => p.dispatchWorkgroups(1));
            run(encoder, initBound, (p) => p.dispatchWorkgroupsIndirect(args, 0));
            run(encoder, histBound, (p) => p.dispatchWorkgroupsIndirect(args, 12));
            run(encoder, scanBound, (p) => p.dispatchWorkgroups(RADIX_PASSES));
            for (let p = 0; p < RADIX_PASSES; p++) {
                run(encoder, binBound[p], (bp) => bp.dispatchWorkgroupsIndirect(args, 24));
            }
        },
        destroy(): void {
            for (const b of owned) b.destroy();
        },
    };
}
