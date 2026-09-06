// GPU butterfly FFT — one workgroup processes each row or column with shared-memory radix-2
// thread per output element, an O(N) inner loop per thread) with a shared-memory radix-2
// decimation-in-time FFT: one workgroup per row (or column), `N/2` threads, `log2(N)` synchronized
// butterfly stages in workgroup-shared memory (`var<workgroup>`, `gpu.md`'s LDS convention).
//
// `N` must be a power of two — every kernel here is baked for one specific `N` at factory-call time
// (`workgroupSize` and the shared-memory array length must be compile-time constants in WGSL, so `N`
// is a closed-over JS constant, the same factory pattern `standard/glaze/composite.ts`'s
// `glazeKernel(layout)` uses). `getRowFftKernel`/`getColFftKernel` memoize by `N` so the two shipped
// cascades (and the N-invariance probe, which reuses the same two `N`) share one compiled kernel
// object each rather than rebuilding per cascade instance.
//
// TGSL compiles plain `u32 / u32` as real division. This kernel derives a butterfly pair's
// (block, local-index) split from the thread id, so integer division goes through `idiv`; `%`
// (WGSL's native integer remainder) is unaffected and used directly, per the same rule.
//
// Direction: unnormalized inverse only (`sign = +1`, matching `fft.ts`'s `fft1dInPlace` and the
// `idft2` phase convention `Σ_k in[k]·exp(+i·2π·k·n/N)`, no `1/N`). Nothing in this
// package ever needs the forward direction — `generateH0` already produces its output in the
// frequency domain.

import { Compute, probeBuffer } from "@dylanebert/shallot";
import { idiv } from "@dylanebert/shallot/utils/core";
import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/** Complex number as vec2f: .x = real, .y = imag — same layout `ocean.ts`'s `Complex` uses. */
const Complex = d.vec2f;

const PI = Math.PI;

function isPow2(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}

/**
 * Twiddle angle for the butterfly stage of length `len` at local index `blockLocal` — the SAME
 * expression `makeRowFftKernel`/`makeColFftKernel` evaluate at every stage, every frame. Exported
 * so a CPU-side measurement of the device's own trig accuracy (`measureTwiddleTrigError`, below)
 * drives the real kernel through this one shared formula rather than a hand-copied second
 * version that could silently drift from what the WGSL actually emits. Calling this directly from JS still round-trips
 * through typegpu's `d.f32` return-type cast, so it is an f32-ROUNDED value, not a device
 * measurement — see that function's own docblock for why a real GPU dispatch is still required.
 */
export const twiddleAngle = tgpu.fn(
    [d.u32, d.u32],
    d.f32,
)((blockLocal, len) => {
    "use gpu";
    return (d.f32(2) * PI * d.f32(blockLocal)) / d.f32(len);
});

/** Per-N bind group layout: unsized complex storage arrays, input read-only, output mutable. */
export function makeFftLayout(N: number) {
    void N; // N is baked into the kernel body, not the layout — kept as a parameter for call-site symmetry
    return tgpu.bindGroupLayout({
        input: { storage: d.arrayOf(Complex), access: "readonly" },
        output: { storage: d.arrayOf(Complex), access: "mutable" },
    });
}

export type FftLayout = ReturnType<typeof makeFftLayout>;

/** Builds a fresh row-FFT `computeFn` for one specific power-of-two `N` (see the module docblock for
 * why `N` must be baked at factory-call time). One workgroup per row, `N/2` threads; bit-reversal
 * permutation happens on load into shared memory, so no separate unshuffle pass is needed. */
export function makeRowFftKernel(N: number, layout: FftLayout) {
    if (!isPow2(N)) throw new Error(`makeRowFftKernel: N must be a power of two, got ${N}`);
    const bits = Math.log2(N);
    const half = N / 2;
    const reLds = tgpu.workgroupVar(d.arrayOf(d.f32, N));
    const imLds = tgpu.workgroupVar(d.arrayOf(d.f32, N));

    return tgpu
        .computeFn({
            workgroupSize: [half],
            in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
        })((input) => {
            "use gpu";
            const tid = input.lid.x;
            const row = input.wid.x;

            // bit-reversal permutation on load — each of the N/2 threads loads its own pair
            // (tid, tid+half) from the row, bit-reverses each index, and seeds shared memory there.
            let revA = d.u32(0);
            let vA = tid;
            for (let b = 0; b < bits; b++) {
                revA = (revA << d.u32(1)) | (vA & d.u32(1));
                vA = vA >> d.u32(1);
            }
            let revB = d.u32(0);
            let vB = tid + d.u32(half);
            for (let b = 0; b < bits; b++) {
                revB = (revB << d.u32(1)) | (vB & d.u32(1));
                vB = vB >> d.u32(1);
            }
            const gA = layout.$.input[row * d.u32(N) + tid];
            const gB = layout.$.input[row * d.u32(N) + tid + d.u32(half)];
            reLds.$[revA] = gA.x;
            imLds.$[revA] = gA.y;
            reLds.$[revB] = gB.x;
            imLds.$[revB] = gB.y;
            std.workgroupBarrier();

            // iterative radix-2 DIT butterfly stages — N/2 butterflies per stage, one per thread.
            let len = d.u32(2);
            while (len <= d.u32(N)) {
                const halfLen = len >> d.u32(1);
                const block = idiv(tid, halfLen);
                const blockLocal = tid % halfLen; // native u32 remainder — the u32/u32 hazard is `/` only
                const evenIdx = block * len + blockLocal;
                const oddIdx = evenIdx + halfLen;
                const angle = twiddleAngle(blockLocal, len);
                const wr = std.cos(angle);
                const wi = std.sin(angle);
                const evenR = reLds.$[evenIdx];
                const evenIm = imLds.$[evenIdx];
                const oddR = reLds.$[oddIdx];
                const oddIm = imLds.$[oddIdx];
                const tr = oddR * wr - oddIm * wi;
                const ti = oddR * wi + oddIm * wr;
                reLds.$[evenIdx] = evenR + tr;
                imLds.$[evenIdx] = evenIm + ti;
                reLds.$[oddIdx] = evenR - tr;
                imLds.$[oddIdx] = evenIm - ti;
                std.workgroupBarrier();
                len = len << d.u32(1);
            }

            // naturally-ordered output (bit-reversal happened on load) — each thread writes its pair.
            layout.$.output[row * d.u32(N) + tid] = Complex(reLds.$[tid], imLds.$[tid]);
            layout.$.output[row * d.u32(N) + tid + d.u32(half)] = Complex(
                reLds.$[tid + d.u32(half)],
                imLds.$[tid + d.u32(half)],
            );
        })
        .$name(`ocean-fft-row-${N}`);
}

/** Builds a fresh column-FFT `computeFn` for one specific power-of-two `N` — identical butterfly
 * logic to {@link makeRowFftKernel}, differing only in the (strided) global index formula: one
 * workgroup per COLUMN, reading/writing `input[k*N + col]` instead of `input[row*N + k]`. */
export function makeColFftKernel(N: number, layout: FftLayout) {
    if (!isPow2(N)) throw new Error(`makeColFftKernel: N must be a power of two, got ${N}`);
    const bits = Math.log2(N);
    const half = N / 2;
    const reLds = tgpu.workgroupVar(d.arrayOf(d.f32, N));
    const imLds = tgpu.workgroupVar(d.arrayOf(d.f32, N));

    return tgpu
        .computeFn({
            workgroupSize: [half],
            in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
        })((input) => {
            "use gpu";
            const tid = input.lid.x;
            const col = input.wid.x;

            let revA = d.u32(0);
            let vA = tid;
            for (let b = 0; b < bits; b++) {
                revA = (revA << d.u32(1)) | (vA & d.u32(1));
                vA = vA >> d.u32(1);
            }
            let revB = d.u32(0);
            let vB = tid + d.u32(half);
            for (let b = 0; b < bits; b++) {
                revB = (revB << d.u32(1)) | (vB & d.u32(1));
                vB = vB >> d.u32(1);
            }
            const gA = layout.$.input[tid * d.u32(N) + col];
            const gB = layout.$.input[(tid + d.u32(half)) * d.u32(N) + col];
            reLds.$[revA] = gA.x;
            imLds.$[revA] = gA.y;
            reLds.$[revB] = gB.x;
            imLds.$[revB] = gB.y;
            std.workgroupBarrier();

            let len = d.u32(2);
            while (len <= d.u32(N)) {
                const halfLen = len >> d.u32(1);
                const block = idiv(tid, halfLen);
                const blockLocal = tid % halfLen;
                const evenIdx = block * len + blockLocal;
                const oddIdx = evenIdx + halfLen;
                const angle = twiddleAngle(blockLocal, len);
                const wr = std.cos(angle);
                const wi = std.sin(angle);
                const evenR = reLds.$[evenIdx];
                const evenIm = imLds.$[evenIdx];
                const oddR = reLds.$[oddIdx];
                const oddIm = imLds.$[oddIdx];
                const tr = oddR * wr - oddIm * wi;
                const ti = oddR * wi + oddIm * wr;
                reLds.$[evenIdx] = evenR + tr;
                imLds.$[evenIdx] = evenIm + ti;
                reLds.$[oddIdx] = evenR - tr;
                imLds.$[oddIdx] = evenIm - ti;
                std.workgroupBarrier();
                len = len << d.u32(1);
            }

            layout.$.output[tid * d.u32(N) + col] = Complex(reLds.$[tid], imLds.$[tid]);
            layout.$.output[(tid + d.u32(half)) * d.u32(N) + col] = Complex(
                reLds.$[tid + d.u32(half)],
                imLds.$[tid + d.u32(half)],
            );
        })
        .$name(`ocean-fft-col-${N}`);
}

interface FftKernelSet {
    layout: FftLayout;
    rowKernel: ReturnType<typeof makeRowFftKernel>;
    colKernel: ReturnType<typeof makeColFftKernel>;
}

const kernelCache = new Map<number, FftKernelSet>();

/** memoized per-N kernel set (layout + row/col compute fns) — every cascade/probe at the same `N`
 *  shares one compiled kernel pair rather than re-authoring identical WGSL. */
export function getFftKernels(N: number): FftKernelSet {
    let entry = kernelCache.get(N);
    if (!entry) {
        const layout = makeFftLayout(N);
        entry = {
            layout,
            rowKernel: makeRowFftKernel(N, layout),
            colKernel: makeColFftKernel(N, layout),
        };
        kernelCache.set(N, entry);
    }
    return entry;
}

// ── exhaustive device twiddle-trig measurement ──────────────────────────────────────────────
//
// Higham's radix-2 FFT forward-error theorem (below, `examples/showcase/ocean/src/ocean/slope-seam.ts`'s
// `higham242Bound`) bounds the FFT's OWN floating-point rounding error under unit roundoff `u`,
// but says nothing about how far a real device's `cos`/`sin` intrinsics land from the
// mathematically exact value — WGSL leaves transcendental built-ins to an accuracy CLASS, not an
// exact result, so that term has to be MEASURED on the seat under test, never authored (the I3g
// re-verdict's own convicted class: "a computation-accuracy term taken from a permitted floor
// either cannot discriminate... or is a fit... measure it under solve/held-out; never author it").
// This is that measurement: dispatch a real compute pass evaluating `twiddleAngle` + `std.cos`/
// `std.sin` at EVERY `(len, blockLocal)` pair the row/col FFT kernels evaluate for transform
// length `N` — there are exactly `N - 1` such pairs (`len` doubling from 2 to `N`, `blockLocal`
// spanning `[0, len/2)` at each) — and compare each against `Math.cos`/`Math.sin` of the exact f64
// angle. `μ_max` is the maximum absolute deviation over the whole population, on EITHER function.

const twiddleProbeParams = d.struct({ count: d.u32, pad: d.vec3u }).$name("TwiddleProbeParams");
const twiddleProbeLayout = tgpu.bindGroupLayout({
    pairs: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    out: { storage: d.arrayOf(d.vec2f), access: "mutable" },
    params: { uniform: twiddleProbeParams },
});

/** `pairs[i] = (len, blockLocal)`; `out[i] = (cos(twiddleAngle(blockLocal, len)),
 *  sin(twiddleAngle(blockLocal, len)))`, evaluated by the SAME `twiddleAngle` the production FFT
 *  kernels call — never a hand-copied angle formula. */
const twiddleProbeKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const idx = input.gid.x;
        if (idx >= twiddleProbeLayout.$.params.count) return;
        const pair = twiddleProbeLayout.$.pairs[idx];
        const angle = twiddleAngle(pair.y, pair.x);
        twiddleProbeLayout.$.out[idx] = d.vec2f(std.cos(angle), std.sin(angle));
    })
    .$name("ocean-fft-twiddle-probe");

export interface TwiddleErrorReading {
    /** population size — exactly `N - 1` for a transform of length `N`. */
    pairs: number;
    /** max(maxCosError, maxSinError) — the single number `slope-seam.ts` feeds into E0. */
    muMax: number;
    maxCosError: number;
    maxSinError: number;
}

/** Exhaustively measures this device's own `cos`/`sin` deviation from the mathematically exact
 *  value at every twiddle angle the length-`N` radix-2 FFT evaluates. Requires a real GPU adapter
 *  (`Compute.device`) — there is no CPU stand-in: typegpu's CPU execution of a `tgpu.fn` only
 *  rounds its RETURN value once (verified: `twiddleAngle(3, 16)` called directly from JS reads
 *  `Math.fround((2*Math.PI*3)/16)` exactly), which cannot represent what a real WGSL runtime's
 *  `cos`/`sin` intrinsics compute on real hardware — only a device dispatch can. */
export async function measureTwiddleTrigError(N: number): Promise<TwiddleErrorReading> {
    if (!isPow2(N)) throw new Error(`measureTwiddleTrigError: N must be a power of two, got ${N}`);
    const { device, root } = Compute;
    const pairs: number[] = [];
    for (let len = 2; len <= N; len <<= 1) {
        for (let blockLocal = 0; blockLocal < len / 2; blockLocal++) pairs.push(len, blockLocal);
    }
    const count = pairs.length / 2;
    const pairsBuffer = device.createBuffer({
        label: "ocean-fft-twiddle-pairs",
        size: count * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(pairsBuffer, 0, new Uint32Array(pairs));
    const outBuffer = device.createBuffer({
        label: "ocean-fft-twiddle-out",
        size: count * 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const paramsBuffer = device.createBuffer({
        label: "ocean-fft-twiddle-params",
        size: d.sizeOf(twiddleProbeParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([count, 0, 0, 0]));
    const pairsT = root.createBuffer(d.arrayOf(d.vec2u, count), pairsBuffer).$usage("storage");
    const outT = root.createBuffer(d.arrayOf(d.vec2f, count), outBuffer).$usage("storage");
    const paramsT = root.createBuffer(twiddleProbeParams, paramsBuffer).$usage("uniform");
    const pipeline = root.createComputePipeline({ compute: twiddleProbeKernel });
    const group = root.createBindGroup(twiddleProbeLayout, {
        pairs: pairsT,
        out: outT,
        params: paramsT,
    });

    const probe = await probeBuffer(device, outBuffer, {
        label: "ocean-fft-twiddle-probe",
        encode: (encoder) => {
            const pass = encoder.beginComputePass({
                label: "ocean-fft-twiddle-probe",
                timestampWrites: Compute.span?.("ocean-fft-twiddle-probe"),
            });
            pipeline
                .with(group)
                .with(pass)
                .dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        },
    });
    const out = new Float32Array(probe.bytes);
    pairsBuffer.destroy();
    outBuffer.destroy();
    paramsBuffer.destroy();

    let maxCosError = 0;
    let maxSinError = 0;
    for (let i = 0; i < count; i++) {
        const len = pairs[i * 2];
        const blockLocal = pairs[i * 2 + 1];
        const angleExact = (2 * Math.PI * blockLocal) / len;
        const cosErr = Math.abs(out[i * 2] - Math.cos(angleExact));
        const sinErr = Math.abs(out[i * 2 + 1] - Math.sin(angleExact));
        if (cosErr > maxCosError) maxCosError = cosErr;
        if (sinErr > maxSinError) maxSinError = sinErr;
    }
    return { pairs: count, muMax: Math.max(maxCosError, maxSinError), maxCosError, maxSinError };
}

export type { TgpuComputePipeline };
