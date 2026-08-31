// GPU butterfly FFT (I1) — replaces the water-surface spike's `dftRowKernel`/`dftColKernel` (one
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
// TGSL compiles plain `u32 / u32` to a REAL division (`gpu.md`'s promoted rule, already carried in
// `@dylanebert/shallot`'s `packages/shallot/AGENTS.md`: "TGSL integer division uses `idiv` from
// `utils/core`, never `/`"). This kernel derives a butterfly pair's (block, local-index) split from
// the thread id via exactly that division, so every division below goes through `idiv`; `%`
// (WGSL's native integer remainder) is unaffected and used directly, per the same rule.
//
// Direction: unnormalized inverse only (`sign = +1`, matching `fft.ts`'s `fft1dInPlace` and the
// spike's own `idft2` phase convention `Σ_k in[k]·exp(+i·2π·k·n/N)`, no `1/N`). Nothing in this
// package ever needs the forward direction — `generateH0` already produces its output in the
// frequency domain.

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
                const angle = (d.f32(2) * PI * d.f32(blockLocal)) / d.f32(len);
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
                const angle = (d.f32(2) * PI * d.f32(blockLocal)) / d.f32(len);
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

export type { TgpuComputePipeline };
