// The exact storage-seam pyramid claim (I3g-r2). Two independent numeric claims, and the whole
// point of the split is that only ONE of them carries a tolerance:
//
//  - STORAGE SEAM (discrete, no tolerance anywhere): at every level and channel, the published
//    rgba16float texel is one of the f16-representable values immediately bracketing the f32
//    value it was rounded from. This is WGSL's own numeric-conversion guarantee, not an
//    engineering approximation: "When a value is converted to a defined floating-point
//    representation, finite values falling between two representable finite values are rounded
//    to one or the other" (WGSL specification, "Floating Point Conversion",
//    https://www.w3.org/TR/WGSL/) — the CHOICE between the two neighbours is left
//    implementation-defined, never a third value. `f16Neighbors` below returns exactly that pair.
//  - COMPUTATION (the one place a real tolerance sits): the level-0 real-space field the GPU's own
//    f32 inverse FFT computes, against an f64 recompute of the identical transform on the
//    identical input — bounded by Higham's Theorem 24.2, transcribed in `higham242RelativeBound`'s
//    own docblock, with every input to that bound MEASURED (never authored): `u` is IEEE754
//    single-precision unit roundoff, `muMax` is this device's own exhaustively-measured twiddle
//    trig error (`gpu-fft.ts`'s `measureTwiddleTrigError`), and `normX64` is the reference field's
//    own L2 norm.
//
// No seed set, margin, partition, or per-channel scale lands anywhere in this file (I3g-r2's own
// Approach bullet) — the storage-seam claim needs none of those (it is a discrete membership test
// against the SAME texel's own f32 source, never a scaled tolerance), and the computation claim's
// only free inputs are the three measured/derived quantities above.
//
// Mutation table (device kernels), re-run at the I3g-r2 review-round-1 correction against the
// LANDED I3g-r2 commits (`bun bench --scenario ocean-slope` against a real WebGPU device,
// nvidia/lovelace). Every mutation below targets `slope.ts`/`gpu-fft.ts` — files this correction
// round does not otherwise edit — so each reverts cleanly with `git show HEAD:<path>`:
//
//   1. `slope.ts`'s `mipKernel`, mean weight 0.25 -> 0.2 (wrong divisor, same 4 texels averaged):
//      exit 1. "storage seam claim holds over every texel of every level and channel" failed —
//      total 349524, violations=87322 (every channel of every level >=1 moved >=2 steps).
//   2. `slope.ts`'s `mipKernel`, dropped texel (`std.add(std.add(a, b), c)`, `e` never added,
//      divisor left at 0.25): exit 1. Same check failed — violations=87237.
//   3. `slope.ts`'s `slopePostKernel`, swapped channel (`vec4f(valueZ.x, valueX.x, energy, 0)`):
//      exit 1. "mip 0 storage seam histogram" failed — violations=131020 (level 0 only: L>=1
//      compares against its own already-swapped published parent, so the self-referential claim
//      cannot see a defect that landed consistently across the whole chain at level 0 — the
//      independent live-buffer comparison at level 0 is what catches it).
//   4. `gpu-fft.ts`'s `makeRowFftKernel`, dropped stage (butterfly loop starts at `len = 4` instead
//      of `2`, skipping the first stage): exit 1. "level-0 computation claim holds on slopeX" AND
//      "...slopeZ" failed — deviation L2 ~2.46e1 / ~2.38e1 against E0 ~1.6e-4 (reach 0.0x). The
//      storage-seam claim stayed green (self-referential — it does not care whether the FFT is
//      correct, only whether storage rounded consistently), which is exactly why the computation
//      claim exists as an independent check.
//   5. `slope.ts`'s `slopeKernel`, sign flip on slopeX only (`vec2f(kx * h.y, -kx * h.x)`): exit 1.
//      "level-0 computation claim holds on slopeX" failed — deviation L2 ~4.02e1 against E0
//      ~1.6e-4 (reach 0.0x); slopeZ's claim and the storage-seam claim both stayed green, showing
//      the computation claim discriminates per-channel rather than being fooled by one good side.
//
// I3g-r2's re-verdict retired the sixth row this table used to carry: `slope-seam.test.ts` no
// longer builds a "published" CPU pyramid by running level>=1 through `reduceSlopeMip` and
// comparing it against `expectedFromPublished` on the same inputs — that was one derivation
// (`reduceSlopeMip` is itself a transcription of the mean/residual formula below, with
// `Math.fround` discipline inserted) checked against another restatement of itself, not an
// independent reference, and its "seam claim" title was false — `reduceSlopeMip` is I3c's own
// production function (mip-residual variance for shading), not part of this file's seam math, and
// its correctness is covered by `slope.test.ts`'s literal-fixture arm. `expectedLevel0` and
// `expectedFromPublished`'s OWN correctness — that they compute the formula their docblocks below
// claim — is mutation-proven directly, against literals no other package export can move: see the
// mutation table in `slope-seam.test.ts`'s own header (`bun test
// packages/shallot-ocean/tests/slope-seam.test.ts`).

export const CHANNELS = ["slopeX", "slopeZ", "energy", "residual"] as const;
export type Channel = (typeof CHANNELS)[number];

function f16BitsOf(value: number): number {
    return new Uint16Array(new Float16Array([value]).buffer)[0];
}

function f16FromRawBits(bits: number): number {
    return Float32Array.from(new Float16Array(new Uint16Array([bits & 0xffff]).buffer))[0];
}

/** The f16-representable value nearest `value` (round-to-nearest-ties-to-even — `Float16Array`'s
 *  own, well-defined JS-spec conversion; WGSL's own device-side conversion is only guaranteed to
 *  land on one of the two neighbours this nearest value anchors, never necessarily THIS one). */
export function f16Round(value: number): number {
    return f16FromRawBits(f16BitsOf(value));
}

/** The next f16-representable value strictly above `v` (`v` must already be f16-representable). */
export function f16NextUp(v: number): number {
    if (v === 0) return f16FromRawBits(1);
    const bits = f16BitsOf(v);
    const sign = bits & 0x8000;
    const magnitude = bits & 0x7fff;
    if (sign === 0) return f16FromRawBits(magnitude + 1);
    return magnitude === 0 ? 0 : f16FromRawBits(sign | (magnitude - 1));
}

/** The next f16-representable value strictly below `v` (`v` must already be f16-representable). */
export function f16NextDown(v: number): number {
    if (v === 0) return f16FromRawBits(0x8001);
    const bits = f16BitsOf(v);
    const sign = bits & 0x8000;
    const magnitude = bits & 0x7fff;
    if (sign !== 0) return f16FromRawBits(sign | (magnitude + 1));
    return magnitude === 0 ? f16FromRawBits(0x8001) : f16FromRawBits(magnitude - 1);
}

/** The two f16-representable values bracketing `value` — `[value, value]` when `value` is already
 *  f16-exact (both "neighbours" coincide, matching `residual`'s architectural exact zero). */
export function f16Neighbors(value: number): [number, number] {
    const nearest = f16Round(value);
    if (nearest === value) return [nearest, nearest];
    return value > nearest ? [nearest, f16NextUp(nearest)] : [f16NextDown(nearest), nearest];
}

/** Signed f16-ULP hop count from `published`'s nearest-rounding of `expected` — 0 means an exact
 *  round-to-nearest match, 1 means the OTHER of the two WGSL-legal neighbours, and anything else is
 *  a genuine violation of the storage-seam claim (no slack term sits in front of this comparison —
 *  see the I3g-r re-verdict this file's header cites for why an FMA-fusion slack was deleted here
 *  rather than sized: the consult's own bench reading found every texel-channel on this device at
 *  `steps <= 1`, so an authored allowance the reading proves inert is deleted, not justified).
 *  Same-sign texels only (every channel here is far from a sign change at the f16 subnormal floor
 *  in practice; a sign flip reports a large sentinel distance rather than a false small one). */
export function f16StepDistance(expected: number, published: number): number {
    const nearest = f16Round(expected);
    if (nearest === published) return 0;
    const nb = f16BitsOf(nearest);
    const pb = f16BitsOf(published);
    if ((nb & 0x8000) !== (pb & 0x8000) && nearest !== 0 && published !== 0) {
        return 1000; // not a legitimate rounding neighbour under any realistic reading
    }
    return Math.abs((nb & 0x7fff) - (pb & 0x7fff));
}

/**
 * Level-0 "value it was rounded from", replicating `slopePostKernel`'s own f32 expression order
 * exactly (`Math.fround` at every intermediate step, never JS's default f64 arithmetic) from the
 * GPU's own real-space FFT output buffers — `xReal`/`zReal` are the `.x` (real) component of each
 * complex texel `state.x`/`state.z` holds after both FFT passes complete, read back once via
 * `getSlopeBuffers`. Residual is the kernel's own hardcoded literal 0.
 */
export function expectedLevel0(xReal: Float32Array, zReal: Float32Array): Float32Array {
    const n = xReal.length;
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        const sx = xReal[i];
        const sz = zReal[i];
        const energy = Math.fround(Math.fround(sx * sx) + Math.fround(sz * sz));
        out[i * 4] = sx;
        out[i * 4 + 1] = sz;
        out[i * 4 + 2] = energy;
        out[i * 4 + 3] = 0;
    }
    return out;
}

/**
 * Level L>=1 "value it was rounded from", replicating `mipKernel`'s own f32 expression order and
 * grouping exactly (`mean = mul(add(add(a,b),add(c,e)), vec4f(0.25))`, then
 * `residual = max(0, secondMoment - mean.x*mean.x - mean.y*mean.y)`) from the PUBLISHED
 * (already f16-quantized, then widened back to f32) parent level — never the CPU reference, since
 * the seam claim is self-referential through the GPU's own publication chain: every level's
 * "expected" value is what the GPU's own mip kernel would compute from what the GPU itself
 * already published one level up.
 */
export function expectedFromPublished(parent: Float32Array, size: number): Float32Array {
    const nextSize = size / 2;
    const out = new Float32Array(nextSize * nextSize * 4);
    const meanOf = (a: number, b: number, c: number, e: number) =>
        Math.fround(Math.fround(Math.fround(a + b) + Math.fround(c + e)) * 0.25);
    for (let y = 0; y < nextSize; y++) {
        for (let x = 0; x < nextSize; x++) {
            const offsets = [
                (2 * y * size + 2 * x) * 4,
                (2 * y * size + 2 * x + 1) * 4,
                ((2 * y + 1) * size + 2 * x) * 4,
                ((2 * y + 1) * size + 2 * x + 1) * 4,
            ];
            const meanX = meanOf(
                parent[offsets[0]],
                parent[offsets[1]],
                parent[offsets[2]],
                parent[offsets[3]],
            );
            const meanZ = meanOf(
                parent[offsets[0] + 1],
                parent[offsets[1] + 1],
                parent[offsets[2] + 1],
                parent[offsets[3] + 1],
            );
            const secondMoment = meanOf(
                parent[offsets[0] + 2],
                parent[offsets[1] + 2],
                parent[offsets[2] + 2],
                parent[offsets[3] + 2],
            );
            const t1 = Math.fround(secondMoment - Math.fround(meanX * meanX));
            const t2 = Math.fround(t1 - Math.fround(meanZ * meanZ));
            const residual = Math.max(0, t2);
            const index = (y * nextSize + x) * 4;
            out[index] = meanX;
            out[index + 1] = meanZ;
            out[index + 2] = secondMoment;
            out[index + 3] = residual;
        }
    }
    return out;
}

/**
 * Higham's Theorem 24.2 (N.J. Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.,
 * SIAM 2002, Chapter 24 "The Fast Fourier Transform and Applications", §24.1, pp. 452-454;
 * primary-source verified against the book's own text): for `y = F_n x` computed by the radix-2
 * Cooley-Tukey FFT with `n = 2^t` and twiddle factors `w_hat` satisfying `|w_hat - w| <= mu`, the
 * computed result `y_hat` satisfies
 *
 *     ||y - y_hat||_2 / ||y||_2  <=  t*eta / (1 - t*eta),      eta := mu + gamma_4*(sqrt(2) + mu)
 *
 * where `gamma_4 = 4u/(1 - 4u)` is Higham's standard rounding-error constant for four chained
 * floating-point operations and `u` is the unit roundoff (`2^-24` for f32 — matching the theorem's
 * own convention). This module applies the bound with `t = stages = 2*log2(N)`: the ROW pass and
 * the COLUMN pass of the package's separable 2D transform, each independently a length-N 1D FFT
 * contributing `log2(N)` butterfly stages. Folding both dimensions into one theorem application
 * (rather than composing two half-size applications and adding their relative errors) is the
 * CONSERVATIVE direction, never the tight one: `x/(1-x)` is convex on `[0,1)`, so
 * `f(2s) >= 2*f(s)` for `s = log2(N)*eta > 0` — one application at double the stage count bounds
 * at least as much error as two applications at the true stage count summed.
 */
export function higham242RelativeBound(muMax: number, u: number, stages: number): number {
    const gamma4 = (4 * u) / (1 - 4 * u);
    const eta = muMax + gamma4 * (Math.SQRT2 + muMax);
    const product = stages * eta;
    if (product >= 1) return Number.POSITIVE_INFINITY;
    return product / (1 - product);
}

/** `E0 = higham242RelativeBound(...) * normX64` — the theorem is stated relative to `||y||_2`,
 *  which IS `||x64||_2` here: `ifft2Exact`'s (`fft.ts`) own f64 radix-2 FFT output, computed at
 *  full f64 precision (`fft.ts`'s own docblock wording) rather than the theorem's own EXACT
 *  transform, which `ifft2Exact` is not — it carries its own rounding error at `u ~ 2^-53` per
 *  chained operation, roughly nine orders of magnitude below E0's own f32-driven `u = 2^-24` scale,
 *  negligible against E0 rather than zero. */
export function higham242AbsoluteBound(
    muMax: number,
    u: number,
    stages: number,
    normX64: number,
): number {
    return higham242RelativeBound(muMax, u, stages) * normX64;
}

/** L2 norm of an interleaved re/im `Float64Array` pair (both halves of one complex field). */
export function complexL2Norm(re: Float64Array, im: Float64Array): number {
    let sumSquares = 0;
    for (let i = 0; i < re.length; i++) sumSquares += re[i] * re[i] + im[i] * im[i];
    return Math.sqrt(sumSquares);
}
