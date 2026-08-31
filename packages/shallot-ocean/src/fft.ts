// CPU radix-2 Cooley-Tukey FFT — the butterfly transform used by the CPU reference and the
// GPU kernels in `gpu-fft.ts` both realize. `N` must be a power of two (`spectrum.ts`'s
// `assertAllPowerOfTwo`, checked at module load for every shipped cascade).
//
// The transform computed here is UNNORMALIZED and uses the inverse phase convention
// exactly: `out[n] = Σ_k in[k]·exp(sign·i·2π·k·n/N)`, no `1/N` factor (the amplitude convention is
// baked into `generateH0`, per that file's own docblock). `sign = +1` is verified against a direct
// O(N²) summation in
// `tests/fft.test.ts`); nothing here ever needs `sign = -1` (the forward direction) — `generateH0`
// already produces its output directly in the frequency domain, so no forward transform is needed
// anywhere in this package.

/** true iff `n` is a power of two — re-derived here rather than imported, so this module stays
 *  import-free (it is the leaf every CPU and GPU FFT path bottoms out at). */
function isPow2(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}

function bitReverse(i: number, bits: number): number {
    let r = 0;
    let v = i;
    for (let b = 0; b < bits; b++) {
        r = (r << 1) | (v & 1);
        v >>= 1;
    }
    return r;
}

/**
 * In-place radix-2 decimation-in-time FFT over interleaved-free `re`/`im` arrays of length `N`
 * (`N` a power of two). `sign = +1` is the unnormalized inverse direction `idft2` uses; `sign = -1`
 * would be the standard forward DFT (unused in this package, kept only because the butterfly stage
 * is identical either way — flipping the twiddle sign is the whole difference).
 */
export function fft1dInPlace(re: Float64Array, im: Float64Array, N: number, sign: 1 | -1): void {
    if (!isPow2(N)) throw new Error(`fft1dInPlace: N must be a power of two, got ${N}`);
    const bits = Math.log2(N);

    // bit-reversal permutation
    for (let i = 0; i < N; i++) {
        const j = bitReverse(i, bits);
        if (j > i) {
            const tr = re[i];
            re[i] = re[j];
            re[j] = tr;
            const ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }

    // iterative Cooley-Tukey butterfly stages
    for (let len = 2; len <= N; len <<= 1) {
        const half = len >> 1;
        const angleStep = (sign * 2 * Math.PI) / len;
        for (let start = 0; start < N; start += len) {
            for (let k = 0; k < half; k++) {
                const angle = angleStep * k;
                const wr = Math.cos(angle);
                const wi = Math.sin(angle);
                const evenI = start + k;
                const oddI = start + k + half;
                const evenR = re[evenI];
                const evenIm = im[evenI];
                const oddR = re[oddI];
                const oddIm = im[oddI];
                const tr = oddR * wr - oddIm * wi;
                const ti = oddR * wi + oddIm * wr;
                re[evenI] = evenR + tr;
                im[evenI] = evenIm + ti;
                re[oddI] = evenR - tr;
                im[oddI] = evenIm - ti;
            }
        }
    }
}

/**
 * Unnormalized 2D inverse FFT — a row pass (along x) then a column pass (along y), matching the
 * `idft2`'s signature and semantics exactly (interleaved re/im `Float32Array` in and out,
 * length `N*N*2`). Row-major layout: `input[(y*N + x)*2 + {0,1}]`.
 */
export function ifft2(input: Float32Array, N: number): Float32Array {
    const re = new Float64Array(N * N);
    const im = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) {
        re[i] = input[i * 2];
        im[i] = input[i * 2 + 1];
    }

    // row pass
    const rowRe = new Float64Array(N);
    const rowIm = new Float64Array(N);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            rowRe[x] = re[y * N + x];
            rowIm[x] = im[y * N + x];
        }
        fft1dInPlace(rowRe, rowIm, N, 1);
        for (let x = 0; x < N; x++) {
            re[y * N + x] = rowRe[x];
            im[y * N + x] = rowIm[x];
        }
    }

    // column pass
    const colRe = new Float64Array(N);
    const colIm = new Float64Array(N);
    for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
            colRe[y] = re[y * N + x];
            colIm[y] = im[y * N + x];
        }
        fft1dInPlace(colRe, colIm, N, 1);
        for (let y = 0; y < N; y++) {
            re[y * N + x] = colRe[y];
            im[y * N + x] = colIm[y];
        }
    }

    const out = new Float32Array(N * N * 2);
    for (let i = 0; i < N * N; i++) {
        out[i * 2] = re[i];
        out[i * 2 + 1] = im[i];
    }
    return out;
}

/** Direct O(N²) unnormalized inverse DFT — a reference implementation kept only for the
 *  brute-force reference `tests/fft.test.ts` checks `ifft2` against (never called from production). */
export function directIdft2(input: Float32Array, N: number): Float32Array {
    const mid = new Float64Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            let sumRe = 0;
            let sumIm = 0;
            for (let k = 0; k < N; k++) {
                const hr = input[(y * N + k) * 2];
                const hi = input[(y * N + k) * 2 + 1];
                const angle = (2 * Math.PI * x * k) / N;
                const c = Math.cos(angle);
                const s = Math.sin(angle);
                sumRe += hr * c - hi * s;
                sumIm += hr * s + hi * c;
            }
            mid[(y * N + x) * 2] = sumRe;
            mid[(y * N + x) * 2 + 1] = sumIm;
        }
    }
    const out = new Float64Array(N * N * 2);
    for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
            let sumRe = 0;
            let sumIm = 0;
            for (let k = 0; k < N; k++) {
                const hr = mid[(k * N + x) * 2];
                const hi = mid[(k * N + x) * 2 + 1];
                const angle = (2 * Math.PI * y * k) / N;
                const c = Math.cos(angle);
                const s = Math.sin(angle);
                sumRe += hr * c - hi * s;
                sumIm += hr * s + hi * c;
            }
            out[(y * N + x) * 2] = sumRe;
            out[(y * N + x) * 2 + 1] = sumIm;
        }
    }
    return Float32Array.from(out);
}
