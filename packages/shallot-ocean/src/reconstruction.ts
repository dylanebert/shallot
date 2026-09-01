// Displacement-field reconstruction kernels — the sub-texel interpolation a mesh vertex (or a
// per-pixel shading reader) uses to read the displacement texture at an off-grid world position.
//
// A 4-tap bilinear kernel is only C0 continuous: its gradient jumps discontinuously at every texel
// boundary, which is visible on a displaced mesh as a lattice of hard creases rather than a smooth
// wave surface. Bicubic Catmull-Rom (16 wrapped taps, C1) removes that discontinuity — the
// gradient's two one-sided estimates at a texel boundary converge to the same value, which
// `maxGradientJump` below measures directly. `nearestSample` (no interpolation, C^-1) brackets
// bicubic from the opposite side bilinear does: no negative lobes to overshoot with.
//
// This module replicates both the GPU vertex-stage kernel and a plain-TS reference in exact
// lockstep (same taps, same wrapping, same Catmull-Rom coefficients) so a divergence between the
// two localizes to whichever side changed, rather than being explained away as "an equivalent
// formula".

/** a periodic NxN scalar field, `field[y][x]`, wrapping in both axes — stands in for one channel of
 *  a cascade's displacement texture. The reconstruction kernels below don't care what the data
 *  means, only that it's generic (non-collinear) enough that a C0 kernel's gradient jump is visible;
 *  degenerate data (e.g. all zero, or perfectly linear) would hide a real discontinuity. */
export type Field = number[][];

function wrap(i: number, n: number): number {
    return ((i % n) + n) % n;
}

/** a small deterministic PRNG (LCG) — reproducible test data, no external dependency. */
export function syntheticField(n: number, seed = 1): Field {
    let s = seed >>> 0 || 1;
    const rand = (): number => {
        s = (s * 1103515245 + 12345) >>> 0;
        return s / 0xffffffff;
    };
    const field: Field = [];
    for (let y = 0; y < n; y++) {
        const row: number[] = [];
        for (let x = 0; x < n; x++) row.push(rand() * 2 - 1);
        field.push(row);
    }
    return field;
}

/** 4-tap bilinear over wrapped corners, C0 only. */
export function bilinearSample(field: Field, n: number, u: number, v: number): number {
    const ix = Math.floor(u);
    const iy = Math.floor(v);
    const fx = u - ix;
    const fy = v - iy;
    const xa = wrap(ix, n);
    const ya = wrap(iy, n);
    const xb = wrap(ix + 1, n);
    const yb = wrap(iy + 1, n);
    const t00 = field[ya][xa];
    const t10 = field[ya][xb];
    const t01 = field[yb][xa];
    const t11 = field[yb][xb];
    return t00 * (1 - fx) * (1 - fy) + t10 * fx * (1 - fy) + t01 * (1 - fx) * fy + t11 * fx * fy;
}

/** one dimension of uniform Catmull-Rom (tau=0.5) — the exact coefficients the GPU vertex-stage
 *  kernel evaluates (matrix form expanded, Horner-evaluated). `p0..p3` are samples at local offsets
 *  -1,0,1,2; interpolates between `p1` and `p2` at `t ∈ [0,1]`. */
export function catmullRom1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const c = -0.5 * p0 + 0.5 * p2;
    return p1 + t * (c + t * (b + t * a));
}

/** the shipped kernel — 16-tap bicubic Catmull-Rom, C1. Mirrors the vertex-stage sampler: 4 rows of
 *  4 wrapped texel taps, `catmullRom1D` across each row's x, then across the 4 row results in y. */
export function bicubicSample(field: Field, n: number, u: number, v: number): number {
    const ix = Math.floor(u);
    const iy = Math.floor(v);
    const fx = u - ix;
    const fy = v - iy;
    const rows: number[] = [];
    for (let j = -1; j <= 2; j++) {
        const yy = wrap(iy + j, n);
        const p0 = field[yy][wrap(ix - 1, n)];
        const p1 = field[yy][wrap(ix, n)];
        const p2 = field[yy][wrap(ix + 1, n)];
        const p3 = field[yy][wrap(ix + 2, n)];
        rows.push(catmullRom1D(p0, p1, p2, p3, fx));
    }
    return catmullRom1D(rows[0], rows[1], rows[2], rows[3], fy);
}

/** raw texel-nearest sampling: no interpolation at all, C^-1 (piecewise-constant, a step at every
 *  texel boundary). Brackets `bicubicSample` from the opposite direction `bilinearSample` does: no
 *  negative lobes at all, so if bicubic's negative-lobe overshoot is what drives some defect,
 *  nearest should show less of it (or none), not more. */
export function nearestSample(field: Field, n: number, u: number, v: number): number {
    const ix = wrap(Math.round(u), n);
    const iy = wrap(Math.round(v), n);
    return field[iy][ix];
}

export type ReconstructionKernel = (field: Field, n: number, u: number, v: number) => number;

/**
 * The maximum |gradient jump| in `u` across any texel boundary, over every boundary `k` and a
 * spread of off-grid `v` offsets (never exactly on a boundary or the 0.5 midpoint — deliberately
 * asymmetric offsets so degenerate symmetric data can't hide a real jump). A central finite
 * difference of spacing `h` samples just inside and just outside each seam; a C1 kernel's two
 * one-sided derivative estimates converge to the same value as `h -> 0` (residual jump ~ O(h) from
 * truncation error, not a real discontinuity), while a C0 kernel's stay apart at any `h`.
 */
export function maxGradientJump(
    kernel: ReconstructionKernel,
    field: Field,
    n: number,
    h = 1e-4,
): number {
    let maxJump = 0;
    const vOffsets = [0.17, 0.44, 0.71]; // off-grid, off-midpoint row offsets
    for (let k = 0; k < n; k++) {
        for (const vFrac of vOffsets) {
            const v = 3 + vFrac; // arbitrary row, clear of the x boundary under test
            const gradLeft = (kernel(field, n, k - h, v) - kernel(field, n, k - 2 * h, v)) / h;
            const gradRight = (kernel(field, n, k + 2 * h, v) - kernel(field, n, k + h, v)) / h;
            const jump = Math.abs(gradRight - gradLeft);
            if (jump > maxJump) maxJump = jump;
        }
    }
    return maxJump;
}

export interface ReconstructionFinding {
    ok: boolean;
    message: string;
}

/** a reconstruction-continuity check: `kernel`'s worst-case gradient jump across a texel boundary,
 *  over a synthetic field of resolution `n`, must be ≤ `tolerance`. */
export function checkReconstructionContinuity(
    label: string,
    kernel: ReconstructionKernel,
    n: number,
    tolerance: number,
): ReconstructionFinding {
    const field = syntheticField(n);
    const jump = maxGradientJump(kernel, field, n);
    return {
        ok: jump <= tolerance,
        message: `${label} reconstruction: max gradient jump across a texel boundary = ${jump.toFixed(6)} — must be ≤ ${tolerance} (C1 tolerance) — ratio ${(jump / tolerance).toFixed(2)}x`,
    };
}
