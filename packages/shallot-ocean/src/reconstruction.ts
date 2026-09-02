// Displacement-field reconstruction kernels — the sub-texel interpolation a mesh vertex (or a
// per-pixel shading reader) uses to read the displacement texture at an off-grid world position.
//
// A 4-tap bilinear kernel is only C0 continuous: its gradient jumps discontinuously at every texel
// boundary, which is visible on a displaced mesh as a lattice of hard creases rather than a smooth
// wave surface. Bicubic Catmull-Rom (16 wrapped taps, C1) removes that discontinuity — its seam value
// and seam derivative on each side of a texel boundary agree exactly, which
// `reconstruction-kernel-claim.test.ts`'s seam-identity arm measures via closed-form in-cell Lagrange
// fits (the heir to this module's retired `maxGradientJump` order-of-convergence probe). `nearestSample`
// (no interpolation, C^-1) brackets bicubic from the opposite side bilinear does: no negative lobes to
// overshoot with.
//
// This module and `vertex-displacement.ts`'s GPU vertex-stage kernel express the same taps, wrapping
// and Catmull-Rom coefficients twice, hand-authored on each side; `wrap-catmullrom-lockstep.test.ts`
// drives both from the same inputs and asserts they agree, so a divergence between the two localizes
// to whichever side changed rather than being explained away as "an equivalent formula".

/** a periodic NxN scalar field, `field[y][x]`, wrapping in both axes — stands in for one channel of
 *  a cascade's displacement texture. The reconstruction kernels below don't care what the data
 *  means, only that it's generic (non-collinear) enough that a C0 kernel's gradient jump is visible;
 *  degenerate data (e.g. all zero, or perfectly linear) would hide a real discontinuity. */
export type Field = number[][];

/** wrap a possibly-negative texel index into `[0, n)`. Exported so
 *  `wrap-catmullrom-lockstep.test.ts` can drive it against `vertex-displacement.ts`'s `wrapIndex`. */
export function wrap(i: number, n: number): number {
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
