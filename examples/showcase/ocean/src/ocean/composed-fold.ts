// The composed-world-grid fold statistic: what the shipped surface actually renders is the SUM of
// every displacement cascade's gradient at a shared world point, not a per-cascade reading pooled by
// texel count. `fold-anchor.oracle.ts` (CPU λ solve) and the GPU scenario `ocean-fold` (CPU-only
// composed print, no GPU composed sampler exists — see that scenario's own header) both import this
// module so the composition arithmetic is authored once.
//
// SAMPLING: no reconstruction kernel (bilinear/bicubic) samples either cascade here. A fold fraction
// is a tail statistic with no derivative-error bound (`field-mesh-agreement.test.ts`'s own measured
// finding: a 1% RMS slope-derivative attenuation produced a 12.5-point fold-fraction deficit), so a
// kernel's smoothing bias would land directly in this stage's GATE, not merely in a printed reading.
// Each cascade is read by NEAREST-NEIGHBOR (round + periodic wrap, `reconstruction.ts`'s
// `nearestSample` semantics on flat arrays) — the only reconstruction "kernel" that injects zero
// derivative smoothing, at the cost of a position quantization bounded by half that cascade's own
// texel, an order of magnitude below the physical scale (see `worldGridSpec` below).
//
// GRID: spacing is the FINEST cascade's own native texel (so every cascade's own resolution is at
// least met — "resolve cascade 1's texel"); extent is the COARSEST cascade's own period ("extent
// spans cascade 0's period") — the composed field is genuinely aperiodic at the joint (cascades'
// world lengths are coprime, `assertCoprimeL`), so one coarse-cascade period is a declared, stated
// truncation of the true (LCM) joint period, not a second period the field actually has.
import type { CascadeConfig } from "./spectrum";

/** One cascade's real-space gradient fields, unit-λ (the λ-independent half of `jacobianStats`) —
 *  `runCpuPipeline`'s own `gxxHeight`/`gxzHeight`/`gzzHeight` (real part only), so a caller already
 *  holding a `CpuStageResult` reads this directly off it rather than re-deriving. */
export interface CascadeGradientField {
    N: number;
    L: number;
    /** real part of ∂Dx/∂x, per texel, row-major, length N*N. */
    gxx: Float64Array;
    /** real part of ∂Dx/∂z (== ∂Dz/∂x), per texel, row-major, length N*N. */
    gxz: Float64Array;
    /** real part of ∂Dz/∂z, per texel, row-major, length N*N. */
    gzz: Float64Array;
}

/** Extracts the real part of a `ComplexArray` (interleaved re/im, `cpu-reference.ts`'s layout) into
 *  a flat `Float64Array`, the shape `composeWorldGrid` reads. */
export function realPart(complex: Float32Array, N: number): Float64Array {
    const out = new Float64Array(N * N);
    for (let i = 0; i < N * N; i++) out[i] = complex[i * 2];
    return out;
}

export interface WorldGridSpec {
    /** world-space spacing between adjacent grid samples, in metres — the finest cascade's own
     *  native texel (`min(L/N)` over the composed configs). */
    spacing: number;
    /** grid points per axis. */
    gridN: number;
    /** the declared world extent, `gridN * spacing` (near, not exactly, the coarsest cascade's own
     *  period — see this module's header). */
    extent: number;
}

/** Declares the shared composed-world-grid from the configs being composed: spacing resolves the
 *  finest cascade's texel, extent spans the coarsest cascade's period. Both are read off the
 *  `CascadeConfig`s themselves, never authored as a second, independent constant. */
export function worldGridSpec(configs: readonly CascadeConfig[]): WorldGridSpec {
    const spacing = Math.min(...configs.map((c) => c.L / c.N));
    const extent = Math.max(...configs.map((c) => c.L));
    const gridN = Math.round(extent / spacing);
    return { spacing, gridN, extent: gridN * spacing };
}

/** Nearest periodic texel index for world coordinate `world` on a cascade whose own native spacing
 *  is `texel` over `N` texels — the zero-derivative-smoothing read this module's header names. */
function nearestIndex(world: number, texel: number, N: number): number {
    return ((Math.round(world / texel) % N) + N) % N;
}

export interface ComposedField {
    gridN: number;
    /** unit-λ composed (summed across every cascade) gradient fields, row-major, length gridN². */
    gxx: Float64Array;
    gxz: Float64Array;
    gzz: Float64Array;
}

/** Superposes every cascade's unit-λ gradient field onto one shared world grid: at each grid point,
 *  each cascade contributes its own nearest-texel value (never interpolated — this module's header),
 *  summed. The result is unit-λ (no `lambda` applied), reusable across every trial λ a bisection
 *  evaluates without rebuilding the composition. */
export function composeWorldGrid(
    fields: readonly CascadeGradientField[],
    grid: WorldGridSpec,
): ComposedField {
    const { spacing, gridN, extent } = grid;
    const half = extent / 2;
    const gxx = new Float64Array(gridN * gridN);
    const gxz = new Float64Array(gridN * gridN);
    const gzz = new Float64Array(gridN * gridN);
    const texels = fields.map((f) => f.L / f.N);
    for (let gy = 0; gy < gridN; gy++) {
        const z = (gy + 0.5) * spacing - half;
        for (let gx = 0; gx < gridN; gx++) {
            const x = (gx + 0.5) * spacing - half;
            let sxx = 0;
            let sxz = 0;
            let szz = 0;
            for (let c = 0; c < fields.length; c++) {
                const f = fields[c];
                const ix = nearestIndex(x, texels[c], f.N);
                const iy = nearestIndex(z, texels[c], f.N);
                const idx = iy * f.N + ix;
                sxx += f.gxx[idx];
                sxz += f.gxz[idx];
                szz += f.gzz[idx];
            }
            const i = gy * gridN + gx;
            gxx[i] = sxx;
            gxz[i] = sxz;
            gzz[i] = szz;
        }
    }
    return { gridN, gxx, gxz, gzz };
}

/** Composed fold fraction at trial `lambda` — det(I + λ(G0+G1+…)) < 0, as a fraction of world-grid
 *  AREA (every cell the same size, so no per-cascade texel-count pooling weight enters here). */
export function foldFractionAt(composed: ComposedField, lambda: number): number {
    const n = composed.gridN * composed.gridN;
    let count = 0;
    for (let i = 0; i < n; i++) {
        const Jxx = 1 + lambda * composed.gxx[i];
        const Jzz = 1 + lambda * composed.gzz[i];
        const Jxz = lambda * composed.gxz[i];
        if (Jxx * Jzz - Jxz * Jxz < 0) count++;
    }
    return count / n;
}

/** RMS of the composed unit-λ Jacobian TRACE (gxx + gzz) over a composed field — the measured σ
 *  `fold-anchor.oracle.ts`'s erfc corroboration and `FOLD_REGIME.effectiveSlopeSigma` derive from,
 *  never an isotropic-Gaussian projection factor (deleted — see that oracle's own docblock). Callers
 *  average this across a seed×phase ensemble themselves; this reads one composed field only. */
export function traceRms(composed: ComposedField): number {
    const n = composed.gridN * composed.gridN;
    let sumSquares = 0;
    for (let i = 0; i < n; i++) {
        const trace = composed.gxx[i] + composed.gzz[i];
        sumSquares += trace * trace;
    }
    return Math.sqrt(sumSquares / n);
}
