// The field-vs-field cross-instrument agreement arm: per cascade, the spectral Jacobian's fold
// fraction (`cpu-reference.ts`'s `jacobianStats`, an analytic-derivative instrument) must agree with
// central differences taken on the SAME bicubic reconstruction the vertex-stage displacement sampler
// uses (`reconstruction.ts`'s `bicubicSample`), sampled at RANDOM world points — never the mesh's own
// vertices, so this reads the field alone, independent of any mesh
// (`mesh-inversion-sweep.oracle.ts` is the companion instrument that reads the mesh).
//
// The tolerance is the sum of two independently-derived error sources, neither of which the subject
// (the imported, mutable `bicubicSample`) can move:
//
//   1. Reconstruction truncation error — `referenceTruncationError` drives a FROZEN, test-local
//      Catmull-Rom implementation (`refBicubicSample`/`refCatmullRom1D` below, written from the
//      textbook uniform Catmull-Rom (tau=0.5) formula, never importing `reconstruction.ts`) against a
//      SYNTHETIC single-mode field at each cascade's own `kHi` (the worst-case, highest-frequency mode
//      the declared band admits), at the SAME central-difference step the field comparison uses. This
//      number is fixed for a given cascade regardless of what `reconstruction.ts` ships — it can only
//      move if this file's own reference formula changes. (Earlier shape: `reconstructionRelativeError`
//      called the SAME mutable `kernel` this arm was grading, so a broken kernel widened its own
//      tolerance in lockstep with its own defect — the Gate law's named prohibition.)
//   2. Finite-sample error — `fdFold` is a `RANDOM_POINTS`-sample Monte Carlo estimate of a Bernoulli
//      proportion whose population value is `spectralFold` (`cpu.jacobian.foldFraction`: an analytic,
//      full-N² -grid quantity from `cpu-reference.ts`, independent of any reconstruction kernel).
//      `samplingRelativeError` is the standard 95% normal-approximation relative-error bound on that
//      estimator (the same 1.96 z-quantile convention `slope.ts`'s `slopeMomentAgreementTolerance`
//      uses — read that function before this one). `RANDOM_POINTS` is raised from the original 400 to
//      20000 so the tightest cascade's expected fold count (`spectralFold * RANDOM_POINTS`, ~1.1% ×
//      20000 ≈ 220 for cascade 1) clears a few hundred — at a few folds out of 400 the normal
//      approximation to a binomial proportion is not a reliable model, and a relative difference
//      between two near-zero fractions is a fragile statistic either way (both `spectralFold` and
//      `fdFold` are printed beside the reading so this is visible, not hidden behind a pass/fail).
//
// No authored margin constant: the old `TOLERANCE_MARGIN = 1.2` claimed "the same margin convention
// `slope.ts`'s agreement arms use" — `slope.ts` carries no such margin literal, so the citation was
// false. Deleted rather than replaced.
//
// The bound this derivation buys must actually discriminate the reconstruction kernel it was
// measured on from a materially worse one — `nearestSample` (no interpolation, C^-1) run through the
// exact same comparison at the exact same shipped `N` doubles as this arm's committed red witness:
// only the reconstruction kernel (the subject) is mutated, never the tolerance or the expectation.
//
// Mutation table (each applied in place at this stage's own ref, run, reverted with
// `git show HEAD:<path>`, never shipped):
//   - `texelUV` u/v transposed in `centralDiffDet`'s `sampleDx`/`sampleDz` → RED (all 4 tests fail;
//     the shipped `RANDOM_POINTS=400` let this reach byte-identical fold counts by coincidence at low
//     sample count — the 20000-point raise above fixes both this and the fragile-statistic concern).
//   - `reconstruction.ts`'s `catmullRom1D` `c` coefficient (`0.5*p2` → `0.35*p2`) → RED on both this
//     file (cascade 1: relDiff 65.48% > bound 48.99%) and `bun run test:ocean-mesh-inversion`; this
//     file's printed bound (48.99%) is IDENTICAL before and after, since it is derived from the
//     frozen `refCatmullRom1D`, never from the mutated import.
//   - `reconstruction.ts`'s `wrap` widened to `i % n` (drops the negative-index wraparound the -1 tap
//     needs) → RED (all 4 tests fail, several by thrown error on an out-of-range array read). A
//     narrower single-tap shift (`p3` at `ix+2` → `ix+3`) stayed within this file's own loose bound
//     (cascade 1 relDiff moved 14.88% → 6.22%, well under 48.99%) but reds
//     `mesh-inversion-sweep.oracle.ts`'s leg (b) red witness — this file's bound is dominated by the
//     fixed truncation-error term and is not the tighter instrument for every reconstruction defect;
//     the mesh-side ring 0 comparison is.
import { describe, expect, test } from "bun:test";
import { runCpuPipeline } from "../src/cpu-reference";
import {
    bicubicSample,
    type Field,
    nearestSample,
    type ReconstructionKernel,
} from "../src/reconstruction";
import { CASCADE_CONFIGS, generateH0 } from "../src/spectrum";

const H_STEP_FRAC = 1 / 8; // central-difference step, as a fraction of one texel — matches the
// scale a mesh's near-field ring samples the field at.
// RANDOM_POINTS: see this file's header — sized so the tightest cascade's expected fold count clears
// a few hundred, keeping the finite-sample normal approximation meaningful rather than a coin flip.
const RANDOM_POINTS = 20000;
const RANDOM_SEED = 0x5eed;
const Z95 = 1.96; // matches `slope.ts`'s own `slopeMomentAgreementTolerance` z-quantile convention.

function toField(flat: Float64Array, N: number): Field {
    const f: Field = [];
    for (let y = 0; y < N; y++) {
        const row: number[] = new Array(N);
        for (let x = 0; x < N; x++) row[x] = flat[y * N + x];
        f.push(row);
    }
    return f;
}

/** deterministic PRNG (mulberry32) — reproducible random world points, no external dependency. */
function mulberry32(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (s + 0x6d2b79f5) >>> 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function texelUV(world: number, L: number, N: number): number {
    return (world / L + 0.5) * N - 0.5;
}

function worldForTexel(x: number, N: number, L: number): number {
    return ((x + 0.5) / N - 0.5) * L;
}

/** central-difference Jacobian of the (dx, dz) displacement field, reconstructed via `kernel` at
 *  world point (x, z), step `h` in world metres. */
function centralDiffDet(
    kernel: ReconstructionKernel,
    dxField: Field,
    dzField: Field,
    N: number,
    L: number,
    x: number,
    z: number,
    h: number,
): number {
    const sampleDx = (wx: number, wz: number) =>
        kernel(dxField, N, texelUV(wx, L, N), texelUV(wz, L, N));
    const sampleDz = (wx: number, wz: number) =>
        kernel(dzField, N, texelUV(wx, L, N), texelUV(wz, L, N));
    const dDxdx = (sampleDx(x + h, z) - sampleDx(x - h, z)) / (2 * h);
    const dDxdz = (sampleDx(x, z + h) - sampleDx(x, z - h)) / (2 * h);
    const dDzdx = (sampleDz(x + h, z) - sampleDz(x - h, z)) / (2 * h);
    const dDzdz = (sampleDz(x, z + h) - sampleDz(x, z - h)) / (2 * h);
    const Jxx = 1 + dDxdx;
    const Jzz = 1 + dDzdz;
    const Jxz = (dDxdz + dDzdx) / 2;
    return Jxx * Jzz - Jxz * Jxz;
}

interface AgreementReading {
    spectralFold: number;
    fdFold: number;
    relDiff: number;
}

function agreementForCascade(
    kernel: ReconstructionKernel,
    cfgIndex: number,
    N: number,
    L: number,
    cpu: ReturnType<typeof runCpuPipeline>,
): AgreementReading {
    const dxField = toField(cpu.jacobian.dxRaw, N);
    const dzField = toField(cpu.jacobian.dzRaw, N);
    const texel = L / N;
    const h = texel * H_STEP_FRAC;

    const rand = mulberry32(RANDOM_SEED + cfgIndex);
    let foldCount = 0;
    for (let i = 0; i < RANDOM_POINTS; i++) {
        const x = (rand() - 0.5) * L;
        const z = (rand() - 0.5) * L;
        const det = centralDiffDet(kernel, dxField, dzField, N, L, x, z, h);
        if (det < 0) foldCount++;
    }
    const fdFold = foldCount / RANDOM_POINTS;
    const spectralFold = cpu.jacobian.foldFraction;
    const denom = Math.max(spectralFold, fdFold, 1e-6);
    const relDiff = Math.abs(spectralFold - fdFold) / denom;
    return { spectralFold, fdFold, relDiff };
}

// ── frozen, test-local reference kernel — the bound's inputs, never the subject ────────────────────
// Written from the textbook uniform Catmull-Rom (tau=0.5) formula, independently of
// `reconstruction.ts`. This is what makes `referenceTruncationError` immune to a defect in the
// SHIPPED `bicubicSample`/`catmullRom1D`: mutating the subject moves the READING (`agreementForCascade`
// above, driven by the imported `kernel` parameter) without moving this reference's own output.
function refWrap(i: number, n: number): number {
    return ((i % n) + n) % n;
}
function refCatmullRom1D(p0: number, p1: number, p2: number, p3: number, t: number): number {
    const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
    const c = -0.5 * p0 + 0.5 * p2;
    return p1 + t * (c + t * (b + t * a));
}
function refBicubicSample(field: Field, n: number, u: number, v: number): number {
    const ix = Math.floor(u);
    const iy = Math.floor(v);
    const fx = u - ix;
    const fy = v - iy;
    const rows: number[] = [];
    for (let j = -1; j <= 2; j++) {
        const yy = refWrap(iy + j, n);
        const p0 = field[yy][refWrap(ix - 1, n)];
        const p1 = field[yy][refWrap(ix, n)];
        const p2 = field[yy][refWrap(ix + 1, n)];
        const p3 = field[yy][refWrap(ix + 2, n)];
        rows.push(refCatmullRom1D(p0, p1, p2, p3, fx));
    }
    return refCatmullRom1D(rows[0], rows[1], rows[2], rows[3], fy);
}

/** samples the (row-invariant, x-only) synthetic field via the frozen reference kernel at a constant
 *  off-grid row offset — the row dimension is inert since `field[y][x]` is the same for every y. */
function refSampleAt(field: Field, N: number, world: number, L: number): number {
    const u = texelUV(world, L, N);
    return refBicubicSample(field, N, u, 3.0);
}

/** measures the FROZEN reference kernel's own central-difference derivative error, over a SYNTHETIC
 *  single-mode field at wavenumber `k` (the declared band's own `kHi`, the worst case), sampled at N
 *  texels over domain L — independent of any ocean field and, critically, independent of whatever
 *  `reconstruction.ts` ships (see this file's header). This is one half of the tolerance's derivation:
 *  the reconstruction kernel FORM's own truncation error at the SAME step `h` the field comparison
 *  uses, at the highest frequency the declared band admits. */
function referenceTruncationError(N: number, L: number, k: number): number {
    const texel = L / N;
    const h = texel * H_STEP_FRAC;
    const field: Field = [];
    for (let y = 0; y < N; y++) {
        const row: number[] = new Array(N);
        for (let x = 0; x < N; x++) {
            const world = worldForTexel(x, N, L);
            row[x] = Math.cos(k * world);
        }
        field.push(row);
    }
    const rand = mulberry32(0xbeef);
    let sumSq = 0;
    let sumAnalyticSq = 0;
    const samples = 64;
    for (let i = 0; i < samples; i++) {
        const world = (rand() - 0.5) * L;
        const estD =
            (refSampleAt(field, N, world + h, L) - refSampleAt(field, N, world - h, L)) / (2 * h);
        const analyticD = -k * Math.sin(k * world);
        sumSq += (estD - analyticD) ** 2;
        sumAnalyticSq += analyticD * analyticD;
    }
    return Math.sqrt(sumSq / samples) / Math.sqrt(Math.max(sumAnalyticSq / samples, 1e-9));
}

/** 95% normal-approximation relative-error bound on a `n`-sample Bernoulli-proportion estimator of
 *  population value `p` (`slope.ts`'s `slopeMomentAgreementTolerance` z-quantile convention, applied
 *  to a binomial rather than a Fourier-mode-weight estimator). `Math.max(p, 1e-6)` in the denominator
 *  matches `agreementForCascade`'s own `relDiff` denominator floor — when `p` is exactly 0 (cascade 0,
 *  the shipped config's own reading), `se` is also exactly 0 so this returns 0, never NaN. */
function samplingRelativeError(p: number, n: number): number {
    const clampedP = Math.max(0, Math.min(1, p));
    const se = Math.sqrt((clampedP * (1 - clampedP)) / n);
    return (Z95 * se) / Math.max(clampedP, 1e-6);
}

function deriveTolerance(
    cfg: (typeof CASCADE_CONFIGS)[number],
    spectralFold: number,
): {
    truncation: number;
    sampling: number;
    total: number;
} {
    const truncation = referenceTruncationError(cfg.N, cfg.L, cfg.kHi);
    const sampling = samplingRelativeError(spectralFold, RANDOM_POINTS);
    return { truncation, sampling, total: truncation + sampling };
}

describe("cross-instrument agreement — spectral Jacobian vs FD-on-reconstruction at random world points", () => {
    for (let i = 0; i < CASCADE_CONFIGS.length; i++) {
        const cfg = CASCADE_CONFIGS[i];

        test(`cascade ${i} (N=${cfg.N}, L=${cfg.L}m): spectral and FD-on-bicubic-reconstruction agree within the reconstruction-error-derived tolerance`, () => {
            const h0 = generateH0(cfg);
            const cpu = runCpuPipeline(h0, cfg, 0);
            const reading = agreementForCascade(bicubicSample, i, cfg.N, cfg.L, cpu);
            const bound = deriveTolerance(cfg, reading.spectralFold);
            console.log(
                `cascade ${i}: spectral=${(reading.spectralFold * 100).toFixed(3)}% FD(bicubic)=${(reading.fdFold * 100).toFixed(3)}% relDiff=${(reading.relDiff * 100).toFixed(2)}% ` +
                    `tolerance=${(bound.total * 100).toFixed(2)}% (truncation=${(bound.truncation * 100).toFixed(2)}% + sampling=${(bound.sampling * 100).toFixed(2)}% at n=${RANDOM_POINTS})`,
            );
            expect(reading.relDiff).toBeLessThanOrEqual(bound.total);
        });

        test(`cascade ${i} (N=${cfg.N}, L=${cfg.L}m): RED-WITNESS — nearest-texel sampling on the shipped N breaks the same tolerance`, () => {
            // re-runs the guarded arm's own comparison with only the subject (the reconstruction
            // kernel) mutated to `nearestSample`, at the SAME shipped N/L and the SAME
            // reference-kernel-derived tolerance — never an overridden N, never a separately authored
            // bound. This is also the discrimination proof the tolerance owes: a bound derived from
            // the frozen reference kernel's own truncation error must not also pass for a materially
            // worse kernel.
            const h0 = generateH0(cfg);
            const cpu = runCpuPipeline(h0, cfg, 0);
            const greenReading = agreementForCascade(bicubicSample, i, cfg.N, cfg.L, cpu);
            const mutatedReading = agreementForCascade(nearestSample, i, cfg.N, cfg.L, cpu);
            const bound = deriveTolerance(cfg, greenReading.spectralFold);
            console.log(
                `cascade ${i} red-witness (nearest): green relDiff=${(greenReading.relDiff * 100).toFixed(2)}%, ` +
                    `mutated relDiff=${(mutatedReading.relDiff * 100).toFixed(2)}%, bound=${(bound.total * 100).toFixed(2)}%, ` +
                    `reach=${(mutatedReading.relDiff / bound.total).toFixed(1)}x`,
            );
            expect(greenReading.relDiff).toBeLessThanOrEqual(bound.total);
            expect(mutatedReading.relDiff).toBeGreaterThan(bound.total);
        });
    }
});
