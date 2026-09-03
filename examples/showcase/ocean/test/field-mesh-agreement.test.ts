// Field-only fold-fraction reading — per cascade, prints the shipped kernel's finite-difference-on-
// reconstruction fold estimate (`fdFold`, sampled at RANDOM world points, never a mesh vertex) beside
// the spectral Jacobian's own analytic fold fraction (`cpu-reference.ts`'s `jacobianStats`, exact over
// the full N² grid) and `fdFold`'s 95% Bernoulli sampling interval. This reads the field alone,
// independent of any mesh (`mesh-inversion-sweep.oracle.ts` is the companion instrument that reads the
// mesh and asserts a mesh-only property).
//
// I3m re-verdict (2026-09-01): this file used to GATE on `relDiff` between these two readings against
// a tolerance built from a frozen, test-local Catmull-Rom transcription's own central-difference
// truncation error plus the sampling error. The consult ruled the comparison's *kind* wrong rather than
// its constant: a fold fraction is a tail statistic (a count of sign flips of `det J`, dominated by the
// density of `det J` near zero) and carries no derivative-error bound — measured 2026-09-01 at cascade
// 1, a 1% RMS slope-derivative attenuation produced a 12.5-point fold-fraction deficit, an order of
// magnitude larger effect than the derivative error that produced it. A reconstruction kernel's fold
// effect is a reading, never a gate (spec `Gate law` and Locked decision). The frozen reference
// transcription this file used to compare against (`refWrap`/`refCatmullRom1D`/`refBicubicSample`) is
// deleted rather than kept and re-sized: a "frozen reference" transcribed from the very kernel being
// read is a regression pin wearing a reference's name, not an independent check.
//
// No assertion below claims agreement. `spectralFold`, `fdFold`, `relDiff` and the sampling interval
// are printed per cascade so a reader can see the effect this spec's Locked decision measures, but
// nothing here gates on its size.
import { describe, test } from "bun:test";
import { runCpuPipeline } from "../src/ocean/cpu-reference";
import { bicubicSample, type Field } from "../src/ocean/reconstruction";
import { CASCADE_CONFIGS, generateH0 } from "../src/ocean/spectrum";

const H_STEP_FRAC = 1 / 8; // central-difference step, as a fraction of one texel — matches the
// scale a mesh's near-field ring samples the field at.
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

/** central-difference Jacobian of the (dx, dz) displacement field, reconstructed through the SHIPPED
 *  `bicubicSample` at world point (x, z), step `h` in world metres. */
function centralDiffDet(
    dxField: Field,
    dzField: Field,
    N: number,
    L: number,
    x: number,
    z: number,
    h: number,
): number {
    const sampleDx = (wx: number, wz: number) =>
        bicubicSample(dxField, N, texelUV(wx, L, N), texelUV(wz, L, N));
    const sampleDz = (wx: number, wz: number) =>
        bicubicSample(dzField, N, texelUV(wx, L, N), texelUV(wz, L, N));
    const dDxdx = (sampleDx(x + h, z) - sampleDx(x - h, z)) / (2 * h);
    const dDxdz = (sampleDx(x, z + h) - sampleDx(x, z - h)) / (2 * h);
    const dDzdx = (sampleDz(x + h, z) - sampleDz(x - h, z)) / (2 * h);
    const dDzdz = (sampleDz(x, z + h) - sampleDz(x, z - h)) / (2 * h);
    const Jxx = 1 + dDxdx;
    const Jzz = 1 + dDzdz;
    const Jxz = (dDxdz + dDzdx) / 2;
    return Jxx * Jzz - Jxz * Jxz;
}

interface FoldReading {
    spectralFold: number;
    fdFold: number;
    relDiff: number;
}

function foldReadingForCascade(
    cfgIndex: number,
    N: number,
    L: number,
    cpu: ReturnType<typeof runCpuPipeline>,
): FoldReading {
    const dxField = toField(cpu.jacobian.dxRaw, N);
    const dzField = toField(cpu.jacobian.dzRaw, N);
    const texel = L / N;
    const h = texel * H_STEP_FRAC;

    const rand = mulberry32(RANDOM_SEED + cfgIndex);
    let foldCount = 0;
    for (let i = 0; i < RANDOM_POINTS; i++) {
        const x = (rand() - 0.5) * L;
        const z = (rand() - 0.5) * L;
        const det = centralDiffDet(dxField, dzField, N, L, x, z, h);
        if (det < 0) foldCount++;
    }
    const fdFold = foldCount / RANDOM_POINTS;
    const spectralFold = cpu.jacobian.foldFraction;
    const denom = Math.max(spectralFold, fdFold, 1e-6);
    const relDiff = Math.abs(spectralFold - fdFold) / denom;
    return { spectralFold, fdFold, relDiff };
}

/** 95% normal-approximation relative-error bound on an `n`-sample Bernoulli-proportion estimator of
 *  population value `p` (`slope.ts`'s `slopeMomentAgreementTolerance` z-quantile convention, applied to
 *  a binomial rather than a Fourier-mode-weight estimator). `Math.max(p, 1e-6)` matches
 *  `foldReadingForCascade`'s own `relDiff` denominator floor — when `p` is exactly 0, `se` is also
 *  exactly 0 so this returns 0, never NaN. */
function samplingRelativeError(p: number, n: number): number {
    const clampedP = Math.max(0, Math.min(1, p));
    const se = Math.sqrt((clampedP * (1 - clampedP)) / n);
    return (Z95 * se) / Math.max(clampedP, 1e-6);
}

describe("field-only fold-fraction reading — printed per cascade, never gated (I3m re-verdict)", () => {
    for (let i = 0; i < CASCADE_CONFIGS.length; i++) {
        const cfg = CASCADE_CONFIGS[i];

        test(`cascade ${i} (N=${cfg.N}, L=${cfg.L}m): spectral fold vs FD-on-shipped-reconstruction fold, printed with its Bernoulli interval`, () => {
            const h0 = generateH0(cfg);
            const cpu = runCpuPipeline(h0, cfg, 0);
            const reading = foldReadingForCascade(i, cfg.N, cfg.L, cpu);
            const interval = samplingRelativeError(reading.fdFold, RANDOM_POINTS);
            console.log(
                `cascade ${i}: spectral=${(reading.spectralFold * 100).toFixed(3)}% FD(bicubic)=${(reading.fdFold * 100).toFixed(3)}% ` +
                    `relDiff=${(reading.relDiff * 100).toFixed(2)}% fdFold 95% interval=±${(interval * 100).toFixed(2)}% (n=${RANDOM_POINTS}) — reading only, not gated`,
            );
        });
    }
});
