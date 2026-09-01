// The field-vs-field cross-instrument agreement arm: per cascade, the spectral Jacobian's fold
// fraction (`cpu-reference.ts`'s `jacobianStats`, an analytic-derivative instrument) must agree with
// central differences taken on the SAME bicubic reconstruction the vertex-stage displacement sampler
// uses (`reconstruction.ts`'s `bicubicSample`), sampled at RANDOM world points — never the mesh's own
// vertices, so this reads the field alone, independent of any mesh
// (`mesh-inversion-sweep.oracle.ts` is the companion instrument that reads the mesh).
//
// The tolerance is not authored: it is measured directly, by driving `bicubicSample` +
// central-difference against a SYNTHETIC single-mode field at each cascade's own `kHi` (the
// worst-case, highest-frequency mode the declared band admits) and reading the derivative estimate's
// relative error against the mode's own analytic derivative, at the SAME central-difference step
// this file's field comparison uses. `TOLERANCE_MARGIN` is a stated headroom, not a fit (the same
// margin convention `slope.ts`'s agreement arms use).
//
// The bound this derivation buys must actually discriminate the reconstruction kernel it was
// measured on from a materially worse one — `nearestSample` (no interpolation, C^-1) run through the
// exact same comparison at the exact same shipped `N` doubles as this arm's committed red witness:
// only the reconstruction kernel (the subject) is mutated, never the tolerance or the expectation.
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
const TOLERANCE_MARGIN = 1.2; // stated headroom, not a fit (`slope.ts`'s own margin convention).
const RANDOM_POINTS = 400;
const RANDOM_SEED = 0x5eed;

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

/** measures `kernel`'s own central-difference derivative error, over a SYNTHETIC single-mode field
 *  at wavenumber `k` (the declared band's own `kHi`, the worst case), sampled at N texels over
 *  domain L — independent of any ocean field. This is the tolerance's derivation: the
 *  reconstruction's own truncation error at the SAME step `h` the field comparison uses, at the
 *  highest frequency the declared band admits, is a property of the kernel and the step — never of
 *  the reading it is about to bound. */
function reconstructionRelativeError(
    kernel: ReconstructionKernel,
    N: number,
    L: number,
    k: number,
): number {
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
            (sampleAt(kernel, field, N, world + h, L) - sampleAt(kernel, field, N, world - h, L)) /
            (2 * h);
        const analyticD = -k * Math.sin(k * world);
        sumSq += (estD - analyticD) ** 2;
        sumAnalyticSq += analyticD * analyticD;
    }
    return Math.sqrt(sumSq / samples) / Math.sqrt(Math.max(sumAnalyticSq / samples, 1e-9));
}

/** samples the (row-invariant, x-only) synthetic field via `kernel` at a constant off-grid row
 *  offset — the row dimension is inert since `field[y][x]` is the same for every y. */
function sampleAt(
    kernel: ReconstructionKernel,
    field: Field,
    N: number,
    world: number,
    L: number,
): number {
    const u = texelUV(world, L, N);
    return kernel(field, N, u, 3.0);
}

describe("cross-instrument agreement — spectral Jacobian vs FD-on-reconstruction at random world points", () => {
    for (let i = 0; i < CASCADE_CONFIGS.length; i++) {
        const cfg = CASCADE_CONFIGS[i];

        test(`cascade ${i} (N=${cfg.N}, L=${cfg.L}m): spectral and FD-on-bicubic-reconstruction agree within the reconstruction-error-derived tolerance`, () => {
            const h0 = generateH0(cfg);
            const cpu = runCpuPipeline(h0, cfg, 0);
            const reading = agreementForCascade(bicubicSample, i, cfg.N, cfg.L, cpu);
            const tolerance =
                reconstructionRelativeError(bicubicSample, cfg.N, cfg.L, cfg.kHi) *
                TOLERANCE_MARGIN;
            console.log(
                `cascade ${i}: spectral=${(reading.spectralFold * 100).toFixed(3)}% FD(bicubic)=${(reading.fdFold * 100).toFixed(3)}% relDiff=${(reading.relDiff * 100).toFixed(2)}% tolerance=${(tolerance * 100).toFixed(2)}%`,
            );
            expect(reading.relDiff).toBeLessThanOrEqual(tolerance);
        });

        test(`cascade ${i} (N=${cfg.N}, L=${cfg.L}m): RED-WITNESS — nearest-texel sampling on the shipped N breaks the same tolerance`, () => {
            // re-runs the guarded arm's own comparison with only the subject (the reconstruction
            // kernel) mutated to `nearestSample`, at the SAME shipped N/L and the SAME
            // bicubic-derived tolerance — never an overridden N, never a separately authored bound.
            // This is also the discrimination proof the tolerance owes: a bound derived from
            // bicubic's own truncation error must not also pass for a materially worse kernel.
            const h0 = generateH0(cfg);
            const cpu = runCpuPipeline(h0, cfg, 0);
            const greenReading = agreementForCascade(bicubicSample, i, cfg.N, cfg.L, cpu);
            const mutatedReading = agreementForCascade(nearestSample, i, cfg.N, cfg.L, cpu);
            const tolerance =
                reconstructionRelativeError(bicubicSample, cfg.N, cfg.L, cfg.kHi) *
                TOLERANCE_MARGIN;
            console.log(
                `cascade ${i} red-witness (nearest): green relDiff=${(greenReading.relDiff * 100).toFixed(2)}%, ` +
                    `mutated relDiff=${(mutatedReading.relDiff * 100).toFixed(2)}%, bound=${(tolerance * 100).toFixed(2)}%, ` +
                    `reach=${(mutatedReading.relDiff / tolerance).toFixed(1)}x`,
            );
            expect(greenReading.relDiff).toBeLessThanOrEqual(tolerance);
            expect(mutatedReading.relDiff).toBeGreaterThan(tolerance);
        });
    }
});
