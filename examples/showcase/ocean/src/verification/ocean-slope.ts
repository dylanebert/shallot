// ocean-slope — the exact storage-seam pyramid claim, replacing a prior tolerance-based
// mip-agreement arm after two correction rounds convicted every version of that tolerance as
// either a permitted-floor term that cannot discriminate, or a term sized to the very readings it
// was meant to gate. The ruling: the comparison sits on a representation seam (the published value
// IS a rounding of the f32 value it came from), so the claim is DISCRETE — no tolerance belongs
// there at all — and the one place a real numeric tolerance is warranted is one layer up, at the
// FFT's own floating-point arithmetic. Two independent claims, never one number standing in for
// both (`examples/showcase/ocean/src/ocean/slope-seam.ts` owns the math for both):
//
//  - STORAGE SEAM (every level, every channel, every texel — no tolerance): the published
//    rgba16float texel is one of the f16-representable values immediately bracketing the f32
//    value it was rounded from (WGSL's own numeric-conversion clause). Level 0 compares against
//    the GPU's OWN post-inverse-FFT f32 buffers (`getSlopeBuffers`, read back once via COPY_SRC —
//    never a second GPU state built from the same kernels); level L>=1 compares against the mean
//    of the four PUBLISHED level-(L-1) texels — self-referential through the GPU's own publication
//    chain, never the CPU reference.
//  - COMPUTATION (level 0's slopeX/slopeZ ONLY — the one place a tolerance sits): the GPU's own
//    f32 inverse-FFT output against an f64 recompute of the identical transform on the identical
//    spectral input, bounded by Higham's Theorem 24.2 (`slope-seam.ts`'s own docblock transcribes
//    it) with every input MEASURED: `u` is IEEE754 single-precision unit roundoff, `muMax` is this
//    device's own exhaustively-measured twiddle-trig error (`measureTwiddleTrigError`, every
//    `(len, blockLocal)` pair the FFT kernels evaluate), and `||x64||_2` is the f64 reference
//    field's own norm. No seed set, margin, partition, or per-channel scale lands anywhere here.
//
// `bun bench` itself already skips loudly (`scripts/bench.ts`'s `skipReason()`) when the seat has
// no real GPU adapter, before any scenario boots — this file adds no second adapter check.

import { Compute, probeBuffer, probeTexture, type State } from "@dylanebert/shallot";
import {
    CHANNELS,
    type Channel,
    complexL2Norm,
    expectedFromPublished,
    expectedLevel0,
    f16StepDistance,
    generateH0,
    getCascadeConfigs,
    getSlopeBuffers,
    higham242AbsoluteBound,
    ifft2Exact,
    measureSlopePhaseTrigError,
    measureTwiddleTrigError,
    OceanPlugin,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeCompute,
    slopeMipSize,
} from "../ocean/index";

interface Check {
    name: string;
    pass: boolean;
    detail?: string;
}

const [SLOPE_CONFIG] = SLOPE_CASCADE_CONFIGS;
const N = SLOPE_CONFIG.N;

/** IEEE754 single-precision unit roundoff — Higham's own `u` (2^-24, matching the theorem's own
 *  convention for f32, `slope-seam.ts`'s `higham242RelativeBound`). */
const F32_UNIT_ROUNDOFF = 2 ** -24;

/** `ifft2` is separable — a row pass then a column pass, each a full radix-2 Cooley-Tukey
 *  transform of length N (`log2(N)` butterfly stages) — so Higham's theorem is applied here with
 *  the TOTAL stage count across both passes (`slope-seam.ts`'s own docblock justifies why this is
 *  the conservative, not the tight, choice). */
const FFT_STAGES = 2 * Math.log2(N);

/** Extract the real component of an interleaved complex `Float32Array` buffer readback. */
function realComponent(bytes: ArrayBuffer, count: number): Float32Array {
    const raw = new Float32Array(bytes);
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = raw[i * 2];
    return out;
}

function phaseInputPerturbationNorms(
    h0: Float32Array,
    maxTrigError: number,
): { x: number; z: number } {
    const { N, L } = SLOPE_CONFIG;
    const dk = (2 * Math.PI) / L;
    let xSquares = 0;
    let zSquares = 0;
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const index = y * N + x;
            const neg = ((N - y) % N) * N + ((N - x) % N);
            const amplitude =
                Math.abs(h0[index * 2]) +
                Math.abs(h0[index * 2 + 1]) +
                Math.abs(h0[neg * 2]) +
                Math.abs(h0[neg * 2 + 1]);
            // Each complex spectrum component has two sums of four products. Bounding every
            // measured trig-lane perturbation by maxTrigError gives sqrt(2) times this amplitude;
            // the slope input then scales it by |k|. The normalized inverse FFT cannot increase
            // this input L2 norm, so it is a conservative one-shot output perturbation term.
            const modeError = Math.SQRT2 * amplitude * maxTrigError;
            const kx = (x <= N / 2 ? x : x - N) * dk;
            const kz = (y <= N / 2 ? y : y - N) * dk;
            xSquares += (kx * modeError) ** 2;
            zSquares += (kz * modeError) ** 2;
        }
    }
    return { x: Math.sqrt(xSquares), z: Math.sqrt(zSquares) };
}

interface ComplexDeviation {
    /** L2 norm over the whole complex field — what Higham's theorem bounds and what E0 gates. */
    l2: number;
    /** largest single-texel complex-magnitude deviation. */
    max: number;
    /** RMS single-texel complex-magnitude deviation. */
    rms: number;
}

/** Deviation between a GPU complex buffer readback and an f64 complex reference — the whole
 *  population Higham's theorem bounds (both real AND imaginary components: the theorem bounds the
 *  full transformed vector, and the imaginary component of a real-valued field's inverse FFT is
 *  analytically zero, so its magnitude here is purely the FFT's own numerical error, part of what
 *  the bound must cover). `l2` is the gated quantity; `max`/`rms` are the printed reading. */
function complexDeviation(
    bytes: ArrayBuffer,
    re64: Float64Array,
    im64: Float64Array,
): ComplexDeviation {
    const raw = new Float32Array(bytes);
    let sumSquares = 0;
    let max = 0;
    for (let i = 0; i < re64.length; i++) {
        const dr = raw[i * 2] - re64[i];
        const di = raw[i * 2 + 1] - im64[i];
        const magnitude = Math.hypot(dr, di);
        if (magnitude > max) max = magnitude;
        sumSquares += dr * dr + di * di;
    }
    return { l2: Math.sqrt(sumSquares), max, rms: Math.sqrt(sumSquares / re64.length) };
}

/** Widen one mip level's published rgba16float texture content to f32. */
async function publishedLevel(texture: GPUTexture, level: number): Promise<Float32Array> {
    const probe = await probeTexture(Compute.device, texture, { mipLevel: level });
    return Float32Array.from(new Float16Array(probe.bytes));
}

function allowedSteps(level: number, channel: Channel): number {
    if (level === 0 && channel === "residual") return 0; // hardcoded literal, no rounding ambiguity
    return 1; // bare discrete claim — no slack term (I3g-r re-verdict; `slope-seam.ts`'s own header)
}

interface SeamHistogram {
    exact: number;
    neighbor: number;
    slack: number;
    violations: string[];
}

/** Per-channel maximum observed step distance at one level — the durable, attributable replacement
 *  for the deleted scalar `seamSlack`: this device's own reading printed alongside the histogram so
 *  a future >=2-step texel is traceable to the channel that carries it, never anonymous. */
function emptyMaxSteps(): Record<Channel, number> {
    const out = {} as Record<Channel, number>;
    for (const channel of CHANNELS) out[channel] = 0;
    return out;
}

/** Runs the discrete storage-seam claim over one level's `expected`/`published` pair, accumulating
 *  into `histogram` and `maxSteps` (per channel) — the caller decides whether to assert. */
function checkSeamLevel(
    level: number,
    expected: Float32Array,
    published: Float32Array,
    histogram: SeamHistogram,
    maxSteps: Record<Channel, number>,
): void {
    const count = expected.length / 4;
    for (let i = 0; i < count; i++) {
        CHANNELS.forEach((channel, c) => {
            const steps = f16StepDistance(expected[i * 4 + c], published[i * 4 + c]);
            if (steps === 0) histogram.exact++;
            else if (steps === 1) histogram.neighbor++;
            else histogram.slack++;
            if (steps > maxSteps[channel]) maxSteps[channel] = steps;
            const limit = allowedSteps(level, channel);
            if (steps > limit) {
                histogram.violations.push(
                    `L${level} ${channel} texel${i} steps=${steps} limit=${limit}`,
                );
            }
        });
    }
}

async function runChecks(state: State): Promise<Check[]> {
    const checks: Check[] = [];

    const expectedNames = SLOPE_CASCADE_CONFIGS.map((_config, i) => `slope${i}`);
    checks.push({
        name: "slope publication names are non-empty",
        pass: expectedNames.length > 0,
        detail: `[${expectedNames.join(", ")}]`,
    });

    const displacementNames = getCascadeConfigs().map((_config, i) => `displace${i}`);
    checks.push({
        name: "slope publication names never collide with the displacement cascade namespace",
        pass: expectedNames.every((name) => !displacementNames.includes(name)),
        detail: `slope=[${expectedNames.join(", ")}] displace=[${displacementNames.join(", ")}]`,
    });

    // Identity check over the real registered System array, never a cardinality proxy.
    const systems = OceanPlugin.systems ?? [];
    checks.push({
        name: "OceanPlugin registers the slope compute system",
        pass: systems.includes(slopeCompute),
        detail: `systems=[${systems.map((system) => system.name ?? "?").join(", ")}]`,
    });

    const [textureName] = expectedNames;
    const texture = Compute.textures.get(textureName);
    if (!texture) {
        checks.push({ name: `${textureName} texture published`, pass: false, detail: "missing" });
        return checks;
    }
    checks.push({
        name: `${textureName} texture published with every declared mip level`,
        pass: texture.mipLevelCount === SLOPE_MIP_LEVELS,
        detail: `mipLevelCount=${texture.mipLevelCount}, expected ${SLOPE_MIP_LEVELS}`,
    });

    const buffers = getSlopeBuffers(0);
    if (!buffers) {
        checks.push({ name: "slope0 post-IFFT buffers reachable", pass: false, detail: "missing" });
        return checks;
    }

    const claimTime = state.time.elapsed;
    checks.push({
        name: "the level-0 reading uses the paused verify-harness clock",
        pass: claimTime > 0 && state.time.elapsed === claimTime,
        detail: `state.time.elapsed=${state.time.elapsed}`,
    });
    if (!(claimTime > 0) || state.time.elapsed !== claimTime) return checks;

    const h0 = generateH0(SLOPE_CONFIG, 0);
    const { x: xSpectrum, z: zSpectrum } = runSlopeCpuPipeline(h0, SLOPE_CONFIG, claimTime);
    const x64 = ifft2Exact(xSpectrum, N);
    const z64 = ifft2Exact(zSpectrum, N);
    const normX64 = complexL2Norm(x64.re, x64.im);
    const normZ64 = complexL2Norm(z64.re, z64.im);

    const twiddle = await measureTwiddleTrigError(N);
    const e0X = higham242AbsoluteBound(twiddle.muMax, F32_UNIT_ROUNDOFF, FFT_STAGES, normX64);
    const e0Z = higham242AbsoluteBound(twiddle.muMax, F32_UNIT_ROUNDOFF, FFT_STAGES, normZ64);
    const phaseTrig = await measureSlopePhaseTrigError(SLOPE_CONFIG, claimTime);
    checks.push({
        name: "phase trig error is measured over every level-0 kernel argument and readback",
        pass: phaseTrig.arguments === N * N && phaseTrig.readbackArguments === N * N,
        detail:
            `compared=${phaseTrig.arguments}/${N * N}, readback=${phaseTrig.readbackArguments}/${N * N}, ` +
            `max=${phaseTrig.maxError.toExponential(4)}, rms=${phaseTrig.rmsError.toExponential(4)}`,
    });
    const inputPerturbation = phaseInputPerturbationNorms(h0, phaseTrig.maxError);
    const boundX = e0X + inputPerturbation.x;
    const boundZ = e0Z + inputPerturbation.z;
    checks.push({
        name: "level-0 bound keeps FFT twiddle and input perturbation terms separate",
        pass: boundX === e0X + inputPerturbation.x && boundZ === e0Z + inputPerturbation.z,
        detail:
            `state.time.elapsed=${claimTime}, u=${F32_UNIT_ROUNDOFF}, muMax=${twiddle.muMax.toExponential(4)}, ` +
            `phaseMax=${phaseTrig.maxError.toExponential(4)}, stages=${FFT_STAGES}, ` +
            `E0X=${e0X.toExponential(4)}, E0Z=${e0Z.toExponential(4)}, ` +
            `inputX=${inputPerturbation.x.toExponential(4)}, inputZ=${inputPerturbation.z.toExponential(4)}, ` +
            `boundX=${boundX.toExponential(4)}, boundZ=${boundZ.toExponential(4)}`,
    });

    const [liveXProbe, liveZProbe] = await Promise.all([
        probeBuffer(Compute.device, buffers.x),
        probeBuffer(Compute.device, buffers.z),
    ]);
    const devX = complexDeviation(liveXProbe.bytes, x64.re, x64.im);
    const devZ = complexDeviation(liveZProbe.bytes, z64.re, z64.im);
    const staleCpu = runSlopeCpuPipeline(h0, SLOPE_CONFIG, 0);
    const staleX64 = ifft2Exact(staleCpu.x, N);
    const staleZ64 = ifft2Exact(staleCpu.z, N);
    const staleDevX = complexDeviation(liveXProbe.bytes, staleX64.re, staleX64.im);
    const staleDevZ = complexDeviation(liveZProbe.bytes, staleZ64.re, staleZ64.im);
    checks.push({
        name: "different-clock reference reds the level-0 computation claim on slopeX",
        pass: staleDevX.l2 > boundX,
        detail: `referenceTime=0, state.time.elapsed=${claimTime}, deviation(L2)=${staleDevX.l2.toExponential(4)}, bound=${boundX.toExponential(4)}`,
    });
    checks.push({
        name: "different-clock reference reds the level-0 computation claim on slopeZ",
        pass: staleDevZ.l2 > boundZ,
        detail: `referenceTime=0, state.time.elapsed=${claimTime}, deviation(L2)=${staleDevZ.l2.toExponential(4)}, bound=${boundZ.toExponential(4)}`,
    });
    checks.push({
        name: "paused-clock level-0 computation claim holds on slopeX",
        pass: devX.l2 <= boundX,
        detail: `deviation: L2=${devX.l2.toExponential(4)}, max=${devX.max.toExponential(4)}, rms=${devX.rms.toExponential(4)}, bound=${boundX.toExponential(4)}`,
    });
    checks.push({
        name: "paused-clock level-0 computation claim holds on slopeZ",
        pass: devZ.l2 <= boundZ,
        detail: `deviation: L2=${devZ.l2.toExponential(4)}, max=${devZ.max.toExponential(4)}, rms=${devZ.rms.toExponential(4)}, bound=${boundZ.toExponential(4)}`,
    });

    // Energy/residual recomputed in f64 from the SAME x64/z64 inputs — informational only (no
    // tolerance sits here; these two channels are covered by the discrete storage-seam claim
    // below, never by a second numeric bound).
    const xReal0 = realComponent(liveXProbe.bytes, N * N);
    const zReal0 = realComponent(liveZProbe.bytes, N * N);
    const expected0 = expectedLevel0(xReal0, zReal0);
    let energyMax = 0;
    let energySumSquares = 0;
    for (let i = 0; i < N * N; i++) {
        const energy64 = x64.re[i] * x64.re[i] + z64.re[i] * z64.re[i];
        const d = Math.abs(energy64 - expected0[i * 4 + 2]);
        if (d > energyMax) energyMax = d;
        energySumSquares += d * d;
    }
    checks.push({
        name: "energy/residual recomputed in f64 (informational — no tolerance sits here)",
        pass: true,
        detail:
            `energy: max=${energyMax.toExponential(4)} rms=${Math.sqrt(energySumSquares / (N * N)).toExponential(4)} ` +
            "(f64 recompute vs f32-disciplined recompute); residual is architecturally exact zero at level 0",
    });

    // ── Storage seam claim: every texel, every level, every channel ────────────────────────────
    const published: Float32Array[] = [];
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        published.push(await publishedLevel(texture, level));
    }

    const total: SeamHistogram = { exact: 0, neighbor: 0, slack: 0, violations: [] };
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const expected =
            level === 0
                ? expected0
                : expectedFromPublished(
                      published[level - 1],
                      slopeMipSize(SLOPE_CONFIG, level - 1),
                  );
        const levelHistogram: SeamHistogram = { exact: 0, neighbor: 0, slack: 0, violations: [] };
        const levelMaxSteps = emptyMaxSteps();
        checkSeamLevel(level, expected, published[level], levelHistogram, levelMaxSteps);
        const size = slopeMipSize(SLOPE_CONFIG, level);
        const levelTexels = levelHistogram.exact + levelHistogram.neighbor + levelHistogram.slack;
        checks.push({
            name: `mip ${level} storage seam histogram (0/1/>=2 step)`,
            pass: levelHistogram.violations.length === 0,
            detail:
                `size=${size}x${size}, texels=${levelTexels}, 0-step=${levelHistogram.exact}, ` +
                `1-step=${levelHistogram.neighbor}, >=2-step=${levelHistogram.slack}, ` +
                `maxSteps=[${CHANNELS.map((channel) => `${channel}:${levelMaxSteps[channel]}`).join(", ")}], ` +
                `violations=${levelHistogram.violations.length}` +
                (levelHistogram.violations.length > 0
                    ? ` (${levelHistogram.violations.slice(0, 5).join("; ")})`
                    : ""),
        });
        total.exact += levelHistogram.exact;
        total.neighbor += levelHistogram.neighbor;
        total.slack += levelHistogram.slack;
        // `.concat`, never a spread `push(...huge array)` — a pathological mutation can produce a
        // violations array well past V8's max call-argument count, and a spread there throws
        // `RangeError: Maximum call stack size exceeded` before the real (failing) check is even
        // reported (witnessed against a swapped-channel mutation: 65534 violations at one level).
        total.violations = total.violations.concat(levelHistogram.violations);
    }
    const totalTexels = total.exact + total.neighbor + total.slack;
    checks.push({
        name: "storage seam claim holds over every texel of every level and channel",
        pass: total.violations.length === 0,
        detail:
            `total=${totalTexels}, 0-step=${total.exact}, 1-step=${total.neighbor}, ` +
            `>=2-step=${total.slack}, violations=${total.violations.length}`,
    });

    // residual at level 0 is exactly zero on both sides — asserted directly, not merely folded
    // into the aggregate seam claim above.
    let residualNonZero = 0;
    for (let i = 0; i < N * N; i++) if (published[0][i * 4 + 3] !== 0) residualNonZero++;
    checks.push({
        name: "residual at level 0 is exactly zero",
        pass: residualNonZero === 0,
        detail: `${residualNonZero} nonzero texels of ${N * N}`,
    });

    // Zeroed-payload vacuity witness per level (Gate law): zero the published level and re-run
    // the SAME seam comparison at the SAME allowed-steps limit — a red here is required.
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const expected =
            level === 0
                ? expected0
                : expectedFromPublished(
                      published[level - 1],
                      slopeMipSize(SLOPE_CONFIG, level - 1),
                  );
        const zeroed = new Float32Array(expected.length);
        const zeroHistogram: SeamHistogram = { exact: 0, neighbor: 0, slack: 0, violations: [] };
        checkSeamLevel(level, expected, zeroed, zeroHistogram, emptyMaxSteps());
        checks.push({
            name: `mip ${level} zeroed-payload vacuity witness`,
            pass: zeroHistogram.violations.length > 0,
            detail: `${zeroHistogram.violations.length} violations against a zeroed payload`,
        });
    }

    return checks;
}

export async function runDeviceClaim(state: State): Promise<Check[]> {
    state.pause();
    return runChecks(state);
}
