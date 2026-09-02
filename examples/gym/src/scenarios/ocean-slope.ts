// ocean-slope — the exact storage-seam pyramid claim, replacing a prior tolerance-based
// mip-agreement arm after two correction rounds convicted every version of that tolerance as
// either a permitted-floor term that cannot discriminate, or a term sized to the very readings it
// was meant to gate. The ruling: the comparison sits on a representation seam (the published value
// IS a rounding of the f32 value it came from), so the claim is DISCRETE — no tolerance belongs
// there at all — and the one place a real numeric tolerance is warranted is one layer up, at the
// FFT's own floating-point arithmetic. Two independent claims, never one number standing in for
// both (`packages/shallot-ocean/src/slope-seam.ts` owns the math for both):
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

import {
    Compute,
    probeBuffer,
    probeTexture,
    RenderPlugin,
    run,
    SlabPlugin,
    type State,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
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
} from "@dylanebert/shallot-ocean";
import { type Check, frames, register, type Scenario } from "../gym";

const [SLOPE_CONFIG] = SLOPE_CASCADE_CONFIGS;
const N = SLOPE_CONFIG.N;
const DECLARED_TIME = 1 / 30;

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

    // Pre-publish control: read the exact buffers `slopeCompute` writes BEFORE any frame steps —
    // `buildSlopes` has allocated them but `slopeCompute` has never run, so by WebGPU's
    // zero-initialization guarantee this is exactly the state a plugin permanently missing
    // `slopeCompute` from its `systems` array would leave forever.
    // `Promise.all`, never two sequential `await`s: `state.pause()` freezes the virtual clock but
    // NOT the frame loop itself (`slopeCompute` is a `draw`-group system and still runs every real
    // animation frame while paused), so a real frame tick landing between two separately-awaited
    // probes would let the SECOND buffer publish before the FIRST is read — witnessed exactly this
    // way once: `x` read zero as expected while `z` had already been written. Starting both
    // `probeBuffer` calls in the same microtask keeps their synchronous encode+submit prefix atomic
    // with respect to the frame loop.
    const [prePublishXProbe, prePublishZProbe] = await Promise.all([
        probeBuffer(Compute.device, buffers.x),
        probeBuffer(Compute.device, buffers.z),
    ]);

    await frames(2);

    checks.push({
        name: "the compared reading is taken at declared time zero",
        pass: state.time.elapsed === 0,
        detail: `state.time.elapsed=${state.time.elapsed}`,
    });
    if (state.time.elapsed !== 0) return checks;

    // f64 CPU reference — from the SAME h0 seed (`buildSlopes`/`createSlopeState` seeds every
    // cascade with `generateH0(config, 0)`), transformed WITHOUT the final f32-truncating cast
    // (`ifft2Exact`, never `ifft2`). NOT "the exact same spectral input": `slopeSpectra` (CPU, this
    // module's `runSlopeCpuPipeline` call) and `slopeKernel` (GPU, WGSL) are two SEPARATE f32
    // computations of the same `i*k*h̃` formula from the same seed, each accumulating its own f32
    // rounding, so they can differ by ~1 ULP even before either FFT runs. That difference is priced
    // out rather than ignored: it sits at the same `u = 2^-24` order Higham's own bound already
    // carries as its per-operation unit roundoff, so E0 (derived below) already covers it.
    const h0 = generateH0(SLOPE_CONFIG, 0);
    const { x: xSpectrum, z: zSpectrum } = runSlopeCpuPipeline(h0, SLOPE_CONFIG, 0);
    const x64 = ifft2Exact(xSpectrum, N);
    const z64 = ifft2Exact(zSpectrum, N);
    const normX64 = complexL2Norm(x64.re, x64.im);
    const normZ64 = complexL2Norm(z64.re, z64.im);

    const twiddle = await measureTwiddleTrigError(N);
    const e0X = higham242AbsoluteBound(twiddle.muMax, F32_UNIT_ROUNDOFF, FFT_STAGES, normX64);
    const e0Z = higham242AbsoluteBound(twiddle.muMax, F32_UNIT_ROUNDOFF, FFT_STAGES, normZ64);
    // Informational: E0's own derivation inputs, printed once beside the two claims that use it.
    checks.push({
        name: "E0 (Higham Theorem 24.2) derivation inputs",
        pass: true,
        detail:
            `u=${F32_UNIT_ROUNDOFF}, muMax=${twiddle.muMax.toExponential(4)} (measured over ` +
            `${twiddle.pairs} twiddle pairs), stages=${FFT_STAGES}, ||x64||2=${normX64.toFixed(4)}, ` +
            `||z64||2=${normZ64.toFixed(4)} -> E0(slopeX)=${e0X.toExponential(4)}, ` +
            `E0(slopeZ)=${e0Z.toExponential(4)}`,
    });

    // Pre-publish read: the buffers are still all-zero at this point, so the deviation against a
    // nonzero real field is (up to floating point) the field's own norm — this MUST red.
    const preDevX = complexDeviation(prePublishXProbe.bytes, x64.re, x64.im);
    const preDevZ = complexDeviation(prePublishZProbe.bytes, z64.re, z64.im);
    checks.push({
        name: "pre-publish read reds the level-0 computation claim on slopeX",
        pass: preDevX.l2 > e0X,
        detail: `deviation(L2)=${preDevX.l2.toExponential(4)}, E0=${e0X.toExponential(4)}`,
    });
    checks.push({
        name: "pre-publish read reds the level-0 computation claim on slopeZ",
        pass: preDevZ.l2 > e0Z,
        detail: `deviation(L2)=${preDevZ.l2.toExponential(4)}, E0=${e0Z.toExponential(4)}`,
    });

    // Live (post-publish) buffers — the SAME resources the pre-publish read above just probed,
    // now after `slopeCompute` has actually run.
    const [liveXProbe, liveZProbe] = await Promise.all([
        probeBuffer(Compute.device, buffers.x),
        probeBuffer(Compute.device, buffers.z),
    ]);
    const postDevX = complexDeviation(liveXProbe.bytes, x64.re, x64.im);
    const postDevZ = complexDeviation(liveZProbe.bytes, z64.re, z64.im);
    checks.push({
        name: "level-0 computation claim holds on slopeX",
        pass: postDevX.l2 <= e0X,
        detail:
            `deviation: L2=${postDevX.l2.toExponential(4)} (gated), max=${postDevX.max.toExponential(4)}, ` +
            `rms=${postDevX.rms.toExponential(4)}, E0=${e0X.toExponential(4)}, ` +
            `reach=${(e0X / Math.max(postDevX.l2, 1e-30)).toFixed(1)}x`,
    });
    checks.push({
        name: "level-0 computation claim holds on slopeZ",
        pass: postDevZ.l2 <= e0Z,
        detail:
            `deviation: L2=${postDevZ.l2.toExponential(4)} (gated), max=${postDevZ.max.toExponential(4)}, ` +
            `rms=${postDevZ.rms.toExponential(4)}, E0=${e0Z.toExponential(4)}, ` +
            `reach=${(e0Z / Math.max(postDevZ.l2, 1e-30)).toFixed(1)}x`,
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

    state.resume();
    state.step(DECLARED_TIME);
    state.pause();
    checks.push({
        name: "level-0 phase reading is taken at the declared nonzero time",
        pass: state.time.elapsed === DECLARED_TIME,
        detail: `state.time.elapsed=${state.time.elapsed}, declared=${DECLARED_TIME}`,
    });
    if (state.time.elapsed !== DECLARED_TIME) return checks;

    const phaseTrig = await measureSlopePhaseTrigError(SLOPE_CONFIG, DECLARED_TIME);
    checks.push({
        name: "phase trig error is measured over every level-0 kernel argument",
        pass: phaseTrig.arguments === N * N,
        detail:
            `arguments=${phaseTrig.arguments}/${N * N}, maxCos=${phaseTrig.maxCosError.toExponential(4)}, ` +
            `maxSin=${phaseTrig.maxSinError.toExponential(4)}, max=${phaseTrig.maxError.toExponential(4)}, ` +
            `rms=${phaseTrig.rmsError.toExponential(4)}`,
    });

    const timedCpu = runSlopeCpuPipeline(h0, SLOPE_CONFIG, DECLARED_TIME);
    const timedX64 = ifft2Exact(timedCpu.x, N);
    const timedZ64 = ifft2Exact(timedCpu.z, N);
    const timedNormX = complexL2Norm(timedX64.re, timedX64.im);
    const timedNormZ = complexL2Norm(timedZ64.re, timedZ64.im);
    const timedMu = Math.max(twiddle.muMax, phaseTrig.maxError);
    const timedE0X = higham242AbsoluteBound(timedMu, F32_UNIT_ROUNDOFF, FFT_STAGES, timedNormX);
    const timedE0Z = higham242AbsoluteBound(timedMu, F32_UNIT_ROUNDOFF, FFT_STAGES, timedNormZ);
    const [timedXProbe, timedZProbe] = await Promise.all([
        probeBuffer(Compute.device, buffers.x),
        probeBuffer(Compute.device, buffers.z),
    ]);
    const timedDevX = complexDeviation(timedXProbe.bytes, timedX64.re, timedX64.im);
    const timedDevZ = complexDeviation(timedZProbe.bytes, timedZ64.re, timedZ64.im);
    checks.push({
        name: "declared-time level-0 computation claim holds on slopeX",
        pass: timedDevX.l2 <= timedE0X,
        detail: `deviation(L2)=${timedDevX.l2.toExponential(4)}, E0=${timedE0X.toExponential(4)}, mu=${timedMu.toExponential(4)}`,
    });
    checks.push({
        name: "declared-time level-0 computation claim holds on slopeZ",
        pass: timedDevZ.l2 <= timedE0Z,
        detail: `deviation(L2)=${timedDevZ.l2.toExponential(4)}, E0=${timedE0Z.toExponential(4)}, mu=${timedMu.toExponential(4)}`,
    });

    return checks;
}

let checks: Check[] = [];

const scenario: Scenario = {
    name: "ocean-slope",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [
                ProfilePlugin,
                SlabPlugin,
                TransformsPlugin,
                RenderPlugin,
                { ...OceanPlugin, dependencies: [] },
            ],
        });
        // Pin `elapsed` at 0 before the auto frame loop's first `requestAnimationFrame` can fire.
        state.pause();
        try {
            checks = await runChecks(state);
        } catch (err) {
            checks = [{ name: "ocean-slope runChecks threw", pass: false, detail: String(err) }];
        }
        return { state, dispose };
    },
    async assert() {
        return checks;
    },
};

register(scenario);
