// ocean-slope — the published slope-cascade product-texture verdict (`shallot-water-surface` I3g).
// One real-device arm reads the SAME texture `buildSlopes` builds and `slopeCompute` steps every
// frame (never a second GPU state built from the same kernels — the texture is read straight off
// `Compute.textures`, the plugin's own publication surface). For every mip level and channel it
// derives a comparison bound from the compared representation's own error model — the rgba16float
// storage quantum plus the GPU inverse FFT's own twiddle-factor trig error (below), scaled per
// level AND PER CHANNEL by that channel's own CPU-reference magnitude at that level (never a single
// level-0 scale reused across channels or projected onto deeper levels) — asserts the resulting
// verified/unverifiable partition, requires at least one verified channel per level, then zeroes
// each verified GPU payload and re-runs the same comparison at the same bound. A control reading
// taken BEFORE any frame steps — the exact state a plugin missing `slopeCompute` would leave
// forever, by WebGPU's zero-initialization guarantee — reds the same comparison at every verified
// mip-0 channel.
//
// The bound carries NO term for `slopeKernel`'s phase evolution (cos/sin(ω·t), cos/sin(-ω·t)) — the
// one place a NONZERO elapsed time would make the CPU and GPU sides structurally diverge. The arm
// asserts `state.time.elapsed === 0` before it reads any level (below) — the same declared time the
// CPU reference uses — so both sides evaluate cos(0)/sin(0), exact on both implementations, and
// that error path is unreachable by construction rather than merely small; a nonzero elapsed reds
// the assertion before any comparison runs. The bound DOES still carry a trig-accuracy term for a
// DIFFERENT, still-live consumer: the inverse FFT's OWN twiddle factors (`gpu-fft.ts`'s
// `std.cos(angle)`/`std.sin(angle)`, `angle = 2π·blockLocal/len`) are structural, time-independent
// angles evaluated at every butterfly stage every frame regardless of `t` — see `levelBound`.
//
// `bun bench` itself already skips loudly (`scripts/bench.ts`'s `skipReason()`) when the seat has
// no real GPU adapter, before any scenario boots — this file adds no second adapter check.

import {
    Compute,
    probeTexture,
    RenderPlugin,
    run,
    SlabPlugin,
    type State,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    type CascadeConfig,
    generateH0,
    getCascadeConfigs,
    OceanPlugin,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeCompute,
    slopeMipSize,
} from "@dylanebert/shallot-ocean";
import { type Check, frames, register, type Scenario } from "../gym";

const [SLOPE_CONFIG] = SLOPE_CASCADE_CONFIGS;
const CHANNEL_NAMES = ["slopeX", "slopeZ", "energy", "residual"] as const;

/** rgba16float round-to-nearest relative quantum: 10 explicit mantissa bits, half a ULP. */
const HALF_FLOAT_REL_QUANTUM = 2 ** -11;

/**
 * rgba16float's subnormal step is a FIXED 2^-24 (5-bit exponent field zero, 10-bit mantissa),
 * independent of the represented value's own magnitude — `HALF_FLOAT_REL_QUANTUM` above only
 * models half float's NORMAL range (|x| >= 2^-14); below it, round-to-nearest error is bounded by
 * half this fixed step rather than by any fraction of the (near-zero) value itself. A deep mip
 * level's channel can land here: level 8 of a 256x256 field averages slope down to ~1e-7.
 */
const HALF_FLOAT_SUBNORMAL_ABS_QUANTUM = 2 ** -25;

/**
 * WGSL's documented absolute error bound for the trigonometric built-ins over their accurate
 * domain (WebGPU Shading Language spec, "Floating Point Evaluation" — built-in function accuracy).
 * The live consumer is the inverse FFT's OWN twiddle factors (`gpu-fft.ts`'s `std.cos(angle)`/
 * `std.sin(angle)`, `angle = 2π·blockLocal/len`) — a structural angle independent of elapsed time,
 * evaluated at every butterfly stage every frame regardless of `t` (never the phase-evolution
 * `cos/sin(ω·t)` term this bound used to carry — that term is unreachable by construction, per the
 * file header's `state.time.elapsed === 0` assertion).
 */
const WGSL_TRIG_ABS_ERROR = 2 ** -11;

/** `ifft2` is separable — a row pass then a column pass, each a full radix-2 Cooley-Tukey
 *  transform over `SLOPE_CONFIG.N` (`fft.ts`) — so the twiddle term below counts TWO independent
 *  stage sequences, each `log2(N)` complex butterfly stages deep. */
const FFT_DIMENSIONS = 2;
const FFT_STAGES_PER_DIMENSION = Math.log2(SLOPE_CONFIG.N);

/**
 * Absolute comparison bound for one mip level and channel, scaled by `channelScale` — that
 * channel's own CPU-reference magnitude AT THAT LEVEL (measured directly from `cpuSlopeLevels()`,
 * never level 0's range projected onto a deeper level or a different channel — `energy` and
 * `residual` are quadratic in `slopeX`/`slopeZ` and shrink through mip-averaging at a different
 * rate, so a shared scale would misprice them). Two additive routes, both re-derived here rather
 * than authored:
 *
 * - STORAGE: level 0 is quantized once when `slopePostKernel` writes it; each further mip level
 *   reads the previous (already half-float-quantized) level and re-quantizes its own reduced
 *   output on write, so level `L` has been through `L + 1` independent half-float round trips —
 *   the accumulation-depth term the Gate law requires beside the quantum itself.
 * - FFT TWIDDLE: baked into level 0's real-space value once by the GPU inverse FFT's own
 *   twiddle-factor trig evaluations (`WGSL_TRIG_ABS_ERROR` above), then carried forward roughly
 *   UNCHANGED in relative magnitude through every later level's linear mip-averaging
 *   (`reduceSlopeMip`/`mipKernel` perform no further trig evaluations) — so this term does NOT
 *   compound with level the way the storage term does.
 *
 * A floor covers the regime the relative model above cannot: below half float's normal range
 * (`HALF_FLOAT_SUBNORMAL_ABS_QUANTUM`), the two relative terms shrink toward zero alongside the
 * (also shrinking) reference scale, while the ACTUAL round-to-nearest error floors at a fixed
 * absolute step — so the per-round-trip bound is the max of the relative and absolute routes,
 * summed over the same `L + 1` accumulation depth as the storage term.
 */
function levelBound(level: number, channelScale: number): number {
    const trips = level + 1;
    const storageTerm = trips * HALF_FLOAT_REL_QUANTUM * channelScale;
    const fftTwiddleTerm =
        FFT_DIMENSIONS * FFT_STAGES_PER_DIMENSION * WGSL_TRIG_ABS_ERROR * channelScale;
    const relativeBound = storageTerm + fftTwiddleTerm;
    const absoluteFloor = trips * HALF_FLOAT_SUBNORMAL_ABS_QUANTUM;
    return Math.max(relativeBound, absoluteFloor);
}

/** The CPU mip chain, mirroring the GPU's own `slopePostKernel` → `mipKernel` chain exactly
 *  (`reduceSlopeMip` is the package's own CPU mirror of `mipKernel`). */
function cpuSlopeLevels(): Float32Array[] {
    const h0 = generateH0(SLOPE_CONFIG, 0);
    const { xField, zField } = runSlopeCpuPipeline(h0, SLOPE_CONFIG, 0);
    const N = SLOPE_CONFIG.N;
    const level0 = new Float32Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
        const sx = xField[i * 2];
        const sz = zField[i * 2];
        level0[i * 4] = sx;
        level0[i * 4 + 1] = sz;
        level0[i * 4 + 2] = sx * sx + sz * sz;
        level0[i * 4 + 3] = 0;
    }
    const levels: Float32Array[] = [level0];
    for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
        levels.push(reduceSlopeMip(levels[level - 1], slopeMipSize(SLOPE_CONFIG, level - 1)));
    }
    return levels;
}

/** One mip level's published RGBA payload, widened from the texture's native half-float storage —
 *  `Float16Array` is an already-relied-on platform floor in this repo (`skin.ts`'s VAT path). */
async function gpuSlopeLevel(texture: GPUTexture, level: number): Promise<Float32Array> {
    const probe = await probeTexture(Compute.device, texture, { mipLevel: level });
    return Float32Array.from(new Float16Array(probe.bytes));
}

function maxAbs(rgba: Float32Array, channel: number): number {
    let max = 0;
    for (let i = 0; i < rgba.length / 4; i++) {
        const v = Math.abs(rgba[i * 4 + channel]);
        if (v > max) max = v;
    }
    return max;
}

function maxDeviation(cpu: Float32Array, gpu: Float32Array, channel: number): number {
    let max = 0;
    for (let i = 0; i < cpu.length / 4; i++) {
        const d = Math.abs(cpu[i * 4 + channel] - gpu[i * 4 + channel]);
        if (d > max) max = d;
    }
    return max;
}

type Classification = "verified" | "unverifiable" | "failed";

/** Payload below its own derived bound enters the unverifiable partition (Gate law) rather than a
 *  universal pass/fail claim — only a payload that clears the bound and still agrees is "verified",
 *  and only a payload that clears the bound and disagrees is a real "failed" reading. */
function classify(payloadMax: number, deviation: number, bound: number): Classification {
    if (payloadMax < bound) return "unverifiable";
    return deviation < bound ? "verified" : "failed";
}

interface LevelReading {
    level: number;
    gpu: Float32Array;
    /** per-channel: each channel's own scale-derived bound, never one bound shared across channels. */
    bound: number[];
    payloadMax: number[];
    deviation: number[];
    classification: Classification[];
}

/** `cpu` is THIS level's CPU reference (from `cpuSlopeLevels()`), so `channelScale` is measured at
 *  the same level it bounds — never level 0's range projected onto a deeper level. A channel whose
 *  CPU reference is identically zero at this level (`residual` at level 0 — no sub-texel content by
 *  design) still gets a nonzero bound from `levelBound`'s absolute floor, so an exact zero-vs-zero
 *  match reads "unverifiable" (nothing to check) rather than tripping `classify`'s strict `<` on a
 *  degenerate zero bound. */
function readLevel(level: number, cpu: Float32Array, gpu: Float32Array): LevelReading {
    const channelScale = CHANNEL_NAMES.map((_, c) => maxAbs(cpu, c));
    const bound = channelScale.map((scale) => levelBound(level, scale));
    const payloadMax = CHANNEL_NAMES.map((_, c) => maxAbs(gpu, c));
    const deviation = CHANNEL_NAMES.map((_, c) => maxDeviation(cpu, gpu, c));
    const classification = CHANNEL_NAMES.map((_, c) =>
        classify(payloadMax[c], deviation[c], bound[c]),
    );
    return { level, gpu, bound, payloadMax, deviation, classification };
}

async function runChecks(state: State): Promise<Check[]> {
    const checks: Check[] = [];

    const expectedNames = SLOPE_CASCADE_CONFIGS.map(
        (_config: CascadeConfig, i: number) => `slope${i}`,
    );
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

    // Identity check over the real registered System array, never a cardinality proxy — a count
    // like `otherSystems.length === 1` passes on any substitute system and reds on a legitimate
    // third one; `slopeCompute` is on the package barrel beside `oceanCompute`, so pinning the
    // concrete identity is consistent with the existing public surface rather than a widening
    // invented for this arm.
    const systems = OceanPlugin.systems ?? [];
    checks.push({
        name: "OceanPlugin registers the slope compute system",
        pass: systems.includes(slopeCompute),
        detail: `systems=[${systems.map((system) => system.name ?? "?").join(", ")}]`,
    });

    // Mutation-red control for the two presence checks above: this branch is a real, reachable
    // path (not a decorative always-true assertion) — it fires exactly when `buildSlopes` never
    // publishes `textureName`, so a regression that stops publishing under this name reds here
    // rather than passing through an untaken branch.
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

    const cpu = cpuSlopeLevels();

    // Pre-publish control: read the exact texture the plugin publishes BEFORE any frame steps —
    // `buildSlopes` has allocated it but `slopeCompute` has never run, so by WebGPU's
    // zero-initialization guarantee this is exactly the state a plugin permanently missing
    // `slopeCompute` from its `systems` array would leave forever.
    const prePublish = await gpuSlopeLevel(texture, 0);

    // `state.pause()` was called synchronously right after `run()` resolved, before the auto frame
    // loop's first `requestAnimationFrame` could fire, so `elapsed` has been pinned at exactly 0
    // this whole time (`scheduler.ts`: `elapsed += paused ? 0 : real * scale`) — the one declared
    // deterministic time this arm's CPU reference (`runSlopeCpuPipeline(..., 0)`) also uses. `draw`
    // group systems (`slopeCompute` included) still run every frame while paused; only the virtual
    // clock freezes.
    await frames(2);

    // The premise `levelBound` rests on: with no term for `slopeKernel`'s phase evolution, every
    // downstream comparison assumes cos/sin(ω·elapsed) was never exercised (the FFT twiddle-factor
    // trig term levelBound DOES carry is unaffected by this — it comes from the transform's own
    // structural angles, not from elapsed time). Assert the premise as an arm rather than leave it
    // as a comment — a red here means the bound above is no longer honest and every reading past
    // this point is measuring the wrong thing.
    checks.push({
        name: "the compared reading is taken at declared time zero",
        pass: state.time.elapsed === 0,
        detail: `state.time.elapsed=${state.time.elapsed} — levelBound's phase-evolution omission depends on this`,
    });
    if (state.time.elapsed !== 0) return checks;

    const readings: LevelReading[] = [];
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const gpu = await gpuSlopeLevel(texture, level);
        readings.push(readLevel(level, cpu[level], gpu));
    }

    for (const reading of readings) {
        for (let c = 0; c < CHANNEL_NAMES.length; c++) {
            const kind = reading.classification[c];
            checks.push({
                name: `mip ${reading.level} channel ${CHANNEL_NAMES[c]} verified/unverifiable partition`,
                pass: kind !== "failed",
                detail:
                    `payloadMax=${reading.payloadMax[c]}, deviation=${reading.deviation[c]}, ` +
                    `bound=${reading.bound[c]}, classification=${kind}`,
                data: {
                    payloadMax: reading.payloadMax[c],
                    deviation: reading.deviation[c],
                    bound: reading.bound[c],
                },
            });
        }
        const verifiedCount = reading.classification.filter((k) => k === "verified").length;
        checks.push({
            name: `mip ${reading.level} has at least one verified channel`,
            pass: verifiedCount > 0,
            detail: `verified=[${reading.classification.join(", ")}]`,
        });
    }

    // Vacuity witness (Gate law): zero each verified GPU payload and re-run the SAME comparison at
    // the SAME bound — a red here is required, or the comparison isn't actually reading the GPU.
    for (const reading of readings) {
        for (let c = 0; c < CHANNEL_NAMES.length; c++) {
            if (reading.classification[c] !== "verified") continue;
            const zeroed = reading.gpu.slice();
            for (let i = 0; i < zeroed.length / 4; i++) zeroed[i * 4 + c] = 0;
            const mutatedDeviation = maxDeviation(cpu[reading.level], zeroed, c);
            checks.push({
                name: `mip ${reading.level} channel ${CHANNEL_NAMES[c]} zeroed-payload vacuity witness`,
                pass: mutatedDeviation >= reading.bound[c],
                detail:
                    `green=${reading.deviation[c]}, mutated(zeroed)=${mutatedDeviation}, ` +
                    `bound=${reading.bound[c]}, reach=${(mutatedDeviation / reading.bound[c]).toFixed(1)}x`,
            });
        }
    }

    // "removing slopeCompute from the plugin must red": every verified mip-0 channel's comparison,
    // re-run against the pre-publish (never-stepped) reading above, must fail — proving the arm's
    // green depends on `slopeCompute` having actually run, not merely on the texture existing.
    const mip0 = readings[0];
    const mip0VerifiedChannels = mip0.classification
        .map((kind, c) => (kind === "verified" ? c : -1))
        .filter((c) => c >= 0);
    checks.push({
        name: "mip 0 carries at least one verified channel to red-witness slopeCompute removal",
        pass: mip0VerifiedChannels.length > 0,
        detail: `verified=[${mip0.classification.join(", ")}]`,
    });
    for (const c of mip0VerifiedChannels) {
        const preDeviation = maxDeviation(cpu[0], prePublish, c);
        checks.push({
            name: `mip 0 channel ${CHANNEL_NAMES[c]} reds against the pre-publish (no-slopeCompute) state`,
            pass: preDeviation >= mip0.bound[c],
            detail:
                `pre-publish deviation=${preDeviation}, bound=${mip0.bound[c]} — the zero-init state ` +
                "a plugin missing slopeCompute leaves forever",
        });
    }

    return checks;
}

let checks: Check[] = [];

const scenario: Scenario = {
    name: "ocean-slope",
    noRender: true,
    async build(_canvas) {
        const { state, dispose } = await run({
            defaults: false,
            plugins: [ProfilePlugin, SlabPlugin, TransformsPlugin, RenderPlugin, OceanPlugin],
        });
        // Pin `elapsed` at 0 before the auto frame loop's first `requestAnimationFrame` can fire —
        // see the comment beside `frames(2)` in `runChecks` for why this is the arm's one declared
        // deterministic time.
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
