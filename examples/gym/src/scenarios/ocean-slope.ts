// ocean-slope — the published slope-cascade product-texture verdict (`shallot-water-surface` I3g).
// One real-device arm reads the SAME texture `buildSlopes` builds and `slopeCompute` steps every
// frame (never a second GPU state built from the same kernels — the texture is read straight off
// `Compute.textures`, the plugin's own publication surface). For every mip level and channel it
// derives a comparison bound from the compared representation's own error model — WGSL's
// documented trigonometric imprecision plus the rgba16float storage quantum, compounded through
// the mip chain — asserts the resulting verified/unverifiable partition, requires at least one
// verified channel per level, then zeroes each verified GPU payload and re-runs the same
// comparison at the same bound. A control reading taken BEFORE any frame steps — the exact state a
// plugin missing `slopeCompute` would leave forever, by WebGPU's zero-initialization guarantee —
// reds the same comparison at every verified mip-0 channel.
//
// `bun bench` itself already skips loudly (`scripts/bench.ts`'s `skipReason()`) when the seat has
// no real GPU adapter, before any scenario boots — this file adds no second adapter check.

import {
    Compute,
    probeTexture,
    RenderPlugin,
    run,
    SlabPlugin,
    TransformsPlugin,
} from "@dylanebert/shallot";
import { ProfilePlugin } from "@dylanebert/shallot/extras";
import {
    type CascadeConfig,
    generateH0,
    getCascadeConfigs,
    OceanPlugin,
    oceanCompute,
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipSize,
} from "@dylanebert/shallot-ocean";
import { type Check, frames, register, type Scenario } from "../gym";

const [SLOPE_CONFIG] = SLOPE_CASCADE_CONFIGS;
const CHANNEL_NAMES = ["slopeX", "slopeZ", "energy", "residual"] as const;

/**
 * WGSL's documented absolute error bound for the trigonometric built-ins over their accurate
 * domain (WebGPU Shading Language spec, "Floating Point Evaluation" — built-in function accuracy).
 * The phase evolution (`cos`/`sin(ω·t)`, `cos`/`sin(-ω·t)`, `slopeKernel` in `slope.ts`) is the one
 * place this arm's two sides structurally diverge: the CPU reference (`runSlopeCpuPipeline`) goes
 * through plain `Math.cos`/`Math.sin` (near-correctly-rounded on every JS engine this repo
 * targets), while the GPU kernel's `std.cos`/`std.sin` carry no such guarantee — this is the
 * dominant CROSS-implementation error source below the storage format's own quantum.
 */
const WGSL_TRIG_ABS_ERROR = 2 ** -11;

/** rgba16float round-to-nearest relative quantum: 10 explicit mantissa bits, half a ULP. */
const HALF_FLOAT_REL_QUANTUM = 2 ** -11;

/** Independent trig evaluations feeding one H(k,t) sample: cos/sin(ωt) and cos/sin(-ωt). */
const TRIG_EVAL_COUNT = 4;

/**
 * Absolute comparison bound for one mip level, scaled by `referenceScale` (the level-0 CPU
 * payload's own dynamic range — a measured quantity, never an authored constant). Level 0 is
 * quantized once when `slopePostKernel` writes it; each further mip level reads the previous
 * (already half-float-quantized) level and re-quantizes its own reduced output on write, so level
 * `L` has been through `L + 1` independent half-float round trips — the accumulation-depth term
 * the Gate law requires beside the quantum itself.
 */
function levelBound(level: number, referenceScale: number): number {
    const phaseTerm = TRIG_EVAL_COUNT * WGSL_TRIG_ABS_ERROR * referenceScale;
    const storageTerm = (level + 1) * HALF_FLOAT_REL_QUANTUM * referenceScale;
    return phaseTerm + storageTerm;
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
    bound: number;
    payloadMax: number[];
    deviation: number[];
    classification: Classification[];
}

function readLevel(
    level: number,
    cpu: Float32Array,
    gpu: Float32Array,
    referenceScale: number,
): LevelReading {
    const bound = levelBound(level, referenceScale);
    const payloadMax = CHANNEL_NAMES.map((_, c) => maxAbs(gpu, c));
    const deviation = CHANNEL_NAMES.map((_, c) => maxDeviation(cpu, gpu, c));
    const classification = CHANNEL_NAMES.map((_, c) =>
        classify(payloadMax[c], deviation[c], bound),
    );
    return { level, gpu, bound, payloadMax, deviation, classification };
}

async function runChecks(): Promise<Check[]> {
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

    // structural "external consumer" check — over the real registered System array, never a text
    // search: OceanPlugin registers exactly one system besides the (known, publicly exported)
    // displacement compute pass, and that other system is the one this arm is about.
    const otherSystems = OceanPlugin.systems?.filter((system) => system !== oceanCompute) ?? [];
    checks.push({
        name: "OceanPlugin registers exactly one system besides the displacement compute pass",
        pass: otherSystems.length === 1,
        detail: `${otherSystems.length} other system(s) registered`,
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
    let referenceScale = 0;
    for (let i = 0; i < cpu[0].length / 4; i++) {
        for (let c = 0; c < 3; c++) {
            const v = Math.abs(cpu[0][i * 4 + c]);
            if (v > referenceScale) referenceScale = v;
        }
    }

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

    const readings: LevelReading[] = [];
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const gpu = await gpuSlopeLevel(texture, level);
        readings.push(readLevel(level, cpu[level], gpu, referenceScale));
    }

    for (const reading of readings) {
        for (let c = 0; c < CHANNEL_NAMES.length; c++) {
            const kind = reading.classification[c];
            checks.push({
                name: `mip ${reading.level} channel ${CHANNEL_NAMES[c]} verified/unverifiable partition`,
                pass: kind !== "failed",
                detail:
                    `payloadMax=${reading.payloadMax[c]}, deviation=${reading.deviation[c]}, ` +
                    `bound=${reading.bound}, classification=${kind}`,
                data: {
                    payloadMax: reading.payloadMax[c],
                    deviation: reading.deviation[c],
                    bound: reading.bound,
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
                pass: mutatedDeviation >= reading.bound,
                detail:
                    `green=${reading.deviation[c]}, mutated(zeroed)=${mutatedDeviation}, ` +
                    `bound=${reading.bound}, reach=${(mutatedDeviation / reading.bound).toFixed(1)}x`,
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
            pass: preDeviation >= mip0.bound,
            detail:
                `pre-publish deviation=${preDeviation}, bound=${mip0.bound} — the zero-init state ` +
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
            checks = await runChecks();
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
