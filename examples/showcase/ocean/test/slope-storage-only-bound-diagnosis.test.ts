// Diagnosis, re-run here as a committed CPU-only regression fixture (no GPU adapter needed —
// everything below is `generateH0`/`runSlopeCpuPipeline`/`reduceSlopeMip` plus a CPU
// `Float16Array` round trip, the same simulation `ocean-slope`'s bench scenario uses to model
// what the device's own rgba16float storage does). Two prior correction rounds each authored a
// computation-accuracy term sized to make this exact population green, and both were convicted for
// it. The ruling: the comparison sits on a representation seam (the published value IS a rounding
// of the f32 value it came from) and no numeric tolerance belongs there at all — the only real
// (computed, not stored) error lives one layer up, at the FFT's own floating-point arithmetic
// (`slope-seam.ts`'s Higham-bound claim, exercised in `slope-seam.test.ts` and the `ocean-slope`
// gym scenario). This file exists so that ruling stays checkable: it reproduces the population a
// naive storage-quantum-only bound convicts, on a frozen capture, and pins the count at exactly 8
// — never re-measured against a live device here, since a live reading would only restate the seam
// claim `slope-seam.ts` already makes properly. The last test below is a diagnostic PRINT, at
// every level (no assertion anywhere in it): max/rms deviation and the f16 rounding-direction
// split for a live (non-frozen, self-consistent) CPU field — informational context for the frozen
// count above, never a second claim. The storage-seam claim itself has no CPU instance in this
// package; it is asserted exclusively on-device, in the `ocean-slope` gym scenario.
//
// FROZEN/PAYLOAD below are copied verbatim from a real WebGPU device capture taken during the
// retired I3g-r review (`scratch/shallot-water-surface/eight-probe.ts`, gitignored — not a
// runnable reference from this file, since scratch is per-session and not committed). FROZEN is
// that capture's own printed per-level/channel `deviation` reading (CPU reference vs the
// published GPU texture, unquantized reference); PAYLOAD is the same capture's per-level/channel
// GPU payload maximum. Neither is re-derived here — reproducing them is the whole test.
import { expect, test } from "bun:test";
import {
    reduceSlopeMip,
    runSlopeCpuPipeline,
    SLOPE_CASCADE_CONFIGS,
    SLOPE_MIP_LEVELS,
    slopeMipSize,
} from "../src/ocean/slope";
import { generateH0 } from "../src/ocean/spectrum";

const [config] = SLOPE_CASCADE_CONFIGS;
const CHANNEL_NAMES = ["slopeX", "slopeZ", "energy", "residual"] as const;

/** level -> channel -> the frozen device capture's own `deviation` (CPU unquantized reference
 *  minus published GPU texel, absolute value) at that level/channel. */
const FROZEN_DEVIATION: Record<number, Record<(typeof CHANNEL_NAMES)[number], number>> = {
    0: {
        slopeX: 0.00024300813674926758,
        slopeZ: 0.00024020671844482422,
        energy: 0.000061027705669403076,
        residual: 0,
    },
    1: {
        slopeX: 0.000328749418258667,
        slopeZ: 0.0001955777406692505,
        energy: 0.00008881837129592896,
        residual: 0.00004886370152235031,
    },
    2: {
        slopeX: 0.00014813989400863647,
        slopeZ: 0.00016273558139801025,
        energy: 0.00006495416164398193,
        residual: 0.000046253204345703125,
    },
    3: {
        slopeX: 0.00008302181959152222,
        slopeZ: 0.00006751716136932373,
        energy: 0.000037726014852523804,
        residual: 0.00003762170672416687,
    },
    4: {
        slopeX: 0.000019568949937820435,
        slopeZ: 0.000022061169147491455,
        energy: 0.00003363564610481262,
        residual: 0.00003502704203128815,
    },
    5: {
        slopeX: 0.000006591202691197395,
        slopeZ: 0.0000071818940341472626,
        energy: 0.00002795923501253128,
        residual: 0.000027204863727092743,
    },
    6: {
        slopeX: 0.0000020234729163348675,
        slopeZ: 0.0000024863402359187603,
        energy: 0.000028141774237155914,
        residual: 0.000031177885830402374,
    },
    7: {
        slopeX: 0.0000014032702893018723,
        slopeZ: 0.0000005801557563245296,
        energy: 0.000029403716325759888,
        residual: 0.00002925470471382141,
    },
    8: {
        slopeX: 5.962283466942608e-8,
        slopeZ: 1.192074705613777e-7,
        energy: 0.000031517818570137024,
        residual: 0.000031517818570137024,
    },
};

/** level -> channel -> the same capture's published GPU payload maximum (widened f16 -> f32). */
const FROZEN_PAYLOAD: Record<number, Record<(typeof CHANNEL_NAMES)[number], number>> = {
    0: { slopeX: 0.341552734375, slopeZ: 0.33154296875, energy: 0.1483154296875, residual: 0 },
    1: {
        slopeX: 0.256591796875,
        slopeZ: 0.2227783203125,
        energy: 0.0887451171875,
        residual: 0.03759765625,
    },
    2: {
        slopeX: 0.13720703125,
        slopeZ: 0.1417236328125,
        energy: 0.038726806640625,
        residual: 0.03753662109375,
    },
    3: {
        slopeX: 0.04595947265625,
        slopeZ: 0.05181884765625,
        energy: 0.024444580078125,
        residual: 0.0235595703125,
    },
    4: {
        slopeX: 0.01438140869140625,
        slopeZ: 0.01275634765625,
        energy: 0.0171356201171875,
        residual: 0.017120361328125,
    },
    5: {
        slopeX: 0.0038547515869140625,
        slopeZ: 0.0034961700439453125,
        energy: 0.01410675048828125,
        residual: 0.01409912109375,
    },
    6: {
        slopeX: 0.0009551048278808594,
        slopeZ: 0.0008707046508789062,
        energy: 0.0126800537109375,
        residual: 0.01267242431640625,
    },
    7: {
        slopeX: 0.0003745555877685547,
        slopeZ: 0.00016736984252929688,
        energy: 0.01213836669921875,
        residual: 0.01213836669921875,
    },
    8: {
        slopeX: 2 ** -24,
        slopeZ: 2 ** -23,
        energy: 0.0119171142578125,
        residual: 0.0119171142578125,
    },
};

/** rgba16float round-to-nearest relative quantum (10 explicit mantissa bits, half a ULP). */
const HALF_FLOAT_REL_QUANTUM = 2 ** -11;

/** The CPU mip chain, quantized at every level exactly the way the GPU's own rgba16float storage
 *  does (`quant`) or left at full f32 precision (`!quant`) — the two readings this file compares. */
function cpuSlopeLevels(quant: boolean): Float32Array[] {
    const h0 = generateH0(config, 0);
    const { xField, zField } = runSlopeCpuPipeline(h0, config, 0);
    const N = config.N;
    const level0 = new Float32Array(N * N * 4);
    for (let i = 0; i < N * N; i++) {
        const sx = xField[i * 2];
        const sz = zField[i * 2];
        level0[i * 4] = sx;
        level0[i * 4 + 1] = sz;
        level0[i * 4 + 2] = sx * sx + sz * sz;
    }
    const roundTrip = (a: Float32Array) => (quant ? Float32Array.from(new Float16Array(a)) : a);
    const levels = [roundTrip(level0)];
    for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
        levels.push(roundTrip(reduceSlopeMip(levels[level - 1], slopeMipSize(config, level - 1))));
    }
    return levels;
}

function maxAbsChannel(field: Float32Array, channel: number): number {
    let max = 0;
    for (let i = 0; i < field.length / 4; i++) {
        const v = Math.abs(field[i * 4 + channel]);
        if (v > max) max = v;
    }
    return max;
}

test("exactly eight (level, channel) pairs exceed a storage-quantum-only bound on the frozen capture", () => {
    // The naive bound both convicted I3g-r rounds started from: (level+1) round trips of
    // half-float relative quantum, scaled by that channel's own CPU-reference magnitude AT THAT
    // LEVEL — no twiddle/computation term at all. `levels(false)` gives that per-level/channel
    // scale.
    const cpuLevels = cpuSlopeLevels(false);
    const fails: string[] = [];
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        CHANNEL_NAMES.forEach((name, channel) => {
            const scale = maxAbsChannel(cpuLevels[level], channel);
            const bound = (level + 1) * HALF_FLOAT_REL_QUANTUM * scale;
            const deviation = FROZEN_DEVIATION[level][name];
            const payloadMax = FROZEN_PAYLOAD[level][name];
            // Gate law's own unverifiable partition: a payload below its own bound proves nothing.
            // `residual` at level 0 is architecturally exactly zero on BOTH sides (no sub-texel
            // content by design — the Validation bullet this stage carries), so `scale === 0`
            // here is the degenerate zero-vs-zero case, never a real comparison subject; a bound
            // of exactly zero would otherwise read `payloadMax < bound` as false and `deviation >=
            // bound` as `0 >= 0`, manufacturing a ninth "failure" out of an exact match.
            if (scale === 0) return;
            if (payloadMax < bound) return;
            if (deviation >= bound) {
                fails.push(
                    `L${level} ${name} deviation=${deviation.toExponential(3)} bound=${bound.toExponential(3)} ` +
                        `ratio=${(deviation / bound).toFixed(2)}x`,
                );
            }
        });
    }
    console.log(`storage-only-bound diagnosis: ${fails.length} of 36 (level, channel) pairs FAIL`);
    console.log(fails.join("\n"));
    expect(fails.length).toBe(8);
});

test("f32-intermediate reading: max/rms deviation and the f16 rounding-direction histogram", () => {
    // Purely a CPU simulation: the exact f32 value the GPU's own storage kernel would round from
    // (`exact`) against the SAME field's own f16 round trip (`quantized`) — no device capture, no
    // frozen table, just the two readings this whole diagnosis rests on existing independently.
    //
    // PRINT ONLY AT EVERY LEVEL, level 0 included (I3g-r2's re-verdict retired the per-element
    // assertion this block used to carry at level 0: `expect(quantized[i]).toBe(f16Round(exact[i]))`
    // is an `x === f16Round(x)` self-comparison — `quantized[0]` is built by round-tripping
    // `exact[0]` through `Float16Array`, so asserting it equals `f16Round(exact[0])` checks that
    // one JS-spec-conversion helper agrees with another JS-spec-conversion helper on the same
    // input, never a property of `slope-seam.ts`'s own math, and it is not the storage-seam claim
    // either — that claim compares a PUBLISHED (GPU-computed) texel against the f32 value it was
    // rounded from, and nothing here is GPU-computed).
    //
    // Level >= 1 was never assertable here at all: `quantizedLevels[level]` is
    // `reduceSlopeMip(quantizedLevels[level - 1], ...)` (the SELF-REFERENTIAL chain, quantizing at
    // every level, mirroring the GPU's own publish-then-read-back mip chain) while
    // `exactLevels[level]` is `reduceSlopeMip(exactLevels[level - 1], ...)` (the NEVER-quantized
    // chain) — two DIFFERENT parent inputs, not one value and its own rounding. Their divergence at
    // level >= 1 is real and compounding, not a single f16 step (this file's OWN prior test above
    // pins exactly 8 (level, channel) pairs where a naive per-level bound built from that
    // divergence fails), so max/rms deviation and the rounding-direction split are diagnostic reads
    // at every level, asserting nothing about their magnitude anywhere. The seam claim itself — a
    // published GPU texel against the f32 value it was rounded from — has no CPU instance in this
    // package; it is asserted exclusively on-device (`ocean-slope` gym scenario).
    const exactLevels = cpuSlopeLevels(false);
    const quantizedLevels = cpuSlopeLevels(true);
    for (let level = 0; level < SLOPE_MIP_LEVELS; level++) {
        const exact = exactLevels[level];
        const quantized = quantizedLevels[level];
        let maxDeviation = 0;
        let sumSquares = 0;
        let down = 0;
        let exactMatch = 0;
        let up = 0;
        const count = exact.length;
        for (let i = 0; i < count; i++) {
            const deviation = quantized[i] - exact[i];
            const absDeviation = Math.abs(deviation);
            if (absDeviation > maxDeviation) maxDeviation = absDeviation;
            sumSquares += deviation * deviation;
            if (deviation < 0) down++;
            else if (deviation > 0) up++;
            else exactMatch++;
        }
        const rms = Math.sqrt(sumSquares / count);
        console.log(
            `L${level} f32->f16 (print only — see test header): max=${maxDeviation.toExponential(3)} ` +
                `rms=${rms.toExponential(3)} rounding[down=${down}, exact=${exactMatch}, up=${up}] of ${count}`,
        );
    }
});
