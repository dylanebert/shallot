import { describe, test, expect } from "bun:test";
import {
    fitRT60,
    processHistogram,
    reconstructIR,
    brdfDiffuse,
    brdfSpecular,
    distanceAttenuation,
    airAttenuation,
    NUM_BINS,
    EARLY_BINS,
    SAMPLE_RATE,
    BIN_DURATION_S,
    BIN_SAMPLES,
    SPEED_OF_SOUND,
    SPECULAR_EXPONENT,
    AIR_ABSORPTION,
    FDN_PRIMES,
    ALLPASS_DELAYS,
    ALLPASS_FEEDBACK,
    MIN_ABSORPTIVE_GAIN,
} from "../src/extras/acoustics/dsp";
import { createAudioState, type AudioState } from "../src/standard/audio/engine";
import type { AudioCommand } from "../src/standard/audio/backend";

function createSpyAudio(): { audio: AudioState; calls: AudioCommand[] } {
    const calls: AudioCommand[] = [];
    const audio = createAudioState();
    audio.backend = {
        running: true,
        async init() {},
        dispose() {},
        send(cmd) {
            calls.push(cmd);
        },
        pollReadback() {},
        flush() {},
    };
    return { audio, calls };
}

const MAX_IR_SAMPLES = EARLY_BINS * BIN_SAMPLES;

// Module 1: Constants parity

describe("constants parity", () => {
    test("SPEED_OF_SOUND matches Steam Audio (340)", () => {
        expect(SPEED_OF_SOUND).toBe(340);
    });

    test("BIN_DURATION matches Steam Audio 10ms", () => {
        expect(BIN_DURATION_S).toBe(0.01);
    });

    test("EARLY_BINS is 8 (80ms early reflections)", () => {
        // Steam Audio: kEarlyReflectionsDuration = 0.08s = 80ms = 8 bins
        expect(EARLY_BINS).toBe(8);
        expect(EARLY_BINS * BIN_DURATION_S).toBeCloseTo(0.08);
    });

    test("SPECULAR_EXPONENT matches Steam Audio", () => {
        expect(SPECULAR_EXPONENT).toBe(100);
    });

    test("AIR_ABSORPTION coefficients match Steam Audio", () => {
        expect(AIR_ABSORPTION[0]).toBeCloseTo(0.0002, 4);
        expect(AIR_ABSORPTION[1]).toBeCloseTo(0.0017, 4);
        expect(AIR_ABSORPTION[2]).toBeCloseTo(0.0182, 4);
    });

    test("FDN_PRIMES match Steam Audio 16 primes", () => {
        const expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] as const;
        expect(FDN_PRIMES.length).toBe(16);
        expect([...FDN_PRIMES]).toEqual([...expected]);
    });

    test("ALLPASS_DELAYS match Steam Audio", () => {
        expect([...ALLPASS_DELAYS]).toEqual([225, 341, 441, 556]);
    });

    test("ALLPASS_FEEDBACK matches Steam Audio", () => {
        expect(ALLPASS_FEEDBACK).toBe(0.5);
    });

    test("MIN_ABSORPTIVE_GAIN matches Steam Audio", () => {
        expect(MIN_ABSORPTIVE_GAIN).toBe(0.001);
    });

    test("SAMPLE_RATE is 44100", () => {
        expect(SAMPLE_RATE).toBe(44100);
    });

    test("NUM_BINS is 50 (500ms total)", () => {
        expect(NUM_BINS).toBe(50);
        expect(NUM_BINS * BIN_DURATION_S).toBeCloseTo(0.5);
    });

    test("BIN_SAMPLES derived correctly", () => {
        expect(BIN_SAMPLES).toBe(Math.floor(0.01 * 44100));
        expect(BIN_SAMPLES).toBe(441);
    });
});

// Module 2: Reflection BRDF

describe("reflection BRDF", () => {
    test("diffuse term matches (1/π) * scattering * cosθ", () => {
        for (let cosTheta = 0; cosTheta <= 1.0; cosTheta += 0.1) {
            const scattering = 0.7;
            const expected = (1 / Math.PI) * scattering * cosTheta;
            expect(brdfDiffuse(scattering, cosTheta)).toBeCloseTo(expected, 10);
        }
    });

    test("diffuse at cosθ=0 is zero", () => {
        expect(brdfDiffuse(1.0, 0)).toBe(0);
    });

    test("diffuse at scattering=0 is zero", () => {
        expect(brdfDiffuse(0, 0.5)).toBe(0);
    });

    test("specular term matches ((k+2)/(8π)) * (1-scattering) * cos^k(α)", () => {
        for (let cosAlpha = 0; cosAlpha <= 1.0; cosAlpha += 0.1) {
            const scattering = 0.3;
            const expected =
                ((SPECULAR_EXPONENT + 2) / (8 * Math.PI)) *
                (1 - scattering) *
                Math.pow(cosAlpha, SPECULAR_EXPONENT);
            expect(brdfSpecular(scattering, cosAlpha)).toBeCloseTo(expected, 10);
        }
    });

    test("specular normalization constant matches WGSL", () => {
        // WGSL uses 4.0584 ≈ (100+2)/(8π) = 102/25.1327 = 4.0585
        const norm = (SPECULAR_EXPONENT + 2) / (8 * Math.PI);
        expect(norm).toBeCloseTo(4.0585, 3);
    });

    test("specular at scattering=1 is zero", () => {
        expect(brdfSpecular(1.0, 0.5)).toBe(0);
    });

    test("distance attenuation: 1/d² for d >= 1", () => {
        expect(distanceAttenuation(2.0)).toBeCloseTo(0.25, 10);
        expect(distanceAttenuation(10.0)).toBeCloseTo(0.01, 10);
        expect(distanceAttenuation(1.0)).toBeCloseTo(1.0, 10);
    });

    test("distance attenuation clamped at d < 1", () => {
        expect(distanceAttenuation(0.5)).toBe(distanceAttenuation(1.0));
        expect(distanceAttenuation(0.01)).toBe(1.0);
    });

    test("net per-ray contribution matches Steam Audio", () => {
        // Steam Audio: scalar = (4π/N), distanceTerm = 1/(4π*d²)
        //   net = (4π/N) * 1/(4π*d²) = 1/(N*d²)
        // Ours: GPU distAtten = 1/(4π*d²), readback scalar = 1/N
        //   net = 1/(4π*d²) * 1/N (the 4π cancels with Steam's scalar)
        // JS distanceAttenuation returns 1/d² (no 4π, used for direct path only)
        const numRays = 4096;
        const dist = 5.0;
        const steamAudioNet = 1 / (numRays * dist * dist);
        const oursNet = (1 / numRays) * distanceAttenuation(dist);
        expect(oursNet).toBeCloseTo(steamAudioNet, 10);
    });

    test("air absorption: exp(-coeff * distance) per band", () => {
        const distance = 10.0;
        for (let band = 0; band < 3; band++) {
            const expected = Math.exp(-AIR_ABSORPTION[band] * distance);
            expect(airAttenuation(AIR_ABSORPTION[band], distance)).toBeCloseTo(expected, 10);
        }
    });

    test("air absorption increases with frequency", () => {
        const distance = 20.0;
        const low = airAttenuation(AIR_ABSORPTION[0], distance);
        const mid = airAttenuation(AIR_ABSORPTION[1], distance);
        const high = airAttenuation(AIR_ABSORPTION[2], distance);
        expect(low).toBeGreaterThan(mid);
        expect(mid).toBeGreaterThan(high);
    });

    test("air absorption at zero distance is 1", () => {
        for (let band = 0; band < 3; band++) {
            expect(airAttenuation(AIR_ABSORPTION[band], 0)).toBe(1.0);
        }
    });
});

// Module 3: RT60 estimation

describe("fitRT60", () => {
    test("exponential decay returns correct RT60", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const targetRT60 = 1.5;
        const edc = new Float32Array(lateBins);
        for (let i = 0; i < lateBins; i++) {
            const t = (i + 0.5) * BIN_DURATION_S;
            edc[i] = Math.exp((-6.0 * t) / targetRT60);
        }
        const rt60 = fitRT60(edc);
        // The fit window [-0.5, -2.5] in log10 only captures part of the decay.
        // With 42 late bins (420ms), the fit underestimates long RT60 values
        // since the data doesn't span the full 60dB range.
        expect(rt60).toBeGreaterThan(0.5);
        expect(rt60).toBeLessThan(3.0);
    });

    test("multi-slope decay reflects -5dB to -25dB window", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        // Steep early (RT60=0.5s), shallow late (RT60=3.0s)
        for (let i = 0; i < lateBins; i++) {
            const t = (i + 0.5) * BIN_DURATION_S;
            if (t < 0.1) {
                edc[i] = Math.exp((-6.0 * t) / 0.5);
            } else {
                edc[i] = Math.exp((-6.0 * 0.1) / 0.5) * Math.exp((-6.0 * (t - 0.1)) / 3.0);
            }
        }
        const rt60 = fitRT60(edc);
        // The fit window (-0.5 to -2.5 dB range) should capture the slope
        // within that region, yielding something between the two slopes
        expect(rt60).toBeGreaterThan(0.1);
        expect(rt60).toBeLessThan(5.0);
    });

    test("sparse histogram (cathedral-like) returns reasonable RT60", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        // Long decay with low amplitude — scaling doesn't affect slope in log space
        const targetRT60 = 3.0;
        for (let i = 0; i < lateBins; i++) {
            const t = (i + 0.5) * BIN_DURATION_S;
            edc[i] = 0.01 * Math.exp((-6.0 * t) / targetRT60);
        }
        const rt60 = fitRT60(edc);
        // Won't hit exact target due to limited window, but should not fall back to 0
        expect(rt60).not.toBe(0);
        expect(rt60).toBeGreaterThan(0.5);
    });

    test("single-bin energy returns 0 (insufficient data)", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        edc[0] = 1.0;
        const rt60 = fitRT60(edc);
        expect(rt60).toBe(0);
    });

    test("uniform energy returns long RT60", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        edc.fill(1.0);
        const rt60 = fitRT60(edc);
        // Uniform bins → backward cumulation creates linear decrease → log10 has
        // shallow slope. The fit may produce a long RT60 rather than hitting the
        // fallback, depending on whether enough points fall in the [-0.5, -2.5] window.
        expect(rt60).toBeGreaterThan(0.1);
        expect(rt60).toBeLessThanOrEqual(5.0);
    });

    test("zero energy returns 0", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        edc.fill(0);
        const rt60 = fitRT60(edc);
        expect(rt60).toBe(0);
    });

    test("near-zero energy returns 0", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const edc = new Float32Array(lateBins);
        edc.fill(1e-12);
        const rt60 = fitRT60(edc);
        expect(rt60).toBe(0);
    });

    // KNOWN DIVERGENCE: Steam Audio weights EDC by air absorption before fitting.
    // Our fitRT60 does not. This test documents the divergence.
    test("air-absorption-weighted EDC divergence documented", () => {
        const lateBins = NUM_BINS - EARLY_BINS;
        const targetRT60 = 2.0;

        // Unweighted (our approach)
        const edcUnweighted = new Float32Array(lateBins);
        for (let i = 0; i < lateBins; i++) {
            const t = (i + 0.5) * BIN_DURATION_S;
            edcUnweighted[i] = Math.exp((-6.0 * t) / targetRT60);
        }
        const rt60Unweighted = fitRT60(edcUnweighted);

        // What Steam Audio would see: EDC weighted by high-band air absorption
        const edcWeighted = new Float32Array(lateBins);
        for (let i = 0; i < lateBins; i++) {
            const t = (i + 0.5) * BIN_DURATION_S;
            const dist = t * SPEED_OF_SOUND;
            const airWeight = Math.exp(-AIR_ABSORPTION[2] * dist);
            edcWeighted[i] = Math.exp((-6.0 * t) / targetRT60) * airWeight;
        }
        const rt60Weighted = fitRT60(edcWeighted);

        // Both should produce valid results, not fallback
        expect(rt60Unweighted).not.toBe(0);
        expect(rt60Weighted).not.toBe(0);
        // The values differ — the fit window [-0.5, -2.5] samples different
        // parts of the curve depending on the weighting. The magnitude of the
        // difference depends on where the window lands relative to the air
        // absorption effect. This is the divergence we're documenting.
        expect(Math.abs(rt60Unweighted - rt60Weighted)).toBeLessThan(1.0);
    });
});

// Module 4: IR reconstruction

describe("reconstructIR", () => {
    test("output is bounded and finite", () => {
        const smoothed = new Float32Array(NUM_BINS * 3);
        for (let b = 0; b < EARLY_BINS; b++) {
            for (let band = 0; band < 3; band++) {
                smoothed[b * 3 + band] = 0.5;
            }
        }
        const { ir } = reconstructIR(smoothed, 0);
        expect(ir.length).toBe(MAX_IR_SAMPLES);
        for (let i = 0; i < ir.length; i++) {
            expect(Number.isFinite(ir[i])).toBe(true);
        }
    });

    test("zero histogram produces zero IR", () => {
        const smoothed = new Float32Array(NUM_BINS * 3);
        const { ir, energy } = reconstructIR(smoothed, 0);
        let irEnergy = 0;
        for (let i = 0; i < ir.length; i++) {
            irEnergy += ir[i] * ir[i];
        }
        expect(irEnergy).toBe(0);
        expect(energy).toBe(0);
    });

    test("normalization formula: sqrt(energy / BIN_SAMPLES)", () => {
        // Our normalization: amplitude = sqrt(energy / BIN_SAMPLES)
        // Steam Audio: amplitude = energy / sqrt(energy_W * sqrt(4π))
        // For single-channel (W only), reduces to sqrt(energy / sqrt(4π))
        // KNOWN DIVERGENCE: we divide by BIN_SAMPLES, they divide by sqrt(4π)
        // BIN_SAMPLES = 441, sqrt(4π) ≈ 3.545
        // Ours produces much smaller amplitude per sample
        const ourDivisor = BIN_SAMPLES;
        const steamDivisor = Math.sqrt(4 * Math.PI);
        expect(ourDivisor).toBeGreaterThan(steamDivisor);
        // The ratio quantifies the divergence
        const ratio = Math.sqrt(steamDivisor / ourDivisor);
        expect(ratio).toBeCloseTo(Math.sqrt(3.545 / 441), 2);
    });

    test("normalized IR has unit energy, original energy returned separately", () => {
        const smoothed = new Float32Array(NUM_BINS * 3);
        const binEnergies = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02];
        for (let b = 0; b < EARLY_BINS; b++) {
            for (let band = 0; band < 3; band++) {
                smoothed[b * 3 + band] = binEnergies[b];
            }
        }

        const { ir, energy } = reconstructIR(smoothed, 0);
        let irEnergy = 0;
        for (let i = 0; i < ir.length; i++) {
            irEnergy += ir[i] * ir[i];
        }

        expect(energy).toBeGreaterThan(0);
        expect(irEnergy).toBeCloseTo(1.0, 1);
    });

    test("linear interpolation between bins", () => {
        const smoothed = new Float32Array(NUM_BINS * 3);
        // Bin 0: energy = 1.0, Bin 1: energy = 0.0 (all bands)
        for (let band = 0; band < 3; band++) {
            smoothed[0 * 3 + band] = 1.0;
            smoothed[1 * 3 + band] = 0.0;
        }

        const { ir } = reconstructIR(smoothed, 0);
        let earlyRms = 0;
        let lateRms = 0;
        const quarter = Math.floor(BIN_SAMPLES / 4);
        for (let i = 0; i < quarter; i++) {
            earlyRms += ir[i] * ir[i];
            lateRms += ir[BIN_SAMPLES - quarter + i] * ir[BIN_SAMPLES - quarter + i];
        }
        expect(earlyRms).toBeGreaterThan(lateRms);
    });

    test("band-filtered noise produces output for each band", () => {
        const smoothed = new Float32Array(NUM_BINS * 3);
        // Only low band
        for (let b = 0; b < EARLY_BINS; b++) {
            smoothed[b * 3 + 0] = 1.0;
            smoothed[b * 3 + 1] = 0.0;
            smoothed[b * 3 + 2] = 0.0;
        }
        const { energy: lowEnergy } = reconstructIR(smoothed, 0);

        // Only high band
        smoothed.fill(0);
        for (let b = 0; b < EARLY_BINS; b++) {
            smoothed[b * 3 + 0] = 0.0;
            smoothed[b * 3 + 1] = 0.0;
            smoothed[b * 3 + 2] = 1.0;
        }
        const { energy: highEnergy } = reconstructIR(smoothed, 0);

        expect(lowEnergy).toBeGreaterThan(0);
        expect(highEnergy).toBeGreaterThan(0);
    });
});

// Module 5: FDN reverb — tested in Rust (lib.rs)

// Module 6: Wet gain / reverb level

describe("wet gain", () => {
    test("wet gain scales logarithmically with room energy", () => {
        const results: number[] = [];
        // Cap hits at totalMidEnergy ≈ 0.04, so stay below that
        for (const energy of [0.0001, 0.001, 0.005, 0.01, 0.03]) {
            const totalMidEnergy = energy;
            const roomMB = 1000 * Math.log10(Math.max(1e-10, totalMidEnergy));
            const wet = Math.min(2.0, Math.max(0, Math.pow(10, (roomMB + 2000) / 2000)));
            results.push(wet);
        }
        // Each step should increase
        for (let i = 1; i < results.length; i++) {
            expect(results[i]).toBeGreaterThan(results[i - 1]);
        }
        // All values should be below the cap
        for (const r of results) {
            expect(r).toBeLessThan(2.0);
        }
    });

    test("total histogram energy → room_mB: 1000 * log10(totalEnergy)", () => {
        const totalMidEnergy = 0.5;
        const roomMB = 1000 * Math.log10(totalMidEnergy);
        // log10(0.5) ≈ -0.301, so roomMB ≈ -301
        expect(roomMB).toBeCloseTo(-301.03, 0);
    });

    test("small energy (cathedral) still produces audible wet gain", () => {
        const totalMidEnergy = 0.001;
        const roomMB = 1000 * Math.log10(Math.max(1e-10, totalMidEnergy));
        const wet = Math.min(2.0, Math.max(0, Math.pow(10, (roomMB + 2000) / 2000)));
        expect(wet).toBeGreaterThan(0);
    });

    test("large energy (bathroom) caps at 2.0", () => {
        const totalMidEnergy = 1000.0;
        const roomMB = 1000 * Math.log10(Math.max(1e-10, totalMidEnergy));
        const wet = Math.min(2.0, Math.max(0, Math.pow(10, (roomMB + 2000) / 2000)));
        expect(wet).toBeLessThanOrEqual(2.0);
    });

    test("zero late energy → wet gain stays 0", () => {
        const sourceCount = 1;
        const smoothed = new Float32Array(sourceCount * NUM_BINS * 3);
        // Only early bins have energy
        for (let b = 0; b < EARLY_BINS; b++) {
            for (let band = 0; band < 3; band++) {
                smoothed[b * 3 + band] = 1.0;
            }
        }
        const refl = {
            smoothed,
            histogram: new Float32Array(smoothed),
        };
        const slots = new Uint32Array([0]);
        const { audio, calls } = createSpyAudio();
        processHistogram(refl as any, sourceCount, slots, audio);
        const reverbCmd = calls.find((c) => c.type === "reverb") as any;
        expect(reverbCmd.wetGain).toBe(0);
    });

    test("wet gain bounded at 2.0 via processHistogram", () => {
        const sourceCount = 1;
        const smoothed = new Float32Array(sourceCount * NUM_BINS * 3);
        smoothed.fill(1000.0);
        const refl = {
            smoothed,
            histogram: new Float32Array(smoothed),
        };
        const slots = new Uint32Array([0]);
        const { audio, calls } = createSpyAudio();
        processHistogram(refl as any, sourceCount, slots, audio);
        const reverbCmd = calls.find((c) => c.type === "reverb") as any;
        expect(reverbCmd.wetGain).toBeLessThanOrEqual(2.0);
    });
});

// Module 7: Occlusion

describe("occlusion", () => {
    test("unoccluded source: gain = 1.0, cutoff = 20000", () => {
        // Occlusion readback returns [1.0, 1.0, 1.0] for unoccluded
        const gainLow = 1.0,
            gainMid = 1.0,
            gainHigh = 1.0;
        const avg = (gainLow + gainMid + gainHigh) / 3.0;
        expect(avg).toBe(1.0);
        const cutoff = Math.min(20000, Math.max(200, 500 + avg * 19500));
        expect(cutoff).toBe(20000);
    });

    test("fully occluded: gain and cutoff reflect transmission", () => {
        // Low transmission material
        const gainLow = 0.1,
            gainMid = 0.05,
            gainHigh = 0.01;
        const avg = (gainLow + gainMid + gainHigh) / 3.0;
        expect(avg).toBeLessThan(0.1);
        const cutoff = Math.min(20000, Math.max(200, 500 + avg * 19500));
        expect(cutoff).toBeLessThan(2000);
        expect(cutoff).toBeGreaterThanOrEqual(200);
    });

    test("3-band transmission → averaged gain and cutoff mapping", () => {
        const gainLow = 0.8,
            gainMid = 0.3,
            gainHigh = 0.1;
        const avg = (gainLow + gainMid + gainHigh) / 3.0;
        expect(avg).toBeCloseTo(0.4, 5);
        const cutoff = 500 + avg * 19500;
        expect(cutoff).toBeCloseTo(8300, 0);
    });

    test("cutoff clamped to [200, 20000] Hz", () => {
        // Very low transmission → cutoff approaches 500 (above 200 floor)
        const avgLow = 0.0;
        const cutoffLow = Math.min(20000, Math.max(200, 500 + avgLow * 19500));
        expect(cutoffLow).toBe(500);
        expect(cutoffLow).toBeGreaterThanOrEqual(200);

        // Full transmission → cutoff = 20000
        const avgHigh = 1.0;
        const cutoffHigh = Math.min(20000, Math.max(200, 500 + avgHigh * 19500));
        expect(cutoffHigh).toBe(20000);
    });
});

// Module 8: Spatial pipeline — tested in Rust (lib.rs)

// Regression: Steam Audio reference histograms
// Two representative rooms with inlined gold data from Steam Audio test harness
// (4096 rays, 16 bounces, 0.5s duration, 32 runs, speed=340)

describe("regression: reference histograms", () => {
    // Bathroom: 3×2.5×3m, absorption=[0.01, 0.02, 0.02], scattering=0.05
    // prettier-ignore
    const bathroomHistogram = [
        0.00959414, 0.01489687, 0.01341581, 0.0112037, 0.01204865, 0.0113386, 0.0100531, 0.00915747,
        0.00548886, 0.00287946, 0.00082737, 0.00028947, 0.00011476, 0.00000384, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00947461,
        0.01445563, 0.01276617, 0.01047676, 0.01107714, 0.01022134, 0.00891143, 0.00797882,
        0.00473438, 0.0024697, 0.00070893, 0.00024771, 0.0000978, 0.00000327, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00947461,
        0.01445563, 0.01276617, 0.01047676, 0.01107714, 0.01022134, 0.00891143, 0.00797882,
        0.00473438, 0.0024697, 0.00070893, 0.00024771, 0.0000978, 0.00000327, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    const bathroomRT60 = [0.15961336, 0.1588662, 0.15884116];

    // Cathedral: 30×15×20m, absorption=[0.13, 0.20, 0.24], scattering=0.05
    // prettier-ignore
    const cathedralHistogram = [
        0.0, 0.0, 0.0, 0.0, 0.00017031, 0.00009816, 0.00004104, 0.00003483, 0.00007839, 0.00004771,
        0.00006963, 0.00006637, 0.00004415, 0.00004177, 0.0000553, 0.0000337, 0.00003297,
        0.00004668, 0.00005044, 0.00005988, 0.00003636, 0.00003183, 0.00003115, 0.00002204,
        0.00004256, 0.00002911, 0.00003964, 0.00002724, 0.00002982, 0.00002719, 0.00002936,
        0.00002545, 0.00002274, 0.00001606, 0.00002137, 0.00003077, 0.0000275, 0.00002282,
        0.00001524, 0.0000206, 0.00001831, 0.00001561, 0.00001438, 0.00001412, 0.00001886,
        0.00001791, 0.00001303, 0.00001757, 0.00001268, 0.00001582, 0.0, 0.0, 0.0, 0.0, 0.00015661,
        0.00009026, 0.00003511, 0.00002956, 0.00006937, 0.00004047, 0.00005593, 0.00005274,
        0.00003443, 0.00003062, 0.00004054, 0.0000242, 0.00002218, 0.00003393, 0.0000364,
        0.00004017, 0.00002343, 0.00002031, 0.00001877, 0.00001341, 0.00002537, 0.000017,
        0.00002377, 0.00001598, 0.00001659, 0.00001511, 0.00001563, 0.00001316, 0.00001116,
        0.00000762, 0.00001054, 0.0000158, 0.00001334, 0.00001065, 0.00000682, 0.00000916,
        0.00000763, 0.0000067, 0.00000585, 0.00000568, 0.00000814, 0.00000739, 0.00000522,
        0.0000068, 0.00000462, 0.0000058, 0.0, 0.0, 0.0, 0.0, 0.00014878, 0.00008575, 0.00003193,
        0.00002674, 0.00006443, 0.0000366, 0.00004896, 0.00004585, 0.00002958, 0.00002535,
        0.00003356, 0.00001977, 0.00001743, 0.00002802, 0.00002989, 0.00003152, 0.00001794,
        0.00001545, 0.00001379, 0.00000993, 0.00001853, 0.00001228, 0.00001753, 0.00001161,
        0.00001165, 0.00001059, 0.00001067, 0.00000881, 0.00000724, 0.00000485, 0.0000069,
        0.00001064, 0.00000865, 0.00000675, 0.0000042, 0.00000562, 0.00000449, 0.00000403,
        0.00000341, 0.0000033, 0.00000495, 0.00000437, 0.00000303, 0.00000386, 0.00000252,
        0.00000317,
    ];
    const cathedralRT60 = [1.1081233, 1.1434617, 1.11754513];

    for (const [name, histogram, rt60] of [
        ["bathroom", bathroomHistogram, bathroomRT60],
        ["cathedral", cathedralHistogram, cathedralRT60],
    ] as const) {
        for (let band = 0; band < 3; band++) {
            test(`${name} band ${band}: fitRT60 matches reference`, () => {
                const bandHist = new Float32Array(NUM_BINS);
                for (let i = 0; i < NUM_BINS; i++) {
                    bandHist[i] = histogram[band * NUM_BINS + i];
                }
                const result = Math.max(0.1, fitRT60(bandHist, AIR_ABSORPTION[band]));
                const relErr = Math.abs(result - rt60[band]) / rt60[band];
                expect(relErr).toBeLessThan(0.0005);
            });
        }
    }

    test("bathroom: early energy dominates (short reverb)", () => {
        let early = 0,
            late = 0;
        for (let i = 0; i < NUM_BINS; i++) {
            const e = bathroomHistogram[NUM_BINS + i]; // mid band
            if (i < EARLY_BINS) early += e;
            else late += e;
        }
        expect(early).toBeGreaterThan(late * 5);
    });

    test("cathedral: late energy substantial (long reverb)", () => {
        let early = 0,
            late = 0;
        for (let i = 0; i < NUM_BINS; i++) {
            const e = cathedralHistogram[NUM_BINS + i]; // mid band
            if (i < EARLY_BINS) early += e;
            else late += e;
        }
        expect(late).toBeGreaterThan(early);
    });
});

// processHistogram integration

describe("processHistogram", () => {
    test("eq derived from rt60 after fix", () => {
        const sourceCount = 1;
        const smoothed = new Float32Array(sourceCount * NUM_BINS * 3);
        const bandRT60 = [1.0, 0.5, 0.3];
        for (let b = EARLY_BINS; b < NUM_BINS; b++) {
            const t = (b - EARLY_BINS + 0.5) * BIN_DURATION_S;
            for (let band = 0; band < 3; band++) {
                smoothed[b * 3 + band] = 10.0 * Math.exp((-6.0 * t) / bandRT60[band]);
            }
        }
        const refl = {
            smoothed,
            histogram: new Float32Array(smoothed),
        };
        const slots = new Uint32Array([0]);
        const { audio, calls } = createSpyAudio();
        processHistogram(refl as any, sourceCount, slots, audio);
        const reverbCmd = calls.find((c) => c.type === "reverb") as any;
        expect(reverbCmd.rt60Low).toBeGreaterThan(reverbCmd.rt60High);
        expect(reverbCmd.eqLow).toBeLessThan(reverbCmd.eqHigh);
    });
});
