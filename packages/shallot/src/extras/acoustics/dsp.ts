import type { AudioState } from "../../standard/audio/engine";
import { setReflectionIR, setReflectionGain, setReverb } from "../../standard/audio/engine";
import type { ReflectionState } from "./reflection";

export const MAX_SOURCES = 64;
export const NUM_BINS = 50;
export const EARLY_BINS = 8;
export const SAMPLE_RATE = 44100;
export const BIN_DURATION_S = 0.01;
export const BIN_SAMPLES = Math.floor(BIN_DURATION_S * SAMPLE_RATE);
export const SPEED_OF_SOUND = 340;
export const SPECULAR_EXPONENT = 100;
export const AIR_ABSORPTION = [0.0002, 0.0017, 0.0182] as const;
export const FDN_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] as const;
export const ALLPASS_DELAYS = [225, 341, 441, 556] as const;
export const ALLPASS_FEEDBACK = 0.5;
export const MIN_ABSORPTIVE_GAIN = 0.001;
const MAX_IR_SAMPLES = EARLY_BINS * BIN_SAMPLES;
const edcScratch = [
    new Float32Array(NUM_BINS),
    new Float32Array(NUM_BINS),
    new Float32Array(NUM_BINS),
];
const irScratch = new Float32Array(MAX_IR_SAMPLES);

const bandNoise: Float32Array[] = [];
{
    for (let band = 0; band < 3; band++) {
        const noise = new Float32Array(MAX_IR_SAMPLES);
        let seed = 0x12345678 + band * 0x9e3779b9;
        for (let i = 0; i < MAX_IR_SAMPLES; i++) {
            seed ^= seed << 13;
            seed ^= seed >>> 17;
            seed ^= seed << 5;
            noise[i] = ((seed >>> 0) / 0xffffffff) * 2 - 1;
        }
        bandNoise.push(noise);
    }

    const lpCoeff = (freq: number) => 1 - Math.exp((-2 * Math.PI * freq) / SAMPLE_RATE);
    const lp800 = lpCoeff(800);
    const lp8k = lpCoeff(8000);

    for (let band = 0; band < 3; band++) {
        const buf = bandNoise[band];
        let s800 = 0;
        let s8k = 0;
        for (let i = 0; i < buf.length; i++) {
            const raw = buf[i];
            s800 += lp800 * (raw - s800);
            s8k += lp8k * (raw - s8k);
            if (band === 0) buf[i] = s800;
            else if (band === 1) buf[i] = s8k - s800;
            else buf[i] = raw - s8k;
        }
    }
}

export function brdfDiffuse(scattering: number, cosTheta: number): number {
    return (1 / Math.PI) * scattering * cosTheta;
}

export function brdfSpecular(scattering: number, cosAlpha: number): number {
    return (
        ((SPECULAR_EXPONENT + 2) / (8 * Math.PI)) *
        (1 - scattering) *
        Math.pow(cosAlpha, SPECULAR_EXPONENT)
    );
}

export function distanceAttenuation(dist: number): number {
    const d = Math.max(dist, 1.0);
    return 1 / (d * d);
}

export function airAttenuation(coeff: number, distance: number): number {
    return Math.exp(-coeff * distance);
}

const a0 = new Float32Array(3);
const a1 = new Float32Array(3);
const invBin = 1 / BIN_SAMPLES;

export function reconstructIR(
    smoothed: Float32Array,
    base: number,
): { ir: Float32Array; energy: number } {
    for (let bin = 0; bin < EARLY_BINS; bin++) {
        const baseIdx = (base + bin) * 3;
        const nextIdx = (base + bin + 1) * 3;
        const hasNext = bin + 1 < EARLY_BINS;
        for (let band = 0; band < 3; band++) {
            a0[band] = Math.sqrt(Math.max(0, smoothed[baseIdx + band]) * invBin);
            a1[band] = hasNext ? Math.sqrt(Math.max(0, smoothed[nextIdx + band]) * invBin) : 0;
        }
        const startS = bin * BIN_SAMPLES;
        const endS = startS + BIN_SAMPLES;
        for (let s = startS; s < endS; s++) {
            const frac = (s - startS) * invBin;
            const oneMinusFrac = 1 - frac;
            irScratch[s] =
                (a0[0] * oneMinusFrac + a1[0] * frac) * bandNoise[0][s] +
                (a0[1] * oneMinusFrac + a1[1] * frac) * bandNoise[1][s] +
                (a0[2] * oneMinusFrac + a1[2] * frac) * bandNoise[2][s];
        }
    }
    let energy = 0;
    for (let s = 0; s < MAX_IR_SAMPLES; s++) {
        energy += irScratch[s] * irScratch[s];
    }
    if (energy > 1e-10) {
        const scale = 1 / Math.sqrt(energy);
        for (let s = 0; s < MAX_IR_SAMPLES; s++) {
            irScratch[s] *= scale;
        }
    }
    return { ir: irScratch, energy };
}

export function fitRT60(histogram: Float32Array, airAbsorptionCoeff = 0): number {
    const numBins = histogram.length;

    let totalEnergy = 0;
    let x = 0;
    for (let i = 0; i < numBins; i++) {
        totalEnergy += histogram[i] * Math.exp(-airAbsorptionCoeff * x);
        x += BIN_DURATION_S;
    }
    if (totalEnergy < 1e-4) return 0;

    let energy = 0;
    let sumX = 0,
        sumY = 0,
        sumXY = 0,
        sumXX = 0,
        n = 0;

    for (let i = numBins - 1; i >= 0; i--) {
        energy += histogram[i] * Math.exp(-airAbsorptionCoeff * x);
        const y = Math.log10(energy / totalEnergy);

        if (y >= -2.5 && y <= -0.5) {
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumXY += x * y;
            n++;
        }

        x -= BIN_DURATION_S;
    }

    const numerator = n * sumXY - sumX * sumY;
    const denominator = n * sumXX - sumX * sumX;
    if (Math.abs(numerator) < 1e-30) return 0;
    const rt60 = -6 * (denominator / numerator);
    return Math.max(0, Math.min(rt60, 20.0));
}

export function processHistogram(
    refl: ReflectionState,
    sourceCount: number,
    slots: Uint32Array,
    audio: AudioState,
) {
    const raw = refl.histogram;
    const rawLen = sourceCount * NUM_BINS * 3;

    const alpha = 0.3;
    for (let i = 0; i < rawLen; i++) {
        refl.smoothed[i] += alpha * (raw[i] - refl.smoothed[i]);
    }

    let totalLateEnergy = 0;
    let totalEarlyEnergy = 0;

    for (let si = 0; si < sourceCount; si++) {
        const base = si * NUM_BINS;
        const slot = slots[si];
        const { ir, energy } = reconstructIR(refl.smoothed, base);
        setReflectionIR(audio, slot, ir);
        const reflGain = Math.min(1.0, Math.sqrt(energy));
        setReflectionGain(audio, slot, reflGain);

        for (let b = 0; b < EARLY_BINS; b++) {
            for (let band = 0; band < 3; band++) {
                totalEarlyEnergy += refl.smoothed[(base + b) * 3 + band];
            }
        }
        for (let b = EARLY_BINS; b < NUM_BINS; b++) {
            for (let band = 0; band < 3; band++) {
                totalLateEnergy += refl.smoothed[(base + b) * 3 + band];
            }
        }
    }

    const rt60Out: [number, number, number] = [0.1, 0.1, 0.1];
    let wetGainOut = 0.0;

    if (totalLateEnergy > 1e-8 && sourceCount > 0) {
        for (let band = 0; band < 3; band++) {
            const histogram = edcScratch[band];
            histogram.fill(0);
            for (let si = 0; si < sourceCount; si++) {
                const base = si * NUM_BINS;
                for (let b = 0; b < NUM_BINS; b++) {
                    histogram[b] += refl.smoothed[(base + b) * 3 + band];
                }
            }
            rt60Out[band] = Math.max(0.1, fitRT60(histogram, AIR_ABSORPTION[band]));
        }
        const totalEnergy = totalEarlyEnergy + totalLateEnergy;
        wetGainOut = Math.min(2.0, (totalLateEnergy / totalEnergy) * 2.0);
    }

    const eqOut: [number, number, number] = [1, 1, 1];
    let maxEq = 0;
    for (let b = 0; b < 3; b++) {
        eqOut[b] = Math.sqrt(1 / Math.max(0.01, rt60Out[b]));
        if (eqOut[b] > maxEq) maxEq = eqOut[b];
    }
    if (maxEq > 0) {
        for (let b = 0; b < 3; b++) eqOut[b] /= maxEq;
    }

    setReverb(audio, rt60Out, wetGainOut, eqOut);
}
