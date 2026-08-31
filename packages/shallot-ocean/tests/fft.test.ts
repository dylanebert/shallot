// Hardware-invariant CPU logic tier (`testing.md`'s "Logic — bun test"): the butterfly FFT (`fft.ts`)
// against the direct O(N²) DFT it replaces, and the CPU/GPU shared cascade config sanity (power-of-two
// N and reported patch period). No device.
import { describe, expect, test } from "bun:test";
import { directIdft2, fft1dInPlace, ifft2 } from "../src/fft";
import {
    assertAllPowerOfTwo,
    assertCoprimeL,
    CASCADE_CONFIGS,
    generateH0,
    isPowerOfTwo,
    tilePeriod,
} from "../src/spectrum";

/** deterministic PRNG (mulberry32) — reproducible test fixtures, no external dependency. */
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

function randomComplex(N: number, seed: number): Float32Array {
    const rand = mulberry32(seed);
    const out = new Float32Array(N * N * 2);
    for (let i = 0; i < out.length; i++) out[i] = rand() * 2 - 1;
    return out;
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
    return m;
}

describe("fft1dInPlace — single mode, analytic check", () => {
    test("a single-frequency impulse round-trips to the exact cosine/sine basis (N=64)", () => {
        const N = 64;
        const re = new Float64Array(N);
        const im = new Float64Array(N);
        re[3] = 1; // energy at mode k=3, real=1 imag=0
        fft1dInPlace(re, im, N, 1);
        // expected: out[n] = exp(i*2*pi*3*n/N) — re[n]=cos, im[n]=sin
        for (let n = 0; n < N; n++) {
            const angle = (2 * Math.PI * 3 * n) / N;
            expect(re[n]).toBeCloseTo(Math.cos(angle), 6);
            expect(im[n]).toBeCloseTo(Math.sin(angle), 6);
        }
    });
});

describe("ifft2 vs directIdft2 — the FFT swap's own correctness gate", () => {
    for (const N of [16, 64, 128]) {
        test(`agrees with the direct O(N²) inverse DFT at N=${N} (random complex field)`, () => {
            const input = randomComplex(N, 0xc0ffee + N);
            const viaFft = ifft2(input, N);
            const viaDft = directIdft2(input, N);
            const diff = maxAbsDiff(viaFft, viaDft);
            // both accumulate in f64 internally and only round to f32 at the end — the residual is
            // pure f32-rounding noise, not truncation error, so the bound is tight.
            expect(diff).toBeLessThan(1e-3);
        });
    }

    test("agrees on a real production H0 draw (cascade 0, N=64)", () => {
        const cfg = CASCADE_CONFIGS[0];
        const h0 = generateH0(cfg);
        const viaFft = ifft2(h0, cfg.N);
        const viaDft = directIdft2(h0, cfg.N);
        expect(maxAbsDiff(viaFft, viaDft)).toBeLessThan(1e-3);
    });

    test("agrees on a real production H0 draw (cascade 1, N=128)", () => {
        const cfg = CASCADE_CONFIGS[1];
        const h0 = generateH0(cfg);
        const viaFft = ifft2(h0, cfg.N);
        const viaDft = directIdft2(h0, cfg.N);
        expect(maxAbsDiff(viaFft, viaDft)).toBeLessThan(1e-3);
    });
});

describe("cascade config — I1's own structural invariants", () => {
    test("isPowerOfTwo classifies correctly", () => {
        expect(isPowerOfTwo(1)).toBe(true);
        expect(isPowerOfTwo(64)).toBe(true);
        expect(isPowerOfTwo(128)).toBe(true);
        expect(isPowerOfTwo(81)).toBe(false);
        expect(isPowerOfTwo(0)).toBe(false);
        expect(isPowerOfTwo(-4)).toBe(false);
    });

    test("every shipped cascade N is a power of two — the FFT's own precondition", () => {
        expect(assertAllPowerOfTwo(CASCADE_CONFIGS)).toBe(true);
        for (const c of CASCADE_CONFIGS) expect(isPowerOfTwo(c.N)).toBe(true);
    });

    test("assertCoprimeL classifies coprime and non-coprime patch lengths", () => {
        expect(
            assertCoprimeL([
                { ...CASCADE_CONFIGS[0], L: 80 },
                { ...CASCADE_CONFIGS[1], L: 31 },
            ]),
        ).toBe(true);
        expect(
            assertCoprimeL([
                { ...CASCADE_CONFIGS[0], L: 80 },
                { ...CASCADE_CONFIGS[1], L: 40 },
            ]),
        ).toBe(false);
    });

    test("the shipped cascades use coprime world-space patch lengths", () => {
        expect(assertCoprimeL(CASCADE_CONFIGS)).toBe(true);
        expect(CASCADE_CONFIGS[0].L).toBe(80);
        expect(CASCADE_CONFIGS[1].L).toBe(31);
        expect(tilePeriod(CASCADE_CONFIGS)).toBe(2480);
    });

    test("cascade 1's declared band Nyquist headroom: N=128,L=31 covers kHi with margin", () => {
        const cfg = CASCADE_CONFIGS[1];
        const dk = (2 * Math.PI) / cfg.L;
        const kNyquist = (cfg.N / 2) * dk;
        expect(kNyquist).toBeGreaterThan(cfg.kHi);
    });
});
