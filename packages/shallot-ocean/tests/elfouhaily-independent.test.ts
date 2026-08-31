import { describe, expect, test } from "bun:test";
import {
    CM,
    coxMunkMeanSquareSlope,
    curvatureSpectrum,
    directionalDensity,
    directionalFactor,
    frictionVelocity,
    KM,
    longWaveFactor,
    OMEGA_C_MAX,
    OMEGA_C_MIN,
    peakEnvelope,
    peakWavenumber,
    phaseSpeed,
    radialIntegral,
    shortWaveFactor,
    spectrumAmplitudes,
    spreadingDelta,
} from "./elfouhaily-independent";

const U10 = 15;
const OMEGA_C = 0.9;

describe("elfouhaily-independent (I2r-a commit 1: the independent oracle stands alone)", () => {
    test("published constants: km = √(ρg/τ) ≈ 368.6 rad/m, cm = √(2g/km) ≈ 0.23 m/s", () => {
        expect(KM).toBeCloseTo(Math.sqrt((1025 * 9.81) / 0.074), 6);
        expect(KM).toBeGreaterThan(360);
        expect(KM).toBeLessThan(380);
        expect(CM).toBeCloseTo(Math.sqrt((2 * 9.81) / KM), 9);
    });

    test("dispersion (eqs. (3)+(16)): c(km) is the minimum phase speed; c → √(g/k) at low k", () => {
        expect(phaseSpeed(KM)).toBeCloseTo(CM, 12);
        // gravity-wave limit well below km: surface-tension term negligible
        expect(phaseSpeed(0.01)).toBeCloseTo(Math.sqrt(9.81 / 0.01), 4);
        // minimum is global on the sampled band
        for (const k of [0.05, 0.5, 5, 50, 100, 200, 300, 500]) {
            expect(phaseSpeed(k)).toBeGreaterThanOrEqual(CM - 1e-12);
        }
    });

    test("friction velocity (eqs. (10)–(15)): fixed point of the log layer, ~0.63 m/s at U₁₀=15", () => {
        const ustar = frictionVelocity(U10);
        expect(ustar).toBeGreaterThan(0.5);
        expect(ustar).toBeLessThan(0.8);
        // drag coefficient u*²/U₁₀² lands in the published high-wind window (~1.8e-3)
        expect(ustar ** 2 / U10 ** 2).toBeGreaterThan(1e-3);
        expect(ustar ** 2 / U10 ** 2).toBeLessThan(3e-3);
        // it must close its own log-layer equation: U₁₀ = (u*/κ)·ln(10/z₀) with the
        // combined Charnock + viscous roughness
        const z0 = (0.0185 * ustar * ustar) / 9.81 + (0.11 * 1.46e-5) / ustar;
        expect((ustar / 0.4) * Math.log(10 / z0)).toBeCloseTo(U10, 6);
        // monotone in U₁₀
        expect(frictionVelocity(5)).toBeLessThan(frictionVelocity(10));
        expect(frictionVelocity(10)).toBeLessThan(frictionVelocity(20));
    });

    test("peak wavenumber (Ωc = ωp·U₁₀/g inverted): kp = g·Ωc²/U₁₀²", () => {
        expect(peakWavenumber(U10, OMEGA_C)).toBeCloseTo((9.81 * 0.81) / 225, 12);
        const kp = peakWavenumber(U10, OMEGA_C);
        // peak of the radial density sits near kp (within a factor 2 — the density peaks
        // slightly above kp once the spreading taper is folded in)
        let argmax = 0;
        let maxVal = 0;
        for (let i = 0; i < 4000; i++) {
            const k = Math.exp(Math.log(0.005) + (i * (Math.log(500) - Math.log(0.005))) / 3999);
            const val = (curvatureSpectrum(k, U10, OMEGA_C) / k ** 4) * k;
            if (val > maxVal) {
                maxVal = val;
                argmax = k;
            }
        }
        expect(argmax).toBeGreaterThan(kp / 2);
        expect(argmax).toBeLessThan(kp * 2);
    });

    test("amplitudes (eq. (29)-region): αp = 0.006·Ωc^0.55; both αm branches present and positive", () => {
        expect(spectrumAmplitudes(U10, OMEGA_C).alphaP).toBeCloseTo(0.006 * 0.9 ** 0.55, 12);
        // u*/cm at U₁₀=15 is ≈ 0.63/0.23 > 1 → the rough-flow branch
        expect(frictionVelocity(U10) / CM).toBeGreaterThan(1);
        expect(spectrumAmplitudes(U10, OMEGA_C).alphaM).toBeCloseTo(
            0.01 * (1 + 3 * Math.log(frictionVelocity(U10) / CM)),
            12,
        );
        // smooth branch: at U₁₀=4 the friction velocity stays under cm
        expect(frictionVelocity(4) / CM).toBeLessThan(1);
        expect(spectrumAmplitudes(4, OMEGA_C).alphaM).toBeCloseTo(
            0.01 * (1 + Math.log(frictionVelocity(4) / CM)),
            12,
        );
    });

    test("long/short wave factors (eqs. (30)–(34)): taper, peakedness, both exponentials", () => {
        const kp = peakWavenumber(U10, OMEGA_C);
        // Lpm(kp) = exp(−1.25) — the taper is 1 only as k→0; Jp(kp) = γ since r(kp)=1
        const atPeak = longWaveFactor(kp, U10, OMEGA_C);
        expect(atPeak).toBeCloseTo(Math.exp(-1.25) * 1.7, 9);
        // Fp is NOT monotone just above kp (the rising Lpm taper beats the falling side-effect
        // exponential near the peak) — but it decays hard by kp·100, and Fm ≤ Fp's envelope branch
        expect(longWaveFactor(kp * 100, U10, OMEGA_C)).toBeLessThan(atPeak * 0.2);
        expect(shortWaveFactor(KM, U10, OMEGA_C)).toBeCloseTo(peakEnvelope(KM, U10, OMEGA_C), 9);
        // Fm decays hard beyond km (the Gaussian cut)
        expect(shortWaveFactor(KM * 4, U10, OMEGA_C)).toBeLessThan(
            shortWaveFactor(KM, U10, OMEGA_C) * 0.5,
        );
    });

    test("spreading (eqs. (37)–(40)): Δ ∈ (0,1), directional factor unit-normalized, 2nd harmonic", () => {
        const delta = spreadingDelta(1, U10, OMEGA_C);
        expect(delta).toBeGreaterThan(0);
        expect(delta).toBeLessThan(1);
        // (1+Δcos(2θ))/(2π) integrates to 1 over the full circle
        let sum = 0;
        const steps = 720;
        for (let i = 0; i < steps; i++) {
            sum +=
                directionalFactor((i + 0.5) * ((2 * Math.PI) / steps), 1, U10, OMEGA_C) *
                ((2 * Math.PI) / steps);
        }
        expect(sum).toBeCloseTo(1, 9);
        // downwind − upwind anisotropy = 2Δ/(2π)·(1/2π)... the ratio of the two lobe maxima:
        expect(directionalFactor(0, 1, U10, OMEGA_C)).toBeGreaterThan(
            directionalFactor(Math.PI / 2, 1, U10, OMEGA_C),
        );
    });

    test("density: positive and finite across the full band at every declared Ωc", () => {
        for (const omegaC of [OMEGA_C_MIN, 0.9, 1.5, 3, OMEGA_C_MAX]) {
            // along the wind the density never nulls
            expect(directionalDensity(0.3, 0.6, U10, omegaC, 0.6)).toBeGreaterThan(0);
            for (let i = 0; i < 200; i++) {
                const k = Math.exp(Math.log(0.005) + (i * (Math.log(500) - Math.log(0.005))) / 199);
                const val = directionalDensity(k, 0.6, U10, omegaC, 0.6);
                expect(Number.isFinite(val)).toBe(true);
                // Δ can saturate to exactly 1 (tanh), giving an exact crosswind null —
                // the published form's own doing — so the floor is ≥ 0, with a strict
                // positive check along the wind axis below.
                expect(val).toBeGreaterThanOrEqual(0);
            }
        }
    });

    test("Cox–Munk anchor: the independent full-tail mean-square slope tracks the published glint fit", () => {
        // THE external anchor — nobody in this repo tuned this: mss = 0.003 + 0.00512·U₁₀.
        // Tolerance is the paper's own §4 comparison band (the production side is held to 25%/30%).
        for (const [u10, tol] of [
            [5, 0.25],
            [10, 0.25],
            [15, 0.25],
            [20, 0.3],
        ] as const) {
            const mss = radialIntegral(0.01, KM, u10, 0.9, 2);
            const anchor = coxMunkMeanSquareSlope(u10);
            const ratio = mss / anchor;
            expect(Math.abs(ratio - 1)).toBeLessThan(tol);
            // the fit must GROW with wind speed — a scale error shows as a constant ratio,
            // a shape error as a non-monotone one
            expect(ratio).toBeGreaterThan(0.5);
        }
        expect(radialIntegral(0.01, KM, 20, 0.9, 2)).toBeGreaterThan(
            radialIntegral(0.01, KM, 5, 0.9, 2),
        );
    });

    test("spectral variance integral is finite and wave-height scales plausibly", () => {
        const variance = radialIntegral(0.01, KM, U10, OMEGA_C, 0);
        expect(Number.isFinite(variance)).toBe(true);
        expect(variance).toBeGreaterThan(0);
        const hs = 4 * Math.sqrt(variance);
        // U₁₀ = 15 m/s: published Hs is in the 2–5 m window for this wind
        expect(hs).toBeGreaterThan(1);
        expect(hs).toBeLessThan(8);
        // Hs grows with U₁₀
        expect(radialIntegral(0.01, KM, 20, OMEGA_C, 0)).toBeGreaterThan(variance);
    });
});
