/**
 * I2r-a, commit 1 — INDEPENDENT test-side implementation of the published Elfouhaily directional
 * spectrum (Elfouhaily, Chapron, Katsaros & Vandemark, "A unified directional spectrum for long and
 * short wind-driven waves", JGR 102(C7), 1997, DOI 10.1029/97JC00467) and of the published full-tail
 * mean-square-slope integral that Cox–Munk glint statistics anchor externally.
 *
 * This file is deliberately a SEPARATE authoring pass from `src/spectrum.ts`: it exists so the
 * production implementation can be compared against an oracle that was not read out of the
 * production code. Each factor carries its equation-number citation inline; the citation
 * provenance (what was and was not re-verified against the journal text in this environment) is
 * recorded in `tests/elfouhaily-source-review.md`, which must be read together with this file.
 *
 * Structure notes (independence): this module works in the paper's polar quantities — phase speed
 * c(k), the two curvature-spectrum branches B_l and B_h, and the directional spreading function
 * Δ(k, θ) — and only assembles a Cartesian density at the very edge, where the realization
 * convention (a 2-D Fourier density integrated against dk_x dk_z) forces it. Fixed points are
 * solved with plain loops; no shared helper with the production file is imported.
 */

/** gravitational acceleration [m/s²] (paper works in MKS units throughout). */
export const G = 9.81;

/** Kinematic viscosity of air at ~15 °C [m²/s] — enters the smooth/rough roughness closure's
 * viscous term (eqs. (10)–(15) region: the wind-stress chapter's z₀ closure). */
const NU_AIR = 1.46e-5;

/** von Kármán constant used by the paper's wind-stress closure (eqs. (10)–(15) region). */
const KAPPA = 0.4;

/** Charnock constant of the revised Charnock relation the paper's stress chapter adopts
 * (eqs. (10)–(15) region): z₀ = α_ch·u*²/g + 0.11·ν/u*. */
const CHARNOCK = 0.0185;

/** Gravity-capillary transition wavenumber [rad/m], eq. (17): km = √(ρ_w·g/τ) — the wavenumber
 * where gravity and surface-tension restoring forces balance (minimum phase speed). */
export const KM = Math.sqrt((1025 * G) / 0.074);

/** Minimum phase speed of the dispersion relation [m/s], eq. (16): cm = √(2g/km) — c(km) is the
 * minimum of c(k) = √(g/k·(1+(k/km)²)). */
export const CM = Math.sqrt((2 * G) / KM);

/**
 * Phase speed of the gravity–capillary dispersion relation [m/s], eqs. (3)+(16):
 * ω² = g·k·(1 + (k/km)²), c = ω/k = √(g/k·(1+(k/km)²)).
 * Pure. Valid for k > 0; the (k/km)² term is the surface-tension correction.
 */
export function phaseSpeed(k: number): number {
    return Math.sqrt((G / k) * (1 + (k / KM) ** 2));
}

/**
 * Friction velocity u* [m/s] from the 10 m wind speed U₁₀ [m/s], eqs. (10)–(15) region: the
 * paper's wind-stress chapter closes the profile through the roughness length
 * z₀ = α_ch·u*²/g + 0.11·ν/u* and the logarithmic layer law U₁₀ = (u* over κ)·ln(10/z₀).
 * Solved by fixed-point iteration from a smooth-flow start (32 sweeps — the same order of
 * convergence discipline the production loop uses, reached independently).
 * Pure. Positive for U₁₀ > 0.
 */
export function frictionVelocity(u10: number): number {
    let ustar = (0.4 * u10) / Math.log(10 / (CHARNOCK * 0.001));
    for (let sweep = 0; sweep < 32; sweep++) {
        const z0 = (CHARNOCK * ustar * ustar) / G + (0.11 * NU_AIR) / ustar;
        ustar = (KAPPA * u10) / Math.log(10 / z0);
    }
    return ustar;
}

/**
 * Peak wavenumber kp [rad/m] from the inverse wave age Ωc: the paper defines
 * Ωc = ωp·U₁₀/g (dimensionless inverse wave age; Ωc = 0.84 is fully developed), and with the
 * deep-water gravity-wave dispersion ωp² = g·kp this inverts to kp = g·Ωc²/U₁₀².
 * Pure.
 */
export function peakWavenumber(u10: number, omegaC: number): number {
    return (G * omegaC * omegaC) / (u10 * u10);
}

/**
 * The shared peak envelope of both curvature-spectrum branches, eqs. (32)–(33) region:
 *   Lpm = exp(−1.25·(kp/k)²)               — the Phillips-equilibrium taper,
 *   r   = exp(−(√(k/kp)−1)²/(2σ²))          — the peakedness weighting,
 *   Jp  = γ^r                               — the peakedness enhancement,
 * with the paper's long-wave shape parameters (eq. (31)-region):
 *   σ   = 0.08·(1 + 4·Ωc⁻³)                — the spread of the peak around kp,
 *   γ   = 1.7 for Ωc ≤ 1, else 1.7 + 6·log₁₀(Ωc) — the Ωc ≤ 1 and log₁₀ branches.
 * Both Fp and Fm multiply THIS envelope and their own private exponential — Fp's
 * side-effect exponential belongs to Fp alone and must not leak into Fm.
 * Pure.
 */
export function peakEnvelope(k: number, u10: number, omegaC: number): number {
    const kp = peakWavenumber(u10, omegaC);
    const sigma = 0.08 * (1 + 4 * omegaC ** -3);
    const gamma = omegaC <= 1 ? 1.7 : 1.7 + 6 * Math.log10(omegaC);
    const r = Math.exp(-((Math.sqrt(k / kp) - 1) ** 2) / (2 * sigma * sigma));
    const jp = gamma ** r;
    const lpm = Math.exp(-1.25 * (kp / k) ** 2);
    return lpm * jp;
}

/**
 * The low-wavenumber (long gravity wave) branch of the curvature spectrum's directional
 * factor, eqs. (30)–(34) region: Fp = Lpm·Jp·exp(−(Ωc/√10)(√(k/kp)−1)) — the peak envelope
 * times Fp's own side-effect exponential.
 * Pure.
 */
export function longWaveFactor(k: number, u10: number, omegaC: number): number {
    const kp = peakWavenumber(u10, omegaC);
    return (
        peakEnvelope(k, u10, omegaC) * Math.exp(-(omegaC / Math.sqrt(10)) * (Math.sqrt(k / kp) - 1))
    );
}

/**
 * The high-wavenumber (short gravity wave) branch of the curvature spectrum's directional
 * factor, eqs. (30)–(34) region: Fm = Lpm·Jp·exp(−0.25·(k/km−1)²) — the same Lpm/Jp peak
 * envelope the long-wave branch carries (and NOTHING of Fp's exponential), cut off beyond
 * the gravity–capillary transition km by a Gaussian in (k/km − 1)² (eq. (34)-region).
 * Pure.
 */
export function shortWaveFactor(k: number, u10: number, omegaC: number): number {
    return peakEnvelope(k, u10, omegaC) * Math.exp(-0.25 * (k / KM - 1) ** 2);
}

/**
 * The saturation-spectrum prefactors, eqs. (29)+(35) region:
 *   αp = 0.006·Ωc^0.55 (the long-wave amplitude; the 0.55 exponent is the published one),
 *   αm: BOTH branches — for u* > cm the rough-flow form 0.01·(1 + 3·ln(u* / cm)), else the
 *   smooth-flow form 0.01·(1 + ln(u* / cm)).
 * Returns { alphaP, alphaM }. Pure.
 */
export function spectrumAmplitudes(
    u10: number,
    omegaC: number,
): { alphaP: number; alphaM: number } {
    const alphaP = 0.006 * omegaC ** 0.55;
    const ustar = frictionVelocity(u10);
    const ratio = ustar / CM;
    const alphaM = ratio > 1 ? 0.01 * (1 + 3 * Math.log(ratio)) : 0.01 * (1 + Math.log(ratio));
    return { alphaP, alphaM };
}

/**
 * The curvature spectrum B(k) = B_l(k) + B_h(k), eq. (4)+eqs. (29)–(34):
 *   B_l = 0.5·αp·(cp/c)·Fp      (long-wave branch, eq. (35)-region),
 *   B_h = 0.5·αm·(cm/c)·Fm      (short-wave branch, eq. (36)-region).
 * Pure. Positive for k in the paper's validity range (kp ≲ k ≲ several·km).
 */
export function curvatureSpectrum(k: number, u10: number, omegaC: number): number {
    const { alphaP, alphaM } = spectrumAmplitudes(u10, omegaC);
    const c = phaseSpeed(k);
    const cp = phaseSpeed(peakWavenumber(u10, omegaC));
    const bl = 0.5 * alphaP * (cp / c) * longWaveFactor(k, u10, omegaC);
    const bh = 0.5 * alphaM * (CM / c) * shortWaveFactor(k, u10, omegaC);
    return bl + bh;
}

/**
 * The directional spreading function's anisotropy parameter, eqs. (37)–(40) region: the paper's
 * Δ(k) enters through tanh of the harmonic mean of the long- and short-wave contributions
 *   4·(c/cp)^2.5 + am·(cm/c)^2.5     with am = 0.13·u* / cm (the wind-shifted asymmetry),
 * wrapped by the constant ln(2)/4 that keeps Δ → 1⁻ in the fully developed limit.
 * Returns the tanh argument's evaluation Δ itself. Pure.
 */
export function spreadingDelta(k: number, u10: number, omegaC: number): number {
    const cp = phaseSpeed(peakWavenumber(u10, omegaC));
    const c = phaseSpeed(k);
    const ustar = frictionVelocity(u10);
    const am = (0.13 * ustar) / CM;
    return Math.tanh(Math.log(2) / 4 + 4 * (c / cp) ** 2.5 + am * (CM / c) ** 2.5);
}

/**
 * The directional spreading normalization, eqs. (37)–(40) region: the paper's angular factor is
 * Φ(k, θ) = (1 + Δ·cos(2(θ − θw)))/(2π) — unit integral over θ (checked by
 * `elfouhaily-independent.test.ts`), second harmonic only, θw the wind direction [rad].
 * Pure.
 */
export function directionalFactor(
    theta: number,
    k: number,
    u10: number,
    omegaC: number,
    windDir = 0,
): number {
    const delta = spreadingDelta(k, u10, omegaC);
    return (1 + delta * Math.cos(2 * (theta - windDir))) / (2 * Math.PI);
}

/**
 * The full directional Cartesian wavenumber density Ψ(k, θ) [m²·rad²... per (kx, kz) cell]:
 * Ψ = (B_l + B_h)/k⁴ · Φ(k, θ). The k⁴ is the realization convention: a 2-D Fourier density
 * integrated against dk_x·dk_z reproduces the variance the paper's polar Ψ carries against
 * k·dk·dθ (reviewed in `elfouhaily-source-review.md`, "conventions"). This is the only place
 * the polar → Cartesian step happens in this file.
 * Pure. Returns the density at (k, θ) for the declared (U₁₀, Ωc) pair.
 */
export function directionalDensity(
    k: number,
    theta: number,
    u10: number,
    omegaC: number,
    windDir = 0,
): number {
    const b = curvatureSpectrum(k, u10, omegaC);
    return (b / k ** 4) * directionalFactor(theta, k, u10, omegaC, windDir);
}

/**
 * The omnidirectional radial moment ∫₀^{2π} Ψ(k, θ)·k^p dθ for p ∈ {0, 1, 2} — the building
 * block of the independent variance (p = 1, the paper's polar measure) and of the independent
 * full-tail mean-square-slope integral (p = 3 against the polar k·dk measure, i.e. the slope
 * moment k²·Ψ). Log-spaced Simpson quadrature over ln k — the integrand is smooth in ln k and
 * the band spans four decades. Pure.
 */
export function radialMoment(k: number, u10: number, omegaC: number, power: 0 | 1 | 2): number {
    // ∫₀^{2π} Ψ dθ = B/k⁴ exactly: the spreading's second harmonic (1 + Δ·cos(2(θ−θw)))/(2π)
    // integrates to 1 over the full circle — Δ and θw drop out of every omnidirectional moment.
    return (curvatureSpectrum(k, u10, omegaC) / k ** 4) * k ** power;
}

/**
 * The independent full-tail variance ∫∫Ψ·k^p·k dk dθ over k ∈ [kMin, kMax] (p = 0 for variance,
 * p = 2 for mean-square slope): the polar measure ∫k dk dθ with the extra k^p moment.
 * Log-spaced composite Simpson with `steps` panels. Pure.
 */
export function radialIntegral(
    kMin: number,
    kMax: number,
    u10: number,
    omegaC: number,
    power: 0 | 2,
    steps = 256,
): number {
    const logLo = Math.log(kMin);
    const logHi = Math.log(kMax);
    const h = (logHi - logLo) / steps;
    let sum = 0;
    for (let i = 0; i <= steps; i++) {
        const weight = i === 0 || i === steps ? 1 : i % 2 === 1 ? 4 : 2;
        const k = Math.exp(logLo + i * h);
        // log-space measure: dk = k·d(ln k), so the x-space integrand gains one more k than the
        // dk-space one: (angular mean of Ψ)·k^{power+1}·k — the extra k is the Jacobian.
        const mean = radialMoment(k, u10, omegaC, 0);
        sum += weight * mean * k ** (power + 2);
    }
    return (sum * h) / 3;
}

/**
 * The independent Cox–Munk anchor: the published empirical glint fit
 * mss = 0.003 + 0.00512·U₁₀ (Cox & Munk 1954, as quoted by the Elfouhaily paper's own §4 when it
 * compares its integral against the glint statistics). Exposed so the oracle's full-tail
 * mean-square-slope can be gated against an EXTERNAL number nobody in this repo tuned.
 */
export function coxMunkMeanSquareSlope(u10: number): number {
    return 0.003 + 0.00512 * u10;
}

/** The published inverse-wave-age validity window, eq. (29)-region: Ωc ∈ [0.84, 5] —
 * 0.84 is the fully developed sea, 5 the aggressively young limit the paper plots. */
export const OMEGA_C_MIN = 0.84;
export const OMEGA_C_MAX = 5;
