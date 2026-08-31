// Physical ocean spectrum for the FFT substrate.
// Grid resolution is constrained by the radix-2 transform; world-space patch lengths define the
// spatial repeat period independently.

export type LabelFn = (i: number, N: number) => number;

export interface CascadeConfig {
    /** FFT grid resolution (N×N), constrained to a power of two by the radix-2 transform. */
    N: number;
    /** Physical domain size in meters (the patch the N×N grid covers). */
    L: number;
    /** Declared spectral band (rad/m), applied in `generateH0`. */
    kLo: number;
    /** Upper edge of the declared spectral band (rad/m). */
    kHi: number;
}

/** The physical inputs shared by every cascade. */
export interface SeaState {
    /** Significant wave height in meters; variance is `(significantWaveHeight / 4)²`. */
    significantWaveHeight: number;
    /** Ten-metre wind speed in m/s, shared by the composed spectrum. */
    windSpeed: number;
    /** Wind direction in radians, shared by the composed spectrum. */
    windDir: number;
    /** Choppiness derived from the composed-field whitecap anchor. */
    lambda: number;
    /** Whitecap coverage target derived from the wind speed. */
    whitecapFraction: number;
}

/**
 * The two cascades use one physical sea state and retain their distinct world-space periods. Their
 * lengths are pairwise coprime, so the shared repeat period is 2480 m. The sea-state normalization
 * is computed below from the composed declared bands; no per-cascade amplitude or wind exists.
 */
export const CASCADE_CONFIGS: CascadeConfig[] = [
    { N: 64, L: 80, kLo: 0.07853981633974483, kHi: 0.75 },
    { N: 128, L: 31, kLo: 0.75, kHi: 8.482300164692441 },
];

/** Standard DFT/FFT mode index: `k = (i <= N/2 ? i : i-N) * dk`. */
export function kIndex(i: number, N: number): number {
    return i <= N / 2 ? i : i - N;
}

/** true iff `n` is a power of two (`n >= 1`). */
export function isPowerOfTwo(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}

/** greatest common divisor for integer patch lengths. */
export function gcd(a: number, b: number): number {
    while (b > 0) [a, b] = [b, a % b];
    return a;
}

/** Assert every cascade's `N` is a power of two. */
export function assertAllPowerOfTwo(configs: CascadeConfig[]): boolean {
    return configs.every((c) => isPowerOfTwo(c.N));
}

/** Check the anti-alignment invariant for world-space patch lengths. */
export function assertCoprimeL(configs: CascadeConfig[]): boolean {
    for (let i = 0; i < configs.length; i++) {
        for (let j = i + 1; j < configs.length; j++) {
            if (!Number.isInteger(configs[i].L) || !Number.isInteger(configs[j].L)) return false;
            if (gcd(configs[i].L, configs[j].L) !== 1) return false;
        }
    }
    return true;
}

function lcm(a: number, b: number): number {
    return (a * b) / gcd(a, b);
}

/** The world-space distance at which all cascade boundaries re-align. */
export function tilePeriod(configs: CascadeConfig[]): number {
    if (configs.length < 2) return Number.POSITIVE_INFINITY;
    return configs.reduce((period, c) => lcm(period, c.L), 1);
}

if (!assertAllPowerOfTwo(CASCADE_CONFIGS)) {
    throw new Error(
        `shallot-ocean: every cascade N must be a power of two for the butterfly FFT — got ${CASCADE_CONFIGS.map((c) => c.N).join(", ")}`,
    );
}

/** gravitational acceleration in m/s². */
export const G = 9.81;
const SURFACE_TENSION = 0.074;
const WATER_DENSITY = 1025;
const ELF_ALPHA_P = 0.006;
const ELF_ALPHA_M = 0.01;
const ELF_REFERENCE_SPEED = 0.23;
const TARGET_HS = 3;
const BASE_WIND = 15;
const BASE_WIND_DIR = 0.6;

/**
 * Monahan & O'Muircheartaigh's wind-driven whitecap coverage fit, W = 3.84e-6 U₁₀^3.41.
 * Source: https://doi.org/10.1029/JC087iC01p00457. It supplies the physical fold anchor; it is not
 * a visual threshold selected from the fold gate.
 */
export function whitecapFraction(windSpeed: number): number {
    if (windSpeed <= 0) return 0;
    return Math.min(0.5, 3.84e-6 * windSpeed ** 3.41);
}

/**
 * H0 is seeded by the physical wavenumber rather than the grid, so a represented mode is identical
 * at every N. The draw is deterministic only to make CPU/GPU and N-invariance readings repeatable.
 */
function hashBits(a: number, b: number, salt: number): number {
    const buf = new ArrayBuffer(16);
    const dv = new DataView(buf);
    dv.setFloat64(0, a);
    dv.setFloat64(8, b);
    const w0 = dv.getUint32(0);
    const w1 = dv.getUint32(4);
    const w2 = dv.getUint32(8);
    const w3 = dv.getUint32(12);
    let h = (0x9e3779b9 ^ salt) >>> 0;
    h = Math.imul(h ^ w0, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h ^ w1, 0xc2b2ae35);
    h ^= h >>> 16;
    h = Math.imul(h ^ w2, 0x85ebca6b);
    h ^= h >>> 13;
    h = Math.imul(h ^ w3, 0xc2b2ae35);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
}

function gaussFromK(kx: number, kz: number, salt: number): number {
    const u1 = Math.max(hashBits(kx, kz, salt * 2), 1e-10);
    const u2 = hashBits(kx, kz, salt * 2 + 1);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function rawCellAmplitude(
    cfg: CascadeConfig,
    kx: number,
    kz: number,
    densityScale: number,
    windSpeed: number,
    windDir: number,
): number {
    const k = Math.hypot(kx, kz);
    if (k < cfg.kLo || k > cfg.kHi) return 0;
    return (
        (Math.sqrt(Math.max(0, unifiedSpectrum(kx, kz, windSpeed, windDir) * densityScale)) *
            ((2 * Math.PI) / cfg.L)) /
        Math.SQRT2
    );
}

/** Parseval variance before the final sea-state realization correction. */
function unscaledRealizedFieldVariance(
    population: CascadeConfig[],
    significantWaveHeight = TARGET_HS,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
): number {
    const densityScale = spectrumNormalization(
        population,
        significantWaveHeight,
        windSpeed,
        windDir,
    );
    let sum = 0;
    for (const cfg of population) {
        const dk = (2 * Math.PI) / cfg.L;
        let cascadeSum = 0;
        for (let y = 0; y < cfg.N; y++) {
            for (let x = 0; x < cfg.N; x++) {
                const kx = kIndex(x, cfg.N) * dk;
                const kz = kIndex(y, cfg.N) * dk;
                const amp = rawCellAmplitude(cfg, kx, kz, densityScale, windSpeed, windDir);
                const re = gaussFromK(kx, kz, 0) * amp;
                const im = gaussFromK(kx, kz, 1) * amp;
                const nx = (cfg.N - x) % cfg.N;
                const ny = (cfg.N - y) % cfg.N;
                const nkx = kIndex(nx, cfg.N) * dk;
                const nkz = kIndex(ny, cfg.N) * dk;
                const namp = rawCellAmplitude(cfg, nkx, nkz, densityScale, windSpeed, windDir);
                const nre = gaussFromK(nkx, nkz, 0) * namp;
                const nim = gaussFromK(nkx, nkz, 1) * namp;
                const hre = re + nre;
                const him = im - nim;
                cascadeSum += hre * hre + him * him;
            }
        }
        // The inverse FFT is unnormalized; Parseval's average is the coefficient-energy sum.
        sum += cascadeSum;
    }
    return sum;
}

/** Parseval variance after updateH combines ±k draws, before an optional realization scale. */
export function realizedFieldVariance(
    population: CascadeConfig[],
    significantWaveHeight = TARGET_HS,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
    realizationScale = 1,
): number {
    return (
        unscaledRealizedFieldVariance(population, significantWaveHeight, windSpeed, windDir) *
        realizationScale ** 2
    );
}

/**
 * Elfouhaily et al.'s unified directional spectrum, expressed as a 2-D height PSD. The low and
 * high branches share phase speed and wind inputs, including the gravity/capillary transition;
 * this is deliberately not a Phillips tail with a hand amplitude. Source: Elfouhaily et al., JGR
 * 102(C7), 1997, https://doi.org/10.1029/97JC00467.
 */
export function unifiedSpectrum(
    kx: number,
    kz: number,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
): number {
    const k2 = kx * kx + kz * kz;
    if (k2 < 1e-12 || windSpeed <= 0) return 0;
    const k = Math.sqrt(k2);
    const kPeak = G / (windSpeed * windSpeed);
    const kCapillary = Math.sqrt((WATER_DENSITY * G) / SURFACE_TENSION);
    const phaseSpeed = Math.sqrt((G / k) * (1 + (k / kCapillary) ** 2));
    const peakSpeed = Math.sqrt((G / kPeak) * (1 + (kPeak / kCapillary) ** 2));
    const capillarySpeed = Math.sqrt((G / kCapillary) * 2);
    const alphaP = ELF_ALPHA_P * Math.sqrt(windSpeed / peakSpeed);
    const alphaM = ELF_ALPHA_M * (1 + Math.log(Math.max(windSpeed / ELF_REFERENCE_SPEED, 1)));
    const low = 0.5 * alphaP * (peakSpeed / phaseSpeed) * Math.exp(-1.25 * (kPeak / k) ** 2);
    const high =
        0.5 * alphaM * (capillarySpeed / phaseSpeed) * Math.exp(-0.25 * (k / kCapillary - 1) ** 2);
    const directionalCos = (kx * Math.cos(windDir) + kz * Math.sin(windDir)) / k;
    const directionalSpread = Math.tanh(
        Math.log(2) / 4 +
            4 * (phaseSpeed / peakSpeed) ** 2.5 +
            0.13 * (capillarySpeed / phaseSpeed) ** 2.5,
    );
    const directional =
        (1 + directionalSpread * (2 * directionalCos * directionalCos - 1)) / (2 * Math.PI);
    // Elfouhaily's Ψ is a radial spectrum integrated with k·dk·dθ. A Cartesian 2-D PSD
    // therefore divides Ψ by one additional k before integrating dkx·dkz.
    return Math.max(0, ((low + high) * directional) / k ** 4);
}

interface SpectrumMetrics {
    densityVariance: number;
    normalizedVariance: number;
    slopeVariance: number;
    strainVariance: number;
}

function integrateSpectrum(
    configs: CascadeConfig[],
    significantWaveHeight: number,
    windSpeed: number,
    windDir: number,
): SpectrumMetrics {
    let densityVariance = 0;
    let slopeVariance = 0;
    let strainVariance = 0;
    for (const cfg of configs) {
        const dk = (2 * Math.PI) / cfg.L;
        for (let y = 0; y < cfg.N; y++) {
            for (let x = 0; x < cfg.N; x++) {
                const kx = kIndex(x, cfg.N) * dk;
                const kz = kIndex(y, cfg.N) * dk;
                const k = Math.hypot(kx, kz);
                if (k < cfg.kLo || k > cfg.kHi) continue;
                const cell = unifiedSpectrum(kx, kz, windSpeed, windDir) * dk * dk;
                densityVariance += cell;
                slopeVariance += cell * k * k;
                // Dxx = ∂x(Dx) has Fourier multiplier kx²/k. Keep this directional moment
                // instead of assuming that the wind-aligned spectrum is isotropic.
                strainVariance += k > 0 ? (cell * kx ** 4) / (k * k) : 0;
            }
        }
    }
    const targetVariance = (significantWaveHeight / 4) ** 2;
    const normalization = densityVariance > 0 ? targetVariance / densityVariance : 0;
    return {
        densityVariance,
        normalizedVariance: densityVariance * normalization,
        slopeVariance: slopeVariance * normalization,
        strainVariance: strainVariance * normalization,
    };
}

/** Return the multiplicative sea-state normalization applied to each spectral cell. */
export function spectrumNormalization(
    configs: CascadeConfig[],
    significantWaveHeight = TARGET_HS,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
): number {
    let densityVariance = 0;
    for (const cfg of configs) {
        const dk = (2 * Math.PI) / cfg.L;
        for (let y = 0; y < cfg.N; y++) {
            for (let x = 0; x < cfg.N; x++) {
                const kx = kIndex(x, cfg.N) * dk;
                const kz = kIndex(y, cfg.N) * dk;
                const k = Math.hypot(kx, kz);
                if (k >= cfg.kLo && k <= cfg.kHi)
                    densityVariance += unifiedSpectrum(kx, kz, windSpeed, windDir) * dk * dk;
            }
        }
    }
    return densityVariance > 0 ? (significantWaveHeight / 4) ** 2 / densityVariance : 0;
}

/** A deterministic fold band derived from the wind whitecap anchor and composed slope variance. */
export interface FoldBand {
    whitecapAnchor: number;
    lambda: number;
    lambdaCeiling: number;
    slopeRms: number;
    /** one-sided negative-strain probability at the derived λ */
    foldAnchor: number;
    /** one-sided negative-strain probability at the unit-RMS ceiling */
    foldCeiling: number;
}

function erfc(x: number): number {
    const sign = x < 0 ? -1 : 1;
    const a = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * a);
    const p =
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)));
    const value = p * t * Math.exp(-a * a);
    return sign >= 0 ? value : 2 - value;
}

function inverseErfc(p: number): number {
    let low = 0;
    let high = 8;
    for (let i = 0; i < 64; i++) {
        const mid = (low + high) / 2;
        if (erfc(mid) > p) low = mid;
        else high = mid;
    }
    return (low + high) / 2;
}

/** One-sided negative Gaussian strain probability at a given λ and RMS. */
export function foldProbability(lambda: number, slopeRms: number): number {
    if (lambda <= 0 || slopeRms <= 0) return 0;
    return 0.5 * erfc(1 / (Math.SQRT2 * lambda * slopeRms));
}

/**
 * Derive λ from the composed spectrum: the anchor is the wind whitecap probability and the ceiling
 * is the unit-RMS fold limit. The one-sided Gaussian tail relation is `P(X < -1/λ) = W`, so λ is
 * solved from the anchor and then compared with the ceiling; no observed fold pass or fail selects it.
 */
export function deriveFoldBand(
    configs: CascadeConfig[],
    significantWaveHeight = TARGET_HS,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
): FoldBand {
    const metrics = integrateSpectrum(configs, significantWaveHeight, windSpeed, windDir);
    // The whitecap relation is one-sided in a diagonal strain component. Use its directional
    // spectral moment directly rather than assuming that this wind-aligned field is isotropic.
    const slopeRms = Math.sqrt(Math.max(metrics.strainVariance, Number.MIN_VALUE));
    const anchor = whitecapFraction(windSpeed);
    const tailSigma = Math.SQRT2 * inverseErfc(2 * Math.max(anchor, 1e-6));
    const lambda = 1 / (slopeRms * tailSigma);
    const lambdaCeiling = 1 / slopeRms;
    return {
        whitecapAnchor: anchor,
        lambda,
        lambdaCeiling,
        slopeRms,
        foldAnchor: foldProbability(lambda, slopeRms),
        foldCeiling: foldProbability(lambdaCeiling, slopeRms),
    };
}

/** One physical sea state shared by every cascade. */
export const SEA_STATE: SeaState = Object.freeze({
    significantWaveHeight: TARGET_HS,
    windSpeed: BASE_WIND,
    windDir: BASE_WIND_DIR,
    lambda: deriveFoldBand(CASCADE_CONFIGS, TARGET_HS, BASE_WIND, BASE_WIND_DIR).lambda,
    whitecapFraction: whitecapFraction(BASE_WIND),
});

/** Expected variance represented by one declared band after the shared sea-state normalization. */
export function declaredBandVariance(
    cfg: CascadeConfig,
    population: CascadeConfig[],
    significantWaveHeight = TARGET_HS,
    windSpeed = BASE_WIND,
    windDir = BASE_WIND_DIR,
): number {
    const dk = (2 * Math.PI) / cfg.L;
    let variance = 0;
    const scale = spectrumNormalization(population, significantWaveHeight, windSpeed, windDir);
    for (let y = 0; y < cfg.N; y++) {
        for (let x = 0; x < cfg.N; x++) {
            const kx = kIndex(x, cfg.N) * dk;
            const kz = kIndex(y, cfg.N) * dk;
            const k = Math.hypot(kx, kz);
            if (k >= cfg.kLo && k <= cfg.kHi)
                variance += unifiedSpectrum(kx, kz, windSpeed, windDir) * scale * dk * dk;
        }
    }
    return variance;
}

/** The normalization applied to a discrete spectral cell, including `sqrt(Δk²)`. */
export function spectralCellAmplitude(
    cfg: CascadeConfig,
    kx: number,
    kz: number,
    population: CascadeConfig[],
    seaState: SeaState = SEA_STATE,
): number {
    const k = Math.hypot(kx, kz);
    if (k < cfg.kLo || k > cfg.kHi) return 0;
    const dk = (2 * Math.PI) / cfg.L;
    const density = unifiedSpectrum(kx, kz, seaState.windSpeed, seaState.windDir);
    const densityScale = spectrumNormalization(
        population,
        seaState.significantWaveHeight,
        seaState.windSpeed,
        seaState.windDir,
    );
    return Math.sqrt(Math.max(0, density * densityScale) * (dk * dk));
}

/**
 * Generate Fourier-domain H₀(k). `sqrt(Δk²)` converts the published density into a Fourier-cell
 * amplitude; the unnormalized inverse FFT's Parseval normalization is applied by the realized-field
 * calibration below.
 */
export function generateH0(
    cfg: CascadeConfig,
    labelFn: LabelFn,
    seaState: SeaState,
    population: CascadeConfig[],
): Float32Array {
    const N = cfg.N;
    const dk = (2 * Math.PI) / cfg.L;
    const h0 = new Float32Array(N * N * 2);
    const densityScale = spectrumNormalization(
        population,
        seaState.significantWaveHeight,
        seaState.windSpeed,
        seaState.windDir,
    );
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const kx = labelFn(x, N) * dk;
            const kz = labelFn(y, N) * dk;
            const idx = (y * N + x) * 2;
            const k = Math.hypot(kx, kz);
            if (k < cfg.kLo || k > cfg.kHi) continue;
            const pCell =
                unifiedSpectrum(kx, kz, seaState.windSpeed, seaState.windDir) * densityScale;
            const cellAmplitude = Math.sqrt(Math.max(pCell, 0) * (dk * dk)) / Math.SQRT2;
            h0[idx] = gaussFromK(kx, kz, 0) * cellAmplitude;
            h0[idx + 1] = gaussFromK(kx, kz, 1) * cellAmplitude;
        }
    }
    // updateH adds an independently drawn conjugate -k term, so calibrate the realized Parseval
    // variance rather than assuming the density integral is the realized field variance.
    const realized = unscaledRealizedFieldVariance(
        population,
        seaState.significantWaveHeight,
        seaState.windSpeed,
        seaState.windDir,
    );
    const correction = Math.sqrt((seaState.significantWaveHeight / 4) ** 2 / realized);
    for (let i = 0; i < h0.length; i++) h0[i] *= correction;
    return h0;
}

/** Theoretical FFT butterfly FLOP count for one cascade. */
export function theoreticalFlops(cfg: CascadeConfig): number {
    const N = cfg.N;
    return 6 * N * N * Math.log2(N) + 4 * N * N;
}

/** Total theoretical FLOPs across all cascades. */
export function totalTheoreticalFlops(configs: CascadeConfig[]): number {
    return configs.reduce((sum, c) => sum + theoreticalFlops(c), 0);
}

/** Counterfactual direct-DFT FLOP count retained for the FFT performance reading. */
export function directDftFlops(cfg: CascadeConfig): number {
    const N = cfg.N;
    return 16 * N * N * N + 4 * N * N;
}
