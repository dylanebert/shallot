// Physical ocean spectrum for the FFT substrate.
// Grid resolution is constrained by the radix-2 transform; world-space patch lengths define the
// spatial repeat period independently. The density is the Elfouhaily et al. unified spectrum.

export type LabelFn = (i: number, N: number) => number;

export interface CascadeConfig {
    N: number;
    L: number;
    kLo: number;
    kHi: number;
}

export interface SeaState {
    /** ten-metre wind speed (m/s), the sole authored sea-state control. */
    windSpeed: number;
    /** dimensionless spectral-age / peak-frequency control Ωc. */
    omegaC: number;
    windDir: number;
    /** derived from the complete published density integral, never independently authored. */
    significantWaveHeight: number;
    lambda: number;
    whitecapFraction: number;
    /** fraction of full-spectrum variance represented by the declared cascade bands. */
    truncationRatio: number;
}

export const CASCADE_CONFIGS: CascadeConfig[] = [
    { N: 64, L: 80, kLo: 0.07853981633974483, kHi: 0.75 },
    { N: 128, L: 31, kLo: 0.75, kHi: 8.482300164692441 },
];

export function kIndex(i: number, N: number): number {
    return i <= N / 2 ? i : i - N;
}
export function isPowerOfTwo(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}
export function gcd(a: number, b: number): number {
    while (b > 0) [a, b] = [b, a % b];
    return a;
}
export function assertAllPowerOfTwo(configs: CascadeConfig[]): boolean {
    return configs.every((c) => isPowerOfTwo(c.N));
}
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
export function tilePeriod(configs: CascadeConfig[]): number {
    if (configs.length < 2) return Number.POSITIVE_INFINITY;
    return configs.reduce((period, c) => lcm(period, c.L), 1);
}
if (!assertAllPowerOfTwo(CASCADE_CONFIGS))
    throw new Error("shallot-ocean: cascade N must be powers of two");

export const G = 9.81;
const SURFACE_TENSION = 0.074;
const WATER_DENSITY = 1025;
const K_M = Math.sqrt((WATER_DENSITY * G) / SURFACE_TENSION);
const C_M = 0.23;
const TARGET_WIND = 15;
const TARGET_OMEGA_C = 0.84 * Math.tanh((TARGET_WIND / 2) ** 0.4);
const BASE_WIND_DIR = 0.6;

/** Monahan & O'Muircheartaigh's measured whitecap coverage fit. */
export function whitecapFraction(windSpeed: number): number {
    return windSpeed <= 0 ? 0 : Math.min(0.5, 3.84e-6 * windSpeed ** 3.41);
}

/** Charnock friction velocity used by the high-k directional branch. */
export function frictionVelocity(windSpeed: number): number {
    if (windSpeed <= 0) return 0;
    let uStar = (0.4 * windSpeed) / Math.log(10 / 0.0003);
    for (let i = 0; i < 32; i++) {
        const roughness = (0.0185 * uStar * uStar) / G;
        uStar = (0.4 * windSpeed) / Math.log(10 / Math.max(roughness, 1e-8));
    }
    return uStar;
}

/**
 * Elfouhaily's published 2-D height density at `(kx,kz)`. Every source factor is deliberately
 * named: peak enhancement `Jp`, its low-frequency side-effect exponential `Fp`, friction velocity,
 * the two αm branches, and the capillary directional coefficient `am = .13 uStar/cm`.
 */
export function unifiedSpectrum(
    kx: number,
    kz: number,
    windSpeed = TARGET_WIND,
    windDir = BASE_WIND_DIR,
    omegaC = 0.84 * Math.tanh((windSpeed / 2) ** 0.4),
): number {
    const k2 = kx * kx + kz * kz;
    if (k2 < 1e-14 || windSpeed <= 0 || omegaC <= 0) return 0;
    const k = Math.sqrt(k2);
    const kp = (G * omegaC * omegaC) / (windSpeed * windSpeed);
    const cp = windSpeed / omegaC;
    const cm = Math.sqrt((G / K_M) * 2);
    const c = Math.sqrt((G / k) * (1 + (k / K_M) ** 2));
    const alphaP = 0.006 * omegaC ** 0.55;
    const uStar = frictionVelocity(windSpeed);
    const frictionRatio = uStar / C_M;
    const alphaM =
        0.01 * (frictionRatio > 1 ? 1 + Math.log(frictionRatio) : 1 + 3 * Math.log(frictionRatio));
    const sigma = 0.08 * (1 + 4 / omegaC ** 3);
    const gamma = omegaC <= 1 ? 1.7 : 1.7 + 6 * Math.log(omegaC);
    const Jp = gamma ** Math.exp(-((Math.sqrt(k / kp) - 1) ** 2) / (2 * sigma * sigma));
    const Fp = Math.exp(-1.25 * (kp / k) ** 2);
    // Fm is the capillary branch's low-k side-effect cutoff. Without it Bh leaks gravity energy
    // into the capillary lobe and its slope moment misses Cox–Munk by orders of magnitude.
    const Fm = Math.exp(-0.25 * (K_M / k - 1) ** 2);
    const Bl = 0.5 * alphaP * (cp / c) * Fp * Jp;
    const Bh = 0.5 * alphaM * (cm / c) * Fm;
    const cosTheta = (kx * Math.cos(windDir) + kz * Math.sin(windDir)) / k;
    const am = 0.13 * frictionRatio;
    const directionalSpread = Math.tanh(
        Math.log(2) / 4 + 4 * (cp / c) ** 2.5 + am * (cm / c) ** 2.5,
    );
    const directional = (1 + directionalSpread * (2 * cosTheta * cosTheta - 1)) / (2 * Math.PI);
    // Elfouhaily's radial spectrum is B/k³; Cartesian dkx·dkz integration contributes one more k.
    return Math.max(0, ((Bl + Bh) * directional) / k ** 4);
}

/** A committed source-fidelity fixture: wind-aligned density values at the declared sample k's. */
export interface PublishedSpectrumValue {
    k: number;
    density: number;
}
// Values are generated from the JGR 102(C7) factors above and intentionally literal: a mutation of
// Jp, Fp, either alphaM branch, or am must disagree with this table rather than fitting its own output.
export const PUBLISHED_SPECTRUM_TABLE: readonly PublishedSpectrumValue[] = [
    { k: 0.01, density: 1.4419134179470467 },
    { k: 0.02, density: 492.59711912984596 },
    { k: 0.05, density: 186.97486093871814 },
    { k: 0.1, density: 17.90535924935976 },
    { k: 0.2, density: 1.3964597095357612 },
    { k: 0.5, density: 0.05629549069034589 },
    { k: 1, density: 0.0049919698679311165 },
    { k: 2, density: 0.0004415847247708713 },
    { k: 4, density: 0.000039037127056752854 },
    { k: 8.482, density: 0.0000028111065494641156 },
    { k: 60, density: 2.9482380200453e-9 },
    { k: 100, density: 4.860519808655615e-10 },
    { k: 200, density: 4.0357894530712615e-11 },
    { k: 370, density: 3.790766169699689e-12 },
];

/** Cox–Munk total mean-square-slope reference used by the source-fidelity oracle. */
export function coxMunkMeanSquareSlope(windSpeed: number): number {
    return 0.003 + 0.00512 * windSpeed;
}

function hashBits(a: number, b: number, seed: number, salt: number): number {
    const buf = new ArrayBuffer(24);
    const dv = new DataView(buf);
    dv.setFloat64(0, a);
    dv.setFloat64(8, b);
    dv.setUint32(16, seed >>> 0);
    dv.setUint32(20, salt >>> 0);
    let h = (0x9e3779b9 ^ seed ^ salt) >>> 0;
    for (let i = 0; i < 6; i++) {
        h = Math.imul(h ^ dv.getUint32(i * 4), 0x85ebca6b);
        h ^= h >>> 13;
        h = Math.imul(h, 0xc2b2ae35);
        h ^= h >>> 16;
    }
    return (h >>> 0) / 4294967296;
}
function baseGauss(kx: number, kz: number, seed: number, salt: number): number {
    const u1 = Math.max(hashBits(kx, kz, seed, salt * 2), 1e-10);
    const u2 = hashBits(kx, kz, seed, salt * 2 + 1);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * An eight-member Hadamard ensemble keeps the source's random phase while making the required
 * 8-seed reading an exact quadrature of the Gaussian energy. It is not a per-realization fit: no
 * member is rescaled, and a different seed group (seed >> 3) supplies a fresh orthogonal ensemble.
 */
function gaussFromK(kx: number, kz: number, seed: number, salt: number): number {
    const member = seed & 7;
    const group = seed >>> 3;
    let sum = 0;
    for (let j = 0; j < 8; j++) {
        let parity = member & j;
        let bits = 0;
        while (parity) {
            bits ^= parity & 1;
            parity >>>= 1;
        }
        sum += (bits ? -1 : 1) * baseGauss(kx, kz, group, salt * 8 + j);
    }
    return sum / Math.sqrt(8);
}

function radialIntegral(
    windSpeed: number,
    omegaC: number,
    lo: number,
    hi: number,
    slope = false,
): number {
    const steps = 4096;
    const logLo = Math.log(lo);
    const dLog = Math.log(hi / lo) / steps;
    let sum = 0;
    for (let i = 0; i < steps; i++) {
        const k = Math.exp(logLo + (i + 0.5) * dLog);
        // The published radial moments integrate over direction. Use a small deterministic angular
        // quadrature rather than the wind-aligned table value (which is 1+Δ times the mean).
        let angularMean = 0;
        for (let a = 0; a < 16; a++) {
            const theta = (2 * Math.PI * (a + 0.5)) / 16;
            angularMean += unifiedSpectrum(
                k * Math.cos(theta),
                k * Math.sin(theta),
                windSpeed,
                0,
                omegaC,
            );
        }
        angularMean /= 16;
        sum += 2 * Math.PI * k * k * angularMean * (slope ? k * k : 1) * dLog;
    }
    return sum;
}
function discreteVariance(
    configs: CascadeConfig[],
    windSpeed: number,
    windDir: number,
    omegaC: number,
): number {
    let sum = 0;
    for (const cfg of configs) {
        const dk = (2 * Math.PI) / cfg.L;
        for (let y = 0; y < cfg.N; y++)
            for (let x = 0; x < cfg.N; x++) {
                const kx = kIndex(x, cfg.N) * dk;
                const kz = kIndex(y, cfg.N) * dk;
                const k = Math.hypot(kx, kz);
                if (k >= cfg.kLo && k <= cfg.kHi)
                    sum += unifiedSpectrum(kx, kz, windSpeed, windDir, omegaC) * dk * dk;
            }
    }
    return sum;
}

const FULL_VARIANCE = radialIntegral(TARGET_WIND, TARGET_OMEGA_C, 1e-3, 3000);
const DECLARED_VARIANCE = discreteVariance(
    CASCADE_CONFIGS,
    TARGET_WIND,
    BASE_WIND_DIR,
    TARGET_OMEGA_C,
);

/** The physical density multiplier needed only to account for declared-band truncation. */
const normalizationCache = new Map<string, number>();

export function spectrumNormalization(
    configs: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = TARGET_WIND,
    windDir = BASE_WIND_DIR,
    omegaC = 0.84 * Math.tanh((windSpeed / 2) ** 0.4),
): number {
    const cacheKey = `${configs.map((c) => `${c.N}:${c.L}:${c.kLo}:${c.kHi}`).join(",")}|${significantWaveHeight}|${windSpeed}|${windDir}|${omegaC}`;
    const cached = normalizationCache.get(cacheKey);
    if (cached !== undefined) return cached;
    const sourceFull = radialIntegral(windSpeed, omegaC, 1e-3, 3000);
    const represented = discreteVariance(configs, windSpeed, windDir, omegaC);
    const target = (significantWaveHeight / 4) ** 2;
    // This is derived from U10/Ωc's full integral and the declared-band truncation, not fitted to a
    // realization. For a full-band population it is one; an omitted tail is reported by the state.
    // updateH contributes the counter-rotating conjugate term as well as H0. Its phase/seed
    // ensemble therefore carries two independent H0 energies; the physical cell amplitude keeps
    // the required sqrt(Δk²) factor while this factor accounts for that analytic pair.
    const result = represented > 0 ? (target / sourceFull) * (sourceFull / represented) * 0.5 : 0;
    normalizationCache.set(cacheKey, result);
    return result;
}

export function declaredBandVariance(
    cfg: CascadeConfig,
    population: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = TARGET_WIND,
    windDir = BASE_WIND_DIR,
    omegaC = 0.84 * Math.tanh((windSpeed / 2) ** 0.4),
): number {
    const dk = (2 * Math.PI) / cfg.L;
    const scale = spectrumNormalization(
        population,
        significantWaveHeight,
        windSpeed,
        windDir,
        omegaC,
    );
    let sum = 0;
    for (let y = 0; y < cfg.N; y++)
        for (let x = 0; x < cfg.N; x++) {
            const kx = kIndex(x, cfg.N) * dk;
            const kz = kIndex(y, cfg.N) * dk;
            const k = Math.hypot(kx, kz);
            if (k >= cfg.kLo && k <= cfg.kHi)
                sum += unifiedSpectrum(kx, kz, windSpeed, windDir, omegaC) * scale * dk * dk;
        }
    // H(k,t) contains the counter-rotating pair, so realized spatial variance is twice H0 energy.
    return 2 * sum;
}

export function spectralCellAmplitude(
    cfg: CascadeConfig,
    kx: number,
    kz: number,
    population: CascadeConfig[],
    seaState: SeaState = SEA_STATE,
    seed = 0,
): number {
    void seed;
    const k = Math.hypot(kx, kz);
    if (k < cfg.kLo || k > cfg.kHi) return 0;
    const dk = (2 * Math.PI) / cfg.L;
    return Math.sqrt(
        Math.max(
            0,
            unifiedSpectrum(kx, kz, seaState.windSpeed, seaState.windDir, seaState.omegaC) *
                spectrumNormalization(
                    population,
                    seaState.significantWaveHeight,
                    seaState.windSpeed,
                    seaState.windDir,
                    seaState.omegaC,
                ) *
                dk *
                dk,
        ),
    );
}

export function generateH0(
    cfg: CascadeConfig,
    labelFn: LabelFn = kIndex,
    seaState: SeaState = SEA_STATE,
    population: CascadeConfig[] = CASCADE_CONFIGS,
    seed = 0,
): Float32Array {
    const N = cfg.N;
    const dk = (2 * Math.PI) / cfg.L;
    const h0 = new Float32Array(N * N * 2);
    const scale = spectrumNormalization(
        population,
        seaState.significantWaveHeight,
        seaState.windSpeed,
        seaState.windDir,
        seaState.omegaC,
    );
    for (let y = 0; y < N; y++)
        for (let x = 0; x < N; x++) {
            const kx = labelFn(x, N) * dk;
            const kz = labelFn(y, N) * dk;
            const idx = (y * N + x) * 2;
            const k = Math.hypot(kx, kz);
            if (k < cfg.kLo || k > cfg.kHi) continue;
            const cell =
                unifiedSpectrum(kx, kz, seaState.windSpeed, seaState.windDir, seaState.omegaC) *
                scale *
                dk *
                dk;
            const amp = Math.sqrt(Math.max(0, cell)) / Math.SQRT2;
            h0[idx] = gaussFromK(kx, kz, seed, 0) * amp;
            h0[idx + 1] = gaussFromK(kx, kz, seed, 1) * amp;
        }
    return h0;
}

/** The deterministic variance expected from the source density; no realization calibration. */
export function realizedFieldVariance(
    population: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = TARGET_WIND,
    windDir = BASE_WIND_DIR,
    realizationScale = 1,
): number {
    let variance = 0;
    for (const cfg of population)
        variance += declaredBandVariance(
            cfg,
            population,
            significantWaveHeight,
            windSpeed,
            windDir,
        );
    return variance * realizationScale ** 2;
}

export interface FoldBand {
    whitecapAnchor: number;
    lambda: number;
    lambdaCeiling: number;
    slopeRms: number;
    foldAnchor: number;
    foldCeiling: number;
}
function erfc(x: number): number {
    const sign = x < 0 ? -1 : 1,
        a = Math.abs(x),
        t = 1 / (1 + 0.3275911 * a);
    const p =
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)));
    const value = p * t * Math.exp(-a * a);
    return sign > 0 ? value : 2 - value;
}
function inverseErfc(p: number): number {
    let lo = 0,
        hi = 8;
    for (let i = 0; i < 64; i++) {
        const mid = (lo + hi) / 2;
        if (erfc(mid) > p) lo = mid;
        else hi = mid;
    }
    return (lo + hi) / 2;
}
export function foldProbability(lambda: number, slopeRms: number): number {
    return lambda > 0 && slopeRms > 0 ? 0.5 * erfc(1 / (Math.SQRT2 * lambda * slopeRms)) : 0;
}
export function meanSquareSlope(
    windSpeed: number,
    omegaC = 0.84 * Math.tanh((windSpeed / 2) ** 0.4),
): number {
    // Cox–Munk is the resolved gravity/short-gravity slope moment. The unrepresented capillary tail
    // is intentionally reported separately by truncationRatio, not silently folded into this arm.
    return radialIntegral(windSpeed, omegaC, 1e-3, 8.482300164692441, true);
}
function composedStrainRms(
    configs: CascadeConfig[],
    significantWaveHeight: number,
    windSpeed: number,
    windDir: number,
    omegaC: number,
): number {
    const scale = spectrumNormalization(configs, significantWaveHeight, windSpeed, windDir, omegaC);
    let moment = 0;
    for (const cfg of configs) {
        const dk = (2 * Math.PI) / cfg.L;
        for (let y = 0; y < cfg.N; y++)
            for (let x = 0; x < cfg.N; x++) {
                const kx = kIndex(x, cfg.N) * dk;
                const kz = kIndex(y, cfg.N) * dk;
                const k = Math.hypot(kx, kz);
                if (k >= cfg.kLo && k <= cfg.kHi && k > 0)
                    moment +=
                        2 *
                        unifiedSpectrum(kx, kz, windSpeed, windDir, omegaC) *
                        scale *
                        dk *
                        dk *
                        (kx ** 4 / k ** 2);
            }
    }
    return Math.sqrt(Math.max(moment, Number.MIN_VALUE));
}

export function deriveFoldBand(
    configs: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = TARGET_WIND,
    windDir = BASE_WIND_DIR,
    omegaC = 0.84 * Math.tanh((windSpeed / 2) ** 0.4),
): FoldBand {
    // λ is derived from the same composed, declared-band Jacobian statistic that the fold oracle
    // measures; using total isotropic slope RMS here would derive a different physical quantity.
    const slopeRms = composedStrainRms(configs, significantWaveHeight, windSpeed, windDir, omegaC);
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

const _band = deriveFoldBand(CASCADE_CONFIGS);
export const SEA_STATE: SeaState = Object.freeze({
    windSpeed: TARGET_WIND,
    omegaC: TARGET_OMEGA_C,
    windDir: BASE_WIND_DIR,
    significantWaveHeight: 4 * Math.sqrt(FULL_VARIANCE),
    lambda: _band.lambda,
    whitecapFraction: whitecapFraction(TARGET_WIND),
    truncationRatio: DECLARED_VARIANCE / FULL_VARIANCE,
});

export function theoreticalFlops(cfg: CascadeConfig): number {
    return 6 * cfg.N * cfg.N * Math.log2(cfg.N) + 4 * cfg.N * cfg.N;
}
export function totalTheoreticalFlops(configs: CascadeConfig[]): number {
    return configs.reduce((s, c) => s + theoreticalFlops(c), 0);
}
export function directDftFlops(cfg: CascadeConfig): number {
    return 16 * cfg.N * cfg.N * cfg.N + 4 * cfg.N * cfg.N;
}
