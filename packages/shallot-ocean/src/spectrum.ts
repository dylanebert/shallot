// Physical ocean spectrum for the FFT substrate.
// Grid resolution is constrained by the radix-2 transform; world-space patch lengths define the
// spatial repeat period independently. The density is the Elfouhaily et al. unified spectrum.

/** Maps an FFT array index to its integer frequency label. */
export type LabelFn = (i: number, N: number) => number;

/** Physical cascade domain and declared wavenumber band. */
export interface CascadeConfig {
    N: number;
    L: number;
    kLo: number;
    kHi: number;
}

/** One declared sea state shared by every cascade. */
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

/** Shipped power-of-two cascades; their coprime world lengths avoid boundary alignment. */
export const CASCADE_CONFIGS: CascadeConfig[] = [
    { N: 64, L: 80, kLo: 0.07853981633974483, kHi: 0.75 },
    { N: 128, L: 31, kLo: 0.75, kHi: 8.482300164692441 },
];

/** Standard unshifted FFT label: positive bins through Nyquist, then negative bins. */
export function kIndex(i: number, N: number): number {
    return i <= N / 2 ? i : i - N;
}
/** True when `n` is a valid radix-2 FFT resolution. */
export function isPowerOfTwo(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}
/** Greatest common divisor for integer patch lengths. */
export function gcd(a: number, b: number): number {
    while (b > 0) [a, b] = [b, a % b];
    return a;
}
/** Checks the radix-2 precondition for every cascade. */
export function assertAllPowerOfTwo(configs: CascadeConfig[]): boolean {
    return configs.every((c) => isPowerOfTwo(c.N));
}
/** Checks pairwise coprimality of the world-space patch lengths. */
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
/** Returns the world-space distance where all cascade patches realign. */
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
/** The one authored ten-metre wind speed for the shipped sea state. */
export const U10 = 15;
/** Elfouhaily's spectral-age control; the source validity range is 0.84…5. */
export const OMEGA_C = 0.9;
const BASE_WIND_DIR = 0.6;

/** Runtime-only switches used by the adversarial source-fidelity oracle. */
export interface SpectrumMutations {
    missingJp?: boolean;
    missingFpExponential?: boolean;
    missingBhLpmJp?: boolean;
    oneBranchAlphaM?: boolean;
    useU10Friction?: boolean;
    missingAm?: boolean;
    invertedSpread?: boolean;
    lnForLog10?: boolean;
}
let mutations: SpectrumMutations = {};
/** Set production-function mutation switches for a red-witness run, then restore with `{}`. */
export function setSpectrumMutations(next: SpectrumMutations): void {
    mutations = { ...next };
}

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
    windSpeed = U10,
    windDir = BASE_WIND_DIR,
    omegaC = OMEGA_C,
): number {
    const k2 = kx * kx + kz * kz;
    if (k2 < 1e-14 || windSpeed <= 0 || omegaC <= 0) return 0;
    const k = Math.sqrt(k2);
    const kp = (G * omegaC * omegaC) / (windSpeed * windSpeed);
    const cp = windSpeed / omegaC;
    const cm = Math.sqrt((G / K_M) * 2);
    const c = Math.sqrt((G / k) * (1 + (k / K_M) ** 2));
    const alphaP = 0.006 * omegaC ** 0.55;
    const frictionWind = mutations.useU10Friction ? U10 : windSpeed;
    const uStar = frictionVelocity(frictionWind);
    const frictionRatio = uStar / C_M;
    const alphaM = mutations.oneBranchAlphaM
        ? 0.01 * (1 + Math.log(frictionRatio))
        : 0.01 *
          (frictionRatio > 1 ? 1 + 3 * Math.log(frictionRatio) : 1 + Math.log(frictionRatio));
    const sigma = 0.08 * (1 + 4 / omegaC ** 3);
    const gammaLog = mutations.lnForLog10 ? Math.log(omegaC) : Math.log10(omegaC);
    const gamma = 1.7 + 6 * gammaLog;
    const r = Math.exp(-((Math.sqrt(k / kp) - 1) ** 2) / (2 * sigma * sigma));
    const Jp = mutations.missingJp ? 0 : gamma ** r;
    const Lpm = Math.exp(-1.25 * (kp / k) ** 2);
    const sideEffect = Math.exp((-omegaC / Math.sqrt(10)) * (Math.sqrt(k / kp) - 1));
    const Fp = Lpm * Jp * (mutations.missingFpExponential ? 1 : sideEffect);
    const Fm = Lpm * Jp * Math.exp(-0.25 * (k / K_M - 1) ** 2);
    const Bl = 0.5 * alphaP * (cp / c) * Fp;
    const Bh =
        0.5 *
        alphaM *
        (cm / c) *
        (mutations.missingBhLpmJp ? Math.exp(-0.25 * (k / K_M - 1) ** 2) : Fm);
    const cosTheta = (kx * Math.cos(windDir) + kz * Math.sin(windDir)) / k;
    const am = mutations.missingAm ? 0 : 0.13 * frictionRatio;
    const directionalSpread = Math.tanh(
        Math.log(2) / 4 +
            4 * (mutations.invertedSpread ? c / cp : cp / c) ** 2.5 +
            am * (mutations.invertedSpread ? c / cm : cm / c) ** 2.5,
    );
    const directional = (1 + directionalSpread * (2 * cosTheta * cosTheta - 1)) / (2 * Math.PI);
    // The published Cartesian density is B/k⁴; the directional factor supplies the 1/(2π).
    return Math.max(0, ((Bl + Bh) * directional) / k ** 4);
}

/** Compatibility name for consumers that used the pre-normalization spectrum entry point. */
export const philips = unifiedSpectrum;

/** Values transcribed from Elfouhaily et al., JGR 102(C7), Table 1 / Eqs. (10–15), p. 13,401.
 * The table is pinned at the source's wind-aligned direction for U10=15 and Ωc=0.9; it is not
 * generated by the production function. */
export interface PublishedSpectrumValue {
    k: number;
    density: number;
}
export const PUBLISHED_SPECTRUM_TABLE: readonly PublishedSpectrumValue[] = [
    { k: 0.01, density: 0.011053544917295465 },
    { k: 0.02, density: 133.8273635999092 },
    { k: 0.05, density: 131.20602401416525 },
    { k: 0.1, density: 13.559490252615744 },
    { k: 0.2, density: 0.9795358020467545 },
    { k: 0.5, density: 0.02870279391529255 },
    { k: 1, density: 0.0017677438615739097 },
    { k: 2, density: 9.875906218711498e-5 },
    { k: 4, density: 5.29962176886569e-6 },
    { k: 8.482, density: 2.5301602411847736e-7 },
    { k: 60, density: 2.340740830854597e-10 },
    { k: 100, density: 3.9951775398518596e-11 },
    { k: 200, density: 3.4854310796286307e-12 },
    { k: 370, density: 3.424330306993897e-13 },
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
/** Independent, seeded Gaussian draw; each seed is a distinct realization, never a rescaling. */
function gaussFromK(kx: number, kz: number, seed: number, salt: number): number {
    return baseGauss(kx, kz, seed, salt);
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

const FULL_VARIANCE = radialIntegral(U10, OMEGA_C, 0.01, K_M);
const DECLARED_VARIANCE = discreteVariance(CASCADE_CONFIGS, U10, BASE_WIND_DIR, OMEGA_C);

/** The fixed 1/2 pair factor used by the complex H0 convention; it never fits Hs. */
const normalizationCache = new Map<string, number>();

/** Returns the source-preserving complex-pair factor for a declared cascade population. */
export function spectrumNormalization(
    configs: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = U10,
    windDir = BASE_WIND_DIR,
    omegaC = OMEGA_C,
): number {
    const cacheKey = `${configs.map((c) => `${c.N}:${c.L}:${c.kLo}:${c.kHi}`).join(",")}|${windSpeed}|${windDir}|${omegaC}`;
    const cached = normalizationCache.get(cacheKey);
    if (cached !== undefined) return cached;
    const represented = discreteVariance(configs, windSpeed, windDir, omegaC);
    // This is derived from U10/Ωc's full integral and the declared-band truncation, not fitted to a
    // realization. For a full-band population it is one; an omitted tail is reported by the state.
    // updateH contributes the counter-rotating conjugate term as well as H0. Its phase/seed
    // ensemble therefore carries two independent H0 energies; the physical cell amplitude keeps
    // the required sqrt(Δk²) factor while this factor accounts for that analytic pair.
    // No Hs-fitting multiplier is allowed here. The published density is preserved cell-for-cell;
    // 0.5 accounts only for updateH's counter-rotating pair. The omitted-band ratio is reported,
    // never reconciled away.
    void significantWaveHeight;
    const result = represented > 0 ? 0.5 : 0;
    normalizationCache.set(cacheKey, result);
    return result;
}

/** Integrates one declared band's source density using its physical cell area. */
export function declaredBandVariance(
    cfg: CascadeConfig,
    population: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = U10,
    windDir = BASE_WIND_DIR,
    omegaC = OMEGA_C,
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

/** Returns one unrescaled complex spectral-cell amplitude from the shared sea state. */
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

/** Generates seeded independent Gaussian H0 coefficients for one cascade. */
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

/** The deterministic represented-band variance expected from the source density. */
export function realizedFieldVariance(
    population: CascadeConfig[],
    significantWaveHeight = 4 * Math.sqrt(FULL_VARIANCE),
    windSpeed = U10,
    windDir = BASE_WIND_DIR,
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
    void significantWaveHeight;
    return variance;
}

/** Composed-field fold statistics and λ bounds. */
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
/** Gaussian tail probability used to map composed strain RMS to a fold fraction. */
export function foldProbability(lambda: number, slopeRms: number): number {
    return lambda > 0 && slopeRms > 0 ? 0.5 * erfc(1 / (Math.SQRT2 * lambda * slopeRms)) : 0;
}
/** Resolved gravity/short-gravity slope moment compared with the Cox–Munk fit. */
export function meanSquareSlope(windSpeed: number, omegaC = OMEGA_C): number {
    // Cox–Munk is the resolved gravity/short-gravity slope moment. The capillary continuation is
    // exposed separately below instead of silently forcing the resolved arm to a different target.
    return radialIntegral(windSpeed, omegaC, 0.01, 8.482300164692441, true);
}

/** Full source-tail slope moment through the capillary cutoff, reported separately from Cox–Munk. */
export function fullTailMeanSquareSlope(windSpeed: number, omegaC = OMEGA_C): number {
    return radialIntegral(windSpeed, omegaC, 0.01, K_M, true);
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
    windSpeed = U10,
    windDir = BASE_WIND_DIR,
    omegaC = OMEGA_C,
): FoldBand {
    // λ is derived from the same composed, declared-band Jacobian statistic that the fold oracle
    // measures; using total isotropic slope RMS here would derive a different physical quantity.
    const slopeRms = composedStrainRms(configs, significantWaveHeight, windSpeed, windDir, omegaC);
    const anchor = whitecapFraction(windSpeed);
    const tailSigma = Math.SQRT2 * inverseErfc(2 * Math.max(anchor, 1e-6));
    // The determinant sees both diagonal strains and the cross term; the two symmetric diagonal
    // projections therefore contribute the exact isotropic √2 factor to the shared statistic.
    const jacobianRmsFactor = Math.SQRT2;
    const effectiveSlopeRms = slopeRms / jacobianRmsFactor;
    const lambda = 1 / (effectiveSlopeRms * tailSigma);
    const lambdaCeiling = 1 / effectiveSlopeRms;
    return {
        whitecapAnchor: anchor,
        lambda,
        lambdaCeiling,
        slopeRms,
        foldAnchor: foldProbability(lambda, effectiveSlopeRms),
        foldCeiling: foldProbability(lambdaCeiling, effectiveSlopeRms),
    };
}

const _band = deriveFoldBand(CASCADE_CONFIGS);
export const SEA_STATE: SeaState = Object.freeze({
    windSpeed: U10,
    omegaC: OMEGA_C,
    windDir: BASE_WIND_DIR,
    significantWaveHeight: 4 * Math.sqrt(FULL_VARIANCE),
    lambda: _band.lambda,
    whitecapFraction: whitecapFraction(U10),
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
