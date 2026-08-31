// Physical Elfouhaily ocean spectrum for the FFT substrate. Grid resolution controls sampling;
// world-space patch length controls the repeat period, and the declared physical band is independent
// of both.

/** Maps an FFT array index to its integer frequency label. */
export type LabelFn = (i: number, N: number) => number;

/** Physical cascade domain and declared wavenumber band. */
export interface CascadeConfig {
    N: number;
    L: number;
    /** Shared sea-state wind mirrored for compatibility with the committed spike harness. */
    windSpeed: number;
    /** Shared sea-state wind direction mirrored for compatibility with the committed spike harness. */
    windDir: number;
    /** Existing displacement strength; physical derivation belongs to the later fold stage. */
    lambda: number;
    kLo: number;
    kHi: number;
}

/** One physical sea state shared by every cascade. */
export interface SeaState {
    /** Ten-metre wind speed in m/s. */
    windSpeed: number;
    /** Dimensionless inverse wave age, valid over 0.84…5. */
    omegaC: number;
    /** Wind direction in radians. */
    windDir: number;
    /** Derived from the full published height-density integral. */
    significantWaveHeight: number;
    /** Discrete declared-band variance divided by full published variance. */
    truncationRatio: number;
}

/** Shipped power-of-two cascades; coprime world lengths avoid aligned repeat boundaries. */
export const CASCADE_CONFIGS: CascadeConfig[] = [
    {
        N: 64,
        L: 80,
        windSpeed: 15,
        windDir: 0.6,
        lambda: 1.925088,
        kLo: 0.07853981633974483,
        kHi: 0.75,
    },
    {
        N: 128,
        L: 31,
        windSpeed: 15,
        windDir: 0.6,
        lambda: 5.492399,
        kLo: 0.75,
        kHi: 8.482300164692441,
    },
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
export function assertAllPowerOfTwo(configs: readonly CascadeConfig[]): boolean {
    return configs.every((config) => isPowerOfTwo(config.N));
}

/** Checks pairwise coprimality of the world-space patch lengths. */
export function assertCoprimeL(configs: readonly CascadeConfig[]): boolean {
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
export function tilePeriod(configs: readonly CascadeConfig[]): number {
    if (configs.length < 2) return Number.POSITIVE_INFINITY;
    return configs.reduce((period, config) => lcm(period, config.L), 1);
}

/** Gravitational acceleration in m/s². */
export const G = 9.81;
/** The authored ten-metre wind speed for the shipped sea state. */
export const U10 = 15;
/** Elfouhaily inverse wave age for the shipped sea state. */
export const OMEGA_C = 0.9;
const WIND_DIR = 0.6;
const WATER_DENSITY = 1025;
const SURFACE_TENSION = 0.074;
/** Gravity-capillary transition wavenumber in rad/m. */
export const K_M = Math.sqrt((WATER_DENSITY * G) / SURFACE_TENSION);
/** Minimum gravity-capillary phase speed in m/s. */
export const C_M = Math.sqrt((2 * G) / K_M);

/** Named factor removals used only by the production mutation oracle. */
export interface SpectrumMutation {
    constantDensity?: boolean;
    kMinusThreeDensity?: boolean;
    missingJp?: boolean;
    missingFpExponential?: boolean;
    missingBhEnvelope?: boolean;
    oneBranchAlphaM?: boolean;
    useU10ForFriction?: boolean;
    missingAm?: boolean;
    invertedSpread?: boolean;
    lnForLog10?: boolean;
}

/** Named realization defects used only by the production-path red witnesses. */
export interface RealizationMutation {
    omitCellArea?: boolean;
    rescalePerRealization?: boolean;
}

/** Neutral drag estimate used to realize Elfouhaily's `u* = √Cd10N U10` relation. */
export function frictionVelocity(windSpeed: number): number {
    if (windSpeed <= 0) return 0;
    const drag = (0.8 + 0.065 * windSpeed) * 1e-3;
    return windSpeed * Math.sqrt(drag);
}

function phaseSpeed(k: number): number {
    return Math.sqrt((G / k) * (1 + (k / K_M) ** 2));
}

function curvatureSpectrum(
    k: number,
    windSpeed: number,
    omegaC: number,
    mutation: SpectrumMutation,
): number {
    const kp = (G * omegaC * omegaC) / (windSpeed * windSpeed);
    const c = phaseSpeed(k);
    const cp = phaseSpeed(kp);
    const uStar = frictionVelocity(mutation.useU10ForFriction ? U10 : windSpeed);
    const frictionRatio = uStar / C_M;
    const alphaP = 0.006 * omegaC ** 0.55;
    const alphaM = mutation.oneBranchAlphaM
        ? 0.01 * (1 + Math.log(frictionRatio))
        : 0.01 *
          (frictionRatio > 1 ? 1 + 3 * Math.log(frictionRatio) : 1 + Math.log(frictionRatio));
    const sigma = 0.08 * (1 + 4 * omegaC ** -3);
    const gammaLog = mutation.lnForLog10 ? Math.log(omegaC) : Math.log10(omegaC);
    const gamma = omegaC <= 1 ? 1.7 : 1.7 + 6 * gammaLog;
    const peakWeight = Math.exp(-((Math.sqrt(k / kp) - 1) ** 2) / (2 * sigma * sigma));
    const jp = mutation.missingJp ? 1 : gamma ** peakWeight;
    const lpm = Math.exp(-1.25 * (kp / k) ** 2);
    const fpExponential = Math.exp(-(omegaC / Math.sqrt(10)) * (Math.sqrt(k / kp) - 1));
    const fp = lpm * jp * (mutation.missingFpExponential ? 1 : fpExponential);
    const fmCutoff = Math.exp(-0.25 * (k / K_M - 1) ** 2);
    const fm = mutation.missingBhEnvelope ? fmCutoff : lpm * jp * fmCutoff;
    const bl = 0.5 * alphaP * (cp / c) * fp;
    const bh = 0.5 * alphaM * (C_M / c) * fm;
    return bl + bh;
}

/**
 * Production Elfouhaily Cartesian height density at `(kx, kz)`. The implementation follows the
 * published curvature branches and converts `S(k) = B(k)/k³` through eq. (45), `Ψ = SΦ/k`.
 */
export function directionalDensity(
    kx: number,
    kz: number,
    windSpeed = U10,
    windDir = WIND_DIR,
    omegaC = OMEGA_C,
    mutation: SpectrumMutation = {},
): number {
    const k = Math.hypot(kx, kz);
    if (k < 1e-12 || windSpeed <= 0 || omegaC < 0.84 || omegaC > 5) return 0;
    if (mutation.constantDensity) return 1;
    if (mutation.kMinusThreeDensity) return k ** -3;

    const c = phaseSpeed(k);
    const kp = (G * omegaC * omegaC) / (windSpeed * windSpeed);
    const cp = phaseSpeed(kp);
    const uStar = frictionVelocity(mutation.useU10ForFriction ? U10 : windSpeed);
    const am = mutation.missingAm ? 0 : (0.13 * uStar) / C_M;
    const longRatio = mutation.invertedSpread ? cp / c : c / cp;
    const shortRatio = mutation.invertedSpread ? c / C_M : C_M / c;
    const delta = Math.tanh(Math.log(2) / 4 + 4 * longRatio ** 2.5 + am * shortRatio ** 2.5);
    const theta = Math.atan2(kz, kx);
    const angular = (1 + delta * Math.cos(2 * (theta - windDir))) / (2 * Math.PI);
    return Math.max(0, (curvatureSpectrum(k, windSpeed, omegaC, mutation) * angular) / k ** 4);
}

function assertCascadeSeaState(
    config: CascadeConfig,
    seaState: Pick<SeaState, "windSpeed" | "windDir">,
): void {
    if (config.windSpeed !== seaState.windSpeed || config.windDir !== seaState.windDir) {
        throw new Error("shallot-ocean: cascade wind must match the shared sea state");
    }
}

/** Compatibility name used by the committed spike's spectral diagnostic. */
export function philips(kx: number, kz: number, config: CascadeConfig): number {
    assertCascadeSeaState(config, SEA_STATE);
    return directionalDensity(kx, kz, SEA_STATE.windSpeed, SEA_STATE.windDir, SEA_STATE.omegaC);
}

function integrateRadialMoment(windSpeed: number, omegaC: number, slope: boolean): number {
    const steps = 4096;
    const logLo = Math.log(0.01);
    const dLog = Math.log(K_M / 0.01) / steps;
    let sum = 0;
    for (let i = 0; i < steps; i++) {
        const k = Math.exp(logLo + (i + 0.5) * dLog);
        const b = curvatureSpectrum(k, windSpeed, omegaC, {});
        // Height: S(k) dk = B(k)/k³ · k d(log k). Slope adds k², leaving B d(log k).
        sum += (slope ? b : b / (k * k)) * dLog;
    }
    return sum;
}

/** Full published mean-square-slope integral through the gravity-capillary peak. */
export function fullMeanSquareSlope(windSpeed: number, omegaC = OMEGA_C): number {
    return integrateRadialMoment(windSpeed, omegaC, true);
}

/** Integrates production density over one discrete cascade band, including Fourier-cell area. */
export function declaredBandVariance(
    config: CascadeConfig,
    seaState: Pick<SeaState, "windSpeed" | "windDir" | "omegaC"> = SEA_STATE,
): number {
    const dk = (2 * Math.PI) / config.L;
    let sum = 0;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = kIndex(x, config.N) * dk;
            const kz = kIndex(y, config.N) * dk;
            const k = Math.hypot(kx, kz);
            if (k < config.kLo || k > config.kHi) continue;
            sum +=
                directionalDensity(kx, kz, seaState.windSpeed, seaState.windDir, seaState.omegaC) *
                dk *
                dk;
        }
    }
    return sum;
}

const FULL_VARIANCE = integrateRadialMoment(U10, OMEGA_C, false);
const DECLARED_VARIANCE = CASCADE_CONFIGS.reduce(
    (sum, config) =>
        sum + declaredBandVariance(config, { windSpeed: U10, windDir: WIND_DIR, omegaC: OMEGA_C }),
    0,
);

/** Short-gravity/capillary cascade configuration; it is slope-only and contributes no displacement. */
export const SLOPE_CASCADE_CONFIGS: readonly CascadeConfig[] = Object.freeze([
    {
        N: 256,
        L: 13,
        windSpeed: U10,
        windDir: WIND_DIR,
        lambda: 0,
        kLo: 8.482300164692441,
        kHi: 60,
    },
]);

/** Every cascade domain, including slope-only domains, for shared invariant checks. */
export const ALL_CASCADE_CONFIGS: readonly CascadeConfig[] = Object.freeze([
    ...CASCADE_CONFIGS,
    ...SLOPE_CASCADE_CONFIGS,
]);

if (!assertAllPowerOfTwo(ALL_CASCADE_CONFIGS)) {
    throw new Error("shallot-ocean: cascade N must be powers of two");
}
if (!assertCoprimeL(ALL_CASCADE_CONFIGS)) {
    throw new Error("shallot-ocean: cascade patch lengths must be pairwise coprime");
}

/** Shared physical sea state; significant wave height is derived rather than independently authored. */
export const SEA_STATE: SeaState = Object.freeze({
    windSpeed: U10,
    omegaC: OMEGA_C,
    windDir: WIND_DIR,
    significantWaveHeight: 4 * Math.sqrt(FULL_VARIANCE),
    truncationRatio: DECLARED_VARIANCE / FULL_VARIANCE,
});

for (const config of ALL_CASCADE_CONFIGS) assertCascadeSeaState(config, SEA_STATE);

function hashBits(kx: number, kz: number, seed: number, salt: number): number {
    const bytes = new ArrayBuffer(24);
    const view = new DataView(bytes);
    view.setFloat64(0, kx);
    view.setFloat64(8, kz);
    view.setUint32(16, seed >>> 0);
    view.setUint32(20, salt >>> 0);
    let hash = (0x9e3779b9 ^ seed ^ salt) >>> 0;
    for (let i = 0; i < 6; i++) {
        hash = Math.imul(hash ^ view.getUint32(i * 4), 0x85ebca6b);
        hash ^= hash >>> 13;
        hash = Math.imul(hash, 0xc2b2ae35);
        hash ^= hash >>> 16;
    }
    return (hash >>> 0) / 4294967296;
}

function gaussian(kx: number, kz: number, seed: number, salt: number): number {
    const u1 = Math.max(hashBits(kx, kz, seed, salt * 2), 1e-10);
    const u2 = hashBits(kx, kz, seed, salt * 2 + 1);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Generates an unrescaled seeded Fourier realization. The `Δk²` cell area converts the published
 * Cartesian density to a mode weight; the one-half pair factor accounts for time evolution's
 * counter-rotating `H0(k)` and conjugate `H0(-k)` terms.
 */
export function generateH0(
    config: CascadeConfig,
    seed = 0,
    labelFn: LabelFn = kIndex,
    seaState: SeaState = SEA_STATE,
    mutation: RealizationMutation = {},
): Float32Array {
    const dk = (2 * Math.PI) / config.L;
    const h0 = new Float32Array(config.N * config.N * 2);
    let expectedEnergy = 0;
    let realizedEnergy = 0;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = labelFn(x, config.N) * dk;
            const kz = labelFn(y, config.N) * dk;
            const index = (y * config.N + x) * 2;
            const k = Math.hypot(kx, kz);
            if (k < config.kLo || k > config.kHi) continue;
            const density = directionalDensity(
                kx,
                kz,
                seaState.windSpeed,
                seaState.windDir,
                seaState.omegaC,
            );
            const correctCell = density * dk * dk;
            const cell = mutation.omitCellArea ? density : correctCell;
            const amplitude = Math.sqrt(Math.max(cell, 0)) / 2;
            const real = gaussian(kx, kz, seed, 0) * amplitude;
            const imaginary = gaussian(kx, kz, seed, 1) * amplitude;
            h0[index] = real;
            h0[index + 1] = imaginary;
            expectedEnergy += correctCell / 2;
            realizedEnergy += real * real + imaginary * imaginary;
        }
    }
    if (mutation.rescalePerRealization && realizedEnergy > 0) {
        const scale = Math.sqrt(expectedEnergy / realizedEnergy);
        for (let i = 0; i < h0.length; i++) h0[i] *= scale;
    }
    return h0;
}

/** Theoretical FFT butterfly FLOP count for one cascade. */
export function theoreticalFlops(config: CascadeConfig): number {
    return 6 * config.N * config.N * Math.log2(config.N) + 4 * config.N * config.N;
}

/** Total theoretical FLOPs across all cascades. */
export function totalTheoreticalFlops(configs: CascadeConfig[]): number {
    return configs.reduce((sum, config) => sum + theoreticalFlops(config), 0);
}

/** Counterfactual direct two-dimensional DFT FLOP count. */
export function directDftFlops(config: CascadeConfig): number {
    return 16 * config.N * config.N * config.N + 4 * config.N * config.N;
}
