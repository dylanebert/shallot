// JONSWAP-modified Phillips spectrum generation for the FFT ocean substrate.
// Grid resolution is constrained by the radix-2 transform; world-space patch lengths define the
// spatial repeat period independently.

export type LabelFn = (i: number, N: number) => number;

export interface CascadeConfig {
    /** FFT grid resolution (N×N), constrained to a power of two by the radix-2 transform. */
    N: number;
    /** Physical domain size in meters (the patch the N×N grid covers). */
    L: number;
    /** Wind speed in m/s. */
    windSpeed: number;
    /** Wind direction in radians. */
    windDir: number;
    /** Wave amplitude scale. */
    amplitude: number;
    /** Choppiness (horizontal displacement strength). */
    lambda: number;
    /**
     * Declared spectral band (rad/m), applied in `generateH0`. N-INDEPENDENT: this cascade
     * represents wavenumbers in [kLo, kHi] regardless of grid resolution — `kMax = Nπ/L` would make
     * the represented sea a function of N, which is not part of the declared band contract. The
     * two cascades below are declared non-overlapping and contiguous at kSplit=0.75 rad/m: cascade 0
     * owns the swell/large-scale band [dk₀, 0.75], cascade 1 owns the chop/ripple band [0.75, its
     * own Nyquist]. `kLo`/`kHi` define the band independently of the grid resolution.
     */
    kLo: number;
    /** See `kLo`. */
    kHi: number;
}

/**
 * The shipped cascades use `N=64` and `N=128`, the smallest power-of-two choices that preserve their
 * declared bands. Their patch lengths are `L=80` and `L=31`; pairwise-coprime world-space lengths
 * maximize the shared repeat period, which is 2480 m here. `tilePeriod` exposes that least common
 * multiple for callers that intentionally choose repeating patches.
 *
 * The amplitude and choppiness values are configuration inputs. Spectrum normalization and derived
 * physical parameters belong to the caller that selects them.
 */
export const CASCADE_CONFIGS: CascadeConfig[] = [
    {
        N: 64,
        L: 80,
        windSpeed: 18,
        windDir: 0.6,
        amplitude: 0.00005,
        lambda: 1.925088,
        kLo: 0.07853981633974483,
        kHi: 0.75,
    },
    {
        N: 128,
        L: 31,
        windSpeed: 12,
        windDir: 0.6,
        amplitude: 0.00003,
        lambda: 5.492399,
        kLo: 0.75,
        kHi: 8.482300164692441,
    },
];

/** Standard DFT/FFT mode index: `k = (i <= N/2 ? i : i-N) * dk`. Every k-labelled kernel
 * (`generateH0`'s band mask, the CPU reference, and the matching GPU kernels) computes a physical
 * wavenumber this way, and the FFT transforms with the matching unshifted phase `2π·x·k/N`. A mode's
 * physical value is therefore stable when it is represented at more than one resolution. */
export function kIndex(i: number, N: number): number {
    return i <= N / 2 ? i : i - N;
}

/** true iff `n` is a power of two (`n >= 1`). The butterfly FFT's own precondition on `N`. */
export function isPowerOfTwo(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}

/** greatest common divisor for integer patch lengths. */
export function gcd(a: number, b: number): number {
    while (b > 0) [a, b] = [b, a % b];
    return a;
}

/**
 * Assert every cascade's `N` is a power of two — the FFT's own precondition, checked once at module
 * load (below) rather than discovered at the first `assertPowerOfTwo(N)` deep in a kernel factory.
 */
export function assertAllPowerOfTwo(configs: CascadeConfig[]): boolean {
    return configs.every((c) => isPowerOfTwo(c.N));
}

/**
 * tests whether integer patch lengths are pairwise coprime. Pairwise-coprime world-space lengths
 * are the anti-alignment invariant for the shipped cascades.
 */
export function assertCoprimeL(configs: CascadeConfig[]): boolean {
    for (let i = 0; i < configs.length; i++) {
        for (let j = i + 1; j < configs.length; j++) {
            if (!Number.isInteger(configs[i].L) || !Number.isInteger(configs[j].L)) return false;
            if (gcd(configs[i].L, configs[j].L) !== 1) return false;
        }
    }
    return true;
}

/** Least common multiple for integer patch lengths (`lcm(a,b) = a*b/gcd(a,b)`). */
function lcm(a: number, b: number): number {
    return (a * b) / gcd(a, b);
}

/** the world-space distance at which all cascade boundaries re-align, folded across every `L`.
 * `Infinity` for a single cascade. Consumers can use this value when sizing periodic resources. */
export function tilePeriod(configs: CascadeConfig[]): number {
    if (configs.length < 2) return Number.POSITIVE_INFINITY;
    return configs.reduce((period, c) => lcm(period, c.L), 1);
}

if (!assertAllPowerOfTwo(CASCADE_CONFIGS)) {
    throw new Error(
        `shallot-ocean: every cascade N must be a power of two for the butterfly FFT — got ${CASCADE_CONFIGS.map((c) => c.N).join(", ")}`,
    );
}

/** exported so a caller deriving a quantity from the dominant wave period shares this one value
 *  rather than re-declaring it. */
export const G = 9.81;

/**
 * H0 is a deterministic hash of the WAVENUMBER `(kx, kz)`, not of the grid — so a given physical
 * wavenumber draws the identical amplitude/phase pair at every `N` that can represent it, which is
 * what cross-resolution comparisons depend on: it operates purely in the frequency domain occupied
 * by `generateH0`.
 *
 * The mixer is a murmur3-style avalanche over the raw IEEE-754 bits of kx/kz plus a salt.
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

/** Box-Muller transform for a standard normal random variable, drawn deterministically from the
 * wavenumber `(kx, kz)` plus a salt selecting which of H0's draws (real/imag) this is. */
function gaussFromK(kx: number, kz: number, salt: number): number {
    const u1 = Math.max(hashBits(kx, kz, salt * 2), 1e-10);
    const u2 = hashBits(kx, kz, salt * 2 + 1);
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * JONSWAP-modified Phillips spectrum P(k). Exported so a parity witness can build an energy-weighted
 * lag-1 autocorrelation prediction directly from the declared spectral shape.
 *
 * Phillips base: A·exp(-1/(kL)²) / k⁴ · |k̂·D̂|²
 * JONSWAP enhancement: γ^r where r = exp(-((ω-ω_p)/(σ·ω_p))²/2)
 */
export function philips(kx: number, kz: number, cfg: CascadeConfig): number {
    const k2 = kx * kx + kz * kz;
    if (k2 < 1e-12) return 0;
    const k = Math.sqrt(k2);
    const Lwind = (cfg.windSpeed * cfg.windSpeed) / G;
    const kHatX = kx / k;
    const kHatZ = kz / k;
    const dx = Math.cos(cfg.windDir);
    const dz = Math.sin(cfg.windDir);
    const dot = kHatX * dx + kHatZ * dz;
    const phillips =
        (cfg.amplitude / (k2 * k2)) * Math.exp(-1 / (k * Lwind * (k * Lwind))) * dot * dot;

    // JONSWAP peak enhancement
    const omega = Math.sqrt(G * k);
    const omegaP = G / cfg.windSpeed;
    const gamma = 3.3;
    const sigma = omega <= omegaP ? 0.07 : 0.09;
    const r = Math.exp(-(((omega - omegaP) / (sigma * omegaP)) ** 2) / 2);
    return phillips * gamma ** r;
}

/**
 * Generate the Fourier-domain initial height field H₀(k) for one cascade.
 * Returns a flat Float32Array of length N*N*2 (interleaved real, imag per element).
 *
 * H₀(k) = (1/√2)·(ξᵣ + i·ξᵢ)·√P(k). The bin-width normalization is left to the caller's
 * spectrum calibration; `lambda` and `amplitude` are configuration inputs.
 *
 * `cfg.kLo`/`cfg.kHi` (rad/m, N-independent) zero every mode outside the cascade's declared band.
 * `labelFn` defaults to `kIndex`; supplying another label is useful for independent diagnostic
 * comparisons, but does not change the transform's phase convention.
 * The conjugate H₀*(-k) needed for time evolution is derived in the GPU update kernel by looking up
 * H₀ at index (-k) and conjugating — so we only store H₀.
 */
export function generateH0(cfg: CascadeConfig, labelFn: LabelFn = kIndex): Float32Array {
    const N = cfg.N;
    const dk = (2 * Math.PI) / cfg.L;
    const h0 = new Float32Array(N * N * 2);

    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const kx = labelFn(x, N) * dk;
            const kz = labelFn(y, N) * dk;
            const idx = (y * N + x) * 2;
            const kMag = Math.sqrt(kx * kx + kz * kz);

            if (kMag < cfg.kLo || kMag > cfg.kHi) {
                h0[idx] = 0;
                h0[idx + 1] = 0;
                continue;
            }

            const p = philips(kx, kz, cfg);
            const sqrtP = Math.sqrt(Math.max(p, 0)) / Math.SQRT2;

            // salts 0/1 select the real/imag draw; each draw itself consumes two hashes (u1, u2)
            // via gaussFromK's own *2/*2+1 split — the (kx, kz) pair is the only seed.
            const xiR = gaussFromK(kx, kz, 0);
            const xiI = gaussFromK(kx, kz, 1);

            h0[idx] = xiR * sqrtP;
            h0[idx + 1] = xiI * sqrtP;
        }
    }

    return h0;
}

/**
 * Theoretical FFT butterfly FLOP count for one cascade (2D inverse FFT — one N-point FFT per
 * row, one per column, N of each).
 * Each butterfly: 1 complex multiply (4 mul + 2 add) + 1 complex add (2 add) = ~6 FLOPs.
 * Per dimension: N transforms × (N/2)·log₂(N) butterflies → 3·N²·log₂(N) FLOPs.
 * Two dimensions: 6·N²·log₂(N) FLOPs.
 * Plus the displacement/Jacobian post-process: ~4·N² FLOPs (finite differences).
 * Total per cascade: 6·N²·log₂(N) + 4·N².
 */
export function theoreticalFlops(cfg: CascadeConfig): number {
    const N = cfg.N;
    const log2N = Math.log2(N);
    const fftFlops = 6 * N * N * log2N; // N row FFTs + N column FFTs, ~6 FLOPs/butterfly
    const postFlops = 4 * N * N; // finite-difference displacement + Jacobian
    return fftFlops + postFlops;
}

/** Total theoretical FLOPs across all cascades. */
export function totalTheoreticalFlops(configs: CascadeConfig[]): number {
    return configs.reduce((sum, c) => sum + theoreticalFlops(c), 0);
}

/**
 * The flop cost a direct O(N²)-per-dimension DFT
 * would pay at this cascade's `N` — kept for the achieved-vs-legacy comparison the ALU-ratio reading
 * cites (`totalTheoreticalFlops` is this package's own FFT cost; this is the counterfactual it
 * replaced). Per dimension pass: N² output elements, each an N-term complex sum, 8 FLOPs/term
 * (4 mul + 4 add; the 2 transcendentals per term are excluded, matching `theoreticalFlops`'
 * convention). 8·N³ per dimension, 16·N³ both, plus the 4·N² post-process.
 */
export function directDftFlops(cfg: CascadeConfig): number {
    const N = cfg.N;
    const dftFlops = 8 * N * N * N * 2; // N² outputs × N terms × 8 FLOPs, two dimensions
    const postFlops = 4 * N * N;
    return dftFlops + postFlops;
}
