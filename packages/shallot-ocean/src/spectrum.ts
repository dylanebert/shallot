// JONSWAP-modified Phillips spectrum generation for the FFT ocean substrate.
// Ported from the water-surface spike's `spectrum.ts` (I1) — the spike's O(N²) direct DFT becomes a
// butterfly FFT here, which is what forces `N` to power-of-two on every cascade. The spike's cascade
// 1 ran at `N=81` (chosen for coprimality with cascade 0's `N=64`); that coprimality argument was
// itself the defect the Locked decision names: **coprimality belongs to the world-space patch length
// `L`, never to the grid resolution `N`** — reading it onto `N` is what cost the spike its FFT. Both
// `N` are now powers of two (64, 128); `L` stays untouched (see `CASCADE_CONFIGS`'s own docblock for
// why this stage does not also move `L` to a coprime pair — the foam pass's tiling domain is what
// stopped it, and it is a documented, deliberate deferral, not an oversight).

export interface CascadeConfig {
    /** FFT grid resolution (N×N). Power-of-two on every cascade — the butterfly FFT's own
     * requirement (I1). Cross-cascade tile alignment is NOT this field's concern (see `L`). */
    N: number;
    /** Physical domain size in meters (the patch the N×N grid covers). Coprime across cascades
     * (as integers) to prevent cross-cascade tile alignment — this is the field the spike's
     * `N`-coprimality argument actually belonged to (Locked decision). */
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
     * the represented sea a function of N, which is exactly what the spike's S2h stage closed. The
     * two cascades below are declared non-overlapping and contiguous at kSplit=0.75 rad/m: cascade 0
     * owns the swell/large-scale band [dk₀, 0.75], cascade 1 owns the chop/ripple band [0.75, its
     * own Nyquist]. `kLo`/`kHi` are carried forward byte-for-byte from the spike (only `N`/`L`
     * changed at I1) — I1 does not touch spectrum normalization or the fold band (I2's scope).
     */
    kLo: number;
    /** See `kLo`. */
    kHi: number;
}

/**
 * I1 — both cascades moved to power-of-two `N` for the butterfly FFT. Cascade 0 keeps its spike
 * values verbatim (`N=64` was already a power of two). Cascade 1 moves `N: 81 → 128`; `L` stays
 * `30`, UNCHANGED. `N=128` is the smallest power of two whose Nyquist (`kNyquist = Nπ/L = 128π/30 ≈
 * 13.40 rad/m`) still covers the unchanged declared band's `kHi` (8.4823 rad/m) — the spike's own
 * `N=81,L=30` pair sat exactly AT its Nyquist edge (`81π/30 = 8.4823 = kHi`, zero headroom); this
 * pair carries real headroom above `kHi` instead, a byproduct of the smallest-power-of-two choice,
 * not a retuned target.
 *
 * **`L` deliberately stays untouched, and cross-cascade non-alignment is NOT strict coprimality
 * here.** The Locked decision names `L` (not `N`) as where a non-alignment guarantee belongs, but
 * does not mandate `gcd(L₀, L₁) = 1` — the shipped pair (`80, 30`) shares a factor of 10
 * (`gcd = 10`, `lcm = 240`), a 240 m repeat period the spike's own foam pass (`spike/src/foam.ts`)
 * already built its tiling domain around and no prior stage flagged as a visible defect. Moving `L₁`
 * to a value coprime with 80 (e.g. 31) was tried and reverted: it forces `lcm(L₀, L₁) = L₀·L₁`
 * (2480 m instead of 240 m), which blows up the foam pass's own domain/texel budget — a foam-side
 * recalibration outside I1's scope (foam is I4's territory) for a property this spec does not
 * actually gate. `tilePeriod` below reports the real (unchanged) 240 m period; `assertCoprimeL` is
 * kept as a general-purpose utility for a later stage that chooses to decorrelate `L` further, not
 * asserted true for the shipped configs (see the spec's Residue).
 *
 * `lambda` is carried forward UNCHANGED (1.925088 / 5.492399) — I1 does not re-derive it. The spike's
 * S2h/S3c bisection that produced these values targeted the OLD `N` (this stage's `L` is unchanged);
 * I2 ("physical spectrum normalization") owns re-deriving both lambda and the fold-fraction band for
 * the new grid, per the spec's Locked decision ("a third hand-fitted amplitude/λ pair is the exact
 * defect the spike spent five stages on" — re-fitting lambda here, against this stage's own gates,
 * would be exactly that).
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
        L: 30,
        windSpeed: 12,
        windDir: 0.6,
        amplitude: 0.00003,
        lambda: 5.492399,
        kLo: 0.75,
        kHi: 8.482300164692441,
    },
];

/** Standard DFT/FFT mode index: `k = (i <= N/2 ? i : i-N) * dk`. Every k-labelled kernel
 * (`generateH0`'s band mask below, `updateH`/`chop`/`spectralGradient` in `cpu-reference.ts`, and the
 * matching GPU kernels in `ocean.ts`) computes a physical wavenumber this way, and the FFT transforms
 * with the matching unshifted phase `2π·x·k/N` — carried forward verbatim from the spike (S3c),
 * unaffected by the FFT swap: re-indexing changes which `(kx,kz)` value each grid index labels, not
 * whether a shared wavenumber draws identically at both N, so `kIndex` needs no change here. */
export function kIndex(i: number, N: number): number {
    return i <= N / 2 ? i : i - N;
}

/** true iff `n` is a power of two (`n >= 1`). The butterfly FFT's own precondition on `N`. */
export function isPowerOfTwo(n: number): boolean {
    return n >= 1 && (n & (n - 1)) === 0;
}

/** Greatest common divisor — used to assert coprime domain sizes. */
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
 * Assert all cascade world-space patch lengths (`L`) are pairwise coprime (as integers) — a
 * SUFFICIENT (not necessary) condition for cross-cascade tile non-alignment, which the Locked
 * decision names as living on `L`, never `N`. General-purpose utility for a future stage that
 * chooses to decorrelate `L` further; **not asserted true for the shipped `CASCADE_CONFIGS`** — see
 * that constant's own docblock for why (the shipped pair shares a factor of 10, a 240 m repeat
 * period the foam pass already tiles around; `tilePeriod` below reports it).
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

/** Least common multiple — the real cross-cascade repeat period in world-space meters when `L` is
 *  not coprime (`lcm(a,b) = a*b/gcd(a,b)`), reported by `tilePeriod` below. */
function lcm(a: number, b: number): number {
    return (a * b) / gcd(a, b);
}

/** The world-space distance at which every shipped cascade's tile boundary re-aligns — `lcm` of
 *  every pairwise `L`, folded across more than two cascades. `Infinity` for a single cascade (no
 *  cross-cascade alignment to speak of). Informational: nothing in this package gates on this value,
 *  it is the honest number `assertCoprimeL`'s absence leaves on the table for the shipped configs. */
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
 * what the N-invariance arm's premise depends on (ported verbatim from the spike's S2e; unaffected by
 * the FFT swap since it operates purely in the frequency domain `generateH0` already occupies).
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
 * JONSWAP-modified Phillips spectrum P(k). Exported (unlike the spike's private `philips`) so the
 * mode-labeling oracle's parity witness can build an energy-weighted lag-1 autocorrelation prediction
 * directly from the declared spectral shape (see `tests/mode-labeling.oracle.ts`).
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
 * H₀(k) = (1/√2)·(ξᵣ + i·ξᵢ)·√P(k) — carried forward from the spike without the `Δk` (bin-width)
 * factor a fully normalized Tessendorf spectrum carries; `lambda`/`amplitude` absorb the missing
 * normalization (I2's scope — see the spec's Residue).
 *
 * `cfg.kLo`/`cfg.kHi` (rad/m, N-independent) zero every mode outside the cascade's declared band.
 * The conjugate H₀*(-k) needed for time evolution is derived in the GPU update kernel by looking up
 * H₀ at index (-k) and conjugating — so we only store H₀.
 */
export function generateH0(cfg: CascadeConfig): Float32Array {
    const N = cfg.N;
    const dk = (2 * Math.PI) / cfg.L;
    const h0 = new Float32Array(N * N * 2);

    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const kx = kIndex(x, N) * dk;
            const kz = kIndex(y, N) * dk;
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
 * The flop cost a direct O(N²)-per-dimension DFT (what the water-surface spike actually computed)
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
