// CPU reference fixture for the fold-fraction probe and the CPU/GPU agreement arms. A second,
// independent implementation of the kernels `ocean.ts` actually runs on the GPU — `updateKernel`,
// `chopKernel`, `gradientKernel`, and the FFT-based 2D inverse transform (`fft1dInPlace`/`ifft2` from
// `./fft`) — transcribed by hand from `ocean.ts`'s own WGSL/TGSL source, sharing nothing with the GPU
// path except its inputs (`CASCADE_CONFIGS`, `generateH0`, imported from `./spectrum`, both plain TS
// already shared between CPU and GPU). This file exists to answer one question: does the GPU probe
// agree with the operator's own arithmetic; the inverse transform uses the same butterfly FFT as
// the GPU path.
//
// Every function below is a literal transcription — same formula, same operation order as the
// WGSL source it mirrors — so a divergence from the GPU reading localizes to whichever stage
// disagrees first, rather than being explained away as "a different but equivalent formula".

import { ifft2 } from "./fft";
import { type CascadeConfig, kIndex, SEA_STATE } from "./spectrum";

const G = 9.81;
const PI = Math.PI;

/** Complex array, interleaved (re, im) per element — the same layout `Complex` (`d.vec2f`) uses
 * on the GPU side, so a GPU readback and a CPU array compare index-for-index with no repacking. */
export type ComplexArray = Float32Array;

/**
 * H(k,t) = H0(k)·exp(iωt) + conj(H0(-k))·exp(-iωt) — verbatim `updateKernel`.
 */
export function updateH(h0: ComplexArray, N: number, L: number, t: number): ComplexArray {
    const dk = (2 * PI) / L;
    const h = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;
            const kx = kIndex(x, N) * dk;
            const kz = kIndex(y, N) * dk;
            const kMag = Math.sqrt(kx * kx + kz * kz);
            const omega = Math.sqrt(G * Math.max(kMag, 1e-6));

            const negX = (N - x) % N;
            const negY = (N - y) % N;
            const negIdx = negY * N + negX;

            const h0r = h0[idx * 2];
            const h0i = h0[idx * 2 + 1];
            const negR = h0[negIdx * 2];
            const negI = -h0[negIdx * 2 + 1]; // conj(H0(-k))

            const cw = Math.cos(omega * t);
            const sw = Math.sin(omega * t);
            const cmw = Math.cos(-omega * t);
            const smw = Math.sin(-omega * t);

            const term1r = h0r * cw - h0i * sw;
            const term1i = h0r * sw + h0i * cw;
            const term2r = negR * cmw - negI * smw;
            const term2i = negR * smw + negI * cmw;

            h[idx * 2] = term1r + term2r;
            h[idx * 2 + 1] = term1i + term2i;
        }
    }
    return h;
}

/**
 * D(k) = i·k̂·H̃(k,t), emitted for x and z — verbatim `chopKernel`. The shared sea-state λ is NOT
 * applied here (ocean.ts scales the transformed real-space field in `postKernel`).
 */
export function chop(
    h: ComplexArray,
    N: number,
    L: number,
): { dxSpec: ComplexArray; dzSpec: ComplexArray } {
    const dk = (2 * PI) / L;
    const dxSpec = new Float32Array(N * N * 2);
    const dzSpec = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;
            const kx = kIndex(x, N) * dk;
            const kz = kIndex(y, N) * dk;
            const kMag = Math.sqrt(kx * kx + kz * kz);
            const invK = kMag > 1e-6 ? 1 / kMag : 0;
            const kHatX = kx * invK;
            const kHatZ = kz * invK;

            const hr = h[idx * 2];
            const hi = h[idx * 2 + 1];
            dxSpec[idx * 2] = -kHatX * hi;
            dxSpec[idx * 2 + 1] = kHatX * hr;
            dzSpec[idx * 2] = -kHatZ * hi;
            dzSpec[idx * 2 + 1] = kHatZ * hr;
        }
    }
    return { dxSpec, dzSpec };
}

/**
 * Unnormalized 2D inverse transform — a row FFT then a column FFT (`ifft2`, `./fft`), matching the
 * physical-cell amplitude convention implemented by `generateH0`. The function keeps the stable
 * public signature and phase convention; `tests/fft.test.ts` compares it with a direct sum.
 */
export function idft2(input: ComplexArray, N: number): ComplexArray {
    return ifft2(input, N);
}

/**
 * Spectral gradient source, verbatim `gradientKernel`: ∂Dx/∂x, ∂Dx/∂z (= ∂Dz/∂x), ∂Dz/∂z
 * as their OWN spectra, derived from the ALREADY-COMPUTED chop spectra by the derivative theorem
 * (multiply by i·k). Each is inverse-transformed by the same `idft2` used for dxSpec/dzSpec.
 */
export function spectralGradient(
    dxSpec: ComplexArray,
    dzSpec: ComplexArray,
    N: number,
    L: number,
): { gxxSpec: ComplexArray; gxzSpec: ComplexArray; gzzSpec: ComplexArray } {
    const dk = (2 * PI) / L;
    const gxxSpec = new Float32Array(N * N * 2);
    const gxzSpec = new Float32Array(N * N * 2);
    const gzzSpec = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;
            const kx = kIndex(x, N) * dk;
            const kz = kIndex(y, N) * dk;

            const dxr = dxSpec[idx * 2];
            const dxi = dxSpec[idx * 2 + 1];
            const dzr = dzSpec[idx * 2];
            const dzi = dzSpec[idx * 2 + 1];

            // multiply a complex value by i·k: (a + b·i)·(i·k) = -k·b + i·k·a
            gxxSpec[idx * 2] = -kx * dxi;
            gxxSpec[idx * 2 + 1] = kx * dxr;
            gxzSpec[idx * 2] = -kz * dxi;
            gxzSpec[idx * 2 + 1] = kz * dxr;
            gzzSpec[idx * 2] = -kz * dzi;
            gzzSpec[idx * 2 + 1] = kz * dzr;
        }
    }
    return { gxxSpec, gxzSpec, gzzSpec };
}

export interface JacobianStats {
    N: number;
    /** real-part-only h/Dx/Dz fields, per texel (length N*N — real part of the raw IFFT output). */
    height: Float64Array;
    dxRaw: Float64Array;
    dzRaw: Float64Array;
    /** detJ per texel, length N*N — same layout as the GPU probe's `ProbeRow.detJ`. */
    detJ: Float64Array;
    dDxdx: Float64Array;
    foldCount: number;
    foldFraction: number;
    rmsH: number;
    rmsDx: number;
    rmsDz: number;
    rmsDDxdx: number;
}

/**
 * Spectral Jacobian on the (already spectrally chopped and inverse-transformed) displacement
 * field, verbatim `postKernel`'s Jacobian half (the texture-write and probe-write side effects are
 * irrelevant here; this returns the same per-texel quantities `postKernel` would have written).
 * `gxxHeight`/`gxzHeight`/`gzzHeight` are the real parts of the spectral gradient fields'
 * inverse transforms — no discretized `d/dx`, no `dx`/`1/dx` anywhere in this function.
 */
export function jacobianStats(
    heightField: ComplexArray,
    dxHeight: ComplexArray,
    dzHeight: ComplexArray,
    gxxHeight: ComplexArray,
    gxzHeight: ComplexArray,
    gzzHeight: ComplexArray,
    N: number,
    lambda: number,
): JacobianStats {
    const height = new Float64Array(N * N);
    const dxRaw = new Float64Array(N * N);
    const dzRaw = new Float64Array(N * N);
    const detJ = new Float64Array(N * N);
    const dDxdx = new Float64Array(N * N);
    let foldCount = 0;
    let sumH2 = 0;
    let sumDx2 = 0;
    let sumDz2 = 0;
    let sumDDxdx2 = 0;

    const re = (f: ComplexArray, i: number) => f[i * 2];

    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const idx = y * N + x;

            const h = re(heightField, idx);
            const dxr = lambda * re(dxHeight, idx);
            const dzr = lambda * re(dzHeight, idx);

            const a = lambda * re(gxxHeight, idx); // dDxdx
            const b = lambda * re(gxzHeight, idx); // dDxdz == dDzdx
            const dd = lambda * re(gzzHeight, idx); // dDzdz

            const Jxx = 1 + a;
            const Jzz = 1 + dd;
            const Jxz = b;
            const det = Jxx * Jzz - Jxz * Jxz;

            height[idx] = h;
            dxRaw[idx] = dxr;
            dzRaw[idx] = dzr;
            detJ[idx] = det;
            dDxdx[idx] = a;
            if (det < 0) foldCount++;
            sumH2 += h * h;
            sumDx2 += dxr * dxr;
            sumDz2 += dzr * dzr;
            sumDDxdx2 += a * a;
        }
    }

    const total = N * N;
    return {
        N,
        height,
        dxRaw,
        dzRaw,
        detJ,
        dDxdx,
        foldCount,
        foldFraction: foldCount / total,
        rmsH: Math.sqrt(sumH2 / total),
        rmsDx: Math.sqrt(sumDx2 / total),
        rmsDz: Math.sqrt(sumDz2 / total),
        rmsDDxdx: Math.sqrt(sumDDxdx2 / total),
    };
}

/** Run the whole pipeline (update → chop → gradient → IFFT → Jacobian) for one config at one
 * time — the CPU side of the stage-by-stage comparison the CPU/GPU agreement arms drive against
 * the GPU readback (`ocean.ts`'s `readStageBuffers`). */
export interface WorldJacobianStats {
    worldSize: number;
    foldCount: number;
    sampleCount: number;
    foldFraction: number;
    variance: number;
}

/** Fields needed to evaluate a composed world-grid Jacobian. */
export interface WorldJacobianFields {
    height: Float32Array | Float64Array;
    gxxHeight: ComplexArray;
    gxzHeight: ComplexArray;
    gzzHeight: ComplexArray;
}

/** Compose every cascade on one shared world-space lattice before taking the determinant. */
export function composeWorldJacobian(
    results: WorldJacobianFields[],
    configs: CascadeConfig[],
    worldSize = 128,
    lambda = SEA_STATE.lambda,
    worldExtent = Math.max(...configs.map((config) => config.L)),
): WorldJacobianStats {
    if (results.length !== configs.length)
        throw new Error("world-grid fields/configs length mismatch");
    let foldCount = 0;
    let variance = 0;
    for (let z = 0; z < worldSize; z++) {
        for (let x = 0; x < worldSize; x++) {
            const worldX = ((x + 0.5) / worldSize - 0.5) * worldExtent;
            const worldZ = ((z + 0.5) / worldSize - 0.5) * worldExtent;
            let h = 0;
            let dxdx = 0;
            let dxdz = 0;
            let dzdz = 0;
            for (let c = 0; c < configs.length; c++) {
                const cfg = configs[c];
                const result = results[c];
                const localX =
                    ((Math.floor((worldX / cfg.L + 0.5) * cfg.N) % cfg.N) + cfg.N) % cfg.N;
                const localZ =
                    ((Math.floor((worldZ / cfg.L + 0.5) * cfg.N) % cfg.N) + cfg.N) % cfg.N;
                const i = localZ * cfg.N + localX;
                h += result.height[i];
                dxdx += lambda * result.gxxHeight[i * 2];
                dxdz += lambda * result.gxzHeight[i * 2];
                dzdz += lambda * result.gzzHeight[i * 2];
            }
            const det = (1 + dxdx) * (1 + dzdz) - dxdz * dxdz;
            if (det < 0) foldCount++;
            variance += h * h;
        }
    }
    const sampleCount = worldSize * worldSize;
    return {
        worldSize,
        foldCount,
        sampleCount,
        foldFraction: foldCount / sampleCount,
        variance: variance / sampleCount,
    };
}

export interface CpuStageResult {
    h: ComplexArray;
    dxSpec: ComplexArray;
    dzSpec: ComplexArray;
    height: ComplexArray;
    dxHeight: ComplexArray;
    dzHeight: ComplexArray;
    gxxHeight: ComplexArray;
    gxzHeight: ComplexArray;
    gzzHeight: ComplexArray;
    jacobian: JacobianStats;
}

export function runCpuPipeline(h0: ComplexArray, config: CascadeConfig, time = 0): CpuStageResult {
    const { N, L } = config;
    const h = updateH(h0, N, L, time);
    const { dxSpec, dzSpec } = chop(h, N, L);
    const { gxxSpec, gxzSpec, gzzSpec } = spectralGradient(dxSpec, dzSpec, N, L);
    const height = idft2(h, N);
    const dxHeight = idft2(dxSpec, N);
    const dzHeight = idft2(dzSpec, N);
    const gxxHeight = idft2(gxxSpec, N);
    const gxzHeight = idft2(gxzSpec, N);
    const gzzHeight = idft2(gzzSpec, N);
    const jacobian = jacobianStats(
        height,
        dxHeight,
        dzHeight,
        gxxHeight,
        gxzHeight,
        gzzHeight,
        N,
        SEA_STATE.lambda,
    );
    return {
        h,
        dxSpec,
        dzSpec,
        height,
        dxHeight,
        dzHeight,
        gxxHeight,
        gxzHeight,
        gzzHeight,
        jacobian,
    };
}
