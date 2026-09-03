// FFT ocean compute substrate — two displacement cascades over world-space patches. The separate
// high-wavenumber cascade is in `slope.ts` and contributes slope textures only.
// Compute passes per displacement cascade: update H(k,t); chop spectrum (i·k̂·H̃ for x and z); three inverse 2D
// FFTs (height, Dx, Dz) plus the spectral-gradient chain's own three (gxx, gxz, gzz — six total);
// post-process (Jacobian from the resulting displacement field, texture + probe write).
//
// The compute path uses a shared-memory butterfly FFT instead of a direct O(N²)-per-dimension DFT.
// shared-memory butterfly FFT in `./gpu-fft.ts`, which forces `N` to power-of-two on every cascade
// (`spectrum.ts`'s `CASCADE_CONFIGS`). Update, chop, gradient, and post-process retain the same
// spectral formulas around the inverse-transform passes.
//
// The canonical displacement operator (i·k̂·H̃(k,t), inverse-transformed and scaled by `lambda` after
// the transform — never a finite difference on `h`) and the spectral Jacobian (analytic derivative of
// the spectral representation, with no finite-difference spacing factor. `spectrum.ts` defines the
// represented bands, while the package tests cover cross-resolution behavior.

import {
    Compute,
    PartPlugin,
    type Plugin,
    RenderPlugin,
    SearPlugin,
    type State,
    type System,
} from "@dylanebert/shallot";
import { BeginFrameSystem, Frame, Render } from "@dylanebert/shallot/render/core";
import { PrepassSystem } from "@dylanebert/shallot/sear/core";
import { idiv } from "@dylanebert/shallot/utils/core";
import tgpu, { type TgpuBindGroup, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { getFftKernels } from "./gpu-fft";
import { buildSlopes, slopeCompute, teardownSlopes } from "./slope";
import { CASCADE_CONFIGS, type CascadeConfig, generateH0 } from "./spectrum";
import { buildFoamStrength, registerOceanSurface, teardownFoamStrength } from "./surface";

// Re-export the cascade configuration alongside the plugin API.
export { CASCADE_CONFIGS, type CascadeConfig };

// ── schemas ──────────────────────────────────────────────────────────────────

/** Complex number as vec2f: .x = real, .y = imag. */
export const Complex = d.vec2f;

/** Per-cascade compute params (uniform). */
export const CascadeParams = d
    .struct({
        N: d.u32,
        L: d.f32,
        lambda: d.f32,
        time: d.f32,
        gravity: d.f32,
        pad0: d.f32,
        pad1: d.f32,
        pad2: d.f32,
    })
    .$name("CascadeParams");

/** Readback data per cascade: probe statistics. */
export const ProbeData = d
    .struct({
        energy: d.f32,
        minDet: d.f32,
        maxDet: d.f32,
        negDetCount: d.u32,
        finiteCount: d.u32,
        totalCount: d.u32,
        sampleHash: d.u32,
        _pad: d.u32, // explicit padding to 32 bytes
    })
    .$name("ProbeData");

// ── bind group layouts ──────────────────────────────────────────────────────

export const updateLayout = tgpu.bindGroupLayout({
    h0: { storage: d.arrayOf(Complex), access: "readonly" },
    h: { storage: d.arrayOf(Complex), access: "mutable" },
    params: { uniform: CascadeParams },
});

/** i·k̂·H̃(k,t) — the spectral displacement source: one bind group emits BOTH the x and z chop
 * spectra from the same H(k,t) buffer in a single pass. */
const chopLayout = tgpu.bindGroupLayout({
    h: { storage: d.arrayOf(Complex), access: "readonly" },
    dxOut: { storage: d.arrayOf(Complex), access: "mutable" },
    dzOut: { storage: d.arrayOf(Complex), access: "mutable" },
    params: { uniform: CascadeParams },
});

/**
 * Spectral gradient source: ∂Dx/∂x, ∂Dx/∂z (= ∂Dz/∂x), ∂Dz/∂z as their OWN inverse transforms,
 * derived by the derivative theorem (spatial d/dx ⇔ multiply the spectrum by i·k) on the
 * ALREADY-COMPUTED chop spectra: gxx = i·kx·dxSpec, gxz = i·kz·dxSpec (= i·kx·dzSpec by the symmetry
 * `dxSpec = i·k̂ₓ·H`, `dzSpec = i·k̂_z·H`), gzz = i·kz·dzSpec. Replaces a central-difference Jacobian,
 * which would understate the true fold fraction (`sinc(k·dx)` attenuation, worst at the band edge).
 */
const gradientLayout = tgpu.bindGroupLayout({
    dxSpec: { storage: d.arrayOf(Complex), access: "readonly" },
    dzSpec: { storage: d.arrayOf(Complex), access: "readonly" },
    gxxOut: { storage: d.arrayOf(Complex), access: "mutable" },
    gxzOut: { storage: d.arrayOf(Complex), access: "mutable" },
    gzzOut: { storage: d.arrayOf(Complex), access: "mutable" },
    params: { uniform: CascadeParams },
});

const postLayout = tgpu.bindGroupLayout({
    height: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    dxField: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    dzField: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    gxxField: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    gxzField: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    gzzField: { storage: d.arrayOf(Complex), access: "readonly", visibility: ["compute"] },
    output: {
        storageTexture: d.textureStorage2d("rgba16float", "write-only"),
        visibility: ["compute"],
    },
    probe: { storage: d.arrayOf(ProbeData), access: "mutable", visibility: ["compute"] },
    params: { uniform: CascadeParams, visibility: ["compute"] },
});

// ── compute kernels ──────────────────────────────────────────────────────────

const PI = Math.PI;

/**
 * Update H(k,t) = H₀(k)·exp(iωt) + conj(H₀(-k))·exp(-iωt).
 * One thread per k-point. Row index derived via `idiv` (`gpu.md`'s promoted TGSL-integer-division
 * rule — plain `idx / N` compiles to a REAL `f32` division on integer operands, `idiv` from
 * `utils/core` is the sanctioned fix; `%` is native integer remainder and unaffected).
 */
export const updateKernel = tgpu
    .computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = updateLayout.$.params.N;
        if (idx >= N * N) return;

        const dk = (d.f32(2) * PI) / updateLayout.$.params.L;
        const x = idx % N;
        const y = idiv(idx, N);
        // standard DFT mode index: k = (i <= N/2 ? i : i-N) * dk, matching the FFT's unshifted
        // phase 2π·i·k/N (see spectrum.ts's `kIndex` docblock — the CPU/TS mirror of this select).
        // Compares in f32 (never u32/u32) to sidestep the u32-division hazard noted above.
        const halfN = d.f32(N) / d.f32(2);
        const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= halfN) * dk;
        const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= halfN) * dk;
        const kMag = std.sqrt(kx * kx + kz * kz);
        const omega = std.sqrt(updateLayout.$.params.gravity * std.max(kMag, d.f32(1e-6)));
        const t = updateLayout.$.params.time;

        const h0 = updateLayout.$.h0[idx];
        const expWt = d.vec2f(std.cos(omega * t), std.sin(omega * t));
        const expMwt = d.vec2f(std.cos(-omega * t), std.sin(-omega * t));

        // conj(H0(-k)): look up H0 at index (-k) = ((N - x) % N, (N - y) % N)
        const negX = (N - x) % N;
        const negY = (N - y) % N;
        const h0Neg = updateLayout.$.h0[negY * N + negX];
        const conjH0Neg = d.vec2f(h0Neg.x, -h0Neg.y);

        // H(k,t) = H0(k) * exp(iωt) + conj(H0(-k)) * exp(-iωt)
        const term1 = d.vec2f(h0.x * expWt.x - h0.y * expWt.y, h0.x * expWt.y + h0.y * expWt.x);
        const term2 = d.vec2f(
            conjH0Neg.x * expMwt.x - conjH0Neg.y * expMwt.y,
            conjH0Neg.x * expMwt.y + conjH0Neg.y * expMwt.x,
        );
        updateLayout.$.h[idx] = d.vec2f(term1.x + term2.x, term1.y + term2.y);
    })
    .$name("ocean-update");

/**
 * Chop spectrum: D(k) = i·k̂·H̃(k,t), emitted for x and z in one pass.
 * i·H = i·(hx + i·hy) = -hy + i·hx, so i·k̂·H = (-k̂·hy, k̂·hx). `lambda` is applied AFTER the
 * inverse transform (postKernel), not here.
 */
const chopKernel = tgpu
    .computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = chopLayout.$.params.N;
        if (idx >= N * N) return;

        const dk = (d.f32(2) * PI) / chopLayout.$.params.L;
        const x = idx % N;
        const y = idiv(idx, N);
        const halfN = d.f32(N) / d.f32(2);
        const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= halfN) * dk;
        const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= halfN) * dk;
        const kMag = std.sqrt(kx * kx + kz * kz);
        const invK = std.select(d.f32(0), d.f32(1) / kMag, kMag > d.f32(1e-6));
        const kHatX = kx * invK;
        const kHatZ = kz * invK;

        const h = chopLayout.$.h[idx];
        chopLayout.$.dxOut[idx] = d.vec2f(-kHatX * h.y, kHatX * h.x);
        chopLayout.$.dzOut[idx] = d.vec2f(-kHatZ * h.y, kHatZ * h.x);
    })
    .$name("ocean-chop");

/**
 * Spectral gradient kernel: emits the three second-derivative spectra (gxx, gxz, gzz) by
 * multiplying the already-computed chop spectra by i·k (see `gradientLayout`'s docblock). Each
 * output is itself inverse-transformed by the SAME FFT kernels below, exactly like dxSpec/dzSpec.
 */
const gradientKernel = tgpu
    .computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = gradientLayout.$.params.N;
        if (idx >= N * N) return;

        const dk = (d.f32(2) * PI) / gradientLayout.$.params.L;
        const x = idx % N;
        const y = idiv(idx, N);
        const halfN = d.f32(N) / d.f32(2);
        const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= halfN) * dk;
        const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= halfN) * dk;

        const dxs = gradientLayout.$.dxSpec[idx];
        const dzs = gradientLayout.$.dzSpec[idx];

        // multiply a complex value by i·k: (a + b·i)·(i·k) = -k·b + i·k·a
        gradientLayout.$.gxxOut[idx] = d.vec2f(-kx * dxs.y, kx * dxs.x);
        gradientLayout.$.gxzOut[idx] = d.vec2f(-kz * dxs.y, kz * dxs.x);
        gradientLayout.$.gzzOut[idx] = d.vec2f(-kz * dzs.y, kz * dzs.x);
    })
    .$name("ocean-gradient");

/**
 * Post-process: extract height (real part of the height IFFT); read the already-transformed Dx/Dz
 * displacement fields (real parts of their own IFFTs) and scale by `lambda`; compute the Jacobian
 * from the SPECTRAL gradient fields (`gxxField`/`gxzField`/`gzzField` — each already the inverse
 * transform of i·k times the corresponding chop spectrum) rather than central differences on the
 * displacement field. Analytic derivative of the spectral representation, dimensionless by
 * construction (no `dx` or `1/dx` anywhere in this kernel). `Jxz` needs no averaging — `gxzField`
 * already carries the (analytically equal) `∂Dx/∂z` and `∂Dz/∂x` cross term directly. Write to
 * storage texture + probe.
 *
 * Texture channels: (Dx, h, Dz, detJ)
 */
const postKernel = tgpu
    .computeFn({
        workgroupSize: [8, 8],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const N = postLayout.$.params.N;
        const x = input.gid.x;
        const y = input.gid.y;
        if (x >= N || y >= N) return;

        const idx = y * N + x;
        const h = postLayout.$.height[idx].x; // real part = height
        const lambda = postLayout.$.params.lambda;

        // Displacement field (already the true spectral chop; lambda scales the transformed
        // real-space field, per Tessendorf, not the spectrum).
        const dxRaw = postLayout.$.dxField[idx].x;
        const dzRaw = postLayout.$.dzField[idx].x;
        const Dx = lambda * dxRaw;
        const Dz = lambda * dzRaw;

        // Spectral second-derivative fields — already the exact analytic gradient of the
        // (already spectrally chopped) displacement field, no discretization of `d/dx` needed.
        const dDxdx = lambda * postLayout.$.gxxField[idx].x;
        const dDxdz = lambda * postLayout.$.gxzField[idx].x; // == dDzdx by construction
        const dDzdz = lambda * postLayout.$.gzzField[idx].x;

        // Jacobian of the (x + Dx, z + Dz) map.
        const Jxx = d.f32(1) + dDxdx;
        const Jzz = d.f32(1) + dDzdz;
        const Jxz = dDxdz;
        const detJ = Jxx * Jzz - Jxz * Jxz;

        // Write to storage texture
        std.textureStore(
            postLayout.$.output,
            d.vec2i(d.i32(x), d.i32(y)),
            d.vec4f(Dx, h, Dz, detJ),
        );

        // Accumulate probe data. Written per-pixel; the CPU reads and reduces via mirror/readback.
        const finite = std.select(d.u32(0), d.u32(1), std.abs(detJ) < d.f32(1e30));
        const neg = std.select(d.u32(0), d.u32(1), detJ < d.f32(0));
        postLayout.$.probe[idx] = ProbeData({
            energy: h * h,
            minDet: detJ,
            maxDet: detJ,
            negDetCount: neg,
            finiteCount: finite,
            totalCount: d.u32(1),
            sampleHash: d.u32(0),
            _pad: d.u32(0),
        });
    })
    .$name("ocean-post");

// ── peak-FMA measurement (the achieved-vs-theoretical ratio's denominator) ─────

/**
 * measured achievable FMA rate: 1M threads × 1024 iterations × 4 dependent FMA chains × 2 FLOPs.
 * Thread count is the load-bearing half: too few threads cannot occupy the GPU (the chains are
 * dependent, so without enough resident threads the SMs idle on latency). 1M threads × fewer
 * iterations each holds the total work while the GPU runs full.
 */
const PEAK_THREADS = 1 << 20;
const PEAK_ITERS = 1024;
const PEAK_CHAINS = 4;
/** each chain is a vec4 FMA — 4 lanes × 2 FLOPs per chain per iteration. */
export const PEAK_FMA_FLOPS = PEAK_THREADS * PEAK_ITERS * PEAK_CHAINS * 4 * 2;

/** the harness flips this around its benchmark window so the peak pass costs nothing at rest */
let _peakFmaEnabled = false;
export function setPeakFmaEnabled(on: boolean): void {
    _peakFmaEnabled = on;
}

const peakLayout = tgpu.bindGroupLayout({
    seed: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    sink: { storage: d.arrayOf(d.vec4f), access: "mutable" },
});

/** pure-FMA compute kernel: 4 interleaved dependent `x = x·s + b` chains per thread, results sunk */
const peakKernel = tgpu
    .computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const tid = input.gid.x;
        if (tid >= d.u32(PEAK_THREADS)) return;
        const s = peakLayout.$.seed[tid];
        let a = std.copy(s); // array access is a reference — copy into a reassignable value
        let b = std.add(s, d.vec4f(0.1));
        let c = std.sub(s, d.vec4f(0.1));
        let e = std.mul(s, d.vec4f(0.9));
        for (let i = d.u32(0); i < d.u32(PEAK_ITERS); i = i + d.u32(1)) {
            a = std.add(std.mul(a, s), b);
            b = std.add(std.mul(b, s), c);
            c = std.add(std.mul(c, s), e);
            e = std.add(std.mul(e, s), a);
        }
        peakLayout.$.sink[tid % d.u32(256)] = std.add(std.add(a, b), std.add(c, e));
    })
    .$name("ocean-peakfma-kernel");

// ── per-cascade GPU state ─────────────────────────────────────────────────────────────────

interface CascadeState {
    config: CascadeConfig;
    N: number;
    h0Buffer: GPUBuffer;
    hBuffer: GPUBuffer;
    tempBuffer: GPUBuffer;
    heightBuffer: GPUBuffer;
    dxSpecBuffer: GPUBuffer;
    dzSpecBuffer: GPUBuffer;
    dxTempBuffer: GPUBuffer;
    dzTempBuffer: GPUBuffer;
    dxHeightBuffer: GPUBuffer;
    dzHeightBuffer: GPUBuffer;
    gxxSpecBuffer: GPUBuffer;
    gxzSpecBuffer: GPUBuffer;
    gzzSpecBuffer: GPUBuffer;
    gxxTempBuffer: GPUBuffer;
    gxzTempBuffer: GPUBuffer;
    gzzTempBuffer: GPUBuffer;
    gxxHeightBuffer: GPUBuffer;
    gxzHeightBuffer: GPUBuffer;
    gzzHeightBuffer: GPUBuffer;
    texture: GPUTexture;
    probeBuffer: GPUBuffer;
    paramsBuffer: GPUBuffer;
    updatePipeline: TgpuComputePipeline;
    chopPipeline: TgpuComputePipeline;
    gradientPipeline: TgpuComputePipeline;
    fftRowPipeline: TgpuComputePipeline;
    fftColPipeline: TgpuComputePipeline;
    postPipeline: TgpuComputePipeline;
    updateGroup: TgpuBindGroup;
    chopGroup: TgpuBindGroup;
    gradientGroup: TgpuBindGroup;
    heightRowGroup: TgpuBindGroup;
    heightColGroup: TgpuBindGroup;
    dxRowGroup: TgpuBindGroup;
    dxColGroup: TgpuBindGroup;
    dzRowGroup: TgpuBindGroup;
    dzColGroup: TgpuBindGroup;
    gxxRowGroup: TgpuBindGroup;
    gxxColGroup: TgpuBindGroup;
    gxzRowGroup: TgpuBindGroup;
    gxzColGroup: TgpuBindGroup;
    gzzRowGroup: TgpuBindGroup;
    gzzColGroup: TgpuBindGroup;
    postGroup: TgpuBindGroup;
}

const cascades: CascadeState[] = [];

/** Total probe entries across all cascades (for readback buffer sizing). */
export const PROBE_TOTAL = CASCADE_CONFIGS.reduce((s, c) => s + c.N * c.N, 0);

/** Per-cascade probe entry count. */
export const PROBE_COUNTS = CASCADE_CONFIGS.map((c) => c.N * c.N);

/** Get the displacement texture for a cascade (by index). */
export function getDisplacementTexture(cascade: number): GPUTexture | null {
    return cascades[cascade]?.texture ?? null;
}

/** Get the probe buffer for a specific cascade (for readback). */
export function getProbeBufferForCascade(cascade: number): GPUBuffer | null {
    return cascades[cascade]?.probeBuffer ?? null;
}

/** Get cascade configs. */
export function getCascadeConfigs(): CascadeConfig[] {
    return CASCADE_CONFIGS;
}

// ── shared cascade construction (build() and the N-invariance probe both use this) ────────

/** Allocate all GPU state for one cascade config (buffers, pipelines, bind groups). Does NOT
 * publish textures/samplers into Compute — callers that want the cascade rendered do that. */
function createCascadeState(config: CascadeConfig): CascadeState {
    const { device, root } = Compute;
    const N = config.N;
    const complexBytes = N * N * 8; // vec2f = 8 bytes
    const probeBytes = d.sizeOf(d.arrayOf(ProbeData, N * N));

    const h0Data = generateH0(config, 0);
    const h0Buffer = device.createBuffer({
        label: `ocean-h0-${N}`,
        size: complexBytes,
        // COPY_SRC: the stage-by-stage CPU/GPU agreement probe (readStageBuffers) reads this buffer
        // back — without it, copyBufferToBuffer's source-usage validation fails asynchronously (no
        // throw) and the destination stays zero-initialized.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(h0Buffer, 0, h0Data);

    // COPY_SRC: same reason as h0Buffer above — every complex buffer readStageBuffers reads back
    // needs it; the *Temp buffers are never read back and don't need it, but sharing one constructor
    // keeps every complex buffer copy-readable rather than hand-picking which ones a future probe
    // will want.
    const makeComplexBuffer = (label: string) =>
        device.createBuffer({
            label,
            size: complexBytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

    const hBuffer = makeComplexBuffer(`ocean-h-${N}`);
    const tempBuffer = makeComplexBuffer(`ocean-temp-${N}`);
    const heightBuffer = makeComplexBuffer(`ocean-height-${N}`);
    const dxSpecBuffer = makeComplexBuffer(`ocean-dxspec-${N}`);
    const dzSpecBuffer = makeComplexBuffer(`ocean-dzspec-${N}`);
    const dxTempBuffer = makeComplexBuffer(`ocean-dxtemp-${N}`);
    const dzTempBuffer = makeComplexBuffer(`ocean-dztemp-${N}`);
    const dxHeightBuffer = makeComplexBuffer(`ocean-dxheight-${N}`);
    const dzHeightBuffer = makeComplexBuffer(`ocean-dzheight-${N}`);
    // spectral gradient buffers: gxx/gxz/gzz spectra + their IFFT chains (temp, then the real-space
    // height-analog buffer postKernel reads for the Jacobian).
    const gxxSpecBuffer = makeComplexBuffer(`ocean-gxxspec-${N}`);
    const gxzSpecBuffer = makeComplexBuffer(`ocean-gxzspec-${N}`);
    const gzzSpecBuffer = makeComplexBuffer(`ocean-gzzspec-${N}`);
    const gxxTempBuffer = makeComplexBuffer(`ocean-gxxtemp-${N}`);
    const gxzTempBuffer = makeComplexBuffer(`ocean-gxztemp-${N}`);
    const gzzTempBuffer = makeComplexBuffer(`ocean-gzztemp-${N}`);
    const gxxHeightBuffer = makeComplexBuffer(`ocean-gxxheight-${N}`);
    const gxzHeightBuffer = makeComplexBuffer(`ocean-gxzheight-${N}`);
    const gzzHeightBuffer = makeComplexBuffer(`ocean-gzzheight-${N}`);

    const texture = device.createTexture({
        label: `ocean-displace-${N}`,
        size: { width: N, height: N },
        format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    const probeBuffer = device.createBuffer({
        label: `ocean-probe-${N}`,
        size: probeBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramsBuffer = device.createBuffer({
        label: `ocean-params-${N}`,
        size: d.sizeOf(CascadeParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const h0Typed = root.createBuffer(d.arrayOf(Complex, N * N), h0Buffer).$usage("storage");
    const hTyped = root.createBuffer(d.arrayOf(Complex, N * N), hBuffer).$usage("storage");
    const tempTyped = root.createBuffer(d.arrayOf(Complex, N * N), tempBuffer).$usage("storage");
    const heightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), heightBuffer)
        .$usage("storage");
    const dxSpecTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dxSpecBuffer)
        .$usage("storage");
    const dzSpecTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dzSpecBuffer)
        .$usage("storage");
    const dxTempTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dxTempBuffer)
        .$usage("storage");
    const dzTempTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dzTempBuffer)
        .$usage("storage");
    const dxHeightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dxHeightBuffer)
        .$usage("storage");
    const dzHeightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), dzHeightBuffer)
        .$usage("storage");
    const gxxSpecTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxxSpecBuffer)
        .$usage("storage");
    const gxzSpecTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxzSpecBuffer)
        .$usage("storage");
    const gzzSpecTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gzzSpecBuffer)
        .$usage("storage");
    const gxxTempTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxxTempBuffer)
        .$usage("storage");
    const gxzTempTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxzTempBuffer)
        .$usage("storage");
    const gzzTempTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gzzTempBuffer)
        .$usage("storage");
    const gxxHeightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxxHeightBuffer)
        .$usage("storage");
    const gxzHeightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gxzHeightBuffer)
        .$usage("storage");
    const gzzHeightTyped = root
        .createBuffer(d.arrayOf(Complex, N * N), gzzHeightBuffer)
        .$usage("storage");
    const probeTyped = root
        .createBuffer(d.arrayOf(ProbeData, N * N), probeBuffer)
        .$usage("storage");
    const paramsTyped = root.createBuffer(CascadeParams, paramsBuffer).$usage("uniform");

    const updatePipeline = root
        .createComputePipeline({ compute: updateKernel })
        .$name(`ocean-update-${N}`);
    const chopPipeline = root
        .createComputePipeline({ compute: chopKernel })
        .$name(`ocean-chop-${N}`);
    const gradientPipeline = root
        .createComputePipeline({ compute: gradientKernel })
        .$name(`ocean-gradient-${N}`);
    const fft = getFftKernels(N);
    const fftRowPipeline = root
        .createComputePipeline({ compute: fft.rowKernel })
        .$name(`ocean-fft-row-${N}`);
    const fftColPipeline = root
        .createComputePipeline({ compute: fft.colKernel })
        .$name(`ocean-fft-col-${N}`);
    const postPipeline = root
        .createComputePipeline({ compute: postKernel })
        .$name(`ocean-post-${N}`);

    const updateGroup = root.createBindGroup(updateLayout, {
        h0: h0Typed,
        h: hTyped,
        params: paramsTyped,
    });
    const chopGroup = root.createBindGroup(chopLayout, {
        h: hTyped,
        dxOut: dxSpecTyped,
        dzOut: dzSpecTyped,
        params: paramsTyped,
    });
    const gradientGroup = root.createBindGroup(gradientLayout, {
        dxSpec: dxSpecTyped,
        dzSpec: dzSpecTyped,
        gxxOut: gxxSpecTyped,
        gxzOut: gxzSpecTyped,
        gzzOut: gzzSpecTyped,
        params: paramsTyped,
    });
    const heightRowGroup = root.createBindGroup(fft.layout, { input: hTyped, output: tempTyped });
    const heightColGroup = root.createBindGroup(fft.layout, {
        input: tempTyped,
        output: heightTyped,
    });
    const dxRowGroup = root.createBindGroup(fft.layout, {
        input: dxSpecTyped,
        output: dxTempTyped,
    });
    const dxColGroup = root.createBindGroup(fft.layout, {
        input: dxTempTyped,
        output: dxHeightTyped,
    });
    const dzRowGroup = root.createBindGroup(fft.layout, {
        input: dzSpecTyped,
        output: dzTempTyped,
    });
    const dzColGroup = root.createBindGroup(fft.layout, {
        input: dzTempTyped,
        output: dzHeightTyped,
    });
    const gxxRowGroup = root.createBindGroup(fft.layout, {
        input: gxxSpecTyped,
        output: gxxTempTyped,
    });
    const gxxColGroup = root.createBindGroup(fft.layout, {
        input: gxxTempTyped,
        output: gxxHeightTyped,
    });
    const gxzRowGroup = root.createBindGroup(fft.layout, {
        input: gxzSpecTyped,
        output: gxzTempTyped,
    });
    const gxzColGroup = root.createBindGroup(fft.layout, {
        input: gxzTempTyped,
        output: gxzHeightTyped,
    });
    const gzzRowGroup = root.createBindGroup(fft.layout, {
        input: gzzSpecTyped,
        output: gzzTempTyped,
    });
    const gzzColGroup = root.createBindGroup(fft.layout, {
        input: gzzTempTyped,
        output: gzzHeightTyped,
    });
    const postGroup = root.createBindGroup(postLayout, {
        height: heightTyped,
        dxField: dxHeightTyped,
        dzField: dzHeightTyped,
        gxxField: gxxHeightTyped,
        gxzField: gxzHeightTyped,
        gzzField: gzzHeightTyped,
        output: texture.createView(),
        probe: probeTyped,
        params: paramsTyped,
    });

    return {
        config,
        N,
        h0Buffer,
        hBuffer,
        tempBuffer,
        heightBuffer,
        dxSpecBuffer,
        dzSpecBuffer,
        dxTempBuffer,
        dzTempBuffer,
        dxHeightBuffer,
        dzHeightBuffer,
        gxxSpecBuffer,
        gxzSpecBuffer,
        gzzSpecBuffer,
        gxxTempBuffer,
        gxzTempBuffer,
        gzzTempBuffer,
        gxxHeightBuffer,
        gxzHeightBuffer,
        gzzHeightBuffer,
        texture,
        probeBuffer,
        paramsBuffer,
        updatePipeline,
        chopPipeline,
        gradientPipeline,
        fftRowPipeline,
        fftColPipeline,
        postPipeline,
        updateGroup,
        chopGroup,
        gradientGroup,
        heightRowGroup,
        heightColGroup,
        dxRowGroup,
        dxColGroup,
        dzRowGroup,
        dzColGroup,
        gxxRowGroup,
        gxxColGroup,
        gxzRowGroup,
        gxzColGroup,
        gzzRowGroup,
        gzzColGroup,
        postGroup,
    };
}

function destroyCascadeState(cs: CascadeState): void {
    cs.h0Buffer.destroy();
    cs.hBuffer.destroy();
    cs.tempBuffer.destroy();
    cs.heightBuffer.destroy();
    cs.dxSpecBuffer.destroy();
    cs.dzSpecBuffer.destroy();
    cs.dxTempBuffer.destroy();
    cs.dzTempBuffer.destroy();
    cs.dxHeightBuffer.destroy();
    cs.dzHeightBuffer.destroy();
    cs.gxxSpecBuffer.destroy();
    cs.gxzSpecBuffer.destroy();
    cs.gzzSpecBuffer.destroy();
    cs.gxxTempBuffer.destroy();
    cs.gxzTempBuffer.destroy();
    cs.gzzTempBuffer.destroy();
    cs.gxxHeightBuffer.destroy();
    cs.gxzHeightBuffer.destroy();
    cs.gzzHeightBuffer.destroy();
    cs.texture.destroy();
    cs.probeBuffer.destroy();
    cs.paramsBuffer.destroy();
}

/** Write params for a cascade (must happen before any compute pass is encoded — writeBuffer isn't
 * allowed while a pass is open). */
function writeCascadeParams(cs: CascadeState, time: number): void {
    const paramsStaging = new ArrayBuffer(d.sizeOf(CascadeParams));
    const p32 = new Uint32Array(paramsStaging);
    const pf = new Float32Array(paramsStaging);
    p32[0] = cs.N;
    pf[1] = cs.config.L;
    pf[2] = cs.config.lambda;
    pf[3] = time;
    pf[4] = 9.81;
    pf[5] = 0;
    pf[6] = 0;
    pf[7] = 0;
    Compute.device.queue.writeBuffer(cs.paramsBuffer, 0, paramsStaging);
}

/** Encode the full per-cascade pass sequence (update, chop, gradient, then six FFT chains
 * [height, Dx, Dz, gxx, gxz, gzz] — each row-FFT then col-FFT — then post) into an already-open
 * command encoder. Shared by the per-frame render loop and the one-shot N-invariance probe below, so
 * both exercise the exact same production kernels. */
function encodeCascadePasses(encoder: GPUCommandEncoder, cs: CascadeState): void {
    const N = cs.N;
    const dispatch = Math.ceil((N * N) / 64);
    const postDispatch: [number, number] = [Math.ceil(N / 8), Math.ceil(N / 8)];

    const run1D = (label: string, pipeline: TgpuComputePipeline, group: TgpuBindGroup) => {
        const pass = encoder.beginComputePass({
            label,
            timestampWrites: Compute.span?.(label),
        });
        pipeline.with(group).with(pass).dispatchWorkgroups(dispatch);
        pass.end();
    };

    /** one FFT chain: N workgroups (one per row/column), each `N/2` threads — the dispatch count is
     * the row/column count, not `N*N/64` (the FFT kernels are one-workgroup-per-line, not
     * one-thread-per-element like the surrounding kernels). */
    const runFft = (label: string, pipeline: TgpuComputePipeline, group: TgpuBindGroup) => {
        const pass = encoder.beginComputePass({
            label,
            timestampWrites: Compute.span?.(label),
        });
        pipeline.with(group).with(pass).dispatchWorkgroups(N);
        pass.end();
    };

    run1D("ocean-update", cs.updatePipeline, cs.updateGroup);
    run1D("ocean-chop", cs.chopPipeline, cs.chopGroup);
    run1D("ocean-gradient", cs.gradientPipeline, cs.gradientGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.heightRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.heightColGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.dxRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.dxColGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.dzRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.dzColGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.gxxRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.gxxColGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.gxzRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.gxzColGroup);
    runFft("ocean-fft-row", cs.fftRowPipeline, cs.gzzRowGroup);
    runFft("ocean-fft-col", cs.fftColPipeline, cs.gzzColGroup);

    const passPost = encoder.beginComputePass({
        label: "ocean-post",
        timestampWrites: Compute.span?.("ocean-post"),
    });
    cs.postPipeline
        .with(cs.postGroup)
        .with(passPost)
        .dispatchWorkgroups(postDispatch[0], postDispatch[1]);
    passPost.end();
}

/**
 * N-invariance probe — runs the ACTUAL production cascade pipeline (createCascadeState +
 * encodeCascadePasses, the same functions the live render loop uses) at an arbitrary one-off
 * config, off the persistent `cascades[]` array, and returns the fold fraction (det<0 share over
 * the full N*N grid). Used by the N-invariance arm to compare fold fraction at two grid
 * resolutions over one held-fixed sea state (same L, physical density, band, and lambda; only N
 * differs).
 */
export async function measureFoldFraction(config: CascadeConfig, time = 0): Promise<number> {
    const { device } = Compute;
    const cs = createCascadeState(config);
    writeCascadeParams(cs, time);

    const encoder = device.createCommandEncoder({ label: "fold-fraction-probe" });
    encodeCascadePasses(encoder, cs);
    device.queue.submit([encoder.finish()]);

    const N = cs.N;
    const entrySize = 32; // ProbeData = 8 × 4 bytes
    const readSize = N * N * entrySize;
    const probeReadback = device.createBuffer({
        label: "ocean-fold-probe-read",
        size: readSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const copyEncoder = device.createCommandEncoder({ label: "fold-fraction-probe-copy" });
    copyEncoder.copyBufferToBuffer(cs.probeBuffer, 0, probeReadback, 0, readSize);
    device.queue.submit([copyEncoder.finish()]);

    await probeReadback.mapAsync(GPUMapMode.READ);
    const mapped = probeReadback.getMappedRange();
    const u32View = new Uint32Array(mapped);
    let negDetCount = 0;
    let totalCount = 0;
    for (let i = 0; i < N * N; i++) {
        const offset = i * 8;
        negDetCount += u32View[offset + 3];
        totalCount += u32View[offset + 5];
    }
    probeReadback.unmap();
    probeReadback.destroy();
    destroyCascadeState(cs);

    return negDetCount / Math.max(totalCount, 1);
}

/** Read one complex (vec2f) storage buffer back to a plain Float32Array (interleaved re, im). */
async function readComplexBuffer(
    device: GPUDevice,
    buffer: GPUBuffer,
    N: number,
): Promise<Float32Array> {
    const size = N * N * 8; // vec2f = 8 bytes
    const stageReadback = device.createBuffer({
        label: "ocean-stage-probe-read",
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label: "stage-probe-copy" });
    encoder.copyBufferToBuffer(buffer, 0, stageReadback, 0, size);
    device.queue.submit([encoder.finish()]);
    await stageReadback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stageReadback.getMappedRange().slice(0));
    stageReadback.unmap();
    stageReadback.destroy();
    return out;
}

/** Per-texel probe row: `postKernel` writes ONE ProbeData per texel (never reduced on-GPU), so
 * `minDet`/`maxDet` both carry that texel's own `detJ` and `energy` carries that texel's `h²` —
 * this is a full per-texel readout, not an aggregate. */
export interface ProbeRow {
    energy: Float32Array; // h² per texel
    detJ: Float32Array; // detJ per texel
}

async function readProbeBuffer(device: GPUDevice, buffer: GPUBuffer, N: number): Promise<ProbeRow> {
    const entrySize = 32; // ProbeData = 8 × 4 bytes
    const size = N * N * entrySize;
    const probeReadback = device.createBuffer({
        label: "ocean-stage-probe-read",
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder({ label: "stage-probe-copy" });
    encoder.copyBufferToBuffer(buffer, 0, probeReadback, 0, size);
    device.queue.submit([encoder.finish()]);
    await probeReadback.mapAsync(GPUMapMode.READ);
    const mapped = probeReadback.getMappedRange().slice(0);
    const f32 = new Float32Array(mapped);
    const energy = new Float32Array(N * N);
    const detJ = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) {
        energy[i] = f32[i * 8 + 0]; // ProbeData.energy
        detJ[i] = f32[i * 8 + 1]; // ProbeData.minDet (== maxDet == this texel's own detJ)
    }
    probeReadback.unmap();
    probeReadback.destroy();
    return { energy, detJ };
}

/** Stage-by-stage GPU readback for the CPU/GPU agreement arms (`cpu-reference.ts` + the ported
 * `smoke.ts`-style harness in the consuming project). Runs the ACTUAL production kernel sequence
 * (`createCascadeState` + `encodeCascadePasses`, same as `measureFoldFraction`) and reads back
 * every intermediate buffer along the pipeline: H0 (as written to the GPU, byte-identical to
 * `generateH0`'s own output — the sanity leg), H(k,t) after `updateKernel`, the chop spectra
 * after `chopKernel`, the three RAW (pre-lambda) inverse-transformed fields after the FFT chains,
 * and the per-texel probe row (detJ + h² — see `ProbeRow`) written by `postKernel`. Nothing here
 * re-derives any of these values; every buffer is the SAME one the per-frame render loop reads. */
export interface StageBuffers {
    N: number;
    h0: Float32Array;
    h: Float32Array;
    dxSpec: Float32Array;
    dzSpec: Float32Array;
    height: Float32Array;
    dxHeight: Float32Array;
    dzHeight: Float32Array;
    probe: ProbeRow;
}

export async function readStageBuffers(config: CascadeConfig, time = 0): Promise<StageBuffers> {
    const { device } = Compute;
    const cs = createCascadeState(config);
    writeCascadeParams(cs, time);

    const encoder = device.createCommandEncoder({ label: "stage-probe" });
    encodeCascadePasses(encoder, cs);
    device.queue.submit([encoder.finish()]);

    const N = cs.N;
    const [h0, h, dxSpec, dzSpec, height, dxHeight, dzHeight, probe] = await Promise.all([
        readComplexBuffer(device, cs.h0Buffer, N),
        readComplexBuffer(device, cs.hBuffer, N),
        readComplexBuffer(device, cs.dxSpecBuffer, N),
        readComplexBuffer(device, cs.dzSpecBuffer, N),
        readComplexBuffer(device, cs.heightBuffer, N),
        readComplexBuffer(device, cs.dxHeightBuffer, N),
        readComplexBuffer(device, cs.dzHeightBuffer, N),
        readProbeBuffer(device, cs.probeBuffer, N),
    ]);
    destroyCascadeState(cs);

    return { N, h0, h, dxSpec, dzSpec, height, dxHeight, dzHeight, probe };
}

// ── compute system ───────────────────────────────────────────────────────────

// peak-FMA GPU state (built alongside the cascades)
let peakPipeline: TgpuComputePipeline | null = null;
let peakGroup: TgpuBindGroup | null = null;
let sinkBuffer: GPUBuffer | null = null;

/** the ocean compute system — exported so a foam pass can pin its order after it */
export const oceanCompute: System = {
    name: "ocean-compute",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state: State) {
        const encoder = Render.encoder;
        if (!encoder || cascades.length === 0) return;
        const time = state.time.elapsed;

        // Write ALL params BEFORE beginning any compute passes (queue.writeBuffer is not allowed while a pass is open)
        for (const cs of cascades) {
            writeCascadeParams(cs, time);
        }

        for (const cs of cascades) {
            encodeCascadePasses(encoder, cs);
        }

        // peak-FMA measurement (a harness enables it around its benchmark window only)
        if (_peakFmaEnabled && peakPipeline && peakGroup) {
            const passPeak = encoder.beginComputePass({
                label: "ocean-peakfma",
                timestampWrites: Compute.span?.("ocean-peakfma"),
            });
            peakPipeline
                .with(peakGroup)
                .with(passPeak)
                .dispatchWorkgroups(PEAK_THREADS / 64);
            passPeak.end();
        }
    },
};

// ── build / teardown ─────────────────────────────────────────────────────────

function build(): void {
    const { device, root } = Compute;
    cascades.length = 0;

    for (const config of CASCADE_CONFIGS) {
        const cs = createCascadeState(config);
        cascades.push(cs);

        // Publish displacement textures for the render surface
        const texIdx = CASCADE_CONFIGS.indexOf(config);
        Compute.textures.set(`displace${texIdx}`, cs.texture);
    }

    // Publish sampler
    const sampler = device.createSampler({
        label: "ocean-sampler",
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat",
    });
    Compute.samplers.set("oceanSampler", sampler);

    // Peak-FMA measurement buffers + pipeline (dispatched only while the flag is on)
    const seed = new Float32Array(PEAK_THREADS * 4);
    for (let i = 0; i < seed.length; i++) seed[i] = Math.random() * 2 - 1;
    const seedBuffer = device.createBuffer({
        label: "ocean-peakfma-seed",
        size: seed.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(seedBuffer, 0, seed);
    sinkBuffer = device.createBuffer({
        label: "ocean-peakfma-sink",
        size: 256 * 16,
        usage: GPUBufferUsage.STORAGE,
    });
    peakPipeline = root
        .createComputePipeline({ compute: peakKernel })
        .$name("ocean-peakfma-pipeline");
    peakGroup = root.createBindGroup(peakLayout, {
        seed: root.createBuffer(d.arrayOf(d.vec4f, PEAK_THREADS), seedBuffer).$usage("storage"),
        sink: root.createBuffer(d.arrayOf(d.vec4f, 256), sinkBuffer).$usage("storage"),
    });
}

function teardown(): void {
    sinkBuffer?.destroy();
    sinkBuffer = null;
    peakPipeline = null;
    peakGroup = null;
    for (const cs of cascades) {
        destroyCascadeState(cs);
    }
    cascades.length = 0;
    for (let i = 0; i < CASCADE_CONFIGS.length; i++) {
        Compute.textures.delete(`displace${i}`);
    }
    Compute.samplers.delete("oceanSampler");
}

// ── plugin ───────────────────────────────────────────────────────────────────

export const OceanPlugin: Plugin = {
    name: "Ocean",
    dependencies: [PartPlugin, RenderPlugin, SearPlugin],
    systems: [oceanCompute, slopeCompute],
    initialize: registerOceanSurface,
    warm() {
        buildFoamStrength();
        build();
        buildSlopes();
    },
    dispose() {
        teardown();
        teardownSlopes();
        teardownFoamStrength();
    },
};

export default OceanPlugin;

// re-exported so a consumer never needs to import `render/core` directly just to reference the
// ordering anchors this plugin pins against
export { Frame };
