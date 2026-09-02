// High-wavenumber slope cascades. These modes are rendered as slope textures only: no height,
// horizontal displacement, or Jacobian output is allocated for this band. The source is a seeded
// height spectrum, converted to directional slope spectra (`i·kx·h̃`, `i·kz·h̃`) and inverse-transformed with the same
// radix-2 FFT used by the displacement cascades. Each output texture owns every mip level so a
// shading can select measured residual variance from hardware sample-coordinate gradients.

import { Compute, probeBuffer, type System } from "@dylanebert/shallot";
import { BeginFrameSystem, Render } from "@dylanebert/shallot/render/core";
import { PrepassSystem } from "@dylanebert/shallot/sear/core";
import { idiv } from "@dylanebert/shallot/utils/core";
import tgpu, { type TgpuBindGroup, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { ifft2 } from "./fft";
import { getFftKernels } from "./gpu-fft";
import {
    type CascadeConfig,
    directionalDensity,
    G,
    generateH0,
    kIndex,
    SEA_STATE,
    SLOPE_CASCADE_CONFIGS,
    type SpectrumMutation,
} from "./spectrum";

export { SLOPE_CASCADE_CONFIGS } from "./spectrum";

/** Number of complete mip levels in each slope texture, including level zero. */
export const SLOPE_MIP_LEVELS = Math.floor(Math.log2(SLOPE_CASCADE_CONFIGS[0].N)) + 1;

/** Checks the slope-only resource contract before GPU state is allocated. */
export function assertSlopeOnly(config: CascadeConfig): void {
    if (config.lambda !== 0) {
        throw new Error("shallot-ocean: slope cascade must not couple to displacement");
    }
}

/** CPU mirror of the GPU mip reduction: XY means plus second moment and sub-texel residual variance. */
export function reduceSlopeMip(source: Float32Array, size: number): Float32Array {
    if (!Number.isInteger(size) || size < 2 || source.length !== size * size * 4) {
        throw new RangeError("reduceSlopeMip: source must be an RGBA square with size >= 2");
    }
    const nextSize = size / 2;
    if (!Number.isInteger(nextSize)) throw new RangeError("reduceSlopeMip: size must be even");
    const out = new Float32Array(nextSize * nextSize * 4);
    for (let y = 0; y < nextSize; y++) {
        for (let x = 0; x < nextSize; x++) {
            const offsets = [
                (2 * y * size + 2 * x) * 4,
                (2 * y * size + 2 * x + 1) * 4,
                ((2 * y + 1) * size + 2 * x) * 4,
                ((2 * y + 1) * size + 2 * x + 1) * 4,
            ];
            const meanX = offsets.reduce((sum, i) => sum + source[i], 0) / 4;
            const meanZ = offsets.reduce((sum, i) => sum + source[i + 1], 0) / 4;
            const second = offsets.reduce((sum, i) => sum + source[i + 2], 0) / 4;
            const residual = Math.max(0, second - meanX * meanX - meanZ * meanZ);
            const index = (y * nextSize + x) * 4;
            out[index] = meanX;
            out[index + 1] = meanZ;
            out[index + 2] = second;
            out[index + 3] = residual;
        }
    }
    return out;
}

/** Per-level dimensions for a square slope texture. */
export function slopeMipSize(config: CascadeConfig, level: number): number {
    if (!Number.isInteger(level) || level < 0 || level > Math.floor(Math.log2(config.N))) {
        throw new RangeError(`slopeMipSize: invalid mip level ${level}`);
    }
    return Math.max(1, config.N >> level);
}

const PI = Math.PI;

/** Production slope-kernel dispersion frequency for one discrete mode. */
export const slopeOmega = tgpu.fn(
    [d.u32, d.u32, d.u32, d.f32],
    d.f32,
)((x, y, N, L) => {
    "use gpu";
    const dk = (d.f32(2) * PI) / L;
    const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= d.f32(N) / d.f32(2)) * dk;
    const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= d.f32(N) / d.f32(2)) * dk;
    return std.sqrt(d.f32(G) * std.max(std.sqrt(kx * kx + kz * kz), d.f32(1e-6)));
});

/** Production slope-kernel phase trig at one frequency and declared elapsed time. */
export const slopePhaseTrig = tgpu.fn(
    [d.f32, d.f32],
    d.vec4f,
)((omega, time) => {
    "use gpu";
    const phase = omega * time;
    return d.vec4f(std.cos(phase), std.sin(phase), std.cos(-phase), std.sin(-phase));
});

/** Deliberately omit the gradient-k factor from the slope spectra (`slopeSpectra`, the only
 *  function that reads `missingGradientK`). `SlopeMutation` also types `runSlopeCpuPipeline`'s own
 *  `mutation` parameter, which is the entry point the dropped-gradient red-witness oracle's
 *  production consumers (`slope.test.ts`, `slope.oracle.ts`) actually call — both pass
 *  `missingGradientK` to `runSlopeCpuPipeline`, which forwards it into `slopeSpectra` unchanged;
 *  neither test calls `slopeSpectra` directly. `integrateComposedSlopePsd`/`composedSlopePsd` never
 *  took this flag: the correct red-witness for the composed-quadrature side compares against the
 *  *unmutated* `composedSlopePsd` (I3c-r), so there is no caller left that needs a mutable gradient
 *  term in the quadrature. */
export interface SlopeMutation {
    missingGradientK?: boolean;
}

/**
 * Independently integrate the restricted slope moment over the declared band. The polar Jacobian
 * supplies `k`, the gradient operator supplies `k²`, and the angular quadrature evaluates the
 * published Cartesian density without reusing the discrete height-variance helper.
 */
function integrateComposedSlopePsd(
    config: CascadeConfig,
    mutation: SpectrumMutation,
    radialSteps: number,
    angularSteps: number,
): number {
    const logLo = Math.log(config.kLo);
    const dLog = Math.log(config.kHi / config.kLo) / radialSteps;
    const dTheta = (2 * Math.PI) / angularSteps;
    let total = 0;
    for (let radial = 0; radial < radialSteps; radial++) {
        const k = Math.exp(logLo + (radial + 0.5) * dLog);
        let angularMean = 0;
        for (let angular = 0; angular < angularSteps; angular++) {
            const theta = (angular + 0.5) * dTheta;
            angularMean += directionalDensity(
                k * Math.cos(theta),
                k * Math.sin(theta),
                SEA_STATE.windSpeed,
                SEA_STATE.windDir,
                SEA_STATE.omegaC,
                mutation,
            );
        }
        // k² from |∇h|², then k·dk = k²·dLog from the polar/log-radial measure: the radial weight
        // is k⁴·dLog·dTheta.
        const gradientMoment = k ** 4;
        total += angularMean * gradientMoment * dLog * dTheta;
    }
    return total;
}

/** Restricted slope moment from the production density and a fixed midpoint quadrature. */
export function composedSlopePsd(config: CascadeConfig, mutation: SpectrumMutation = {}): number {
    return integrateComposedSlopePsd(config, mutation, 512, 256);
}

/**
 * Relative one-sigma sampling error of the seeded Fourier energy estimator.
 * Each mode's weight is its published cell density times |k|²; the ratio is the effective
 * inverse sample count and is used by deterministic agreement gates instead of a magic tolerance.
 */
export function slopeMomentSamplingError(config: CascadeConfig): number {
    const dk = (2 * Math.PI) / config.L;
    let sum = 0;
    let squared = 0;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = kIndex(x, config.N) * dk;
            const kz = kIndex(y, config.N) * dk;
            const k = Math.hypot(kx, kz);
            if (k < config.kLo || k > config.kHi) continue;
            const weight =
                directionalDensity(
                    kx,
                    kz,
                    SEA_STATE.windSpeed,
                    SEA_STATE.windDir,
                    SEA_STATE.omegaC,
                ) *
                dk *
                dk *
                k *
                k;
            sum += weight;
            squared += weight * weight;
        }
    }
    return sum > 0 ? Math.sqrt(squared) / sum : Infinity;
}

/** 95% deterministic agreement bound derived from the mode-weight sampling error. */
export function slopeMomentAgreementTolerance(config: CascadeConfig): number {
    return 1.96 * slopeMomentSamplingError(config);
}

/** CPU reference for the directional slope spectra used by the GPU slope kernel. */
export function slopeSpectra(
    h: Float32Array,
    config: CascadeConfig,
    mutation: SlopeMutation = {},
): { x: Float32Array; z: Float32Array } {
    const { N, L } = config;
    const dk = (2 * Math.PI) / L;
    const x = new Float32Array(N * N * 2);
    const z = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let ix = 0; ix < N; ix++) {
            const i = y * N + ix;
            const kx = kIndex(ix, N) * dk;
            const kz = kIndex(y, N) * dk;
            const hr = h[i * 2];
            const hi = h[i * 2 + 1];
            const gradientX = mutation.missingGradientK ? 1 : kx;
            const gradientZ = mutation.missingGradientK ? 1 : kz;
            x[i * 2] = -gradientX * hi;
            x[i * 2 + 1] = gradientX * hr;
            z[i * 2] = -gradientZ * hi;
            z[i * 2 + 1] = gradientZ * hr;
        }
    }
    return { x, z };
}

/** CPU reference for one slope realization, including the two inverse FFTs. */
export function runSlopeCpuPipeline(
    h0: Float32Array,
    config: CascadeConfig = SLOPE_CASCADE_CONFIGS[0],
    time = 0,
    mutation: SlopeMutation = {},
): {
    h: Float32Array;
    x: Float32Array;
    z: Float32Array;
    xField: Float32Array;
    zField: Float32Array;
} {
    const N = config.N;
    const dk = (2 * Math.PI) / config.L;
    const h = new Float32Array(N * N * 2);
    for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
            const index = y * N + x;
            const kx = kIndex(x, N) * dk;
            const kz = kIndex(y, N) * dk;
            const omega = Math.sqrt(G * Math.max(Math.hypot(kx, kz), 1e-6));
            const neg = ((N - y) % N) * N + ((N - x) % N);
            const ar = h0[index * 2];
            const ai = h0[index * 2 + 1];
            const br = h0[neg * 2];
            const bi = -h0[neg * 2 + 1];
            const c = Math.cos(omega * time);
            const s = Math.sin(omega * time);
            const cm = Math.cos(-omega * time);
            const sm = Math.sin(-omega * time);
            h[index * 2] = ar * c - ai * s + br * cm - bi * sm;
            h[index * 2 + 1] = ar * s + ai * c + br * sm + bi * cm;
        }
    }
    const { x, z } = slopeSpectra(h, config, mutation);
    return { h, x, z, xField: ifft2(x, N), zField: ifft2(z, N) };
}

/** Mean-square slope of a realized CPU field, measured over the raster's real components. */
export function realizedSlopeMss(
    result: Pick<ReturnType<typeof runSlopeCpuPipeline>, "xField" | "zField">,
): number {
    const samples = result.xField.length / 2;
    let sum = 0;
    for (let i = 0; i < samples; i++) {
        sum += result.xField[i * 2] ** 2 + result.zField[i * 2] ** 2;
    }
    return sum / samples;
}

/** Rasterized realized slope moment; its sample count and FFT field are explicitly N-dependent. */
export function rasterSlopeMoment(h0: Float32Array, config: CascadeConfig, time = 0): number {
    return realizedSlopeMss(runSlopeCpuPipeline(h0, config, time));
}

const slopeParams = d
    .struct({ n: d.u32, L: d.f32, time: d.f32, pad: d.vec3u })
    .$name("SlopeParams");
const mipParams = d.struct({ width: d.u32, height: d.u32, pad: d.vec2u }).$name("SlopeMipParams");
const slopeLayout = tgpu.bindGroupLayout({
    h0: { storage: d.arrayOf(d.vec2f), access: "readonly" },
    x: { storage: d.arrayOf(d.vec2f), access: "mutable" },
    z: { storage: d.arrayOf(d.vec2f), access: "mutable" },
    params: { uniform: slopeParams },
});
const slopePostLayout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(d.vec2f), access: "readonly" },
    z: { storage: d.arrayOf(d.vec2f), access: "readonly" },
    output: { storageTexture: d.textureStorage2d("rgba16float", "write-only") },
    params: { uniform: slopeParams },
});
const mipLayout = tgpu.bindGroupLayout({
    source: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    output: {
        storageTexture: d.textureStorage2d("rgba16float", "write-only"),
        visibility: ["compute"],
    },
    params: { uniform: mipParams, visibility: ["compute"] },
});

const slopeKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = slopeLayout.$.params.n;
        if (idx >= N * N) return;
        const x = idx % N;
        const y = idiv(idx, N);
        const dk = (d.f32(2) * PI) / slopeLayout.$.params.L;
        const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= d.f32(N) / d.f32(2)) * dk;
        const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= d.f32(N) / d.f32(2)) * dk;
        const omega = slopeOmega(x, y, N, slopeLayout.$.params.L);
        const negX = (N - x) % N;
        const negY = (N - y) % N;
        const h0 = slopeLayout.$.h0[idx];
        const hn = slopeLayout.$.h0[negY * N + negX];
        const phaseTrig = slopePhaseTrig(omega, slopeLayout.$.params.time);
        const c = d.vec2f(phaseTrig.x, phaseTrig.y);
        const cm = d.vec2f(phaseTrig.z, phaseTrig.w);
        const nr = d.vec2f(hn.x, -hn.y);
        const h = d.vec2f(
            h0.x * c.x - h0.y * c.y + nr.x * cm.x - nr.y * cm.y,
            h0.x * c.y + h0.y * c.x + nr.x * cm.y + nr.y * cm.x,
        );
        slopeLayout.$.x[idx] = d.vec2f(-kx * h.y, kx * h.x);
        slopeLayout.$.z[idx] = d.vec2f(-kz * h.y, kz * h.x);
    })
    .$name("ocean-slope-spectrum");

const phaseProbeLayout = tgpu.bindGroupLayout({
    output: { storage: d.arrayOf(d.vec4f), access: "mutable" },
    params: { uniform: slopeParams },
});

const phaseProbeKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = phaseProbeLayout.$.params.n;
        if (idx >= N * N) return;
        const x = idx % N;
        const y = idiv(idx, N);
        const omega = slopeOmega(x, y, N, phaseProbeLayout.$.params.L);
        phaseProbeLayout.$.output[idx] = slopePhaseTrig(omega, phaseProbeLayout.$.params.time);
    })
    .$name("ocean-slope-phase-probe");

export interface PhaseTrigErrorReading {
    /** Number of production mode arguments evaluated exhaustively. */
    arguments: number;
    maxCosError: number;
    maxSinError: number;
    maxError: number;
    rmsError: number;
}

/** Exhaustively measure device phase-trig error over every mode argument the slope kernel evaluates. */
export async function measureSlopePhaseTrigError(
    config: CascadeConfig,
    time: number,
): Promise<PhaseTrigErrorReading> {
    if (!Number.isFinite(time) || time === 0)
        throw new RangeError("phase reading time must be finite and nonzero");
    const { device, root } = Compute;
    const count = config.N * config.N;
    const output = device.createBuffer({
        label: "ocean-slope-phase-out",
        size: count * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = device.createBuffer({
        label: "ocean-slope-phase-params",
        size: d.sizeOf(slopeParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const bytes = new ArrayBuffer(d.sizeOf(slopeParams));
    new Uint32Array(bytes)[0] = config.N;
    const f32 = new Float32Array(bytes);
    f32[1] = config.L;
    f32[2] = time;
    device.queue.writeBuffer(params, 0, bytes);
    const pipeline = root.createComputePipeline({ compute: phaseProbeKernel });
    const group = root.createBindGroup(phaseProbeLayout, {
        output: root.createBuffer(d.arrayOf(d.vec4f, count), output).$usage("storage"),
        params: root.createBuffer(slopeParams, params).$usage("uniform"),
    });
    const probe = await probeBuffer(device, output, {
        label: "ocean-slope-phase-probe",
        encode: (encoder) => {
            const pass = encoder.beginComputePass({ label: "ocean-slope-phase-probe" });
            pipeline
                .with(group)
                .with(pass)
                .dispatchWorkgroups(Math.ceil(count / 64));
            pass.end();
        },
    });
    const actual = new Float32Array(probe.bytes);
    let maxCosError = 0;
    let maxSinError = 0;
    let sumSquares = 0;
    for (let idx = 0; idx < count; idx++) {
        const x = idx % config.N;
        const y = Math.floor(idx / config.N);
        const dk = (2 * Math.PI) / config.L;
        const kx = (x <= config.N / 2 ? x : x - config.N) * dk;
        const kz = (y <= config.N / 2 ? y : y - config.N) * dk;
        const omega = Math.sqrt(G * Math.max(Math.hypot(kx, kz), 1e-6));
        const phase = omega * time;
        const expected = [Math.cos(phase), Math.sin(phase), Math.cos(-phase), Math.sin(-phase)];
        for (let lane = 0; lane < 4; lane++) {
            const error = Math.abs(actual[idx * 4 + lane] - expected[lane]);
            if (lane % 2 === 0) maxCosError = Math.max(maxCosError, error);
            else maxSinError = Math.max(maxSinError, error);
            sumSquares += error * error;
        }
    }
    output.destroy();
    params.destroy();
    return {
        arguments: count,
        maxCosError,
        maxSinError,
        maxError: Math.max(maxCosError, maxSinError),
        rmsError: Math.sqrt(sumSquares / (count * 4)),
    };
}

const slopePostKernel = tgpu
    .computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const x = input.gid.x;
        const y = input.gid.y;
        const N = slopePostLayout.$.params.n;
        if (x >= N || y >= N) return;
        const valueX = slopePostLayout.$.x[y * N + x];
        const valueZ = slopePostLayout.$.z[y * N + x];
        const energy = valueX.x * valueX.x + valueZ.x * valueZ.x;
        // Level zero has no sub-texel residual; each later mip computes the residual from the
        // four source texels' second moment and its new mean.
        std.textureStore(
            slopePostLayout.$.output,
            d.vec2i(d.i32(x), d.i32(y)),
            d.vec4f(valueX.x, valueZ.x, energy, 0),
        );
    })
    .$name("ocean-slope-post");

const mipKernel = tgpu
    .computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const x = input.gid.x;
        const y = input.gid.y;
        const width = mipLayout.$.params.width;
        const height = mipLayout.$.params.height;
        if (x >= width || y >= height) return;
        const a = std.textureLoad(mipLayout.$.source, d.vec2i(d.i32(x * 2), d.i32(y * 2)), 0);
        const b = std.textureLoad(mipLayout.$.source, d.vec2i(d.i32(x * 2 + 1), d.i32(y * 2)), 0);
        const c = std.textureLoad(mipLayout.$.source, d.vec2i(d.i32(x * 2), d.i32(y * 2 + 1)), 0);
        const e = std.textureLoad(
            mipLayout.$.source,
            d.vec2i(d.i32(x * 2 + 1), d.i32(y * 2 + 1)),
            0,
        );
        const mean = std.mul(std.add(std.add(a, b), std.add(c, e)), d.vec4f(0.25));
        const secondMoment = mean.z;
        const residual = std.max(d.f32(0), secondMoment - mean.x * mean.x - mean.y * mean.y);
        std.textureStore(
            mipLayout.$.output,
            d.vec2i(d.i32(x), d.i32(y)),
            d.vec4f(mean.x, mean.y, secondMoment, residual),
        );
    })
    .$name("ocean-slope-mip");

interface SlopeState {
    config: CascadeConfig;
    n: number;
    h0: GPUBuffer;
    x: GPUBuffer;
    z: GPUBuffer;
    xTemp: GPUBuffer;
    zTemp: GPUBuffer;
    params: GPUBuffer;
    mipParams: GPUBuffer[];
    texture: GPUTexture;
    slope: TgpuComputePipeline;
    post: TgpuComputePipeline;
    mip: TgpuComputePipeline;
    fftRow: TgpuComputePipeline;
    fftCol: TgpuComputePipeline;
    slopeGroup: TgpuBindGroup;
    postGroup: TgpuBindGroup;
    xRowGroup: TgpuBindGroup;
    xColGroup: TgpuBindGroup;
    zRowGroup: TgpuBindGroup;
    zColGroup: TgpuBindGroup;
    mipGroups: TgpuBindGroup[];
}
const states: SlopeState[] = [];

function createSlopeState(config: CascadeConfig): SlopeState {
    assertSlopeOnly(config);
    const { device, root } = Compute;
    const N = config.N;
    const complexBytes = N * N * 8;
    const make = (label: string, usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE) =>
        device.createBuffer({
            label,
            size: complexBytes,
            usage,
        });
    const h0 = device.createBuffer({
        label: `ocean-slope-h0-${N}`,
        size: complexBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(h0, 0, generateH0(config, 0));
    // COPY_SRC: the I3g-r2 level-0 computation claim (`ocean-slope` gym scenario) reads these
    // buffers straight back — the SAME post-inverse-FFT f32 complex values `slopePostKernel`
    // rounds to the published rgba16float texture — never a second GPU state built from the same
    // kernels. `xTemp`/`zTemp` are pure intra-pass scratch and carry no COPY_SRC.
    const x = make(`ocean-slope-x-${N}`, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const z = make(`ocean-slope-z-${N}`, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const xTemp = make(`ocean-slope-x-temp-${N}`);
    const zTemp = make(`ocean-slope-z-temp-${N}`);
    const params = device.createBuffer({
        label: `ocean-slope-params-${N}`,
        size: d.sizeOf(slopeParams),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const mipParamsBuffers = Array.from({ length: SLOPE_MIP_LEVELS - 1 }, (_, level) =>
        device.createBuffer({
            label: `ocean-slope-mip-params-${N}-${level + 1}`,
            size: d.sizeOf(mipParams),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
    );
    const texture = device.createTexture({
        label: `ocean-slope-${N}`,
        size: { width: N, height: N },
        mipLevelCount: SLOPE_MIP_LEVELS,
        format: "rgba16float",
        // COPY_SRC: the published-texture readback arm (`ocean-slope` gym scenario) probes this
        // exact resource via `probeTexture`, which requires the source to carry COPY_SRC — without
        // it the only way to inspect this texture's content is a second GPU state built from the
        // same kernels, which the arm is written to avoid.
        usage:
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_SRC,
    });
    const h0T = root.createBuffer(d.arrayOf(d.vec2f, N * N), h0).$usage("storage");
    const xT = root.createBuffer(d.arrayOf(d.vec2f, N * N), x).$usage("storage");
    const zT = root.createBuffer(d.arrayOf(d.vec2f, N * N), z).$usage("storage");
    const xTempT = root.createBuffer(d.arrayOf(d.vec2f, N * N), xTemp).$usage("storage");
    const zTempT = root.createBuffer(d.arrayOf(d.vec2f, N * N), zTemp).$usage("storage");
    const paramsT = root.createBuffer(slopeParams, params).$usage("uniform");
    const mipParamsT = mipParamsBuffers.map((buffer) =>
        root.createBuffer(mipParams, buffer).$usage("uniform"),
    );
    const fft = getFftKernels(N);
    const slope = root
        .createComputePipeline({ compute: slopeKernel })
        .$name(`ocean-slope-spectrum-${N}`);
    const post = root
        .createComputePipeline({ compute: slopePostKernel })
        .$name(`ocean-slope-post-${N}`);
    const mip = root.createComputePipeline({ compute: mipKernel }).$name(`ocean-slope-mip-${N}`);
    const fftRow = root
        .createComputePipeline({ compute: fft.rowKernel })
        .$name(`ocean-slope-fft-row-${N}`);
    const fftCol = root
        .createComputePipeline({ compute: fft.colKernel })
        .$name(`ocean-slope-fft-col-${N}`);
    const slopeGroup = root.createBindGroup(slopeLayout, {
        h0: h0T,
        x: xT,
        z: zT,
        params: paramsT,
    });
    const postGroup = root.createBindGroup(slopePostLayout, {
        x: xT,
        z: zT,
        output: texture.createView({ baseMipLevel: 0, mipLevelCount: 1 }),
        params: paramsT,
    });
    const xRowGroup = root.createBindGroup(fft.layout, { input: xT, output: xTempT });
    const xColGroup = root.createBindGroup(fft.layout, { input: xTempT, output: xT });
    const zRowGroup = root.createBindGroup(fft.layout, { input: zT, output: zTempT });
    const zColGroup = root.createBindGroup(fft.layout, { input: zTempT, output: zT });
    const mipGroups: TgpuBindGroup[] = [];
    for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
        const source = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
        const output = texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
        mipGroups.push(
            root.createBindGroup(mipLayout, {
                source,
                output,
                params: mipParamsT[level - 1],
            }),
        );
    }
    return {
        config,
        n: N,
        h0,
        x,
        z,
        xTemp,
        zTemp,
        params,
        mipParams: mipParamsBuffers,
        texture,
        slope,
        post,
        mip,
        fftRow,
        fftCol,
        slopeGroup,
        postGroup,
        xRowGroup,
        xColGroup,
        zRowGroup,
        zColGroup,
        mipGroups,
    };
}

function destroy(state: SlopeState): void {
    for (const buffer of [
        state.h0,
        state.x,
        state.z,
        state.xTemp,
        state.zTemp,
        state.params,
        ...state.mipParams,
    ])
        buffer.destroy();
    state.texture.destroy();
}

function writeParams(state: SlopeState, time: number): void {
    const bytes = new ArrayBuffer(d.sizeOf(slopeParams));
    const u32 = new Uint32Array(bytes);
    const f32 = new Float32Array(bytes);
    u32[0] = state.n;
    f32[1] = state.config.L;
    f32[2] = time;
    Compute.device.queue.writeBuffer(state.params, 0, bytes);
}

function writeMipParams(state: SlopeState, level: number, width: number, height: number): void {
    const bytes = new ArrayBuffer(d.sizeOf(mipParams));
    const u32 = new Uint32Array(bytes);
    u32[0] = width;
    u32[1] = height;
    Compute.device.queue.writeBuffer(state.mipParams[level - 1], 0, bytes);
}

function encode(state: SlopeState, encoder: GPUCommandEncoder): void {
    const run = (
        label: string,
        pipeline: TgpuComputePipeline,
        group: TgpuBindGroup,
        x: number,
        y = 1,
    ) => {
        const pass = encoder.beginComputePass({ label, timestampWrites: Compute.span?.(label) });
        pipeline.with(group).with(pass).dispatchWorkgroups(x, y);
        pass.end();
    };
    const lines = Math.ceil((state.n * state.n) / 64);
    run("ocean-slope-spectrum", state.slope, state.slopeGroup, lines);
    runFft(state, encoder, "ocean-slope-fft-row", state.fftRow, state.xRowGroup);
    runFft(state, encoder, "ocean-slope-fft-col", state.fftCol, state.xColGroup);
    runFft(state, encoder, "ocean-slope-fft-row", state.fftRow, state.zRowGroup);
    runFft(state, encoder, "ocean-slope-fft-col", state.fftCol, state.zColGroup);
    // The post reads the completed inverse transforms in x/z. Reuse the FFT output buffers as the
    // post inputs; no displacement buffer or lambda write exists on this path.
    const post = encoder.beginComputePass({
        label: "ocean-slope-post",
        timestampWrites: Compute.span?.("ocean-slope-post"),
    });
    state.post
        .with(state.postGroup)
        .with(post)
        .dispatchWorkgroups(Math.ceil(state.n / 8), Math.ceil(state.n / 8));
    post.end();
    for (let level = 1; level < SLOPE_MIP_LEVELS; level++) {
        const width = slopeMipSize(state.config, level);
        writeMipParams(state, level, width, width);
        run(
            `ocean-slope-mip-${level}`,
            state.mip,
            state.mipGroups[level - 1],
            Math.ceil(width / 8),
            Math.ceil(width / 8),
        );
    }
}

function runFft(
    state: SlopeState,
    encoder: GPUCommandEncoder,
    label: string,
    pipeline: TgpuComputePipeline,
    group: TgpuBindGroup,
): void {
    const pass = encoder.beginComputePass({ label, timestampWrites: Compute.span?.(label) });
    pipeline.with(group).with(pass).dispatchWorkgroups(state.n);
    pass.end();
}

/** GPU system for the slope-only cascade. */
export const slopeCompute: System = {
    name: "ocean-slope-compute",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update(state) {
        const encoder = Render.encoder;
        if (!encoder) return;
        for (const slope of states) {
            writeParams(slope, state.time.elapsed);
            encode(slope, encoder);
        }
    },
};

/** Build the slope-only GPU resources. Called by OceanPlugin's warm hook. */
export function buildSlopes(): void {
    if (states.length > 0) teardownSlopes();
    for (const config of SLOPE_CASCADE_CONFIGS) {
        assertSlopeOnly(config);
        const state = createSlopeState(config);
        states.push(state);
        Compute.textures.set(`slope${states.length - 1}`, state.texture);
    }
    Compute.samplers.set(
        "slopeSampler",
        Compute.device.createSampler({
            label: "ocean-slope-sampler",
            minFilter: "linear",
            magFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        }),
    );
}

/** Release slope-only GPU resources and named texture publications. */
export function teardownSlopes(): void {
    for (const state of states) destroy(state);
    states.length = 0;
    for (let i = 0; i < SLOPE_CASCADE_CONFIGS.length; i++) Compute.textures.delete(`slope${i}`);
    Compute.samplers.delete("slopeSampler");
}

/** render-only access to the published slope texture. */
export function getSlopeTexture(cascade: number): GPUTexture | null {
    return states[cascade]?.texture ?? null;
}

/** Read-only accessor for one slope cascade's own post-inverse-FFT f32 complex buffers (`.x`
 *  real component = the real-space slopeX/slopeZ value `slopePostKernel` reads to build the
 *  published texture). `states` is module-private so the I3g-r2 level-0 computation claim can
 *  reach the SAME live buffers `slopeCompute` writes every frame — never a second GPU state
 *  built from the same kernels — only through this accessor. Returns `null` before `buildSlopes`
 *  has run for the requested cascade index. */
export function getSlopeBuffers(cascade: number): { x: GPUBuffer; z: GPUBuffer } | null {
    const state = states[cascade];
    return state ? { x: state.x, z: state.z } : null;
}
