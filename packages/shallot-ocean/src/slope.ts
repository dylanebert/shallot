// High-wavenumber slope cascades. These modes are rendered as slope textures only: no height,
// horizontal displacement, or Jacobian output is allocated for this band. The source is a seeded
// height spectrum, converted to directional slope spectra and inverse-transformed with the same
// radix-2 FFT used by the displacement cascades. Each output texture owns every mip level so a
// later shading pass can select measured residual variance without inventing a footprint heuristic.

import { Compute, type System } from "@dylanebert/shallot";
import { BeginFrameSystem, Render } from "@dylanebert/shallot/render/core";
import { PrepassSystem } from "@dylanebert/shallot/sear/core";
import { idiv } from "@dylanebert/shallot/utils/core";
import tgpu, { type TgpuBindGroup, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { ifft2 } from "./fft";
import { getFftKernels } from "./gpu-fft";
import { type CascadeConfig, directionalDensity, generateH0, kIndex, SEA_STATE } from "./spectrum";

/** The short-gravity/capillary cascade. Its Nyquist edge leaves headroom above 60 rad/m. */
export const SLOPE_CASCADE_CONFIGS: readonly CascadeConfig[] = Object.freeze([
    {
        N: 256,
        L: 13,
        windSpeed: SEA_STATE.windSpeed,
        windDir: SEA_STATE.windDir,
        lambda: 0,
        kLo: 8.482300164692441,
        kHi: 60,
    },
]);

/** Number of complete mip levels in each slope texture, including level zero. */
export const SLOPE_MIP_LEVELS = Math.floor(Math.log2(SLOPE_CASCADE_CONFIGS[0].N)) + 1;

/** Per-level dimensions for a square slope texture. */
export function slopeMipSize(config: CascadeConfig, level: number): number {
    if (!Number.isInteger(level) || level < 0 || level > Math.floor(Math.log2(config.N))) {
        throw new RangeError(`slopeMipSize: invalid mip level ${level}`);
    }
    return Math.max(1, config.N >> level);
}

/** GPU slope texture for a cascade, or null before the Ocean plugin warms. */
const slopeTextures: GPUTexture[] = [];
const PI = Math.PI;
export function getSlopeTexture(cascade = 0): GPUTexture | null {
    return slopeTextures[cascade] ?? null;
}

/** Returns the immutable slope-cascade configuration list. */
export function getSlopeCascadeConfigs(): readonly CascadeConfig[] {
    return SLOPE_CASCADE_CONFIGS;
}

/**
 * Sum of the composed slope-PSD over the declared band. `i·k̂·h̃` preserves the source PSD's
 * energy while carrying the directional slope vector into the texture; the two components sum to
 * one mode's height-PSD because k̂x² + k̂z² = 1.
 */
export function composedSlopePsd(config: CascadeConfig): number {
    const dk = (2 * Math.PI) / config.L;
    let total = 0;
    for (let y = 0; y < config.N; y++) {
        for (let x = 0; x < config.N; x++) {
            const kx = kIndex(x, config.N) * dk;
            const kz = kIndex(y, config.N) * dk;
            const k = Math.hypot(kx, kz);
            if (k < config.kLo || k > config.kHi) continue;
            total +=
                directionalDensity(
                    kx,
                    kz,
                    SEA_STATE.windSpeed,
                    SEA_STATE.windDir,
                    SEA_STATE.omegaC,
                ) *
                dk *
                dk;
        }
    }
    return total;
}

/** CPU reference for the directional slope spectra used by the GPU slope kernel. */
export function slopeSpectra(
    h: Float32Array,
    config: CascadeConfig,
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
            const magnitude = Math.hypot(kx, kz);
            const xHat = magnitude > 1e-6 ? kx / magnitude : 0;
            const zHat = magnitude > 1e-6 ? kz / magnitude : 0;
            const hr = h[i * 2];
            const hi = h[i * 2 + 1];
            x[i * 2] = -xHat * hi;
            x[i * 2 + 1] = xHat * hr;
            z[i * 2] = -zHat * hi;
            z[i * 2 + 1] = zHat * hr;
        }
    }
    return { x, z };
}

/** CPU reference for one slope realization, including the two inverse FFTs. */
export function runSlopeCpuPipeline(
    h0: Float32Array,
    config: CascadeConfig = SLOPE_CASCADE_CONFIGS[0],
    time = 0,
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
            const omega = Math.sqrt(9.81 * Math.max(Math.hypot(kx, kz), 1e-6));
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
    const { x, z } = slopeSpectra(h, config);
    return { h, x, z, xField: ifft2(x, N), zField: ifft2(z, N) };
}

const slopeParams = d
    .struct({ n: d.u32, L: d.f32, time: d.f32, width: d.u32, height: d.u32, pad: d.vec3u })
    .$name("SlopeParams");
const slopeUpdateLayout = tgpu.bindGroupLayout({
    h0: { storage: d.arrayOf(d.vec2f), access: "readonly" },
    h: { storage: d.arrayOf(d.vec2f), access: "mutable" },
    params: { uniform: slopeParams },
});
const slopeLayout = tgpu.bindGroupLayout({
    h: { storage: d.arrayOf(d.vec2f), access: "readonly" },
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
const mipParams = d.struct({ width: d.u32, height: d.u32, pad: d.vec2u }).$name("SlopeMipParams");
const mipLayout = tgpu.bindGroupLayout({
    source: { texture: d.texture2d(d.f32), visibility: ["compute"] },
    output: {
        storageTexture: d.textureStorage2d("rgba16float", "write-only"),
        visibility: ["compute"],
    },
    params: { uniform: mipParams, visibility: ["compute"] },
});

const slopeUpdateKernel = tgpu
    .computeFn({ workgroupSize: [64], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const idx = input.gid.x;
        const N = slopeUpdateLayout.$.params.n;
        if (idx >= N * N) return;
        const x = idx % N;
        const y = idiv(idx, N);
        const dk = (d.f32(2) * PI) / slopeUpdateLayout.$.params.L;
        const kx = std.select(d.f32(x) - d.f32(N), d.f32(x), d.f32(x) <= d.f32(N) / d.f32(2)) * dk;
        const kz = std.select(d.f32(y) - d.f32(N), d.f32(y), d.f32(y) <= d.f32(N) / d.f32(2)) * dk;
        const omega = std.sqrt(d.f32(9.81) * std.max(std.sqrt(kx * kx + kz * kz), d.f32(1e-6)));
        const negX = (N - x) % N;
        const negY = (N - y) % N;
        const h0 = slopeUpdateLayout.$.h0[idx];
        const hn = slopeUpdateLayout.$.h0[negY * N + negX];
        const c = d.vec2f(
            std.cos(omega * slopeUpdateLayout.$.params.time),
            std.sin(omega * slopeUpdateLayout.$.params.time),
        );
        const cm = d.vec2f(
            std.cos(-omega * slopeUpdateLayout.$.params.time),
            std.sin(-omega * slopeUpdateLayout.$.params.time),
        );
        const nr = d.vec2f(hn.x, -hn.y);
        slopeUpdateLayout.$.h[idx] = d.vec2f(
            h0.x * c.x - h0.y * c.y + nr.x * cm.x - nr.y * cm.y,
            h0.x * c.y + h0.y * c.x + nr.x * cm.y + nr.y * cm.x,
        );
    })
    .$name("ocean-slope-update");

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
        const magnitude = std.sqrt(kx * kx + kz * kz);
        const inv = std.select(d.f32(0), d.f32(1) / magnitude, magnitude > d.f32(1e-6));
        const xHat = kx * inv;
        const zHat = kz * inv;
        const h = slopeLayout.$.h[idx];
        slopeLayout.$.x[idx] = d.vec2f(-xHat * h.y, xHat * h.x);
        slopeLayout.$.z[idx] = d.vec2f(-zHat * h.y, zHat * h.x);
    })
    .$name("ocean-slope-spectrum");

const slopePostKernel = tgpu
    .computeFn({ workgroupSize: [8, 8], in: { gid: d.builtin.globalInvocationId } })((input) => {
        "use gpu";
        const x = input.gid.x;
        const y = input.gid.y;
        const N = slopePostLayout.$.params.n;
        if (x >= N || y >= N) return;
        const valueX = slopePostLayout.$.x[y * N + x];
        const valueZ = slopePostLayout.$.z[y * N + x];
        std.textureStore(
            slopePostLayout.$.output,
            d.vec2i(d.i32(x), d.i32(y)),
            d.vec4f(valueX.x, valueZ.x, 0, 1),
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
        const sum = std.add(std.add(a, b), std.add(c, e));
        std.textureStore(
            mipLayout.$.output,
            d.vec2i(d.i32(x), d.i32(y)),
            std.mul(sum, d.vec4f(0.25)),
        );
    })
    .$name("ocean-slope-mip");

interface SlopeState {
    config: CascadeConfig;
    n: number;
    h0: GPUBuffer;
    h: GPUBuffer;
    x: GPUBuffer;
    z: GPUBuffer;
    xTemp: GPUBuffer;
    zTemp: GPUBuffer;
    params: GPUBuffer;
    mipParams: GPUBuffer[];
    texture: GPUTexture;
    update: TgpuComputePipeline;
    slope: TgpuComputePipeline;
    post: TgpuComputePipeline;
    mip: TgpuComputePipeline;
    fftRow: TgpuComputePipeline;
    fftCol: TgpuComputePipeline;
    updateGroup: TgpuBindGroup;
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
    const { device, root } = Compute;
    const N = config.N;
    const complexBytes = N * N * 8;
    const make = (label: string) =>
        device.createBuffer({ label, size: complexBytes, usage: GPUBufferUsage.STORAGE });
    const h0 = device.createBuffer({
        label: `ocean-slope-h0-${N}`,
        size: complexBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(h0, 0, generateH0(config, 0));
    const h = make(`ocean-slope-h-${N}`);
    const x = make(`ocean-slope-x-${N}`);
    const z = make(`ocean-slope-z-${N}`);
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
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const h0T = root.createBuffer(d.arrayOf(d.vec2f, N * N), h0).$usage("storage");
    const hT = root.createBuffer(d.arrayOf(d.vec2f, N * N), h).$usage("storage");
    const xT = root.createBuffer(d.arrayOf(d.vec2f, N * N), x).$usage("storage");
    const zT = root.createBuffer(d.arrayOf(d.vec2f, N * N), z).$usage("storage");
    const xTempT = root.createBuffer(d.arrayOf(d.vec2f, N * N), xTemp).$usage("storage");
    const zTempT = root.createBuffer(d.arrayOf(d.vec2f, N * N), zTemp).$usage("storage");
    const paramsT = root.createBuffer(slopeParams, params).$usage("uniform");
    const mipParamsT = mipParamsBuffers.map((buffer) =>
        root.createBuffer(mipParams, buffer).$usage("uniform"),
    );
    const fft = getFftKernels(N);
    const update = root
        .createComputePipeline({ compute: slopeUpdateKernel })
        .$name(`ocean-slope-update-${N}`);
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
    const updateGroup = root.createBindGroup(slopeUpdateLayout, {
        h0: h0T,
        h: hT,
        params: paramsT,
    });
    const slopeGroup = root.createBindGroup(slopeLayout, { h: hT, x: xT, z: zT, params: paramsT });
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
        const width = slopeMipSize(config, level);
        const source = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 });
        const output = texture.createView({ baseMipLevel: level, mipLevelCount: 1 });
        mipGroups.push(
            root.createBindGroup(mipLayout, {
                source,
                output,
                params: mipParamsT[level - 1],
            }),
        );
        void width;
    }
    return {
        config,
        n: N,
        h0,
        h,
        x,
        z,
        xTemp,
        zTemp,
        params,
        mipParams: mipParamsBuffers,
        texture,
        update,
        slope,
        post,
        mip,
        fftRow,
        fftCol,
        updateGroup,
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
        state.h,
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
    run("ocean-slope-update", state.update, state.updateGroup, lines);
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
        const state = createSlopeState(config);
        states.push(state);
        slopeTextures.push(state.texture);
        Compute.textures.set(`slope${states.length - 1}`, state.texture);
    }
}

/** Release slope-only GPU resources and named texture publications. */
export function teardownSlopes(): void {
    for (const state of states) destroy(state);
    states.length = 0;
    for (let i = 0; i < SLOPE_CASCADE_CONFIGS.length; i++) Compute.textures.delete(`slope${i}`);
    slopeTextures.length = 0;
}
