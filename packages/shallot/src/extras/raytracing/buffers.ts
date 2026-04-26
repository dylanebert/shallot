import { gbuf, type GBuf } from "../../standard/compute";

export const WAVEFRONT_RAY_STRIDE = 48;
export const WAVEFRONT_HIT_STRIDE = 32;
export const MAX_SHADOW_RAYS_PER_PIXEL = 5; // 1 sun + 4 point lights

export const WAVEFRONT_RAY_STRUCT_WGSL = /* wgsl */ `
struct WavefrontRay {
    origin: vec3<f32>,
    pixelIndex: u32,
    direction: vec3<f32>,
    flags: u32,
    throughput: vec3<f32>,
    _pad: f32,
}

const RAY_STRIDE_U32: u32 = 12u;

fn packPixel(x: u32, y: u32) -> u32 {
    return x | (y << 16u);
}

fn unpackPixelX(p: u32) -> u32 {
    return p & 0xFFFFu;
}

fn unpackPixelY(p: u32) -> u32 {
    return p >> 16u;
}

const BOUNCE_TYPE_MASK: u32 = 0xFu;
const BOUNCE_COUNT_SHIFT: u32 = 4u;
const BOUNCE_COUNT_MASK: u32 = 0xF0u;
const SHADOW_RAY_BIT: u32 = 0x200u;

fn rayBounceCount(flags: u32) -> u32 {
    return (flags & BOUNCE_COUNT_MASK) >> BOUNCE_COUNT_SHIFT;
}

fn isShadowRay(flags: u32) -> bool {
    return (flags & SHADOW_RAY_BIT) != 0u;
}

fn packRayFlags(bounceType: u32, bounceCount: u32) -> u32 {
    return (bounceType & BOUNCE_TYPE_MASK)
         | ((bounceCount << BOUNCE_COUNT_SHIFT) & BOUNCE_COUNT_MASK);
}

fn packShadowFlags(tMax: f32) -> u32 {
    return SHADOW_RAY_BIT | (pack2x16float(vec2(0.0, tMax)) & 0xFFFF0000u);
}

fn unpackShadowTMax(flags: u32) -> f32 {
    let t = unpack2x16float(flags & 0xFFFF0000u).y;
    return select(t, 1000.0, t <= 0.0);
}
`;

export const READ_RAY_WGSL = /* wgsl */ `
fn readRayLocal(idx: u32) -> WavefrontRay {
    let base = idx * RAY_STRIDE_U32;
    var r: WavefrontRay;
    r.origin = vec3(bitcast<f32>(rayBuffer[base]), bitcast<f32>(rayBuffer[base + 1u]), bitcast<f32>(rayBuffer[base + 2u]));
    r.pixelIndex = rayBuffer[base + 3u];
    r.direction = vec3(bitcast<f32>(rayBuffer[base + 4u]), bitcast<f32>(rayBuffer[base + 5u]), bitcast<f32>(rayBuffer[base + 6u]));
    r.flags = rayBuffer[base + 7u];
    r.throughput = vec3(bitcast<f32>(rayBuffer[base + 8u]), bitcast<f32>(rayBuffer[base + 9u]), bitcast<f32>(rayBuffer[base + 10u]));
    return r;
}
`;

export const WAVEFRONT_HIT_STRUCT_WGSL = /* wgsl */ `
struct WavefrontHit {
    normal: vec3<f32>,
    t: f32,
    entityId: u32,
    u: f32,
    v: f32,
    _pad: u32,
}
`;

// 6 u32s per pixel: accum(4) + firstHit(2)
export function pixelStateSize(pixelCount: number): number {
    return pixelCount * 24;
}

// 0: INPUT, 1: OUTPUT, 2: SHADOW, 3: pad
// 4: DBG_RAY_OVERFLOW, 5: DBG_SHADOW_OVERFLOW, 6: DBG_MAX_OUTPUT, 7: DBG_MAX_SHADOW
// 8: DBG_SHADE_PIXEL_OOB, 9: DBG_SHADOW_PIXEL_OOB, 10: DBG_BOUNCE_CLIPPED, 11: DBG_SHADOW_UNCLAMPED
export const COUNTER_BUF_SIZE = 48;

export interface WavefrontBuffers {
    rays0: GBuf;
    hits: GBuf;
    pixelState: GBuf;
    rays1: GBuf;
    shadowRays: GBuf;
    shadowHits: GBuf;
    counters: GBuf;
    indirect: GBuf;
    zeroCounters: Uint32Array;
}

const BYTES_PER_PIXEL_SHADOW_RAYS = WAVEFRONT_RAY_STRIDE * MAX_SHADOW_RAYS_PER_PIXEL;

export function maxPixelsForDevice(device: GPUDevice): number {
    const limit = device.limits.maxStorageBufferBindingSize;
    return Math.floor(limit / BYTES_PER_PIXEL_SHADOW_RAYS);
}

export function createWavefrontBuffers(device: GPUDevice, pixelCount: number): WavefrontBuffers {
    const storageRW = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const maxBounceRays = pixelCount * 4;
    const rayBufSize = WAVEFRONT_RAY_STRIDE * maxBounceRays;
    const hitBufSize = WAVEFRONT_HIT_STRIDE * maxBounceRays;
    const shadowBufSize = MAX_SHADOW_RAYS_PER_PIXEL * pixelCount;
    const pxStateSize = pixelStateSize(pixelCount);

    return {
        rays0: gbuf(device, "wf-rays-0", storageRW, () => rayBufSize),
        hits: gbuf(device, "wf-hits", storageRW, () => hitBufSize),
        pixelState: gbuf(device, "wf-pixel-state", storageRW, () => pxStateSize),
        rays1: gbuf(device, "wf-rays-1", storageRW, () => rayBufSize),
        shadowRays: gbuf(
            device,
            "wf-shadow-rays",
            storageRW,
            () => WAVEFRONT_RAY_STRIDE * shadowBufSize,
        ),
        shadowHits: gbuf(
            device,
            "wf-shadow-hits",
            storageRW,
            () => WAVEFRONT_HIT_STRIDE * shadowBufSize,
        ),
        counters: gbuf(
            device,
            "wf-counters",
            storageRW | GPUBufferUsage.COPY_SRC,
            () => COUNTER_BUF_SIZE,
        ),
        indirect: gbuf(
            device,
            "wf-indirect",
            GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
            () => 12,
        ),
        zeroCounters: new Uint32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    };
}
