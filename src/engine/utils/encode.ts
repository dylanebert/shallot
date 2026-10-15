import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
    clamp,
    max,
    normalize,
    pack2x16float,
    pack2x16snorm,
    pack4x8unorm,
    pow,
    select,
    unpack2x16float,
    unpack2x16snorm,
    unpack4x8unorm,
} from "typegpu/std";
import { chunk } from "./tgsl";

// WGSL chunks resolve independently with strict names; each includes its own dependencies.

// snorm16 represents 0 and ±1 exactly, avoiding biased axis-aligned contact normals.

/** octahedral-encode a unit normal to an snorm16x2 `u32` (the storage normal, 12 B → 4 B;
 *  Cigolle et al. 2014). Not for interpolated or filtered normals: the octahedral seam breaks
 *  under interpolation; cross those as a plain `vec3` and renormalize. */
export const octEncodeNormal = tgpu.fn(
    [d.vec3f],
    d.u32,
)((n) => {
    "use gpu";
    const denom = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z);
    const inv = select(1 / denom, 0, denom <= 0);
    const p = d.vec2f(n.x * inv, n.y * inv);
    const signX = select(d.f32(-1), d.f32(1), p.x >= 0);
    const signY = select(d.f32(-1), d.f32(1), p.y >= 0);
    const folded = d.vec2f((1 - Math.abs(p.y)) * signX, (1 - Math.abs(p.x)) * signY);
    return pack2x16snorm(select(p, folded, n.z < 0));
});

/** decode an snorm16x2 `u32` back to a unit normal: the inverse of {@link octEncodeNormal}. Cardinals
 *  round-trip exactly (0 ↔ 0, ±1 ↔ ±32767). */
export const octDecodeNormal = tgpu.fn(
    [d.u32],
    d.vec3f,
)((enc) => {
    "use gpu";
    const p = unpack2x16snorm(enc);
    const z = 1 - Math.abs(p.x) - Math.abs(p.y);
    const signX = select(d.f32(-1), d.f32(1), p.x >= 0);
    const signY = select(d.f32(-1), d.f32(1), p.y >= 0);
    const folded = d.vec3f((1 - Math.abs(p.y)) * signX, (1 - Math.abs(p.x)) * signY, z);
    return normalize(select(d.vec3f(p.x, p.y, z), folded, z < 0));
});

/** WGSL `octEncodeNormal(n) -> u32` + `octDecodeNormal(enc) -> vec3<f32>`: the snorm16x2 storage-normal
 *  codec, resolved from the CPU-callable TGSL functions. */
export const octEncodeWgsl = chunk("octEncodeWgsl", [octEncodeNormal, octDecodeNormal]);

// snorm16x4 has per-component error at most 1/32767 across [-1, 1]; 0 and ±1 are exact.

/** pack a quaternion into two snorm16x2 `u32` lanes (xy, zw), 16 B → 8 B. */
export const packQuatSnorm16x4 = tgpu.fn(
    [d.vec4f],
    d.vec2u,
)((q) => {
    "use gpu";
    return d.vec2u(pack2x16snorm(d.vec2f(q.x, q.y)), pack2x16snorm(d.vec2f(q.z, q.w)));
});

/** unpack two snorm16x2 `u32` lanes back to a unit quaternion, renormalizing to absorb the per-component
 *  quantization — `quatMul` / `quatRotate` downstream expect unit input. */
export const unpackQuatSnorm16x4 = tgpu.fn(
    [d.vec2u],
    d.vec4f,
)((p) => {
    "use gpu";
    const xy = unpack2x16snorm(p.x);
    const zw = unpack2x16snorm(p.y);
    return normalize(d.vec4f(xy.x, xy.y, zw.x, zw.y));
});

/** WGSL `packQuatSnorm16x4(q) -> vec2<u32>` + `unpackQuatSnorm16x4(p) -> vec4<f32>`: the quaternion
 *  storage codec for a field whose precision feeds a finite-difference downstream (a body quat read
 *  back as angular velocity, an iter-mutated-state case). Worst-case angular error
 *  < 0.06° per round-trip. */
export const quatSnorm16x4Wgsl = chunk("quatSnorm16x4Wgsl", [
    packQuatSnorm16x4,
    unpackQuatSnorm16x4,
]);

// LDR colors store sRGB-encoded RGB + linear alpha as a `pack4x8unorm` u32. sRGB encoding gives uniform
// perceptual precision across the range and lets hex inputs (`0xRRGGBB`) round-trip exactly
// through the byte form. Alpha is linear in [0,1]; the sRGB transfer applies only to color channels.

/** sRGB → linear transfer on one channel (IEC 61966-2-1). */
export const srgbToLinear1 = tgpu.fn(
    [d.f32],
    d.f32,
)((c) => {
    "use gpu";
    return select(pow(max((c + 0.055) / 1.055, 0), 2.4), c / 12.92, c <= 0.04045);
});

/** linear → sRGB transfer on one channel (IEC 61966-2-1). */
export const linearToSrgb1 = tgpu.fn(
    [d.f32],
    d.f32,
)((c) => {
    "use gpu";
    return select(1.055 * pow(max(c, 0), 1 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
});

/** decode an sRGB-packed LDR color to linear rgb + linear alpha. */
export const unpackLdrColor = tgpu.fn(
    [d.u32],
    d.vec4f,
)((p) => {
    "use gpu";
    const v = unpack4x8unorm(p);
    return d.vec4f(srgbToLinear1(v.x), srgbToLinear1(v.y), srgbToLinear1(v.z), v.w);
});

/** sRGB-encode linear `rgb` + linear `alpha` into a `pack4x8unorm` u32. */
export const packLdrColor = tgpu.fn(
    [d.vec3f, d.f32],
    d.u32,
)((rgb, alpha) => {
    "use gpu";
    return pack4x8unorm(
        d.vec4f(linearToSrgb1(rgb.x), linearToSrgb1(rgb.y), linearToSrgb1(rgb.z), alpha),
    );
});

/** WGSL `srgbToLinear1(c) -> f32` + `unpackLdrColor(p: u32) -> vec4<f32>`: the LDR color read side.
 *  Splice into a surface that reads a `u32` color binding. */
export const ldrColorUnpackWgsl = chunk("ldrColorUnpackWgsl", [unpackLdrColor]);

/** WGSL `linearToSrgb1(c) -> f32` + `packLdrColor(rgb, alpha) -> u32`: the LDR color write side, split
 *  from {@link ldrColorUnpackWgsl} so a reader shader doesn't drag in the encode helpers. */
export const ldrColorPackWgsl = chunk("ldrColorPackWgsl", [packLdrColor]);

// Khronos GL_EXT_packed_float layout shares f16's exponent bias (15), allowing mantissa-bit drops.
// Clamp below f16's maximum so conversion cannot round an f11 channel to infinity.

/** decode an r11g11b10ufloat-packed HDR color. */
export const unpackHdrColor = tgpu.fn(
    [d.u32],
    d.vec3f,
)((p) => {
    "use gpu";
    const r11 = p & 0x7ff;
    const g11 = (p >>> 11) & 0x7ff;
    const b10 = (p >>> 22) & 0x3ff;
    const rgv = unpack2x16float((r11 << 4) | (g11 << 20));
    const bbv = unpack2x16float(b10 << 5);
    return d.vec3f(rgv.x, rgv.y, bbv.x);
});

/** encode an HDR color to r11g11b10ufloat, clamped to the format's [0, 65024] range. */
export const packHdrColor = tgpu.fn(
    [d.vec3f],
    d.u32,
)((rgb) => {
    "use gpu";
    const c = d.vec3f(clamp(rgb.x, 0, 65024), clamp(rgb.y, 0, 65024), clamp(rgb.z, 0, 65024));
    const rg = pack2x16float(d.vec2f(c.x, c.y));
    const bb = pack2x16float(d.vec2f(c.z, 0));
    return ((rg >>> 4) & 0x7ff) | (((rg >>> 20) & 0x7ff) << 11) | (((bb >>> 5) & 0x3ff) << 22);
});

/** WGSL `unpackHdrColor(p: u32) -> vec3<f32>`: the r11g11b10ufloat color read side. */
export const hdrColorUnpackWgsl = chunk("hdrColorUnpackWgsl", [unpackHdrColor]);

/** WGSL `packHdrColor(rgb) -> u32`: the HDR color write side, split from {@link hdrColorUnpackWgsl}
 *  so a reader shader doesn't drag in the encode helpers. */
export const hdrColorPackWgsl = chunk("hdrColorPackWgsl", [packHdrColor]);

// Smallest-3 uses 4 bytes with ~0.1° max error, suitable for reads without compounding.
// Not for iter-mutated state that is later finite-differenced.

/** pack a quaternion as smallest-3 (10-10-10-2 in one `u32`). */
export const packQuatSmallest3 = tgpu.fn(
    [d.vec4f],
    d.u32,
)(/* wgsl */ `(q: vec4f) -> u32 {
    let aq = abs(q);
    let largest: u32 = select(
        select(2u, 3u, aq.w > aq.z),
        select(0u, 1u, aq.y > aq.x),
        max(aq.x, aq.y) > max(aq.z, aq.w),
    );
    let s = select(-1.0, 1.0, q[largest] >= 0.0);
    let q2 = q * s;
    var abc: vec3<f32>;
    switch largest {
        case 0u: { abc = q2.yzw; }
        case 1u: { abc = vec3<f32>(q2.x, q2.z, q2.w); }
        case 2u: { abc = vec3<f32>(q2.x, q2.y, q2.w); }
        default: { abc = q2.xyz; }
    }
    let scale = 511.0 / 1.41421356;
    let s0 = i32(clamp(round(abc.x * scale), -511.0, 511.0));
    let s1 = i32(clamp(round(abc.y * scale), -511.0, 511.0));
    let s2 = i32(clamp(round(abc.z * scale), -511.0, 511.0));
    return (u32(s0) & 0x3FFu)
         | ((u32(s1) & 0x3FFu) << 10u)
         | ((u32(s2) & 0x3FFu) << 20u)
         | (largest << 30u);
}`);

/** unpack a smallest-3 `u32` back to a unit quaternion, reconstructing the dropped component. */
export const unpackQuatSmallest3 = tgpu.fn(
    [d.u32],
    d.vec4f,
)(/* wgsl */ `(p: u32) -> vec4f {
    let largest = (p >> 30u) & 3u;
    let s0 = bitcast<i32>(p << 22u) >> 22u;
    let s1 = bitcast<i32>((p << 12u) & 0xFFC00000u) >> 22u;
    let s2 = bitcast<i32>((p << 2u) & 0xFFC00000u) >> 22u;
    let scale = 1.41421356 / 511.0;
    let abc = vec3<f32>(f32(s0), f32(s1), f32(s2)) * scale;
    let m = sqrt(max(0.0, 1.0 - dot(abc, abc)));
    switch largest {
        case 0u: { return vec4<f32>(m, abc.x, abc.y, abc.z); }
        case 1u: { return vec4<f32>(abc.x, m, abc.y, abc.z); }
        case 2u: { return vec4<f32>(abc.x, abc.y, m, abc.z); }
        default: { return vec4<f32>(abc.x, abc.y, abc.z, m); }
    }
}`);

/** WGSL `packQuatSmallest3(q) -> u32` + `unpackQuatSmallest3(p) -> vec4<f32>`: the compact quaternion
 *  storage codec. GPU-only: its WGSL bodies dynamically index vectors and switch on the result,
 *  neither of which TGSL expresses. */
export const smallest3Wgsl = chunk("smallest3Wgsl", [packQuatSmallest3, unpackQuatSmallest3]);

// Plain-JS mirrors avoid TypeGPU CPU dispatch and vector-allocation costs in mesh baking and uploads.
// encode.test.ts checks their packed words against the TGSL functions.

/** {@link octEncodeNormal} over three loose lanes — the mesh builder's per-vertex form. */
export function octEncode(x: number, y: number, z: number): number {
    // `Math.fround` marks every point the TGSL body builds a vector, and so stores f32 — the mirror
    // has to round where the schema would, or the two disagree on values near a lattice midpoint.
    const nx = Math.fround(x);
    const ny = Math.fround(y);
    const nz = Math.fround(z);
    const denom = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
    // Preserve the TGSL select's NaN branch: a NaN denominator takes the reciprocal.
    // A TypeGPU schema refuses non-finite CPU inputs, so the differential cannot cover this case.
    const inv = denom <= 0 ? 0 : 1 / denom;
    const px = Math.fround(nx * inv);
    const py = Math.fround(ny * inv);
    if (nz < 0) {
        const sx = px >= 0 ? 1 : -1;
        const sy = py >= 0 ? 1 : -1;
        return packSnorm2((1 - Math.abs(py)) * sx, (1 - Math.abs(px)) * sy);
    }
    return packSnorm2(px, py);
}

// The plain-JS mirrors' lattice steps: one f32-input, f64-scaling construction. WGSL packing has its
// own floor-based conversion and intermediate accuracy contract, so agreement at ambiguous inputs is
// not guaranteed by the CPU rounding choice.
const snorm16 = (v: number): number =>
    Math.round(Math.max(-1, Math.min(1, Math.fround(v))) * 32767) & 0xffff;
const unorm16 = (v: number): number =>
    Math.round(Math.max(0, Math.min(1, Math.fround(v))) * 65535) & 0xffff;
const unorm8 = (v: number): number =>
    Math.round(Math.max(0, Math.min(1, Math.fround(v))) * 255) & 0xff;

/** `pack2x16unorm` over two loose lanes — the mesh builder's per-vertex form. */
export function packUnorm2(x: number, y: number): number {
    return ((unorm16(y) << 16) | unorm16(x)) >>> 0;
}

function packSnorm2(x: number, y: number): number {
    return ((snorm16(y) << 16) | snorm16(x)) >>> 0;
}

/** {@link packLdrColor} over four loose lanes. */
export function packColor4(r: number, g: number, b: number, a: number): number {
    return (
        (unorm8(srgb(r)) | (unorm8(srgb(g)) << 8) | (unorm8(srgb(b)) << 16) | (unorm8(a) << 24)) >>>
        0
    );
}

// the `linearToSrgb1` transfer, with the f32 rounding its TGSL twin gets from the schema on both the
// argument and the return.
function srgb(c: number): number {
    const v = Math.fround(c);
    return Math.fround(v <= 0.0031308 ? v * 12.92 : 1.055 * Math.max(v, 0) ** (1 / 2.4) - 0.055);
}
