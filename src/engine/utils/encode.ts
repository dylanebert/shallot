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

// The GPU storage codecs, each a single TGSL function: one source that runs on the CPU
// (a `bun test` calls it directly) and resolves to the WGSL a shader splices. Lattice drift between a
// CPU packer and a GPU unpacker is the failure this shape makes unrepresentable: a CPU oct encoder on
// a unorm16 lattice against a GPU decoder on snorm16 puts a steady torque on settled stacks.
//
// The `*Wgsl()` chunks below are the splice surface for a raw-WGSL shader, resolved with `names:
// "strict"` so the emitted function names are the ones documented here. Each chunk is self-contained
// (its own resolve, its own dependencies) and shares none with another, so any of them splice together.

// snorm16 mapping (pack2x16snorm): (-1, 1) ↔ (-32767, 32767), with 0 ↔ 0 exact. A unorm16 mapping
// (-1, 1) ↔ (0, 65535) puts 0 between two integer rails, so axis-aligned vectors decoded as (0, ±1, 0)
// round-trip to (0, 1, ±1.5e-5). For contact normals on a flat ground, that asymmetric z-bias gives a
// non-cancelling residual torque on the four corner contacts and a steady quaternion drift on settled
// boxes. snorm16 makes ±1 and 0 round-trip exactly.

/** octahedral-encode a unit normal to an snorm16x2 `u32` (the storage normal, 12 B → 4 B;
 *  Cigolle et al. 2014). **Never for an interpolated or filtered normal** — the octahedral seam breaks
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
 *  codec. Splice into a producer that packs a normal or a reader that unpacks one; bit-identical to the
 *  CPU-callable {@link octEncodeNormal}, since it *is* that function. */
export const octEncodeWgsl = chunk("octEncodeWgsl", [octEncodeNormal, octDecodeNormal]);

// snorm16x4 quaternion: 4 components packed into 2 u32 via pack2x16snorm. Per-component error bound
// 1/32767 ≈ 3.05e-5 (uniform across [-1, 1]). Identity (0,0,0,1) and the six 180° axis-aligned rotations
// are bit-exact: 0 ↔ 0 and ±1 ↔ ±32767 sit on lattice rails. Decode renormalizes to absorb per-component
// quantization into a unit quat — downstream quatMul/quatRotate expect unit input. Cardinal exactness
// plus the per-component and worst-case angular bounds are validated in encode.test.ts.

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

// HDR colors store as r11g11b10ufloat manual u32 pack — Khronos GL_EXT_packed_float bit layout: R 11-bit
// at 0..10 (5-bit exp + 6-bit mantissa), G 11-bit at 11..21, B 10-bit at 22..31 (5-bit exp + 5-bit
// mantissa). All three share f16's exponent bias (15) and unsigned semantics, so the encode collapses to
// a half-float pack followed by mantissa-bit drops: f11 = (f16 >> 4) & 0x7FF; f10 = (f16 >> 5) & 0x3FF.
// Inputs clamp to [0, 65024] (max f11 normal) so f16's wider 65504 range never produces an f11 inf via
// mantissa overflow. ~3% relative precision per channel — well below the visible threshold for additive
// HDR emission.

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

/** WGSL `unpackHdrColor(p: u32) -> vec3<f32>`: the HDR color read side (`rgb9e5ufloat`
 *  is read-only in WebGPU, so the pack is manual). */
export const hdrColorUnpackWgsl = chunk("hdrColorUnpackWgsl", [unpackHdrColor]);

/** WGSL `packHdrColor(rgb) -> u32`: the HDR color write side, split from {@link hdrColorUnpackWgsl}
 *  so a reader shader doesn't drag in the encode helpers. */
export const hdrColorPackWgsl = chunk("hdrColorPackWgsl", [packHdrColor]);

// smallest-3 quaternion: drop the largest component and store the other three as 10-bit snorm plus a
// 2-bit index, 16 B → 4 B, ~0.1° max error. Fine for narrowphase + per-pair neighbour reads (no
// compounding); NOT for iter-mutated state a downstream pass finite-differences.

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
 *  storage codec. The one codec here whose bodies stay WGSL text — smallest-3 dynamically indexes a
 *  vector (`q[largest]`) and switches on the result, neither of which TGSL expresses — so it is
 *  GPU-only, with no CPU twin to keep in step. */
export const smallest3Wgsl = chunk("smallest3Wgsl", [packQuatSmallest3, unpackQuatSmallest3]);

// The hot-path mirrors. Everything above is authored once as TGSL and callable on the CPU, but a
// typegpu CPU call costs ~270 ns of dispatch plus ~365 ns per vector it builds — ~5 µs for the oct
// encoder, against ~30 ns hand-written. That is the right price for a unit test and the wrong one for a
// per-vertex mesh bake or a GPU table upload.
//
// So the hot packers get a plain-JS mirror, over the same lattice primitives, and encode.test.ts
// pins each against the TGSL function it mirrors across a sweep — bit-identical on the packed word.
// That differential is the point: a hand-paired CPU/WGSL twin is held in step only by a comment, and
// these are held in step by a gate.

/** {@link octEncodeNormal} over three loose lanes — the mesh builder's per-vertex form. */
export function octEncode(x: number, y: number, z: number): number {
    // `Math.fround` marks every point the TGSL body builds a vector, and so stores f32 — the mirror
    // has to round where the schema would, or the two disagree on values near a lattice midpoint.
    const nx = Math.fround(x);
    const ny = Math.fround(y);
    const nz = Math.fround(z);
    const denom = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
    // `denom <= 0`, not `denom > 0`: the two differ on a NaN component, and the TGSL source's `select`
    // takes the reciprocal there (the NaN policy: compute through, don't wallpaper). Untestable
    // through the differential: a typegpu schema refuses a non-finite value, so only a shader (or this
    // mirror) ever sees one.
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
