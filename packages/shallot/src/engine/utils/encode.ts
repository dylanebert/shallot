// snorm16 mapping (pack2x16snorm): (-1, 1) ↔ (-32767, 32767), with 0 ↔ 0 exact.
// The earlier unorm16 mapping (-1, 1) ↔ (0, 65535) puts 0 between two integer
// rails, so axis-aligned vectors decoded as (0, ±1, 0) round-tripped to
// (0, 1, ±1.5e-5). For contact normals on a flat ground, that asymmetric
// z-bias produced a non-cancelling residual torque on the four corner contacts
// and a steady-state quaternion drift on settled boxes (validated 2026-05-08).
// Using snorm16 makes ±1 and 0 round-trip exactly.
//
// The WGSL string and the CPU `octEncodeNormal` below MUST stay bit-identical:
// the same triangle normal flows through CPU packing in `blas.ts` (initial
// build) and GPU packing in `refit.ts` (dynamic atlas copy), and downstream
// shaders unpack them with the same `unpack2x16snorm` lattice. Lattice drift
// between the two paths is exactly the failure mode that injected the
// 2026-05-08 settled-stack torque. One source, two emitters.

interface Vec3 {
    x: number;
    y: number;
    z: number;
}

function pack2x16snorm(x: number, y: number): number {
    const a = Math.round(Math.max(-1, Math.min(1, x)) * 32767) & 0xffff;
    const b = Math.round(Math.max(-1, Math.min(1, y)) * 32767) & 0xffff;
    return ((b << 16) | a) >>> 0;
}

function unpack2x16snorm(u: number): [number, number] {
    const lo = (u << 16) >> 16;
    const hi = u >> 16;
    return [Math.max(-1, lo / 32767), Math.max(-1, hi / 32767)];
}

/** pack two [0,1] lanes into a unorm16x2 `u32` (lane x → low 16 bits) — the CPU twin of WGSL
 *  `pack2x16unorm`, bit-identical so a quantized field round-trips. For object-space positions
 *  (gpu.md rule 6) the caller normalizes `(p - aabbMin) / extent` into [0,1] first, and
 *  {@link POS_QUANT_WGSL} `decodePos` reverses it; fixed-point over a bounded range beats f16
 *  (1.5e-5 × extent uniform vs f16's relative precision, coarse near the high end). */
export function pack2x16unorm(x: number, y: number): number {
    const a = Math.round(Math.max(0, Math.min(1, x)) * 65535) & 0xffff;
    const b = Math.round(Math.max(0, Math.min(1, y)) * 65535) & 0xffff;
    return ((b << 16) | a) >>> 0;
}

/** unpack a unorm16x2 `u32` to two [0,1] lanes (low 16 bits → x) — the CPU twin of WGSL `unpack2x16unorm`. */
export function unpack2x16unorm(u: number): [number, number] {
    return [(u & 0xffff) / 65535, (u >>> 16) / 65535];
}

// snorm16x4 quaternion: 4 components packed into 2 u32 via pack2x16snorm.
// Per-component error bound 1/32767 ≈ 3.05e-5 (uniform across [-1, 1]).
// Identity (0,0,0,1) and the six 180° axis-aligned rotations are bit-exact:
// 0 ↔ 0 and ±1 ↔ ±32767 sit on lattice rails. Decode renormalizes to absorb
// per-component quantization into a unit quat — downstream quatMul/quatRotate
// expect unit input. Cardinal exactness plus the per-component and worst-case
// angular bounds are validated in encode.test.ts.
export function packQuatSnorm16x4(x: number, y: number, z: number, w: number): [number, number] {
    return [pack2x16snorm(x, y), pack2x16snorm(z, w)];
}

export function unpackQuatSnorm16x4(lo: number, hi: number): [number, number, number, number] {
    const [x, y] = unpack2x16snorm(lo);
    const [z, w] = unpack2x16snorm(hi);
    const len = Math.hypot(x, y, z, w);
    return [x / len, y / len, z / len, w / len];
}

/** octahedral-encode a unit normal to a snorm16x2 `u32` (the storage normal, 12 B → 4 B; gpu.md rule 6).
 *  The CPU twin of {@link OCT_ENCODE_WGSL} `octEncodeNormal`, bit-identical (it inlines the same Cigolle
 *  2014 fold) so a normal packed CPU-side decodes the same on the GPU — cardinals round-trip exactly
 *  (0 ↔ 0, ±1 ↔ ±32767). */
export function octEncodeNormal(n: Vec3): number {
    const denom = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z);
    const inv = denom > 0 ? 1 / denom : 0;
    const px = n.x * inv;
    const py = n.y * inv;
    if (n.z < 0) {
        const sx = px >= 0 ? 1 : -1;
        const sy = py >= 0 ? 1 : -1;
        return pack2x16snorm((1 - Math.abs(py)) * sx, (1 - Math.abs(px)) * sy);
    }
    return pack2x16snorm(px, py);
}

/** decode a snorm16x2 `u32` back to a unit normal — the inverse of {@link octEncodeNormal}, bit-identical
 *  to {@link OCT_ENCODE_WGSL} `octDecodeNormal`, so a CPU decode matches the GPU's (e.g. an oct-packed spot
 *  cone axis read back to validate a shader). */
export function octDecodeNormal(enc: number): Vec3 {
    const [px, py] = unpack2x16snorm(enc);
    const z = 1 - Math.abs(px) - Math.abs(py);
    let nx: number;
    let ny: number;
    let nz: number;
    if (z < 0) {
        const sx = px >= 0 ? 1 : -1;
        const sy = py >= 0 ? 1 : -1;
        nx = (1 - Math.abs(py)) * sx;
        ny = (1 - Math.abs(px)) * sy;
        nz = z;
    } else {
        nx = px;
        ny = py;
        nz = z;
    }
    const len = Math.hypot(nx, ny, nz);
    return { x: nx / len, y: ny / len, z: nz / len };
}

/** WGSL `octEncodeNormal(n) -> u32` + `octDecodeNormal(enc) -> vec3<f32>` — the snorm16x2 storage-normal
 *  codec (gpu.md rule 6, Cigolle et al. 2014). Splice into a producer that packs a normal or a reader that
 *  unpacks one; bit-identical to the CPU {@link octEncodeNormal}. */
export const OCT_ENCODE_WGSL = /* wgsl */ `
fn octEncodeNormal(n: vec3<f32>) -> u32 {
    let denom = abs(n.x) + abs(n.y) + abs(n.z);
    let inv = select(1.0 / denom, 0.0, denom <= 0.0);
    let p = n.xy * inv;
    let signX = select(-1.0, 1.0, p.x >= 0.0);
    let signY = select(-1.0, 1.0, p.y >= 0.0);
    let folded = vec2<f32>((1.0 - abs(p.y)) * signX, (1.0 - abs(p.x)) * signY);
    let pq = select(p, folded, n.z < 0.0);
    return pack2x16snorm(pq);
}

fn octDecodeNormal(enc: u32) -> vec3<f32> {
    let p = unpack2x16snorm(enc);
    let z = 1.0 - abs(p.x) - abs(p.y);
    let signX = select(-1.0, 1.0, p.x >= 0.0);
    let signY = select(-1.0, 1.0, p.y >= 0.0);
    let folded = vec3<f32>((1.0 - abs(p.y)) * signX, (1.0 - abs(p.x)) * signY, z);
    let n = select(vec3<f32>(p.x, p.y, z), folded, z < 0.0);
    return normalize(n);
}
`;

/** WGSL `MeshQuant` + `decodePos` / `decodeUv` / `meshIdOf` — the quantized-vertex decode (gpu.md rule 6).
 *  A vertex packs into a 16 B `vec4<u32>`: w0 = unorm16 pos.xy, w1 = unorm16 pos.z | (meshId << 16),
 *  w2 = oct normal, w3 = unorm16 uv. `MeshQuant` is the per-mesh position + uv AABB the decode dequantizes
 *  against, selected by `meshIdOf` from a storage table — AABB-relative with no per-draw uniform, so it works
 *  unchanged in render bundles. A degenerate axis (extent 0) has scale 0, so decode returns the offset there.
 *  Splice into a vertex-pull shader; the encode half is {@link POS_QUANT_PACK_WGSL} (split so a decode-only
 *  reader doesn't drag in the producer's pack helpers). */
export const POS_QUANT_WGSL = /* wgsl */ `
struct MeshQuant {
    posOffset: vec4<f32>,  // posMin.xyz, uvMin.x
    posScale: vec4<f32>,   // posExt.xyz, uvMin.y
    uvScale: vec4<f32>,    // uvExt.xy, _, _
};

fn meshIdOf(w1: u32) -> u32 {
    return w1 >> 16u;
}

fn decodePos(w0: u32, w1: u32, q: MeshQuant) -> vec3<f32> {
    let xy = unpack2x16unorm(w0);
    let z = f32(w1 & 0xFFFFu) / 65535.0;
    return q.posOffset.xyz + vec3<f32>(xy.x, xy.y, z) * q.posScale.xyz;
}

fn decodeUv(w3: u32, q: MeshQuant) -> vec2<f32> {
    return vec2<f32>(q.posOffset.w, q.posScale.w) + unpack2x16unorm(w3) * q.uvScale.xy;
}
`;

/** WGSL `encodePos(p, meshId, q) -> vec2<u32>` + `encodeUv(uv, q) -> u32` — the {@link POS_QUANT_WGSL} encode
 *  half, for a GPU producer (compute-emitted terrain / meshing) that writes the quantized streams directly.
 *  Splice after POS_QUANT_WGSL (references `MeshQuant`) + OCT_ENCODE_WGSL; the producer supplies its mesh's
 *  analytic AABB as the `MeshQuant`. `select` guards a zero-extent axis so a flat producer never divides by zero. */
export const POS_QUANT_PACK_WGSL = /* wgsl */ `
fn encodePos(p: vec3<f32>, meshId: u32, q: MeshQuant) -> vec2<u32> {
    let n = select((p - q.posOffset.xyz) / q.posScale.xyz, vec3<f32>(0.0), q.posScale.xyz == vec3<f32>(0.0));
    let z16 = u32(round(clamp(n.z, 0.0, 1.0) * 65535.0));
    return vec2<u32>(pack2x16unorm(clamp(n.xy, vec2<f32>(0.0), vec2<f32>(1.0))), z16 | (meshId << 16u));
}

fn encodeUv(uv: vec2<f32>, q: MeshQuant) -> u32 {
    let uvOff = vec2<f32>(q.posOffset.w, q.posScale.w);
    let n = select((uv - uvOff) / q.uvScale.xy, vec2<f32>(0.0), q.uvScale.xy == vec2<f32>(0.0));
    return pack2x16unorm(clamp(n, vec2<f32>(0.0), vec2<f32>(1.0)));
}
`;

/** WGSL `Xform` + `xformPoint`/`xformNormal`/`xformMat`: the decomposed transform firehose's format and
 *  decode (the transform analogue of {@link POS_QUANT_WGSL} for the vertex firehose). The firehose stores a
 *  per-entity `{pos, quat, scale}` (48 B AoS) the compose passes write and every reader reconstructs on read
 *  (niagara's reconstruct-on-read: the VS reads it scattered-per-instance, so AoS is one cache line/instance,
 *  gpu.md rule 1). `xformPoint` applies T·R·S to a local point; `xformNormal` is the inverse-transpose
 *  `R·S⁻¹` (correct under non-uniform scale, where `(R·S)·n` tilts the normal wrong);
 *  `xformMat` rebuilds the world matrix for readers that need columns or a matmul (the billboard / glyph
 *  surfaces). `xformMat` is bit-identical to the prior compose's matrix, so column-extracted scale is
 *  unchanged. Splice at module scope; sear splices it for every surface, so a surface preamble must not
 *  redefine `Xform` / `xform*`. */
export const XFORM_WGSL = /* wgsl */ `
struct Xform {
    pos: vec3<f32>,
    quat: vec4<f32>,
    scale: vec3<f32>,
};

fn xformQuat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn xformPoint(x: Xform, p: vec3<f32>) -> vec3<f32> {
    return x.pos + xformQuat(x.quat, p * x.scale);
}

fn xformNormal(x: Xform, n: vec3<f32>) -> vec3<f32> {
    // a flattened (zero-scale) axis divides 0/0 → nan, which poisons every normal on the entity; the
    // select drops that lane to a finite 0 (a degenerate normal the oct codec tolerates)
    let inv = select(n / x.scale, vec3<f32>(0.0), x.scale == vec3<f32>(0.0));
    return xformQuat(x.quat, inv);
}

fn xformMat(x: Xform) -> mat4x4<f32> {
    let q = x.quat;
    let s = x.scale;
    let x2 = q.x + q.x; let y2 = q.y + q.y; let z2 = q.z + q.z;
    let xx = q.x * x2; let xy = q.x * y2; let xz = q.x * z2;
    let yy = q.y * y2; let yz = q.y * z2; let zz = q.z * z2;
    let wx = q.w * x2; let wy = q.w * y2; let wz = q.w * z2;
    return mat4x4<f32>(
        vec4<f32>((1.0 - yy - zz) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0.0),
        vec4<f32>((xy - wz) * s.y, (1.0 - xx - zz) * s.y, (yz + wx) * s.y, 0.0),
        vec4<f32>((xz + wy) * s.z, (yz - wx) * s.z, (1.0 - xx - yy) * s.z, 0.0),
        vec4<f32>(x.pos, 1.0),
    );
}
`;

// LDR colors store sRGB-encoded RGB + linear alpha as `pack4x8unorm` u32.
// sRGB encoding gives uniform perceptual precision across the range and lets
// hex inputs (`Part.color = 0xRRGGBB`) round-trip exactly through the byte form.
// Alpha is linear in [0,1]; sRGB transfer applies only to color channels.
/** WGSL `unpackLdrColor(p: u32) -> vec4<f32>`: decode an sRGB-packed LDR color (the `color` slab's
 *  mirror) to linear rgb + linear alpha. Splice into a surface that reads a `u32` color binding. */
export const LDR_COLOR_UNPACK_WGSL = /* wgsl */ `
fn srgbToLinear1(c: f32) -> f32 {
    return select(pow(max((c + 0.055) / 1.055, 0.0), 2.4), c / 12.92, c <= 0.04045);
}

fn unpackLdrColor(p: u32) -> vec4<f32> {
    let v = unpack4x8unorm(p);
    return vec4<f32>(srgbToLinear1(v.x), srgbToLinear1(v.y), srgbToLinear1(v.z), v.w);
}
`;

export const LDR_COLOR_PACK_WGSL = /* wgsl */ `
fn linearToSrgb1(c: f32) -> f32 {
    return select(1.055 * pow(max(c, 0.0), 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

fn packLdrColor(rgb: vec3<f32>, alpha: f32) -> u32 {
    let s = vec3<f32>(linearToSrgb1(rgb.x), linearToSrgb1(rgb.y), linearToSrgb1(rgb.z));
    return pack4x8unorm(vec4<f32>(s, alpha));
}
`;

/** pack four [0,1] lanes into a `pack4x8unorm` u32 (x → byte 0). The CPU twin of WGSL `pack4x8unorm`,
 *  for slab GPU-mirror packing — bit-identical to the intrinsic so a quantized field round-trips. */
export function pack4x8unorm(x: number, y: number, z: number, w: number): number {
    const b = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255) & 0xff;
    return (b(x) | (b(y) << 8) | (b(z) << 16) | (b(w) << 24)) >>> 0;
}

function linearToSrgb1(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * Math.max(c, 0) ** (1 / 2.4) - 0.055;
}

/** sRGB-encode linear `rgb` + linear `alpha` into a `pack4x8unorm` u32 — the CPU twin of WGSL
 *  {@link LDR_COLOR_PACK_WGSL} `packLdrColor`, for the `color` slab's GPU mirror and round-trip tests. */
export function packLdrColor(r: number, g: number, b: number, a: number): number {
    return pack4x8unorm(linearToSrgb1(r), linearToSrgb1(g), linearToSrgb1(b), a);
}

// HDR colors store as r11g11b10ufloat manual u32 pack — Khronos GL_EXT_packed_float bit
// layout: R 11-bit at 0..10 (5-bit exp + 6-bit mantissa), G 11-bit at 11..21, B 10-bit
// at 22..31 (5-bit exp + 5-bit mantissa). All three components share f16's exponent bias
// (15) and unsigned semantics, so the encode collapses to a half-float pack followed by
// mantissa-bit drops: f11 = (f16 >> 4) & 0x7FF; f10 = (f16 >> 5) & 0x3FF. Inputs are
// clamped to [0, 65024] (max f11 normal) so f16's wider 65504 range never produces an
// f11 inf via mantissa overflow. ~3% relative precision per channel — well below the
// visible threshold for additive HDR emission. Pack and unpack split so reader shaders
// don't drag in unused encode helpers.
export const HDR_COLOR_UNPACK_WGSL = /* wgsl */ `
fn unpackHdrColor(p: u32) -> vec3<f32> {
    let r11 = p & 0x7FFu;
    let g11 = (p >> 11u) & 0x7FFu;
    let b10 = (p >> 22u) & 0x3FFu;
    let rg = (r11 << 4u) | (g11 << 20u);
    let bb = b10 << 5u;
    let rgv = unpack2x16float(rg);
    let bbv = unpack2x16float(bb);
    return vec3<f32>(rgv.x, rgv.y, bbv.x);
}
`;

export const HDR_COLOR_PACK_WGSL = /* wgsl */ `
fn packHdrColor(rgb: vec3<f32>) -> u32 {
    let c = clamp(rgb, vec3<f32>(0.0), vec3<f32>(65024.0));
    let rg = pack2x16float(c.rg);
    let bb = pack2x16float(vec2<f32>(c.b, 0.0));
    let r11 = (rg >> 4u) & 0x7FFu;
    let g11 = (rg >> 20u) & 0x7FFu;
    let b10 = (bb >> 5u) & 0x3FFu;
    return r11 | (g11 << 11u) | (b10 << 22u);
}
`;

// snorm16x4 quat WGSL codec. Storage: 2 u32 lanes (xy, zw) via pack2x16snorm.
// Decode normalizes to absorb per-component quantization (≤ 3.05e-5) into a
// unit quat — required because quatMul / quatRotate expect unit input.
// Identity + axis-aligned 180° round-trip bit-exact (0 ↔ 0, ±1 ↔ ±32767).
// Worst-case angular error ≤ ~0.01° per round-trip (derived in encode.test.ts).
export const QUAT_SNORM16X4_WGSL = /* wgsl */ `
fn packQuatSnorm16x4(q: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16snorm(q.xy), pack2x16snorm(q.zw));
}

fn unpackQuatSnorm16x4(p: vec2<u32>) -> vec4<f32> {
    let xy = unpack2x16snorm(p.x);
    let zw = unpack2x16snorm(p.y);
    return normalize(vec4<f32>(xy.x, xy.y, zw.x, zw.y));
}
`;

export const SMALLEST3_WGSL = /* wgsl */ `
fn packQuatSmallest3(q: vec4<f32>) -> u32 {
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
}

fn unpackQuatSmallest3(p: u32) -> vec4<f32> {
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
}
`;
