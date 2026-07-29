import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
    clamp,
    cross,
    max,
    normalize,
    pack2x16float,
    pow,
    round,
    select,
    unpack2x16float,
    unpack4x8unorm,
} from "typegpu/std";
import {
    chunk,
    packSnorm2x16,
    packUnorm2x16,
    packUnorm4x8,
    snorm16,
    unorm8,
    unorm16,
    unpackSnorm2x16,
    unpackUnorm2x16,
} from "./tgsl";

// The GPU storage codecs (gpu.md rule 6), each a single TGSL function: one source that runs on the CPU
// (a `bun test` calls it directly) and resolves to the WGSL a shader splices. Lattice drift between a
// CPU packer and a GPU unpacker is the failure this shape makes unrepresentable — the 2026-05-08
// settled-stack torque was exactly that, a CPU oct encoder on a unorm16 lattice against a GPU decoder
// on snorm16.
//
// The `*Wgsl()` chunks below are the splice surface for a raw-WGSL shader, resolved with `names:
// "strict"` so the emitted function names are the ones documented here. Two rules govern them:
//
//   - A chunk is self-contained (its own resolve, its own dependencies) — EXCEPT two pairs that share a
//     dependency, and each shares a namespace so the dependency is emitted exactly once, into the half
//     named here. The position quantizers share `quantNs` (`MeshQuant`, emitted into posQuantWgsl());
//     the two snorm16 codecs share `snormNs` (the snorm pack/unpack leaves, emitted into
//     octEncodeWgsl()). Both pinned by the dependent half resolving its base half first, so the strings
//     are the same whatever order a consumer asks in — and the dependent half is spliced after its base.
//   - Splicing two chunks that share a dependency would otherwise define it twice, which is a WGSL
//     error. `encode.test.ts` asserts every pair of chunks stays duplicate-free.

const quantNs = tgpu["~unstable"].namespace({ names: "strict" });
const snormNs = tgpu["~unstable"].namespace({ names: "strict" });

/** the decomposed per-entity world transform the `transforms` firehose stores (48 B AoS: pos, quat,
 *  scale), reconstructed on read rather than stored as a matrix — the VS reads it scattered per
 *  instance, so AoS is one cache line per instance (gpu.md rule 1 / "Instance transforms"). */
export const Xform = d.struct({
    pos: d.vec3f,
    quat: d.vec4f,
    scale: d.vec3f,
});

/** the per-mesh position + uv AABB the quantized vertex stream dequantizes against: `posOffset` =
 *  (posMin.xyz, uvMin.x), `posScale` = (posExt.xyz, uvMin.y), `uvScale` = (uvExt.xy, _, _). AABB-relative
 *  with no per-draw uniform, so it works unchanged in render bundles. */
export const MeshQuant = d.struct({
    posOffset: d.vec4f,
    posScale: d.vec4f,
    uvScale: d.vec4f,
});

// snorm16 mapping (pack2x16snorm): (-1, 1) ↔ (-32767, 32767), with 0 ↔ 0 exact. The earlier unorm16
// mapping (-1, 1) ↔ (0, 65535) puts 0 between two integer rails, so axis-aligned vectors decoded as
// (0, ±1, 0) round-tripped to (0, 1, ±1.5e-5). For contact normals on a flat ground, that asymmetric
// z-bias produced a non-cancelling residual torque on the four corner contacts and a steady-state
// quaternion drift on settled boxes (validated 2026-05-08). snorm16 makes ±1 and 0 round-trip exactly.

/** octahedral-encode a unit normal to an snorm16x2 `u32` (the storage normal, 12 B → 4 B; gpu.md rule 6,
 *  Cigolle et al. 2014). **Never for an interpolated or filtered normal** — the octahedral seam breaks
 *  under interpolation (gpu.md rule 9); cross those as a plain `vec3` and renormalize.
 *  @example const w2 = octEncodeNormal(vec3f(0, 1, 0)); */
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
    return packSnorm2x16(select(p, folded, n.z < 0));
});

/** decode an snorm16x2 `u32` back to a unit normal: the inverse of {@link octEncodeNormal}. Cardinals
 *  round-trip exactly (0 ↔ 0, ±1 ↔ ±32767).
 *  @example const n = octDecodeNormal(w2); */
export const octDecodeNormal = tgpu.fn(
    [d.u32],
    d.vec3f,
)((enc) => {
    "use gpu";
    const p = unpackSnorm2x16(enc);
    const z = 1 - Math.abs(p.x) - Math.abs(p.y);
    const signX = select(d.f32(-1), d.f32(1), p.x >= 0);
    const signY = select(d.f32(-1), d.f32(1), p.y >= 0);
    const folded = d.vec3f((1 - Math.abs(p.y)) * signX, (1 - Math.abs(p.x)) * signY, z);
    return normalize(select(d.vec3f(p.x, p.y, z), folded, z < 0));
});

/** WGSL `octEncodeNormal(n) -> u32` + `octDecodeNormal(enc) -> vec3<f32>`: the snorm16x2 storage-normal
 *  codec. Splice into a producer that packs a normal or a reader that unpacks one; bit-identical to the
 *  CPU-callable {@link octEncodeNormal}, since it *is* that function. */
export const octEncodeWgsl = chunk("octEncodeWgsl", [octEncodeNormal, octDecodeNormal], snormNs);

// snorm16x4 quaternion: 4 components packed into 2 u32 via pack2x16snorm. Per-component error bound
// 1/32767 ≈ 3.05e-5 (uniform across [-1, 1]). Identity (0,0,0,1) and the six 180° axis-aligned rotations
// are bit-exact: 0 ↔ 0 and ±1 ↔ ±32767 sit on lattice rails. Decode renormalizes to absorb per-component
// quantization into a unit quat — downstream quatMul/quatRotate expect unit input. Cardinal exactness
// plus the per-component and worst-case angular bounds are validated in encode.test.ts.

/** pack a quaternion into two snorm16x2 `u32` lanes (xy, zw), 16 B → 8 B.
 *  @example const [lo, hi] = packQuatSnorm16x4(quat); */
export const packQuatSnorm16x4 = tgpu.fn(
    [d.vec4f],
    d.vec2u,
)((q) => {
    "use gpu";
    return d.vec2u(packSnorm2x16(d.vec2f(q.x, q.y)), packSnorm2x16(d.vec2f(q.z, q.w)));
});

/** unpack two snorm16x2 `u32` lanes back to a unit quaternion, renormalizing to absorb the per-component
 *  quantization — `quatMul` / `quatRotate` downstream expect unit input.
 *  @example const q = unpackQuatSnorm16x4(vec2u(lo, hi)); */
export const unpackQuatSnorm16x4 = tgpu.fn(
    [d.vec2u],
    d.vec4f,
)((p) => {
    "use gpu";
    const xy = unpackSnorm2x16(p.x);
    const zw = unpackSnorm2x16(p.y);
    return normalize(d.vec4f(xy.x, xy.y, zw.x, zw.y));
});

const quatChunk = chunk("quatSnorm16x4Wgsl", [packQuatSnorm16x4, unpackQuatSnorm16x4], snormNs);

/** WGSL `packQuatSnorm16x4(q) -> vec2<u32>` + `unpackQuatSnorm16x4(p) -> vec4<f32>`: the quaternion
 *  storage codec for a field whose precision feeds a finite-difference downstream (a body quat read
 *  back as angular velocity — gpu.md rule 6's iter-mutated-state case). Worst-case angular error
 *  ≤ ~0.01° per round-trip. Splice **after** {@link octEncodeWgsl} — the two share the snorm pack/unpack
 *  leaves, and that chunk defines them. */
export function quatSnorm16x4Wgsl(): string {
    // resolve the oct half first, unconditionally: the shared snorm leaves land in whichever chunk
    // resolves first, and pinning that to the oct half (the one every live splice already carries)
    // keeps both chunks the same text whatever order a consumer asks for them in.
    octEncodeWgsl();
    return quatChunk();
}

/** the per-mesh dequant table selector: a quantized vertex's `w1` carries `meshId` in its high 16 bits.
 *  @example let q = meshQuant[meshIdOf(w1)]; */
export const meshIdOf = tgpu.fn(
    [d.u32],
    d.u32,
)((w1) => {
    "use gpu";
    // masked, not a bare shift: WGSL's `>>` on u32 is logical and JS's is arithmetic, so the two arms
    // would disagree on a word with bit 31 set (`>>>` has no TGSL binding). The field is 16 bits wide.
    return (w1 >> 16) & 0xffff;
});

/** decode a quantized vertex position: `w0` = unorm16 pos.xy, `w1` = unorm16 pos.z | (meshId << 16),
 *  dequantized against the mesh's {@link MeshQuant} AABB. A degenerate axis (extent 0) has scale 0, so
 *  the decode returns the offset there.
 *  @example let pos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]); */
export const decodePos = tgpu.fn(
    [d.u32, d.u32, MeshQuant],
    d.vec3f,
)((w0, w1, q) => {
    "use gpu";
    const xy = unpackUnorm2x16(w0);
    const z = d.f32(w1 & 0xffff) / 65535;
    return d.vec3f(
        q.posOffset.x + xy.x * q.posScale.x,
        q.posOffset.y + xy.y * q.posScale.y,
        q.posOffset.z + z * q.posScale.z,
    );
});

/** decode a quantized vertex uv (`w3` = unorm16 uv) against the mesh's uv AABB.
 *  @example let uv = decodeUv(v.w, meshQuant[meshIdOf(v.y)]); */
export const decodeUv = tgpu.fn(
    [d.u32, MeshQuant],
    d.vec2f,
)((w3, q) => {
    "use gpu";
    const uv = unpackUnorm2x16(w3);
    return d.vec2f(q.posOffset.w + uv.x * q.uvScale.x, q.posScale.w + uv.y * q.uvScale.y);
});

/** WGSL `MeshQuant` + `meshIdOf` / `decodePos` / `decodeUv`: the quantized-vertex decode (gpu.md rule 6).
 *  A vertex packs into a 16 B `vec4<u32>`: w0 = unorm16 pos.xy, w1 = unorm16 pos.z | (meshId << 16),
 *  w2 = oct normal, w3 = unorm16 uv. Splice into a vertex-pull shader; the encode half is
 *  {@link posQuantPackWgsl} (split so a decode-only reader doesn't drag in the producer's helpers). */
export const posQuantWgsl = chunk("posQuantWgsl", [meshIdOf, decodePos, decodeUv], quantNs);

/** encode a world/object-space position into the quantized vertex words: `.x` = unorm16 pos.xy,
 *  `.y` = unorm16 pos.z | (meshId << 16). A zero-extent axis normalizes to 0, never a divide by zero.
 *  @example let w = encodePos(p, meshId, q); */
export const encodePos = tgpu.fn(
    [d.vec3f, d.u32, MeshQuant],
    d.vec2u,
)((p, meshId, q) => {
    "use gpu";
    const s = q.posScale;
    const o = q.posOffset;
    const nx = select((p.x - o.x) / s.x, 0, s.x === 0);
    const ny = select((p.y - o.y) / s.y, 0, s.y === 0);
    const nz = select((p.z - o.z) / s.z, 0, s.z === 0);
    const z16 = d.u32(round(clamp(nz, 0, 1) * 65535));
    return d.vec2u(packUnorm2x16(d.vec2f(clamp(nx, 0, 1), clamp(ny, 0, 1))), z16 | (meshId << 16));
});

/** encode a uv into the quantized vertex's `w3` word, against the mesh's uv AABB.
 *  @example let w3 = encodeUv(uv, q); */
export const encodeUv = tgpu.fn(
    [d.vec2f, MeshQuant],
    d.u32,
)((uv, q) => {
    "use gpu";
    const nu = select((uv.x - q.posOffset.w) / q.uvScale.x, 0, q.uvScale.x === 0);
    const nv = select((uv.y - q.posScale.w) / q.uvScale.y, 0, q.uvScale.y === 0);
    return packUnorm2x16(d.vec2f(clamp(nu, 0, 1), clamp(nv, 0, 1)));
});

/** WGSL `encodePos(p, meshId, q) -> vec2<u32>` + `encodeUv(uv, q) -> u32`: the {@link posQuantWgsl}
 *  encode half, for a GPU producer (compute-emitted terrain / meshing) that writes the quantized streams
 *  directly. Splice **after** {@link posQuantWgsl} — that chunk defines the shared `MeshQuant` struct. The
 *  producer supplies its mesh's analytic AABB as the `MeshQuant`. */
const packChunk = chunk("posQuantPackWgsl", [encodePos, encodeUv], quantNs);

export function posQuantPackWgsl(): string {
    // resolve the decode half first, unconditionally: the shared `MeshQuant` lands in whichever chunk
    // resolves first, and pinning that to the decode half keeps both chunks the same text whatever
    // order a consumer asks for them in.
    posQuantWgsl();
    return packChunk();
}

/** rotate `v` by the quaternion `q` (xyzw).
 *  @example let world = xformQuat(x.quat, local); */
export const xformQuat = tgpu.fn(
    [d.vec4f, d.vec3f],
    d.vec3f,
)((q, v) => {
    "use gpu";
    const qv = d.vec3f(q.x, q.y, q.z);
    const c = cross(qv, v);
    const t = d.vec3f(2 * c.x, 2 * c.y, 2 * c.z);
    const u = cross(qv, t);
    return d.vec3f(v.x + q.w * t.x + u.x, v.y + q.w * t.y + u.y, v.z + q.w * t.z + u.z);
});

/** apply T·R·S to a local-space point.
 *  @example let world = xformPoint(x, localPos); */
export const xformPoint = tgpu.fn(
    [Xform, d.vec3f],
    d.vec3f,
)((x, p) => {
    "use gpu";
    const r = xformQuat(x.quat, d.vec3f(p.x * x.scale.x, p.y * x.scale.y, p.z * x.scale.z));
    return d.vec3f(x.pos.x + r.x, x.pos.y + r.y, x.pos.z + r.z);
});

/** apply the inverse-transpose `R·S⁻¹` to a local-space normal — correct under non-uniform scale, where
 *  `(R·S)·n` tilts the normal wrong. A flattened (zero-scale) axis would divide 0/0 → NaN and poison every
 *  normal on the entity, so that lane drops to a finite 0 (a degenerate normal the oct codec tolerates).
 *  @example let n = xformNormal(x, localNormal); */
export const xformNormal = tgpu.fn(
    [Xform, d.vec3f],
    d.vec3f,
)((x, n) => {
    "use gpu";
    const inv = d.vec3f(
        select(n.x / x.scale.x, 0, x.scale.x === 0),
        select(n.y / x.scale.y, 0, x.scale.y === 0),
        select(n.z / x.scale.z, 0, x.scale.z === 0),
    );
    return xformQuat(x.quat, inv);
});

/** rebuild the world matrix, for a reader that needs columns or a matmul (the billboard / glyph
 *  surfaces). Bit-identical to the prior compose's matrix, so column-extracted scale is unchanged.
 *  @example let m = xformMat(x); */
export const xformMat = tgpu.fn(
    [Xform],
    d.mat4x4f,
)((x) => {
    "use gpu";
    const q = x.quat;
    const s = x.scale;
    const x2 = q.x + q.x;
    const y2 = q.y + q.y;
    const z2 = q.z + q.z;
    const xx = q.x * x2;
    const xy = q.x * y2;
    const xz = q.x * z2;
    const yy = q.y * y2;
    const yz = q.y * z2;
    const zz = q.z * z2;
    const wx = q.w * x2;
    const wy = q.w * y2;
    const wz = q.w * z2;
    return d.mat4x4f(
        d.vec4f((1 - yy - zz) * s.x, (xy + wz) * s.x, (xz - wy) * s.x, 0),
        d.vec4f((xy - wz) * s.y, (1 - xx - zz) * s.y, (yz + wx) * s.y, 0),
        d.vec4f((xz + wy) * s.z, (yz - wx) * s.z, (1 - xx - yy) * s.z, 0),
        d.vec4f(x.pos.x, x.pos.y, x.pos.z, 1),
    );
});

/** WGSL `Xform` + `xformQuat` / `xformPoint` / `xformNormal` / `xformMat`: the decomposed transform
 *  firehose's format and decode. Splice at module scope; sear splices it for every surface, so a surface
 *  preamble must not redefine `Xform` / `xform*`. */
export const xformWgsl = chunk("xformWgsl", [xformPoint, xformNormal, xformMat]);

// LDR colors store sRGB-encoded RGB + linear alpha as a `pack4x8unorm` u32. sRGB encoding gives uniform
// perceptual precision across the range and lets hex inputs (`Part.color = 0xRRGGBB`) round-trip exactly
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

/** decode an sRGB-packed LDR color (the `color` slab's GPU mirror) to linear rgb + linear alpha.
 *  @example let c = unpackLdrColor(color[eid]); */
export const unpackLdrColor = tgpu.fn(
    [d.u32],
    d.vec4f,
)((p) => {
    "use gpu";
    const v = unpack4x8unorm(p);
    return d.vec4f(srgbToLinear1(v.x), srgbToLinear1(v.y), srgbToLinear1(v.z), v.w);
});

/** sRGB-encode linear `rgb` + linear `alpha` into a `pack4x8unorm` u32 — the `color` slab's GPU mirror.
 *  @example const packed = packLdrColor(vec3f(1, 0, 0), 1); */
export const packLdrColor = tgpu.fn(
    [d.vec3f, d.f32],
    d.u32,
)((rgb, alpha) => {
    "use gpu";
    return packUnorm4x8(
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

/** decode an r11g11b10ufloat-packed HDR color.
 *  @example let rgb = unpackHdrColor(emissive[eid]); */
export const unpackHdrColor = tgpu.fn(
    [d.u32],
    d.vec3f,
)((p) => {
    "use gpu";
    const r11 = p & 0x7ff;
    const g11 = (p >> 11) & 0x7ff;
    const b10 = (p >> 22) & 0x3ff;
    const rgv = unpack2x16float((r11 << 4) | (g11 << 20));
    const bbv = unpack2x16float(b10 << 5);
    return d.vec3f(rgv.x, rgv.y, bbv.x);
});

/** encode an HDR color to r11g11b10ufloat, clamped to the format's [0, 65024] range.
 *  @example let packed = packHdrColor(rgb); */
export const packHdrColor = tgpu.fn(
    [d.vec3f],
    d.u32,
)((rgb) => {
    "use gpu";
    const c = d.vec3f(clamp(rgb.x, 0, 65024), clamp(rgb.y, 0, 65024), clamp(rgb.z, 0, 65024));
    const rg = pack2x16float(d.vec2f(c.x, c.y));
    const bb = pack2x16float(d.vec2f(c.z, 0));
    return ((rg >> 4) & 0x7ff) | (((rg >> 20) & 0x7ff) << 11) | (((bb >> 5) & 0x3ff) << 22);
});

/** WGSL `unpackHdrColor(p: u32) -> vec3<f32>`: the HDR color read side (gpu.md rule 6 — `rgb9e5ufloat`
 *  is read-only in WebGPU, so the pack is manual). */
export const hdrColorUnpackWgsl = chunk("hdrColorUnpackWgsl", [unpackHdrColor]);

/** WGSL `packHdrColor(rgb) -> u32`: the HDR color write side, split from {@link hdrColorUnpackWgsl}
 *  so a reader shader doesn't drag in the encode helpers. */
export const hdrColorPackWgsl = chunk("hdrColorPackWgsl", [packHdrColor]);

// smallest-3 quaternion: drop the largest component and store the other three as 10-bit snorm plus a
// 2-bit index, 16 B → 4 B, ~0.1° max error. Fine for narrowphase + per-pair neighbour reads (no
// compounding); NOT for iter-mutated state a downstream pass finite-differences (gpu.md rule 6).

/** pack a quaternion as smallest-3 (10-10-10-2 in one `u32`).
 *  @example let p = packQuatSmallest3(q); */
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

/** unpack a smallest-3 `u32` back to a unit quaternion, reconstructing the dropped component.
 *  @example let q = unpackQuatSmallest3(p); */
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
// per-vertex mesh bake or a per-entity slab flush (65k entities × 6.9 µs would be 0.45 s of frame).
//
// So the two hot packers get a plain-JS mirror, over the same lattice primitives, and encode.test.ts
// pins each against the TGSL function it mirrors across a sweep — bit-identical on the packed word.
// That differential is the point: the hand-paired CPU/WGSL twins this file used to carry were held in
// step by a comment, and these are held in step by a gate.

/** {@link octEncodeNormal} over three loose lanes — the mesh builder's per-vertex form.
 *  @example const w2 = octEncode(nx, ny, nz); */
export function octEncode(x: number, y: number, z: number): number {
    // `Math.fround` marks every point the TGSL body builds a vector, and so stores f32 — the mirror
    // has to round where the schema would, or the two disagree on values near a lattice midpoint.
    const nx = Math.fround(x);
    const ny = Math.fround(y);
    const nz = Math.fround(z);
    const denom = Math.abs(nx) + Math.abs(ny) + Math.abs(nz);
    // `denom <= 0`, not `denom > 0`: the two differ on a NaN component, and the TGSL source's `select`
    // takes the reciprocal there (gpu.md's NaN policy — compute through, don't wallpaper). Untestable
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

/** {@link packUnorm2x16} over two loose lanes — the mesh builder's per-vertex form.
 *  @example const w0 = packUnorm2(u, v); */
export function packUnorm2(x: number, y: number): number {
    return ((unorm16(y) << 16) | unorm16(x)) >>> 0;
}

function packSnorm2(x: number, y: number): number {
    return ((snorm16(y) << 16) | snorm16(x)) >>> 0;
}

/** {@link packLdrColor} over four loose lanes — the `color` slab mirror's per-entity form.
 *  @example const packed = packColor4(r, g, b, a); */
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
