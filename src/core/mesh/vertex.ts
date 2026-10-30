import tgpu from "typegpu";
import * as d from "typegpu/data";
import {
    bitcast,
    clamp,
    max,
    normalize,
    pack2x16snorm,
    pack2x16unorm,
    round,
    select,
    unpack2x16snorm,
    unpack2x16unorm,
} from "typegpu/std";
import { chunk } from "../../engine/utils";

const quantNs = tgpu["~unstable"].namespace({ names: "strict" });

/** the per-mesh position + uv AABB the quantized vertex stream dequantizes against: `posOffset` =
 *  (posMin.xyz, uvMin.x), `posScale` = (posExt.xyz, uvMin.y), `uvScale` = uvExt.xy. `streams` is the family's appended stream base in vec4u units. AABB-relative
 *  with no per-draw uniform, so it works unchanged in render bundles. */
export const MeshQuant = d.struct({
    posOffset: d.vec4f,
    posScale: d.vec4f,
    uvScale: d.vec2f,
    streams: d.u32,
});

/** the per-mesh dequant table selector: a quantized vertex's `w1` carries `meshId` in its high 16 bits. */
export const meshIdOf = tgpu.fn(
    [d.u32],
    d.u32,
)((w1) => {
    "use gpu";
    // JS's arithmetic shift would disagree with WGSL's logical u32 shift when the sign bit is set.
    // Both JS shift operators emit WGSL `>>`.
    return (w1 >>> 16) & 0xffff;
});

/** decode a quantized vertex position: `w0` = unorm16 pos.xy, `w1` = unorm16 pos.z | (meshId << 16),
 *  dequantized against the mesh's {@link MeshQuant} AABB. A degenerate axis (extent 0) has scale 0, so
 *  the decode returns the offset there. */
export const decodePos = tgpu.fn(
    [d.u32, d.u32, MeshQuant],
    d.vec3f,
)((w0, w1, q) => {
    "use gpu";
    const xy = unpack2x16unorm(w0);
    const z = d.f32(w1 & 0xffff) / 65535;
    return d.vec3f(
        q.posOffset.x + xy.x * q.posScale.x,
        q.posOffset.y + xy.y * q.posScale.y,
        q.posOffset.z + z * q.posScale.z,
    );
});

/** decode a quantized vertex uv (`w3` = unorm16 uv) against the mesh's uv AABB. */
export const decodeUv = tgpu.fn(
    [d.u32, MeshQuant],
    d.vec2f,
)((w3, q) => {
    "use gpu";
    const uv = unpack2x16unorm(w3);
    return d.vec2f(q.posOffset.w + uv.x * q.uvScale.x, q.posScale.w + uv.y * q.uvScale.y);
});

/** Exact f32 UV_1 words; unlike the base UV these are not AABB-quantized. */
export const decodeUv1 = tgpu.fn(
    [d.vec2u],
    d.vec2f,
)((words) => {
    "use gpu";
    return bitcast(d.vec2u, d.vec2f)(words);
});

/** RGBA on the unorm16 lattice, preserving glTF's unorm8 and unorm16 colors. */
export const decodeColor = tgpu.fn(
    [d.vec2u],
    d.vec4f,
)((words) => {
    "use gpu";
    return d.vec4f(unpack2x16unorm(words.x), unpack2x16unorm(words.y));
});

/** Bevy's octahedral tangent: signed oct x, oct y mapped positive with handedness in its sign. */
export const encodeTangent = tgpu.fn(
    [d.vec4f],
    d.u32,
)((v) => {
    "use gpu";
    const inv = 1 / (Math.abs(v.x) + Math.abs(v.y) + Math.abs(v.z));
    const p = d.vec2f(v.x * inv, v.y * inv);
    const folded = d.vec2f(
        (1 - Math.abs(p.y)) * select(-1, 1, p.x >= 0),
        (1 - Math.abs(p.x)) * select(-1, 1, p.y >= 0),
    );
    const oct = select(p, folded, v.z < 0);
    return pack2x16snorm(d.vec2f(oct.x, max(oct.y * 0.5 + 0.5, 1 / 32767) * v.w));
});

/** Decode before interpolation: octahedral coordinates cannot cross the rasterizer's seam. */
export const decodeTangent = tgpu.fn(
    [d.u32],
    d.vec4f,
)((word) => {
    "use gpu";
    const p = unpack2x16snorm(word);
    const f = d.vec2f(p.x, Math.abs(p.y) * 2 - 1);
    const z = 1 - Math.abs(f.x) - Math.abs(f.y);
    const t = clamp(-z, 0, 1);
    return d.vec4f(
        normalize(d.vec3f(f.x + select(t, -t, f.x >= 0), f.y + select(t, -t, f.y >= 0), z)),
        select(-1, 1, p.y >= 0),
    );
});

/** Loose-lane CPU mirror of encodeTangent, rounding vector construction to f32. */
export function packTangent(x: number, y: number, z: number, w: number): number {
    const f = Math.fround;
    x = f(x);
    y = f(y);
    z = f(z);
    w = f(w);
    const inv = 1 / (Math.abs(x) + Math.abs(y) + Math.abs(z));
    let px = f(x * inv);
    let py = f(y * inv);
    if (z < 0) {
        const ox = px;
        px = f((1 - Math.abs(py)) * (px >= 0 ? 1 : -1));
        py = f((1 - Math.abs(ox)) * (py >= 0 ? 1 : -1));
    }
    const snorm = (v: number) => Math.round(Math.max(-1, Math.min(1, f(v))) * 32767) & 0xffff;
    return (snorm(px) | (snorm(f(Math.max(py * 0.5 + 0.5, 1 / 32767) * w)) << 16)) >>> 0;
}

/** WGSL `MeshQuant` + `meshIdOf` / `decodePos` / `decodeUv`: the quantized-vertex decode.
 *  A vertex packs into a 16 B `vec4<u32>`: w0 = unorm16 pos.xy, w1 = unorm16 pos.z | (meshId << 16),
 *  w2 = oct normal, w3 = unorm16 uv. Splice into a vertex-pull shader; the encode half is
 *  {@link posQuantPackWgsl} (split so a decode-only reader doesn't drag in the producer's helpers). */
export const posQuantWgsl = chunk("posQuantWgsl", [meshIdOf, decodePos, decodeUv], quantNs);

/** encode a world/object-space position into the quantized vertex words: `.x` = unorm16 pos.xy,
 *  `.y` = unorm16 pos.z | (meshId << 16). A zero-extent axis selects normalized zero. */
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
    return d.vec2u(pack2x16unorm(d.vec2f(clamp(nx, 0, 1), clamp(ny, 0, 1))), z16 | (meshId << 16));
});

/** encode a uv into the quantized vertex's `w3` word, against the mesh's uv AABB. */
export const encodeUv = tgpu.fn(
    [d.vec2f, MeshQuant],
    d.u32,
)((uv, q) => {
    "use gpu";
    const nu = select((uv.x - q.posOffset.w) / q.uvScale.x, 0, q.uvScale.x === 0);
    const nv = select((uv.y - q.posScale.w) / q.uvScale.y, 0, q.uvScale.y === 0);
    return pack2x16unorm(d.vec2f(clamp(nu, 0, 1), clamp(nv, 0, 1)));
});

const packChunk = chunk("posQuantPackWgsl", [encodePos, encodeUv], quantNs);

/** WGSL `encodePos(p, meshId, q) -> vec2<u32>` + `encodeUv(uv, q) -> u32`: the {@link posQuantWgsl}
 *  encode half, for a GPU producer (compute-emitted terrain / meshing) that writes the quantized streams
 *  directly. Splice after {@link posQuantWgsl} — that chunk defines the shared `MeshQuant` struct. The
 *  producer supplies its mesh's analytic AABB as the `MeshQuant`. */
export function posQuantPackWgsl(): string {
    // MeshQuant lands in the first resolved chunk; resolve decode first to keep chunk text
    // independent of the caller's resolution order.
    posQuantWgsl();
    return packChunk();
}
