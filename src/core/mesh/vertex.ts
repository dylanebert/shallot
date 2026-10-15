import tgpu from "typegpu";
import * as d from "typegpu/data";
import { clamp, pack2x16unorm, round, select, unpack2x16unorm } from "typegpu/std";
import { chunk } from "../../engine/utils";

const quantNs = tgpu["~unstable"].namespace({ names: "strict" });

/** the per-mesh position + uv AABB the quantized vertex stream dequantizes against: `posOffset` =
 *  (posMin.xyz, uvMin.x), `posScale` = (posExt.xyz, uvMin.y), `uvScale` = (uvExt.xy, _, _). AABB-relative
 *  with no per-draw uniform, so it works unchanged in render bundles. */
export const MeshQuant = d.struct({
    posOffset: d.vec4f,
    posScale: d.vec4f,
    uvScale: d.vec4f,
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
