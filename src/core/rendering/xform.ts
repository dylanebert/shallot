import tgpu from "typegpu";
import * as d from "typegpu/data";
import { cross, select } from "typegpu/std";
import { chunk } from "../../engine/utils";

/** Decomposed placement record (48 B AoS: pos, quat,
 *  scale), reconstructed on read rather than stored as a matrix — the VS reads it scattered per
 *  instance, so AoS is one cache line per instance. */
export const Xform = d.struct({
    pos: d.vec3f,
    quat: d.vec4f,
    scale: d.vec3f,
});

/** rotate `v` by the quaternion `q` (xyzw). */
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

/** apply T·R·S to a local-space point. */
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
 *  normal on the entity, so that lane drops to a finite 0 (a degenerate normal the oct codec tolerates). */
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
 *  surfaces). Columns 0–2 are the rotated basis axes times `scale`, so column i has length `|scale[i]|`. */
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
 *  firehose's format and decode. Splice at module scope; the standard renderer splices it for every surface, so a surface
 *  preamble must not redefine `Xform` / `xform*`. */
export const xformWgsl = chunk("xformWgsl", [xformPoint, xformNormal, xformMat]);
