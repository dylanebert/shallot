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

/** Transforms a local normal by R·S⁻¹ for non-uniform scale; zero-scale lanes return zero instead of NaN. */
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

/** World matrix with rotated, scaled basis columns; column i has length |scale[i]|. */
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

/** Splice at module scope; standard surfaces already include Xform/xform* and must not redefine them. */
export const xformWgsl = chunk("xformWgsl", [xformPoint, xformNormal, xformMat]);
