const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

/** constrain `value` to `[min, max]`: below `min` returns `min`, above `max` returns `max` */
export function clamp(value: number, min: number, max: number): number {
    return value < min ? min : value > max ? max : value;
}

/** linear interpolation from `a` to `b`, `t` in `[0, 1]` (not clamped; `t` outside the range extrapolates) */
export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

/** spherical interpolation between two quaternions, t in [0, 1] */
export function slerp(
    fromX: number,
    fromY: number,
    fromZ: number,
    fromW: number,
    toX: number,
    toY: number,
    toZ: number,
    toW: number,
    t: number,
): { x: number; y: number; z: number; w: number } {
    if (!Number.isFinite(t) || !Number.isFinite(fromW) || !Number.isFinite(toW)) {
        throw new Error(
            `slerp received NaN: from=[${fromX},${fromY},${fromZ},${fromW}], to=[${toX},${toY},${toZ},${toW}], t=${t}`,
        );
    }

    let dot = fromX * toX + fromY * toY + fromZ * toZ + fromW * toW;

    if (dot < 0) {
        dot = -dot;
        toX = -toX;
        toY = -toY;
        toZ = -toZ;
        toW = -toW;
    }

    let s0: number, s1: number;
    if (dot > 0.9995) {
        s0 = 1 - t;
        s1 = t;
    } else {
        const theta0 = Math.acos(dot);
        const sinTheta0 = Math.sqrt(1 - dot * dot);
        const theta = theta0 * t;
        const sinTheta = Math.sin(theta);
        s0 = Math.cos(theta) - (dot * sinTheta) / sinTheta0;
        s1 = sinTheta / sinTheta0;
    }

    return {
        x: s0 * fromX + s1 * toX,
        y: s0 * fromY + s1 * toY,
        z: s0 * fromZ + s1 * toZ,
        w: s0 * fromW + s1 * toW,
    };
}

/** quaternion from euler angles in degrees (XYZ order) */
export function quat(
    x: number,
    y: number,
    z: number,
): { x: number; y: number; z: number; w: number } {
    const hx = x * DEG_TO_RAD * 0.5;
    const hy = y * DEG_TO_RAD * 0.5;
    const hz = z * DEG_TO_RAD * 0.5;
    const cx = Math.cos(hx),
        sx = Math.sin(hx);
    const cy = Math.cos(hy),
        sy = Math.sin(hy);
    const cz = Math.cos(hz),
        sz = Math.sin(hz);

    return {
        x: sx * cy * cz + cx * sy * sz,
        y: cx * sy * cz - sx * cy * sz,
        z: cx * cy * sz + sx * sy * cz,
        w: cx * cy * cz - sx * sy * sz,
    };
}

/** euler angles in degrees (XYZ order) from a quaternion */
export function euler(
    x: number,
    y: number,
    z: number,
    w: number,
): { x: number; y: number; z: number } {
    const x2 = x + x,
        y2 = y + y,
        z2 = z + z;
    const xx = x * x2,
        xy = x * y2,
        xz = x * z2;
    const yy = y * y2,
        yz = y * z2,
        zz = z * z2;
    const wx = w * x2,
        wy = w * y2,
        wz = w * z2;

    const m13 = xz + wy;
    const ey = Math.asin(m13 < -1 ? -1 : m13 > 1 ? 1 : m13);

    if (m13 > -0.9999999 && m13 < 0.9999999) {
        return {
            x: Math.atan2(wx - yz, 1 - (xx + yy)) * RAD_TO_DEG,
            y: ey * RAD_TO_DEG,
            z: Math.atan2(wz - xy, 1 - (yy + zz)) * RAD_TO_DEG,
        };
    } else {
        return {
            x: Math.atan2(yz + wx, 1 - (xx + zz)) * RAD_TO_DEG,
            y: ey * RAD_TO_DEG,
            z: 0,
        };
    }
}

/** apply an euler delta (degrees, XYZ order) to a quaternion */
export function rotate(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    dx: number,
    dy: number,
    dz: number,
): { x: number; y: number; z: number; w: number } {
    const hx = dx * DEG_TO_RAD * 0.5;
    const hy = dy * DEG_TO_RAD * 0.5;
    const hz = dz * DEG_TO_RAD * 0.5;
    const cx = Math.cos(hx),
        sx = Math.sin(hx);
    const cy = Math.cos(hy),
        sy = Math.sin(hy);
    const cz = Math.cos(hz),
        sz = Math.sin(hz);

    const bx = sx * cy * cz + cx * sy * sz;
    const by = cx * sy * cz - sx * cy * sz;
    const bz = cx * cy * sz + sx * sy * cz;
    const bw = cx * cy * cz - sx * sy * sz;

    return {
        x: qw * bx + qx * bw + qy * bz - qz * by,
        y: qw * by - qx * bz + qy * bw + qz * bx,
        z: qw * bz + qx * by - qy * bx + qz * bw,
        w: qw * bw - qx * bx - qy * by - qz * bz,
    };
}

/** perspective projection mat4 (column-major), reverse-Z (near→1, far→0); fov in degrees */
export function perspective(
    fov: number,
    aspect: number,
    near: number,
    far: number,
    out?: Float32Array,
): Float32Array {
    if (fov <= 0) throw new Error(`Invalid FOV: ${fov} (must be > 0)`);
    if (aspect <= 0) throw new Error(`Invalid aspect ratio: ${aspect} (must be > 0)`);
    if (near === far) throw new Error(`Invalid depth planes: near === far (${near})`);
    if (!out) out = new Float32Array(16);
    const f = 1 / Math.tan((fov * Math.PI) / 360);
    const nf = 1 / (near - far);
    out[0] = f / aspect;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = f;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    // reverse-Z: the depth row is the standard mapping with near/far swapped, so near→1 and
    // far→0 (`out[11] = -1` keeps w_clip = z). Float depth + reverse-Z holds near-constant
    // relative precision across the range; forward-Z crowds it all at the far plane.
    out[10] = -near * nf;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = -near * far * nf;
    out[15] = 0;
    return out;
}

/** orthographic projection mat4 (column-major), reverse-Z (near→1, far→0); size is the half-height in world units */
export function orthographic(
    size: number,
    aspect: number,
    near: number,
    far: number,
    out?: Float32Array,
): Float32Array {
    if (size <= 0) throw new Error(`Invalid orthographic size: ${size} (must be > 0)`);
    if (aspect <= 0) throw new Error(`Invalid aspect ratio: ${aspect} (must be > 0)`);
    if (near === far) throw new Error(`Invalid depth planes: near === far (${near})`);
    if (!out) out = new Float32Array(16);
    const lr = 1 / (size * aspect);
    const bt = 1 / size;
    const nf = 1 / (near - far);
    out[0] = lr;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = bt;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    // reverse-Z: the depth row with near/far swapped, so near→1 and far→0 — the convention the
    // perspective path uses, kept uniform so the sun's ortho shadow map matches the receivers
    out[10] = -nf;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = -far * nf;
    out[15] = 1;
    return out;
}

/** column-major mat4 from translation (px, py, pz), quaternion (qx, qy, qz, qw), scale (sx, sy, sz) */
export function compose(
    px: number,
    py: number,
    pz: number,
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    sx: number,
    sy: number,
    sz: number,
    out?: Float32Array,
): Float32Array {
    if (!out) out = new Float32Array(16);
    const x2 = qx + qx;
    const y2 = qy + qy;
    const z2 = qz + qz;
    const xx = qx * x2;
    const xy = qx * y2;
    const xz = qx * z2;
    const yy = qy * y2;
    const yz = qy * z2;
    const zz = qz * z2;
    const wx = qw * x2;
    const wy = qw * y2;
    const wz = qw * z2;
    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}

/** mat4 × mat4, column-major */
export function multiply(a: Float32Array, b: Float32Array, out?: Float32Array): Float32Array {
    if (!out) out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
            out[j * 4 + i] =
                a[i] * b[j * 4] +
                a[i + 4] * b[j * 4 + 1] +
                a[i + 8] * b[j * 4 + 2] +
                a[i + 12] * b[j * 4 + 3];
        }
    }
    return out;
}

/** general mat4 inverse; returns the input unchanged when the matrix is singular */
export function invert(m: Float32Array, out?: Float32Array): Float32Array {
    if (!out) out = new Float32Array(16);

    const a00 = m[0],
        a01 = m[1],
        a02 = m[2],
        a03 = m[3];
    const a10 = m[4],
        a11 = m[5],
        a12 = m[6],
        a13 = m[7];
    const a20 = m[8],
        a21 = m[9],
        a22 = m[10],
        a23 = m[11];
    const a30 = m[12],
        a31 = m[13],
        a32 = m[14],
        a33 = m[15];

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (Math.abs(det) < 1e-10) {
        return out;
    }
    det = 1 / det;

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

    return out;
}

/** view matrix from eye looking at target */
export function lookAt(
    eyeX: number,
    eyeY: number,
    eyeZ: number,
    targetX: number,
    targetY: number,
    targetZ: number,
    upX = 0,
    upY = 1,
    upZ = 0,
    out?: Float32Array,
): Float32Array {
    let zx = eyeX - targetX;
    let zy = eyeY - targetY;
    let zz = eyeZ - targetZ;
    let zLen = Math.sqrt(zx * zx + zy * zy + zz * zz);

    if (zLen < 1e-6) {
        zx = 0;
        zy = 0;
        zz = 1;
    } else {
        zLen = 1 / zLen;
        zx *= zLen;
        zy *= zLen;
        zz *= zLen;
    }

    let xx = upY * zz - upZ * zy;
    let xy = upZ * zx - upX * zz;
    let xz = upX * zy - upY * zx;
    let xLen = Math.sqrt(xx * xx + xy * xy + xz * xz);

    if (xLen < 1e-6) {
        if (Math.abs(zy) > 0.9) {
            xx = 1;
            xy = 0;
            xz = 0;
        } else {
            xx = -zz;
            xy = 0;
            xz = zx;
        }
        xLen = Math.sqrt(xx * xx + xy * xy + xz * xz);
    }

    xLen = 1 / xLen;
    xx *= xLen;
    xy *= xLen;
    xz *= xLen;

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const tx = -(xx * eyeX + xy * eyeY + xz * eyeZ);
    const ty = -(yx * eyeX + yy * eyeY + yz * eyeZ);
    const tz = -(zx * eyeX + zy * eyeY + zz * eyeZ);

    if (!out) out = new Float32Array(16);
    out[0] = xx;
    out[1] = yx;
    out[2] = zx;
    out[3] = 0;
    out[4] = xy;
    out[5] = yy;
    out[6] = zy;
    out[7] = 0;
    out[8] = xz;
    out[9] = yz;
    out[10] = zz;
    out[11] = 0;
    out[12] = tx;
    out[13] = ty;
    out[14] = tz;
    out[15] = 1;
    return out;
}

/** rotation quaternion that points an object at eye toward target */
export function aim(
    eyeX: number,
    eyeY: number,
    eyeZ: number,
    targetX: number,
    targetY: number,
    targetZ: number,
    upX = 0,
    upY = 1,
    upZ = 0,
): { x: number; y: number; z: number; w: number } {
    if (
        !Number.isFinite(eyeX) ||
        !Number.isFinite(eyeY) ||
        !Number.isFinite(eyeZ) ||
        !Number.isFinite(targetX) ||
        !Number.isFinite(targetY) ||
        !Number.isFinite(targetZ)
    ) {
        throw new Error(
            `aim received NaN: eye=[${eyeX},${eyeY},${eyeZ}], target=[${targetX},${targetY},${targetZ}]`,
        );
    }

    let zx = eyeX - targetX;
    let zy = eyeY - targetY;
    let zz = eyeZ - targetZ;
    let zLen = Math.sqrt(zx * zx + zy * zy + zz * zz);

    if (zLen === 0) {
        zz = 1;
    } else {
        zLen = 1 / zLen;
        zx *= zLen;
        zy *= zLen;
        zz *= zLen;
    }

    let xx = upY * zz - upZ * zy;
    let xy = upZ * zx - upX * zz;
    let xz = upX * zy - upY * zx;
    let xLen = Math.sqrt(xx * xx + xy * xy + xz * xz);

    if (xLen < 1e-6) {
        if (Math.abs(zz) > Math.abs(zx)) {
            upX += 1e-4;
        } else {
            upZ += 1e-4;
        }
        xx = upY * zz - upZ * zy;
        xy = upZ * zx - upX * zz;
        xz = upX * zy - upY * zx;
        xLen = Math.sqrt(xx * xx + xy * xy + xz * xz);
    }

    if (xLen < 1e-6) {
        xx = 1;
        xy = 0;
        xz = 0;
    } else {
        xLen = 1 / xLen;
        xx *= xLen;
        xy *= xLen;
        xz *= xLen;
    }

    const yx = zy * xz - zz * xy;
    const yy = zz * xx - zx * xz;
    const yz = zx * xy - zy * xx;

    const trace = xx + yy + zz;
    let qw: number, qx: number, qy: number, qz: number;

    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        qw = 0.25 / s;
        qx = (yz - zy) * s;
        qy = (zx - xz) * s;
        qz = (xy - yx) * s;
    } else if (xx > yy && xx > zz) {
        const s = 2 * Math.sqrt(1 + xx - yy - zz);
        qw = (yz - zy) / s;
        qx = 0.25 * s;
        qy = (yx + xy) / s;
        qz = (zx + xz) / s;
    } else if (yy > zz) {
        const s = 2 * Math.sqrt(1 + yy - xx - zz);
        qw = (zx - xz) / s;
        qx = (yx + xy) / s;
        qy = 0.25 * s;
        qz = (yz + zy) / s;
    } else {
        const s = 2 * Math.sqrt(1 + zz - xx - yy);
        qw = (xy - yx) / s;
        qx = (zx + xz) / s;
        qy = (yz + zy) / s;
        qz = 0.25 * s;
    }

    return { x: qx, y: qy, z: qz, w: qw };
}
