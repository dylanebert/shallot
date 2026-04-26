export interface Vec3 {
    x: number;
    y: number;
    z: number;
}

export interface Ray {
    origin: Vec3;
    direction: Vec3;
}

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

const _rgb = { r: 0, g: 0, b: 0 };

export function srgbToLinear(c: number): number {
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function linearToSrgb(c: number): number {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
}

export function unpackColor(packed: number): { r: number; g: number; b: number } {
    _rgb.r = srgbToLinear(((packed >> 16) & 0xff) / 255);
    _rgb.g = srgbToLinear(((packed >> 8) & 0xff) / 255);
    _rgb.b = srgbToLinear((packed & 0xff) / 255);
    return _rgb;
}

const _dir: [number, number, number] = [0, 0, 0];

export function normalizeDirection(x: number, y: number, z: number): [number, number, number] {
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len < 0.0001) {
        _dir[0] = 0;
        _dir[1] = -1;
        _dir[2] = 0;
    } else {
        _dir[0] = x / len;
        _dir[1] = y / len;
        _dir[2] = z / len;
    }
    return _dir;
}

export function clamp(value: number, min: number, max: number): number {
    return value < min ? min : value > max ? max : value;
}

export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

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

export function eulerToQuaternion(
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

export function quaternionToEuler(
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
    out[10] = far * nf;
    out[11] = -1;
    out[12] = 0;
    out[13] = 0;
    out[14] = far * near * nf;
    out[15] = 0;
    return out;
}

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
    out[10] = nf;
    out[11] = 0;
    out[12] = 0;
    out[13] = 0;
    out[14] = near * nf;
    out[15] = 1;
    return out;
}

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

export function invert(m: Float32Array, out?: Float32Array): Float32Array {
    if (!out) out = new Float32Array(16);
    const r00 = m[0],
        r01 = m[1],
        r02 = m[2];
    const r10 = m[4],
        r11 = m[5],
        r12 = m[6];
    const r20 = m[8],
        r21 = m[9],
        r22 = m[10];
    const tx = m[12],
        ty = m[13],
        tz = m[14];

    out[0] = r00;
    out[1] = r10;
    out[2] = r20;
    out[3] = 0;
    out[4] = r01;
    out[5] = r11;
    out[6] = r21;
    out[7] = 0;
    out[8] = r02;
    out[9] = r12;
    out[10] = r22;
    out[11] = 0;
    out[12] = -(r00 * tx + r01 * ty + r02 * tz);
    out[13] = -(r10 * tx + r11 * ty + r12 * tz);
    out[14] = -(r20 * tx + r21 * ty + r22 * tz);
    out[15] = 1;

    return out;
}

export function extractFrustumPlanes(viewProj: Float32Array, out?: Float32Array): Float32Array {
    const planes = out ?? new Float32Array(24);
    const m = viewProj;

    planes[0] = m[3] + m[0];
    planes[1] = m[7] + m[4];
    planes[2] = m[11] + m[8];
    planes[3] = m[15] + m[12];

    planes[4] = m[3] - m[0];
    planes[5] = m[7] - m[4];
    planes[6] = m[11] - m[8];
    planes[7] = m[15] - m[12];

    planes[8] = m[3] + m[1];
    planes[9] = m[7] + m[5];
    planes[10] = m[11] + m[9];
    planes[11] = m[15] + m[13];

    planes[12] = m[3] - m[1];
    planes[13] = m[7] - m[5];
    planes[14] = m[11] - m[9];
    planes[15] = m[15] - m[13];

    planes[16] = m[2];
    planes[17] = m[6];
    planes[18] = m[10];
    planes[19] = m[14];

    planes[20] = m[3] - m[2];
    planes[21] = m[7] - m[6];
    planes[22] = m[11] - m[10];
    planes[23] = m[15] - m[14];
    for (let i = 0; i < 6; i++) {
        const len = Math.hypot(planes[i * 4], planes[i * 4 + 1], planes[i * 4 + 2]);
        if (len > 0) {
            planes[i * 4] /= len;
            planes[i * 4 + 1] /= len;
            planes[i * 4 + 2] /= len;
            planes[i * 4 + 3] /= len;
        }
    }
    return planes;
}

export function lookAtMatrix(
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

export function orthographicBounds(
    left: number,
    right: number,
    bottom: number,
    top: number,
    near: number,
    far: number,
    out?: Float32Array,
): Float32Array {
    if (!out) out = new Float32Array(16);
    const lr = 1 / (right - left);
    const bt = 1 / (top - bottom);
    const nf = 1 / (near - far);
    out[0] = 2 * lr;
    out[1] = 0;
    out[2] = 0;
    out[3] = 0;
    out[4] = 0;
    out[5] = 2 * bt;
    out[6] = 0;
    out[7] = 0;
    out[8] = 0;
    out[9] = 0;
    out[10] = nf;
    out[11] = 0;
    out[12] = -(right + left) * lr;
    out[13] = -(top + bottom) * bt;
    out[14] = near * nf;
    out[15] = 1;
    return out;
}

export function extractFrustumCorners(
    invViewProj: Float32Array,
    nearZ: number,
    farZ: number,
    out?: Float32Array,
): Float32Array {
    const corners = out ?? new Float32Array(24);
    const ndcCorners = [
        [-1, -1, nearZ],
        [1, -1, nearZ],
        [-1, 1, nearZ],
        [1, 1, nearZ],
        [-1, -1, farZ],
        [1, -1, farZ],
        [-1, 1, farZ],
        [1, 1, farZ],
    ];

    for (let i = 0; i < 8; i++) {
        const [nx, ny, nz] = ndcCorners[i];
        const m = invViewProj;

        const wx = m[0] * nx + m[4] * ny + m[8] * nz + m[12];
        const wy = m[1] * nx + m[5] * ny + m[9] * nz + m[13];
        const wz = m[2] * nx + m[6] * ny + m[10] * nz + m[14];
        const ww = m[3] * nx + m[7] * ny + m[11] * nz + m[15];

        corners[i * 3] = wx / ww;
        corners[i * 3 + 1] = wy / ww;
        corners[i * 3 + 2] = wz / ww;
    }

    return corners;
}

export function invertMatrix(m: Float32Array, out?: Float32Array): Float32Array {
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

export function testAABBFrustum(
    minX: number,
    minY: number,
    minZ: number,
    maxX: number,
    maxY: number,
    maxZ: number,
    planes: Float32Array,
): boolean {
    for (let i = 0; i < 6; i++) {
        const nx = planes[i * 4];
        const ny = planes[i * 4 + 1];
        const nz = planes[i * 4 + 2];
        const d = planes[i * 4 + 3];
        const px = nx >= 0 ? maxX : minX;
        const py = ny >= 0 ? maxY : minY;
        const pz = nz >= 0 ? maxZ : minZ;
        if (nx * px + ny * py + nz * pz + d < 0) return false;
    }
    return true;
}

export function testAABBSphere(
    minX: number,
    minY: number,
    minZ: number,
    maxX: number,
    maxY: number,
    maxZ: number,
    cx: number,
    cy: number,
    cz: number,
    radius: number,
): boolean {
    const dx = Math.max(minX - cx, 0, cx - maxX);
    const dy = Math.max(minY - cy, 0, cy - maxY);
    const dz = Math.max(minZ - cz, 0, cz - maxZ);
    return dx * dx + dy * dy + dz * dz <= radius * radius;
}

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
            `lookAt received NaN: eye=[${eyeX},${eyeY},${eyeZ}], target=[${targetX},${targetY},${targetZ}]`,
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
