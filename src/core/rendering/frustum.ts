/** floats per packed frustum: 6 planes × vec4 */
export const FRUSTUM_FLOATS = 24;

/** Floats per cull slot: header vec4 (tag in x) followed by six frustum planes. */
export const CULL_VOLUME_FLOATS = 4 + FRUSTUM_FLOATS;

/** cull-volume tag (header `vec4`'s `.x`): a frustum, a 6-plane AND (every camera + shadow caster view) */
export const CULL_FRUSTUM = 0;

/**
 * Writes six inward frustum planes from column-major viewProj into `out` at `base`, as xyz normal/w offset.
 * Nondegenerate planes are normalized: dot(n,p)+w is signed world distance. Infinite perspective leaves
 * the far plane degenerate; {@link frustumVolume} supplies the finite cull bound.
 */
export function frustumPlanes(viewProj: Float32Array, out: Float32Array, base = 0): Float32Array {
    const m = viewProj;

    // WebGPU reverse-Z uses z for the far plane and w-z for near.
    out[base + 0] = m[3] + m[0];
    out[base + 1] = m[7] + m[4];
    out[base + 2] = m[11] + m[8];
    out[base + 3] = m[15] + m[12];

    out[base + 4] = m[3] - m[0];
    out[base + 5] = m[7] - m[4];
    out[base + 6] = m[11] - m[8];
    out[base + 7] = m[15] - m[12];

    out[base + 8] = m[3] + m[1];
    out[base + 9] = m[7] + m[5];
    out[base + 10] = m[11] + m[9];
    out[base + 11] = m[15] + m[13];

    out[base + 12] = m[3] - m[1];
    out[base + 13] = m[7] - m[5];
    out[base + 14] = m[11] - m[9];
    out[base + 15] = m[15] - m[13];

    out[base + 16] = m[2];
    out[base + 17] = m[6];
    out[base + 18] = m[10];
    out[base + 19] = m[14];

    out[base + 20] = m[3] - m[2];
    out[base + 21] = m[7] - m[6];
    out[base + 22] = m[11] - m[10];
    out[base + 23] = m[15] - m[14];

    for (let i = 0; i < 6; i++) {
        const o = base + i * 4;
        const len = Math.sqrt(out[o] * out[o] + out[o + 1] * out[o + 1] + out[o + 2] * out[o + 2]);
        if (len > 0) {
            out[o] /= len;
            out[o + 1] /= len;
            out[o + 2] /= len;
            out[o + 3] /= len;
        }
    }
    return out;
}

/**
 * Writes a cull slot to `out`; header.y is one for depth-only views, zero for shading views.
 * Supplying cameraWorld and far replaces an infinite perspective's degenerate far plane with a finite bound.
 */
export function frustumVolume(
    out: Float32Array,
    slot: number,
    viewProj: Float32Array,
    cameraWorld?: Float32Array,
    far?: number,
): void {
    const base = slot * CULL_VOLUME_FLOATS;
    out[base] = CULL_FRUSTUM; // header vec4: tag in .x
    frustumPlanes(viewProj, out, base + 4); // planes follow the header vec4
    if (!cameraWorld || far === undefined) return;

    const x = cameraWorld[8]!;
    const y = cameraWorld[9]!;
    const z = cameraWorld[10]!;
    const invLength = 1 / (Math.hypot(x, y, z) || 1);
    const nx = x * invLength;
    const ny = y * invLength;
    const nz = z * invLength;
    const plane = base + 4 + 16;
    out[plane] = nx;
    out[plane + 1] = ny;
    out[plane + 2] = nz;
    out[plane + 3] = far - nx * cameraWorld[12]! - ny * cameraWorld[13]! - nz * cameraWorld[14]!;
}
