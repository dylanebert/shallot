/** floats per packed frustum: 6 planes × vec4 */
export const FRUSTUM_FLOATS = 24;

/**
 * floats per packed cull volume: a leading header `vec4` (the tag in `.x`) + the 6 frustum planes. Each
 * slot of `RenderContext.cullVolumes` is one of these; the producer pack indexes a slot at `slot *
 * CULL_VOLUME_FLOATS` and tests the planes after the header vec4. Every view culls by frustum: cameras,
 * the sun, and each point/spot shadow combo (its own frustum-culled depth view).
 */
export const CULL_VOLUME_FLOATS = 4 + FRUSTUM_FLOATS;

/** cull-volume tag (header `vec4`'s `.x`): a frustum, a 6-plane AND (every camera + shadow caster view) */
export const CULL_FRUSTUM = 0;

/**
 * extract the six clip-space frustum planes from a column-major `viewProj`,
 * packed as 6 × `vec4<f32>` (xyz = inward normal, w = offset) into `out` at
 * `base`. With reverse-Z in WebGPU's [0, 1] depth range, the far plane is the
 * bare z-row and the near plane is w−z. Infinite perspective makes the far row
 * degenerate; {@link frustumVolume} replaces it with the finite cull bound. Each
 * nondegenerate plane is normalized so `dot(n, p) + w` is a world-space signed
 * distance. A sphere of radius `r` is outside when that distance drops below `-r`.
 * Pure; the cull pass reads the packed planes per view, the producer pack tests
 * instance bounds against them
 */
export function frustumPlanes(viewProj: Float32Array, out: Float32Array, base = 0): Float32Array {
    const m = viewProj;

    // rows of the matrix recovered from column-major storage (m[col*4 + row]):
    // left = w + x, right = w - x, bottom = w + y, top = w - y,
    // far = z, near = w - z (reverse-Z)
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
 * pack a frustum cull volume into `out` at view `slot`: the tag word ({@link CULL_FRUSTUM}) in the header
 * vec4, then the camera's 6 planes ({@link frustumPlanes}). Infinite perspective projections replace the
 * degenerate far clip plane with the camera's finite culling bound when `cameraWorld` and `far` are supplied.
 * The view pack sets header.y to one for a
 * depth-only view and zero for a shading view. One source for the per-slot layout: the pack
 * reads the slot at `slot * CULL_VOLUME_FLOATS` and tests the planes after the header vec4.
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
