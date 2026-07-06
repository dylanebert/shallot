// the executable spec for the sprite surface WGSL: each function mirrors one billboard variant's
// vertex math (Godot material.cpp formulas — full screen-aligned = inverse-view columns substituted
// for the model rotation with scale re-applied; Y-locked = the cross(up, toViewer) orthonormal form).
// `world` is a column-major mat4 (the transforms slab layout); `right`/`up` are the camera's
// normalized world-space basis from the View uniform. The unit tests here are the gate the WGSL
// port is validated against.

type Vec3 = [number, number, number];

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function columnScale(world: Float32Array, base: number): number {
    return Math.hypot(world[base], world[base + 1], world[base + 2]);
}

/** world-aligned: the plain instanced transform, no rotation substitution */
export function worldCorner(world: Float32Array, lx: number, ly: number): Vec3 {
    return [
        world[0] * lx + world[4] * ly + world[12],
        world[1] * lx + world[5] * ly + world[13],
        world[2] * lx + world[6] * ly + world[14],
    ];
}

/** screen-aligned: camera right/up replace the model rotation; per-axis scale re-applied */
export function screenCorner(
    world: Float32Array,
    right: Vec3,
    up: Vec3,
    lx: number,
    ly: number,
): Vec3 {
    const sx = lx * columnScale(world, 0);
    const sy = ly * columnScale(world, 4);
    return [
        world[12] + right[0] * sx + up[0] * sy,
        world[13] + right[1] * sx + up[1] * sy,
        world[14] + right[2] * sx + up[2] * sy,
    ];
}

/**
 * Y-locked: the quad stays upright, yawing its horizontal axes toward the viewer.
 * `cross(right, up)` is the camera's backward axis (toward the viewer); its xz projection is the
 * quad's facing. A straight-down camera collapses that projection, so the camera up's xz
 * projection (screen-up on the ground plane) takes over — the quad is edge-on there anyway, the
 * fallback only keeps the basis finite and stable.
 */
export function yLockedCorner(
    world: Float32Array,
    right: Vec3,
    up: Vec3,
    lx: number,
    ly: number,
): Vec3 {
    const v = cross(right, up);
    let fx = v[0];
    let fz = v[2];
    if (fx * fx + fz * fz < 1e-8) {
        fx = up[0];
        fz = up[2];
    }
    const inv = 1 / Math.hypot(fx, fz);
    fx *= inv;
    fz *= inv;
    // quad right = cross((0,1,0), facing) — orthonormal with world up by construction
    const rx = fz;
    const rz = -fx;
    const sx = lx * columnScale(world, 0);
    const sy = ly * columnScale(world, 4);
    return [world[12] + rx * sx, world[13] + sy, world[14] + rz * sx];
}
