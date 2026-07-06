import { mesh } from "../render";

/**
 * register the built-in meshes. All unit-sized (radius / half-extent 0.5),
 * centered at the origin, so an entity's transform scale maps to world size:
 *
 * - `cube` — flat-shaded, per-face normals
 * - `sphere` / `capsule` — smooth-shaded; the rounded primitives physics will
 *   collide as a point/segment + radius
 */
export function initMeshes(): void {
    mesh({ name: "cube", ...cube() });
    mesh({ name: "sphere", ...sphere() });
    mesh({ name: "capsule", ...capsule() });
}

interface Vert {
    px: number;
    py: number;
    pz: number;
    nx: number;
    ny: number;
    nz: number;
    u: number;
    v: number;
}

// interleave per-vertex data into the shared vertex layout — eight floats per
// vertex: (posX posY posZ uvX) (normalX normalY normalZ uvY)
function pack(verts: Vert[]): Float32Array {
    const out = new Float32Array(verts.length * 8);
    for (let i = 0; i < verts.length; i++) {
        const w = verts[i];
        const o = i * 8;
        out[o] = w.px;
        out[o + 1] = w.py;
        out[o + 2] = w.pz;
        out[o + 3] = w.u;
        out[o + 4] = w.nx;
        out[o + 5] = w.ny;
        out[o + 6] = w.nz;
        out[o + 7] = w.v;
    }
    return out;
}

/**
 * unit cube (half-extent 0.5), flat-shaded — four vertices per face for
 * per-face normals, each face's uv running (0,0)→(1,0)→(1,1)→(0,1) over its
 * BL→BR→TR→TL corners. Winding is CCW outward, matching sear's back-face cull
 */
export function cube(): { vertices: Float32Array; indices: Uint32Array } {
    const uv = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ];
    // outward normal, then the BL, BR, TR, TL corners of each face
    const faces = [
        {
            normal: [0, 0, 1],
            corners: [
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
        },
        {
            normal: [0, 0, -1],
            corners: [
                [0.5, -0.5, -0.5],
                [-0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [0.5, 0.5, -0.5],
            ],
        },
        {
            normal: [0, 1, 0],
            corners: [
                [-0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
            ],
        },
        {
            normal: [0, -1, 0],
            corners: [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, -0.5, 0.5],
            ],
        },
        {
            normal: [1, 0, 0],
            corners: [
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.5, 0.5],
            ],
        },
        {
            normal: [-1, 0, 0],
            corners: [
                [-0.5, -0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
            ],
        },
    ];

    const verts: Vert[] = [];
    const indices: number[] = [];
    for (const { normal, corners } of faces) {
        const base = verts.length;
        for (let i = 0; i < 4; i++) {
            verts.push({
                px: corners[i][0],
                py: corners[i][1],
                pz: corners[i][2],
                nx: normal[0],
                ny: normal[1],
                nz: normal[2],
                u: uv[i][0],
                v: uv[i][1],
            });
        }
        indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }

    return { vertices: pack(verts), indices: new Uint32Array(indices) };
}

/**
 * UV sphere of radius 0.5, smooth-shaded (vertex normal = surface direction).
 * `segments` divisions around the axis, `rings` from pole to pole. Winding is
 * CCW outward, matching sear's back-face cull
 */
export function sphere(
    segments = 32,
    rings = 16,
): { vertices: Float32Array; indices: Uint32Array } {
    const verts: Vert[] = [];
    const indices: number[] = [];
    const radius = 0.5;

    for (let y = 0; y <= rings; y++) {
        const v = y / rings;
        const theta = v * Math.PI;
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            verts.push({ px: nx * radius, py: ny * radius, pz: nz * radius, nx, ny, nz, u, v });
        }
    }

    const stride = segments + 1;
    for (let y = 0; y < rings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = y * stride + x;
            const b = a + stride;
            indices.push(a, a + 1, b, a + 1, b + 1, b);
        }
    }

    return { vertices: pack(verts), indices: new Uint32Array(indices) };
}

/**
 * capsule of radius 0.5 and cylinder half-height 0.5 — two hemispherical caps
 * over a cylindrical mid-section, spanning y ∈ [-1, 1], smooth-shaded.
 * `segments` divisions around the axis, `rings` from cap pole to cap pole
 * (rounded down to an even count). The bottom cap winds the opposite direction
 * from the top because its rings run pole→equator (the top runs equator→pole),
 * keeping every triangle CCW outward
 */
export function capsule(
    segments = 32,
    rings = 16,
): { vertices: Float32Array; indices: Uint32Array } {
    const verts: Vert[] = [];
    const indices: number[] = [];
    const radius = 0.5;
    const halfHeight = 0.5;
    const halfRings = Math.floor(rings / 2);

    // top cap: pole (theta 0) down to the equator (theta PI/2), lifted +halfHeight
    for (let y = 0; y <= halfRings; y++) {
        const theta = (y / halfRings) * (Math.PI / 2);
        const v = (y / halfRings) * 0.25;
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            verts.push({
                px: nx * radius,
                py: ny * radius + halfHeight,
                pz: nz * radius,
                nx,
                ny,
                nz,
                u,
                v,
            });
        }
    }

    // cylinder seam rings — top then bottom, horizontal normals
    for (let x = 0; x <= segments; x++) {
        const u = x / segments;
        const phi = u * Math.PI * 2;
        const nx = Math.cos(phi);
        const nz = Math.sin(phi);
        verts.push({ px: nx * radius, py: halfHeight, pz: nz * radius, nx, ny: 0, nz, u, v: 0.25 });
    }
    for (let x = 0; x <= segments; x++) {
        const u = x / segments;
        const phi = u * Math.PI * 2;
        const nx = Math.cos(phi);
        const nz = Math.sin(phi);
        verts.push({
            px: nx * radius,
            py: -halfHeight,
            pz: nz * radius,
            nx,
            ny: 0,
            nz,
            u,
            v: 0.75,
        });
    }

    // bottom cap: pole (theta 0, pointing down) up to the equator (theta PI/2), dropped -halfHeight
    for (let y = 0; y <= halfRings; y++) {
        const theta = (y / halfRings) * (Math.PI / 2);
        const v = 0.75 + (y / halfRings) * 0.25;
        for (let x = 0; x <= segments; x++) {
            const u = x / segments;
            const phi = u * Math.PI * 2;
            const nx = Math.sin(theta) * Math.cos(phi);
            const ny = -Math.cos(theta);
            const nz = Math.sin(theta) * Math.sin(phi);
            verts.push({
                px: nx * radius,
                py: ny * radius - halfHeight,
                pz: nz * radius,
                nx,
                ny,
                nz,
                u,
                v,
            });
        }
    }

    const stride = segments + 1;
    for (let y = 0; y < halfRings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = y * stride + x;
            const b = a + stride;
            indices.push(a, a + 1, b, a + 1, b + 1, b);
        }
    }

    const cylTop = (halfRings + 1) * stride;
    const cylBot = cylTop + stride;
    for (let x = 0; x < segments; x++) {
        const a = cylTop + x;
        const b = cylBot + x;
        indices.push(a, a + 1, b, a + 1, b + 1, b);
    }

    const botStart = cylBot + stride;
    for (let y = 0; y < halfRings; y++) {
        for (let x = 0; x < segments; x++) {
            const a = botStart + y * stride + x;
            const b = a + stride;
            indices.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }

    return { vertices: pack(verts), indices: new Uint32Array(indices) };
}
