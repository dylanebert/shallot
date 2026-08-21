import * as d from "typegpu/data";

// The terrain grid's addressing + world scale — a fixed-topology XZ vertex grid, not voxel's 3D density
// field. A heightfield needs only one height sample per (x, z) column, so the mesh is a flat W×H quad
// grid displaced in Y, not a volumetric field marched into faces.
//
// Scale: 256×256 quads at 4 m spacing → a 1024 m × 1024 m footprint (~1.05 km²), matching the spec's
// "~1 km² showcase scale" (the voxel-example spirit: round numbers a reader can check, not a tuned
// figure). 4 m spacing keeps the vertex count small (257×257 = 66,049 verts, well under any storage-
// binding pressure) while staying fine enough to read the rolling hills at showcase camera distances —
// a road's own marking legibility is stage 8's problem (analytic fs evaluation, not this grid), not this
// stage's. CELLS must be even so the grid centres exactly on the world origin (HALF is a whole number of cells).

export const SPACING = 4; // world metres per grid cell
export const CELLS = 256; // quads per side
export const VERTS = CELLS + 1; // vertices per side (fence-post)
export const HALF = CELLS / 2; // cells from centre to edge

export const VERTEX_COUNT = VERTS * VERTS;
export const INDEX_COUNT = CELLS * CELLS * 6;

export const WORLD_EXTENT = CELLS * SPACING; // metres per side (1024)
export const WORLD_HALF = WORLD_EXTENT / 2;

export const GridVertices = d.arrayOf(d.vec4u, VERTEX_COUNT);
export const GridPosition = d.arrayOf(d.vec2u, VERTEX_COUNT);
export const GridIndices = d.arrayOf(d.u32, INDEX_COUNT);

/** grid column (ix, iz) → world (x, z), centred on the origin. */
export function worldX(ix: number): number {
    return (ix - HALF) * SPACING;
}
export function worldZ(iz: number): number {
    return (iz - HALF) * SPACING;
}

/** the inverse of {@link worldX}/{@link worldZ} — a grid-aligned world (x, z) → its exact column (ix, iz),
 *  used by the overlay capture gate (`capture.ts`) to read a probe point's real generated height off
 *  `readVertices()` rather than re-deriving the noise function in JS. `Math.round` rather than a plain
 *  divide since a caller passing an exact multiple of SPACING can still land a hair off an integer to
 *  floating-point error. */
export function gridX(x: number): number {
    return Math.round(x / SPACING) + HALF;
}
export function gridZ(z: number): number {
    return Math.round(z / SPACING) + HALF;
}

/** grid column (ix, iz) → flat vertex index, row-major over x within z. The GPU kernel and this CPU
 *  twin share the same formula (the mesh-topology oracle asserts they can't drift). */
export function vertexIndex(ix: number, iz: number): number {
    return iz * VERTS + ix;
}

/**
 * the CPU-built index buffer for the fixed W×H quad grid — pure function of {@link CELLS}, no dependency
 * on generated heights (unlike voxel's mesher, the triangle list here is data-independent, so it's
 * authored once on the CPU and uploaded once, never recomputed on the GPU).
 *
 * Winding: each quad (ix, iz) splits into (i0, i2, i1) and (i1, i2, i3) where i0..i3 are its four
 * corners in row-major order. For p0=(x,y,z), p1=(x+1,y,z), p2=(x,y,z+1): cross(p2-p0, p1-p0) = (0,1,0)
 * — that triangle order faces +Y (up), the convention sear expects for a front-facing ground plane.
 */
export function gridIndices(cells = CELLS): Uint32Array {
    const verts = cells + 1;
    const out = new Uint32Array(cells * cells * 6);
    let k = 0;
    for (let iz = 0; iz < cells; iz++) {
        for (let ix = 0; ix < cells; ix++) {
            const i0 = iz * verts + ix;
            const i1 = i0 + 1;
            const i2 = i0 + verts;
            const i3 = i2 + 1;
            out[k++] = i0;
            out[k++] = i2;
            out[k++] = i1;
            out[k++] = i1;
            out[k++] = i2;
            out[k++] = i3;
        }
    }
    return out;
}
