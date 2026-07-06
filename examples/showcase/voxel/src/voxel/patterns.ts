// The canonical scenario patterns: hand-authored deterministic grids whose exposed-face count is
// analytically known. They are the mesher-correctness gate — every distinct code path a face-cull mesher
// can take has a fixture here, and the mesher asserts its atomic face count against each. These functions
// only author the grid; the analytic occupancy each one satisfies is the spec the tests check.

import { CHUNK, coord, DENSITY, DIM, set } from "./grid";

const SOLID = DENSITY; // patterns write full density → the threshold reproduces clean binary occupancy

/** one solid voxel in air — the trivial base (6 faces). */
export function single(data: Float32Array, x: number, y: number, z: number): void {
    set(data, x, y, z, SOLID);
}

/** a fully-solid chunk — only the outer shell emits; the interior must occlude. */
export function solidChunk(data: Float32Array, sx: number, sy: number, sz: number): void {
    const x0 = sx * CHUNK;
    const y0 = sy * CHUNK;
    const z0 = sz * CHUNK;
    for (let z = z0; z < z0 + CHUNK; z++) {
        for (let y = y0; y < y0 + CHUNK; y++) {
            for (let x = x0; x < x0 + CHUNK; x++) {
                set(data, x, y, z, SOLID);
            }
        }
    }
}

/** alternating solid / air over a box — worst case, every solid voxel fully exposed. */
export function checker(
    data: Float32Array,
    x0: number,
    y0: number,
    z0: number,
    w: number,
    h: number,
    d: number,
): void {
    for (let z = 0; z < d; z++) {
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (((x + y + z) & 1) === 0) set(data, x0 + x, y0 + y, z0 + z, SOLID);
            }
        }
    }
}

/** a flat ground plane — solid below `height`, the common-case degenerate heightmap. */
export function slab(data: Float32Array, height: number): void {
    for (let z = 0; z < DIM.z; z++) {
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < DIM.x; x++) {
                set(data, x, y, z, SOLID);
            }
        }
    }
}

/** a solid cube with a 1-cell air channel bored along z through its center — the interior-face case a
 * heightmap never reaches (emits inward-facing quads). */
export function tunnel(data: Float32Array, x0: number, y0: number, z0: number, size: number): void {
    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                set(data, x0 + x, y0 + y, z0 + z, SOLID);
            }
        }
    }
    const c = size >> 1;
    for (let z = 0; z < size; z++) {
        set(data, x0 + c, y0 + c, z0 + z, 0);
    }
}

/** a solid ball; with r > CHUNK it spans several chunks — the curved-surface + watertight-seam fixture. */
export function sphere(data: Float32Array, cx: number, cy: number, cz: number, r: number): void {
    const r2 = r * r;
    const x0 = Math.max(0, Math.floor(cx - r));
    const x1 = Math.min(DIM.x - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r));
    const y1 = Math.min(DIM.y - 1, Math.ceil(cy + r));
    const z0 = Math.max(0, Math.floor(cz - r));
    const z1 = Math.min(DIM.z - 1, Math.ceil(cz + r));
    for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) {
                const dx = x - cx;
                const dy = y - cy;
                const dz = z - cz;
                if (dx * dx + dy * dy + dz * dz <= r2) set(data, x, y, z, SOLID);
            }
        }
    }
}

/** translate every solid cell so the content's bounding box centres in the grid. The patterns above are
 * authored at the placements the oracle tests pin (a corner, the floor, a chunk index), but the scenario
 * orbits the world origin — the grid centre — so an off-centre fixture reads flung into an octant. A rigid
 * translation leaves the exposed-face count unchanged (an out-of-bounds neighbour and an interior-air
 * neighbour both emit a face), so the mesher-correctness gate is unaffected: `faces()` recomputes from the
 * shifted grid and the GPU meshes the same. Returns the input untouched when already centred (the sphere)
 * or empty. */
export function recenter(data: Float32Array): Float32Array {
    let minX = DIM.x;
    let minY = DIM.y;
    let minZ = DIM.z;
    let maxX = -1;
    let maxY = -1;
    let maxZ = -1;
    for (let i = 0; i < data.length; i++) {
        if (data[i] === 0) continue;
        const [x, y, z] = coord(i);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }
    if (maxX < 0) return data; // empty grid — nothing to centre

    // shift so the bbox midpoint lands on the grid midpoint (DIM-1)/2
    const dx = Math.round((DIM.x - 1 - minX - maxX) / 2);
    const dy = Math.round((DIM.y - 1 - minY - maxY) / 2);
    const dz = Math.round((DIM.z - 1 - minZ - maxZ) / 2);
    if (dx === 0 && dy === 0 && dz === 0) return data;

    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
        if (data[i] === 0) continue;
        const [x, y, z] = coord(i);
        set(out, x + dx, y + dy, z + dz, data[i]);
    }
    return out;
}
