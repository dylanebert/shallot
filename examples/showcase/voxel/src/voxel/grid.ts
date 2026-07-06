// The voxel data substrate: an f32 DENSITY per cell on a 3D grid partitioned into fixed-size chunks, laid
// out in one linear buffer with chunk-major addressing. A cell is solid (meshed) where density ≥ ISO; below
// it is air. A scalar field (not a hard 0/1 flag) is what lets the carve brush *add weight* with a smooth
// falloff so the surface grows continuously across the threshold — the Astroneer / Planet-Coaster sculpt
// model on a blocky isosurface. Generated terrain + the canonical patterns write a clean 0 / DENSITY, so the
// threshold reproduces the binary occupancy the face-count oracle gates against. This is the CPU
// authoritative copy; the GPU mesher uploads `data` verbatim into one monolithic `Compute` storage buffer
// (read as `array<f32>`) and mirrors `index` in WGSL.
//
// The active (editable, resident) region is sized to fit ONE storage binding on the portable floor:
// `maxStorageBufferBindingSize` defaults to 128 MiB and the platform floor never assumes more, so a
// 64 MiB grid fits with margin on Apple / Steam Deck / integrated, not just the desktop ceiling. View
// distance beyond this region comes from a baked/LOD tier, decoupled from the voxel buffer.
//
// Chunk-major: each chunk's cells occupy one contiguous range, so a chunk uploads / evicts / remeshes as
// a single slice — what makes per-chunk streaming + dispatch a localized write. CHUNK and SLOTS are
// powers of two so the WGSL mirror bakes the same arithmetic as shift/mask from these constants.

export const CHUNK = 32;
const LOG2_CHUNK = 5;
const CHUNK_MASK = CHUNK - 1;
export const CHUNK_CELLS = CHUNK * CHUNK * CHUNK;

export const SLOTS = { x: 8, y: 8, z: 8 } as const;
export const DIM = { x: SLOTS.x * CHUNK, y: SLOTS.y * CHUNK, z: SLOTS.z * CHUNK } as const;

export const TOTAL_CELLS = SLOTS.x * SLOTS.y * SLOTS.z * CHUNK_CELLS;
export const BYTES = TOTAL_CELLS * 4;

/** the portable single-binding cap: maxStorageBufferBindingSize's spec default. The grid must fit here. */
export const BINDING_FLOOR = 128 * 1024 * 1024;

/** world units per voxel cell — the cell↔world mapping. The mesher emits cell `c` spanning world
 *  `[(c − DIM/2)·VOXEL, (c+1 − DIM/2)·VOXEL]`; the cursor pick (edit.ts) inverts it. One source so a scale
 *  change can't desync the rendered mesh from the carve ray. */
export const VOXEL = 1.0;

/** the full-solid density a generated / authored cell holds, and the isosurface threshold the mesher,
 *  oracle, and pick all test against. A cell is solid where density ≥ {@link ISO}. The brush blends partial
 *  values between 0 and DENSITY, so cells cross ISO gradually (the continuous-growth sculpt). */
export const DENSITY = 1.0;
export const ISO = 0.5;

/** cell coordinate → chunk slot (the chunk-major outer index). A chunk's cells occupy the contiguous range
 *  `[slot·CHUNK_CELLS, (slot+1)·CHUNK_CELLS)`, so a carve re-uploads exactly the touched chunk slices. */
export function chunkSlot(x: number, y: number, z: number): number {
    return ((z >> LOG2_CHUNK) * SLOTS.y + (y >> LOG2_CHUNK)) * SLOTS.x + (x >> LOG2_CHUNK);
}

/** cell coordinate → flat buffer offset. Chunk-major: `slot * CHUNK_CELLS + local`. */
export function index(x: number, y: number, z: number): number {
    const local = ((z & CHUNK_MASK) * CHUNK + (y & CHUNK_MASK)) * CHUNK + (x & CHUNK_MASK);
    return chunkSlot(x, y, z) * CHUNK_CELLS + local;
}

/** flat buffer offset → cell coordinate; the inverse of `index`. */
export function coord(offset: number): [number, number, number] {
    const slot = Math.floor(offset / CHUNK_CELLS);
    const local = offset % CHUNK_CELLS;
    const sx = slot % SLOTS.x;
    const sy = Math.floor(slot / SLOTS.x) % SLOTS.y;
    const sz = Math.floor(slot / (SLOTS.x * SLOTS.y));
    const lx = local & CHUNK_MASK;
    const ly = (local >> LOG2_CHUNK) & CHUNK_MASK;
    const lz = local >> (2 * LOG2_CHUNK);
    return [sx * CHUNK + lx, sy * CHUNK + ly, sz * CHUNK + lz];
}

/** WGSL mirror of {@link index} (the chunk-major cell→offset addressing) plus the DIM bounds constants.
 *  The mesher and the generator both bake this, so a generated grid and the mesh that reads it share one
 *  addressing source — divergence would mesh garbage. Declares `DIM_X/Y/Z` and `voxelIndex(x,y,z)`. */
export function addressingWgsl(): string {
    return /* wgsl */ `
const DIM_X: i32 = ${DIM.x};
const DIM_Y: i32 = ${DIM.y};
const DIM_Z: i32 = ${DIM.z};

fn voxelIndex(x: u32, y: u32, z: u32) -> u32 {
    let slot = ((z >> ${LOG2_CHUNK}u) * ${SLOTS.y}u + (y >> ${LOG2_CHUNK}u)) * ${SLOTS.x}u + (x >> ${LOG2_CHUNK}u);
    let local = ((z & ${CHUNK_MASK}u) * ${CHUNK}u + (y & ${CHUNK_MASK}u)) * ${CHUNK}u + (x & ${CHUNK_MASK}u);
    return slot * ${CHUNK_CELLS}u + local;
}
`;
}

export function get(data: Float32Array, x: number, y: number, z: number): number {
    return data[index(x, y, z)];
}

export function set(data: Float32Array, x: number, y: number, z: number, value: number): void {
    data[index(x, y, z)] = value;
}

/** solid = density ≥ {@link ISO} (the isosurface test). The grid stores a scalar weight, not a 0/1 flag. */
export function solid(data: Float32Array, x: number, y: number, z: number): boolean {
    return get(data, x, y, z) >= ISO;
}

/** bounds-checked solidity (the CPU twin of the WGSL `solidAt`): out-of-bounds reads as air, so a voxel
 *  on the grid edge emits a face outward. The unchecked {@link solid} trusts the caller's bounds. */
function solidAt(data: Float32Array, x: number, y: number, z: number): boolean {
    if (x < 0 || y < 0 || z < 0 || x >= DIM.x || y >= DIM.y || z >= DIM.z) return false;
    return solid(data, x, y, z);
}

/**
 * the analytic exposed-face count — the mesher-correctness oracle. A face exists iff exactly one of the
 * two voxels sharing it is solid, so each solid cell contributes one face per air (or out-of-bounds)
 * neighbour. This is the GPU mesher's invariant in CPU form: its atomic counter must match `faces(data)`
 * exactly for every grid (the watertight-seam gate — a doubled border face over-counts, a gap under-counts).
 */
export function faces(data: Float32Array): number {
    let n = 0;
    for (let z = 0; z < DIM.z; z++) {
        for (let y = 0; y < DIM.y; y++) {
            for (let x = 0; x < DIM.x; x++) {
                if (!solidAt(data, x, y, z)) continue;
                if (!solidAt(data, x + 1, y, z)) n++;
                if (!solidAt(data, x - 1, y, z)) n++;
                if (!solidAt(data, x, y + 1, z)) n++;
                if (!solidAt(data, x, y - 1, z)) n++;
                if (!solidAt(data, x, y, z + 1)) n++;
                if (!solidAt(data, x, y, z - 1)) n++;
            }
        }
    }
    return n;
}
