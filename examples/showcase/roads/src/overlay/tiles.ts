import { WORLD_EXTENT, WORLD_HALF } from "../terrain/grid";

// The overlay's fixed world-space tile grid: pure addressing + packing math, no GPU/engine imports (the
// same device-free split terrain/grid.ts and terrain/noise.ts use) — `bun test` exercises every formula
// here without a device (testing.md's logic tier). `atlas.ts` is the GPU-resident half that consumes it.
//
// Tile size: 64 m (spec-given, matching Anno's own 512² tile precedent). 1024 m / 64 m = 16 tiles/side
// exactly (grid.ts's WORLD_EXTENT), so the tile grid tiles the terrain footprint with no partial edge tile
// — a round number a reader can check, not a fit. Atlas resolution: 512² texels/tile (Anno precedent,
// spec's Locked decision) — one row is 512 texels regardless of TILE_SIZE, so a texel is
// TILE_SIZE / TILE_RES = 0.125 m, the sub-texel-crisp unit the fwidth-thresholded distance channel
// (terrain.ts's fs) reads against.

export const TILE_SIZE = 64; // world metres per tile side
export const TILE_RES = 512; // atlas texels per tile side (Anno's 512² slices)
export const TEXEL_SIZE = TILE_SIZE / TILE_RES; // world metres per atlas texel (0.125 m)

export const TILES_PER_SIDE = WORLD_EXTENT / TILE_SIZE; // 16
export const TILE_COUNT = TILES_PER_SIDE * TILES_PER_SIDE; // 256

// Atlas capacity: smaller than TILE_COUNT (the "small indirection" the spec's Approach names — every
// unwritten tile costs no atlas layer, only an indirection slot). A layer is allocated the first time a
// tile is marked dirty and never evicted (residency/eviction is the explicitly out-of-scope "tier above" —
// this world is small enough that the spec's own argument against full VT paging applies to overlay
// capacity too). Sized off the redraw throttle below: ATLAS_LAYERS = 8 × THROTTLE gives 8 fully-throttled
// frames of sustained new-tile allocation before capacity is reached — well past stage 6's "a handful of
// roads + one carpark" network, which (a few hundred metres of centreline over 64 m tiles) touches on the
// order of ten tiles. Exceeding it is a fail-loud error (atlas.ts), not silent eviction — a residue for a
// later stage if the network ever grows past this working set.
export const THROTTLE = 8; // dirty-tile redraws (writeTexture calls) per frame
export const ATLAS_LAYERS = 8 * THROTTLE; // 64

export const ALBEDO_FORMAT = "rgba8unorm" as const;
export const ALBEDO_BYTES_PER_TEXEL = 4;
// the boundary-distance channel: a signed distance in world metres, quantized to r8unorm (one byte/texel —
// no float16 pack/unpack boundary to get wrong). DIST_RANGE clamps the encoded range to ±1 m: the
// fwidth-thresholded composite (terrain.ts) only ever reads distance within a few texels of its zero
// crossing (0.125 m/texel × a handful of texels), so saturating the unorm range at ±1 m spends every
// quantization step (2 m / 255 ≈ 0.0078 m ≈ 0.06 texel) where the antialiasing actually samples, instead of
// wasting range on interior distances no reader needs precisely.
export const DIST_FORMAT = "r8unorm" as const;
export const DIST_BYTES_PER_TEXEL = 1;
export const DIST_RANGE = 1; // metres, half-range

// the fwidth-thresholded coverage band's half-width, in screen pixels: the fs's `fw = COVERAGE_BAND_PX *
// fwidth(dist)` coefficient (terrain.ts). `fwidth` is the change in its argument over one screen pixel, and
// coverage leaves [0, 1] exactly when `dist` moves by `fw` — so this coefficient *is* the band width the
// composite antialiases over, and the capture gate's pixel tolerance is a fixed multiple of it
// (capture.ts's TRANSITION_TOLERANCE_PX). Two readers, one value: the shader can't import a JS constant
// through TGSL's resolver, but it can be *interpolated* into it, which is what terrain.ts does.
export const COVERAGE_BAND_PX = 0.5;

/** encode a signed world-metre distance to its r8unorm byte — the CPU write-path's half of the codec
 *  `terrain.ts`'s fs decodes (`(sampled - 0.5) * 2 * DIST_RANGE`), read back verbatim by unorm sampling. */
export function encodeDist(metres: number): number {
    const clamped = Math.max(-DIST_RANGE, Math.min(DIST_RANGE, metres));
    const unit = clamped / (2 * DIST_RANGE) + 0.5; // → [0, 1]
    return Math.round(unit * 255);
}

/** the exact inverse of {@link encodeDist} — the CPU-side check reads a packed byte back through this
 *  rather than re-deriving encode's arithmetic, so the readback oracle exercises a real round trip. */
export function decodeDist(byte: number): number {
    return ((byte / 255 - 0.5) * 2 * DIST_RANGE) as number;
}

/** tile grid column (tx, tz) → flat tile id, row-major over tx within tz — the same convention
 *  `grid.ts`'s `vertexIndex` uses for the mesh grid, and the id the fs's indirection lookup and this
 *  module's `dirtyTiles` both address by. */
export function tileId(tx: number, tz: number): number {
    return tz * TILES_PER_SIDE + tx;
}

export function tileCoordOf(id: number): [tx: number, tz: number] {
    return [id % TILES_PER_SIDE, Math.floor(id / TILES_PER_SIDE)];
}

/** world (x, z) → the tile column that contains it, clamped to the grid (a point exactly on the far edge,
 *  `x === WORLD_HALF`, would otherwise floor to `TILES_PER_SIDE`, one past the last column). */
export function tileOf(x: number, z: number): [tx: number, tz: number] {
    const tx = Math.min(TILES_PER_SIDE - 1, Math.max(0, Math.floor((x + WORLD_HALF) / TILE_SIZE)));
    const tz = Math.min(TILES_PER_SIDE - 1, Math.max(0, Math.floor((z + WORLD_HALF) / TILE_SIZE)));
    return [tx, tz];
}

/** an axis-aligned world-space edit region — a stroke's dirty bounds, in world metres. */
export interface Rect {
    minX: number;
    minZ: number;
    maxX: number;
    maxZ: number;
}

/**
 * the analytic dirty-tile set for an edit rect: every tile column whose 64 m footprint overlaps `rect`,
 * clamped to the grid, sorted ascending by {@link tileId}. Pure — `atlas.ts`'s `markDirty` is a thin
 * GPU-side wrapper over this.
 *
 * @example dirtyTiles({ minX: -10, minZ: -10, maxX: 10, maxZ: 10 }) // the four tiles meeting at the origin
 */
export function dirtyTiles(rect: Rect): number[] {
    const [tx0, tz0] = tileOf(rect.minX, rect.minZ);
    const [tx1, tz1] = tileOf(rect.maxX, rect.maxZ);
    const loTx = Math.min(tx0, tx1);
    const hiTx = Math.max(tx0, tx1);
    const loTz = Math.min(tz0, tz1);
    const hiTz = Math.max(tz0, tz1);
    const out: number[] = [];
    for (let tz = loTz; tz <= hiTz; tz++) {
        for (let tx = loTx; tx <= hiTx; tx++) out.push(tileId(tx, tz));
    }
    return out;
}

/** a tile column's world-space origin (its minX/minZ corner) — the offset a texel-local (u, v) is added to. */
export function tileOrigin(tx: number, tz: number): [x: number, z: number] {
    return [tx * TILE_SIZE - WORLD_HALF, tz * TILE_SIZE - WORLD_HALF];
}

/** byte offset of texel (x, y)'s first channel within one tile's tightly packed row-major buffer — no
 *  WebGPU copy-alignment padding (that padding is `atlas.ts`'s `writeTexture` call's own concern, not the
 *  CPU pattern buffer's). */
export function texelOffset(x: number, y: number, bytesPerTexel: number): number {
    return (y * TILE_RES + x) * bytesPerTexel;
}
