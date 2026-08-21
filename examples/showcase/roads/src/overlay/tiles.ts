import { WORLD_EXTENT, WORLD_HALF } from "../terrain/grid";

// The overlay's fixed world-space tile grid: pure addressing + packing math, no GPU/engine imports (the
// same device-free split terrain/grid.ts and terrain/noise.ts use) — `bun test` exercises every formula
// here without a device. `atlas.ts` is the GPU-resident half that consumes it.
//
// Tile size: 64 m (spec-given, matching Anno's own 512² tile precedent). 1024 m / 64 m = 16 tiles/side
// exactly (grid.ts's WORLD_EXTENT), so the tile grid tiles the terrain footprint with no partial edge tile
// — a round number a reader can check, not a fit. Atlas resolution: 512² texels/tile (Anno precedent,
// spec's Locked decision) — one row is 512 texels regardless of TILE_SIZE, so a texel is
// TILE_SIZE / TILE_RES = 0.125 m, the sub-texel-crisp unit the fwidth-thresholded coverage distance
// channel (terrain.ts's fs) reads against. The marking channel is no longer baked into a texel —
// stage 8 moved it to analytic fs evaluation from the chord uniform.

export const TILE_SIZE = 64; // world metres per tile side
export const TILE_RES = 512; // atlas texels per tile side (Anno's 512² slices)
export const TEXEL_SIZE = TILE_SIZE / TILE_RES; // world metres per atlas texel (0.125 m)

export const TILES_PER_SIDE = WORLD_EXTENT / TILE_SIZE; // 16
export const TILE_COUNT = TILES_PER_SIDE * TILES_PER_SIDE; // 256

// Atlas capacity: a layer is allocated the first time a tile is marked dirty and never evicted (the
// spec's Locked decision forbids eviction — this atlas is single-resolution with no coarse tier to fall
// back to, so an eviction miss mid-drag degrades to a hole). Sized by measurement (stage 4d): the
// worst-case single-document footprint under the capsule (swath) dirty-set test — a corner-to-corner
// diagonal chord at 45° across the bounded 1024 m world (~1437 m long, 8 m wide + 1-texel margin) —
// touches **46** tiles. Measured over a scan of orientations across the admissible domain (0–180° at
// 1° steps); the worst case is the 45° diagonal, not an axis-aligned chord.
//
// The null controls: an axis-aligned full-width chord reads exactly **32** (the AABB was already exact
// there, so the capsule test did not narrow it); the diagonal dropped from **256** (the whole grid under
// the old AABB) to **46** — the true swath. A straight chord crosses at most `2 × TILES_PER_SIDE − 1 =
// 31` tiles before width, so 46 is 1.48× the hand-derived bound (the extra tiles are the halfWidth +
// margin band on each side), not ~8× — the instrument is now measuring the artifact's real footprint,
// not the AABB's. Stage 4c's residue rule: a measurement ~8× the bound means the instrument is wrong,
// not that the buffer is small; 46 is well within the plausible range.
//
// `ATLAS_LAYERS = 64` gives ~39% headroom over the measured 46 (64 − 46 = 18 spare layers). This is also
// the original pre-4c value, so the ceiling deletion the spec predicted ("the plausible reading is
// under 64 and the ceiling deletes with no capacity change at all") is confirmed — the ceiling was
// hiding the AABB, and the capsule test brings the true footprint back under 64. Memory: 64 layers ×
// (512² texels × 4 B/texel rgba8unorm + 512² × 1 B/texel r8unorm) = 64 × 512² × 5 B = 83,886,080 B ≈ 80 MiB.
// Exceeding it is a fail-loud error (atlas.ts), not silent eviction.
export const THROTTLE = 8; // dirty-tile redraws (writeTexture calls) per frame
export const ATLAS_LAYERS = 64; // measured worst-case swath 46 + headroom (stage 4d)

export const ALBEDO_FORMAT = "rgba8unorm" as const;
export const ALBEDO_BYTES_PER_TEXEL = 4;
// the boundary-distance channel: a signed distance in world metres, quantized to r8unorm (one byte/texel —
// no float16 pack/unpack boundary to get wrong). DIST_RANGE clamps the encoded range to ±1 m: the
// fwidth-thresholded composite (terrain.ts) only ever reads distance within a few texels of its zero
// crossing (0.125 m/texel × a handful of texels), so saturating the unorm range at ±1 m spends every
// quantization step (2 m / 255 ≈ 0.0078 m ≈ 0.06 texel) where the antialiasing actually samples, instead of
// wasting range on interior distances no reader needs precisely. The marking channel no longer uses this
// codec — stage 8 moved markings to analytic fs evaluation.
export const DIST_FORMAT = "r8unorm" as const;
export const DIST_BYTES_PER_TEXEL = 1;
export const DIST_RANGE = 1; // metres, half-range

// Road markings — evaluated analytically in the fs from the chord uniform (endpoints + halfWidth),
// not baked into a texel (roads-interactive.md Locked decision, stage 8). The marking channel is a
// two-edge analytic pixel-coverage form, not a distance threshold — a feature narrower than a texel is
// a regime mismatch to bake, not a tuning problem. Dimensions from the MUTCD's normal line and
// broken-line pattern, in metres — a legibility standard, not a compliance claim.

// MUTCD normal line width: 4–6 in (0.1016–0.1524 m). Upper bound (6 in = 0.1524 m) — the lower bound was
// chosen to survive the 0.125 m texel the bake stored the marking in, and stage 8 removes that texel
// from the path entirely, so the parameter is free to take the standard's upper bound: 50 % more pixel
// coverage at every distance, still inside the standard.
export const LINE_WIDTH = 0.1524; // metres — MUTCD normal line, 6 in (0.1524 m), upper bound
export const LINE_HALF_WIDTH = LINE_WIDTH / 2; // derived: half the line width

// Edge line inset from the road edge — a showcase design choice, not a MUTCD dimension. The edge
// line's centre sits EDGE_INSET metres inside the road from the edge (d = −EDGE_INSET on the existing
// edge distance), so the line is fully on the asphalt with margin to the road boundary.
export const EDGE_INSET = 0.3; // metres — design choice, places the edge line inside the road

// MUTCD broken-line pattern: 10 ft segment, 30 ft gap. 1 ft = 0.3048 m.
export const DASH_SEGMENT = 3.048; // metres — MUTCD 10 ft segment (10 ft × 0.3048 m/ft)
export const DASH_GAP = 9.144; // metres — MUTCD 30 ft gap (30 ft × 0.3048 m/ft)
export const DASH_PERIOD = DASH_SEGMENT + DASH_GAP; // derived: one full dash cycle (segment + gap)
export const DASH_DUTY = DASH_SEGMENT / DASH_PERIOD; // derived: fraction of the period that is dash
// Dash phase offset — shifts the dash pattern along the chord so the road's midpoint (the capture gate's
// on-road probe point) falls in a gap, not on a dash. A design choice: the spec names the dash pattern
// (segment/gap ratio) but not where it starts, and one chord means no joint to break at. Offset = one
// DASH_SEGMENT shifts the phase by exactly DASH_DUTY, swapping dash and gap at the midpoint.
export const DASH_OFFSET = DASH_SEGMENT;

// Marking albedo — two-lane two-way road: solid white edges, broken yellow centreline.
export const EDGE_ALBEDO: readonly [number, number, number] = [0.85, 0.85, 0.85]; // white edge lines
export const CENTRE_ALBEDO: readonly [number, number, number] = [0.85, 0.75, 0.15]; // yellow centreline

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
