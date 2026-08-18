import {
    ALBEDO_BYTES_PER_TEXEL,
    DIST_BYTES_PER_TEXEL,
    encodeDist,
    type Rect,
    TEXEL_SIZE,
    TILE_RES,
    texelOffset,
    tileOrigin,
} from "./tiles";

// The stage's hand-authored "known pattern": a single straight road-width stroke, standing in for the real
// stroke document a later stage authors (spec's Approach step 5 — "Road rasterizer" owns polylines/stamps
// and a compute rasterizer). Stage 4's job is the substrate (atlas + indirection + composite + dirty/
// redraw), so this is deliberately the simplest shape that exercises every layer of it: a pure
// distance-to-segment function (the same primitive the real rasterizer will differential-test against,
// `checks.md`'s two-independent-derivations shape for stage 5) plus a CPU-side packer that stands in for
// the TGSL compute rasterizer stage 5 builds.
//
// Run along +X through the world origin at Z=0 (grid.ts centres the terrain there, and `heightAt(0, 0)` is
// the one analytically known height on the seeded surface — an exact perlin-lattice point samples zero
// gradient contribution at every octave, so `GROUND_LEVEL + fbm2(0,0)*RELIEF === GROUND_LEVEL` regardless
// of seed — useful for a fixed-camera capture that needs a real world height with no device readback).
// HALF_WIDTH is one grid cell (grid.ts's SPACING) so an on-road/off-road probe pair can sit on exact grid
// vertices too (test/roads.spec.ts's capture arm reads their heights via `readVertices`).
export const STROKE_HALF_LENGTH = 150; // metres either side of the origin
export const STROKE_HALF_WIDTH = 4; // metres — matches terrain/grid.ts's SPACING
export const STROKE_Z = 0;

// dark asphalt, the same "not sampled from real materials, just a legible showcase color" spirit as
// terrain.ts's grass/rock/snow constants.
export const ROAD_ALBEDO: readonly [number, number, number] = [0.05, 0.05, 0.06];

/** signed distance in world metres from (x, z) to the stroke's road edge — negative inside the road,
 *  positive outside, zero at the boundary. The primitive stage 5's differential oracle will re-derive
 *  independently against a GPU rasterization of the same shape (`checks.md`'s two-derivations discipline);
 *  here it's the one source both the CPU packer and this stage's unit tests read. */
export function strokeDistance(x: number, z: number): number {
    const clampedX = Math.max(-STROKE_HALF_LENGTH, Math.min(STROKE_HALF_LENGTH, x));
    const dx = x - clampedX;
    const dz = z - STROKE_Z;
    const toSegment = Math.hypot(dx, dz);
    return toSegment - STROKE_HALF_WIDTH;
}

/** the stroke's world-space dirty bounds — its width/length AABB plus a one-texel margin so the
 *  fwidth-antialiased edge (terrain.ts's fs) never samples a texel outside the written region. Feeds
 *  `dirtyTiles` (tiles.ts). */
export function strokeRect(): Rect {
    const margin = TEXEL_SIZE;
    return {
        minX: -STROKE_HALF_LENGTH - STROKE_HALF_WIDTH - margin,
        maxX: STROKE_HALF_LENGTH + STROKE_HALF_WIDTH + margin,
        minZ: STROKE_Z - STROKE_HALF_WIDTH - margin,
        maxZ: STROKE_Z + STROKE_HALF_WIDTH + margin,
    };
}

/** pack one tile's albedo + distance content for the stroke pattern — the CPU write-path `atlas.ts` calls
 *  per dirty tile. Every texel gets a distance sample (so a tile far from the stroke still encodes "clearly
 *  outside"); albedo is only meaningful where distance is negative, but is written everywhere too (cheap,
 *  and the fs never reads it past the coverage the distance channel derives). */
export function packStrokeTile(tx: number, tz: number): { albedo: Uint8Array; dist: Uint8Array } {
    const [ox, oz] = tileOrigin(tx, tz);
    const albedo = new Uint8Array(TILE_RES * TILE_RES * ALBEDO_BYTES_PER_TEXEL);
    const dist = new Uint8Array(TILE_RES * TILE_RES * DIST_BYTES_PER_TEXEL);
    for (let y = 0; y < TILE_RES; y++) {
        const worldZ = oz + (y + 0.5) * TEXEL_SIZE;
        for (let x = 0; x < TILE_RES; x++) {
            const worldX = ox + (x + 0.5) * TEXEL_SIZE;
            const d = strokeDistance(worldX, worldZ);
            const distAt = texelOffset(x, y, DIST_BYTES_PER_TEXEL);
            dist[distAt] = encodeDist(d);
            const albedoAt = texelOffset(x, y, ALBEDO_BYTES_PER_TEXEL);
            albedo[albedoAt] = Math.round(ROAD_ALBEDO[0] * 255);
            albedo[albedoAt + 1] = Math.round(ROAD_ALBEDO[1] * 255);
            albedo[albedoAt + 2] = Math.round(ROAD_ALBEDO[2] * 255);
            albedo[albedoAt + 3] = 255;
        }
    }
    return { albedo, dist };
}

/** a world point guaranteed inside the stroke — the capture gate's on-road probe. Sits at the origin
 *  (analytically zero height, see the module header) with a lateral margin from either edge for the
 *  antialiased boundary. */
export const ON_ROAD_POINT: readonly [x: number, z: number] = [0, STROKE_Z];

/** a world point guaranteed outside the stroke, one grid cell (grid.ts SPACING) beyond the road edge —
 *  the capture gate's off-road probe, also grid-aligned for an exact `readVertices` height. */
export const OFF_ROAD_POINT: readonly [x: number, z: number] = [
    0,
    STROKE_Z + STROKE_HALF_WIDTH + 4,
];
