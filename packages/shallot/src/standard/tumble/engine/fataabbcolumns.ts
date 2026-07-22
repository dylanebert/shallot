// The persistent fat-AABB region (kernel/src/fataabb.rs) — one enlarged broad-phase AABB per shape,
// held resident in the kernel's linear memory so the in-kernel recycle loop (4b) can test contact
// overlap without a per-step marshal. A second low persistent region, directly above the body region;
// keyed by shapeId (grow-on-createShape), so — unlike the body region — no record migration: a shape's
// slot is fixed for its life. This module owns the grow-only sizing to the shape high-water; the write
// (at refit) and the kernel read land in 4b.3.
//
// A region grow relocates the manifold + geometry regions above it (kernel-side, in place) and, like
// any memory.grow, detaches every typed-array view — so callers refresh the stores over the relocated
// regions after a grow (the same discipline reserveBodies follows).

import { kernel } from "./kernel";
import type { AABB } from "./math";
import type { Shape } from "./shape";
import type { WorldState } from "./world";

/** f32 stride of one shape's fat AABB in the column (lower.xyz + upper.xyz), mirroring `fataabb.rs`. */
const AABB_STRIDE = 6;

/** @returns the smallest power-of-two capacity ≥ `need`, at least 16 (amortizes region grows). */
function growCap(need: number): number {
    let cap = 16;
    while (cap < need) cap *= 2;
    return cap;
}

/**
 * Size the persistent fat-AABB region to hold `shapeCount` shapes (the shape high-water). Grows the
 * kernel region — relocating the manifold + geometry regions above it in place — only when the count
 * exceeds the current capacity. @returns true if the region grew (the caller must refresh any views
 * over the relocated regions above it).
 */
export function reserveFatAabb(shapeCount: number): boolean {
    return kernel().reserveFatAabb(growCap(shapeCount)) !== 0;
}

/**
 * A typed-array view over the resident fat-AABB column plus the shapeId-keyed write. One per world.
 * The column is the source the in-kernel recycle overlap test (4b.3c) and the in-kernel finalize refit's
 * escape test read; every TS site that writes `shape.fatAABB` mirrors it here inline (create, user
 * moves, the CCD/fast paths, the finalize commit). Re-derives its view whenever a grow detaches it.
 */
export class FatAabbStore {
    /** Resident fat-AABB column (`AABB_STRIDE` f32 per shape). Re-derived after every grow. */
    fatF = new Float32Array(0);

    /** Re-derive the column view over the current region. No-op before the first `reserveFatAabb`. */
    refreshViews(): void {
        const k = kernel();
        const cap = k.fatAabbCap();
        if (cap === 0) return;
        const buf = k.memory.buffer;
        const layout = new Uint32Array(buf, k.fatAabbLayoutPtr(), 1);
        this.fatF = new Float32Array(buf, layout[0], cap * AABB_STRIDE);
    }

    /** Write shape `shapeId`'s fat AABB into the column (lower.xyz + upper.xyz). */
    write(shapeId: number, fat: AABB): void {
        const f = this.fatF;
        const o = shapeId * AABB_STRIDE;
        f[o] = fat.lowerBound.x;
        f[o + 1] = fat.lowerBound.y;
        f[o + 2] = fat.lowerBound.z;
        f[o + 3] = fat.upperBound.x;
        f[o + 4] = fat.upperBound.y;
        f[o + 5] = fat.upperBound.z;
    }
}

/** Create an empty fat-AABB store for a new world. Its view is derived on the first write. */
export function createFatAabbStore(): FatAabbStore {
    return new FatAabbStore();
}

/**
 * Mirror a shape's just-written `shape.fatAABB` into the resident column, sizing the region to the shape
 * high-water first. The event-time write path (shape create, `Body_SetTransform`, `Body_SetType`) — not
 * a per-step hot loop, so it reserves + refreshes each call. A grow relocates the shape + manifold +
 * geometry regions above the fat-AABB region and detaches every view, so the stores that read through
 * them are refreshed before anything else runs. The per-step finalize/CCD commit sites skip this and
 * write `world.fatAabbStore` directly (the column is already sized, the view refreshed once at the top of
 * the pass).
 */
export function writeFatAabb(world: WorldState, shape: Shape): void {
    if (reserveFatAabb(world.shapes.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
        world.shapeStore.refreshViews();
    }
    world.fatAabbStore.refreshViews();
    world.fatAabbStore.write(shape.id, shape.fatAABB);
}
