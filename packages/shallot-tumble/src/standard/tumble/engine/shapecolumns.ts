// The persistent shape region (kernel/src/shapes.rs) — one record per shapeId (type code, local
// geometry, nextShapeId), held resident in the kernel's linear memory so the in-kernel finalize refit
// can walk a body's shape list and compute its AABBs without a per-step marshal. A third low persistent
// region, above the fat-AABB region; keyed by shapeId (grow-on-createShape), so — like fat-AABB and
// unlike the body region — no record migration: a shape's slot is fixed for its life.
//
// The write sites are the shape lifecycle itself: `createShape` writes the whole record (a recycled
// shapeId inherits nothing), `destroyShape` patches the predecessor's `next` slot. There is no lazy
// dirty set — a stale record is invisible to TS and would silently feed the kernel garbage, so the
// column is written where the shape record is.
//
// A region grow relocates the manifold + geometry regions above it (kernel-side, in place) and, like
// any memory.grow, detaches every typed-array view — so callers refresh the stores over the relocated
// regions after a grow (the same discipline reserveBodies/reserveFatAabb follow).

import { NULL_INDEX } from "./array";
import type { Capsule, Sphere } from "./geometry";
import type { HullData } from "./hull";
import { kernel } from "./kernel";
import type { Shape } from "./shape";
import { ShapeType } from "./types";
import type { WorldState } from "./world";

/** 4-byte stride of one shape record, mirroring `shapes.rs`: type(1) next(1) geometry(7) refit(7). */
export const SHAPE_STRIDE = 16;
/** Shape type code — the `ShapeType` value verbatim (sphere/capsule/hull dispatch in-kernel; every
 * other value is the TS-fallback partition the kernel skips). */
export const S_TYPE = 0;
/** Next shape in the body's list, or `NULL_INDEX` (0xFFFFFFFF through the u32 view). */
export const S_NEXT = 1;
/** Local geometry the AABB compute needs: sphere center(3)+radius(1), capsule center1(3)+center2(3)+
 * radius(1), hull local-AABB lower(3)+upper(3). Unwritten for the fallback types. */
export const S_GEOM = 2;
/** Finalize-refit output the kernel writes per convex shape and TS reads in `finalizeBodies`: the
 * candidate fat AABB (`[lower.xyz, upper.xyz]`, 6 f32) then the escaped flag (u32, 0/1). */
export const S_CAND = 9;
export const S_ESCAPED = 15;

/** Which shape types the in-kernel finalize refit computes; the rest (mesh/height-field/compound) fall
 * back to the TS AABB path at their list position. Mirrors kernel `is_convex_refit` (`finalize.rs`). */
export function isConvexRefit(type: ShapeType): boolean {
    return type === ShapeType.Sphere || type === ShapeType.Capsule || type === ShapeType.Hull;
}

/** @returns the smallest power-of-two capacity ≥ `need`, at least 16 (amortizes region grows). */
function growCap(need: number): number {
    let cap = 16;
    while (cap < need) cap *= 2;
    return cap;
}

/**
 * Size the persistent shape region to hold `shapeCount` shapes (the shape high-water). Grows the kernel
 * region — relocating the manifold + geometry regions above it in place — only when the count exceeds
 * the current capacity. @returns true if the region grew (the caller must refresh any views over the
 * relocated regions above it, and over every region a `memory.grow` detached).
 */
export function reserveShapes(shapeCount: number): boolean {
    return kernel().reserveShapes(growCap(shapeCount)) !== 0;
}

/**
 * Typed-array views over the resident shape column plus the shapeId-keyed writes. One per world. The
 * column is what the in-kernel finalize refit reads; TS writes it at shape create/destroy. Re-derives
 * its views whenever a grow detaches or relocates them.
 */
export class ShapeStore {
    /** Resident shape column as u32 (type + nextShapeId). Re-derived after every grow. */
    shapeU = new Uint32Array(0);
    /** The same bytes as f32 — the geometry payload's natural type. */
    shapeF = new Float32Array(0);

    /** Re-derive the column views over the current region. No-op before the first `reserveShapes`. */
    refreshViews(): void {
        const k = kernel();
        const cap = k.shapeCap();
        if (cap === 0) return;
        const buf = k.memory.buffer;
        const layout = new Uint32Array(buf, k.shapeLayoutPtr(), 1);
        this.shapeU = new Uint32Array(buf, layout[0], cap * SHAPE_STRIDE);
        this.shapeF = new Float32Array(buf, layout[0], cap * SHAPE_STRIDE);
    }

    /** Write shape `shape.id`'s whole record — type, `nextShapeId`, geometry. Every slot is written
     * (the payload zeroed past the type's fields), so a recycled shapeId inherits nothing. */
    write(shape: Shape): void {
        const u = this.shapeU;
        const f = this.shapeF;
        const o = shape.id * SHAPE_STRIDE;
        u[o + S_TYPE] = shape.type;
        u[o + S_NEXT] = shape.nextShapeId;
        for (let i = S_GEOM; i < SHAPE_STRIDE; ++i) f[o + i] = 0;

        const g = o + S_GEOM;
        if (shape.type === ShapeType.Sphere) {
            const s = shape.sphere as Sphere;
            f[g] = s.center.x;
            f[g + 1] = s.center.y;
            f[g + 2] = s.center.z;
            f[g + 3] = s.radius;
        } else if (shape.type === ShapeType.Capsule) {
            const c = shape.capsule as Capsule;
            f[g] = c.center1.x;
            f[g + 1] = c.center1.y;
            f[g + 2] = c.center1.z;
            f[g + 3] = c.center2.x;
            f[g + 4] = c.center2.y;
            f[g + 5] = c.center2.z;
            f[g + 6] = c.radius;
        } else if (shape.type === ShapeType.Hull) {
            // The hull's local AABB is the whole hull-AABB path (`computeShapeAABBOut` transforms it);
            // the topology the narrowphase needs lives in the geometry pools, not here.
            const box = (shape.hull as HullData).aabb;
            f[g] = box.lowerBound.x;
            f[g + 1] = box.lowerBound.y;
            f[g + 2] = box.lowerBound.z;
            f[g + 3] = box.upperBound.x;
            f[g + 4] = box.upperBound.y;
            f[g + 5] = box.upperBound.z;
        }
    }

    /** Patch shape `shapeId`'s `next` slot after a shape-list unlink. */
    writeNext(shapeId: number, nextShapeId: number): void {
        this.shapeU[shapeId * SHAPE_STRIDE + S_NEXT] = nextShapeId;
    }
}

/** Create an empty shape store for a new world. Its views are derived on the first write. */
export function createShapeStore(): ShapeStore {
    return new ShapeStore();
}

/**
 * Write a newly created shape's record into the resident column, sizing the region to the new shape
 * high-water first. A grow relocates the manifold + geometry regions above the shape region and detaches
 * every view, so the stores that read through them are refreshed before anything else runs.
 */
export function writeShape(world: WorldState, shape: Shape): void {
    if (reserveShapes(world.shapes.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    world.shapeStore.refreshViews();
    world.shapeStore.write(shape);
}

/**
 * Patch the shape column after `shape` is unlinked from its body's list: its predecessor now points at
 * `shape.nextShapeId`. The destroyed shape's own record is left as-is — its id is freed, so nothing
 * reaches it, and a create that recycles the id rewrites every slot.
 */
export function unlinkShape(world: WorldState, shape: Shape): void {
    if (shape.prevShapeId === NULL_INDEX) return;
    world.shapeStore.refreshViews();
    world.shapeStore.writeNext(shape.prevShapeId, shape.nextShapeId);
}
