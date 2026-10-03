import type { World } from "../../../engine";
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

import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import type { AABB, Transform } from "../common/math";
import { ShapeType, type SurfaceMaterial } from "../common/types";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HullData } from "../shapes/hull";
import type { Shape } from "../shapes/shape";
import { type Body, getBodySim } from "../world/body";
import type { WorldState } from "../world/world";
import { kernel } from "./kernel";

/** 4-byte stride of one shape record, mirroring `shapes.rs`: type(1) next(1) geometry(7) refit(7) attachment(2). */
export const SHAPE_STRIDE = 34;
/** Shape type code — the `ShapeType` value verbatim (sphere/capsule/hull dispatch in-kernel; every
 * other value is the TS-fallback partition the kernel skips). */
export const S_TYPE = 0;
/** Next shape in the body's list, or `NULL_INDEX` (0xFFFFFFFF through the u32 view). */
export const S_NEXT = 1;
/** Local geometry the AABB compute needs: sphere center(3)+radius(1), capsule center1(3)+center2(3)+
 * radius(1), hull local-AABB lower(3)+upper(3). Unwritten for the fallback types. */
export const S_GEOM = 2;
/** Hull record index or non-convex geometry word offset; capsule uses this lane for its radius. */
export const S_GEO_REFERENCE = 8;
/** Finalize-refit output the kernel writes per convex shape and TS reads in `finalizeBodies`: the
 * candidate fat AABB (`[lower.xyz, upper.xyz]`, 6 f32) then the escaped flag (u32, 0/1). */
export const S_CAND = 9;
export const S_ESCAPED = 15;
/** Kernel shape-record attachment lanes, outside finalize output. */
export const S_MATERIAL_HEAD = 16;
export const S_MATERIAL_COUNT = 17;

/** Kernel material record: friction, restitution, rolling, tangent xyz, u64 user id, color, link,
 * generation and alive. */
export const MATERIAL_STRIDE = 12;
const M_NEXT = 9;

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
export function reserveShapes(world: World | undefined, shapeCount: number): boolean {
    const cap = growCap(shapeCount);
    const fatGrew = kernel(world).reserveFatAabb(cap) !== 0;
    const shapeGrew = kernel(world).reserveShapes(cap) !== 0;
    return fatGrew || shapeGrew;
}

/** Allocate a world-local shape slot in the kernel pool. The shape record itself is authored below,
 * but index reuse, generation and validity are never decided by TypeScript. */
export function createShapeSlot(world: WorldState): number {
    if (reserveShapes(world.ecsState, world.shapes.length + 1)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    const id = kernel(world.ecsState).shapeCreate(world.worldId);
    world.shapeStore.refreshViews();
    return id;
}

export function destroyShapeSlot(world: WorldState, shapeId: number): void {
    kernel(world.ecsState).shapeDestroy(world.worldId, shapeId);
}

/**
 * Typed-array views over the resident shape column plus the shapeId-keyed writes. One per world. The
 * column is what the in-kernel finalize refit reads; TS writes it at shape create/destroy. Re-derives
 * its views whenever a grow detaches or relocates them.
 */
export class ShapeStore {
    readonly ecsState: World | undefined;

    private readonly _worldId: number;

    constructor(ecsState: World | undefined, worldId: number) {
        this.ecsState = ecsState;

        this._worldId = worldId;
    }

    /** Resident shape column as u32 (type + nextShapeId). Re-derived after every grow. */
    shapeU = new Uint32Array(0);
    /** The same bytes as f32 — the geometry payload's natural type. */
    shapeF = new Float32Array(0);
    /** Resident fat-AABB column owned by this shape store, not a second helper store. */
    fatF = new Float32Array(0);
    /** Kernel-owned material records for this world's shape slots. */
    materialU = new Uint32Array(0);
    materialF = new Float32Array(0);
    // The held layout header views are derived from.
    private _layout = new Uint32Array(0);
    private _fatLayout = new Uint32Array(0);
    private _materialLayout = new Uint32Array(0);

    /** Re-derive the column views over the current region. No-op before the first `reserveShapes`, and
     * when the buffer, offset and capacity are those the views were derived at. */
    refreshViews(): void {
        const k = kernel(this.ecsState);
        const cap = k.shapeCap();
        const fatCap = k.fatAabbCap();
        if (cap === 0 && fatCap === 0) return;
        const buf = k.memory.buffer;
        const ptr = k.shapeLayoutPtr();
        if (this._layout.buffer !== buf || this._layout.byteOffset !== ptr)
            this._layout = new Uint32Array(buf, ptr, 1);
        const fatPtr = k.fatAabbLayoutPtr();
        if (this._fatLayout.buffer !== buf || this._fatLayout.byteOffset !== fatPtr)
            this._fatLayout = new Uint32Array(buf, fatPtr, 1);
        const materialPtr = k.materialLayoutPtr();
        if (this._materialLayout.buffer !== buf || this._materialLayout.byteOffset !== materialPtr)
            this._materialLayout = new Uint32Array(buf, materialPtr, 1);
        const layout = this._layout;
        const fatLayout = this._fatLayout;
        const materialLayout = this._materialLayout;
        if (
            this.shapeU.buffer !== buf ||
            this.shapeU.byteOffset !== layout[0] + this._worldId * cap * SHAPE_STRIDE * 4 ||
            this.shapeU.length !== cap * SHAPE_STRIDE
        ) {
            const worldOffset = this._worldId * cap * SHAPE_STRIDE * 4;
            this.shapeU = new Uint32Array(buf, layout[0] + worldOffset, cap * SHAPE_STRIDE);
            this.shapeF = new Float32Array(buf, layout[0] + worldOffset, cap * SHAPE_STRIDE);
        }
        if (
            this.fatF.buffer !== buf ||
            this.fatF.byteOffset !== fatLayout[0] + this._worldId * fatCap * 6 * 4 ||
            this.fatF.length !== fatCap * 6
        ) {
            const worldOffset = this._worldId * fatCap * 6 * 4;
            this.fatF = new Float32Array(buf, fatLayout[0] + worldOffset, fatCap * 6);
        }
        const materialCap = k.materialCap();
        if (
            this.materialU.buffer !== buf ||
            this.materialU.byteOffset !==
                materialLayout[0] + this._worldId * materialCap * MATERIAL_STRIDE * 4 ||
            this.materialU.length !== materialCap * MATERIAL_STRIDE
        ) {
            const worldOffset = this._worldId * materialCap * MATERIAL_STRIDE * 4;
            this.materialU = new Uint32Array(
                buf,
                materialLayout[0] + worldOffset,
                materialCap * MATERIAL_STRIDE,
            );
            this.materialF = new Float32Array(
                buf,
                materialLayout[0] + worldOffset,
                materialCap * MATERIAL_STRIDE,
            );
        }
    }

    /** Write authored type, list link and geometry while preserving the kernel attachment and finalize
     * output lanes. A material list is published before this write on create/reuse. */
    write(world: WorldState, shape: Shape): void {
        const u = this.shapeU;
        const f = this.shapeF;
        const o = shape.id * SHAPE_STRIDE;
        const materialHead = u[o + S_MATERIAL_HEAD];
        const materialCount = u[o + S_MATERIAL_COUNT];
        u[o + S_TYPE] = shape.type;
        u[o + S_NEXT] = shape.nextShapeId;
        for (let i = S_GEOM; i < SHAPE_STRIDE; ++i) f[o + i] = 0;
        u[o + S_MATERIAL_HEAD] = materialHead;
        u[o + S_MATERIAL_COUNT] = materialCount;

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
        } else if (shape.mesh) {
            f[g] = shape.mesh.scale.x;
            f[g + 1] = shape.mesh.scale.y;
            f[g + 2] = shape.mesh.scale.z;
        }
        this.writeGeometryReference(world, shape);
        this.writeQueryProperties(shape);
        const body = world.bodies[shape.bodyId];
        this.writeQueryPose(shape.id, body, getBodySim(world, body).transform);
    }

    writeQueryProperties(shape: Shape): void {
        const n = shape.id * SHAPE_STRIDE;
        const u = this.shapeU;
        u[n + 25] = shape.filter.categoryHi;
        u[n + 26] = shape.filter.categoryLo;
        u[n + 27] = shape.filter.maskHi;
        u[n + 28] = shape.filter.maskLo;
        u[n + 29] = shape.bodyId;
        u[n + 30] = Number(shape.enableSensorEvents);
        u[n + 31] = shape.filter.groupIndex;
    }

    writeQueryPose(shapeId: number, body: Body, pose?: Transform): void {
        const n = shapeId * SHAPE_STRIDE;
        this.shapeU[n + 32] = body.setIndex === SetType.Awake ? body.localIndex + 1 : 0;
        if (body.setIndex === SetType.Awake) return;
        if (!pose) throw new Error("physics: a non-awake query shape requires its sleeping pose");
        const f = this.shapeF;
        f[n + 18] = pose.p.x;
        f[n + 19] = pose.p.y;
        f[n + 20] = pose.p.z;
        f[n + 21] = pose.q.v.x;
        f[n + 22] = pose.q.v.y;
        f[n + 23] = pose.q.v.z;
        f[n + 24] = pose.q.s;
    }

    /** Refresh a shape's pool reference without touching its material or finalize lanes. */
    writeGeometryReference(world: WorldState, shape: Shape): void {
        const o = shape.id * SHAPE_STRIDE + S_GEO_REFERENCE;
        if (shape.hull) this.shapeU[o] = shape.hull.geoIndex;
        else if (shape.mesh) this.shapeU[o] = world.meshDatabase.get(shape.mesh.data)!.geoIndex;
        else if (shape.heightField)
            this.shapeU[o] = world.heightFieldDatabase.get(shape.heightField)!.geoIndex;
        else if (shape.compound)
            this.shapeU[o] = world.compoundDatabase.get(shape.compound)!.geoIndex;
    }

    /** Patch shape `shapeId`'s `next` slot after a shape-list unlink. */
    writeNext(shapeId: number, nextShapeId: number): void {
        this.shapeU[shapeId * SHAPE_STRIDE + S_NEXT] = nextShapeId;
    }

    /** Upload authored materials into kernel-owned records and attach their linked list to a shape. */
    writeMaterials(world: WorldState, shape: Shape, materials: SurfaceMaterial[]): void {
        this.refreshViews();
        let head = -1;
        for (let i = materials.length - 1; i >= 0; --i) {
            const id = kernel(world.ecsState).materialCreate(world.worldId);
            world.manifoldStore.refreshViews();
            world.bodyStore.refreshViews();
            this.refreshViews();
            const o = id * MATERIAL_STRIDE;
            const m = materials[i];
            this.materialF[o] = m.friction;
            this.materialF[o + 1] = m.restitution;
            this.materialF[o + 2] = m.rollingResistance;
            this.materialF[o + 3] = m.tangentVelocity.x;
            this.materialF[o + 4] = m.tangentVelocity.y;
            this.materialF[o + 5] = m.tangentVelocity.z;
            const bits = BigInt.asUintN(64, m.userMaterialId);
            this.materialU[o + 6] = Number(bits & 0xffffffffn);
            this.materialU[o + 7] = Number((bits >> 32n) & 0xffffffffn);
            this.materialU[o + 8] = m.customColor;
            this.materialU[o + M_NEXT] = head < 0 ? 0xffffffff : head;
            head = id;
        }
        // Publish the completed list once. The shape record, not Shape, owns this attachment.
        const o = shape.id * SHAPE_STRIDE;
        this.shapeU[o + S_MATERIAL_HEAD] = head < 0 ? 0xffffffff : head;
        this.shapeU[o + S_MATERIAL_COUNT] = materials.length;
    }

    /** Detach and release the kernel material records owned by a shape. */
    destroyMaterials(world: WorldState, shape: Shape): void {
        this.refreshViews();
        const k = kernel(world.ecsState);
        const head = k.shapeMaterialHead(world.worldId, shape.id) >>> 0;
        const count = k.shapeMaterialCount(world.worldId, shape.id) >>> 0;
        const listCount = k.materialListCount(world.worldId, head) >>> 0;
        if (listCount !== count) {
            throw new Error(`physics: material attachment mismatch on shape ${shape.id}`);
        }
        const o = shape.id * SHAPE_STRIDE;
        // Detach first. Each next link is captured before materialDestroy replaces it with the free link.
        this.shapeU[o + S_MATERIAL_HEAD] = 0xffffffff;
        this.shapeU[o + S_MATERIAL_COUNT] = 0;
        let id = head;
        for (let i = 0; i < count; ++i) {
            const next = this.materialU[id * MATERIAL_STRIDE + M_NEXT] >>> 0;
            k.materialDestroy(world.worldId, id);
            id = next;
        }
    }

    /** Write the shape's enlarged proxy AABB into the same resident shape-owned store. */
    writeFatAabb(shapeId: number, fat: AABB): void {
        const o = shapeId * 6;
        this.fatF[o] = fat.lowerBound.x;
        this.fatF[o + 1] = fat.lowerBound.y;
        this.fatF[o + 2] = fat.lowerBound.z;
        this.fatF[o + 3] = fat.upperBound.x;
        this.fatF[o + 4] = fat.upperBound.y;
        this.fatF[o + 5] = fat.upperBound.z;
    }
}

export function syncBodyQuery(world: WorldState, body: Body): void {
    const store = world.shapeStore;
    store.refreshViews();
    const pose = body.setIndex === SetType.Awake ? undefined : getBodySim(world, body).transform;
    for (let id = body.headShapeId; id !== NULL_INDEX; id = world.shapes[id].nextShapeId)
        store.writeQueryPose(id, body, pose);
}

/** Create an empty shape store for a new world. Its views are derived on the first write. */
export function createShapeStore(world: World | undefined, worldId: number): ShapeStore {
    return new ShapeStore(world, worldId);
}

/** Read live material records from the kernel-owned linked list. The returned objects are bridge values;
 * simulation decisions always re-read this column rather than a Shape.materials authoring array. */
export function readShapeMaterials(world: World | undefined, shape: Shape): SurfaceMaterial[] {
    const k = kernel(world);
    const head = k.shapeMaterialHead(shape.worldId, shape.id) >>> 0;
    const count = k.shapeMaterialCount(shape.worldId, shape.id) >>> 0;
    const listCount = k.materialListCount(shape.worldId, head) >>> 0;
    if (listCount !== count) {
        throw new Error(`physics: material attachment mismatch on shape ${shape.id}`);
    }
    if (count === 0) return [];
    const cap = k.materialCap();
    const ptr = k.materialLayoutPtr();
    const buf = k.memory.buffer;
    const base = new Uint32Array(buf, ptr, 1)[0] + shape.worldId * cap * MATERIAL_STRIDE * 4;
    const u = new Uint32Array(buf, base, cap * MATERIAL_STRIDE);
    const f = new Float32Array(buf, base, cap * MATERIAL_STRIDE);
    const out: SurfaceMaterial[] = [];
    let id = head;
    for (let i = 0; i < count; ++i) {
        const o = id * MATERIAL_STRIDE;
        const bits = BigInt(u[o + 6]) | (BigInt(u[o + 7]) << 32n);
        out.push({
            friction: f[o],
            restitution: f[o + 1],
            rollingResistance: f[o + 2],
            tangentVelocity: { x: f[o + 3], y: f[o + 4], z: f[o + 5] },
            userMaterialId: BigInt.asUintN(64, bits),
            customColor: u[o + 8],
        });
        id = u[o + M_NEXT] >>> 0;
    }
    return out;
}

/** The authoritative live material count for a shape. */
export function shapeMaterialCount(world: World | undefined, shape: Shape): number {
    const k = kernel(world);
    const head = k.shapeMaterialHead(shape.worldId, shape.id) >>> 0;
    const count = k.shapeMaterialCount(shape.worldId, shape.id) >>> 0;
    if (k.materialListCount(shape.worldId, head) >>> 0 !== count) {
        throw new Error(`physics: material attachment mismatch on shape ${shape.id}`);
    }
    return count;
}

/**
 * Write a newly created shape's record into the resident column, sizing the region to the new shape
 * high-water first. A grow relocates the manifold + geometry regions above the shape region and detaches
 * every view, so the stores that read through them are refreshed before anything else runs.
 */
export function writeShape(world: WorldState, shape: Shape): void {
    if (reserveShapes(world.ecsState, world.shapes.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    world.shapeStore.refreshViews();
    world.shapeStore.write(world, shape);
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

/** Size and write the resident fat-AABB lane owned by the shape store. */
export function writeFatAabb(world: WorldState, shape: Shape): void {
    if (reserveShapes(world.ecsState, world.shapes.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    world.shapeStore.refreshViews();
    world.shapeStore.writeFatAabb(shape.id, shape.fatAABB);
}
