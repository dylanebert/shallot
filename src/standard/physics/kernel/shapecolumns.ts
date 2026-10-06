import type { World } from "../../../engine";
import { type BodySimRef, simField } from "./bodycolumns";
// The persistent shape region (kernel/src/shapes.rs) — one record per shapeId (type code, local
// geometry, nextShapeId), held resident in the kernel's linear memory so the in-kernel finalize refit
// can walk a body's shape list and compute its AABBs without a per-step marshal. Each World owns
// its columns; a shape's slot is fixed for its life.
//
// The write sites are the shape lifecycle itself: `createShape` writes the whole record (a recycled
// shapeId inherits nothing), `destroyShape` patches the predecessor's `next` slot. There is no lazy
// dirty set — a stale record is invisible to TS and would silently feed the kernel garbage, so the
// column is written where the shape record is.
//
// Reserves can reallocate columns or grow memory; callers refresh views afterward.

import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import type { AABB } from "../common/math";
import { ShapeType, type SurfaceMaterial } from "../common/types";
import type { Capsule, Sphere } from "../shapes/geometry";
import type { HullData } from "../shapes/hull";
import type { Shape } from "../shapes/shape";
import { type Body, getBodySim } from "../world/body";
import type { WorldState } from "../world/world";
import { shapeBodyId } from "./filtercolumns";
import { kernel } from "./kernel";
import { KernelViews } from "./views";

/** Word stride of one kernel shape record, mirroring `shapes.rs`. */
export const SHAPE_STRIDE = 52;
export const S_PROXY_KEY = 50;
/** Shape type code — the `ShapeType` value verbatim. */
export const S_TYPE = 0;
/** Next shape in the body's list, or `NULL_INDEX` (0xFFFFFFFF through the u32 view). */
export const S_NEXT = 1;
/** Local geometry the AABB compute needs: sphere center(3)+radius(1), capsule center1(3)+center2(3)+
 * radius(1), hull local-AABB lower(3)+upper(3). Non-convex bounds use the geometry pools. */
export const S_GEOM = 2;
/** Hull record index or non-convex geometry word offset; capsule uses this lane for its radius. */
export const S_GEO_REFERENCE = 8;
/** Refit escaped its fat margin; consumed and cleared by the serial tree enlarge pass. */
export const S_ESCAPED = 15;
/** Kernel shape-record attachment lanes, outside finalize output. */
export const S_MATERIAL_HEAD = 16;
export const S_MATERIAL_COUNT = 17;

/** Kernel material record: friction, restitution, rolling, tangent xyz, u64 user id, color, link,
 * generation and alive. */
export const MATERIAL_STRIDE = 12;
const M_NEXT = 9;

/** @returns the smallest power-of-two capacity ≥ `need`, at least 16 (amortizes region grows). */
function growCap(need: number): number {
    let cap = 16;
    while (cap < need) cap *= 2;
    return cap;
}

/**
 * Size the persistent shape region to hold `shapeCount` shapes (the shape high-water). Grows the kernel
 * columns only when the count exceeds the current capacity. @returns true if they grew;
 * callers refresh views after growth.
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
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
    if (reserveShapes(world.ecsState, world.shapes.length + 1)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    const id = kernel(world.ecsState).shapeCreate(world.worldId);
    world.shapeStore.refreshViews();
    world.shapeStore.shapeF.fill(0, id * SHAPE_STRIDE + 34, id * SHAPE_STRIDE + 40);
    world.shapeStore.fatF.fill(0, id * 6, id * 6 + 6);
    return id;
}

export function destroyShapeSlot(world: WorldState, shapeId: number): void {
    kernel(world.ecsState).shapeDestroy(world.worldId, shapeId);
}

/**
 * Typed-array views over the resident shape column plus the shapeId-keyed writes. One per world. The
 * column is what the in-kernel finalize refit reads; TS writes it at shape create/destroy. Re-derives
 * its views when the shared kernel key changes.
 */
export class ShapeStore extends KernelViews {
    private readonly _worldId: number;

    constructor(ecsState: World | undefined, worldId: number) {
        super(ecsState);
        this._worldId = worldId;
        this.guardViews();
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
    // User material ids built per material record, with the id words each was built from. Held in an
    // object so the view guard leaves its words alone; ids are rebuilt only when a record's words change.
    private readonly _ids = { ids: [] as bigint[], words: new Uint32Array(0) };
    // The held layout header views are derived from.
    private _layout = new Uint32Array(0);
    private _fatLayout = new Uint32Array(0);
    private _materialLayout = new Uint32Array(0);

    // Shape/material lifecycle metadata and contents are entirely native region data.
    captureCheckpoint(): null {
        return null;
    }
    restoreCheckpoint(_state: unknown): void {}

    /** Re-derive the column views over the current region. No-op before the first `reserveShapes`, and
     * when the buffer, offset and capacity are those the views were derived at. */
    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        k.shapeSetActiveWorld(this._worldId);
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
            this.shapeU.byteOffset !== layout[0] ||
            this.shapeU.length !== cap * SHAPE_STRIDE
        ) {
            this.shapeU = new Uint32Array(buf, layout[0], cap * SHAPE_STRIDE);
            this.shapeF = new Float32Array(buf, layout[0], cap * SHAPE_STRIDE);
        }
        if (
            this.fatF.buffer !== buf ||
            this.fatF.byteOffset !== fatLayout[0] ||
            this.fatF.length !== fatCap * 6
        ) {
            this.fatF = new Float32Array(buf, fatLayout[0], fatCap * 6);
        }
        const materialCap = k.materialCap();
        if (
            this.materialU.buffer !== buf ||
            this.materialU.byteOffset !== materialLayout[0] ||
            this.materialU.length !== materialCap * MATERIAL_STRIDE
        ) {
            this.materialU = new Uint32Array(buf, materialLayout[0], materialCap * MATERIAL_STRIDE);
            this.materialF = new Float32Array(
                buf,
                materialLayout[0],
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
        for (let i = S_GEOM; i < SHAPE_STRIDE; ++i) {
            if (
                (i < 9 || i > 14) &&
                (i < 25 || i > 29) &&
                i !== 31 &&
                i !== 41 &&
                (i < 34 || i > 39)
            )
                f[o + i] = 0;
        }
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
        const body = world.bodies[shapeBodyId(world, shape.id)];
        this.writeQueryPose(world, shape.id, body, getBodySim(world, body));
    }

    writeQueryProperties(shape: Shape): void {
        const n = shape.id * SHAPE_STRIDE;
        const u = this.shapeU;
        u[n + 30] = Number(shape.enableSensorEvents);
        u[n + S_PROXY_KEY] = shape.proxyKey;
        this.shapeF[n + 40] = shape.aabbMargin;
        this.shapeF[n + 43] = shape.hull?.innerRadius ?? 0;
        u[n + 51] = Number(shape.enableHitEvents);
    }

    writeQueryPose(world: WorldState, shapeId: number, body: Body, sim?: BodySimRef): void {
        const n = shapeId * SHAPE_STRIDE;
        this.shapeU[n + 32] = body.setIndex === SetType.Awake ? body.localIndex + 1 : 0;
        if (body.setIndex === SetType.Awake) return;
        if (!sim) throw new Error("physics: a non-awake query shape requires its sleeping pose");
        const pose = simField(world, sim, "transform");
        this.shapeU[n + 42] = simField(world, sim, "flags");
        const f = this.shapeF;
        f[n + 44] = simField(world, sim, "center").x;
        f[n + 45] = simField(world, sim, "center").y;
        f[n + 46] = simField(world, sim, "center").z;
        f[n + 47] = simField(world, sim, "localCenter").x;
        f[n + 48] = simField(world, sim, "localCenter").y;
        f[n + 49] = simField(world, sim, "localCenter").z;
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

    /** Material record `id`'s user material id, built once per distinct value of its id words. */
    userMaterialId(id: number): bigint {
        const o = id * MATERIAL_STRIDE;
        const low = this.materialU[o + 6];
        const high = this.materialU[o + 7];
        const cache = this._ids;
        if (cache.words.length < 2 * id + 2) {
            const words = new Uint32Array(Math.max(32, 4 * id + 4));
            words.set(cache.words);
            cache.words = words;
        }
        const words = cache.words;
        if (cache.ids[id] === undefined || words[2 * id] !== low || words[2 * id + 1] !== high) {
            words[2 * id] = low;
            words[2 * id + 1] = high;
            cache.ids[id] = BigInt(low) | (BigInt(high) << 32n);
        }
        return cache.ids[id];
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
        const f = this.fatF;
        f[o] = fat.lowerBound.x;
        f[o + 1] = fat.lowerBound.y;
        f[o + 2] = fat.lowerBound.z;
        f[o + 3] = fat.upperBound.x;
        f[o + 4] = fat.upperBound.y;
        f[o + 5] = fat.upperBound.z;
    }
}

export function syncBodyQuery(world: WorldState, body: Body): void {
    const store = world.shapeStore;
    store.refreshViews();
    const sim = body.setIndex === SetType.Awake ? undefined : getBodySim(world, body);
    for (let id = body.headShapeId; id !== NULL_INDEX; id = world.shapes[id].nextShapeId) {
        store.writeQueryPose(world, id, body, sim);
    }
}

/** Create an empty shape store for a new world. Its views are derived on the first write. */
export function createShapeStore(world: World | undefined, worldId: number): ShapeStore {
    return new ShapeStore(world, worldId);
}

/** Read live material records from the kernel-owned linked list. The returned objects are bridge values;
 * simulation decisions always re-read this column rather than a Shape.materials authoring array. */
export function readShapeMaterials(world: World | undefined, shape: Shape): SurfaceMaterial[] {
    const k = kernel(world);
    k.shapeSetActiveWorld(shape.worldId);
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
    const base = new Uint32Array(buf, ptr, 1)[0];
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
 * high-water first. Refresh views after a grow-capable call before writing.
 */
export function writeShape(world: WorldState, shape: Shape): void {
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
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

/** Keep the last shape bounds resident in `shapeF`, the shape store's current view, for the next
 * continuous sweep. */
export function writeTightAabb(shapeF: Float32Array, shapeId: number, box: AABB): void {
    const o = shapeId * SHAPE_STRIDE + 34;
    shapeF[o] = box.lowerBound.x;
    shapeF[o + 1] = box.lowerBound.y;
    shapeF[o + 2] = box.lowerBound.z;
    shapeF[o + 3] = box.upperBound.x;
    shapeF[o + 4] = box.upperBound.y;
    shapeF[o + 5] = box.upperBound.z;
}

/** Copy the kernel-owned tight bounds into caller-owned scratch. */
export function readShapeAabb(world: WorldState, shapeId: number, out: AABB): AABB {
    world.shapeStore.refreshViews();
    return readBounds(world.shapeStore.shapeF, shapeId * SHAPE_STRIDE + 34, out);
}

/** Copy the kernel-owned fat bounds into caller-owned scratch. */
export function readFatAabb(world: WorldState, shapeId: number, out: AABB): AABB {
    world.shapeStore.refreshViews();
    return readBounds(world.shapeStore.fatF, shapeId * 6, out);
}

function readBounds(f: Float32Array, o: number, out: AABB): AABB {
    out.lowerBound.x = f[o];
    out.lowerBound.y = f[o + 1];
    out.lowerBound.z = f[o + 2];
    out.upperBound.x = f[o + 3];
    out.upperBound.y = f[o + 4];
    out.upperBound.z = f[o + 5];
    return out;
}

/** Size and write the resident fat-AABB lane owned by the shape store. */
export function writeFatAabb(world: WorldState, shapeId: number, box: AABB): void {
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
    if (reserveShapes(world.ecsState, world.shapes.length)) {
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
    }
    world.shapeStore.refreshViews();
    world.shapeStore.writeFatAabb(shapeId, box);
}
