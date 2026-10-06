import type { World } from "../../../engine";
import type { AABB } from "../common/math";
import type { ShapeDef, ShapeType, SurfaceMaterial } from "../common/types";
import type { Shape } from "../shapes/shape";
import type { WorldState } from "../world/world";
import { shapeBodyId } from "./filtercolumns";
import { kernel } from "./kernel";
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

import { KernelViews } from "./views";

/** Word stride of one kernel shape record, mirroring `shapes.rs`. */
export const SHAPE_STRIDE = 69;
export const S_PROXY_KEY = 50;
/** b3Shape union: inline sphere or capsule, data reference plus mesh scale, or data reference. */
export const S_GEOM = 2;
/** First union word: retained hull pointer or non-convex pool word offset. */
export const S_GEO_REFERENCE = S_GEOM;
/** Kernel shape-record attachment lanes, outside finalize output. */
export const S_MATERIAL_HEAD = 16;
export const S_MATERIAL_COUNT = 17;

/** Surface material: friction, restitution, rolling, tangent xyz, u64 user id and color. */
export const MATERIAL_STRIDE = 9;

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
export function createShapeSlot(
    world: WorldState,
    body: number,
    type: ShapeType,
    def: ShapeDef,
): number {
    const flags =
        Number(def.enableSensorEvents) |
        (Number(def.enableContactEvents) << 1) |
        (Number(def.enableCustomFiltering) << 2) |
        (Number(def.enableHitEvents) << 3) |
        (Number(def.enablePreSolveEvents) << 4) |
        (Number(def.enableSpeculativeContact) << 6);
    const id = kernel(world.ecsState).shapeCreate(
        world.worldId,
        body,
        type,
        def.density,
        def.explosionScale,
        flags,
    );
    world.shapeStore.refreshViews();
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
    /** Borrowed synchronous geometry result; copied before another kernel call. */
    geometryOutput = new Float32Array(0);
    /** Resident fat-AABB column owned by this shape store, not a second helper store. */
    fatF = new Float32Array(0);
    /** Addressable memory for inline materials and owned contiguous material arrays. */
    materialU = new Uint32Array(0);
    materialF = new Float32Array(0);
    private _materialData = new DataView(new ArrayBuffer(0));
    // BigInts are reused while a material address holds the same user-id words.
    private readonly _ids = new Map<number, { low: number; high: number; value: bigint }>();
    // The held layout header views are derived from.
    private _layout = new Uint32Array(0);
    private _fatLayout = new Uint32Array(0);

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
        const layout = this._layout;
        const fatLayout = this._fatLayout;
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
        if (this.geometryOutput.buffer !== buf)
            this.geometryOutput = new Float32Array(buf, k.shapeGeometryOutputPtr(), 13);
        if (this.materialU.buffer !== buf || this.materialU.byteLength !== buf.byteLength) {
            this.materialU = new Uint32Array(buf);
            this.materialF = new Float32Array(buf);
            this._materialData = new DataView(buf);
        }
    }

    /** Attach the body's query pose; geometry references were written at shape creation. */
    write(world: WorldState, shape: Shape): void {
        const body = shapeBodyId(world, shape);
        this.writeQueryPose(world, shape, body);
    }

    writeQueryPose(world: WorldState, shapeId: number, body: number): void {
        kernel(world.ecsState).shapeQueryPose(world.worldId, shapeId, body);
    }

    /** Refresh a shape's pool reference without touching its material or finalize lanes. */
    /** Copy authored materials into the shape's inline material or owned contiguous array. */
    writeMaterials(
        world: WorldState,
        shape: Shape,
        materials: SurfaceMaterial[] | SurfaceMaterial,
    ): void {
        const count = Array.isArray(materials) ? materials.length : 1;
        this.refreshViews();
        const ptr = kernel(world.ecsState).shapeAllocateMaterials(world.worldId, shape, count);
        world.manifoldStore.refreshViews();
        world.bodyStore.refreshViews();
        this.refreshViews();
        for (let i = 0; i < count; ++i) {
            const o = ptr / 4 + i * MATERIAL_STRIDE;
            const m = Array.isArray(materials) ? materials[i] : materials;
            this._materialData.setBigUint64((o + 6) * 4, m.userMaterialId, true);
            kernel(world.ecsState).shapeMaterialSet(
                world.worldId,
                shape,
                i,
                m.friction,
                m.restitution,
                m.rollingResistance,
                m.tangentVelocity.x,
                m.tangentVelocity.y,
                m.tangentVelocity.z,
                this.materialU[o + 6],
                this.materialU[o + 7],
                m.customColor,
            );
            let cached = this._ids.get(o);
            if (!cached) {
                cached = { low: 0, high: 0, value: 0n };
                this._ids.set(o, cached);
            }
            cached.low = this.materialU[o + 6];
            cached.high = this.materialU[o + 7];
            cached.value = m.userMaterialId;
        }
    }

    /** Material record `id`'s user material id, built once per distinct value of its id words. */
    userMaterialId(id: number): bigint {
        const o = id;
        const low = this.materialU[o + 6];
        const high = this.materialU[o + 7];
        let cached = this._ids.get(id);
        if (!cached || cached.low !== low || cached.high !== high) {
            cached = { low, high, value: BigInt(low) | (BigInt(high) << 32n) };
            this._ids.set(id, cached);
        }
        if (cached.value < 0n || cached.value > 0xffffffffffffffffn)
            cached.value = this._materialData.getBigUint64((o + 6) * 4, true);
        return cached.value;
    }

    /** Copy one live inline or owned material into caller-owned scratch. */
    readMaterialAt(shape: Shape, index: number, out: SurfaceMaterial): SurfaceMaterial {
        this.refreshViews();
        const k = kernel(this.ecsState);
        const count = k.shapeMaterialCount(this._worldId, shape) >>> 0;
        if (count === 0) throw new Error(`physics: no material on shape ${shape}`);
        if (index < 0 || index >= count)
            throw new Error(`physics: no material ${index} on shape ${shape}`);
        const o = k.shapeMaterialPtr(this._worldId, shape) / 4 + index * MATERIAL_STRIDE;
        const f = this.materialF;
        out.friction = f[o];
        out.restitution = f[o + 1];
        out.rollingResistance = f[o + 2];
        out.tangentVelocity.x = f[o + 3];
        out.tangentVelocity.y = f[o + 4];
        out.tangentVelocity.z = f[o + 5];
        out.userMaterialId = this.userMaterialId(o);
        out.customColor = this.materialU[o + 8];
        return out;
    }

    /** Read a material's user id without constructing a TypeScript material copy. */
    materialUserIdAt(shape: Shape, index: number): bigint {
        this.refreshViews();
        const k = kernel(this.ecsState);
        if (k.shapeMaterialCount(this._worldId, shape) >>> 0 === 0) return 0n;
        const o = k.shapeMaterialPtr(this._worldId, shape) / 4 + index * MATERIAL_STRIDE;
        return this.userMaterialId(o);
    }

    /** Detach and release the kernel material records owned by a shape. */
    destroyMaterials(world: WorldState, shape: Shape): void {
        this.refreshViews();
        kernel(world.ecsState).shapeFreeMaterials(world.worldId, shape);
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

export function syncBodyQuery(world: WorldState, body: number): void {
    kernel(world.ecsState).shapeSyncBody(world.worldId, body);
}

/** Create an empty shape store for a new world. Its views are derived on the first write. */
export function createShapeStore(world: World | undefined, worldId: number): ShapeStore {
    return new ShapeStore(world, worldId);
}

/** Read live materials from the kernel. The returned objects are bridge values;
 * simulation decisions always re-read this column rather than a Shape.materials authoring array. */
export function readShapeMaterials(world: WorldState, shape: Shape): SurfaceMaterial[] {
    const k = kernel(world.ecsState);
    k.shapeSetActiveWorld(world.worldId);
    const count = k.shapeMaterialCount(world.worldId, shape) >>> 0;
    if (count === 0) return [];
    const ptr = k.shapeMaterialPtr(world.worldId, shape);
    const buf = k.memory.buffer;
    const u = new Uint32Array(buf, ptr, count * MATERIAL_STRIDE);
    const f = new Float32Array(buf, ptr, count * MATERIAL_STRIDE);
    const out: SurfaceMaterial[] = [];
    for (let i = 0; i < count; ++i) {
        const o = i * MATERIAL_STRIDE;
        const bits = BigInt(u[o + 6]) | (BigInt(u[o + 7]) << 32n);
        out.push({
            friction: f[o],
            restitution: f[o + 1],
            rollingResistance: f[o + 2],
            tangentVelocity: { x: f[o + 3], y: f[o + 4], z: f[o + 5] },
            userMaterialId: BigInt.asUintN(64, bits),
            customColor: u[o + 8],
        });
    }
    return out;
}

/** The authoritative live material count for a shape. */
export function shapeMaterialCount(world: WorldState, shape: Shape): number {
    const k = kernel(world.ecsState);
    return k.shapeMaterialCount(world.worldId, shape) >>> 0;
}

/**
 * Write a newly created shape's record into the resident column, sizing the region to the new shape
 * high-water first. Refresh views after a grow-capable call before writing.
 */
export function writeShape(world: WorldState, shape: Shape): void {
    kernel(world.ecsState).shapeSetActiveWorld(world.worldId);
    world.manifoldStore.refreshViews();
    world.bodyStore.refreshViews();
    world.shapeStore.refreshViews();
    world.shapeStore.write(world, shape);
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
    world.shapeStore.refreshViews();
    world.shapeStore.writeFatAabb(shapeId, box);
}
