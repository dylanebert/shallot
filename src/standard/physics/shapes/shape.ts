import { shapeSensorIndex, writeShapeFilterValue } from "../kernel/filtercolumns";
import { acquireCompoundData, acquireHeightFieldData, acquireMeshData } from "../kernel/geocolumns";
import { ShapeField, shapeField, shapeScalar } from "../kernel/shaperecords";
// Shape API and authoring bridge to Box3D's shape.c operations in the kernel (Erin Catto, MIT).

import type * as bp from "../collision/broadphase";
import { NULL_INDEX } from "../common/array";
import { hi32, lo32 } from "../common/bits";
import type { EntityId } from "../common/ids";
import type { AABB, Transform, Vec3, WorldTransform } from "../common/records";

const f32 = Math.fround;

import {
    type BodyType,
    type Filter,
    type ShapeDef,
    ShapeType,
    type SurfaceMaterial,
} from "../common/types";
import { kernel } from "../kernel/kernel";
import {
    createShapeSlot,
    readShapeMaterials,
    S_GEO_REFERENCE,
    SHAPE_STRIDE,
    shapeMaterialCount,
} from "../kernel/shapecolumns";
import { releaseGeometryIdentities } from "../world/body";
import type { Visitor } from "../world/sensor";
import { addHullToDatabase, type WorldState } from "../world/world";
import { type CompoundData, getCompoundMaterials } from "./compound";
import type { Capsule, MassData, Sphere } from "./geometry";
import type { HeightFieldData } from "./heightfield";
import type { HullData } from "./hull";
import type { MeshData } from "./mesh";

/** Min extent (smallest sphere fitting inside) and max extent per axis, for sleeping (b3ShapeExtent). */
export type ShapeExtent = { minExtent: number; maxExtent: Vec3 };

/** Shape identity; the kernel column is its only nongeometry record. */
export type Shape = number;
export function shapeIsValid(world: WorldState, id: EntityId): boolean {
    if (!world.inUse) return false;
    world.shapeStore.refreshViews();
    const shape = id.index1 - 1;
    const k = kernel(world.ecsState);
    return (
        shape >= 0 &&
        shape < world.shapeStore.shapeU.length / SHAPE_STRIDE &&
        k.shapeAlive(world.worldId, shape) !== 0 &&
        k.shapeGeneration(world.worldId, shape) === id.generation
    );
}

export function readShapeSphere(world: WorldState, shape: Shape): Sphere {
    world.shapeStore.refreshViews();
    const f = world.shapeStore.shapeF,
        o = shape * SHAPE_STRIDE + 48;
    return { center: { x: f[o], y: f[o + 1], z: f[o + 2] }, radius: f[o + 3] };
}

export function readShapeCapsule(world: WorldState, shape: Shape): Capsule {
    world.shapeStore.refreshViews();
    const f = world.shapeStore.shapeF,
        o = shape * SHAPE_STRIDE + 48;
    return {
        center1: { x: f[o], y: f[o + 1], z: f[o + 2] },
        center2: { x: f[o + 3], y: f[o + 4], z: f[o + 5] },
        radius: f[o + 6],
    };
}

export function shapeRadius(world: WorldState, shape: Shape): number {
    world.shapeStore.refreshViews();
    const offset = shapeField(world, shape, ShapeField.type) === ShapeType.Sphere ? 51 : 54;
    return world.shapeStore.shapeF[shape * SHAPE_STRIDE + offset];
}

/**
 * Compounds own their material array, including a single material; other one-material shapes use
 * inline storage. This requested observation returns an independent array (b3GetShapeMaterials).
 */
export function getShapeMaterials(world: WorldState, shape: Shape): SurfaceMaterial[] {
    return readShapeMaterials(world, shape);
}

/** Authoritative live material count, read from the kernel shape record. */
export function getShapeMaterialCount(world: WorldState, shape: Shape): number {
    return shapeMaterialCount(world, shape);
}

/** The shape's material 0 — what a convex contact mixes — read in place into `out`, as Box3D reads
 * `b3GetShapeMaterials(shape)[0]`; valid until the next read into the same record. */
export function getShapeMaterial(
    world: WorldState,
    shape: Shape,
    out: SurfaceMaterial,
): SurfaceMaterial {
    return world.shapeStore.readMaterialAt(shape, 0, out);
}

export function getShapeType(world: WorldState, shape: Shape): ShapeType {
    return shapeField(world, shape, ShapeField.type) as ShapeType;
}

export function getShapeDensity(world: WorldState, shape: Shape): number {
    return shapeScalar(world, shape, ShapeField.density);
}

export function getShapeGeometryReference(world: WorldState, shape: Shape): number {
    return kernel(world.ecsState).shapeGetGeometryReference(world.worldId, shape);
}

export function getShapeGeometryScale(world: WorldState, shape: Shape): Vec3 {
    world.shapeStore.refreshViews();
    const offset = shape * SHAPE_STRIDE + 49;
    const values = world.shapeStore.shapeF;
    return { x: values[offset], y: values[offset + 1], z: values[offset + 2] };
}

export function getShapeFilter(world: WorldState, shape: Shape): Filter {
    world.shapeStore.refreshViews();
    const offset = shape * SHAPE_STRIDE;
    const words = world.shapeStore.shapeU;
    return {
        categoryBits: (BigInt(words[offset + 39] >>> 0) << 32n) | BigInt(words[offset + 38] >>> 0),
        maskBits: (BigInt(words[offset + 41] >>> 0) << 32n) | BigInt(words[offset + 40] >>> 0),
        groupIndex: words[offset + 42] | 0,
    };
}

/**
 * The user material id at a contact point (b3GetShapeUserMaterialId). Selects the per-triangle
 * material for a mesh/height-field, the child's remapped slot for a compound, else material 0.
 */
export function getShapeUserMaterialId(
    world: WorldState,
    shape: Shape,
    childIndex: number,
    triangleIndex: number,
): bigint {
    const materialCount = getShapeMaterialCount(world, shape);
    if (materialCount === 0) {
        return 0n;
    }

    const materialIndex = kernel(world.ecsState).shapeMaterialIndex(
        world.worldId,
        shape,
        childIndex,
        triangleIndex,
    );
    return world.shapeStore.userMaterialIdAt(
        shape,
        Math.max(0, Math.min(materialIndex, materialCount - 1)),
    );
}

// --- geometry dispatch -----------------------------------------------------------------------

/** Mass, center, and inertia of a shape at its density (b3ComputeShapeMass). */
export function computeShapeMass(
    world: WorldState,
    shape: Shape,
    out: MassData = {
        mass: 0,
        center: { x: 0, y: 0, z: 0 },
        inertia: { cx: { x: 0, y: 0, z: 0 }, cy: { x: 0, y: 0, z: 0 }, cz: { x: 0, y: 0, z: 0 } },
    },
): MassData {
    kernel(world.ecsState).shapeComputeMass(world.worldId, shape);
    const values = world.shapeStore.geometryOutput;
    out.mass = values[0];
    copyGeometryVector(values, 1, out.center);
    copyGeometryVector(values, 4, out.inertia.cx);
    copyGeometryVector(values, 7, out.inertia.cy);
    copyGeometryVector(values, 10, out.inertia.cz);
    return out;
}

function copyGeometryVector(values: Float32Array, lane: number, out: Vec3): Vec3 {
    out.x = values[lane];
    out.y = values[lane + 1];
    out.z = values[lane + 2];
    return out;
}

/** Min/max extent of a shape relative to a local center, for sleeping bounds (b3ComputeShapeExtent). */
export function computeShapeExtent(
    world: WorldState,
    shape: Shape,
    localCenter: Vec3,
    out: ShapeExtent = { minExtent: 0, maxExtent: { x: 0, y: 0, z: 0 } },
): ShapeExtent {
    kernel(world.ecsState).shapeComputeExtent(
        world.worldId,
        shape,
        localCenter.x,
        localCenter.y,
        localCenter.z,
    );
    const values = world.shapeStore.geometryOutput;
    out.minExtent = values[0];
    copyGeometryVector(values, 1, out.maxExtent);
    return out;
}

/** Enclosing AABB of a shape under a transform (b3ComputeShapeAABB). */
export function computeShapeAABB(world: WorldState, shape: Shape, transform: Transform): AABB {
    return computeShapeAABBOut(world, shape, transform, {
        lowerBound: { x: 0, y: 0, z: 0 },
        upperBound: { x: 0, y: 0, z: 0 },
    });
}

/** Conservative world AABB inflated by `extra` (b3ComputeFatShapeAABB, single-precision path). */
export function computeFatShapeAABB(
    world: WorldState,
    shape: Shape,
    transform: WorldTransform,
    extra: number,
): AABB {
    return computeFatShapeAABBOut(world, shape, transform, extra, {
        lowerBound: { x: 0, y: 0, z: 0 },
        upperBound: { x: 0, y: 0, z: 0 },
    });
}

/** Enclosing kernel AABB copied into caller-owned storage, without allocating. */
export function computeShapeAABBOut(
    world: WorldState,
    shape: Shape,
    transform: Transform,
    o: AABB,
): AABB {
    return computeFatShapeAABBOut(world, shape, transform, 0, o);
}

/** Inflated kernel AABB copied into caller-owned storage, without allocating. */
export function computeFatShapeAABBOut(
    world: WorldState,
    shape: Shape,
    transform: WorldTransform,
    extra: number,
    o: AABB,
): AABB {
    const p = transform.p,
        q = transform.q;
    kernel(world.ecsState).shapeComputeAABB(
        world.worldId,
        shape,
        p.x,
        p.y,
        p.z,
        q.v.x,
        q.v.y,
        q.v.z,
        q.s,
        extra,
    );
    const values = world.shapeStore.geometryOutput;
    copyGeometryVector(values, 0, o.lowerBound);
    copyGeometryVector(values, 3, o.upperBound);
    return o;
}

/** Local centroid of a shape (b3GetShapeCentroid). */
export function getShapeCentroid(
    world: WorldState,
    shape: Shape,
    out: Vec3 = { x: 0, y: 0, z: 0 },
): Vec3 {
    kernel(world.ecsState).shapeGetCentroid(world.worldId, shape);
    return copyGeometryVector(world.shapeStore.geometryOutput, 0, out);
}

// --- proxy -----------------------------------------------------------------------------------

export function createShapeProxy(
    shape: Shape,
    broadPhase: bp.BroadPhase,
    _type: BodyType,
    transform: WorldTransform,
    forcePairCreation: boolean,
): void {
    const world = broadPhase.store.world;
    if (!world) throw new Error("physics: shape proxies require a world-owned broad phase");
    world.broadPhase.store.initialize();
    const p = transform.p,
        q = transform.q;
    kernel(world.ecsState).shapeCreateProxyTransform(
        world.worldId,
        shape,
        Number(forcePairCreation),
        p.x,
        p.y,
        p.z,
        q.v.x,
        q.v.y,
        q.v.z,
        q.s,
    );
    world.broadPhase.store.refreshViews();
}

export function destroyShapeProxy(shape: Shape, broadPhase: bp.BroadPhase): void {
    const world = broadPhase.store.world!;
    kernel(world.ecsState).shapeDestroyProxy(world.worldId, shape);
}

export function setShapeFilter(world: WorldState, shape: Shape, filter: Filter): void {
    world.broadPhase.store.initialize();
    writeShapeFilterValue(world, shape, filter, true);
    world.broadPhase.store.refreshViews();
}

// --- create / destroy ------------------------------------------------------------------------
const shapeScale = { x: 1, y: 1, z: 1 };

function createShapeInternal(
    world: WorldState,
    body: number,
    def: ShapeDef,
    geometry: Sphere | Capsule | HullData | MeshData | HeightFieldData | CompoundData,
    shapeType: ShapeType,
    scale: Vec3,
): Shape | null {
    // Round every user float to f32 once at ingress (density/explosionScale/material floats); the C def
    // is f32, so an unrounded f64 scalar would reach mass/solve and break bit-exact parity. Filter
    // category/mask bigints and enum/bool fields pass through untouched.
    const shapeId = createShapeSlot(world, body, shapeType, def);
    const shape = shapeId;
    const k = kernel(world.ecsState);
    if (shapeType === ShapeType.Capsule) {
        const c = geometry as Capsule;
        shapeType = k.shapeSetGeometry(
            world.worldId,
            shape,
            c.center1.x,
            c.center1.y,
            c.center1.z,
            c.center2.x,
            c.center2.y,
            c.center2.z,
            c.radius,
        ) as ShapeType;
    } else if (shapeType === ShapeType.Sphere) {
        const s = geometry as Sphere;
        k.shapeSetGeometry(
            world.worldId,
            shape,
            s.center.x,
            s.center.y,
            s.center.z,
            s.radius,
            0,
            0,
            0,
        );
    } else if (shapeType === ShapeType.Mesh) {
        k.shapeSetGeometry(world.worldId, shape, scale.x, scale.y, scale.z, 0, 0, 0, 0);
    }
    switch (shapeType) {
        case ShapeType.Capsule:
        case ShapeType.Sphere:
            break;
        case ShapeType.Hull: {
            const handle = addHullToDatabase(world, geometry as HullData);
            world.shapeStore.refreshViews();
            world.bodyStore.refreshViews();
            world.manifoldStore.refreshViews();
            world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE] = handle;
            break;
        }
        case ShapeType.Mesh: {
            const pointer = acquireMeshData(world, geometry as MeshData);
            world.shapeStore.refreshViews();
            world.bodyStore.refreshViews();
            world.manifoldStore.refreshViews();
            world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE] = pointer;
            break;
        }
        case ShapeType.HeightField: {
            const pointer = acquireHeightFieldData(world, geometry as HeightFieldData);
            world.shapeStore.refreshViews();
            world.bodyStore.refreshViews();
            world.manifoldStore.refreshViews();
            world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE] = pointer;
            break;
        }
        case ShapeType.Compound: {
            const pointer = acquireCompoundData(world, geometry as CompoundData);
            world.shapeStore.refreshViews();
            world.bodyStore.refreshViews();
            world.manifoldStore.refreshViews();
            world.shapeStore.shapeU[shape * SHAPE_STRIDE + S_GEO_REFERENCE] = pointer;
            break;
        }
        default:
            throw new Error(`physics: unknown shape type ${shapeType}`);
    }

    writeShapeFilterValue(world, shape, def.filter);
    world.shapeUserData[shape] = def.userData;
    world.shapeNames[shape] = def.name ?? "";
    // The kernel pool advanced the generation at allocation; the public bridge only carries it.

    const materials =
        shapeType === ShapeType.Compound
            ? getCompoundMaterials(geometry as CompoundData)
            : def.materials?.length
              ? def.materials
              : def.baseMaterial;
    world.shapeStore.writeMaterials(world, shape, materials);
    k.shapeFinishGeometry(world.worldId, shape);

    world.broadPhase.store.initialize();
    k.shapeFinishCreate(
        world.worldId,
        shape,
        def.invokeContactCreation,
        def.isSensor,
        def.updateBodyMass,
    );
    world.broadPhase.store.refreshViews();

    return shape;
}

function createShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    geometry: Sphere | Capsule | HullData | MeshData | HeightFieldData | CompoundData,
    shapeType: ShapeType,
    scale: Vec3 = shapeScale,
): Shape | null {
    if (!kernel(world.ecsState).shapeCanCreate(world.worldId, body, shapeType)) return null;
    world.locked = true;
    const shape = createShapeInternal(world, body, def, geometry, shapeType, scale);
    world.locked = false;
    return shape;
}

export function createSphereShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    sphere: Sphere,
): Shape | null {
    return createShape(world, body, def, sphere, ShapeType.Sphere);
}

export function createCapsuleShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    capsuleInput: Capsule,
): Shape | null {
    return createShape(world, body, def, capsuleInput, ShapeType.Capsule);
}

export function createHullShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    hull: HullData,
): Shape | null {
    return createShape(world, body, def, hull, ShapeType.Hull);
}

export function createMeshShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    mesh: MeshData,
    scale: Vec3,
): Shape | null {
    return createShape(world, body, def, mesh, ShapeType.Mesh, scale);
}

export function createHeightFieldShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    heightField: HeightFieldData,
): Shape | null {
    return createShape(world, body, def, heightField, ShapeType.HeightField);
}

export function createCompoundShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    compound: CompoundData,
): Shape | null {
    return createShape(world, body, def, compound, ShapeType.Compound);
}

export function setShapeSphere(world: WorldState, shape: Shape, sphere: Sphere): void {
    kernel(world.ecsState).shapeSetSphere(
        world.worldId,
        shape,
        f32(sphere.center.x),
        f32(sphere.center.y),
        f32(sphere.center.z),
        f32(sphere.radius),
    );
    releaseGeometryIdentities(world);
}

export function setShapeCapsule(world: WorldState, shape: Shape, capsule: Capsule): void {
    kernel(world.ecsState).shapeSetCapsule(
        world.worldId,
        shape,
        f32(capsule.center1.x),
        f32(capsule.center1.y),
        f32(capsule.center1.z),
        f32(capsule.center2.x),
        f32(capsule.center2.y),
        f32(capsule.center2.z),
        f32(capsule.radius),
    );
    releaseGeometryIdentities(world);
}

export function setShapeHull(world: WorldState, shape: Shape, hull: HullData): void {
    const handle = addHullToDatabase(world, hull);
    kernel(world.ecsState).shapeSetHull(world.worldId, shape, handle);
    releaseGeometryIdentities(world);
}

export function setShapeMesh(world: WorldState, shape: Shape, mesh: MeshData, scale: Vec3): void {
    const handle = acquireMeshData(world, mesh);
    kernel(world.ecsState).shapeSetMesh(
        world.worldId,
        shape,
        handle,
        f32(scale.x),
        f32(scale.y),
        f32(scale.z),
    );
    releaseGeometryIdentities(world);
}

export function setShapeDensity(
    world: WorldState,
    shape: Shape,
    density: number,
    updateBodyMass: boolean,
): void {
    kernel(world.ecsState).shapeSetDensity(world.worldId, shape, f32(density), updateBodyMass);
}

export function setShapeSurfaceMaterial(
    world: WorldState,
    shape: Shape,
    material: SurfaceMaterial,
): void {
    const userMaterialId = BigInt.asUintN(64, material.userMaterialId);
    kernel(world.ecsState).shapeSetSurfaceMaterial(
        world.worldId,
        shape,
        f32(material.friction),
        f32(material.restitution),
        f32(material.rollingResistance),
        f32(material.tangentVelocity[0]),
        f32(material.tangentVelocity[1]),
        f32(material.tangentVelocity[2]),
        lo32(userMaterialId),
        hi32(userMaterialId),
        material.customColor >>> 0,
    );
}

export function setShapeFriction(world: WorldState, shape: Shape, friction: number): void {
    kernel(world.ecsState).shapeSetFriction(world.worldId, shape, f32(friction));
}

export function setShapeRestitution(world: WorldState, shape: Shape, restitution: number): void {
    kernel(world.ecsState).shapeSetRestitution(world.worldId, shape, f32(restitution));
}

export function setShapeMeshMaterial(
    world: WorldState,
    shape: Shape,
    material: SurfaceMaterial,
    index: number,
): void {
    if (!Number.isInteger(index) || index < 0 || index >= getShapeMaterialCount(world, shape))
        throw new RangeError("physics: mesh material index is out of range");
    const userMaterialId = BigInt.asUintN(64, material.userMaterialId);
    kernel(world.ecsState).shapeSetMeshMaterial(
        world.worldId,
        shape,
        index,
        f32(material.friction),
        f32(material.restitution),
        f32(material.rollingResistance),
        f32(material.tangentVelocity[0]),
        f32(material.tangentVelocity[1]),
        f32(material.tangentVelocity[2]),
        lo32(userMaterialId),
        hi32(userMaterialId),
        material.customColor >>> 0,
    );
}

export function destroyShape(world: WorldState, shape: Shape, updateBodyMass: boolean): void {
    world.locked = true;
    kernel(world.ecsState).shapeDestroyWorld(world.worldId, shape, updateBodyMass);
    releaseGeometryIdentities(world);
    world.shapeUserData[shape] = undefined;
    world.shapeNames[shape] = "";
    world.locked = false;
}

/** Whether a shape is a sensor (b3Shape_IsSensor). */
export function isSensorShape(world: WorldState, shape: Shape): boolean {
    return shapeSensorIndex(world, shape) !== NULL_INDEX;
}

/**
 * The shapes currently overlapping a sensor (b3Shape_GetSensorData). Returns a fresh copy of the
 * sensor's current-frame overlaps; empty if the shape is not a sensor.
 */
export function getSensorData(world: WorldState, shape: Shape): Visitor[] {
    if (world.locked || shapeSensorIndex(world, shape) === NULL_INDEX) {
        return [];
    }
    const k = kernel(world.ecsState);
    const index = shapeSensorIndex(world, shape);
    const result: Visitor[] = [];
    for (let i = 0, count = k.sensorVisitorCount(world.worldId, index); i < count; ++i) {
        result.push({
            shapeId: k.sensorVisitorWord(world.worldId, index, i, 0),
            generation: k.sensorVisitorWord(world.worldId, index, i, 1),
        });
    }
    return result;
}
