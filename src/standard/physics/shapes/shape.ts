import { ContactField, contactField } from "../collision/contact";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    bodyType,
    shapeBodyId,
    shapeSensorIndex,
    writeShapeFilterValue,
} from "../kernel/filtercolumns";
import { rebuildGeometry } from "../kernel/geocolumns";
import { ShapeField, shapeField } from "../kernel/shaperecords";
// Shape API and authoring bridge to Box3D's shape.c operations in the kernel (Erin Catto, MIT).

import type * as bp from "../collision/broadphase";
import { destroyContact } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { SetType } from "../common/constants";
import {
    type AABB,
    clampInt,
    mat3,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
    xf,
} from "../common/math";
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
    destroyShapeSlot,
    readShapeMaterials,
    SHAPE_STRIDE,
    shapeMaterialCount,
    writeShape,
} from "../kernel/shapecolumns";
import { readBodyTransform, updateBodyMassData } from "../world/body";
import { createSensor, destroySensor, type Visitor } from "../world/sensor";
import {
    addCompoundToDatabase,
    addGeometryToDatabase,
    addHullToDatabase,
    removeCompoundFromDatabase,
    removeGeometryFromDatabase,
    removeHullFromDatabase,
    type WorldState,
} from "../world/world";
import {
    type CompoundData,
    getCompoundChild,
    getCompoundMaterials,
    MAX_COMPOUND_MESH_MATERIALS,
} from "./compound";
import type { Capsule, MassData, Sphere } from "./geometry";
import { getHeightFieldMaterial, type HeightFieldData } from "./heightfield";
import type { HullData } from "./hull";
import type { Mesh, MeshData } from "./mesh";

/** Min extent (smallest sphere fitting inside) and max extent per axis, for sleeping (b3ShapeExtent). */
export type ShapeExtent = { minExtent: number; maxExtent: Vec3 };

/** Shape identity; the kernel column is its only nongeometry record. */
export type Shape = number;
/** Geometry retained in TypeScript until C2b. */
export type ShapeGeometry = {
    sphere?: Sphere;
    capsule?: Capsule;
    hull?: HullData;
    mesh?: Mesh;
    heightField?: HeightFieldData;
    compound?: CompoundData;
};

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

    let materialIndex = 0;
    if (shapeField(world, shape, ShapeField.type) === ShapeType.Mesh) {
        materialIndex = (world.shapeGeometry[shape].mesh as Mesh).data.materialIndices[
            triangleIndex
        ];
    } else if (shapeField(world, shape, ShapeField.type) === ShapeType.HeightField) {
        materialIndex = getHeightFieldMaterial(
            world.shapeGeometry[shape].heightField as HeightFieldData,
            triangleIndex,
        );
    } else if (shapeField(world, shape, ShapeField.type) === ShapeType.Compound) {
        const child = getCompoundChild(
            world.shapeGeometry[shape].compound as CompoundData,
            childIndex,
        );
        if (child.type === ShapeType.Mesh) {
            const meshMaterialIndex = clampInt(
                (child.mesh as Mesh).data.materialIndices[triangleIndex],
                0,
                MAX_COMPOUND_MESH_MATERIALS - 1,
            );
            materialIndex = child.materialIndices[meshMaterialIndex];
        } else {
            materialIndex = child.materialIndices[0];
        }
    }

    materialIndex = clampInt(materialIndex, 0, materialCount - 1);
    return getShapeMaterials(world, shape)[materialIndex].userMaterialId;
}

// --- geometry dispatch -----------------------------------------------------------------------

/** Mass, center, and inertia of a shape at its density (b3ComputeShapeMass). */
export function computeShapeMass(
    world: WorldState,
    shape: Shape,
    out: MassData = { mass: 0, center: vec3.zero(), inertia: mat3.zero() },
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
    out: ShapeExtent = { minExtent: 0, maxExtent: vec3.zero() },
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
        lowerBound: vec3.zero(),
        upperBound: vec3.zero(),
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
        lowerBound: vec3.zero(),
        upperBound: vec3.zero(),
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
export function getShapeCentroid(world: WorldState, shape: Shape, out: Vec3 = vec3.zero()): Vec3 {
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

function destroyShapeContacts(world: WorldState, shape: Shape, wakeBodies: boolean): void {
    const k = kernel(world.ecsState);
    let key = k.shapeContactNext(world.worldId, shape, -2);
    while (key !== NULL_INDEX) {
        const contact = key >> 1;
        const next = contactField(world, contact, ContactField.nextKeyA + 3 * (key & 1));
        key = k.shapeContactNext(world.worldId, shape, next);
        destroyContact(world, contact, wakeBodies);
    }
}
export function setShapeFilter(world: WorldState, shape: Shape, filter: Filter): void {
    const bodyPoseScratch1 = shapePoseScratch;

    writeShapeFilterValue(world, shape, filter);
    const body = shapeBodyId(world, shape);
    destroyShapeContacts(world, shape, true);
    destroyShapeProxy(shape, world.broadPhase);
    if (bodyField(world, body, BodyField.setIndex) !== SetType.Disabled) {
        createShapeProxy(
            shape,
            world.broadPhase,
            bodyType(world, bodyField(world, body, BodyField.id)),
            readBodyTransform(world, body, bodyPoseScratch1),
            true,
        );
    }
}

export function destroyShapeAllocations(world: WorldState, shape: Shape): void {
    if (shapeField(world, shape, ShapeField.type) === ShapeType.Hull) {
        removeHullFromDatabase(world, world.shapeGeometry[shape].hull as HullData);
        world.shapeGeometry[shape].hull = undefined;
    } else if (shapeField(world, shape, ShapeField.type) === ShapeType.Mesh) {
        removeGeometryFromDatabase(
            world,
            world.meshDatabase,
            world.shapeGeometry[shape].mesh!.data,
        );
        // Keep the inline mesh payload's capacity, but release its caller-owned geometry reference.
        world.shapeGeometry[shape].mesh!.data = undefined as unknown as MeshData;
    } else if (shapeField(world, shape, ShapeField.type) === ShapeType.HeightField) {
        removeGeometryFromDatabase(
            world,
            world.heightFieldDatabase,
            world.shapeGeometry[shape].heightField!,
        );
        world.shapeGeometry[shape].heightField = undefined;
    } else if (shapeField(world, shape, ShapeField.type) === ShapeType.Compound) {
        removeCompoundFromDatabase(world, world.shapeGeometry[shape].compound!);
        world.shapeGeometry[shape].compound = undefined;
    }
    world.shapeStore.destroyMaterials(world, shape);
    world.shapeUserData[shape] = undefined;
    world.shapeNames[shape] = "";
}

// --- create / destroy ------------------------------------------------------------------------
const shapePoseScratch = xf.identity();
const shapeScale = { x: 1, y: 1, z: 1 };

function createShapeInternal(
    world: WorldState,
    body: number,
    bodyTransform: WorldTransform,
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
    world.shapeGeometry[shape] ??= {
        sphere: undefined,
        capsule: undefined,
        hull: undefined,
        mesh: undefined,
        heightField: undefined,
        compound: undefined,
    };

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
    const fields = world.shapeStore.shapeF;
    const g = shape * SHAPE_STRIDE + 2;
    switch (shapeType) {
        case ShapeType.Capsule: {
            const c = (world.shapeGeometry[shape].capsule ??= {
                center1: vec3.zero(),
                center2: vec3.zero(),
                radius: 0,
            });
            copyGeometryVector(fields, g, c.center1);
            copyGeometryVector(fields, g + 3, c.center2);
            c.radius = fields[g + 6];
            break;
        }
        case ShapeType.Sphere: {
            const s = (world.shapeGeometry[shape].sphere ??= { center: vec3.zero(), radius: 0 });
            copyGeometryVector(fields, g, s.center);
            s.radius = fields[g + 3];
            break;
        }
        case ShapeType.Hull:
            world.shapeGeometry[shape].hull = addHullToDatabase(world, geometry as HullData);
            break;
        case ShapeType.Mesh: {
            addGeometryToDatabase(world, world.meshDatabase, geometry as MeshData);
            const mesh = (world.shapeGeometry[shape].mesh ??= {
                data: geometry as MeshData,
                scale: vec3.zero(),
            });
            mesh.data = geometry as MeshData;
            copyGeometryVector(world.shapeStore.shapeF, g, mesh.scale);
            break;
        }
        case ShapeType.HeightField:
            addGeometryToDatabase(world, world.heightFieldDatabase, geometry as HeightFieldData);
            world.shapeGeometry[shape].heightField = geometry as HeightFieldData;
            break;
        case ShapeType.Compound:
            addCompoundToDatabase(world, geometry as CompoundData);
            world.shapeGeometry[shape].compound = geometry as CompoundData;
            break;
        default:
            throw new Error(`physics: unknown shape type ${shapeType}`);
    }

    if (world.geometryDirty) {
        rebuildGeometry(world);
        world.geometryDirty = false;
    }
    writeShapeFilterValue(world, shape, def.filter);
    world.shapeUserData[shape] = def.userData;
    world.shapeNames[shape] = def.name ?? "";
    // The kernel pool advanced the generation at allocation; the public bridge only carries it.

    const materials =
        shapeType === ShapeType.Compound
            ? getCompoundMaterials(world.shapeGeometry[shape].compound as CompoundData)
            : def.materials?.length
              ? def.materials
              : def.baseMaterial;
    world.shapeStore.writeMaterials(world, shape, materials);
    writeShape(world, shape);
    k.shapeFinishGeometry(world.worldId, shape);

    if (bodyField(world, body, BodyField.setIndex) !== SetType.Disabled) {
        // A compound never force-creates pairs: its outer proxy holds no geometry, only children do
        // (b3CreateShapeInternal). The inner tree's proxies are found through the outer query instead.
        const forcePairCreation = def.invokeContactCreation && shapeType !== ShapeType.Compound;
        createShapeProxy(
            shape,
            world.broadPhase,
            bodyType(world, bodyField(world, body, BodyField.id)),
            bodyTransform,
            forcePairCreation,
        );
    }

    kernel(world.ecsState).shapeLink(world.worldId, shape, body);

    if (def.isSensor) createSensor(world, shapeId);

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
    const bodyPoseScratch1 = shapePoseScratch;

    if (!kernel(world.ecsState).shapeCanCreate(world.worldId, body, shapeType)) return null;

    world.locked = true;
    const bodyTransform = readBodyTransform(world, body, bodyPoseScratch1);
    const shape = createShapeInternal(world, body, bodyTransform, def, geometry, shapeType, scale);
    if (shape === null) {
        world.locked = false;
        return null;
    }
    if (def.updateBodyMass) {
        updateBodyMassData(world, body);
    }
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

export function destroyShapeInternal(
    world: WorldState,
    shape: Shape,
    _body: number,
    wakeBodies: boolean,
): void {
    const shapeId = shape;

    kernel(world.ecsState).shapeUnlink(world.worldId, shape);

    destroyShapeProxy(shape, world.broadPhase);

    destroyShapeContacts(world, shape, wakeBodies);

    if (shapeSensorIndex(world, shape) !== NULL_INDEX) {
        destroySensor(world, shape);
    }

    destroyShapeAllocations(world, shape);

    destroyShapeSlot(world, shapeId);
}

export function destroyShape(world: WorldState, shape: Shape, updateBodyMass: boolean): void {
    world.locked = true;
    const body = shapeBodyId(world, shape);
    destroyShapeInternal(world, shape, body, true);
    if (updateBodyMass) {
        updateBodyMassData(world, body);
    }
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
    if (shapeSensorIndex(world, shape) === NULL_INDEX) {
        return [];
    }
    const overlaps = world.sensors[shapeSensorIndex(world, shape)].overlaps2;
    return overlaps.data.slice(0, overlaps.count).map((r) => ({ ...r }));
}
