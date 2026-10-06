import { ContactField, contactField } from "../collision/contact";
import { BodyField, bodyField } from "../kernel/bodyrecords";
import {
    bodyType,
    shapeBodyId,
    shapeSensorIndex,
    writeShapeFilterValue,
} from "../kernel/filtercolumns";
import { ShapeField, shapeField } from "../kernel/shaperecords";
// Shapes: geometry attached to a body, with a broad-phase proxy. Ported from Box3D's shape.c (Erin
// Catto, MIT). A shape's nongeometry record lives in the kernel's id-pooled column; it links to a
// body through a doubly-linked shape list and to the broad-phase through a proxy key.
//
// Sphere/capsule/hull/mesh/height-field/compound shapes are ported: create/destroy, mass/AABB/extent/
// centroid, the proxy, and materials. fround discipline (every f32 result rounds through Math.fround, keeping bit-exact f32 parity).

import type * as bp from "../collision/broadphase";
import { destroyContact } from "../collision/contact";
import { NULL_INDEX } from "../common/array";
import { AABB_MARGIN_FRACTION, MAX_AABB_MARGIN, SetType } from "../common/constants";
import {
    type AABB,
    aabb,
    clampInt,
    f32,
    mat3,
    maxf,
    minf,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
    xf,
} from "../common/math";
import {
    BodyType,
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
    computeCompoundAABB,
    getCompoundChild,
    getCompoundMaterials,
    MAX_COMPOUND_MESH_MATERIALS,
} from "./compound";
import {
    type Capsule,
    computeCapsuleAABB,
    computeCapsuleAABBOut,
    computeCapsuleMassOut,
    computeSphereAABB,
    computeSphereAABBOut,
    computeSphereMassOut,
    type MassData,
    roundCapsule,
    roundSphere,
    type Sphere,
} from "./geometry";
import {
    computeHeightFieldAABB,
    getHeightFieldMaterial,
    type HeightFieldData,
} from "./heightfield";
import { computeHullAABB, computeHullExtent, computeHullMass, type HullData } from "./hull";
import { computeMeshAABB, type Mesh, type MeshData, safeScale } from "./mesh";

/** Min extent (smallest sphere fitting inside) and max extent per axis, for sleeping (b3ShapeExtent). */
export type ShapeExtent = { minExtent: number; maxExtent: Vec3 };

/** Shape identity; the kernel column is its only nongeometry record. */
export type Shape = number;
/** Geometry retained in TypeScript until C2. */
export type ShapeGeometry = {
    sphere?: Sphere;
    capsule?: Capsule;
    hull?: HullData;
    mesh?: Mesh;
    heightField?: HeightFieldData;
    compound?: CompoundData;
};

/**
 * A one-material shape presents its inline material as a length-1 array; multi-material meshes own
 * a heap array. This requested observation returns an independent material array (b3GetShapeMaterials).
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
    const k = kernel(world.ecsState);
    k.shapeSetActiveWorld(world.worldId);
    const ptr = k.shapeMaterialPtr(world.worldId, shape);
    const count = k.shapeMaterialCount(world.worldId, shape) >>> 0;
    if (count === 0) throw new Error(`physics: no material on shape ${shape}`);
    const u = world.shapeStore.materialU;
    const f = world.shapeStore.materialF;
    const o = ptr / 4;
    out.friction = f[o];
    out.restitution = f[o + 1];
    out.rollingResistance = f[o + 2];
    out.tangentVelocity.x = f[o + 3];
    out.tangentVelocity.y = f[o + 4];
    out.tangentVelocity.z = f[o + 5];
    out.userMaterialId = world.shapeStore.userMaterialId(o);
    out.customColor = u[o + 8];
    return out;
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

const massDensity = { density: 0 };
/** Mass, center, and inertia of a shape at its density (b3ComputeShapeMass). */
export function computeShapeMass(
    world: WorldState,
    shape: Shape,
    out: MassData = { mass: 0, center: vec3.zero(), inertia: mat3.zero() },
): MassData {
    const fields = world.shapeStore.shapeF;
    massDensity.density = fields[shape * SHAPE_STRIDE + ShapeField.density];
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Capsule:
            return computeCapsuleMassOut(
                world.shapeGeometry[shape].capsule as Capsule,
                massDensity,
                out,
            );
        case ShapeType.Hull:
            return computeHullMass(
                world.shapeGeometry[shape].hull as HullData,
                fields[shape * SHAPE_STRIDE + ShapeField.density],
                out,
            );
        case ShapeType.Sphere:
            return computeSphereMassOut(
                world.shapeGeometry[shape].sphere as Sphere,
                massDensity,
                out,
            );
        case ShapeType.Mesh:
        case ShapeType.HeightField:
        case ShapeType.Compound:
            // Mesh/height/compound are static-only; they contribute no mass (b3ComputeShapeMass default).
            out.mass = 0;
            out.center.x = out.center.y = out.center.z = 0;
            out.inertia.cx.x =
                out.inertia.cx.y =
                out.inertia.cx.z =
                out.inertia.cy.x =
                out.inertia.cy.y =
                out.inertia.cy.z =
                out.inertia.cz.x =
                out.inertia.cz.y =
                out.inertia.cz.z =
                    0;
            return out;
        default:
            throw new Error(
                `physics: unknown shape type ${shapeField(world, shape, ShapeField.type)}`,
            );
    }
}

/** Min/max extent of a shape relative to a local center, for sleeping bounds (b3ComputeShapeExtent). */
export function computeShapeExtent(
    world: WorldState,
    shape: Shape,
    localCenter: Vec3,
    out: ShapeExtent = { minExtent: 0, maxExtent: vec3.zero() },
): ShapeExtent {
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Capsule: {
            const c = world.shapeGeometry[shape].capsule as Capsule;
            const radius = c.radius;
            out.minExtent = radius;
            out.maxExtent.x = f32(
                maxf(f32(c.center1.x - localCenter.x), f32(c.center2.x - localCenter.x)) + radius,
            );
            out.maxExtent.y = f32(
                maxf(f32(c.center1.y - localCenter.y), f32(c.center2.y - localCenter.y)) + radius,
            );
            out.maxExtent.z = f32(
                maxf(f32(c.center1.z - localCenter.z), f32(c.center2.z - localCenter.z)) + radius,
            );
            return out;
        }
        case ShapeType.Sphere: {
            const s = world.shapeGeometry[shape].sphere as Sphere;
            const radius = s.radius;
            out.minExtent = radius;
            out.maxExtent.x = Math.abs(
                f32(f32(f32(s.center.x - localCenter.x) + radius) - localCenter.x),
            );
            out.maxExtent.y = Math.abs(
                f32(f32(f32(s.center.y - localCenter.y) + radius) - localCenter.y),
            );
            out.maxExtent.z = Math.abs(
                f32(f32(f32(s.center.z - localCenter.z) + radius) - localCenter.z),
            );
            return out;
        }
        case ShapeType.Hull:
            return computeHullExtent(world.shapeGeometry[shape].hull as HullData, localCenter, out);
        case ShapeType.Mesh:
        case ShapeType.Compound: {
            computeShapeAABBOut(world, shape, localIdentity, localBounds);
            vec3.subOut(localBounds.lowerBound, localCenter, extentDifference);
            const r1 = vec3.length(extentDifference);
            vec3.subOut(localBounds.upperBound, localCenter, extentDifference);
            const r2 = vec3.length(extentDifference);
            out.minExtent = minf(r1, r2);
            const lo = localBounds.lowerBound,
                hi = localBounds.upperBound;
            let x = f32(localCenter.x - lo.x) > f32(hi.x - localCenter.x) ? lo.x : hi.x;
            let y = f32(localCenter.y - lo.y) > f32(hi.y - localCenter.y) ? lo.y : hi.y;
            let z = f32(localCenter.z - lo.z) > f32(hi.z - localCenter.z) ? lo.z : hi.z;
            // Box3D's mesh extent is absolute; its compound extent is relative to the local center.
            if (shapeField(world, shape, ShapeField.type) === ShapeType.Compound) {
                x = f32(x - localCenter.x);
                y = f32(y - localCenter.y);
                z = f32(z - localCenter.z);
            }
            out.maxExtent.x = Math.abs(x);
            out.maxExtent.y = Math.abs(y);
            out.maxExtent.z = Math.abs(z);
            return out;
        }
        case ShapeType.HeightField:
            // Height fields are static-only; extent is unused (b3ComputeShapeExtent default → zeros).
            out.minExtent = 0;
            out.maxExtent.x = out.maxExtent.y = out.maxExtent.z = 0;
            return out;
        default:
            throw new Error(
                `physics: unknown shape type ${shapeField(world, shape, ShapeField.type)}`,
            );
    }
}

/** Enclosing AABB of a shape under a transform (b3ComputeShapeAABB). */
export function computeShapeAABB(world: WorldState, shape: Shape, transform: Transform): AABB {
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Capsule:
            return computeCapsuleAABB(world.shapeGeometry[shape].capsule as Capsule, transform);
        case ShapeType.Hull:
            return computeHullAABB(world.shapeGeometry[shape].hull as HullData, transform);
        case ShapeType.Sphere:
            return computeSphereAABB(world.shapeGeometry[shape].sphere as Sphere, transform);
        case ShapeType.Mesh: {
            const m = world.shapeGeometry[shape].mesh as Mesh;
            return computeMeshAABB(m.data, transform, m.scale);
        }
        case ShapeType.HeightField:
            return computeHeightFieldAABB(
                world.shapeGeometry[shape].heightField as HeightFieldData,
                transform,
            );
        case ShapeType.Compound:
            return computeCompoundAABB(
                world.shapeGeometry[shape].compound as CompoundData,
                transform,
            );
        default:
            throw new Error(
                `physics: unknown shape type ${shapeField(world, shape, ShapeField.type)}`,
            );
    }
}

/** Conservative world AABB inflated by `extra` (b3ComputeFatShapeAABB, single-precision path). */
export function computeFatShapeAABB(
    world: WorldState,
    shape: Shape,
    transform: WorldTransform,
    extra: number,
): AABB {
    const r = { x: extra, y: extra, z: extra };
    const box = computeShapeAABB(world, shape, transform);
    return { lowerBound: vec3.sub(box.lowerBound, r), upperBound: vec3.add(box.upperBound, r) };
}

/** {@link computeShapeAABB}, written into `o` — identical expression trees. The convex types (the
 * awake-set bulk) run allocation-free; the mesh/height-field/compound tiers fall back to the
 * allocating compute and copy (identity on already-f32 values). */
export function computeShapeAABBOut(
    world: WorldState,
    shape: Shape,
    transform: Transform,
    o: AABB,
): AABB {
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Capsule:
            return computeCapsuleAABBOut(
                world.shapeGeometry[shape].capsule as Capsule,
                transform,
                o,
            );
        case ShapeType.Hull:
            return aabb.transformOut(
                transform,
                (world.shapeGeometry[shape].hull as HullData).aabb,
                o,
            );
        case ShapeType.Sphere:
            return computeSphereAABBOut(world.shapeGeometry[shape].sphere as Sphere, transform, o);
        case ShapeType.Mesh: {
            const mesh = world.shapeGeometry[shape].mesh!;
            return computeMeshAABB(mesh.data, transform, mesh.scale, o);
        }
        case ShapeType.HeightField:
            return computeHeightFieldAABB(world.shapeGeometry[shape].heightField!, transform, o);
        case ShapeType.Compound:
            return computeCompoundAABB(world.shapeGeometry[shape].compound!, transform, o);
        default:
            throw new Error(
                `physics: unknown shape type ${shapeField(world, shape, ShapeField.type)}`,
            );
    }
}

/** {@link computeFatShapeAABB}, written into `o` — identical expression tree, no allocation for
 * the convex types. */
export function computeFatShapeAABBOut(
    world: WorldState,
    shape: Shape,
    transform: WorldTransform,
    extra: number,
    o: AABB,
): AABB {
    computeShapeAABBOut(world, shape, transform, o);
    o.lowerBound.x = f32(o.lowerBound.x - extra);
    o.lowerBound.y = f32(o.lowerBound.y - extra);
    o.lowerBound.z = f32(o.lowerBound.z - extra);
    o.upperBound.x = f32(o.upperBound.x + extra);
    o.upperBound.y = f32(o.upperBound.y + extra);
    o.upperBound.z = f32(o.upperBound.z + extra);
    return o;
}

const extentDifference = vec3.zero();
const localIdentity = xf.identity();
const localBounds = { lowerBound: vec3.zero(), upperBound: vec3.zero() };
/** Local centroid of a shape (b3GetShapeCentroid). */
export function getShapeCentroid(world: WorldState, shape: Shape, out: Vec3 = vec3.zero()): Vec3 {
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Capsule: {
            const c = world.shapeGeometry[shape].capsule as Capsule;
            return vec3.lerpOut(c.center1, c.center2, f32(0.5), out);
        }
        case ShapeType.Sphere:
            return (world.shapeGeometry[shape].sphere as Sphere).center;
        case ShapeType.Hull:
            return (world.shapeGeometry[shape].hull as HullData).center;
        case ShapeType.Mesh:
        case ShapeType.HeightField:
        case ShapeType.Compound:
            computeShapeAABBOut(world, shape, localIdentity, localBounds);
            vec3.addOut(localBounds.lowerBound, localBounds.upperBound, out);
            return vec3.scaleOut(0.5, out, out);
        default:
            throw new Error(
                `physics: unknown shape type ${shapeField(world, shape, ShapeField.type)}`,
            );
    }
}

function computeShapeMargin(world: WorldState, shape: Shape): number {
    let margin = 0;
    switch (shapeField(world, shape, ShapeField.type)) {
        case ShapeType.Sphere:
            margin = (world.shapeGeometry[shape].sphere as Sphere).radius;
            break;
        case ShapeType.Capsule: {
            const c = world.shapeGeometry[shape].capsule as Capsule;
            margin = f32(f32(0.5 * vec3.distance(c.center2, c.center1)) + c.radius);
            break;
        }
        case ShapeType.Hull: {
            const hull = world.shapeGeometry[shape].hull as HullData;
            let maxExtentSqr = 0;
            for (let i = 0; i < hull.vertexCount; ++i) {
                maxExtentSqr = maxf(
                    maxExtentSqr,
                    vec3.distanceSquared(hull.points[i], hull.center),
                );
            }
            margin = f32(Math.sqrt(maxExtentSqr));
            break;
        }
        default:
            // Static-only shapes use speculative distance for their proxies; the per-shape margin
            // is never consumed. Return the cap.
            return MAX_AABB_MARGIN;
    }
    return minf(MAX_AABB_MARGIN, f32(AABB_MARGIN_FRACTION * margin));
}

// --- proxy -----------------------------------------------------------------------------------

const proxyBounds = { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } };
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
    computeShapeAABBOut(world, shape, transform, proxyBounds);
    kernel(world.ecsState).shapeCreateProxyBounds(
        world.worldId,
        shape,
        Number(forcePairCreation),
        proxyBounds.lowerBound.x,
        proxyBounds.lowerBound.y,
        proxyBounds.lowerBound.z,
        proxyBounds.upperBound.x,
        proxyBounds.upperBound.y,
        proxyBounds.upperBound.z,
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
const centroidScratch = vec3.zero();
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

    switch (shapeType) {
        case ShapeType.Capsule:
            world.shapeGeometry[shape].capsule = roundCapsule(
                geometry as Capsule,
                world.shapeGeometry[shape].capsule,
            );
            break;
        case ShapeType.Sphere:
            world.shapeGeometry[shape].sphere = roundSphere(
                geometry as Sphere,
                world.shapeGeometry[shape].sphere,
            );
            break;
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
            safeScale(scale, mesh.scale);
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

    writeShapeFilterValue(world, shape, def.filter);
    world.shapeUserData[shape] = def.userData;
    world.shapeNames[shape] = def.name ?? "";
    const centroid = getShapeCentroid(world, shape, centroidScratch);
    const report = world.shapeStore.geometryInput;
    report[0] = centroid.x;
    report[1] = centroid.y;
    report[2] = centroid.z;
    report[3] = computeShapeMargin(world, shape);
    kernel(world.ecsState).shapeFinishGeometry(world.worldId, shape);
    // The kernel pool advanced the generation at allocation; the public bridge only carries it.

    const materials =
        shapeType === ShapeType.Compound
            ? getCompoundMaterials(world.shapeGeometry[shape].compound as CompoundData)
            : def.materials?.length
              ? def.materials
              : def.baseMaterial;
    world.shapeStore.writeMaterials(world, shape, materials);
    writeShape(world, shape);

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

    // Compound and height-field shapes must be on static bodies (b3CreateShape). They carry no mass,
    // so a dynamic body with one would have zero mass and blow up; the C returns null here.
    if (
        bodyType(world, bodyField(world, body, BodyField.id)) !== BodyType.Static &&
        (shapeType === ShapeType.Compound || shapeType === ShapeType.HeightField)
    ) {
        return null;
    }

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

const capsuleInputScratch = { center1: vec3.zero(), center2: vec3.zero(), radius: 0 };
const sphereInputScratch = { center: vec3.zero(), radius: 0 };
export function createCapsuleShape(
    world: WorldState,
    body: number,
    def: ShapeDef,
    capsuleInput: Capsule,
): Shape | null {
    // Round to f32 up front so the degenerate test matches the C's float-input path (b3CreateCapsuleShape).
    const capsule = roundCapsule(capsuleInput, capsuleInputScratch);
    // A degenerate capsule collapses to a sphere at its midpoint (matches b3CreateCapsuleShape).
    const kind = kernel(world.ecsState).shapeCapsuleType(
        capsule.center1.x,
        capsule.center1.y,
        capsule.center1.z,
        capsule.center2.x,
        capsule.center2.y,
        capsule.center2.z,
    );
    if (kind === ShapeType.Sphere) {
        const sphere = sphereInputScratch;
        vec3.lerpOut(capsule.center1, capsule.center2, f32(0.5), sphere.center);
        sphere.radius = capsule.radius;
        return createShape(world, body, def, sphere, ShapeType.Sphere);
    }
    return createShape(world, body, def, capsule, ShapeType.Capsule);
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
