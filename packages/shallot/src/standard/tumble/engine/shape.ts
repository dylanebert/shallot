// Shapes: geometry attached to a body, with a broad-phase proxy. Ported from Box3D's shape.c (Erin
// Catto, MIT). A shape is its own record in world.shapes (id-pooled, no separate sim); it links to a
// body through a doubly-linked shape list and to the broad-phase through a proxy key.
//
// Sphere/capsule/hull/mesh/height-field/compound shapes are ported: create/destroy, mass/AABB/extent/
// centroid, the proxy, and materials. fround discipline per the README.

import { NULL_INDEX } from "./array";
import { type Body, getBodyTransformQuick, updateBodyMassData } from "./body";
import { syncHeadShape } from "./bodycolumns";
import * as bp from "./broadphase";
import {
    type CompoundData,
    collideMoverAndCompound,
    computeCompoundAABB,
    getCompoundChild,
    getCompoundMaterials,
    MAX_COMPOUND_MESH_MATERIALS,
    overlapCompound,
    rayCastCompound,
    shapeCastCompound,
} from "./compound";
import { destroyContact } from "./contact";
import {
    AABB_MARGIN_FRACTION,
    LINEAR_SLOP,
    MAX_AABB_MARGIN,
    MAX_SHAPE_CAST_POINTS,
    SetType,
    SPECULATIVE_DISTANCE,
} from "./core";
import {
    type CastOutput,
    emptyCastOutput,
    getSweepTransform,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeProxy,
    type Sweep,
} from "./distance";
import { writeFatAabb } from "./fataabbcolumns";
import {
    type Capsule,
    collideMoverAndCapsule,
    collideMoverAndSphere,
    computeCapsuleAABB,
    computeCapsuleAABBOut,
    computeCapsuleMass,
    computeSphereAABB,
    computeSphereAABBOut,
    computeSphereMass,
    computeSweptCapsuleAABB,
    computeSweptSphereAABB,
    type MassData,
    overlapCapsule,
    overlapSphere,
    rayCastCapsule,
    rayCastSphere,
    roundCapsule,
    roundSphere,
    type Sphere,
    shapeCastCapsule,
    shapeCastSphere,
} from "./geometry";
import {
    collideMoverAndHeightField,
    computeHeightFieldAABB,
    getHeightFieldMaterial,
    type HeightFieldData,
    overlapHeightField,
    rayCastHeightField,
    shapeCastHeightField,
} from "./heightfield";
import {
    collideMoverAndHull,
    computeHullAABB,
    computeHullExtent,
    computeHullMass,
    computeSweptHullAABB,
    type HullData,
    overlapHull,
    rayCastHull,
    shapeCastHull,
} from "./hull";
import { allocId, freeId } from "./ids";
import {
    type AABB,
    aabb,
    clampInt,
    f32,
    froundConfig,
    mat3,
    maxf,
    minf,
    minInt,
    quat,
    type Transform,
    type Vec3,
    vec3,
    type WorldTransform,
    xf,
} from "./math";
import {
    collideMoverAndMesh,
    computeMeshAABB,
    type Mesh,
    type MeshData,
    overlapMesh,
    rayCastMesh,
    safeScale,
    shapeCastMesh,
} from "./mesh";
import type { PlaneResult } from "./mover";
import { createSensor, destroySensor, type Visitor } from "./sensor";
import { unlinkShape, writeShape } from "./shapecolumns";
import {
    BodyType,
    cloneMaterial,
    type FilterBits,
    type ShapeDef,
    ShapeType,
    type SurfaceMaterial,
    toFilterBits,
} from "./types";
import { addHullToDatabase, removeHullFromDatabase, type WorldState } from "./world";

/** Min extent (smallest sphere fitting inside) and max extent per axis, for sleeping (b3ShapeExtent). */
export type ShapeExtent = { minExtent: number; maxExtent: Vec3 };

/** A shape record (b3Shape). The geometry union is modeled as one populated optional field by type. */
export type Shape = {
    id: number;
    bodyId: number;
    prevShapeId: number;
    nextShapeId: number;
    sensorIndex: number;
    proxyKey: number;
    type: ShapeType;
    density: number;
    explosionScale: number;
    aabbMargin: number;
    aabb: AABB;
    fatAABB: AABB;
    localCentroid: Vec3;
    material: SurfaceMaterial;
    materialCount: number;
    materials: SurfaceMaterial[] | null;
    filter: FilterBits;
    userData: unknown;
    generation: number;
    enableSensorEvents: boolean;
    enableContactEvents: boolean;
    enableCustomFiltering: boolean;
    enableHitEvents: boolean;
    enablePreSolveEvents: boolean;
    enlargedAABB: boolean;
    sphere?: Sphere;
    capsule?: Capsule;
    hull?: HullData;
    mesh?: Mesh;
    heightField?: HeightFieldData;
    compound?: CompoundData;
};

/** Farthest AABB corner from a point, per axis (b3FarthestPointOnAABB). */
const farthestPointOnAABB = (b: AABB, p: Vec3): Vec3 => ({
    x: f32(p.x - b.lowerBound.x) > f32(b.upperBound.x - p.x) ? b.lowerBound.x : b.upperBound.x,
    y: f32(p.y - b.lowerBound.y) > f32(b.upperBound.y - p.y) ? b.lowerBound.y : b.upperBound.y,
    z: f32(p.z - b.lowerBound.z) > f32(b.upperBound.z - p.z) ? b.lowerBound.z : b.upperBound.z,
});

/**
 * A one-material shape presents its inline material as a length-1 array; multi-material meshes own
 * a heap array. Reach both the same way (b3GetShapeMaterials). Do not cache — the shapes array moves.
 */
export function getShapeMaterials(shape: Shape): SurfaceMaterial[] {
    return shape.materials !== null ? shape.materials : [shape.material];
}

/** The shape's material 0 — what a convex contact mixes — without the fresh single-element array
 * `getShapeMaterials` builds for a one-material shape. */
export function getShapeMaterial(shape: Shape): SurfaceMaterial {
    return shape.materials !== null ? shape.materials[0] : shape.material;
}

/**
 * The user material id at a contact point (b3GetShapeUserMaterialId). Selects the per-triangle
 * material for a mesh/height-field, the child's remapped slot for a compound, else material 0.
 */
export function getShapeUserMaterialId(
    shape: Shape,
    childIndex: number,
    triangleIndex: number,
): bigint {
    if (shape.materialCount === 0) {
        return 0n;
    }

    let materialIndex = 0;
    if (shape.type === ShapeType.Mesh) {
        materialIndex = (shape.mesh as Mesh).data.materialIndices[triangleIndex];
    } else if (shape.type === ShapeType.HeightField) {
        materialIndex = getHeightFieldMaterial(shape.heightField as HeightFieldData, triangleIndex);
    } else if (shape.type === ShapeType.Compound) {
        const child = getCompoundChild(shape.compound as CompoundData, childIndex);
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

    materialIndex = clampInt(materialIndex, 0, shape.materialCount - 1);
    return getShapeMaterials(shape)[materialIndex].userMaterialId;
}

function emptyShape(): Shape {
    return {
        id: NULL_INDEX,
        bodyId: NULL_INDEX,
        prevShapeId: NULL_INDEX,
        nextShapeId: NULL_INDEX,
        sensorIndex: NULL_INDEX,
        proxyKey: NULL_INDEX,
        type: ShapeType.Sphere,
        density: 0,
        explosionScale: 0,
        aabbMargin: 0,
        aabb: { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } },
        fatAABB: { lowerBound: { x: 0, y: 0, z: 0 }, upperBound: { x: 0, y: 0, z: 0 } },
        localCentroid: { x: 0, y: 0, z: 0 },
        material: {
            friction: 0,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: { x: 0, y: 0, z: 0 },
            userMaterialId: 0n,
            customColor: 0,
        },
        materialCount: 0,
        materials: null,
        filter: { categoryHi: 0, categoryLo: 0, maskHi: 0, maskLo: 0, groupIndex: 0 },
        userData: undefined,
        generation: 0,
        enableSensorEvents: false,
        enableContactEvents: false,
        enableCustomFiltering: false,
        enableHitEvents: false,
        enablePreSolveEvents: false,
        enlargedAABB: false,
    };
}

// --- geometry dispatch -----------------------------------------------------------------------

/** Mass, center, and inertia of a shape at its density (b3ComputeShapeMass). */
export function computeShapeMass(shape: Shape): MassData {
    switch (shape.type) {
        case ShapeType.Capsule:
            return computeCapsuleMass(shape.capsule as Capsule, shape.density);
        case ShapeType.Hull:
            return computeHullMass(shape.hull as HullData, shape.density);
        case ShapeType.Sphere:
            return computeSphereMass(shape.sphere as Sphere, shape.density);
        case ShapeType.Mesh:
        case ShapeType.HeightField:
        case ShapeType.Compound:
            // Mesh/height/compound are static-only; they contribute no mass (b3ComputeShapeMass default).
            return { mass: 0, center: vec3.zero(), inertia: mat3.zero() };
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }
}

/** Min/max extent of a shape relative to a local center, for sleeping bounds (b3ComputeShapeExtent). */
export function computeShapeExtent(shape: Shape, localCenter: Vec3): ShapeExtent {
    switch (shape.type) {
        case ShapeType.Capsule: {
            const c = shape.capsule as Capsule;
            const radius = c.radius;
            const c1 = vec3.sub(c.center1, localCenter);
            const c2 = vec3.sub(c.center2, localCenter);
            const r = { x: radius, y: radius, z: radius };
            return { minExtent: radius, maxExtent: vec3.add(vec3.max(c1, c2), r) };
        }
        case ShapeType.Sphere: {
            const s = shape.sphere as Sphere;
            const radius = s.radius;
            const r = { x: radius, y: radius, z: radius };
            const p = vec3.add(vec3.sub(s.center, localCenter), r);
            return { minExtent: radius, maxExtent: vec3.abs(vec3.sub(p, localCenter)) };
        }
        case ShapeType.Hull:
            return computeHullExtent(shape.hull as HullData, localCenter);
        case ShapeType.Mesh: {
            // Needed for kinematic mesh sleeping. Note maxExtent = |farthest corner|, not relative
            // to localCenter (b3ComputeShapeExtent mesh branch — differs from the compound branch).
            const m = shape.mesh as Mesh;
            const box = computeMeshAABB(m.data, xf.identity(), m.scale);
            const r1 = vec3.length(vec3.sub(box.lowerBound, localCenter));
            const r2 = vec3.length(vec3.sub(box.upperBound, localCenter));
            const p = farthestPointOnAABB(box, localCenter);
            return { minExtent: minf(r1, r2), maxExtent: vec3.abs(p) };
        }
        case ShapeType.Compound: {
            // "Shouldn't be needed but here for completeness" (b3ComputeShapeExtent compound branch).
            // Unlike the mesh branch, maxExtent is |farthest corner − localCenter|.
            const box = computeCompoundAABB(shape.compound as CompoundData, xf.identity());
            const r1 = vec3.length(vec3.sub(box.lowerBound, localCenter));
            const r2 = vec3.length(vec3.sub(box.upperBound, localCenter));
            const p = farthestPointOnAABB(box, localCenter);
            return { minExtent: minf(r1, r2), maxExtent: vec3.abs(vec3.sub(p, localCenter)) };
        }
        case ShapeType.HeightField:
            // Height fields are static-only; extent is unused (b3ComputeShapeExtent default → zeros).
            return { minExtent: 0, maxExtent: vec3.zero() };
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }
}

/** Enclosing AABB of a shape under a transform (b3ComputeShapeAABB). */
export function computeShapeAABB(shape: Shape, transform: Transform): AABB {
    switch (shape.type) {
        case ShapeType.Capsule:
            return computeCapsuleAABB(shape.capsule as Capsule, transform);
        case ShapeType.Hull:
            return computeHullAABB(shape.hull as HullData, transform);
        case ShapeType.Sphere:
            return computeSphereAABB(shape.sphere as Sphere, transform);
        case ShapeType.Mesh: {
            const m = shape.mesh as Mesh;
            return computeMeshAABB(m.data, transform, m.scale);
        }
        case ShapeType.HeightField:
            return computeHeightFieldAABB(shape.heightField as HeightFieldData, transform);
        case ShapeType.Compound:
            return computeCompoundAABB(shape.compound as CompoundData, transform);
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }
}

/** Conservative world AABB inflated by `extra` (b3ComputeFatShapeAABB, single-precision path). */
export function computeFatShapeAABB(shape: Shape, transform: WorldTransform, extra: number): AABB {
    const r = { x: extra, y: extra, z: extra };
    const box = computeShapeAABB(shape, transform);
    return { lowerBound: vec3.sub(box.lowerBound, r), upperBound: vec3.add(box.upperBound, r) };
}

/** {@link computeShapeAABB}, written into `o` — identical expression trees. The convex types (the
 * awake-set bulk) run allocation-free; the mesh/height-field/compound tiers fall back to the
 * allocating compute and copy (identity on already-f32 values). */
export function computeShapeAABBOut(shape: Shape, transform: Transform, o: AABB): AABB {
    switch (shape.type) {
        case ShapeType.Capsule:
            return computeCapsuleAABBOut(shape.capsule as Capsule, transform, o);
        case ShapeType.Hull:
            return aabb.transformOut(transform, (shape.hull as HullData).aabb, o);
        case ShapeType.Sphere:
            return computeSphereAABBOut(shape.sphere as Sphere, transform, o);
        default: {
            const box = computeShapeAABB(shape, transform);
            o.lowerBound.x = box.lowerBound.x;
            o.lowerBound.y = box.lowerBound.y;
            o.lowerBound.z = box.lowerBound.z;
            o.upperBound.x = box.upperBound.x;
            o.upperBound.y = box.upperBound.y;
            o.upperBound.z = box.upperBound.z;
            return o;
        }
    }
}

/** {@link computeFatShapeAABB}, written into `o` — identical expression tree, no allocation for
 * the convex types. */
export function computeFatShapeAABBOut(
    shape: Shape,
    transform: WorldTransform,
    extra: number,
    o: AABB,
): AABB {
    computeShapeAABBOut(shape, transform, o);
    o.lowerBound.x = f32(o.lowerBound.x - extra);
    o.lowerBound.y = f32(o.lowerBound.y - extra);
    o.lowerBound.z = f32(o.lowerBound.z - extra);
    o.upperBound.x = f32(o.upperBound.x + extra);
    o.upperBound.y = f32(o.upperBound.y + extra);
    o.upperBound.z = f32(o.upperBound.z + extra);
    return o;
}

/**
 * AABB enclosing a shape swept along `sweep` over [0, time] (b3ComputeSweptShapeAABB). Convex shapes
 * only — the mesh/height/compound target of a sweep is never the moving (fast) shape.
 */
export function computeSweptShapeAABB(shape: Shape, sweep: Sweep, time: number): AABB {
    const xf1: Transform = {
        p: vec3.sub(sweep.c1, quat.rotate(sweep.q1, sweep.localCenter)),
        q: sweep.q1,
    };
    const xf2 = getSweepTransform(sweep, time);
    switch (shape.type) {
        case ShapeType.Capsule:
            return computeSweptCapsuleAABB(shape.capsule as Capsule, xf1, xf2);
        case ShapeType.Hull:
            return computeSweptHullAABB(shape.hull as HullData, xf1, xf2);
        case ShapeType.Sphere:
            return computeSweptSphereAABB(shape.sphere as Sphere, xf1, xf2);
        default:
            throw new Error("tumble: swept AABB of a non-convex fast shape");
    }
}

/**
 * A shape's convex point cloud + rounding radius for GJK/TOI (b3MakeShapeProxy). The points alias the
 * shape's geometry (read-only in the distance path); mesh/height/compound are handled by their own
 * TOI paths, not this proxy.
 */
export function makeShapeProxy(shape: Shape): ShapeProxy {
    switch (shape.type) {
        case ShapeType.Capsule: {
            const c = shape.capsule as Capsule;
            return { points: [c.center1, c.center2], count: 2, radius: c.radius };
        }
        case ShapeType.Sphere: {
            const s = shape.sphere as Sphere;
            return { points: [s.center], count: 1, radius: s.radius };
        }
        case ShapeType.Hull: {
            const hull = shape.hull as HullData;
            return { points: hull.points, count: hull.vertexCount, radius: 0 };
        }
        default:
            // b3MakeShapeProxy asserts false for mesh/height/compound: they are never the moving
            // shape in a proxy/TOI query, so no convex point cloud is ever requested.
            throw new Error("tumble: mesh/height/compound have no shape proxy");
    }
}

/**
 * Ray vs shape, given the shape's world transform (b3RayCastShape). The ray is pulled into the
 * shape's local frame, dispatched, and the hit point/normal lifted back to world.
 */
export function rayCastShape(shape: Shape, transform: Transform, input: RayCastInput): CastOutput {
    const localInput: RayCastInput = {
        origin: xf.invPoint(transform, input.origin),
        translation: quat.invRotate(transform.q, input.translation),
        maxFraction: input.maxFraction,
    };

    let output: CastOutput;
    switch (shape.type) {
        case ShapeType.Capsule:
            output = rayCastCapsule(shape.capsule as Capsule, localInput);
            break;
        case ShapeType.Sphere:
            output = rayCastSphere(shape.sphere as Sphere, localInput);
            break;
        case ShapeType.Hull:
            output = rayCastHull(shape.hull as HullData, localInput);
            break;
        case ShapeType.Compound:
            output = rayCastCompound(shape.compound as CompoundData, localInput);
            break;
        case ShapeType.Mesh:
            output = rayCastMesh(shape.mesh as Mesh, localInput);
            break;
        case ShapeType.HeightField:
            output = rayCastHeightField(shape.heightField as HeightFieldData, localInput);
            break;
        default:
            return emptyCastOutput();
    }

    output.point = xf.point(transform, output.point);
    output.normal = quat.rotate(transform.q, output.normal);
    return output;
}

/**
 * Shape cast (a swept convex proxy) vs shape, given the shape's world transform (b3ShapeCastShape).
 * The proxy points + translation are pulled into the shape's local frame, dispatched, and the hit
 * point/normal lifted back to world.
 */
export function shapeCastShape(
    shape: Shape,
    transform: Transform,
    input: ShapeCastInput,
): CastOutput {
    const count = minInt(input.proxy.count, MAX_SHAPE_CAST_POINTS);
    const localPoints: Vec3[] = new Array(count);
    for (let i = 0; i < count; ++i) {
        localPoints[i] = xf.invPoint(transform, input.proxy.points[i]);
    }
    const localInput: ShapeCastInput = {
        proxy: { points: localPoints, count, radius: input.proxy.radius },
        translation: quat.invRotate(transform.q, input.translation),
        maxFraction: input.maxFraction,
        canEncroach: input.canEncroach,
    };

    let output: CastOutput;
    switch (shape.type) {
        case ShapeType.Capsule:
            output = shapeCastCapsule(shape.capsule as Capsule, localInput);
            break;
        case ShapeType.Sphere:
            output = shapeCastSphere(shape.sphere as Sphere, localInput);
            break;
        case ShapeType.Hull:
            output = shapeCastHull(shape.hull as HullData, localInput);
            break;
        case ShapeType.Compound:
            output = shapeCastCompound(shape.compound as CompoundData, localInput);
            break;
        case ShapeType.HeightField:
            output = shapeCastHeightField(shape.heightField as HeightFieldData, localInput);
            break;
        case ShapeType.Mesh:
            output = shapeCastMesh(shape.mesh as Mesh, localInput);
            break;
        default:
            return emptyCastOutput();
    }

    output.point = xf.point(transform, output.point);
    output.normal = quat.rotate(transform.q, output.normal);
    return output;
}

/** True if `proxy` overlaps the shape, given the shape's world transform (b3OverlapShape). */
export function overlapShape(shape: Shape, transform: Transform, proxy: ShapeProxy): boolean {
    switch (shape.type) {
        case ShapeType.Capsule:
            return overlapCapsule(shape.capsule as Capsule, transform, proxy);
        case ShapeType.Sphere:
            return overlapSphere(shape.sphere as Sphere, transform, proxy);
        case ShapeType.Hull:
            return overlapHull(shape.hull as HullData, transform, proxy);
        case ShapeType.Compound:
            return overlapCompound(shape.compound as CompoundData, transform, proxy);
        case ShapeType.HeightField:
            return overlapHeightField(shape.heightField as HeightFieldData, transform, proxy);
        case ShapeType.Mesh:
            return overlapMesh(shape.mesh as Mesh, transform, proxy);
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }
}

/**
 * Collision planes between a capsule mover and a shape, given the shape's world transform
 * (b3CollideMover). The mover is pulled into the shape's local frame, dispatched, and each resulting
 * plane's normal/point lifted back to world. At most `capacity` planes are returned.
 */
export function collideMover(
    shape: Shape,
    transform: Transform,
    mover: Capsule,
    capacity: number,
): PlaneResult[] {
    if (capacity === 0) {
        return [];
    }

    const localMover: Capsule = {
        center1: xf.invPoint(transform, mover.center1),
        center2: xf.invPoint(transform, mover.center2),
        radius: mover.radius,
    };

    let planes: PlaneResult[];
    switch (shape.type) {
        case ShapeType.Capsule: {
            const p = collideMoverAndCapsule(shape.capsule as Capsule, localMover);
            planes = p ? [p] : [];
            break;
        }
        case ShapeType.Compound:
            planes = collideMoverAndCompound(shape.compound as CompoundData, capacity, localMover);
            break;
        case ShapeType.Sphere: {
            const p = collideMoverAndSphere(shape.sphere as Sphere, localMover);
            planes = p ? [p] : [];
            break;
        }
        case ShapeType.Hull: {
            const p = collideMoverAndHull(shape.hull as HullData, localMover);
            planes = p ? [p] : [];
            break;
        }
        case ShapeType.Mesh:
            planes = collideMoverAndMesh(shape.mesh as Mesh, capacity, localMover);
            break;
        case ShapeType.HeightField:
            planes = collideMoverAndHeightField(
                shape.heightField as HeightFieldData,
                capacity,
                localMover,
            );
            break;
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }

    for (const pr of planes) {
        pr.plane.normal = quat.rotate(transform.q, pr.plane.normal);
        pr.point = xf.point(transform, pr.point);
    }

    return planes;
}

/** Local centroid of a shape (b3GetShapeCentroid). */
export function getShapeCentroid(shape: Shape): Vec3 {
    switch (shape.type) {
        case ShapeType.Capsule: {
            const c = shape.capsule as Capsule;
            return vec3.lerp(c.center1, c.center2, f32(0.5));
        }
        case ShapeType.Sphere:
            return (shape.sphere as Sphere).center;
        case ShapeType.Hull:
            return (shape.hull as HullData).center;
        case ShapeType.Mesh: {
            const m = shape.mesh as Mesh;
            return aabb.center(computeMeshAABB(m.data, xf.identity(), m.scale));
        }
        case ShapeType.HeightField:
            return aabb.center(
                computeHeightFieldAABB(shape.heightField as HeightFieldData, xf.identity()),
            );
        case ShapeType.Compound:
            return aabb.center(computeCompoundAABB(shape.compound as CompoundData, xf.identity()));
        default:
            throw new Error(`tumble: unknown shape type ${shape.type}`);
    }
}

function computeShapeMargin(shape: Shape): number {
    let margin = 0;
    switch (shape.type) {
        case ShapeType.Sphere:
            margin = (shape.sphere as Sphere).radius;
            break;
        case ShapeType.Capsule: {
            const c = shape.capsule as Capsule;
            margin = f32(f32(0.5 * vec3.distance(c.center2, c.center1)) + c.radius);
            break;
        }
        case ShapeType.Hull: {
            const hull = shape.hull as HullData;
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

function updateShapeAABBs(shape: Shape, transform: WorldTransform, proxyType: BodyType): void {
    const speculativeDistance = SPECULATIVE_DISTANCE;
    const aabbMargin = shape.aabbMargin;

    const box = computeFatShapeAABB(shape, transform, speculativeDistance);
    shape.aabb = box;

    // Smaller margin for static bodies. Cannot be zero due to TOI tolerance.
    const margin = proxyType === BodyType.Static ? speculativeDistance : aabbMargin;
    shape.fatAABB = {
        lowerBound: {
            x: f32(box.lowerBound.x - margin),
            y: f32(box.lowerBound.y - margin),
            z: f32(box.lowerBound.z - margin),
        },
        upperBound: {
            x: f32(box.upperBound.x + margin),
            y: f32(box.upperBound.y + margin),
            z: f32(box.upperBound.z + margin),
        },
    };
}

export function createShapeProxy(
    shape: Shape,
    broadPhase: bp.BroadPhase,
    type: BodyType,
    transform: WorldTransform,
    forcePairCreation: boolean,
): void {
    updateShapeAABBs(shape, transform, type);
    shape.proxyKey = bp.createProxy(
        broadPhase,
        type as bp.BodyTypeValue,
        shape.fatAABB,
        shape.filter.categoryHi,
        shape.filter.categoryLo,
        shape.id,
        forcePairCreation,
    );
}

export function destroyShapeProxy(shape: Shape, broadPhase: bp.BroadPhase): void {
    if (shape.proxyKey !== NULL_INDEX) {
        bp.destroyProxy(broadPhase, shape.proxyKey);
        shape.proxyKey = NULL_INDEX;
    }
}

export function destroyShapeAllocations(world: WorldState, shape: Shape): void {
    if (shape.type === ShapeType.Hull) {
        removeHullFromDatabase(world, shape.hull as HullData);
        shape.hull = undefined;
    }
    if (shape.materials !== null) {
        shape.materials = null;
        shape.materialCount = 0;
    }
}

// --- create / destroy ------------------------------------------------------------------------

function createShapeInternal(
    world: WorldState,
    body: Body,
    bodyTransform: WorldTransform,
    def: ShapeDef,
    geometry: Sphere | Capsule | HullData | MeshData | HeightFieldData | CompoundData,
    shapeType: ShapeType,
    scale: Vec3,
): Shape | null {
    // Round every user float to f32 once at ingress (density/explosionScale/material floats); the C def
    // is f32, so an unrounded f64 scalar would reach mass/solve and break bit-exact parity. Filter
    // category/mask bigints and enum/bool fields pass through untouched.
    def = froundConfig(def);
    const shapeId = allocId(world.shapeIdPool);
    if (shapeId === world.shapes.length) {
        world.shapes.push(emptyShape());
    }

    const shape = world.shapes[shapeId];
    const generation = shape.generation;
    Object.assign(shape, emptyShape());
    shape.generation = generation;

    switch (shapeType) {
        case ShapeType.Capsule:
            shape.capsule = roundCapsule(geometry as Capsule);
            break;
        case ShapeType.Sphere:
            shape.sphere = roundSphere(geometry as Sphere);
            break;
        case ShapeType.Hull:
            shape.hull = addHullToDatabase(world, geometry as HullData);
            break;
        case ShapeType.Mesh:
            // The mesh data is caller-owned and shared (not cloned into a database like hulls).
            shape.mesh = { data: geometry as MeshData, scale: safeScale(scale) };
            break;
        case ShapeType.HeightField:
            // Height-field data is caller-owned and shared (no clone; scale is baked into the data).
            shape.heightField = geometry as HeightFieldData;
            break;
        case ShapeType.Compound:
            // Compound data is caller-owned and shared; the materials are cloned below.
            shape.compound = geometry as CompoundData;
            break;
        default:
            throw new Error(`tumble: unknown shape type ${shapeType}`);
    }

    shape.id = shapeId;
    shape.bodyId = body.id;
    shape.type = shapeType;
    shape.density = def.density;
    shape.explosionScale = def.explosionScale;
    shape.filter = toFilterBits(def.filter);
    shape.userData = def.userData;
    shape.enlargedAABB = false;
    shape.enableSensorEvents = def.enableSensorEvents;
    shape.enableContactEvents = def.enableContactEvents;
    shape.enableCustomFiltering = def.enableCustomFiltering;
    shape.enableHitEvents = def.enableHitEvents;
    shape.enablePreSolveEvents = def.enablePreSolveEvents;
    shape.proxyKey = NULL_INDEX;
    shape.localCentroid = getShapeCentroid(shape);
    shape.aabbMargin = computeShapeMargin(shape);
    shape.generation = generation + 1;

    const materialCount = def.materials ? def.materials.length : 0;
    if (shapeType === ShapeType.Compound) {
        // Own a copy of the compound's materials so every shape frees its array the same way; the
        // per-child material indices resolve against this array (b3CreateShapeInternal compound branch).
        const mats = getCompoundMaterials(shape.compound as CompoundData);
        shape.materialCount = mats.length;
        shape.materials = mats.map(cloneMaterial);
    } else if (materialCount > 1 && def.materials) {
        shape.materialCount = materialCount;
        shape.materials = def.materials.map(cloneMaterial);
    } else {
        shape.material =
            materialCount === 1 && def.materials
                ? cloneMaterial(def.materials[0])
                : cloneMaterial(def.baseMaterial);
        shape.materialCount = 1;
        shape.materials = null;
    }

    if (body.setIndex !== SetType.Disabled) {
        // A compound never force-creates pairs: its outer proxy holds no geometry, only children do
        // (b3CreateShapeInternal). The inner tree's proxies are found through the outer query instead.
        const forcePairCreation = def.invokeContactCreation && shapeType !== ShapeType.Compound;
        createShapeProxy(shape, world.broadPhase, body.type, bodyTransform, forcePairCreation);
        writeFatAabb(world, shape);
    }

    // Add to the body's shape doubly-linked list at the head
    if (body.headShapeId !== NULL_INDEX) {
        world.shapes[body.headShapeId].prevShapeId = shapeId;
    }
    shape.prevShapeId = NULL_INDEX;
    shape.nextShapeId = body.headShapeId;
    body.headShapeId = shapeId;
    body.shapeCount += 1;

    // Mirror the shape into the resident column (type + geometry + its new `next`) and re-point the
    // body's resident head lane at it. Only the new shape's `next` changes — a head insert leaves every
    // other record's link alone.
    writeShape(world, shape);
    syncHeadShape(world, body);

    if (def.isSensor) {
        shape.sensorIndex = world.sensors.length;
        world.sensors.push(createSensor(shapeId));
    } else {
        shape.sensorIndex = NULL_INDEX;
    }

    return shape;
}

function createShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    geometry: Sphere | Capsule | HullData | MeshData | HeightFieldData | CompoundData,
    shapeType: ShapeType,
    scale: Vec3 = { x: 1, y: 1, z: 1 },
): Shape | null {
    // Compound and height-field shapes must be on static bodies (b3CreateShape). They carry no mass,
    // so a dynamic body with one would have zero mass and blow up; the C returns null here.
    if (
        body.type !== BodyType.Static &&
        (shapeType === ShapeType.Compound || shapeType === ShapeType.HeightField)
    ) {
        return null;
    }

    world.locked = true;
    const bodyTransform = getBodyTransformQuick(world, body);
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
    body: Body,
    def: ShapeDef,
    sphere: Sphere,
): Shape | null {
    return createShape(world, body, def, sphere, ShapeType.Sphere);
}

export function createCapsuleShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    capsuleInput: Capsule,
): Shape | null {
    // Round to f32 up front so the degenerate test matches the C's float-input path (b3CreateCapsuleShape).
    const capsule = roundCapsule(capsuleInput);
    // A degenerate capsule collapses to a sphere at its midpoint (matches b3CreateCapsuleShape).
    const lengthSqr = vec3.distanceSquared(capsule.center1, capsule.center2);
    if (lengthSqr <= f32(LINEAR_SLOP * LINEAR_SLOP)) {
        const sphere: Sphere = {
            center: vec3.lerp(capsule.center1, capsule.center2, f32(0.5)),
            radius: capsule.radius,
        };
        return createShape(world, body, def, sphere, ShapeType.Sphere);
    }
    return createShape(world, body, def, capsule, ShapeType.Capsule);
}

export function createHullShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    hull: HullData,
): Shape | null {
    return createShape(world, body, def, hull, ShapeType.Hull);
}

export function createMeshShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    mesh: MeshData,
    scale: Vec3,
): Shape | null {
    return createShape(world, body, def, mesh, ShapeType.Mesh, scale);
}

export function createHeightFieldShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    heightField: HeightFieldData,
): Shape | null {
    return createShape(world, body, def, heightField, ShapeType.HeightField);
}

export function createCompoundShape(
    world: WorldState,
    body: Body,
    def: ShapeDef,
    compound: CompoundData,
): Shape | null {
    return createShape(world, body, def, compound, ShapeType.Compound);
}

export function destroyShapeInternal(
    world: WorldState,
    shape: Shape,
    body: Body,
    wakeBodies: boolean,
): void {
    const shapeId = shape.id;

    // Unlink from the body's shape list
    if (shape.prevShapeId !== NULL_INDEX) {
        world.shapes[shape.prevShapeId].nextShapeId = shape.nextShapeId;
    }
    if (shape.nextShapeId !== NULL_INDEX) {
        world.shapes[shape.nextShapeId].prevShapeId = shape.prevShapeId;
    }
    if (shapeId === body.headShapeId) {
        body.headShapeId = shape.nextShapeId;
    }
    body.shapeCount -= 1;

    // Mirror the unlink into the resident column: the predecessor's `next` slot and the body's head
    // lane. The destroyed shape's own record is stale from here — its id is freed, so no chain reaches
    // it, and a create recycling the id rewrites every slot.
    unlinkShape(world, shape);
    syncHeadShape(world, body);

    destroyShapeProxy(shape, world.broadPhase);

    // Destroy contacts referencing this shape
    let contactKey = body.headContactKey;
    while (contactKey !== NULL_INDEX) {
        const contactId = contactKey >> 1;
        const edgeIndex = contactKey & 1;
        const contact = world.contacts[contactId];
        contactKey = contact.edges[edgeIndex].nextKey;
        if (contact.shapeIdA === shapeId || contact.shapeIdB === shapeId) {
            destroyContact(world, contact, wakeBodies);
        }
    }

    if (shape.sensorIndex !== NULL_INDEX) {
        destroySensor(world, shape);
    }

    destroyShapeAllocations(world, shape);

    freeId(world.shapeIdPool, shapeId);
    shape.id = NULL_INDEX;
}

export function destroyShape(world: WorldState, shape: Shape, updateBodyMass: boolean): void {
    world.locked = true;
    const body = world.bodies[shape.bodyId];
    destroyShapeInternal(world, shape, body, true);
    if (updateBodyMass) {
        updateBodyMassData(world, body);
    }
    world.locked = false;
}

/** Whether a shape is a sensor (b3Shape_IsSensor). */
export function isSensorShape(shape: Shape): boolean {
    return shape.sensorIndex !== NULL_INDEX;
}

/**
 * The shapes currently overlapping a sensor (b3Shape_GetSensorData). Returns a fresh copy of the
 * sensor's current-frame overlaps; empty if the shape is not a sensor.
 */
export function getSensorData(world: WorldState, shape: Shape): Visitor[] {
    if (shape.sensorIndex === NULL_INDEX) {
        return [];
    }
    return world.sensors[shape.sensorIndex].overlaps2.map((r) => ({ ...r }));
}
