// Compound shapes — a static container of child shapes (capsule/hull/mesh/sphere) baked into one
// broad-phase proxy with an inner dynamic tree. Ported from Box3D's compound.c (Erin Catto, MIT).
//
// The C packs the whole compound into one byte blob accessed through offsets; the port models it as a
// plain struct of arrays (mirroring the mesh/height-field batches). Safe because the raw compound
// bytes are never hashed by the sim — only the inner-tree build order (→ child query order → contact
// ids → solver order), the child transforms, and the per-child material indices are load-bearing, all
// preserved here. So `byteCount`/`version`/the offset machinery are dropped; `sharedHullCount` and
// `sharedMeshCount` are kept as the diagnostics the create tests assert.
//
// Materials are de-duplicated in first-seen order across all child types (capsules, then hulls, then
// meshes, then spheres) — a linear find-or-append that yields the same integer indices as the C's
// hash map. Shared hulls de-dup by content hash (b3HashHullData analogue); shared meshes de-dup by
// structural content (b3CompareMeshes, which memcmp's the mesh blob — a deep field compare here, since
// this port dropped the mesh's serialization hash). The shared counts are diagnostics the create tests
// assert; the child instances keep their own geometry references either way.

import { ALL_BITS_HI, ALL_BITS_LO, MAX_SHAPE_CAST_POINTS } from "./core";
import {
    type CastOutput,
    computeProxyAABB,
    emptyCastOutput,
    type RayCastInput,
    type ShapeCastInput,
    type ShapeProxy,
} from "./distance";
import {
    type Capsule,
    collideMoverAndCapsule,
    collideMoverAndSphere,
    computeCapsuleAABB,
    computeSphereAABB,
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
    collideMoverAndHull,
    computeHullAABB,
    type HullData,
    overlapHull,
    rayCastHull,
    shapeCastHull,
} from "./hull";
import { type AABB, aabb, mat3, minInt, quat, type Transform, type Vec3, vec3, xf } from "./math";
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
import * as tree from "./tree";
import { cloneMaterial, ShapeType, type SurfaceMaterial } from "./types";

// A compound mesh child has a fixed number of material slots (B3_MAX_COMPOUND_MESH_MATERIALS); a
// triangle's material index is clamped into this range, then remapped through the child's slots.
export const MAX_COMPOUND_MESH_MATERIALS = 4;

// B3_MAX_CHILD_SHAPES = 1 << B3_CHILD_POWER, B3_CHILD_POWER = 64 - 2 * B3_SHAPE_POWER (22) = 20.
const MAX_CHILD_SHAPES = 1 << 20;

// --- def structs (b3Compound*Def) -----------------------------------------------------------

export type CompoundCapsuleDef = { capsule: Capsule; material: SurfaceMaterial };
export type CompoundHullDef = { hull: HullData; transform: Transform; material: SurfaceMaterial };
export type CompoundMeshDef = {
    meshData: MeshData;
    transform: Transform;
    scale: Vec3;
    materials: SurfaceMaterial[];
    materialCount: number;
};
export type CompoundSphereDef = { sphere: Sphere; material: SurfaceMaterial };

/** Definition for a compound shape. All data is cloned into the runtime compound (b3CompoundDef). */
export type CompoundDef = {
    capsules?: CompoundCapsuleDef[];
    hulls?: CompoundHullDef[];
    meshes?: CompoundMeshDef[];
    spheres?: CompoundSphereDef[];
};

// --- runtime instances (b3Compound*) --------------------------------------------------------

export type CompoundCapsule = { capsule: Capsule; materialIndex: number };
export type CompoundHull = { hull: HullData; transform: Transform; materialIndex: number };
export type CompoundMesh = {
    meshData: MeshData;
    transform: Transform;
    scale: Vec3;
    materialIndices: number[];
};
export type CompoundSphere = { sphere: Sphere; materialIndex: number };

/** Runtime data for a compound shape (b3CompoundData), as a struct of arrays. */
export type CompoundData = {
    tree: tree.DynamicTree;
    materials: SurfaceMaterial[];
    materialCount: number;
    capsules: CompoundCapsule[];
    hulls: CompoundHull[];
    meshes: CompoundMesh[];
    spheres: CompoundSphere[];
    sharedHullCount: number;
    sharedMeshCount: number;
};

/**
 * A compound child resolved to a temporary shape (b3ChildShape). Convex children read material index
 * 0; a mesh child carries all its (up to 4) material indices for the per-triangle remap.
 */
export type ChildShape = {
    type: ShapeType;
    transform: Transform;
    materialIndices: number[];
    capsule?: Capsule;
    hull?: HullData;
    mesh?: Mesh;
    sphere?: Sphere;
};

function materialsEqual(a: SurfaceMaterial, b: SurfaceMaterial): boolean {
    return (
        a.friction === b.friction &&
        a.restitution === b.restitution &&
        a.rollingResistance === b.rollingResistance &&
        a.tangentVelocity.x === b.tangentVelocity.x &&
        a.tangentVelocity.y === b.tangentVelocity.y &&
        a.tangentVelocity.z === b.tangentVelocity.z &&
        a.userMaterialId === b.userMaterialId &&
        a.customColor === b.customColor
    );
}

const vec3Equal = (a: Vec3, b: Vec3): boolean => a.x === b.x && a.y === b.y && a.z === b.z;

// Structural content equality of two baked meshes (b3CompareMeshes, which memcmp's the whole blob). Two
// meshes built from identical input are bit-identical field for field, so a deep compare reproduces the
// C's dedup outcome without the dropped serialization hash. Used only for the diagnostic sharedMeshCount.
function meshesEqual(a: MeshData, b: MeshData): boolean {
    if (a === b) {
        return true;
    }
    if (
        a.surfaceArea !== b.surfaceArea ||
        a.treeHeight !== b.treeHeight ||
        a.degenerateCount !== b.degenerateCount ||
        a.materialCount !== b.materialCount ||
        a.nodes.length !== b.nodes.length ||
        a.vertices.length !== b.vertices.length ||
        a.triangles.length !== b.triangles.length ||
        a.materialIndices.length !== b.materialIndices.length ||
        a.flags.length !== b.flags.length ||
        !vec3Equal(a.bounds.lowerBound, b.bounds.lowerBound) ||
        !vec3Equal(a.bounds.upperBound, b.bounds.upperBound)
    ) {
        return false;
    }
    for (let i = 0; i < a.vertices.length; ++i) {
        if (!vec3Equal(a.vertices[i], b.vertices[i])) {
            return false;
        }
    }
    for (let i = 0; i < a.triangles.length; ++i) {
        const ta = a.triangles[i];
        const tb = b.triangles[i];
        if (ta.index1 !== tb.index1 || ta.index2 !== tb.index2 || ta.index3 !== tb.index3) {
            return false;
        }
    }
    for (let i = 0; i < a.materialIndices.length; ++i) {
        if (a.materialIndices[i] !== b.materialIndices[i]) {
            return false;
        }
    }
    for (let i = 0; i < a.flags.length; ++i) {
        if (a.flags[i] !== b.flags[i]) {
            return false;
        }
    }
    for (let i = 0; i < a.nodes.length; ++i) {
        const na = a.nodes[i];
        const nb = b.nodes[i];
        if (
            na.leaf !== nb.leaf ||
            na.axis !== nb.axis ||
            na.childOffset !== nb.childOffset ||
            na.triangleCount !== nb.triangleCount ||
            na.triangleOffset !== nb.triangleOffset ||
            !vec3Equal(na.lowerBound, nb.lowerBound) ||
            !vec3Equal(na.upperBound, nb.upperBound)
        ) {
            return false;
        }
    }
    return true;
}

/**
 * Resolve a compound child by index (b3GetCompoundChild). Order is capsules, hulls, meshes, spheres —
 * the order they were added to the inner tree, so `childIndex` matches the tree's leaf user data.
 */
export function getCompoundChild(compound: CompoundData, childIndex: number): ChildShape {
    let index = childIndex;

    if (index >= 0 && index < compound.capsules.length) {
        const c = compound.capsules[index];
        return {
            type: ShapeType.Capsule,
            transform: xf.identity(),
            materialIndices: [c.materialIndex, 0, 0, 0],
            capsule: c.capsule,
        };
    }
    index -= compound.capsules.length;

    if (index >= 0 && index < compound.hulls.length) {
        const h = compound.hulls[index];
        return {
            type: ShapeType.Hull,
            transform: h.transform,
            materialIndices: [h.materialIndex, 0, 0, 0],
            hull: h.hull,
        };
    }
    index -= compound.hulls.length;

    if (index >= 0 && index < compound.meshes.length) {
        const m = compound.meshes[index];
        return {
            type: ShapeType.Mesh,
            transform: m.transform,
            materialIndices: [...m.materialIndices],
            mesh: { data: m.meshData, scale: m.scale },
        };
    }
    index -= compound.meshes.length;

    const s = compound.spheres[index];
    return {
        type: ShapeType.Sphere,
        transform: xf.identity(),
        materialIndices: [s.materialIndex, 0, 0, 0],
        sphere: s.sphere,
    };
}

/** All materials of a compound (b3GetCompoundMaterials). */
export function getCompoundMaterials(compound: CompoundData): SurfaceMaterial[] {
    return compound.materials;
}

/**
 * Bake a compound shape from its definition (b3CreateCompound): de-dup materials + shared hulls/meshes,
 * build the inner dynamic tree over every child, then full-rebuild it. Returns null if the child count
 * exceeds B3_MAX_CHILD_SHAPES. The child order (and thus tree leaf user data) is capsules, hulls,
 * meshes, spheres.
 */
export function createCompound(def: CompoundDef): CompoundData | null {
    const capsuleDefs = def.capsules ?? [];
    const hullDefs = def.hulls ?? [];
    const meshDefs = def.meshes ?? [];
    const sphereDefs = def.spheres ?? [];

    const convexCount = capsuleDefs.length + hullDefs.length + sphereDefs.length;
    const shapeCount = convexCount + meshDefs.length;
    if (shapeCount >= MAX_CHILD_SHAPES) {
        return null;
    }

    const t = tree.createTree(shapeCount);
    let childIndex = 0;

    // First-seen material de-dup: linear find-or-append, yielding the same indices as the C hash map.
    const materials: SurfaceMaterial[] = [];
    const getOrInsertMaterial = (mat: SurfaceMaterial): number => {
        for (let i = 0; i < materials.length; ++i) {
            if (materialsEqual(materials[i], mat)) {
                return i;
            }
        }
        materials.push(cloneMaterial(mat));
        return materials.length - 1;
    };

    // Capsules
    const capsules: CompoundCapsule[] = [];
    for (const cd of capsuleDefs) {
        const materialIndex = getOrInsertMaterial(cd.material);
        const capsule = roundCapsule(cd.capsule);
        capsules.push({ capsule, materialIndex });
        const box = computeCapsuleAABB(capsule, xf.identity());
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;
    }

    // Hulls — proxy first (matching the C loop order), then material + shared-hull de-dup by content hash.
    const hulls: CompoundHull[] = [];
    const sharedHullHashes = new Set<number>();
    for (const hd of hullDefs) {
        const box = computeHullAABB(hd.hull, hd.transform);
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;

        const materialIndex = getOrInsertMaterial(hd.material);
        hulls.push({ hull: hd.hull, transform: hd.transform, materialIndex });
        sharedHullHashes.add(hd.hull.hash);
    }

    // Meshes — one material index per mesh material slot; shared meshes de-dup by content (b3CompareMeshes).
    const meshes: CompoundMesh[] = [];
    const sharedMeshes: MeshData[] = [];
    for (const md of meshDefs) {
        const box = computeMeshAABB(md.meshData, md.transform, md.scale);
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;

        const materialIndices = new Array<number>(MAX_COMPOUND_MESH_MATERIALS).fill(0);
        for (let j = 0; j < md.materialCount; ++j) {
            materialIndices[j] = getOrInsertMaterial(md.materials[j]);
        }
        meshes.push({
            meshData: md.meshData,
            transform: md.transform,
            scale: safeScale(md.scale),
            materialIndices,
        });
        if (sharedMeshes.every((m) => meshesEqual(m, md.meshData) === false)) {
            sharedMeshes.push(md.meshData);
        }
    }

    // Spheres
    const spheres: CompoundSphere[] = [];
    for (const sd of sphereDefs) {
        const materialIndex = getOrInsertMaterial(sd.material);
        const sphere = roundSphere(sd.sphere);
        spheres.push({ sphere, materialIndex });
        const box = computeSphereAABB(sphere, xf.identity());
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;
    }

    tree.rebuild(t, true);

    return {
        tree: t,
        materials,
        materialCount: materials.length,
        capsules,
        hulls,
        meshes,
        spheres,
        sharedHullCount: sharedHullHashes.size,
        sharedMeshCount: sharedMeshes.length,
    };
}

/** Enclosing AABB of a compound under a transform (b3ComputeCompoundAABB): its tree root, transformed. */
export function computeCompoundAABB(compound: CompoundData, transform: Transform): AABB {
    const root = compound.tree.root;
    const box = tree.getAABB(compound.tree, root);
    return aabb.transform(transform, box);
}

/** Callback for a compound tree query; return false to stop (b3CompoundQueryFcn). */
export type CompoundQueryCallback = (childIndex: number) => boolean;

/** Query the compound's inner tree with a compound-local AABB (b3QueryCompound). */
export function queryCompound(
    compound: CompoundData,
    box: AABB,
    callback: CompoundQueryCallback,
): void {
    tree.query(compound.tree, box, ALL_BITS_HI, ALL_BITS_LO, false, (_proxyId, childIndex) =>
        callback(childIndex),
    );
}

// Remap a convex/mesh child's dispatched cast output into the compound's material array. Convex
// children read material slot 0; a mesh child's per-triangle material index selects the slot.
const childMaterial = (child: ChildShape, output: CastOutput): number => {
    if (child.type === ShapeType.Mesh) {
        const slot = minInt(output.materialIndex, MAX_COMPOUND_MESH_MATERIALS - 1);
        return child.materialIndices[slot];
    }
    return child.materialIndices[0];
};

/**
 * Ray vs compound (b3RayCastCompound). The input is already in the compound's local frame; the inner
 * tree ray cast visits each candidate child, casts against it in the child's frame, and lifts the
 * nearest hit's point/normal back to the compound frame.
 */
export function rayCastCompound(compound: CompoundData, input: RayCastInput): CastOutput {
    const result = emptyCastOutput();

    tree.rayCast(
        compound.tree,
        input,
        ALL_BITS_HI,
        ALL_BITS_LO,
        false,
        (rayInput, _proxyId, childIndex) => {
            const child = getCompoundChild(compound, childIndex);

            const localInput: RayCastInput = {
                origin: xf.invPoint(child.transform, rayInput.origin),
                translation: quat.invRotate(child.transform.q, rayInput.translation),
                maxFraction: rayInput.maxFraction,
            };

            let output: CastOutput;
            switch (child.type) {
                case ShapeType.Capsule:
                    output = rayCastCapsule(child.capsule as Capsule, localInput);
                    break;
                case ShapeType.Hull:
                    output = rayCastHull(child.hull as HullData, localInput);
                    break;
                case ShapeType.Mesh:
                    output = rayCastMesh(child.mesh as Mesh, localInput);
                    break;
                default:
                    output = rayCastSphere(child.sphere as Sphere, localInput);
                    break;
            }
            output.materialIndex = childMaterial(child, output);

            if (output.hit) {
                output.point = xf.point(child.transform, output.point);
                output.normal = quat.rotate(child.transform.q, output.normal);
                output.childIndex = childIndex;
                copyCastOutput(result, output);
                return output.fraction;
            }
            return rayInput.maxFraction;
        },
    );

    return result;
}

/**
 * Shape cast (a swept convex proxy) vs compound (b3ShapeCastCompound). The input is in the compound's
 * local frame; the inner tree box cast visits each candidate child, pulls the proxy + translation into
 * the child's frame (matrix form, matching the C), casts, and lifts the nearest hit back.
 */
export function shapeCastCompound(compound: CompoundData, input: ShapeCastInput): CastOutput {
    const result = emptyCastOutput();
    if (input.proxy.count === 0) return result;

    const box = aabb.make(input.proxy.points, input.proxy.count, input.proxy.radius);
    const treeInput = { box, translation: input.translation, maxFraction: input.maxFraction };

    tree.boxCast(
        compound.tree,
        treeInput,
        ALL_BITS_HI,
        ALL_BITS_LO,
        false,
        (boxInput, _proxyId, childIndex) => {
            const child = getCompoundChild(compound, childIndex);

            const invTransform = xf.invert(child.transform);
            const m = mat3.fromQuat(invTransform.q);
            const count = minInt(input.proxy.count, MAX_SHAPE_CAST_POINTS);
            const localPoints: Vec3[] = new Array(count);
            for (let i = 0; i < count; ++i) {
                localPoints[i] = vec3.add(mat3.mulV(m, input.proxy.points[i]), invTransform.p);
            }
            const localInput: ShapeCastInput = {
                proxy: { points: localPoints, count, radius: input.proxy.radius },
                translation: mat3.mulV(m, input.translation),
                maxFraction: boxInput.maxFraction,
                canEncroach: input.canEncroach,
            };

            let output: CastOutput;
            switch (child.type) {
                case ShapeType.Capsule:
                    output = shapeCastCapsule(child.capsule as Capsule, localInput);
                    break;
                case ShapeType.Hull:
                    output = shapeCastHull(child.hull as HullData, localInput);
                    break;
                case ShapeType.Mesh:
                    output = shapeCastMesh(child.mesh as Mesh, localInput);
                    break;
                default:
                    output = shapeCastSphere(child.sphere as Sphere, localInput);
                    break;
            }
            output.materialIndex = childMaterial(child, output);

            if (output.hit) {
                output.point = xf.point(child.transform, output.point);
                output.normal = quat.rotate(child.transform.q, output.normal);
                output.childIndex = childIndex;
                copyCastOutput(result, output);
                return output.fraction;
            }
            return boxInput.maxFraction;
        },
    );

    return result;
}

/**
 * True if `proxy` (in world space, with the compound at `transform`) overlaps any child
 * (b3OverlapCompound). Queries the inner tree with the proxy's world-space bounds and tests each
 * candidate child in world space.
 */
export function overlapCompound(
    compound: CompoundData,
    transform: Transform,
    proxy: ShapeProxy,
): boolean {
    // The tree is in compound-local space but the C queries it with the world-space proxy bounds; the
    // per-child overlap test below applies the full world transform, so we mirror that exactly.
    const box = computeProxyAABB(proxy);

    let overlap = false;
    tree.query(compound.tree, box, ALL_BITS_HI, ALL_BITS_LO, false, (_proxyId, childIndex) => {
        const child = getCompoundChild(compound, childIndex);
        const childTransform = xf.mul(transform, child.transform);

        let hit: boolean;
        switch (child.type) {
            case ShapeType.Capsule:
                hit = overlapCapsule(child.capsule as Capsule, childTransform, proxy);
                break;
            case ShapeType.Hull:
                hit = overlapHull(child.hull as HullData, childTransform, proxy);
                break;
            case ShapeType.Mesh:
                hit = overlapMesh(child.mesh as Mesh, childTransform, proxy);
                break;
            default:
                hit = overlapSphere(child.sphere as Sphere, childTransform, proxy);
                break;
        }

        if (hit) {
            overlap = true;
            return false;
        }
        return true;
    });

    return overlap;
}

/**
 * Collision planes between a capsule mover and a compound (b3CollideMoverAndCompound), in the
 * compound's frame. Queries the inner tree with the mover's bounds, pulls the mover into each child's
 * frame (quat form, matching the C), collects that child's planes, and rotates them back to compound
 * space. Stops once `capacity` planes are gathered.
 */
export function collideMoverAndCompound(
    compound: CompoundData,
    capacity: number,
    mover: Capsule,
): PlaneResult[] {
    const planes: PlaneResult[] = [];

    const r = { x: mover.radius, y: mover.radius, z: mover.radius };
    const box: AABB = {
        lowerBound: vec3.sub(vec3.min(mover.center1, mover.center2), r),
        upperBound: vec3.add(vec3.max(mover.center1, mover.center2), r),
    };

    queryCompound(compound, box, (childIndex) => {
        const child = getCompoundChild(compound, childIndex);

        // Transform the mover to child space (quat form, b3InvTransformPoint).
        const localMover: Capsule = {
            center1: xf.invPoint(child.transform, mover.center1),
            center2: xf.invPoint(child.transform, mover.center2),
            radius: mover.radius,
        };

        let childPlanes: PlaneResult[];
        switch (child.type) {
            case ShapeType.Capsule: {
                const p = collideMoverAndCapsule(child.capsule as Capsule, localMover);
                childPlanes = p ? [p] : [];
                break;
            }
            case ShapeType.Hull: {
                const p = collideMoverAndHull(child.hull as HullData, localMover);
                childPlanes = p ? [p] : [];
                break;
            }
            case ShapeType.Mesh:
                childPlanes = collideMoverAndMesh(
                    child.mesh as Mesh,
                    capacity - planes.length,
                    localMover,
                );
                break;
            default: {
                const p = collideMoverAndSphere(child.sphere as Sphere, localMover);
                childPlanes = p ? [p] : [];
                break;
            }
        }

        // Transform each plane back to compound space.
        for (const pr of childPlanes) {
            pr.plane.normal = quat.rotate(child.transform.q, pr.plane.normal);
            pr.point = xf.point(child.transform, pr.point);
            planes.push(pr);
        }

        // Continue the query while there is room for more planes.
        return planes.length < capacity;
    });

    return planes;
}

// b3CompoundRayCastCallback/b3CompoundShapeCastCallback store the hit by overwriting *output; the port
// mirrors that in-place write so the accumulator identity stays stable across the tree walk.
function copyCastOutput(dst: CastOutput, src: CastOutput): void {
    dst.normal = src.normal;
    dst.point = src.point;
    dst.fraction = src.fraction;
    dst.iterations = src.iterations;
    dst.triangleIndex = src.triangleIndex;
    dst.childIndex = src.childIndex;
    dst.materialIndex = src.materialIndex;
    dst.hit = src.hit;
}
