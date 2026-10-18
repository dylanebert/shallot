// Box3D compound.c (Erin Catto, MIT). Inner-tree build order determines child query order,
// contact ids and solver order. Material indices follow first-seen order across child types.
// sharedHullCount and sharedMeshCount are diagnostics; children retain their geometry references.

import { ALL_BITS_HI, ALL_BITS_LO } from "../common/constants";
import type { Transform, Vec3 } from "../common/records";
import { cloneMaterial, ShapeType, type SurfaceMaterial } from "../common/types";
import * as tree from "../kernel/treecolumns";
import {
    type Capsule,
    computeCapsuleAABB,
    computeSphereAABB,
    roundCapsule,
    roundSphere,
    type Sphere,
} from "./geometry";
import { computeHullAABB, type HullData } from "./hull";
import { computeMeshAABB, type Mesh, type MeshData } from "./mesh";

const f32 = Math.fround;

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

/** Compound definition (b3CompoundDef). Materials and sphere/capsule values are copied;
 * hulls, mesh data and child transforms remain caller-owned references. */
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
        a.tangentVelocity[0] === b.tangentVelocity[0] &&
        a.tangentVelocity[1] === b.tangentVelocity[1] &&
        a.tangentVelocity[2] === b.tangentVelocity[2] &&
        a.userMaterialId === b.userMaterialId &&
        a.customColor === b.customColor
    );
}

const vec3Equal = (a: Vec3, b: Vec3): boolean => a.x === b.x && a.y === b.y && a.z === b.z;

// b3CompareMeshes uses serialized bytes; this representation compares fields for sharedMeshCount.
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
            transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
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
        transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } },
        materialIndices: [s.materialIndex, 0, 0, 0],
        sphere: s.sphere,
    };
}

/** All materials of a compound (b3GetCompoundMaterials). */
export function getCompoundMaterials(compound: CompoundData): SurfaceMaterial[] {
    return compound.materials;
}

/**
 * Bake a compound shape (b3CreateCompound). Returns null when the child count is at least
 * B3_MAX_CHILD_SHAPES. Child indices and tree leaf user data follow capsules, hulls, meshes, spheres.
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

    const capsules: CompoundCapsule[] = [];
    for (const cd of capsuleDefs) {
        const materialIndex = getOrInsertMaterial(cd.material);
        const capsule = roundCapsule(cd.capsule);
        capsules.push({ capsule, materialIndex });
        const box = computeCapsuleAABB(capsule, {
            p: { x: 0, y: 0, z: 0 },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        });
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;
    }

    const hulls: CompoundHull[] = [];
    const sharedHullHashes = new Set<bigint>();
    for (const hd of hullDefs) {
        const box = computeHullAABB(hd.hull, hd.transform);
        tree.createProxy(t, box, ALL_BITS_HI, ALL_BITS_LO, childIndex);
        childIndex += 1;

        const materialIndex = getOrInsertMaterial(hd.material);
        hulls.push({ hull: hd.hull, transform: hd.transform, materialIndex });
        sharedHullHashes.add(hd.hull.hash);
    }

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
            // Unclamped: b3SafeScale applies to mesh shapes, not compound mesh children.
            scale: { x: f32(md.scale.x), y: f32(md.scale.y), z: f32(md.scale.z) },
            materialIndices,
        });
        if (sharedMeshes.every((m) => meshesEqual(m, md.meshData) === false)) {
            sharedMeshes.push(md.meshData);
        }
    }

    const spheres: CompoundSphere[] = [];
    for (const sd of sphereDefs) {
        const materialIndex = getOrInsertMaterial(sd.material);
        const sphere = roundSphere(sd.sphere);
        spheres.push({ sphere, materialIndex });
        const box = computeSphereAABB(sphere, {
            p: { x: 0, y: 0, z: 0 },
            q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
        });
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
