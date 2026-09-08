import { describe, expect, test } from "bun:test";
import {
    type CompoundData,
    type CompoundDef,
    collideMoverAndCompound,
    computeCompoundAABB,
    createCompound,
    getCompoundChild,
    getCompoundMaterials,
    overlapCompound,
    queryCompound,
    rayCastCompound,
    shapeCastCompound,
} from "./compound";
import type { ShapeProxy } from "./distance";
import { type Capsule, computeCapsuleAABB, computeSphereAABB, type Sphere } from "./geometry";
import gold from "./geometry.gold.json";
import { computeHullAABB, type HullData, makeBoxHull } from "./hull";
import { aabb, f32, PI, type Quat, quat, type Transform, type Vec3, vec3, xf } from "./math";
import { computeMeshAABB, createBoxMesh, type MeshData } from "./mesh";
import { readNode } from "./tree";
import { defaultSurfaceMaterial, ShapeType, type SurfaceMaterial } from "./types";

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
function vecEqual(got: Vec3, want: string[], label: string) {
    bitEqual(got.x, want[0], `${label}.x`);
    bitEqual(got.y, want[1], `${label}.y`);
    bitEqual(got.z, want[2], `${label}.z`);
}
function quatEqual(got: Quat, want: string[], label: string) {
    bitEqual(got.v.x, want[0], `${label}.x`);
    bitEqual(got.v.y, want[1], `${label}.y`);
    bitEqual(got.v.z, want[2], `${label}.z`);
    bitEqual(got.s, want[3], `${label}.w`);
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

type MaterialGold = {
    friction: string;
    restitution: string;
    rollingResistance: string;
    tangentVelocity: string[];
    userMaterialId: string;
    customColor: number;
};
type NodeGold = {
    index: number;
    leaf: boolean;
    height: number;
    flags: number;
    aabb: { lowerBound: string[]; upperBound: string[] };
    userData?: string;
    child1?: number;
    child2?: number;
};
type ChildGold = {
    type: number;
    transform: { p: string[]; q: string[] };
    materialIndices: number[];
};
type CompoundGold = {
    name: string;
    capsuleCount: number;
    hullCount: number;
    meshCount: number;
    sphereCount: number;
    materialCount: number;
    sharedHullCount: number;
    sharedMeshCount: number;
    materials: MaterialGold[];
    tree: { root: number; nodeCount: number; nodeCapacity: number; nodes: NodeGold[] };
    children: ChildGold[];
};

const cGold = (name: string) =>
    gold.compounds.find((c) => c.name === name) as unknown as CompoundGold;

function assertCompound(c: CompoundData, g: CompoundGold) {
    expect(c.capsules.length).toBe(g.capsuleCount);
    expect(c.hulls.length).toBe(g.hullCount);
    expect(c.meshes.length).toBe(g.meshCount);
    expect(c.spheres.length).toBe(g.sphereCount);
    expect(c.materialCount).toBe(g.materialCount);
    expect(c.sharedHullCount).toBe(g.sharedHullCount);
    expect(c.sharedMeshCount).toBe(g.sharedMeshCount);

    // Materials, bit for bit + user id + color.
    const materials = getCompoundMaterials(c);
    expect(materials.length).toBe(g.materials.length);
    for (let i = 0; i < g.materials.length; ++i) {
        const gm = g.materials[i];
        const m = materials[i];
        bitEqual(m.friction, gm.friction, `${g.name} mat[${i}].friction`);
        bitEqual(m.restitution, gm.restitution, `${g.name} mat[${i}].restitution`);
        bitEqual(
            m.rollingResistance,
            gm.rollingResistance,
            `${g.name} mat[${i}].rollingResistance`,
        );
        vecEqual(m.tangentVelocity, gm.tangentVelocity, `${g.name} mat[${i}].tangentVelocity`);
        expect(m.userMaterialId).toBe(BigInt(gm.userMaterialId));
        expect(m.customColor).toBe(gm.customColor);
    }

    // Inner tree: root + every allocated node, addressed by index (the build order the sim depends on).
    const t = c.tree;
    expect(t.root).toBe(g.tree.root);
    expect(t.nodeCount).toBe(g.tree.nodeCount);
    for (const gn of g.tree.nodes) {
        const n = readNode(t, gn.index);
        const label = `${g.name} node[${gn.index}]`;
        expect((n.flags & 0x0004) !== 0).toBe(gn.leaf);
        expect(n.flags).toBe(gn.flags);
        expect(n.height).toBe(gn.height);
        vecEqual(n.aabb.lowerBound, gn.aabb.lowerBound, `${label}.lo`);
        vecEqual(n.aabb.upperBound, gn.aabb.upperBound, `${label}.hi`);
        if (gn.leaf) {
            expect(String(n.userData)).toBe(gn.userData as string);
        } else {
            expect(n.child1).toBe(gn.child1 as number);
            expect(n.child2).toBe(gn.child2 as number);
        }
    }

    // Children resolved via getCompoundChild (the narrow-phase accessor): type, transform, indices.
    expect(g.children.length).toBe(g.capsuleCount + g.hullCount + g.meshCount + g.sphereCount);
    for (let i = 0; i < g.children.length; ++i) {
        const gc = g.children[i];
        const child = getCompoundChild(c, i);
        expect(child.type).toBe(gc.type as ShapeType);
        vecEqual(child.transform.p, gc.transform.p, `${g.name} child[${i}].p`);
        quatEqual(child.transform.q, gc.transform.q, `${g.name} child[${i}].q`);
        expect(child.materialIndices).toEqual(gc.materialIndices);
    }
}

// Mirrors the mixed scene in fixtures/geometry_gold.c: one of every child type, one shared material.
function buildMixed(): CompoundData {
    const mat = defaultSurfaceMaterial();
    const box = makeBoxHull(0.5, 0.5, 0.5);
    const md = createBoxMesh(vec3.zero(), v(0.5, 0.5, 0.5), false);
    const def: CompoundDef = {
        capsules: [
            { capsule: { center1: v(-1, 0, 0), center2: v(1, 0, 0), radius: 0.25 }, material: mat },
        ],
        hulls: [{ hull: box, transform: xf.identity(), material: mat }],
        meshes: [
            {
                meshData: md,
                transform: xf.identity(),
                scale: v(1, 1, 1),
                materials: [mat],
                materialCount: 1,
            },
        ],
        spheres: [
            { sphere: { center: v(5, 0, 0), radius: 0.5 }, material: mat },
            { sphere: { center: v(-5, 0, 0), radius: 0.5 }, material: mat },
        ],
    };
    return createCompound(def) as CompoundData;
}

function makeMat(friction: number, userId: bigint): SurfaceMaterial {
    // The C stores material floats as f32 (b3SurfaceMaterial is float); fround to match `0.6f` etc.
    const m = defaultSurfaceMaterial();
    m.friction = f32(friction);
    m.userMaterialId = userId;
    return m;
}

// Mirrors the materials scene: 3 capsules, distinct materials in first-seen order.
function buildMaterials(): CompoundData {
    const def: CompoundDef = {
        capsules: [0, 1, 2].map((i) => ({
            capsule: { center1: v(i, 0, 0), center2: v(i + 1, 0, 0), radius: 0.25 },
            material: makeMat(f32(f32(0.1) * f32(i + 1)), BigInt(i + 1)),
        })),
    };
    return createCompound(def) as CompoundData;
}

// Mirrors the transforms scene: the same box hull at two instance transforms (one rotated), sharing one
// material, plus a sphere with a second material.
function buildTransforms(): CompoundData {
    const box = makeBoxHull(0.5, 0.5, 0.5);
    // f32-round the axis/angle literals to match the C `0.3f`/`0.6f` before normalize/fromAxisAngle.
    const q = quat.fromAxisAngle(vec3.normalize(v(f32(0.3), f32(0.7), f32(0.2))), f32(0.6));
    const matA = makeMat(0.6, 10n);
    const matB = makeMat(0.3, 20n);
    const idTransform: Transform = { p: v(1, 0.5, -0.5), q: quat.identity() };
    const rotTransform: Transform = { p: v(-1, 0, 2), q };
    const def: CompoundDef = {
        hulls: [
            { hull: box, transform: idTransform, material: matA },
            { hull: box, transform: rotTransform, material: matA },
        ],
        spheres: [{ sphere: { center: v(0, 3, 0), radius: 0.75 }, material: matB }],
    };
    return createCompound(def) as CompoundData;
}

describe("compound geometry gold", () => {
    test("mixed — all child types, shared material, inner tree", () => {
        assertCompound(buildMixed(), cGold("mixed"));
    });
    test("materials — distinct materials, first-seen index order", () => {
        assertCompound(buildMaterials(), cGold("materials"));
    });
    test("transforms — shared hull at two transforms + sphere, two materials", () => {
        assertCompound(buildTransforms(), cGold("transforms"));
    });
});

// Ported from reference/box3d/test/test_compound.c (create + material + dedup + dispatch subtests). The
// contact/query/mover/serialize subtests land with their own later batches.
describe("test_compound.c parity", () => {
    const mat = defaultSurfaceMaterial();

    test("CompoundCreateMixed — counts + shared diagnostics", () => {
        const c = buildMixed();
        expect(c.capsules.length).toBe(1);
        expect(c.hulls.length).toBe(1);
        expect(c.meshes.length).toBe(1);
        expect(c.spheres.length).toBe(2);
        expect(c.materialCount).toBe(1);
        expect(c.sharedHullCount).toBe(1);
        expect(c.sharedMeshCount).toBe(1);
        expect(c.tree.nodeCount).toBeGreaterThan(0);
        expect(c.tree.root).toBeGreaterThanOrEqual(0);
    });

    test("CompoundCreateSingleType — one child of each type in isolation", () => {
        const cap = createCompound({
            capsules: [
                {
                    capsule: { center1: v(0, 0, 0), center2: v(1, 0, 0), radius: 0.5 },
                    material: mat,
                },
            ],
        }) as CompoundData;
        expect(cap.capsules.length).toBe(1);
        expect(cap.hulls.length).toBe(0);
        expect(cap.meshes.length).toBe(0);
        expect(cap.spheres.length).toBe(0);

        const box = makeBoxHull(1, 1, 1);
        const hull = createCompound({
            hulls: [{ hull: box, transform: xf.identity(), material: mat }],
        }) as CompoundData;
        expect(hull.hulls.length).toBe(1);
        expect(hull.sharedHullCount).toBe(1);
        expect(hull.capsules.length).toBe(0);

        const md = createBoxMesh(vec3.zero(), v(1, 1, 1), false);
        const mesh = createCompound({
            meshes: [
                {
                    meshData: md,
                    transform: xf.identity(),
                    scale: v(1, 1, 1),
                    materials: [mat],
                    materialCount: 1,
                },
            ],
        }) as CompoundData;
        expect(mesh.meshes.length).toBe(1);
        expect(mesh.sharedMeshCount).toBe(1);

        const sphere = createCompound({
            spheres: [{ sphere: { center: v(0, 0, 0), radius: 1 }, material: mat }],
        }) as CompoundData;
        expect(sphere.spheres.length).toBe(1);
    });

    test("CompoundMaterialDedup — identical materials collapse to one slot", () => {
        const m = makeMat(0.4, 7n);
        const c = createCompound({
            capsules: [0, 1, 2].map((i) => ({
                capsule: { center1: v(i, 0, 0), center2: v(i + 1, 0, 0), radius: 0.25 },
                material: m,
            })),
        }) as CompoundData;
        expect(c.materialCount).toBe(1);
        for (let i = 0; i < 3; ++i) {
            expect(c.capsules[i].materialIndex).toBe(0);
        }
    });

    test("CompoundMaterialDistinct — unique materials keep distinct slots", () => {
        const c = createCompound({
            capsules: [0, 1, 2].map((i) => ({
                capsule: { center1: v(i, 0, 0), center2: v(i + 1, 0, 0), radius: 0.25 },
                material: makeMat(f32(f32(0.1) * f32(i + 1)), BigInt(i + 1)),
            })),
        }) as CompoundData;
        expect(c.materialCount).toBe(3);
        const materials = getCompoundMaterials(c);
        for (let i = 0; i < 3; ++i) {
            const idx = c.capsules[i].materialIndex;
            expect(idx).toBeGreaterThanOrEqual(0);
            expect(idx).toBeLessThan(3);
            expect(materials[idx].userMaterialId).toBe(BigInt(i + 1));
        }
    });

    test("CompoundMaterialCrossShape — one material shared across capsule/hull/sphere", () => {
        const m = makeMat(0.5, 99n);
        const box = makeBoxHull(0.5, 0.5, 0.5);
        const c = createCompound({
            capsules: [
                {
                    capsule: { center1: v(0, 0, 0), center2: v(1, 0, 0), radius: 0.25 },
                    material: m,
                },
            ],
            hulls: [{ hull: box, transform: xf.identity(), material: m }],
            spheres: [{ sphere: { center: v(5, 0, 0), radius: 0.5 }, material: m }],
        }) as CompoundData;
        expect(c.materialCount).toBe(1);
        expect(c.capsules[0].materialIndex).toBe(0);
        expect(c.hulls[0].materialIndex).toBe(0);
        expect(c.spheres[0].materialIndex).toBe(0);
    });

    test("CompoundMaterialMeshShared — mesh material dedups against a convex material", () => {
        const m = makeMat(0.3, 11n);
        const md = createBoxMesh(vec3.zero(), v(1, 1, 1), false);
        expect(md.materialCount).toBe(1);
        const c = createCompound({
            meshes: [
                {
                    meshData: md,
                    transform: xf.identity(),
                    scale: v(1, 1, 1),
                    materials: [m],
                    materialCount: 1,
                },
            ],
            spheres: [{ sphere: { center: v(5, 0, 0), radius: 0.5 }, material: m }],
        }) as CompoundData;
        // The mesh's one material and the sphere's identical material share a single slot.
        expect(c.materialCount).toBe(1);
        expect(c.meshes[0].materialIndices[0]).toBe(0);
        expect(c.spheres[0].materialIndex).toBe(0);
    });

    test("CompoundHullSharingPointer / Content / Distinct — shared-hull count by content", () => {
        const box = makeBoxHull(0.5, 0.5, 0.5);
        // Same pointer twice → 1 shared hull.
        const samePtr = createCompound({
            hulls: [
                { hull: box, transform: xf.identity(), material: mat },
                { hull: box, transform: { p: v(2, 0, 0), q: quat.identity() }, material: mat },
            ],
        }) as CompoundData;
        expect(samePtr.hulls.length).toBe(2);
        expect(samePtr.sharedHullCount).toBe(1);

        // Distinct pointers, identical content → still 1 (content dedup by hash).
        const box2 = makeBoxHull(0.5, 0.5, 0.5);
        const sameContent = createCompound({
            hulls: [
                { hull: box, transform: xf.identity(), material: mat },
                { hull: box2, transform: xf.identity(), material: mat },
            ],
        }) as CompoundData;
        expect(sameContent.sharedHullCount).toBe(1);

        // Genuinely different hulls → 2.
        const other = makeBoxHull(1, 0.5, 0.25);
        const distinct = createCompound({
            hulls: [
                { hull: box, transform: xf.identity(), material: mat },
                { hull: other, transform: xf.identity(), material: mat },
            ],
        }) as CompoundData;
        expect(distinct.sharedHullCount).toBe(2);
    });

    test("CompoundMeshSharingPointer / Content / Distinct — shared-mesh count by content", () => {
        const md = createBoxMesh(vec3.zero(), v(1, 1, 1), false);
        const meshDef = (data: MeshData, tp: Vec3) => ({
            meshData: data,
            transform: { p: tp, q: quat.identity() },
            scale: v(1, 1, 1),
            materials: [mat],
            materialCount: 1,
        });

        const samePtr = createCompound({
            meshes: [meshDef(md, v(0, 0, 0)), meshDef(md, v(3, 0, 0))],
        }) as CompoundData;
        expect(samePtr.meshes.length).toBe(2);
        expect(samePtr.sharedMeshCount).toBe(1);

        const md2 = createBoxMesh(vec3.zero(), v(1, 1, 1), false);
        const sameContent = createCompound({
            meshes: [meshDef(md, v(0, 0, 0)), meshDef(md2, v(0, 0, 0))],
        }) as CompoundData;
        expect(sameContent.sharedMeshCount).toBe(1);

        const mdOther = createBoxMesh(vec3.zero(), v(2, 1, 0.5), false);
        const distinct = createCompound({
            meshes: [meshDef(md, v(0, 0, 0)), meshDef(mdOther, v(0, 0, 0))],
        }) as CompoundData;
        expect(distinct.sharedMeshCount).toBe(2);
    });

    test("CompoundChildDispatch — getCompoundChild resolves the right type per index", () => {
        const c = buildMixed();
        // Order: capsule (0), hull (1), mesh (2), spheres (3,4).
        expect(getCompoundChild(c, 0).type).toBe(ShapeType.Capsule);
        expect(getCompoundChild(c, 0).capsule).toBeDefined();
        expect(getCompoundChild(c, 1).type).toBe(ShapeType.Hull);
        expect(getCompoundChild(c, 1).hull).toBeDefined();
        expect(getCompoundChild(c, 2).type).toBe(ShapeType.Mesh);
        expect(getCompoundChild(c, 2).mesh).toBeDefined();
        expect(getCompoundChild(c, 3).type).toBe(ShapeType.Sphere);
        expect(getCompoundChild(c, 4).type).toBe(ShapeType.Sphere);
        expect(getCompoundChild(c, 3).sphere).toBeDefined();
    });

    test("CompoundQuery — inner-tree AABB query visits the overlapping children", () => {
        const c = buildMixed();
        // Child order: capsule (0), hull (1), mesh (2), sphere @+5 (3), sphere @-5 (4).
        const visit = (box: { lowerBound: Vec3; upperBound: Vec3 }): number[] => {
            const hits: number[] = [];
            queryCompound(c, box, (childIndex) => {
                hits.push(childIndex);
                return true;
            });
            return hits.sort((a, b) => a - b);
        };

        // A box far out on +x overlaps only the sphere centered at (5, 0, 0).
        expect(visit({ lowerBound: v(4.5, -0.5, -0.5), upperBound: v(5.5, 0.5, 0.5) })).toEqual([
            3,
        ]);

        // A box enclosing everything visits all five children.
        expect(visit({ lowerBound: v(-10, -10, -10), upperBound: v(10, 10, 10) })).toEqual([
            0, 1, 2, 3, 4,
        ]);
    });

    test("CompoundAABBContainsChildren — root AABB contains every child's AABB", () => {
        const c = buildMixed();
        const root = computeCompoundAABB(c, xf.identity());
        const childCount = c.capsules.length + c.hulls.length + c.meshes.length + c.spheres.length;
        for (let i = 0; i < childCount; ++i) {
            const child = getCompoundChild(c, i);
            let box: ReturnType<typeof computeSphereAABB>;
            switch (child.type) {
                case ShapeType.Capsule:
                    box = computeCapsuleAABB(child.capsule as Capsule, child.transform);
                    break;
                case ShapeType.Sphere:
                    box = computeSphereAABB(child.sphere as Sphere, child.transform);
                    break;
                case ShapeType.Hull:
                    box = computeHullAABB(child.hull as HullData, child.transform);
                    break;
                default: {
                    const m = child.mesh as { data: MeshData; scale: Vec3 };
                    box = computeMeshAABB(m.data, child.transform, m.scale);
                    break;
                }
            }
            expect(aabb.contains(root, box)).toBe(true);
        }
    });
});

// --- query behavioral tests (ported from test_compound.c) ------------------------------------

const rayDown = (origin: Vec3, translation: Vec3) => ({ origin, translation, maxFraction: 1 });

describe("compound queries (behavior vs C reference)", () => {
    test("ray cast misses a compound it passes over", () => {
        const c = createCompound({
            spheres: [
                {
                    sphere: { center: vec3.zero(), radius: 0.5 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        }) as CompoundData;
        const out = rayCastCompound(c, rayDown({ x: -5, y: 5, z: 0 }, { x: 10, y: 0, z: 0 }));
        expect(out.hit).toBe(false);
    });

    test("ray cast returns the nearest child, its normal and material", () => {
        const matA = makeMat(0.4, 100n);
        const matB = makeMat(0.4, 200n);
        const c = createCompound({
            spheres: [
                { sphere: { center: { x: 5, y: 0, z: 0 }, radius: 1 }, material: matA },
                { sphere: { center: { x: 10, y: 0, z: 0 }, radius: 1 }, material: matB },
            ],
        }) as CompoundData;
        const out = rayCastCompound(c, rayDown(vec3.zero(), { x: 20, y: 0, z: 0 }));
        expect(out.hit).toBe(true);
        // Front face of the nearer sphere is at x=4 → fraction 4/20 = 0.2.
        expect(Math.abs(out.fraction - 0.2)).toBeLessThanOrEqual(1e-4);
        expect(Math.abs(out.normal.x + 1)).toBeLessThanOrEqual(1e-4);
        expect(out.childIndex).toBe(0);
        const mats = getCompoundMaterials(c);
        expect(mats[out.materialIndex].userMaterialId).toBe(100n);
    });

    test("ray cast rotates a hull child's normal back into compound space", () => {
        // A unit box rotated 90° about Z at compound +X. The hit normal must come back from hull-local
        // space rotated into compound space (pointing -X toward the incoming ray).
        const c = createCompound({
            hulls: [
                {
                    hull: makeBoxHull(1, 1, 1),
                    transform: {
                        p: { x: 5, y: 0, z: 0 },
                        q: quat.fromAxisAngle(vec3.axisZ(), f32(0.5 * PI)),
                    },
                    material: defaultSurfaceMaterial(),
                },
            ],
        }) as CompoundData;
        const out = rayCastCompound(c, rayDown(vec3.zero(), { x: 20, y: 0, z: 0 }));
        expect(out.hit).toBe(true);
        expect(Math.abs(out.fraction - 0.2)).toBeLessThanOrEqual(1e-4);
        expect(Math.abs(out.normal.x + 1)).toBeLessThanOrEqual(1e-3);
        expect(Math.abs(out.normal.y)).toBeLessThanOrEqual(1e-3);
        expect(Math.abs(out.normal.z)).toBeLessThanOrEqual(1e-3);
    });

    test("shape cast returns the nearest child", () => {
        const mat = defaultSurfaceMaterial();
        const c = createCompound({
            spheres: [
                { sphere: { center: { x: 5, y: 0, z: 0 }, radius: 1 }, material: mat },
                { sphere: { center: { x: 10, y: 0, z: 0 }, radius: 1 }, material: mat },
            ],
        }) as CompoundData;
        const out = shapeCastCompound(c, {
            proxy: { points: [vec3.zero()], count: 1, radius: 0.25 },
            translation: { x: 20, y: 0, z: 0 },
            maxFraction: 1,
            canEncroach: false,
        });
        expect(out.hit).toBe(true);
        // Caster radius 0.25 + sphere radius 1.0 → first contact at x ≈ 3.75.
        expect(Math.abs(out.fraction - 3.75 / 20)).toBeLessThanOrEqual(1e-3);
        expect(out.childIndex).toBe(0);
    });

    test("overlap distinguishes the gap between children from a hit on one", () => {
        const mat = defaultSurfaceMaterial();
        const c = createCompound({
            spheres: [
                { sphere: { center: { x: -3, y: 0, z: 0 }, radius: 0.5 }, material: mat },
                { sphere: { center: { x: 3, y: 0, z: 0 }, radius: 0.5 }, material: mat },
            ],
        }) as CompoundData;
        const gap: ShapeProxy = { points: [vec3.zero()], count: 1, radius: 0.25 };
        expect(overlapCompound(c, xf.identity(), gap)).toBe(false);
        const hit: ShapeProxy = { points: [{ x: 3, y: 0, z: 0 }], count: 1, radius: 0.1 };
        expect(overlapCompound(c, xf.identity(), hit)).toBe(true);
    });

    // Ported from test_compound.c CompoundMover.
    test("collideMover: a mover spanning two boxes gets an up-plane from each", () => {
        const mat = defaultSurfaceMaterial();
        const box = makeBoxHull(0.5, 0.5, 0.5);
        const c = createCompound({
            hulls: [
                {
                    hull: box,
                    transform: { p: { x: -1, y: 0, z: 0 }, q: quat.identity() },
                    material: mat,
                },
                {
                    hull: box,
                    transform: { p: { x: 1, y: 0, z: 0 }, q: quat.identity() },
                    material: mat,
                },
            ],
        }) as CompoundData;

        // A capsule mover over both boxes, low enough to penetrate each +Y face.
        const mover: Capsule = {
            center1: { x: -1, y: 0.6, z: 0 },
            center2: { x: 1, y: 0.6, z: 0 },
            radius: 0.2,
        };
        const planes = collideMoverAndCompound(c, 8, mover);

        expect(planes.length).toBeGreaterThanOrEqual(2);
        const upPlanes = planes.filter((p) => p.plane.normal.y > 0.9).length;
        expect(upPlanes).toBeGreaterThanOrEqual(2);

        // The capacity cap is honored.
        const capped = collideMoverAndCompound(c, 1, mover);
        expect(capped.length).toBeLessThanOrEqual(1);
    });
});
