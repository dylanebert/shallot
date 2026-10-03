// character mover bit-exact gold gate. Asserts the plane solver (solvePlanes), the
// velocity clip (clipVector), and kernel mover queries over sphere, capsule, hull, mesh, height field
// and compound shapes match the frozen historical oracle vectors bit-for-
// bit, over the vectors in mover.gold.json. Current target evidence belongs to the standalone oracle;
// equality, not tolerance (the README).

import { expect, test } from "bun:test";
import { type Transform, type Vec3, xf } from "../common/math";
import { defaultSurfaceMaterial, ShapeType } from "../common/types";
import { type CompoundData, createCompound } from "../shapes/compound";
import type { Capsule, Sphere } from "../shapes/geometry";
import { createGrid } from "../shapes/heightfield";
import { createHull, type HullData, makeBoxHull } from "../shapes/hull";
import { createGridMesh, type Mesh } from "../shapes/mesh";
import { type CollisionPlane, clipVector, type PlaneResult, solvePlanes } from "./mover";
import gold from "./mover.gold.json";
import { kernelMover } from "./shape_query_gold";

const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}
function bitEqual(got: number, want: string, label: string): void {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const vecFromHex = (a: string[]): Vec3 => v(fromBits(a[0]), fromBits(a[1]), fromBits(a[2]));
function vecEqual(got: Vec3, want: string[], label: string): void {
    bitEqual(got.x, want[0], `${label}.x`);
    bitEqual(got.y, want[1], `${label}.y`);
    bitEqual(got.z, want[2], `${label}.z`);
}

type PlaneJson = {
    normal: string[];
    offset: string;
    pushLimit: string;
    push: string;
    clipVelocity: boolean;
};

// Reconstruct a CollisionPlane from its gold entry. b3SolvePlanes resets push at entry, so the gold's
// (post-run) push is only load-bearing for clipVector.
const collisionPlane = (p: PlaneJson): CollisionPlane => ({
    plane: { normal: vecFromHex(p.normal), offset: fromBits(p.offset) },
    pushLimit: fromBits(p.pushLimit),
    push: fromBits(p.push),
    clipVelocity: p.clipVelocity,
});

// Compare a returned plane list against gold's count + per-plane {normal, offset, point}.
function planesEqual(
    got: PlaneResult[],
    want: { count: number; planes: { normal: string[]; offset: string; point: string[] }[] },
    name: string,
): void {
    if (got.length !== want.count) {
        throw new Error(`${name}: got ${got.length} planes, want ${want.count}`);
    }
    for (let i = 0; i < want.count; ++i) {
        vecEqual(got[i].plane.normal, want.planes[i].normal, `${name}[${i}].normal`);
        bitEqual(got[i].plane.offset, want.planes[i].offset, `${name}[${i}].offset`);
        vecEqual(got[i].point, want.planes[i].point, `${name}[${i}].point`);
    }
}

const moverFrom = (m: { center1: string[]; center2: string[]; radius: string }): Capsule => ({
    center1: vecFromHex(m.center1),
    center2: vecFromHex(m.center2),
    radius: fromBits(m.radius),
});

test("the mover plane solver drifts from the Box3D C reference in its delta bits or iteration count, so a character resolves penetration differently than the pinned reference", () => {
    for (const g of gold.solvePlanes) {
        const planes = g.planes.map((p) => collisionPlane(p as PlaneJson));
        const result = solvePlanes(vecFromHex(g.target), planes, planes.length);
        vecEqual(result.delta, g.delta, `${g.name} delta`);
        if (result.iterationCount !== g.iterationCount) {
            throw new Error(
                `${g.name} iterationCount: got ${result.iterationCount}, want ${g.iterationCount}`,
            );
        }
    }
    expect(gold.solvePlanes.length).toBeGreaterThan(0);
});

test("the mover velocity clip drifts from the Box3D C reference, so a character keeps or loses velocity against a contact plane differently than the pinned reference", () => {
    for (const g of gold.clipVector) {
        const planes = g.planes.map((p) => collisionPlane(p as PlaneJson));
        const out = clipVector(vecFromHex(g.vector), planes, planes.length);
        vecEqual(out, g.out, `${g.name} out`);
    }
    expect(gold.clipVector.length).toBeGreaterThan(0);
});

// The box hull, baked identically to fixtures/mover_gold.c's b3CreateHull(boxCorners, 8, 8).
const boxCorners: Vec3[] = [
    v(-0.5, -0.5, -0.5),
    v(0.5, -0.5, -0.5),
    v(0.5, 0.5, -0.5),
    v(-0.5, 0.5, -0.5),
    v(-0.5, -0.5, 0.5),
    v(0.5, -0.5, 0.5),
    v(0.5, 0.5, 0.5),
    v(-0.5, 0.5, 0.5),
];
const box = createHull(boxCorners, 8) as HullData;

// Mesh / height field / compound reconstructed with the same builders the C gold used.
const gridMesh: Mesh = { data: createGridMesh(4, 4, 1, 0, true), scale: v(1, 1, 1) };
const gridField = createGrid(8, 8, v(1, 1, 1), false);
const slab = makeBoxHull(0.5, 0.5, 0.5);
const cmat = defaultSurfaceMaterial();
const identityAt = (x: number, y: number, z: number): Transform => ({
    p: v(x, y, z),
    q: xf.identity().q,
});
const compound = createCompound({
    hulls: [
        { hull: slab, transform: identityAt(-1, 0, 0), material: cmat },
        { hull: slab, transform: identityAt(1, 0, 0), material: cmat },
    ],
}) as CompoundData;

// The per-kind level is the frozen authority; shape-level rotation canonicalizes -0 as Box3D does.
test("world-created kernel shapes answer every immutable mover collision vector bit-exactly", () => {
    for (const g of gold.sphere) {
        const sphere: Sphere = { center: vecFromHex(g.center), radius: fromBits(g.radius) };
        planesEqual(kernelMover(ShapeType.Sphere, sphere, moverFrom(g.mover)), g, g.name);
    }
    for (const g of gold.capsule) {
        const capsule: Capsule = {
            center1: vecFromHex(g.center1),
            center2: vecFromHex(g.center2),
            radius: fromBits(g.radius),
        };
        planesEqual(kernelMover(ShapeType.Capsule, capsule, moverFrom(g.mover)), g, g.name);
    }
    for (const g of gold.hull)
        planesEqual(kernelMover(ShapeType.Hull, box, moverFrom(g.mover)), g, g.name);
    for (const [kind, geometry, cases] of [
        [ShapeType.Mesh, gridMesh, gold.mesh],
        [ShapeType.HeightField, gridField, gold.height],
        [ShapeType.Compound, compound, gold.compound],
    ] as const) {
        for (const g of cases)
            planesEqual(kernelMover(kind, geometry, moverFrom(g.mover)), g, g.name);
    }
});
