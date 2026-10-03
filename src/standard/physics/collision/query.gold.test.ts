import { test } from "bun:test";
// queries bit-exact gold gate. Asserts the per-shape ray casts, shape casts, and overlap
// tests match the frozen historical oracle vectors bit-for-bit, over the vectors in query.gold.json.
// Current target evidence belongs to the standalone oracle. Equality, not tolerance (the README).
// The gold is a reference fixture: never edited to match.

import { type Transform, type Vec3, xf } from "../common/math";
import { defaultSurfaceMaterial, ShapeType } from "../common/types";
import { type CompoundData, createCompound } from "../shapes/compound";
import type { Capsule, Sphere } from "../shapes/geometry";
import { createGrid } from "../shapes/heightfield";
import { createHull, type HullData, makeBoxHull } from "../shapes/hull";
import { createGridMesh, type Mesh } from "../shapes/mesh";
import type { CastOutput, ShapeProxy } from "./distance";
import gold from "./query.gold.json";
import { kernelCast, kernelOverlap, kernelRay } from "./shape_query_gold";

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
function sameValue<T>(got: T, want: T, label: string): void {
    if (!Object.is(got, want)) {
        throw new Error(`${label}: got ${String(got)}, want ${String(want)}`);
    }
}
const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const vecFromHex = (a: string[]): Vec3 => v(fromBits(a[0]), fromBits(a[1]), fromBits(a[2]));
function vecEqual(got: Vec3, want: string[], label: string): void {
    bitEqual(got.x, want[0], `${label}.x`);
    bitEqual(got.y, want[1], `${label}.y`);
    bitEqual(got.z, want[2], `${label}.z`);
}
function outEqual(
    out: CastOutput,
    want: { hit: boolean; fraction: string; point: string[]; normal: string[] },
    name: string,
): void {
    sameValue(out.hit, want.hit, `${name} hit`);
    bitEqual(out.fraction, want.fraction, `${name} fraction`);
    vecEqual(out.point, want.point, `${name} point`);
    vecEqual(out.normal, want.normal, `${name} normal`);
}

// The mesh/height/compound query paths carry the extra output fields; assert them too.
function outFullEqual(
    out: CastOutput,
    want: {
        hit: boolean;
        fraction: string;
        point: string[];
        normal: string[];
        triangleIndex: number;
        childIndex: number;
        materialIndex: number;
    },
    name: string,
): void {
    outEqual(out, want, name);
    sameValue(out.triangleIndex, want.triangleIndex, `${name} triangleIndex`);
    sameValue(out.childIndex, want.childIndex, `${name} childIndex`);
    sameValue(out.materialIndex, want.materialIndex, `${name} materialIndex`);
}

// The cube hull, baked identically to fixtures/query_gold.c's b3CreateHull(cubeCorners, 8, 8).
const cubeCorners: Vec3[] = [
    v(-1, -1, -1),
    v(1, -1, -1),
    v(1, 1, -1),
    v(-1, 1, -1),
    v(-1, -1, 1),
    v(1, -1, 1),
    v(1, 1, 1),
    v(-1, 1, 1),
];
const cube = createHull(cubeCorners, 8) as HullData;

// The proxy point cloud is a single point at the origin (pt0/sphere), mirroring query_gold.c.
const originProxy = (radiusHex: string): ShapeProxy => ({
    points: [v(0, 0, 0)],
    count: 1,
    radius: fromBits(radiusHex),
});

const ray = (origin: string[], translation: string[], maxFraction: string) => ({
    origin: vecFromHex(origin),
    translation: vecFromHex(translation),
    maxFraction: fromBits(maxFraction),
});

const proxyFrom = (point: string[], radius: string): ShapeProxy => ({
    points: [vecFromHex(point)],
    count: 1,
    radius: fromBits(radius),
});
const castInput = (proxy: ShapeProxy, translation: string[], maxFraction: string) => ({
    proxy,
    translation: vecFromHex(translation),
    maxFraction: fromBits(maxFraction),
    canEncroach: false,
});
const identityAt = (p: string[]): Transform => ({ p: vecFromHex(p), q: xf.identity().q });

// Geometry reconstructed with the exact builders fixtures/query_gold.c uses; construction is proven
// bit-exact by the geometry gold, so this reuses it and validates only the query paths.
const gridMesh: Mesh = { data: createGridMesh(4, 4, 1, 0, true), scale: v(1, 1, 1) };
const gridField = createGrid(8, 8, v(1, 1, 1), false);
const slab = makeBoxHull(1.5, 0.25, 1.5);
const cmat = defaultSurfaceMaterial();
const compound = createCompound({
    hulls: [
        { hull: slab, transform: identityAt(["3fb33333", "be800000", "00000000"]), material: cmat },
        { hull: slab, transform: identityAt(["bfb33333", "be800000", "00000000"]), material: cmat },
    ],
}) as CompoundData;

// The per-kind level is the frozen authority; shape-level rotation canonicalizes -0 as Box3D does.
test("world-created kernel shapes answer every immutable per-shape query vector bit-exactly", () => {
    for (const g of gold.raySphere) {
        const sphere: Sphere = { center: vecFromHex(g.center), radius: fromBits(g.radius) };
        outEqual(
            kernelRay(ShapeType.Sphere, sphere, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
    for (const g of gold.rayCapsule) {
        const capsule: Capsule = {
            center1: vecFromHex(g.center1),
            center2: vecFromHex(g.center2),
            radius: fromBits(g.radius),
        };
        outEqual(
            kernelRay(ShapeType.Capsule, capsule, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
    for (const g of gold.rayHull)
        outEqual(
            kernelRay(ShapeType.Hull, cube, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    for (const g of gold.shapeCast) {
        const input = {
            proxy: originProxy(g.proxyRadius),
            translation: vecFromHex(g.translation),
            maxFraction: fromBits(g.maxFraction),
            canEncroach: g.canEncroach,
        };
        const geometry =
            g.shape === "cube"
                ? cube
                : {
                      center: vecFromHex(g.center as string[]),
                      radius: fromBits(g.radius as string),
                  };
        outEqual(
            kernelCast(g.shape === "cube" ? ShapeType.Hull : ShapeType.Sphere, geometry, input),
            g.out,
            g.name,
        );
    }
    for (const g of gold.overlap) {
        const geometry =
            g.shape === "cube"
                ? cube
                : {
                      center: vecFromHex(g.center as string[]),
                      radius: fromBits(g.radius as string),
                  };
        sameValue(
            kernelOverlap(
                g.shape === "cube" ? ShapeType.Hull : ShapeType.Sphere,
                geometry,
                identityAt(g.xfp),
                originProxy(g.proxyRadius),
            ),
            g.out,
            g.name,
        );
    }
    for (const [kind, geometry, rays, casts, overlaps] of [
        [ShapeType.Mesh, gridMesh, gold.rayMesh, gold.shapeCastMesh, gold.overlapMesh],
        [
            ShapeType.HeightField,
            gridField,
            gold.rayHeight,
            gold.shapeCastHeight,
            gold.overlapHeight,
        ],
        [
            ShapeType.Compound,
            compound,
            gold.rayCompound,
            gold.shapeCastCompound,
            gold.overlapCompound,
        ],
    ] as const) {
        for (const g of rays)
            outFullEqual(
                kernelRay(kind, geometry, ray(g.origin, g.translation, g.maxFraction)),
                g.out,
                g.name,
            );
        for (const g of casts)
            outFullEqual(
                kernelCast(
                    kind,
                    geometry,
                    castInput(proxyFrom(g.proxyPoint, g.proxyRadius), g.translation, g.maxFraction),
                ),
                g.out,
                g.name,
            );
        for (const g of overlaps)
            sameValue(
                kernelOverlap(
                    kind,
                    geometry,
                    identityAt(g.xfp),
                    proxyFrom(g.proxyPoint, g.proxyRadius),
                ),
                g.out,
                g.name,
            );
    }
});
