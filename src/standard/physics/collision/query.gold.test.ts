import { test } from "bun:test";
// queries bit-exact gold gate. Asserts the per-shape ray casts, shape casts, and overlap
// tests match the frozen historical oracle vectors bit-for-bit, over the vectors in query.gold.json.
// Current target evidence belongs to the standalone oracle. Equality, not tolerance (the README).
// The gold is a reference fixture: never edited to match.

import { type Transform, type Vec3, xf } from "../common/math";
import { defaultSurfaceMaterial, ShapeType } from "../common/types";
import {
    type CompoundData,
    createCompound,
    overlapCompound,
    rayCastCompound,
    shapeCastCompound,
} from "../shapes/compound";
import {
    type Capsule,
    overlapSphere,
    rayCastCapsule,
    rayCastSphere,
    type Sphere,
    shapeCastSphere,
} from "../shapes/geometry";
import {
    createGrid,
    overlapHeightField,
    rayCastHeightField,
    shapeCastHeightField,
} from "../shapes/heightfield";
import {
    createHull,
    type HullData,
    makeBoxHull,
    overlapHull,
    rayCastHull,
    shapeCastHull,
} from "../shapes/hull";
import { createGridMesh, type Mesh, overlapMesh, rayCastMesh, shapeCastMesh } from "../shapes/mesh";
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

test("a ray against a sphere lands on a different f32 hit point, normal or fraction than the C reference does, including the grazing and ray-origin-inside cases.", () => {
    for (const g of gold.raySphere) {
        const sphere: Sphere = {
            center: vecFromHex(g.center as string[]),
            radius: fromBits(g.radius as string),
        };
        outEqual(rayCastSphere(sphere, ray(g.origin, g.translation, g.maxFraction)), g.out, g.name);
    }
});

test("a ray against a capsule picks the wrong segment region or rounds the endcap and skew hits away from the C reference bits.", () => {
    for (const g of gold.rayCapsule) {
        const capsule: Capsule = {
            center1: vecFromHex(g.center1),
            center2: vecFromHex(g.center2),
            radius: fromBits(g.radius),
        };
        outEqual(
            rayCastCapsule(capsule, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
});

test("the hull slab clip picks a different entering plane or fraction than the C reference on face, edge, corner and interior-origin rays.", () => {
    for (const g of gold.rayHull) {
        outEqual(rayCastHull(cube, ray(g.origin, g.translation, g.maxFraction)), g.out, g.name);
    }
});

test("the GJK-backed shape cast against a sphere or hull drifts from the C reference's conservative-advancement fraction and witness point.", () => {
    for (const g of gold.shapeCast) {
        const proxy = originProxy(g.proxyRadius);
        const input = {
            proxy,
            translation: vecFromHex(g.translation),
            maxFraction: fromBits(g.maxFraction),
            canEncroach: g.canEncroach,
        };
        const out =
            g.shape === "cube"
                ? shapeCastHull(cube, input)
                : shapeCastSphere(
                      {
                          center: vecFromHex(g.center as string[]),
                          radius: fromBits(g.radius as string),
                      },
                      input,
                  );
        outEqual(out, g.out, g.name);
    }
});

test("the sphere and hull overlap predicate answers differently from the C reference at the touching and separated boundary.", () => {
    for (const g of gold.overlap) {
        const proxy = originProxy(g.proxyRadius);
        const transform: Transform = { p: vecFromHex(g.xfp), q: xf.identity().q };
        const result =
            g.shape === "cube"
                ? overlapHull(cube, transform, proxy)
                : overlapSphere(
                      {
                          center: vecFromHex(g.center as string[]),
                          radius: fromBits(g.radius as string),
                      },
                      transform,
                      proxy,
                  );
        sameValue(result, g.out, `${g.name} overlap`);
    }
});

test("a ray through the mesh BVH reports a different triangle index or hit bits than the C reference, so traversal order or triangle clipping has moved.", () => {
    for (const g of gold.rayMesh) {
        outFullEqual(
            rayCastMesh(gridMesh, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
});

test("sweeping a proxy through the mesh selects a different triangle or fraction than the C reference when several triangles are candidates.", () => {
    for (const g of gold.shapeCastMesh) {
        const input = castInput(
            proxyFrom(g.proxyPoint, g.proxyRadius),
            g.translation,
            g.maxFraction,
        );
        outFullEqual(shapeCastMesh(gridMesh, input), g.out, g.name);
    }
});

test("the mesh overlap predicate disagrees with the C reference about a proxy resting on or just clear of the grid surface.", () => {
    for (const g of gold.overlapMesh) {
        const proxy = proxyFrom(g.proxyPoint, g.proxyRadius);
        sameValue(overlapMesh(gridMesh, identityAt(g.xfp), proxy), g.out, `${g.name} overlap`);
    }
});

test("the height-field ray walk visits cells in a different order or clips the sampled column differently from the C reference.", () => {
    for (const g of gold.rayHeight) {
        outFullEqual(
            rayCastHeightField(gridField, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
});

test("sweeping a proxy across the height field returns a different cell triangle or fraction than the C reference.", () => {
    for (const g of gold.shapeCastHeight) {
        const input = castInput(
            proxyFrom(g.proxyPoint, g.proxyRadius),
            g.translation,
            g.maxFraction,
        );
        outFullEqual(shapeCastHeightField(gridField, input), g.out, g.name);
    }
});

test("the height-field overlap predicate disagrees with the C reference about a proxy at the sampled column boundary.", () => {
    for (const g of gold.overlapHeight) {
        const proxy = proxyFrom(g.proxyPoint, g.proxyRadius);
        sameValue(
            overlapHeightField(gridField, identityAt(g.xfp), proxy),
            g.out,
            `${g.name} overlap`,
        );
    }
});

test("a ray against a two-child compound reports the wrong child index or takes the farther child's hit, unlike the C reference.", () => {
    for (const g of gold.rayCompound) {
        outFullEqual(
            rayCastCompound(compound, ray(g.origin, g.translation, g.maxFraction)),
            g.out,
            g.name,
        );
    }
});

test("sweeping a proxy through a compound fails to keep the nearest child's fraction and its child index together, unlike the C reference.", () => {
    for (const g of gold.shapeCastCompound) {
        const input = castInput(
            proxyFrom(g.proxyPoint, g.proxyRadius),
            g.translation,
            g.maxFraction,
        );
        outFullEqual(shapeCastCompound(compound, input), g.out, g.name);
    }
});

test("the compound overlap predicate misses a proxy that touches only one child, or reports one in the gap between children, unlike the C reference.", () => {
    for (const g of gold.overlapCompound) {
        const proxy = proxyFrom(g.proxyPoint, g.proxyRadius);
        sameValue(overlapCompound(compound, identityAt(g.xfp), proxy), g.out, `${g.name} overlap`);
    }
});

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

test("rotated and translated kernel shape ray, cast and overlap dispatch equals TypeScript bit for bit for every query vector", () => {
    for (const g of gold.raySphere)
        kernelRay(
            ShapeType.Sphere,
            { center: vecFromHex(g.center), radius: fromBits(g.radius) },
            ray(g.origin, g.translation, g.maxFraction),
            true,
        );
    for (const g of gold.rayCapsule) {
        const capsule = {
            center1: vecFromHex(g.center1),
            center2: vecFromHex(g.center2),
            radius: fromBits(g.radius),
        };
        const input = ray(g.origin, g.translation, g.maxFraction);
        kernelRay(ShapeType.Capsule, capsule, input, true);
        kernelCast(
            ShapeType.Capsule,
            capsule,
            {
                proxy: { points: [input.origin], count: 1, radius: 0.2 },
                translation: input.translation,
                maxFraction: input.maxFraction,
                canEncroach: false,
            },
            true,
        );
        kernelOverlap(
            ShapeType.Capsule,
            capsule,
            xf.identity(),
            { points: [input.origin], count: 1, radius: 0.2 },
            true,
        );
    }
    for (const g of gold.rayHull)
        kernelRay(ShapeType.Hull, cube, ray(g.origin, g.translation, g.maxFraction), true);
    for (const g of gold.shapeCast) {
        const geometry =
            g.shape === "cube"
                ? cube
                : {
                      center: vecFromHex(g.center as string[]),
                      radius: fromBits(g.radius as string),
                  };
        kernelCast(
            g.shape === "cube" ? ShapeType.Hull : ShapeType.Sphere,
            geometry,
            {
                proxy: originProxy(g.proxyRadius),
                translation: vecFromHex(g.translation),
                maxFraction: fromBits(g.maxFraction),
                canEncroach: g.canEncroach,
            },
            true,
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
        kernelOverlap(
            g.shape === "cube" ? ShapeType.Hull : ShapeType.Sphere,
            geometry,
            identityAt(g.xfp),
            originProxy(g.proxyRadius),
            true,
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
            kernelRay(kind, geometry, ray(g.origin, g.translation, g.maxFraction), true);
        for (const g of casts)
            kernelCast(
                kind,
                geometry,
                castInput(proxyFrom(g.proxyPoint, g.proxyRadius), g.translation, g.maxFraction),
                true,
            );
        for (const g of overlaps)
            kernelOverlap(
                kind,
                geometry,
                identityAt(g.xfp),
                proxyFrom(g.proxyPoint, g.proxyRadius),
                true,
            );
    }
});
