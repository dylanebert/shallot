import { describe, expect, test } from "bun:test";
import gold from "./geometry.gold.json";
import {
    cloneHull,
    computeHullAABB,
    computeHullMass,
    createCone,
    createCylinder,
    createHull,
    createRock,
    findHullSupportFace,
    findHullSupportVertex,
    type HullData,
    makeBoxHull,
    makeTransformedBoxHull,
} from "./hull";
import { f32, quat, type Vec3, vec3, xf } from "./math";

// Reconstruct an exact f32 from the raw hex bits the C generator emitted.
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

type HullGold = (typeof gold.hulls)[number];

// Every geometric field of a built hull, bit-for-bit against the C reference: topology (counts +
// integer indices) and float data (points, planes, center, inertia, mass scalars, aabb).
function assertHull(hull: HullData, g: HullGold) {
    expect(hull.vertexCount).toBe(g.vertexCount);
    expect(hull.edgeCount).toBe(g.edgeCount);
    expect(hull.faceCount).toBe(g.faceCount);

    for (let i = 0; i < hull.vertexCount; ++i) {
        const p = hull.points[i];
        bitEqual(p.x, g.points[i][0], `${g.name} points[${i}].x`);
        bitEqual(p.y, g.points[i][1], `${g.name} points[${i}].y`);
        bitEqual(p.z, g.points[i][2], `${g.name} points[${i}].z`);
        expect(hull.vertices[i].edge).toBe(g.vertexEdge[i]);
    }

    for (let i = 0; i < hull.edgeCount; ++i) {
        const e = hull.edges[i];
        expect([e.next, e.twin, e.origin, e.face]).toEqual([
            g.edges[4 * i + 0],
            g.edges[4 * i + 1],
            g.edges[4 * i + 2],
            g.edges[4 * i + 3],
        ]);
    }

    for (let i = 0; i < hull.faceCount; ++i) {
        expect(hull.faces[i].edge).toBe(g.faceEdge[i]);
        const pl = hull.planes[i];
        bitEqual(pl.normal.x, g.planes[i][0], `${g.name} planes[${i}].nx`);
        bitEqual(pl.normal.y, g.planes[i][1], `${g.name} planes[${i}].ny`);
        bitEqual(pl.normal.z, g.planes[i][2], `${g.name} planes[${i}].nz`);
        bitEqual(pl.offset, g.planes[i][3], `${g.name} planes[${i}].offset`);
    }

    bitEqual(hull.center.x, g.center[0], `${g.name} center.x`);
    bitEqual(hull.center.y, g.center[1], `${g.name} center.y`);
    bitEqual(hull.center.z, g.center[2], `${g.name} center.z`);

    const ci = [
        hull.centralInertia.cx.x,
        hull.centralInertia.cx.y,
        hull.centralInertia.cx.z,
        hull.centralInertia.cy.x,
        hull.centralInertia.cy.y,
        hull.centralInertia.cy.z,
        hull.centralInertia.cz.x,
        hull.centralInertia.cz.y,
        hull.centralInertia.cz.z,
    ];
    for (let i = 0; i < 9; ++i) bitEqual(ci[i], g.centralInertia[i], `${g.name} inertia[${i}]`);

    bitEqual(hull.volume, g.volume, `${g.name} volume`);
    bitEqual(hull.surfaceArea, g.surfaceArea, `${g.name} surfaceArea`);
    bitEqual(hull.innerRadius, g.innerRadius, `${g.name} innerRadius`);

    bitEqual(hull.aabb.lowerBound.x, g.aabbLower[0], `${g.name} aabbLower.x`);
    bitEqual(hull.aabb.lowerBound.y, g.aabbLower[1], `${g.name} aabbLower.y`);
    bitEqual(hull.aabb.lowerBound.z, g.aabbLower[2], `${g.name} aabbLower.z`);
    bitEqual(hull.aabb.upperBound.x, g.aabbUpper[0], `${g.name} aabbUpper.x`);
    bitEqual(hull.aabb.upperBound.y, g.aabbUpper[1], `${g.name} aabbUpper.y`);
    bitEqual(hull.aabb.upperBound.z, g.aabbUpper[2], `${g.name} aabbUpper.z`);
}

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });

const cubeCorners: Vec3[] = [
    v(1, 1, 1),
    v(-1, 1, 1),
    v(-1, -1, 1),
    v(1, -1, 1),
    v(1, 1, -1),
    v(-1, 1, -1),
    v(-1, -1, -1),
    v(1, -1, -1),
];

const tetCorners: Vec3[] = [v(0, 0, 0), v(1, 0, 0), v(0, 1, 0), v(0, 0, 1)];

const redundantCloud: Vec3[] = [
    v(1, 1, 1),
    v(-1, 1, 1),
    v(-1, -1, 1),
    v(1, -1, 1),
    v(1, 1, -1),
    v(-1, 1, -1),
    v(-1, -1, -1),
    v(1, -1, -1),
    v(1, 1, 1),
    v(1, 1, 1),
    v(0, 0, 0),
    v(0.5, 0, 0),
    v(0, 0.5, 0),
    v(0, 0, 0.5),
    v(-0.5, 0, 0),
    v(0, -0.5, 0),
    v(0, 0, -0.5),
    v(0.25, 0.25, 0.25),
    v(-0.25, -0.25, -0.25),
    v(0.5, 0.5, 0.5),
];

const skewCloud: Vec3[] = [
    v(0, 0, 0),
    v(2, 0, 0),
    v(0, 3, 0),
    v(0, 0, 1),
    v(1.5, 1.5, 0.5),
    v(-0.5, 0.5, 0.5),
    v(0.5, -0.5, 0.5),
];

const goldHull = (name: string) => gold.hulls.find((h) => h.name === name) as HullGold;
const goldBox = (name: string) => gold.boxHulls.find((h) => h.name === name) as HullGold;

describe("hull bit-exact vs C reference", () => {
    test("cube", () => assertHull(createHull(cubeCorners, 8) as HullData, goldHull("cube")));
    test("tetrahedron", () =>
        assertHull(createHull(tetCorners, 4) as HullData, goldHull("tetrahedron")));
    test("redundant input", () =>
        assertHull(createHull(redundantCloud, 8) as HullData, goldHull("redundant")));
    test("skew pentahedron", () =>
        assertHull(createHull(skewCloud, 8) as HullData, goldHull("skew")));
    test("cylinder", () => assertHull(createCylinder(2, 1, 0, 8), goldHull("cylinder")));
    test("cylinder6 (nonzero yOffset)", () =>
        assertHull(createCylinder(3, 0.75, 0.25, 6), goldHull("cylinder6")));
    test("cone", () => assertHull(createCone(2, 1, 0.5, 8), goldHull("cone")));
    test("rock", () => assertHull(createRock(1), goldHull("rock")));
});

describe("box hull bit-exact vs C reference", () => {
    test("unit", () => assertHull(makeBoxHull(1, 1, 1), goldBox("unit")));
    test("oblong", () => assertHull(makeBoxHull(0.5, 1, 2), goldBox("oblong")));
    test("transformed", () => {
        // f32-round the literals to match the C `0.3f`/`0.6f` axis + angle bit-for-bit before
        // normalize/fromAxisAngle (both verified bit-exact), else f64 literals diverge by 1 ULP.
        const axis = vec3.normalize(v(f32(0.3), f32(0.7), f32(0.2)));
        const q = quat.fromAxisAngle(axis, f32(0.6));
        const hull = makeTransformedBoxHull(0.75, 1.25, 0.5, { p: v(0.5, -0.25, 1), q });
        assertHull(hull, goldBox("transformed"));
    });
});

// --- topology invariants + rejection, ported from test_hull.c ---------------------------------

// Euler's identity for a convex polyhedron: V - E + F = 2 (E = edgeCount / 2).
const euler = (h: HullData) => h.vertexCount - h.edgeCount / 2 + h.faceCount;

// XorShift32 + Shoemake unit-vector recipe, matching FillSphereSample in test_hull.c exactly so
// the same seeds drive the builder over the same conflict/merge cascades. Uses JS trig (the
// generated cloud only needs to be a valid convex-input sphere sample; invariants are order-free).
function fillSphereSample(count: number, seed: number): Vec3[] {
    const RandLimit = 32767;
    let s = seed >>> 0;
    const out: Vec3[] = [];
    for (let i = 0; i < count; ++i) {
        const u: number[] = [];
        for (let k = 0; k < 3; ++k) {
            s = (s ^ (s << 13)) >>> 0;
            s = (s ^ (s >>> 17)) >>> 0;
            s = (s ^ (s << 5)) >>> 0;
            u.push((s & RandLimit) / RandLimit);
        }
        const u1 = u[0];
        const u2 = 2 * Math.PI * u[1];
        const u3 = 2 * Math.PI * u[2];
        const sqrt1MinusU1 = Math.sqrt(1 - u1);
        const sqrtU1 = Math.sqrt(u1);
        out.push({
            x: sqrt1MinusU1 * Math.sin(u2),
            y: sqrt1MinusU1 * Math.cos(u2),
            z: sqrtU1 * Math.sin(u3),
        });
    }
    return out;
}

describe("hull topology + rejection", () => {
    test("cube counts satisfy Euler", () => {
        const h = createHull(cubeCorners, 8) as HullData;
        expect([h.vertexCount, h.edgeCount, h.faceCount]).toEqual([8, 24, 6]);
        expect(euler(h)).toBe(2);
    });

    test("tetrahedron counts satisfy Euler", () => {
        const h = createHull(tetCorners, 4) as HullData;
        expect([h.vertexCount, h.edgeCount, h.faceCount]).toEqual([4, 12, 4]);
        expect(euler(h)).toBe(2);
    });

    test("maxVertexCount is honored as a strict cap and clamps to [4, 255]", () => {
        const cloud = fillSphereSample(64, 12345);
        const capped = createHull(cloud, 8) as HullData;
        expect(capped.vertexCount).toBeLessThanOrEqual(8);

        const floored = createHull(cloud, 1) as HullData;
        expect(floored.vertexCount).toBeGreaterThanOrEqual(4);
        expect(floored.vertexCount).toBeLessThanOrEqual(255);

        const ceilinged = createHull(cloud, 1000) as HullData;
        expect(ceilinged.vertexCount).toBeGreaterThanOrEqual(4);
        expect(ceilinged.vertexCount).toBeLessThanOrEqual(255);
    });

    test("dense sphere samples build valid hulls (merge-cascade stress)", () => {
        const seeds = [12345, 1, 0xdeadbeef, 0xcafef00d];
        for (const seed of seeds) {
            const cloud = fillSphereSample(512, seed);
            for (const M of [16, 24, 32, 40]) {
                const h = createHull(cloud, M) as HullData;
                expect(h).not.toBeNull();
                expect(h.vertexCount).toBeGreaterThanOrEqual(4);
                expect(h.vertexCount).toBeLessThanOrEqual(M);
                expect(h.faceCount).toBeGreaterThanOrEqual(4);
                expect(euler(h)).toBe(2);
            }
        }
    });

    test("degenerate inputs are rejected", () => {
        const collinear: Vec3[] = [];
        for (let i = 0; i < 8; ++i) collinear.push(v(i, 0, 0));
        expect(createHull(collinear.slice(0, 3), 8)).toBeNull(); // fewer than 4
        expect(createHull([], 8)).toBeNull(); // empty
        expect(createHull(collinear, 8)).toBeNull(); // collinear

        const coincident: Vec3[] = [];
        for (let i = 0; i < 8; ++i) coincident.push(v(1, 2, 3));
        expect(createHull(coincident, 8)).toBeNull(); // coincident

        const coplanar: Vec3[] = [
            v(0, 0, 0),
            v(1, 0, 0),
            v(0, 1, 0),
            v(1, 1, 0),
            v(2, 0.5, 0),
            v(0.5, 2, 0),
        ];
        expect(createHull(coplanar, 8)).toBeNull(); // coplanar
    });
});

describe("hull support + AABB", () => {
    test("support vertex is the extreme along a direction", () => {
        const cube = createHull(cubeCorners, 8) as HullData;
        expect(cube.points[findHullSupportVertex(cube, v(1, 0, 0))].x).toBe(1);
        expect(cube.points[findHullSupportVertex(cube, v(0, -1, 0))].y).toBe(-1);
    });

    test("support face normal is aligned with the direction", () => {
        const cube = createHull(cubeCorners, 8) as HullData;
        const n = cube.planes[findHullSupportFace(cube, v(1, 0, 0))].normal;
        expect(n.x).toBeGreaterThan(0.99);
        expect(n.x).toBeGreaterThan(n.y);
        expect(n.x).toBeGreaterThan(n.z);
    });

    test("computeHullAABB under identity is the local AABB, and translates", () => {
        const cube = createHull(cubeCorners, 8) as HullData;
        const local = computeHullAABB(cube, xf.identity());
        expect(local.lowerBound).toEqual(cube.aabb.lowerBound);
        expect(local.upperBound).toEqual(cube.aabb.upperBound);

        const moved = computeHullAABB(cube, { p: v(1, 2, 3), q: quat.identity() });
        expect(moved.lowerBound).toEqual(v(0, 1, 2));
        expect(moved.upperBound).toEqual(v(2, 3, 4));
    });
});

describe("hull determinism + clone", () => {
    test("same input builds an identical hull and hash", () => {
        const h1 = createHull(cubeCorners, 8) as HullData;
        const h2 = createHull(cubeCorners, 8) as HullData;
        expect(h1.hash).not.toBe(0);
        expect(h2.hash).toBe(h1.hash);
        expect(h2).toEqual(h1);
    });

    test("computeHullMass reads volume and passes the center through", () => {
        // Unit cube: volume 8, centered at the origin. Density 2 -> mass 16.
        const cube = createHull(cubeCorners, 8) as HullData;
        const mass = computeHullMass(cube, 2);
        expect(mass.mass).toBe(16);
        expect(mass.center).toEqual(v(0, 0, 0));
    });

    test("clone is a deep, equal copy", () => {
        const original = createHull(cubeCorners, 8) as HullData;
        const clone = cloneHull(original);
        expect(clone).toEqual(original);
        expect(clone.points).not.toBe(original.points);
        clone.points[0].x = 99;
        expect(original.points[0].x).not.toBe(99);
    });
});
