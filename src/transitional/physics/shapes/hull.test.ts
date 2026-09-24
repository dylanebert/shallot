import { expect } from "bun:test";
import { check } from "../../../harness/check";
import { f32, quat, type Vec3, vec3, xf } from "../common/math";
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

check(
    "hull bit-exact vs C reference",
    {
        claim: "the convex hull builder's quickhull output — points, half-edges, planes, center, inertia, volume, surface area, inner radius and aabb — diverges by a bit from the Box3D C reference on any of the cube, tetrahedron, redundant-cloud, skew, cylinder, cylinder6, cone or rock cases",
    },
    () => {
        const cases: [string, () => HullData][] = [
            ["cube", () => createHull(cubeCorners, 8) as HullData],
            ["tetrahedron", () => createHull(tetCorners, 4) as HullData],
            ["redundant", () => createHull(redundantCloud, 8) as HullData],
            ["skew", () => createHull(skewCloud, 8) as HullData],
            ["cylinder", () => createCylinder(2, 1, 0, 8)],
            ["cylinder6", () => createCylinder(3, 0.75, 0.25, 6)],
            ["cone", () => createCone(2, 1, 0.5, 8)],
            ["rock", () => createRock(1)],
        ];
        for (const [name, build] of cases) {
            assertHull(build(), goldHull(name));
        }
    },
);

check(
    "box hull bit-exact vs C reference",
    {
        claim: "a convex hull built from box half-extents — unit, oblong, or rotated by a quaternion through makeTransformedBoxHull — diverges by a bit from the Box3D C reference",
    },
    () => {
        assertHull(makeBoxHull(1, 1, 1), goldBox("unit"));
        assertHull(makeBoxHull(0.5, 1, 2), goldBox("oblong"));

        // f32-round the literals to match the C `0.3f`/`0.6f` axis + angle bit-for-bit before
        // normalize/fromAxisAngle (both verified bit-exact), else f64 literals diverge by 1 ULP.
        const axis = vec3.normalize(v(f32(0.3), f32(0.7), f32(0.2)));
        const q = quat.fromAxisAngle(axis, f32(0.6));
        assertHull(
            makeTransformedBoxHull(0.75, 1.25, 0.5, { p: v(0.5, -0.25, 1), q }),
            goldBox("transformed"),
        );
    },
);

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

check(
    "hull topology satisfies Euler",
    {
        claim: "a convex hull leaves a broken half-edge topology — counts off the expected 8/24/6 cube or 4/12/4 tetrahedron, or V - E + F away from 2 on dense sphere clouds driven through the merge cascade",
    },
    () => {
        const cube = createHull(cubeCorners, 8) as HullData;
        expect([cube.vertexCount, cube.edgeCount, cube.faceCount]).toEqual([8, 24, 6]);
        expect(euler(cube)).toBe(2);

        const tet = createHull(tetCorners, 4) as HullData;
        expect([tet.vertexCount, tet.edgeCount, tet.faceCount]).toEqual([4, 12, 4]);
        expect(euler(tet)).toBe(2);

        for (const seed of [12345, 1, 0xdeadbeef, 0xcafef00d]) {
            const cloud = fillSphereSample(512, seed);
            for (const M of [16, 24, 32, 40]) {
                const label = `sphere seed ${seed} max ${M}`;
                const h = createHull(cloud, M) as HullData;
                if (h === null) throw new Error(`${label}: builder returned null`);
                if (h.vertexCount < 4 || h.vertexCount > M) {
                    throw new Error(`${label}: vertexCount ${h.vertexCount} outside [4, ${M}]`);
                }
                if (h.faceCount < 4) throw new Error(`${label}: faceCount ${h.faceCount} below 4`);
                if (euler(h) !== 2) throw new Error(`${label}: V - E + F is ${euler(h)}, want 2`);
            }
        }
    },
);

check(
    "hull vertex cap is honored and clamped",
    {
        claim: "the convex hull builder overruns its maxVertexCount cap or fails to clamp an out-of-range cap into [4, 255]",
    },
    () => {
        const cloud = fillSphereSample(64, 12345);
        const cases: [string, number][] = [
            ["cap 8", 8],
            ["cap 1 (floored)", 1],
            ["cap 1000 (ceilinged)", 1000],
        ];
        for (const [label, max] of cases) {
            const h = createHull(cloud, max) as HullData;
            if (h === null) throw new Error(`${label}: builder returned null`);
            const upper = max === 8 ? 8 : 255;
            if (h.vertexCount < 4 || h.vertexCount > upper) {
                throw new Error(`${label}: vertexCount ${h.vertexCount} outside [4, ${upper}]`);
            }
        }
    },
);

check(
    "degenerate hull inputs are rejected",
    {
        claim: "the convex hull builder returns a hull instead of null for a degenerate point cloud: empty, fewer than four points, collinear, coincident or coplanar",
    },
    () => {
        const collinear: Vec3[] = [];
        for (let i = 0; i < 8; ++i) collinear.push(v(i, 0, 0));
        const coincident: Vec3[] = [];
        for (let i = 0; i < 8; ++i) coincident.push(v(1, 2, 3));
        const coplanar: Vec3[] = [
            v(0, 0, 0),
            v(1, 0, 0),
            v(0, 1, 0),
            v(1, 1, 0),
            v(2, 0.5, 0),
            v(0.5, 2, 0),
        ];

        const cases: [string, Vec3[]][] = [
            ["fewer than 4", collinear.slice(0, 3)],
            ["empty", []],
            ["collinear", collinear],
            ["coincident", coincident],
            ["coplanar", coplanar],
        ];
        for (const [label, cloud] of cases) {
            const h = createHull(cloud, 8);
            if (h !== null) throw new Error(`${label}: expected null, got a hull`);
        }
    },
);

check(
    "hull support queries and world AABB",
    {
        claim: "a convex hull's support vertex is not the extreme point along the query direction, its support face normal is not the one aligned with that direction, or computeHullAABB fails to reproduce the local aabb under identity and translate it",
    },
    () => {
        const cube = createHull(cubeCorners, 8) as HullData;

        expect(cube.points[findHullSupportVertex(cube, v(1, 0, 0))].x).toBe(1);
        expect(cube.points[findHullSupportVertex(cube, v(0, -1, 0))].y).toBe(-1);

        const n = cube.planes[findHullSupportFace(cube, v(1, 0, 0))].normal;
        expect(n.x).toBeGreaterThan(0.99);
        expect(n.x).toBeGreaterThan(n.y);
        expect(n.x).toBeGreaterThan(n.z);

        const local = computeHullAABB(cube, xf.identity());
        expect(local.lowerBound).toEqual(cube.aabb.lowerBound);
        expect(local.upperBound).toEqual(cube.aabb.upperBound);

        const moved = computeHullAABB(cube, { p: v(1, 2, 3), q: quat.identity() });
        expect(moved.lowerBound).toEqual(v(0, 1, 2));
        expect(moved.upperBound).toEqual(v(2, 3, 4));
    },
);

check(
    "hull determinism, mass and clone",
    {
        claim: "the convex hull builder is nondeterministic across two identical builds (structure or hash), computeHullMass ignores the hull's volume or center, or cloneHull returns a shallow copy that aliases the original's points",
    },
    () => {
        const h1 = createHull(cubeCorners, 8) as HullData;
        const h2 = createHull(cubeCorners, 8) as HullData;
        expect(h1.hash).not.toBe(0);
        expect(h2.hash).toBe(h1.hash);
        expect(h2).toEqual(h1);

        // Unit cube: volume 8, centered at the origin. Density 2 -> mass 16.
        const mass = computeHullMass(h1, 2);
        expect(mass.mass).toBe(16);
        expect(mass.center).toEqual(v(0, 0, 0));

        const clone = cloneHull(h1);
        expect(clone).toEqual(h1);
        expect(clone.points).not.toBe(h1.points);
        clone.points[0].x = 99;
        expect(h1.points[0].x).not.toBe(99);
    },
);
