import { expect } from "bun:test";
import { check } from "../../../harness/check";
import {
    isWithinSegments,
    lineDistance,
    pointToSegmentDistance,
    type Transform,
    type Vec3,
} from "../common/math";
import type { Capsule, Sphere } from "../shapes/geometry";
import { createCylinder, type HullData, makeBoxHull } from "../shapes/hull";
import { emptyCache } from "./distance";
import {
    collideCapsuleAndSphere,
    collideCapsules,
    collideHullAndCapsule,
    collideHullAndSphere,
    collideHulls,
    collideSpheres,
    emptySATCache,
    type LocalManifold,
    makeFeatureId,
    makeLocalManifold,
    type SATCache,
} from "./manifold";
import gold from "./manifold.gold.json";
import {
    collideCapsuleAndTriangle,
    collideHullAndTriangle,
    collideSphereAndTriangle,
} from "./triangle_manifold";

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

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const vecFromHex = (a: string[]): Vec3 => v(fromBits(a[0]), fromBits(a[1]), fromBits(a[2]));
const xfFromHex = (o: { p: string[]; q: string[] }): Transform => ({
    p: vecFromHex(o.p),
    q: {
        v: v(fromBits(o.q[0]), fromBits(o.q[1]), fromBits(o.q[2])),
        s: fromBits(o.q[3]),
    },
});

function sphereFromHex(o: { center: string[]; radius: string }): Sphere {
    return { center: vecFromHex(o.center), radius: fromBits(o.radius) };
}
function capsuleFromHex(o: { center1: string[]; center2: string[]; radius: string }): Capsule {
    return {
        center1: vecFromHex(o.center1),
        center2: vecFromHex(o.center2),
        radius: fromBits(o.radius),
    };
}
type BoxSpec = { kind: "box"; h: string[] };
type CylSpec = { kind: "cyl"; height: string; radius: string; yOffset: string; sides: number };

function hullFromHex(o: BoxSpec | CylSpec): HullData {
    if (o.kind === "box") {
        const h = vecFromHex(o.h);
        return makeBoxHull(h.x, h.y, h.z);
    }
    return createCylinder(fromBits(o.height), fromBits(o.radius), fromBits(o.yOffset), o.sides);
}

type GoldManifold = {
    normal: string[];
    pointCount: number;
    points: { point: string[]; separation: string; featureId: number }[];
};

function checkManifold(m: LocalManifold, want: GoldManifold, label: string) {
    expect(m.pointCount, `${label}.pointCount`).toBe(want.pointCount);
    // The normal is only meaningful when there are points; C leaves it stale otherwise.
    if (m.pointCount > 0) {
        bitEqual(m.normal.x, want.normal[0], `${label}.normal.x`);
        bitEqual(m.normal.y, want.normal[1], `${label}.normal.y`);
        bitEqual(m.normal.z, want.normal[2], `${label}.normal.z`);
    }
    for (let i = 0; i < want.pointCount; ++i) {
        const pt = m.points[i];
        const wp = want.points[i];
        bitEqual(pt.point.x, wp.point[0], `${label}.points[${i}].point.x`);
        bitEqual(pt.point.y, wp.point[1], `${label}.points[${i}].point.y`);
        bitEqual(pt.point.z, wp.point[2], `${label}.points[${i}].point.z`);
        bitEqual(pt.separation, wp.separation, `${label}.points[${i}].separation`);
        expect(makeFeatureId(pt.pair), `${label}.points[${i}].featureId`).toBe(wp.featureId);
    }
}

function checkSATCache(
    c: SATCache,
    want: { separation: string; type: number; indexA: number; indexB: number },
    label: string,
) {
    bitEqual(c.separation, want.separation, `${label}.separation`);
    expect(c.type, `${label}.type`).toBe(want.type);
    expect(c.indexA, `${label}.indexA`).toBe(want.indexA);
    expect(c.indexB, `${label}.indexB`).toBe(want.indexB);
}

type GoldTriManifold = GoldManifold & { feature: number };

function triFromHex(a: string[][]): [Vec3, Vec3, Vec3] {
    return [vecFromHex(a[0]), vecFromHex(a[1]), vecFromHex(a[2])];
}

function checkTriManifold(m: LocalManifold, want: GoldTriManifold, label: string) {
    checkManifold(m, want, label);
    // The triangle feature (used by mesh-contact reduction) is only meaningful with points.
    if (m.pointCount > 0) {
        expect(m.feature, `${label}.feature`).toBe(want.feature);
    }
}

check(
    "collideSpheres matches the C reference manifold bit for bit",
    {
        claim: "the sphere-sphere manifold drifts from the pinned Box3D C reference on point count, normal, contact point, separation or feature id",
        tier: "step",
    },
    () => {
        for (const scene of gold.spheres) {
            const a = sphereFromHex(scene.a);
            const b = sphereFromHex(scene.b);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(1);
            collideSpheres(m, 1, a, b, xf);
            checkManifold(m, scene.manifold, scene.name);
        }
    },
);

check(
    "collideCapsuleAndSphere matches the C reference manifold bit for bit",
    {
        claim: "the capsule-sphere manifold drifts from the pinned Box3D C reference on point count, normal, contact point, separation or feature id",
        tier: "step",
    },
    () => {
        for (const scene of gold.capsuleSphere) {
            const a = capsuleFromHex(scene.a);
            const b = sphereFromHex(scene.b);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(1);
            collideCapsuleAndSphere(m, 1, a, b, xf);
            checkManifold(m, scene.manifold, scene.name);
        }
    },
);

check(
    "collideHullAndSphere matches the C reference manifold across repeated warm calls",
    {
        claim: "the hull-sphere manifold drifts from the pinned Box3D C reference, or its warm simplex cache changes the answer on the second call for the same pose",
        tier: "step",
    },
    () => {
        for (const scene of gold.hullSphere) {
            const a = hullFromHex({ kind: "box", h: scene.a.h });
            const b = sphereFromHex(scene.b);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(1);
            const cache = emptyCache();
            for (let call = 0; call < scene.manifolds.length; ++call) {
                collideHullAndSphere(m, 1, a, b, xf, cache);
                checkManifold(m, scene.manifolds[call], `${scene.name}[${call}]`);
            }
        }
    },
);

check(
    "collideCapsules matches the C reference manifold bit for bit",
    {
        claim: "the capsule-capsule manifold drifts from the pinned Box3D C reference on point count, normal, contact point, separation or feature id",
        tier: "step",
    },
    () => {
        for (const scene of gold.capsules) {
            const a = capsuleFromHex(scene.a);
            const b = capsuleFromHex(scene.b);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(2);
            collideCapsules(m, 2, a, b, xf);
            checkManifold(m, scene.manifold, scene.name);
        }
    },
);

check(
    "collideHullAndCapsule matches the C reference manifold across repeated warm calls",
    {
        claim: "the hull-capsule manifold drifts from the pinned Box3D C reference, or its warm simplex cache changes the answer on the second call for the same pose",
        tier: "step",
    },
    () => {
        for (const scene of gold.hullCapsule) {
            const a = hullFromHex({ kind: "box", h: scene.a.h });
            const b = capsuleFromHex(scene.b);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(2);
            const cache = emptyCache();
            for (let call = 0; call < scene.manifolds.length; ++call) {
                collideHullAndCapsule(m, 2, a, b, xf, cache);
                checkManifold(m, scene.manifolds[call], `${scene.name}[${call}]`);
            }
        }
    },
);

check(
    "collideHulls matches the C reference manifold and SAT cache across repeated warm calls",
    {
        claim: "the hull-hull SAT and clipping path drifts from the pinned Box3D C reference on the clipped manifold or on the separating-feature cache it warms for the next call",
        tier: "step",
    },
    () => {
        for (const scene of gold.hulls) {
            const a = hullFromHex(scene.a as BoxSpec | CylSpec);
            const b = hullFromHex(scene.b as BoxSpec | CylSpec);
            const xf = xfFromHex(scene.xf);
            const m = makeLocalManifold(8);
            const cache = emptySATCache();
            for (let call = 0; call < scene.manifolds.length; ++call) {
                collideHulls(m, 8, a, b, xf, cache);
                checkManifold(m, scene.manifolds[call], `${scene.name}[${call}]`);
                checkSATCache(cache, scene.caches[call], `${scene.name}.cache[${call}]`);
            }
        }
    },
);

check(
    "collideSphereAndTriangle matches the C reference manifold bit for bit",
    {
        claim: "the sphere-triangle manifold drifts from the pinned Box3D C reference on its manifold or on the triangle feature mesh-contact reduction reads",
        tier: "step",
    },
    () => {
        for (const scene of gold.sphereTriangle) {
            const a = sphereFromHex(scene.a);
            const [v1, v2, v3] = triFromHex(scene.tri);
            const m = makeLocalManifold(1);
            collideSphereAndTriangle(m, 1, a, v1, v2, v3);
            checkTriManifold(m, scene.manifold, scene.name);
        }
    },
);

check(
    "collideCapsuleAndTriangle matches the C reference manifold across repeated warm calls",
    {
        claim: "the capsule-triangle manifold drifts from the pinned Box3D C reference on its manifold, its triangle feature, or under its own warm simplex cache",
        tier: "step",
    },
    () => {
        for (const scene of gold.capsuleTriangle) {
            const a = capsuleFromHex(scene.a);
            const [v1, v2, v3] = triFromHex(scene.tri);
            const m = makeLocalManifold(2);
            const cache = emptyCache();
            for (let call = 0; call < scene.manifolds.length; ++call) {
                collideCapsuleAndTriangle(m, 2, a, v1, v2, v3, cache);
                checkTriManifold(m, scene.manifolds[call], `${scene.name}[${call}]`);
            }
        }
    },
);

check(
    "collideHullAndTriangle matches the C reference manifold and SAT cache across repeated warm calls",
    {
        claim: "the hull-triangle manifold drifts from the pinned Box3D C reference on its manifold, its triangle feature, or on the separating-feature cache it warms",
        tier: "step",
    },
    () => {
        for (const scene of gold.hullTriangle) {
            const a = hullFromHex({ kind: "box", h: scene.a.h });
            const [v1, v2, v3] = triFromHex(scene.tri);
            const m = makeLocalManifold(8);
            const cache = emptySATCache();
            for (let call = 0; call < scene.manifolds.length; ++call) {
                collideHullAndTriangle(m, 8, a, v1, v2, v3, cache);
                checkTriManifold(m, scene.manifolds[call], `${scene.name}[${call}]`);
                checkSATCache(cache, scene.caches[call], `${scene.name}.cache[${call}]`);
            }
        }
    },
);

check(
    "two unit cubes overlapping by 0.1 give a four-point face manifold",
    {
        claim: "hull-hull face clipping loses a corner or mis-scales penetration, so a 0.1 box-box overlap stops producing four points each separated by about -0.1",
        tier: "step",
    },
    () => {
        // Analytic invariant from Box3D's LargeWorldManifoldTest (float path).
        const boxA = makeBoxHull(0.5, 0.5, 0.5);
        const boxB = makeBoxHull(0.5, 0.5, 0.5);
        const xf: Transform = { p: v(0.9, 0, 0), q: { v: v(0, 0, 0), s: 1 } };
        const m = makeLocalManifold(8);
        const cache = emptySATCache();
        collideHulls(m, 8, boxA, boxB, xf, cache);
        expect(m.pointCount).toBe(4);
        for (let i = 0; i < m.pointCount; ++i) {
            expect(Math.abs(m.points[i].separation + 0.1), `point ${i} separation`).toBeLessThan(
                0.01,
            );
        }
    },
);

check(
    "pointToSegmentDistance projects inside and clamps to the endpoints",
    {
        claim: "pointToSegmentDistance fails to clamp a query beyond either end of the segment, returning an extrapolated point instead of the endpoint itself",
        tier: "step",
    },
    () => {
        const a = v(0, 0, 0);
        const b = v(2, 0, 0);
        expect(pointToSegmentDistance(a, b, v(1, 5, 0))).toEqual(v(1, 0, 0));
        // Beyond the a-side returns a; beyond the b-side returns b (reference equality).
        expect(pointToSegmentDistance(a, b, v(-3, 1, 0))).toBe(a);
        expect(pointToSegmentDistance(a, b, v(9, 1, 0))).toBe(b);
    },
);

check(
    "lineDistance finds the closest points on two skew lines",
    {
        claim: "lineDistance solves the wrong pair on two skew infinite lines, so the capsule and edge-edge paths built on it pick the wrong closest points",
        tier: "step",
    },
    () => {
        // Line 1 along x through origin; line 2 along y through (0,0,1). Closest points are the
        // origin and (0,0,1); the connecting segment is within both.
        const r = lineDistance(v(0, 0, 0), v(1, 0, 0), v(0, 0, 1), v(0, 1, 0));
        expect(r.point1).toEqual(v(0, 0, 0));
        expect(r.point2).toEqual(v(0, 0, 1));
        expect(isWithinSegments(r)).toBe(true);
    },
);

check(
    "isWithinSegments rejects an out-of-range fraction",
    {
        claim: "isWithinSegments accepts a fraction outside [0,1], letting an infinite-line solution be used as if it lay on the finite segments",
        tier: "step",
    },
    () => {
        expect(
            isWithinSegments({
                point1: v(0, 0, 0),
                fraction1: 1.5,
                point2: v(0, 0, 0),
                fraction2: 0,
            }),
        ).toBe(false);
    },
);
