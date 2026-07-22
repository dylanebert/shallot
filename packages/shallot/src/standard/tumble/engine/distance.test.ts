import { describe, expect, test } from "bun:test";
import {
    type CastOutput,
    type DistanceOutput,
    emptyCache,
    type ShapeProxy,
    type SimplexCache,
    type Sweep,
    shapeCast,
    shapeDistance,
    type TOIOutput,
    timeOfImpact,
} from "./distance";
import gold from "./distance.gold.json";
import { type Quat, segmentDistance, type Transform, type Vec3 } from "./math";

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

const v = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const vecFromHex = (a: string[]): Vec3 => v(fromBits(a[0]), fromBits(a[1]), fromBits(a[2]));
const quatFromHex = (a: string[]): Quat => ({
    v: v(fromBits(a[0]), fromBits(a[1]), fromBits(a[2])),
    s: fromBits(a[3]),
});
const xfFromHex = (o: { p: string[]; q: string[] }): Transform => ({
    p: vecFromHex(o.p),
    q: quatFromHex(o.q),
});
const sweepFromHex = (o: {
    localCenter: string[];
    c1: string[];
    c2: string[];
    q1: string[];
    q2: string[];
}): Sweep => ({
    localCenter: vecFromHex(o.localCenter),
    c1: vecFromHex(o.c1),
    c2: vecFromHex(o.c2),
    q1: quatFromHex(o.q1),
    q2: quatFromHex(o.q2),
});

// Named proxy point clouds, integer coords mirroring fixtures/distance_gold.c exactly.
const POINTS: Record<string, Vec3[]> = {
    box8: [
        v(-1, -1, -1),
        v(1, -1, -1),
        v(1, 1, -1),
        v(-1, 1, -1),
        v(-1, -1, 1),
        v(1, -1, 1),
        v(1, 1, 1),
        v(-1, 1, 1),
    ],
    quad4: [v(-1, -1, 0), v(1, -1, 0), v(1, 1, 0), v(-1, 1, 0)],
    segB: [v(2, -1, 0), v(2, 1, 0)],
    seg0: [v(0, -1, 0), v(0, 1, 0)],
    pt1: [v(0, 0, 0)],
};
const makeProxy = (name: string, radiusHex: string): ShapeProxy => {
    const points = POINTS[name];
    return { points, count: points.length, radius: fromBits(radiusHex) };
};

// --- bit-exact gold gates -------------------------------------------------------------------

describe("shapeDistance bit-exact vs C reference", () => {
    for (const g of gold.distance) {
        test(g.name, () => {
            const proxyA = makeProxy(g.proxyA, g.radiusA);
            const proxyB = makeProxy(g.proxyB, g.radiusB);
            const cache: SimplexCache = emptyCache();
            const out: DistanceOutput = shapeDistance(
                { proxyA, proxyB, transform: xfFromHex(g.transform), useRadii: g.useRadii },
                cache,
            );
            vecEqual(out.pointA, g.out.pointA, `${g.name} pointA`);
            vecEqual(out.pointB, g.out.pointB, `${g.name} pointB`);
            vecEqual(out.normal, g.out.normal, `${g.name} normal`);
            bitEqual(out.distance, g.out.distance, `${g.name} distance`);
            expect(out.iterations).toBe(g.out.iterations);
            // Cache is written only on the non-overlap exit; on overlap C leaves it untouched (count 0).
            bitEqual(cache.metric, g.out.cache.metric, `${g.name} cache.metric`);
            expect(cache.count).toBe(g.out.cache.count);
            for (let i = 0; i < cache.count; ++i) {
                expect(cache.indexA[i]).toBe(g.out.cache.indexA[i]);
                expect(cache.indexB[i]).toBe(g.out.cache.indexB[i]);
            }
        });
    }
});

describe("shapeCast bit-exact vs C reference", () => {
    for (const g of gold.cast) {
        test(g.name, () => {
            const proxyA = makeProxy(g.proxyA, g.radiusA);
            const proxyB = makeProxy(g.proxyB, g.radiusB);
            const out: CastOutput = shapeCast({
                proxyA,
                proxyB,
                transform: xfFromHex(g.transform),
                translationB: vecFromHex(g.translationB),
                maxFraction: fromBits(g.maxFraction),
                canEncroach: g.canEncroach,
            });
            expect(out.hit).toBe(g.out.hit);
            bitEqual(out.fraction, g.out.fraction, `${g.name} fraction`);
            vecEqual(out.point, g.out.point, `${g.name} point`);
            vecEqual(out.normal, g.out.normal, `${g.name} normal`);
            expect(out.iterations).toBe(g.out.iterations);
        });
    }
});

describe("timeOfImpact bit-exact vs C reference", () => {
    for (const g of gold.toi) {
        test(g.name, () => {
            const proxyA = makeProxy(g.proxyA, g.radiusA);
            const proxyB = makeProxy(g.proxyB, g.radiusB);
            const out: TOIOutput = timeOfImpact({
                proxyA,
                proxyB,
                sweepA: sweepFromHex(g.sweepA),
                sweepB: sweepFromHex(g.sweepB),
                maxFraction: fromBits(g.maxFraction),
            });
            expect(out.state as number).toBe(g.out.state);
            bitEqual(out.fraction, g.out.fraction, `${g.name} fraction`);
            bitEqual(out.distance, g.out.distance, `${g.name} distance`);
            vecEqual(out.point, g.out.point, `${g.name} point`);
            vecEqual(out.normal, g.out.normal, `${g.name} normal`);
            expect(out.distanceIterations).toBe(g.out.distanceIterations);
            expect(out.pushBackIterations).toBe(g.out.pushBackIterations);
            expect(out.rootIterations).toBe(g.out.rootIterations);
        });
    }
});

describe("segmentDistance bit-exact vs C reference", () => {
    for (const g of gold.segment) {
        test(g.name, () => {
            const r = segmentDistance(
                vecFromHex(g.p1),
                vecFromHex(g.q1),
                vecFromHex(g.p2),
                vecFromHex(g.q2),
            );
            vecEqual(r.point1, g.out.point1, `${g.name} point1`);
            bitEqual(r.fraction1, g.out.fraction1, `${g.name} fraction1`);
            vecEqual(r.point2, g.out.point2, `${g.name} point2`);
            bitEqual(r.fraction2, g.out.fraction2, `${g.name} fraction2`);
        });
    }
});

// --- ported upstream test_distance.c subtests (analytic, oracle-independent) -----------------

const EPS = fromBits("34000000"); // FLT_EPSILON = 2^-23

describe("test_distance.c", () => {
    test("SegmentDistanceTest", () => {
        const r = segmentDistance(v(-1, -1, 0), v(-1, 1, 0), v(2, 0, 0), v(1, 0, 0));
        expect(Math.abs(r.fraction1 - 0.5)).toBeLessThan(EPS);
        expect(Math.abs(r.fraction2 - 1)).toBeLessThan(EPS);
        expect(Math.abs(r.point1.x + 1)).toBeLessThan(EPS);
        expect(Math.abs(r.point1.y)).toBeLessThan(EPS);
        expect(Math.abs(r.point1.z)).toBeLessThan(EPS);
        expect(Math.abs(r.point2.x - 1)).toBeLessThan(EPS);
        expect(Math.abs(r.point2.y)).toBeLessThan(EPS);
        expect(Math.abs(r.point2.z)).toBeLessThan(EPS);
    });

    test("ShapeDistanceTest", () => {
        const proxyA: ShapeProxy = {
            points: [v(-1, -1, 0), v(1, -1, 0), v(1, 1, 0), v(-1, 1, 0)],
            count: 4,
            radius: 0,
        };
        const proxyB: ShapeProxy = { points: [v(2, -1, 0), v(2, 1, 0)], count: 2, radius: 0 };
        const out = shapeDistance(
            {
                proxyA,
                proxyB,
                transform: { p: v(0, 0, 0), q: { v: v(0, 0, 0), s: 1 } },
                useRadii: false,
            },
            emptyCache(),
        );
        expect(Math.abs(out.distance - 1)).toBeLessThan(EPS);
    });

    test("ShapeCastTest", () => {
        const proxyA: ShapeProxy = {
            points: [v(-1, -1, 0), v(1, -1, 0), v(1, 1, 0), v(-1, 1, 0)],
            count: 4,
            radius: 0,
        };
        const proxyB: ShapeProxy = { points: [v(2, -1, 0), v(2, 1, 0)], count: 2, radius: 0 };
        const out = shapeCast({
            proxyA,
            proxyB,
            transform: { p: v(0, 0, 0), q: { v: v(0, 0, 0), s: 1 } },
            translationB: v(-2, 0, 0),
            maxFraction: 1,
            canEncroach: false,
        });
        expect(out.hit).toBe(true);
        expect(Math.abs(out.fraction - 0.5)).toBeLessThan(0.005);
    });

    test("TimeOfImpactTest", () => {
        const proxyA: ShapeProxy = {
            points: [v(-1, -1, 0), v(1, -1, 0), v(1, 1, 0), v(-1, 1, 0)],
            count: 4,
            radius: 0,
        };
        const proxyB: ShapeProxy = { points: [v(2, -1, 0), v(2, 1, 0)], count: 2, radius: 0 };
        const id: Quat = { v: v(0, 0, 0), s: 1 };
        const out = timeOfImpact({
            proxyA,
            proxyB,
            sweepA: { localCenter: v(0, 0, 0), c1: v(0, 0, 0), c2: v(0, 0, 0), q1: id, q2: id },
            sweepB: { localCenter: v(0, 0, 0), c1: v(0, 0, 0), c2: v(-2, 0, 0), q1: id, q2: id },
            maxFraction: 1,
        });
        expect(out.state).toBe(3); // TOIState.Hit
        expect(Math.abs(out.fraction - 0.5)).toBeLessThan(0.005);
    });
});
