import { expect, test } from "bun:test";
import { segmentDistance, type Vec3 } from "../common/math";
import { emptyCache, type ShapeProxy, shapeDistance } from "./distance";
import gold from "./distance.gold.json";

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

// --- bit-exact gold gates -------------------------------------------------------------------

test("the closest-points-between-two-segments primitive drifts from the pinned Box3D C reference on either witness point or either clamped fraction", () => {
    for (const g of gold.segment) {
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
    }
});

// --- ported upstream test_distance.c subtests (analytic, oracle-independent) -----------------

const EPS = fromBits("34000000"); // FLT_EPSILON = 2^-23

test("segmentDistance mis-parameterizes or fails to clamp a perpendicular segment pair, so the closest pair lands off the segments instead of at the midpoint and the endpoint", () => {
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

test("shapeDistance returns a gap other than the analytic 1 between a unit quad and a segment standing one unit away, so separated pairs report the wrong distance", () => {
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
