import { describe, expect, test } from "bun:test";
import type { Mat2, Mat3, Quat, Transform, Vec3 } from "./math";
import * as m from "./math";
import gold from "./math.gold.json";

// Reconstruct an exact f32 from the raw hex bits the C generator emitted — no decimal round-trip.
const dv = new DataView(new ArrayBuffer(4));
function fromBits(hex: string): number {
    dv.setUint32(0, Number.parseInt(hex, 16));
    return dv.getFloat32(0);
}

// Bit-exact: both operands are f32-valued, so Object.is catches any divergence including ±0.
function bitEqual(got: number, want: string, label: string) {
    const w = fromBits(want);
    if (!Object.is(got, w)) {
        throw new Error(`${label}: got 0x${bits(got)} (${got}), want ${want} (${w})`);
    }
}
function bits(f: number): string {
    dv.setFloat32(0, f);
    return dv.getUint32(0).toString(16).padStart(8, "0");
}

// --- reconstruct / flatten in the same order the C generator uses -----------------------------

const vec3From = (a: number[], o = 0): Vec3 => ({ x: a[o], y: a[o + 1], z: a[o + 2] });
const quatFrom = (a: number[], o = 0): Quat => ({
    v: { x: a[o], y: a[o + 1], z: a[o + 2] },
    s: a[o + 3],
});
const mat3From = (a: number[], o = 0): Mat3 => ({
    cx: { x: a[o], y: a[o + 1], z: a[o + 2] },
    cy: { x: a[o + 3], y: a[o + 4], z: a[o + 5] },
    cz: { x: a[o + 6], y: a[o + 7], z: a[o + 8] },
});
const mat2From = (a: number[], o = 0): Mat2 => ({
    cx: { x: a[o], y: a[o + 1] },
    cy: { x: a[o + 2], y: a[o + 3] },
});
const xfFrom = (a: number[], o = 0): Transform => ({ p: vec3From(a, o), q: quatFrom(a, o + 3) });

const flatV = (v: Vec3): number[] => [v.x, v.y, v.z];
const flatV2 = (v: { x: number; y: number }): number[] => [v.x, v.y];
const flatQ = (q: Quat): number[] => [q.v.x, q.v.y, q.v.z, q.s];
const flatM = (x: Mat3): number[] => [
    x.cx.x,
    x.cx.y,
    x.cx.z,
    x.cy.x,
    x.cy.y,
    x.cy.z,
    x.cz.x,
    x.cz.y,
    x.cz.z,
];
const flatM2 = (x: Mat2): number[] => [x.cx.x, x.cx.y, x.cy.x, x.cy.y];
const flatT = (t: Transform): number[] => [t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s];

// Each case maps its flat input array to the port's flat output array.
const dispatch: Record<string, (a: number[]) => number[]> = {
    normalizeVec3: (a) => flatV(m.vec3.normalize(vec3From(a))),
    lengthVec3: (a) => [m.vec3.length(vec3From(a))],
    arbitraryPerp: (a) => flatV(m.arbitraryPerp(vec3From(a))),
    scalarTripleProduct: (a) => [
        m.scalarTripleProduct(vec3From(a, 0), vec3From(a, 3), vec3From(a, 6)),
    ],
    normalizeQuat: (a) => flatQ(m.quat.normalize(quatFrom(a))),
    makeQuatFromAxisAngle: (a) => flatQ(m.quat.fromAxisAngle(vec3From(a), a[3])),
    mulQuat: (a) => flatQ(m.quat.mul(quatFrom(a, 0), quatFrom(a, 4))),
    invMulQuat: (a) => flatQ(m.quat.invMul(quatFrom(a, 0), quatFrom(a, 4))),
    rotateVector: (a) => flatV(m.quat.rotate(quatFrom(a), vec3From(a, 4))),
    invRotateVector: (a) => flatV(m.quat.invRotate(quatFrom(a), vec3From(a, 4))),
    computeQuatBetween: (a) =>
        flatQ(m.computeQuatBetweenUnitVectors(vec3From(a, 0), vec3From(a, 3))),
    computeQuatBetweenAntiparallel: (a) =>
        flatQ(m.computeQuatBetweenUnitVectors(vec3From(a, 0), vec3From(a, 3))),
    makeQuatFromMatrix: (a) => flatQ(m.makeQuatFromMatrix(mat3From(a))),
    makeMatrixFromQuat: (a) => flatM(m.mat3.fromQuat(quatFrom(a))),
    invertMatrix: (a) => flatM(m.mat3.invert(mat3From(a))),
    solve3: (a) => flatV(m.mat3.solve(mat3From(a), vec3From(a, 9))),
    mulMV: (a) => flatV(m.mat3.mulV(mat3From(a), vec3From(a, 9))),
    mulMM: (a) => flatM(m.mat3.mul(mat3From(a, 0), mat3From(a, 9))),
    invert2: (a) => flatM2(m.mat2.invert(mat2From(a))),
    solve2: (a) => flatV2(m.mat2.solve(mat2From(a), { x: a[4], y: a[5] })),
    nlerp: (a) => flatQ(m.quat.nlerp(quatFrom(a, 0), quatFrom(a, 4), a[8])),
    getTwistAngle: (a) => [m.quat.getTwistAngle(quatFrom(a))],
    getSwingAngle: (a) => [m.quat.getSwingAngle(quatFrom(a))],
    transformPoint: (a) => flatV(m.xf.point(xfFrom(a), vec3From(a, 7))),
    invTransformPoint: (a) => flatV(m.xf.invPoint(xfFrom(a), vec3From(a, 7))),
    mulTransforms: (a) => flatT(m.xf.mul(xfFrom(a, 0), xfFrom(a, 7))),
    // Gold `in` is [t2, t1]; the reference computed b3InvMulTransforms(t1, t2), so swap.
    invMulTransforms: (a) => flatT(m.xf.invMul(xfFrom(a, 7), xfFrom(a, 0))),
};

type Gold = {
    atan2: [string, string, string][];
    cosSin: [string, string, string][];
    cases: { fn: string; in: string[]; out: string[] }[];
};
const g = gold as Gold;

// --- bit-exact parity vs the C reference (BOX3D_DISABLE_SIMD + FORCE_OVERFLOW) ---------------

describe("bit-exact vs C reference", () => {
    test("atan2 sweep", () => {
        for (const [yh, xh, rh] of g.atan2) {
            bitEqual(m.atan2(fromBits(yh), fromBits(xh)), rh, `atan2(${yh},${xh})`);
        }
    });

    test("computeCosSin sweep", () => {
        for (const [ah, ch, sh] of g.cosSin) {
            const cs = m.computeCosSin(fromBits(ah));
            bitEqual(cs.cosine, ch, `cos(${ah})`);
            bitEqual(cs.sine, sh, `sin(${ah})`);
        }
    });

    test("vec/quat/matrix/transform cases", () => {
        expect(g.cases.length).toBeGreaterThan(20);
        for (const c of g.cases) {
            const fn = dispatch[c.fn];
            if (!fn) throw new Error(`no dispatch for gold case ${c.fn}`);
            const got = fn(c.in.map(fromBits));
            expect(got.length).toBe(c.out.length);
            for (let i = 0; i < got.length; i++) {
                bitEqual(got[i], c.out[i], `${c.fn}[${i}]`);
            }
        }
    });
});

// --- oracle-independent invariants (ported from test_math.c) ---------------------------------
// These hold by algebra regardless of the C build, so they catch conceptual port errors the
// gold can't. Tolerances are Box3D's own (FLT_EPSILON multiples; ATAN_TOL = 0.0023°).

const EPS = m.FLT_EPSILON;

describe("algebraic invariants", () => {
    test("atan2 matches libm within 0.0023°", () => {
        const AtanTol = 4e-5;
        for (let y = -1; y <= 1; y += 0.05) {
            for (let x = -1; x <= 1; x += 0.05) {
                if (x === 0 && y === 0) continue;
                expect(Math.abs(m.atan2(y, x) - Math.atan2(y, x))).toBeLessThan(AtanTol);
            }
        }
    });

    test("computeCosSin matches libm within 0.002", () => {
        for (let t = -10; t < 10; t += 0.05) {
            const a = Math.PI * t;
            const cs = m.computeCosSin(a);
            expect(Math.abs(cs.cosine - Math.cos(a))).toBeLessThan(0.002);
            expect(Math.abs(cs.sine - Math.sin(a))).toBeLessThan(0.002);
        }
    });

    test("normalize yields unit length", () => {
        const u = m.vec3.normalize({ x: 0.2, y: -0.5, z: 3.0 });
        expect(Math.abs(m.vec3.length(u) - 1)).toBeLessThan(4 * EPS);
    });

    test("transform point round-trips through its inverse", () => {
        const axis = m.vec3.normalize({ x: 0.3, y: -0.7, z: 0.5 });
        const t: Transform = { p: { x: 3, y: -5, z: 2 }, q: m.quat.fromAxisAngle(axis, 0.4) };
        const v = { x: 0.5, y: -0.25, z: 1.5 };
        const back = m.xf.invPoint(t, m.xf.point(t, v));
        expect(Math.abs(back.x - v.x)).toBeLessThan(1e-5);
        expect(Math.abs(back.y - v.y)).toBeLessThan(1e-5);
        expect(Math.abs(back.z - v.z)).toBeLessThan(1e-5);
    });

    test("matrix times its inverse is identity", () => {
        const mat: Mat3 = {
            cx: { x: 3, y: 1, z: -1 },
            cy: { x: -1, y: 3, z: 1 },
            cz: { x: 1, y: -1, z: 3 },
        };
        const id = m.mat3.mul(mat, m.mat3.invert(mat));
        const idm = m.mat3.identity();
        for (const col of ["cx", "cy", "cz"] as const) {
            for (const axis of ["x", "y", "z"] as const) {
                expect(Math.abs(id[col][axis] - idm[col][axis])).toBeLessThan(2 * EPS);
            }
        }
    });

    test("solve3 agrees with invert then multiply", () => {
        const mat: Mat3 = {
            cx: { x: 3, y: 1, z: -1 },
            cy: { x: -1, y: 3, z: 1 },
            cz: { x: 1, y: -1, z: 3 },
        };
        const v = { x: 1, y: -2, z: 3 };
        const a = m.mat3.mulV(m.mat3.invert(mat), v);
        const b = m.mat3.solve(mat, v);
        expect(Math.abs(a.x - b.x)).toBeLessThan(EPS);
        expect(Math.abs(a.y - b.y)).toBeLessThan(EPS);
        expect(Math.abs(a.z - b.z)).toBeLessThan(EPS);
    });

    test("quat compose then decompose recovers the operand", () => {
        const q1 = m.quat.fromAxisAngle(m.vec3.axisZ(), -0.5 * Math.PI);
        const q3 = m.quat.normalize({ v: { x: 1, y: -2, z: 3 }, s: 4 });
        const q5 = m.quat.mul(q3, m.quat.invMul(q3, q1));
        expect(Math.abs(q1.v.x - q5.v.x)).toBeLessThan(EPS);
        expect(Math.abs(q1.v.y - q5.v.y)).toBeLessThan(EPS);
        expect(Math.abs(q1.v.z - q5.v.z)).toBeLessThan(EPS);
        expect(Math.abs(q1.s - q5.s)).toBeLessThan(EPS);
    });

    test("computeQuatBetween rotates v1 onto v2", () => {
        const v1 = m.vec3.normalize({ x: 0.2, y: -0.5, z: 3.0 });
        const u = m.vec3.normalize({ x: -0.3, y: 0.8, z: 0.1 });
        const r = m.computeQuatBetweenUnitVectors(v1, u);
        expect(m.quat.isValid(r)).toBe(true);
        const w = m.quat.rotate(r, v1);
        expect(Math.abs(w.x - u.x)).toBeLessThan(0.001);
        expect(Math.abs(w.y - u.y)).toBeLessThan(0.001);
        expect(Math.abs(w.z - u.z)).toBeLessThan(0.001);
    });

    test("nlerp twist angle tracks alpha across a 90° turn", () => {
        const q1 = m.quat.identity();
        const q2 = m.quat.fromAxisAngle(m.vec3.axisZ(), 0.5 * Math.PI);
        for (let i = 0; i <= 20; i++) {
            const alpha = i / 20;
            const angle = m.quat.getTwistAngle(m.quat.nlerp(q1, q2, alpha));
            expect(Math.abs(alpha * 0.5 * Math.PI - angle)).toBeLessThan(m.DEG_TO_RAD);
        }
    });

    test("arbitraryPerp is orthogonal to its input", () => {
        const n = { x: 0.50405544, y: 0.621548057, z: 0.599671543 };
        expect(Math.abs(m.vec3.dot(n, m.arbitraryPerp(n)))).toBeLessThan(2 * EPS);
    });

    test("scalar min/max mirror the C scalar branches, not Math.min/max", () => {
        // b3MinFloat(NaN, b) === b (Math.min would be NaN); b3MinFloat(-0, +0) === +0.
        expect(m.minf(Number.NaN, 5)).toBe(5);
        expect(Object.is(m.minf(-0, 0), 0)).toBe(true);
        expect(m.maxf(Number.NaN, 5)).toBe(5);
    });
});
