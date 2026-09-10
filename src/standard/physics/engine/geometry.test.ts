import { describe, expect, test } from "bun:test";
import {
    type Capsule,
    computeCapsuleAABB,
    computeCapsuleMass,
    computeSphereAABB,
    computeSphereMass,
    type MassData,
    roundCapsule,
    roundSphere,
    type Sphere,
} from "./geometry";
import gold from "./geometry.gold.json";
import { f32, type Transform, type Vec3, xf } from "./math";

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

type MassGold = { name: string; mass: string; center: string[]; inertia: string[] };

function assertMass(m: MassData, g: MassGold) {
    bitEqual(m.mass, g.mass, `${g.name} mass`);
    bitEqual(m.center.x, g.center[0], `${g.name} center.x`);
    bitEqual(m.center.y, g.center[1], `${g.name} center.y`);
    bitEqual(m.center.z, g.center[2], `${g.name} center.z`);
    const inertia = [
        m.inertia.cx.x,
        m.inertia.cx.y,
        m.inertia.cx.z,
        m.inertia.cy.x,
        m.inertia.cy.y,
        m.inertia.cy.z,
        m.inertia.cz.x,
        m.inertia.cz.y,
        m.inertia.cz.z,
    ];
    for (let i = 0; i < 9; ++i) bitEqual(inertia[i], g.inertia[i], `${g.name} inertia[${i}]`);
}

const sphereGold = (name: string) => gold.spheres.find((s) => s.name === name) as MassGold;
const capsuleGold = (name: string) => gold.capsules.find((c) => c.name === name) as MassGold;

describe("sphere/capsule mass bit-exact vs C reference", () => {
    // f32-round non-exact literals (0.35, 0.3) to match the C float inputs bit-for-bit.
    test("unit sphere", () => {
        const s: Sphere = { center: v(0, 0, 0), radius: 1 };
        assertMass(computeSphereMass(s, 1), sphereGold("unit"));
    });
    test("offset sphere", () => {
        const s: Sphere = { center: v(0.5, -1, 2), radius: f32(0.35) };
        assertMass(computeSphereMass(s, 2.5), sphereGold("offset"));
    });
    test("vertical capsule", () => {
        const c: Capsule = { center1: v(0, -1, 0), center2: v(0, 1, 0), radius: 0.5 };
        assertMass(computeCapsuleMass(c, 1), capsuleGold("vertical"));
    });
    test("skew capsule", () => {
        const c: Capsule = {
            center1: v(-1, 0.5, 0.25),
            center2: v(1.5, -0.5, 0.75),
            radius: f32(0.3),
        };
        assertMass(computeCapsuleMass(c, 3), capsuleGold("skew"));
    });
    // Regression: an x-axis ragdoll bone (r = 0.12, water density). The sphere-inertia 0.4f literal
    // must be f32-rounded — f64 0.4 vs C 0.4f round differently for this mass, so the buggy version
    // was ~1 ULP off on the axis inertia (the divergence that broke the rain fixture).
    test("ragdoll bone capsule (0.4f literal)", () => {
        const c: Capsule = { center1: v(0.06, 0, 0), center2: v(-0.06, 0, 0), radius: f32(0.12) };
        assertMass(computeCapsuleMass(c, 1000), capsuleGold("bone"));
    });
});

// AABBs compose xf.point (bit-exact) with min/max/sub/add; concrete extremes pin the wrapper
// without reimplementing the transform.
describe("sphere/capsule AABB", () => {
    const id = xf.identity();

    test("sphere AABB is center +/- radius under identity", () => {
        const aabb = computeSphereAABB({ center: v(0.5, -1, 2), radius: 0.25 }, id);
        expect(aabb.lowerBound).toEqual(v(0.25, -1.25, 1.75));
        expect(aabb.upperBound).toEqual(v(0.75, -0.75, 2.25));
    });

    test("sphere AABB translates with the transform", () => {
        const t: Transform = { p: v(1, 2, 3), q: id.q };
        const aabb = computeSphereAABB({ center: v(0, 0, 0), radius: 1 }, t);
        expect(aabb.lowerBound).toEqual(v(0, 1, 2));
        expect(aabb.upperBound).toEqual(v(2, 3, 4));
    });

    test("capsule AABB encloses both hemispheres under identity", () => {
        const aabb = computeCapsuleAABB(
            { center1: v(0, -1, 0), center2: v(0, 1, 0), radius: 0.5 },
            id,
        );
        expect(aabb.lowerBound).toEqual(v(-0.5, -1.5, -0.5));
        expect(aabb.upperBound).toEqual(v(0.5, 1.5, 0.5));
    });
});

describe("f32 geometry rounding at the storage boundary", () => {
    // The C holds geometry as f32 struct fields; callers pass f64 JS numbers. A field that is not
    // f32-exact (e.g. 0.3) must be rounded on storage or the solver arithmetic diverges from the C.
    test("roundSphere rounds every field to f32", () => {
        const s = roundSphere({ center: { x: 0.1, y: 0.2, z: 0.3 }, radius: 0.3 });
        expect(s.radius).toBe(f32(0.3));
        expect(s.radius).not.toBe(0.3);
        expect(s.center).toEqual({ x: f32(0.1), y: f32(0.2), z: f32(0.3) });
    });

    test("roundCapsule rounds every field to f32", () => {
        const c = roundCapsule({
            center1: { x: -0.3, y: 0, z: 0.7 },
            center2: { x: 0.3, y: 0, z: 0.7 },
            radius: 0.3,
        });
        expect(c.radius).toBe(f32(0.3));
        expect(c.radius).not.toBe(0.3);
        expect(c.center1.x).toBe(f32(-0.3));
        expect(c.center1.z).toBe(f32(0.7));
    });
});
