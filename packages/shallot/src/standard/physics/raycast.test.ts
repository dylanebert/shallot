import { describe, expect, test } from "bun:test";
import { ShapeKind } from "./index";
import {
    generateRay,
    qRotate,
    type RayBody,
    rayCapsule,
    raycast,
    rayOBB,
    raySphere,
    screenToRay,
} from "./raycast";

// Closed-form gold for the §6.5 CPU raycast — every expected t/normal is hand-derived from the geometry,
// not read off the implementation. f64, so the analytic solves are exact to ~1e-9.

describe("qRotate", () => {
    test("90° yaw rotates forward (−z) to −x, and the conjugate inverts it", () => {
        // q = (0, sin45°, 0, cos45°): +90° about Y. Active rotation: −z → −x.
        const s = Math.SQRT1_2;
        const [x, y, z] = qRotate(0, s, 0, s, 0, 0, -1);
        expect(x).toBeCloseTo(-1, 9);
        expect(y).toBeCloseTo(0, 9);
        expect(z).toBeCloseTo(0, 9);
        const [bx, by, bz] = qRotate(-0, -s, -0, s, x, y, z);
        expect(bx).toBeCloseTo(0, 9);
        expect(by).toBeCloseTo(0, 9);
        expect(bz).toBeCloseTo(-1, 9);
    });
});

describe("raySphere", () => {
    test("head-on hit returns the near root + outward normal", () => {
        // unit sphere at origin, ray from z=5 toward −z → enters the +z face at z=1, distance 4
        const h = raySphere(0, 0, 5, 0, 0, -1, 0, 0, 0, 1)!;
        expect(h).not.toBeNull();
        expect(h.t).toBeCloseTo(4, 9);
        expect([h.nx, h.ny, h.nz]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(1, 9),
        ]);
    });

    test("miss returns null", () => {
        expect(raySphere(2, 0, 5, 0, 0, -1, 0, 0, 0, 1)).toBeNull();
    });
});

describe("rayOBB", () => {
    test("axis-aligned box, head-on", () => {
        const h = rayOBB(0, 0, 5, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(4, 9);
        expect(h.nz).toBeCloseTo(1, 9);
    });

    test("hit on the +x face", () => {
        const h = rayOBB(5, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(4, 9);
        expect(h.nx).toBeCloseTo(1, 9);
    });

    test("90° Y-rotated box swaps its x/z extents", () => {
        // half (2,1,1) turned 90° about Y → the 2-extent now spans world z, so a −z ray hits z=2 at t=3,
        // and the local −x entry face rotates back to a world +z normal
        const q: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const h = rayOBB(0, 0, 5, 0, 0, -1, 0, 0, 0, 2, 1, 1, q[0], q[1], q[2], q[3])!;
        expect(h.t).toBeCloseTo(3, 6);
        expect(h.nz).toBeCloseTo(1, 6);
    });

    test("miss returns null", () => {
        expect(rayOBB(3, 0, 5, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)).toBeNull();
    });

    test("origin inside returns the exit distance paired with the EXIT face normal", () => {
        // unit box at origin, ray from the centre toward −z exits the −z face at t=1; the normal is that
        // exit face (0,0,−1), NOT the entry face the slab test tracks for tmin (which would be +z here).
        const h = rayOBB(0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(1, 9);
        expect([h.nx, h.ny, h.nz]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-1, 9),
        ]);
    });

    test("origin inside, off-axis exit picks the correct face", () => {
        // from the centre toward +x exits the +x face at t=1, normal (1,0,0)
        const h = rayOBB(0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(1, 9);
        expect(h.nx).toBeCloseTo(1, 9);
    });
});

describe("rayCapsule", () => {
    test("hits the cylinder body", () => {
        // radius 0.5, half-height 1, identity; a −z ray through the centre enters the side at z=0.5 → t=4.5
        const h = rayCapsule(0, 0, 5, 0, 0, -1, 0, 0, 0, 0.5, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(4.5, 9);
        expect(h.nz).toBeCloseTo(1, 9);
    });

    test("hits the top hemisphere cap", () => {
        // a downward ray enters the top cap at y = halfHeight + radius = 1.5 → t = 3.5, normal +y
        const h = rayCapsule(0, 5, 0, 0, -1, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 1)!;
        expect(h.t).toBeCloseTo(3.5, 9);
        expect(h.ny).toBeCloseTo(1, 9);
    });
});

describe("raycast (nearest hit over a body list)", () => {
    const box = (eid: number, z: number): RayBody => ({
        eid,
        shape: ShapeKind.Box,
        pos: [0, 0, z],
        quat: [0, 0, 0, 1],
        half: [1, 1, 1],
        radius: 0,
    });

    test("returns the nearer of two boxes", () => {
        const ray = { origin: [0, 0, 10] as const, dir: [0, 0, -1] as const };
        const hit = raycast(ray, [box(7, -5), box(3, 0)])!;
        expect(hit.eid).toBe(3); // box at z=0 (t=9) is nearer than z=−5 (t=14)
        expect(hit.distance).toBeCloseTo(9, 9);
        expect(hit.point[2]).toBeCloseTo(1, 9);
    });

    test("maxDist rejects farther hits", () => {
        const ray = { origin: [0, 0, 10] as const, dir: [0, 0, -1] as const };
        expect(raycast(ray, [box(3, 0)], 5)).toBeNull(); // nearest hit at t=9 > 5
        expect(raycast(ray, [box(3, 0)], 12)?.eid).toBe(3);
    });

    test("empty list returns null", () => {
        expect(raycast({ origin: [0, 0, 0], dir: [0, 0, -1] }, [])).toBeNull();
    });
});

describe("generateRay (screen-cursor → world ray)", () => {
    const Id: [number, number, number, number] = [0, 0, 0, 1];
    const fov = 60;
    const aspect = 16 / 9;
    const near = 0.05;

    test("centre NDC fires straight along the camera's −Z, offset to the near plane", () => {
        // identity camera at the origin: ndc (0,0) → camera forward (0,0,-1); origin sits `near` along it
        const r = generateRay(0, 0, aspect, fov, near, [0, 0, 0], Id);
        expect([r.dir[0], r.dir[1], r.dir[2]]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-1, 9),
        ]);
        expect([r.origin[0], r.origin[1], r.origin[2]]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-near, 9),
        ]);
    });

    test("right NDC edge tilts +x by aspect·tan(fov/2), normalized", () => {
        // ndc (1,0) → camera-space dir (aspect·t, 0, -1); the gold is that raw dir, normalized
        const t = Math.tan(((fov / 2) * Math.PI) / 180);
        const len = Math.hypot(aspect * t, 0, -1);
        const r = generateRay(1, 0, aspect, fov, near, [0, 0, 0], Id);
        expect(r.dir[0]).toBeCloseTo((aspect * t) / len, 9);
        expect(r.dir[1]).toBeCloseTo(0, 9);
        expect(r.dir[2]).toBeCloseTo(-1 / len, 9);
    });

    test("a +90° yaw camera fires its forward along world −X", () => {
        // quat = +90° about Y rotates camera-forward (0,0,-1) → (-1,0,0); origin from (3,0,0) steps −X
        const q: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const r = generateRay(0, 0, aspect, fov, near, [3, 0, 0], q);
        expect([r.dir[0], r.dir[1], r.dir[2]]).toEqual([
            expect.closeTo(-1, 6),
            expect.closeTo(0, 6),
            expect.closeTo(0, 6),
        ]);
        expect(r.origin[0]).toBeCloseTo(3 - near, 6);
    });
});

describe("screenToRay (pixel → world ray)", () => {
    const Id: [number, number, number, number] = [0, 0, 0, 1];
    const W = 1600;
    const H = 900;

    test("the canvas centre pixel matches the centre-NDC ray", () => {
        const px = screenToRay(W / 2, H / 2, W, H, 60, 0.05, [0, 0, 0], Id);
        const ndc = generateRay(0, 0, W / H, 60, 0.05, [0, 0, 0], Id);
        expect(px.dir[0]).toBeCloseTo(ndc.dir[0], 9);
        expect(px.dir[1]).toBeCloseTo(ndc.dir[1], 9);
        expect(px.dir[2]).toBeCloseTo(ndc.dir[2], 9);
    });

    test("top-left pixel maps to NDC (−1, +1) — left + up in camera space", () => {
        // pixel (0,0) is screen top-left → ndc (−1, +1): the ray tilts −x and +y
        const r = screenToRay(0, 0, W, H, 60, 0.05, [0, 0, 0], Id);
        expect(r.dir[0]).toBeLessThan(0);
        expect(r.dir[1]).toBeGreaterThan(0);
        expect(r.dir[2]).toBeLessThan(0);
    });
});
