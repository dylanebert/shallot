import { expect } from "bun:test";
import { check } from "../../harness/check";
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

// Closed-form gold for the CPU raycast — every expected t/normal is hand-derived from the geometry,
// not read off the implementation. f64, so the analytic solves are exact to ~1e-9.

check(
    "qRotate applies a 90-degree yaw and its conjugate inverts it",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the raycast quaternion rotation turns a vector the wrong way, so picking a rotated body would test the ray against a mirrored orientation",
    },
    () => {
        // q = (0, sin45, 0, cos45): +90 about Y. Active rotation: -z becomes -x.
        const s = Math.SQRT1_2;
        const [x, y, z] = qRotate(0, s, 0, s, 0, 0, -1);
        expect(x).toBeCloseTo(-1, 9);
        expect(y).toBeCloseTo(0, 9);
        expect(z).toBeCloseTo(0, 9);
        const [bx, by, bz] = qRotate(-0, -s, -0, s, x, y, z);
        expect(bx).toBeCloseTo(0, 9);
        expect(by).toBeCloseTo(0, 9);
        expect(bz).toBeCloseTo(-1, 9);
    },
);

check(
    "raySphere returns the near root with an outward normal, and null on a miss",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the ray-sphere solve takes the far root or an inward normal, so clicking a sphere would report the back surface",
    },
    () => {
        // unit sphere at origin, ray from z=5 toward -z enters the +z face at z=1, distance 4
        const h = raySphere(0, 0, 5, 0, 0, -1, 0, 0, 0, 1);
        expect(h).not.toBeNull();
        expect(h!.t).toBeCloseTo(4, 9);
        expect([h!.nx, h!.ny, h!.nz]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(1, 9),
        ]);
        expect(raySphere(2, 0, 5, 0, 0, -1, 0, 0, 0, 1)).toBeNull();
    },
);

check(
    "rayOBB hits each face at the analytic distance and misses cleanly",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the oriented-box slab test picks the wrong entry face or ignores the body rotation, so picking a turned box would report a wrong distance or normal",
    },
    () => {
        const headOn = rayOBB(0, 0, 5, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(headOn.t).toBeCloseTo(4, 9);
        expect(headOn.nz).toBeCloseTo(1, 9);

        const plusX = rayOBB(5, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(plusX.t).toBeCloseTo(4, 9);
        expect(plusX.nx).toBeCloseTo(1, 9);

        // half (2,1,1) turned 90 about Y: the 2-extent now spans world z, so a -z ray hits z=2 at
        // t=3, and the local -x entry face rotates back to a world +z normal.
        const q: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const turned = rayOBB(0, 0, 5, 0, 0, -1, 0, 0, 0, 2, 1, 1, q[0], q[1], q[2], q[3])!;
        expect(turned.t).toBeCloseTo(3, 6);
        expect(turned.nz).toBeCloseTo(1, 6);

        expect(rayOBB(3, 0, 5, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)).toBeNull();
    },
);

check(
    "rayOBB from inside returns the exit distance paired with the exit face normal",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "an oriented-box hit from a ray origin inside the box reports the entry face the slab test tracks for tmin, so a camera inside geometry would pick a surface behind it",
    },
    () => {
        // unit box at origin, ray from the centre toward -z exits the -z face at t=1; the normal is
        // that exit face (0,0,-1), NOT the entry face (+z here).
        const back = rayOBB(0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(back.t).toBeCloseTo(1, 9);
        expect([back.nx, back.ny, back.nz]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-1, 9),
        ]);

        // from the centre toward +x exits the +x face at t=1, normal (1,0,0)
        const side = rayOBB(0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1)!;
        expect(side.t).toBeCloseTo(1, 9);
        expect(side.nx).toBeCloseTo(1, 9);
    },
);

check(
    "rayCapsule separates the cylinder body from the hemispherical caps",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the ray-capsule solve treats the caps as part of the infinite cylinder, so a ray along the capsule axis would hit at the wrong distance",
    },
    () => {
        // radius 0.5, half-height 1, identity; a -z ray through the centre enters the side at
        // z=0.5, so t=4.5
        const side = rayCapsule(0, 0, 5, 0, 0, -1, 0, 0, 0, 0.5, 1, 0, 0, 0, 1)!;
        expect(side.t).toBeCloseTo(4.5, 9);
        expect(side.nz).toBeCloseTo(1, 9);

        // a downward ray enters the top cap at y = halfHeight + radius = 1.5, so t = 3.5, normal +y
        const cap = rayCapsule(0, 5, 0, 0, -1, 0, 0, 0, 0, 0.5, 1, 0, 0, 0, 1)!;
        expect(cap.t).toBeCloseTo(3.5, 9);
        expect(cap.ny).toBeCloseTo(1, 9);
    },
);

check(
    "raycast returns the nearest body, honours maxDist, and returns null on an empty list",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the raycast sweep over a body list keeps the last hit rather than the nearest or ignores maxDist, so clicking overlapping bodies would select the one behind",
    },
    () => {
        const box = (eid: number, z: number): RayBody => ({
            eid,
            shape: ShapeKind.Box,
            pos: [0, 0, z],
            quat: [0, 0, 0, 1],
            half: [1, 1, 1],
            radius: 0,
        });
        const ray = { origin: [0, 0, 10] as const, dir: [0, 0, -1] as const };

        const hit = raycast(ray, [box(7, -5), box(3, 0)])!;
        expect(hit.eid).toBe(3); // box at z=0 (t=9) is nearer than z=-5 (t=14)
        expect(hit.distance).toBeCloseTo(9, 9);
        expect(hit.point[2]).toBeCloseTo(1, 9);

        expect(raycast(ray, [box(3, 0)], 5)).toBeNull(); // nearest hit at t=9 > 5
        expect(raycast(ray, [box(3, 0)], 12)?.eid).toBe(3);

        expect(raycast({ origin: [0, 0, 0], dir: [0, 0, -1] }, [])).toBeNull();
    },
);

check(
    "generateRay unprojects NDC through the camera's fov, aspect and pose",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the NDC-to-world ray drops the aspect ratio, the near offset or the camera rotation, so a pick would miss the object under the cursor on a non-square canvas or a turned camera",
    },
    () => {
        const Id: [number, number, number, number] = [0, 0, 0, 1];
        const fov = 60;
        const aspect = 16 / 9;
        const near = 0.05;

        // identity camera at the origin: ndc (0,0) gives camera forward (0,0,-1), origin `near` along it
        const centre = generateRay(0, 0, aspect, fov, near, [0, 0, 0], Id);
        expect([centre.dir[0], centre.dir[1], centre.dir[2]]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-1, 9),
        ]);
        expect([centre.origin[0], centre.origin[1], centre.origin[2]]).toEqual([
            expect.closeTo(0, 9),
            expect.closeTo(0, 9),
            expect.closeTo(-near, 9),
        ]);

        // ndc (1,0) gives camera-space dir (aspect*t, 0, -1); the gold is that raw dir, normalized
        const t = Math.tan(((fov / 2) * Math.PI) / 180);
        const len = Math.hypot(aspect * t, 0, -1);
        const right = generateRay(1, 0, aspect, fov, near, [0, 0, 0], Id);
        expect(right.dir[0]).toBeCloseTo((aspect * t) / len, 9);
        expect(right.dir[1]).toBeCloseTo(0, 9);
        expect(right.dir[2]).toBeCloseTo(-1 / len, 9);

        // quat = +90 about Y rotates camera-forward (0,0,-1) to (-1,0,0); origin from (3,0,0) steps -X
        const q: [number, number, number, number] = [0, Math.SQRT1_2, 0, Math.SQRT1_2];
        const yawed = generateRay(0, 0, aspect, fov, near, [3, 0, 0], q);
        expect([yawed.dir[0], yawed.dir[1], yawed.dir[2]]).toEqual([
            expect.closeTo(-1, 6),
            expect.closeTo(0, 6),
            expect.closeTo(0, 6),
        ]);
        expect(yawed.origin[0]).toBeCloseTo(3 - near, 6);
    },
);

check(
    "screenToRay maps pixels to NDC with the y axis flipped",
    {
        class: "pure",
        tier: "step",
        premises: [],
        budget: 1000,
        claim: "the pixel-to-NDC conversion drops the y flip or mis-centres the canvas, so a cursor above the centre would pick a body below it",
    },
    () => {
        const Id: [number, number, number, number] = [0, 0, 0, 1];
        const W = 1600;
        const H = 900;

        const px = screenToRay(W / 2, H / 2, W, H, 60, 0.05, [0, 0, 0], Id);
        const ndc = generateRay(0, 0, W / H, 60, 0.05, [0, 0, 0], Id);
        expect(px.dir[0]).toBeCloseTo(ndc.dir[0], 9);
        expect(px.dir[1]).toBeCloseTo(ndc.dir[1], 9);
        expect(px.dir[2]).toBeCloseTo(ndc.dir[2], 9);

        // pixel (0,0) is screen top-left, i.e. ndc (-1, +1): the ray tilts -x and +y
        const corner = screenToRay(0, 0, W, H, 60, 0.05, [0, 0, 0], Id);
        expect(corner.dir[0]).toBeLessThan(0);
        expect(corner.dir[1]).toBeGreaterThan(0);
        expect(corner.dir[2]).toBeLessThan(0);
    },
);
