import { expect, test } from "bun:test";
import { generateRay, screenToRay } from "./viewport-to-world";

test("the NDC-to-world ray drops the aspect ratio, the near offset or the camera rotation, so a pick would miss the object under the cursor on a non-square canvas or a turned camera", () => {
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
});

test("the pixel-to-NDC conversion drops the y flip or mis-centres the canvas, so a cursor above the centre would pick a body below it", () => {
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
});
