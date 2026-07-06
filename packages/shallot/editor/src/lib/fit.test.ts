import { describe, expect, test } from "bun:test";
import { enclose, frameDistance, frameSize, type Sphere } from "./fit";

describe("enclose", () => {
    test("empty set is null", () => {
        expect(enclose([])).toBeNull();
    });

    test("a single sphere encloses to itself", () => {
        const s: Sphere = { center: [3, -2, 5], radius: 4 };
        const got = enclose([s])!;
        expect(got.center).toEqual([3, -2, 5]);
        expect(got.radius).toBeCloseTo(4, 6);
    });

    test("two points one unit apart enclose to radius 0.5 at their midpoint", () => {
        const got = enclose([
            { center: [0, 0, 0], radius: 0 },
            { center: [1, 0, 0], radius: 0 },
        ])!;
        expect(got.center).toEqual([0.5, 0, 0]);
        expect(got.radius).toBeCloseTo(0.5, 6);
    });

    test("the enclosing sphere includes each sphere's own extent, not just its center", () => {
        // two unit spheres at x=±2: each far point reaches x=±3, so the tight enclosing sphere is
        // radius 3 at the origin — a center-only enclose would give 2.
        const got = enclose([
            { center: [-2, 0, 0], radius: 1 },
            { center: [2, 0, 0], radius: 1 },
        ])!;
        expect(got.center).toEqual([0, 0, 0]);
        expect(got.radius).toBeCloseTo(3, 6);
    });
});

describe("frameDistance", () => {
    // a sphere is tangent to the frustum when distance·sin(fov/2) == radius·padding. With padding 1 and a
    // 90° square viewport, sin(45°) = √½, so distance = radius / √½ = radius·√2.
    test("90° square viewport, no padding: distance = radius·√2", () => {
        expect(frameDistance(1, 90, 1, 1)).toBeCloseTo(Math.SQRT2, 6);
    });

    test("padding scales distance linearly", () => {
        const base = frameDistance(1, 60, 1.6, 1);
        expect(frameDistance(1, 60, 1.6, 1.3)).toBeCloseTo(base * 1.3, 6);
    });

    test("distance scales linearly with radius", () => {
        const unit = frameDistance(1, 60, 1.6);
        expect(frameDistance(10, 60, 1.6)).toBeCloseTo(unit * 10, 6);
    });

    // a portrait viewport (aspect < 1) has a narrower horizontal FOV than vertical, so it's the limiter:
    // the camera must pull back further than the vertical FOV alone would suggest.
    test("portrait viewport frames on the narrower horizontal FOV", () => {
        const wide = frameDistance(1, 60, 1.6); // landscape: vertical FOV limits
        const tall = frameDistance(1, 60, 0.5); // portrait: horizontal FOV limits
        expect(tall).toBeGreaterThan(wide);
    });
});

describe("frameSize", () => {
    test("landscape: size is radius·padding (height limits)", () => {
        expect(frameSize(2, 1.6, 1.3)).toBeCloseTo(2.6, 6);
    });

    test("portrait: size grows so width fits too", () => {
        // aspect 0.5 → width = size·0.5 must cover the radius, so size doubles vs the landscape fit.
        expect(frameSize(2, 0.5, 1)).toBeCloseTo(4, 6);
    });
});
