import { expect, test } from "bun:test";
import { capsule, cube, sphere } from "./primitives";

for (const [name, build] of [
    ["sphere", (segments: number, rings: number) => sphere(0.5, segments, rings)],
    ["capsule", (segments: number, rings: number) => capsule(0.5, 1, segments, rings)],
] as const) {
    for (const [parameter, minimum] of [
        ["segments", 3],
        ["rings", 2],
    ] as const) {
        for (const value of [minimum + 0.5, minimum - 1, 0, -1, NaN, Infinity, -Infinity]) {
            test(`${name} refuses ${parameter}=${value}`, () => {
                expect(() =>
                    build(
                        parameter === "segments" ? value : 16,
                        parameter === "rings" ? value : 16,
                    ),
                ).toThrow(`${parameter} must be an integer >= ${minimum}; received ${value}`);
            });
        }
    }

    for (const segments of [3, 4, 16, 32]) {
        for (const rings of [2, 3, 4, 15, 16]) {
            test(`${name}(${segments}, ${rings}) has in-range indices and every triangle faces outward`, () => {
                const { vertices, indices } = build(segments, rings);
                for (const index of indices) expect(index).toBeLessThan(vertices.length / 8);
                for (let i = 0; i < indices.length; i += 3) {
                    const a = indices[i] * 8;
                    const b = indices[i + 1] * 8;
                    const c = indices[i + 2] * 8;
                    const ab = [0, 1, 2].map((k) => vertices[b + k] - vertices[a + k]);
                    const ac = [0, 1, 2].map((k) => vertices[c + k] - vertices[a + k]);
                    const cross = [
                        ab[1] * ac[2] - ab[2] * ac[1],
                        ab[2] * ac[0] - ab[0] * ac[2],
                        ab[0] * ac[1] - ab[1] * ac[0],
                    ];
                    const normal = [4, 5, 6].map(
                        (k) => vertices[a + k] + vertices[b + k] + vertices[c + k],
                    );
                    expect(
                        cross.reduce((dot, value, k) => dot + value * normal[k], 0),
                    ).toBeGreaterThan(0);
                }
            });
        }
    }
}

// Per-axis [min, max] of the interleaved positions (eight floats per vertex, xyz first).
function bounds(vertices: Float32Array): [number, number][] {
    return [0, 1, 2].map((axis) => {
        let min = Infinity;
        let max = -Infinity;
        for (let i = axis; i < vertices.length; i += 8) {
            min = Math.min(min, vertices[i]);
            max = Math.max(max, vertices[i]);
        }
        return [min, max];
    });
}

test("cube takes full lengths, centered on the origin", () => {
    expect(bounds(cube([2, 4, 6]).vertices)).toEqual([
        [-1, 1],
        [-2, 2],
        [-3, 3],
    ]);
});

test("the default cube spans one meter on each axis", () => {
    expect(bounds(cube().vertices)).toEqual([
        [-0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, 0.5],
    ]);
});

test("capsule takes radius and the mid-section's full length, centered on the origin", () => {
    const [x, y, z] = bounds(capsule(0.25, 2).vertices);
    expect(x[0]).toBeCloseTo(-0.25);
    expect(x[1]).toBeCloseTo(0.25);
    expect(y).toEqual([-1.25, 1.25]);
    expect(z[0]).toBeCloseTo(-0.25);
    expect(z[1]).toBeCloseTo(0.25);
});

test("the default capsule has radius 0.5 and total height 2", () => {
    const [x, y, z] = bounds(capsule().vertices);
    expect(x[0]).toBeCloseTo(-0.5);
    expect(x[1]).toBeCloseTo(0.5);
    expect(y).toEqual([-1, 1]);
    expect(z[0]).toBeCloseTo(-0.5);
    expect(z[1]).toBeCloseTo(0.5);
});
