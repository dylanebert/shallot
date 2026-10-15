import { expect, test } from "bun:test";
import { capsule, cube } from "./primitives";

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
