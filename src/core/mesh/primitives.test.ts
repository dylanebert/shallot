import { expect, test } from "bun:test";
import { cube } from "./primitives";

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
