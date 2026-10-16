import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { World } from "../../engine";
import { SEGMENT_FLOATS, triangleVertex } from "./pipeline";
import { drawArrow, drawLine, head, lineStaging, segmentState } from "./segments";

test("an arrow stages one filled triangle in its shaft's colour and width, facing world up by default", () => {
    const world = new World();
    try {
        drawArrow(world, [0, 0, 0], [5, 0, 0], 0xeee9df, 3);
        const values = lineStaging(world);
        const words = new Uint32Array(values.buffer);
        const o = SEGMENT_FLOATS;
        expect(Array.from(values.slice(o, o + 7))).toEqual([5, 0, 0, 3, 4.5, 0, -0.25]);
        expect(Array.from(values.slice(o + 8, o + 11))).toEqual([4.5, 0, 0.25]);
        expect(words[11]).toBe(0);
        expect(words[o + 11]).toBe(1);
        expect(words[o + 7]).toBe(words[7]);
        expect(values[o + 3]).toBe(values[3]);
        expect(words[o * 2 + 11]).toBe(0);
    } finally {
        world.dispose();
    }
});

test("two different shaft lengths given the same tip length stage the same head", () => {
    const world = new World();
    try {
        drawArrow(world, [0, 0, 0], [5, 0, 0], 0xeee9df, 3, 0.6);
        drawArrow(world, [-5, 0, 0], [5, 0, 0], 0xeee9df, 3, 0.6);
        const words = new Uint32Array(lineStaging(world).buffer);
        expect(Array.from(words.slice(SEGMENT_FLOATS, SEGMENT_FLOATS * 2))).toEqual(
            Array.from(words.slice(SEGMENT_FLOATS * 3, SEGMENT_FLOATS * 4)),
        );
        expect(lineStaging(world)[SEGMENT_FLOATS + 4]).toBeCloseTo(4.4, 6);
    } finally {
        world.dispose();
    }
});

test("the head is one nondegenerate triangle perpendicular to its supplied normal, independent of a view", () => {
    const world = new World();
    try {
        drawLine(world, [0, 2, 3], [5, 2, 3], 0xffffff, 3);
        expect(head(segmentState(world), 1, [0, 0, 2])).toBe(1);
        const row = lineStaging(world).subarray(SEGMENT_FLOATS);
        const a = d.vec3f(row[0], row[1], row[2]);
        const b = d.vec3f(row[4], row[5], row[6]);
        const c = d.vec3f(row[8], row[9], row[10]);
        const normal = d.vec3f(0, 0, 1);
        const vertices = Array.from({ length: 6 }, (_, i) => triangleVertex(a, b, c, i));
        expect(vertices.slice(0, 3)).toEqual([a, b, c]);
        for (const point of vertices) expect(std.dot(std.sub(point, a), normal)).toBe(0);
        expect(std.dot(std.cross(std.sub(b, a), std.sub(c, a)), normal)).toBeCloseTo(1, 6);
        expect(vertices.slice(3)).toEqual([a, a, a]);
    } finally {
        world.dispose();
    }
});

test("a plane normal must be nonzero and perpendicular to the shaft; an empty head emits nothing", () => {
    const world = new World();
    try {
        for (const normal of [
            [0, 0, 0],
            [1, 1, 0],
        ]) {
            expect(() => drawArrow(world, [0, 0, 0], [1, 0, 0], 0xffffff, 1, 1, normal)).toThrow(
                "perpendicular",
            );
        }
        drawLine(world, [0, 0, 0], [0, 0, 0], 0xffffff);
        expect(head(segmentState(world), 1, [0, 1, 0])).toBe(0);
        drawLine(world, [0, 0, 0], [1, 0, 0], 0xffffff);
        expect(head(segmentState(world), 0, [0, 1, 0])).toBe(0);
    } finally {
        world.dispose();
    }
});
