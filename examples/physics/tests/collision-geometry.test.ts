import { expect, test } from "bun:test";
import { World } from "@dylanebert/shallot";
import { vec } from "../src/collision-debug";
import { CollisionGeometry } from "../src/collision-geometry";
import { collisionGeometryDrawing } from "../src/collision-geometry-presentation";

test("Hull 2D draws every input once, both ordered outlines and lifted simplified points", () => {
    const world = new World();
    const s = world.resource(CollisionGeometry);
    Object.assign(s, {
        scene: "hull2D",
        drawScale: 50,
        input: [
            [-1, -1],
            [0, 0],
            [1, -1],
            [1, 1],
            [-1, 1],
        ].map(([x, y], originalIndex) => ({ p: { x: x!, y: y! }, separation: 0, originalIndex })),
    });
    s.hull = [0, 2, 3, 4].map((i) => s.input[i]!);
    s.simplified = [0, 3, 4].map((i) => s.input[i]!);
    const d = collisionGeometryDrawing(world);
    expect(d.segments).toHaveLength(7);
    expect(d.segments[0]).toEqual({ a: [-50, -50, 0], b: [50, -50, 0], color: 0xff8135 });
    expect(d.segments[4]).toEqual({
        a: [-50, -50, Math.fround(0.01)],
        b: [50, 50, Math.fround(0.01)],
        color: 0x38a4b5,
    });
    expect(d.marks).toHaveLength(8);
    expect(d.marks.filter((p) => p.size === 4)).toHaveLength(1);
    expect(d.marks.filter((p) => p.size === 6)).toHaveLength(4);
    expect(d.marks.filter((p) => p.size === 10)).toHaveLength(3);
    expect(d.labels.map((p) => p.label).sort()).toEqual([0, 1, 2, 3, 4].map((i) => `  ${i}`));
    s.drawLabels = false;
    expect(collisionGeometryDrawing(world).labels).toEqual([]);
});
test("Encroach Cast retains start, full translation, stopped box, hit point and normal", () => {
    const world = new World();
    const s = world.resource(CollisionGeometry);
    s.scene = "encroachCast";
    s.transform.p = vec(0, 0.5, 0);
    s.translation = vec(4, 0, 0);
    s.cast = {
        point: vec(1, 0, 0),
        normal: vec(0, 1, 0),
        fraction: 0.25,
        iterations: 3,
        hit: true,
    };
    const d = collisionGeometryDrawing(world);
    expect(d.segments).toHaveLength(38);
    expect(d.segments[24]!.a).toEqual([1.5, 1, 0.5]);
    expect(d.segments[36]).toEqual({ a: [0, 0.5, 0], b: [4, 0.5, 0], color: 0xeee9df });
    expect(d.segments[37]).toEqual({ a: [1, 0, 0], b: [1, 0.5, 0], color: 0x38a4b5 });
    expect(d.marks).toEqual([{ point: [1, 0, 0], size: 8, color: 0xff8135 }]);
    s.cast.hit = false;
    const miss = collisionGeometryDrawing(world);
    expect(miss.segments).toHaveLength(37);
    expect(miss.segments[24]!.a).toEqual([4.5, 1, 0.5]);
    expect(miss.marks).toEqual([]);
});
