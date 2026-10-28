import { ShapeKind, type World } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { Geometry } from "./geometry";
import { JOINT_INK, JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
const point = (v: { x: number; y: number; z: number }): Point => [v.x, v.y, v.z];
export function geometryDrawing(world: World) {
    const s = world.resource(Geometry);
    const segments: { a: Point; b: Point; color: number }[] = [];
    for (const [i, h] of s.hulls.entries()) {
        if (!h) continue;
        const x = s.scene === "hullTransform" ? (i === 0 ? -2 : 2) : 0;
        for (const [j, e] of h.edges.entries()) {
            if (j > e.twin) continue;
            const a = point(h.points[e.origin]!),
                b = point(h.points[h.edges[e.twin]!.origin]!);
            a[0] += x;
            b[0] += x;
            segments.push({ a, b, color: JOINT_INK });
        }
    }
    if (s.scene === "capsuleMass")
        for (const [a, b] of shapeEdges(world, {
            geometry: 0,
            scale: [1, 1, 1],
            sphereCenter: [0, 0, 0],
            kind: ShapeKind.Capsule,
            capsuleCenter1: [-1, 0, 0],
            capsuleCenter2: [1, 0, 0],
            radius: 1,
        }))
            segments.push({ a, b, color: JOINT_INK });
    return segments;
}
export function drawGeometry(
    world: World,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    for (const { a, b, color } of geometryDrawing(world)) line(a, b, color, JOINT_LINE);
}
