import { GlobalTransform, type World } from "@dylanebert/shallot";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import { TreeBenchmarkState } from "./tree";

type Point = [number, number, number];
export const TREE_VISITED = 0xd49560;
const QUERY_LINE = { ...JOINT_LINE, width: 3 };
const CONTEXT_LINE = { ...JOINT_LINE, width: 0.25 };
export function treeGroundHeight(world: World): number {
    const s = world.resource(TreeBenchmarkState);
    return s.nodes[s.root]?.bounds.lowerBound.y ?? 0;
}
export function treeBounds(world: World, eye: Point): number[][] {
    const s = world.resource(TreeBenchmarkState);
    if (!s.active) return [];
    const f = Math.fround;
    const level = s.values.Level,
        distance = f(f(f(s.values.Kilometers * s.values.Kilometers) * 1000) * 1000);
    const rows: number[][] = [];
    for (const n of s.nodes) {
        if (!n || (level < 0 ? !n.leaf : n.depth !== level)) continue;
        const a = n.bounds.lowerBound,
            b = n.bounds.upperBound;
        const x = f(eye[0] - f(f(a.x + b.x) * 0.5)),
            y = f(eye[1] - f(f(a.y + b.y) * 0.5)),
            z = f(eye[2] - f(f(a.z + b.z) * 0.5));
        const squared = f(f(f(x * x) + f(y * y)) + f(z * z));
        if (level < 0 ? squared > distance : level >= 10 && squared >= distance) continue;
        const p = n.leaf ? s.proxies[n.user] : undefined;
        const visited = p && (p.queryStamp === s.stamp || p.rayStamp === s.stamp);
        rows.push([a.x, a.y, a.z, b.x, b.y, b.z, visited ? TREE_VISITED : INK]);
    }
    return rows;
}
export function treeQueryDrawing(world: World) {
    const s = world.resource(TreeBenchmarkState),
        v = s.values;
    const segments: [Point, Point][] = [],
        boxes: number[][] = [],
        marks: Point[] = [];
    if (!s.active || !s.leaves) return { segments, boxes, marks };
    if (v["Ray Cast"]) {
        const r = s.rays[v.Test]!;
        segments.push([
            [r.origin.x, r.origin.y, r.origin.z],
            [
                Math.fround(r.origin.x + r.translation.x),
                Math.fround(r.origin.y + r.translation.y),
                Math.fround(r.origin.z + r.translation.z),
            ],
        ]);
    }
    if (v.Overlap) {
        const b = s.overlaps[v.Test]!;
        boxes.push([
            b.lowerBound.x,
            b.lowerBound.y,
            b.lowerBound.z,
            b.upperBound.x,
            b.upperBound.y,
            b.upperBound.z,
            INK,
        ]);
    }
    if (v["Closet Point"]) {
        const q = s.closestQueries[v.Test]!,
            p: Point = [q.center.x, q.center.y, q.center.z];
        for (let axis = 0; axis < 3; axis++) {
            const ring = (i: number): Point => {
                const v: Point = [...p],
                    angle = (i * 2 * Math.PI) / 24;
                v[(axis + 1) % 3]! += q.radius * Math.cos(angle);
                v[(axis + 2) % 3]! += q.radius * Math.sin(angle);
                return v;
            };
            for (let i = 0; i < 24; i++) segments.push([ring(i), ring(i + 1)]);
        }
        if (s.haveClosest) marks.push([s.closestPoint.x, s.closestPoint.y, s.closestPoint.z]);
    }
    return { segments, boxes, marks };
}
export function drawTreeBenchmark(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const p = world.storage(GlobalTransform).translation;
    const drawing = treeQueryDrawing(world);
    const bounds = (box: number[], options: typeof JOINT_LINE) => {
        const corner = (bits: number): Point => [
            box[bits & 1 ? 3 : 0]!,
            box[bits & 2 ? 4 : 1]!,
            box[bits & 4 ? 5 : 2]!,
        ];
        for (let bits = 0; bits < 8; bits++)
            for (const axis of [1, 2, 4])
                if (!(bits & axis)) line(corner(bits), corner(bits | axis), box[6]!, options);
    };
    const querying = drawing.segments.length > 0 || drawing.boxes.length > 0;
    for (const box of treeBounds(world, [p.x.get(camera), p.y.get(camera), p.z.get(camera)]))
        bounds(box, querying && box[6] === INK ? CONTEXT_LINE : JOINT_LINE);
    for (const box of drawing.boxes) bounds(box, QUERY_LINE);
    for (const [a, b] of drawing.segments) line(a, b, INK, QUERY_LINE);
    for (const p of drawing.marks) drawInkPoint(world, camera, p, line, INK, 15);
}
