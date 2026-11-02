import type { World } from "@dylanebert/shallot";
import type { Point2D } from "@dylanebert/shallot/standard/physics";
import { box } from "./collision-debug";
import { CollisionGeometry } from "./collision-geometry";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
const f = Math.fround;
const AMBER = 0xff8135,
    CYAN = 0x38a4b5,
    MUTED = 0x797369;
export function collisionGeometryDrawing(world: World) {
    const s = world.resource(CollisionGeometry);
    const segments: { a: Point; b: Point; color: number }[] = [];
    const marks: { point: Point; size: number; color: number }[] = [];
    const labels: { point: Point; label: string }[] = [];
    const line = (a: Point, b: Point, color: number) => segments.push({ a, b, color });
    if (s.scene === "hull2D") {
        const position = (p: Point2D, z = 0): Point => [
            f(s.drawScale * p.p.x),
            f(s.drawScale * p.p.y),
            f(z),
        ];
        const onHull = new Set(s.hull.map((p) => p.originalIndex));
        for (const p of s.input)
            if (!onHull.has(p.originalIndex)) {
                const point = position(p);
                marks.push({ point, color: MUTED, size: 4 });
                if (s.drawLabels) labels.push({ point, label: `  ${p.originalIndex}` });
            }
        for (let i = 0; i < s.hull.length; i++) {
            const p = s.hull[i]!;
            const point = position(p);
            line(point, position(s.hull[(i + 1) % s.hull.length]!), AMBER);
            marks.push({ point, color: AMBER, size: 6 });
            if (s.drawLabels) labels.push({ point, label: `  ${p.originalIndex}` });
        }
        for (let i = 0; i < s.simplified.length; i++) {
            const point = position(s.simplified[i]!, 0.01);
            line(point, position(s.simplified[(i + 1) % s.simplified.length]!, 0.01), CYAN);
            marks.push({ point, color: CYAN, size: 10 });
        }
    } else if (s.scene === "encroachCast" && s.cast) {
        const edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ];
        const drawBox = (points: ReturnType<typeof box>, offset: Point, color: number) => {
            const vertices = points.map(
                (p) => [f(p.x + offset[0]), f(p.y + offset[1]), f(p.z + offset[2])] as Point,
            );
            for (const [a, b] of edges) line(vertices[a!]!, vertices[b!]!, color);
        };
        const start: Point = [s.transform.p.x, s.transform.p.y, s.transform.p.z];
        const t = s.translation;
        const fraction = s.cast.hit ? s.cast.fraction : 1;
        const end: Point = [
            f(start[0] + f(fraction * t.x)),
            f(start[1] + f(fraction * t.y)),
            f(start[2] + f(fraction * t.z)),
        ];
        drawBox(box(3, 0.25, 3, -0.25), [0, 0, 0], MUTED);
        drawBox(box(0.5, 0.5, 0.5), start, AMBER);
        drawBox(box(0.5, 0.5, 0.5), end, CYAN);
        line(start, [f(start[0] + t.x), f(start[1] + t.y), f(start[2] + t.z)], INK);
        if (s.cast.hit) {
            const p = s.cast.point,
                n = s.cast.normal;
            const point: Point = [p.x, p.y, p.z];
            marks.push({ point, size: 8, color: AMBER });
            line(
                point,
                [f(p.x + f(0.5 * n.x)), f(p.y + f(0.5 * n.y)), f(p.z + f(0.5 * n.z))],
                CYAN,
            );
        }
    }
    return { segments, marks, labels };
}
export function drawCollisionGeometry(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const drawing = collisionGeometryDrawing(world);
    for (const segment of drawing.segments) line(segment.a, segment.b, segment.color, JOINT_LINE);
    for (const mark of drawing.marks)
        drawInkPoint(world, camera, mark.point, line, mark.color, mark.size);
}
