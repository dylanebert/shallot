import { Body, ShapeKind, type World } from "@dylanebert/shallot";
import { computeBodyAABB, shapeEdges } from "@dylanebert/shallot/standard/physics";
import { CollisionState, type Sweep } from "./collision";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
type Vec = { x: number; y: number; z: number };
const AXIS = 0x797369;
const ORANGE = 0xff8135;
const point = (p: Vec): Point => [p.x, p.y, p.z];
const offset = (p: Vec, v: Vec, t = 1): Point => [p.x + t * v.x, p.y + t * v.y, p.z + t * v.z];
export function collisionDrawing(world: World) {
    const s = world.resource(CollisionState);
    const segments: { a: Point; b: Point; color: number }[] = [],
        marks: { point: Point; color: number }[] = [];
    const line = (a: Point, b: Point, color = INK) => segments.push({ a, b, color });
    const mark = (p: Point, color = INK) => marks.push({ point: p, color });
    const outline = (q: Sweep, t: number, color: number) => {
        const points = q.points.map((p) =>
            offset(
                { x: p.x + q.origin.x, y: p.y + q.origin.y, z: p.z + q.origin.z },
                q.translation,
                t,
            ),
        );
        if (points.length === 8) {
            for (const [a, b] of boxEdges) line(points[a]!, points[b]!, color);
            return;
        }
        if (!points.length) return;
        for (const [a, b] of shapeEdges(world, {
            kind: points.length === 1 ? ShapeKind.Sphere : ShapeKind.Capsule,
            geometry: 0,
            scale: [1, 1, 1],
            radius: q.radius,
            sphereCenter: points[0]!,
            capsuleCenter1: points[0]!,
            capsuleCenter2: points[1] ?? points[0]!,
        }))
            line(a, b, color);
    };
    if (s.scene === "longRayCast") {
        for (let i = 0; i < 5; i++) {
            const truth = s.sweeps[2 * i],
                actual = s.sweeps[2 * i + 1];
            if (!truth || !actual) continue;
            const hit = actual.hits[0],
                reference = truth.hits[0];
            const direction = actual.direction!;
            const color = actual.color ?? AXIS;
            if (hit) {
                line(offset(hit.point, direction, 3), point(hit.point), color);
                line(point(hit.point), offset(hit.point, hit.normal, 1.5), color);
                mark(point(hit.point), color);
            } else if (reference) {
                mark(point(reference.point), ORANGE);
                line(
                    offset(reference.point, direction, 2),
                    offset(reference.point, direction, -2),
                    ORANGE,
                );
            } else {
                const aim = { x: (i - 2) * 5, y: 2.5, z: 0 };
                line(offset(aim, direction, 2), offset(aim, direction, -4), AXIS);
            }
            for (const p of s.trails[i]!) mark(p.point, p.color);
        }
    } else
        for (const q of s.sweeps) {
            const start = point(q.origin),
                end = offset(q.origin, q.translation);
            if (q.overlap !== undefined || s.scene === "overlapWorld") {
                outline(q, 0, q.overlap ? INK : AXIS);
                continue;
            }
            if (s.scene === "initialOverlap") {
                outline(q, 0, q.hits.length ? INK : AXIS);
            } else if (q.points.length) {
                // Mesh Scale draws its swept sphere relative to zero, not to the ray's start.
                const visual = s.scene === "meshScale" ? { ...q, origin: { x: 0, y: 0, z: 0 } } : q;
                if (s.scene === "shapeCast") outline(q, 0, INK);
                if (q.hits.length) for (const h of q.hits) outline(visual, h.fraction, INK);
                else outline(visual, 1, AXIS);
            }
            if (s.scene !== "shapeCast" && s.scene !== "initialOverlap") {
                line(start, end);
                mark(start);
                mark(end);
            }
            for (const h of q.hits) {
                mark(point(h.point));
                if (s.scene !== "capsuleCastRay")
                    line(
                        point(h.point),
                        offset(h.point, h.normal, s.scene === "shapeCast" ? 0.2 : 0.5),
                    );
            }
        }
    if (s.scene === "castWorld")
        for (const eid of s.ignored) {
            if (!world.has(eid, Body)) continue;
            const bounds = computeBodyAABB(world, eid);
            if (!bounds) continue;
            const a = bounds.lowerBound,
                b = bounds.upperBound;
            const p: Point[] = [
                [b.x, b.y, b.z],
                [a.x, b.y, b.z],
                [a.x, a.y, b.z],
                [b.x, a.y, b.z],
                [b.x, b.y, a.z],
                [a.x, b.y, a.z],
                [a.x, a.y, a.z],
                [b.x, a.y, a.z],
            ];
            for (const [i, j] of boxEdges) line(p[i]!, p[j]!, AXIS);
        }
    return { segments, marks };
}
const boxEdges = [
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
] as const;
export function drawCollisionView(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const { segments, marks } = collisionDrawing(world);
    for (const s of segments) line(s.a, s.b, s.color, JOINT_LINE);
    for (const m of marks) drawInkPoint(world, camera, m.point, line, m.color);
}
