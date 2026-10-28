import { ShapeKind, type World } from "@dylanebert/shallot";
import { getSweepTransform, shapeEdges } from "@dylanebert/shallot/standard/physics";
import {
    CollisionDebug,
    castCapsule,
    castTranslation,
    castTriangle,
    identity,
    type Vec,
} from "./collision-debug";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
const CYAN = 0x38a4b5,
    ORANGE = 0xff8135,
    AXIS = 0x797369;
const p = (v: Vec): Point => [v.x, v.y, v.z];
const add = (a: Vec, b: Vec, t = 1): Vec => ({
    x: a.x + t * b.x,
    y: a.y + t * b.y,
    z: a.z + t * b.z,
});
function transform(v: Vec, xf: ReturnType<typeof identity>): Vec {
    const q = xf.q,
        u = q.v;
    const t = {
        x: 2 * (u.y * v.z - u.z * v.y),
        y: 2 * (u.z * v.x - u.x * v.z),
        z: 2 * (u.x * v.y - u.y * v.x),
    };
    return {
        x: xf.p.x + v.x + q.s * t.x + u.y * t.z - u.z * t.y,
        y: xf.p.y + v.y + q.s * t.y + u.z * t.x - u.x * t.z,
        z: xf.p.z + v.z + q.s * t.z + u.x * t.y - u.y * t.x,
    };
}
export function debugDrawing(world: World) {
    const s = world.resource(CollisionDebug);
    const segments: { a: Point; b: Point; color: number }[] = [],
        marks: { point: Point; color: number }[] = [],
        labels: { point: Point; label: string }[] = [];
    const line = (a: Vec, b: Vec, color = INK) => segments.push({ a: p(a), b: p(b), color });
    const mark = (a: Vec, color = INK) => marks.push({ point: p(a), color });
    function shape(
        points: Vec[],
        xf: ReturnType<typeof identity>,
        radius: number,
        color: number,
        indices = false,
    ) {
        const ps = points.map((v) => transform(v, xf));
        if (points.length <= 2) {
            if (!radius) {
                if (ps.length === 1) mark(ps[0]!, color);
                else line(ps[0]!, ps[1]!, color);
            } else
                for (const [a, b] of shapeEdges(world, {
                    kind: points.length === 1 ? ShapeKind.Sphere : ShapeKind.Capsule,
                    geometry: 0,
                    scale: [1, 1, 1],
                    radius,
                    sphereCenter: p(points[0]!),
                    capsuleCenter1: p(points[0]!),
                    capsuleCenter2: p(points[1] ?? points[0]!),
                }))
                    line(
                        transform({ x: a[0], y: a[1], z: a[2] }, xf),
                        transform({ x: b[0], y: b[1], z: b[2] }, xf),
                        color,
                    );
        } else {
            const edges =
                points.length === 3
                    ? [
                          [0, 1],
                          [1, 2],
                          [2, 0],
                      ]
                    : [
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
            for (const [a, b] of edges) line(ps[a!]!, ps[b!]!, color);
        }
        if (indices)
            ps.forEach((v, i) => {
                labels.push({ point: p(v), label: String(i) });
            });
    }
    if (s.scene === "shapeCastDebug") {
        shape(castTriangle, identity(), 0, CYAN);
        shape(castCapsule, s.transform, Math.fround(Math.fround(0.01)), INK);
        if (s.cast?.hit)
            shape(
                castCapsule,
                { ...s.transform, p: add(s.transform.p, castTranslation, s.cast.fraction) },
                0.01,
                ORANGE,
            );
        shape(castCapsule, { ...s.transform, p: add(s.transform.p, castTranslation) }, 0.01, AXIS);
    } else if (s.scene === "timeOfImpact" && s.toi) {
        shape(s.pointsA, identity(), s.typeA === 1 ? 0.02 : 0, CYAN, s.typeA === 2);
        const draw = (t: number, color: number) =>
            shape(
                s.pointsB,
                getSweepTransform(world, s.sweepB!, t),
                s.typeB === 1 ? 0.02 : 0,
                color,
                s.typeB === 2,
            );
        draw(0, INK);
        draw(1, AXIS);
        if (s.toi.fraction < 1) draw(s.toi.fraction, ORANGE);
        if (s.toi.state === 1 || s.toi.state === 3) {
            mark(s.toi.point);
            line(s.toi.point, add(s.toi.point, s.toi.normal, 0.5));
        }
    } else if (s.distance) {
        const d = s.distance,
            debug = s.scene === "distanceDebug";
        shape(s.pointsA, identity(), s.radiusA, CYAN, !debug && s.showIndices);
        shape(s.pointsB, s.transform, s.radiusB, INK, debug || s.showIndices);
        if (debug || !s.drawSimplex) {
            mark(d.pointA, CYAN);
            mark(d.pointB, INK);
            line(d.pointA, d.pointB);
            line(d.pointA, add(d.pointA, d.normal, debug ? 1 : 0.5));
        }
        if (debug || s.drawSimplex) {
            const index = Math.min(s.index, d.simplexCount - 1),
                simplex = d.simplexes[index];
            if (simplex && index >= 0) {
                if (debug || index > 0) {
                    let a = { x: 0, y: 0, z: 0 },
                        b = { x: 0, y: 0, z: 0 };
                    for (let i = 0; i < simplex.count; i++) {
                        const v = simplex.vertices[i]!;
                        a = add(a, v.wA, v.a);
                        b = add(b, v.wB, v.a);
                    }
                    if (simplex.count === 1) {
                        a = simplex.vertices[0]!.wA;
                        b = simplex.vertices[0]!.wB;
                    }
                    if (simplex.count === 4) b = a;
                    mark(a);
                    mark(b);
                    if (!debug) line(a, b);
                }
                for (let i = 0; i < simplex.count; i++) {
                    const v = simplex.vertices[i]!,
                        color = [INK, CYAN, ORANGE, 0xffb62e][i]!;
                    mark(v.wA, color);
                    mark(v.wB, color);
                }
            }
        }
    }
    return { segments, marks, labels };
}
export function drawCollisionDebug(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
) {
    const drawing = debugDrawing(world);
    for (const s of drawing.segments) line(s.a, s.b, s.color, JOINT_LINE);
    for (const m of drawing.marks) drawInkPoint(world, camera, m.point, line, m.color);
}
