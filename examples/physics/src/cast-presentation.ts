import { GlobalTransform, qRotate, Shape, ShapeKind, type World } from "@dylanebert/shallot";
import { worldToViewport } from "@dylanebert/shallot/rendering";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import {
    CastState,
    mover,
    moverOrigin,
    overlapOrigin,
    overlapProxy,
    rayOrigin,
    rayTranslation,
    sphereOrigin,
    sphereTranslation,
} from "./cast";
import { JOINT_DOT, JOINT_INK, JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
type Vec = { x: number; y: number; z: number };
const point = (v: Vec): Point => [v.x, v.y, v.z];
const offset = (a: Vec, b: Vec, scale = 1): Point => [
    a.x + scale * b.x,
    a.y + scale * b.y,
    a.z + scale * b.z,
];
export function castDrawing(world: World) {
    const s = world.resource(CastState);
    const segments: { a: Point; b: Point; color: number }[] = [];
    const marks: Point[] = [];
    if (!s.body) return { segments, marks };
    const line = (a: Point, b: Point, color = JOINT_INK) => segments.push({ a, b, color });
    const shape = world.get(s.body, Shape)!;
    const outline = (edges: [Point, Point][], origin: Point, color = JOINT_INK) => {
        for (const [a, b] of edges)
            line(
                a.map((v, i) => v + origin[i]!) as Point,
                b.map((v, i) => v + origin[i]!) as Point,
                color,
            );
    };
    const q = s.placement.q;
    outline(
        shapeEdges(world, shape).map(
            (edge) => edge.map((p) => qRotate(q.v.x, q.v.y, q.v.z, q.s, ...p)) as [Point, Point],
        ),
        point(s.placement.p),
    );
    line(point(rayOrigin), offset(rayOrigin, rayTranslation));
    if (s.rayHit) {
        marks.push(point(s.ray.point));
        line(point(s.ray.point), offset(s.ray.point, s.ray.normal, 0.2));
    }
    const sphere = shapeEdges(world, {
        ...shape,
        kind: ShapeKind.Sphere,
        sphereCenter: [0, 0, 0],
        radius: 0.2,
    });
    outline(sphere, point(sphereOrigin));
    outline(sphere, offset(sphereOrigin, sphereTranslation, s.sphereHit ? s.sphere.fraction : 1));
    const capsule = (origin: Vec, a: Vec, b: Vec, radius: number, color: number) =>
        outline(
            shapeEdges(world, {
                ...shape,
                kind: ShapeKind.Capsule,
                capsuleCenter1: point(a),
                capsuleCenter2: point(b),
                radius,
            }),
            point(origin),
            color,
        );
    capsule(
        overlapOrigin,
        overlapProxy.points[0]!,
        overlapProxy.points[1]!,
        overlapProxy.radius,
        s.overlap ? JOINT_INK : 0x797369,
    );
    capsule(moverOrigin, mover.center1, mover.center2, mover.radius, JOINT_INK);
    for (const { plane } of s.planes) {
        const p = offset(moverOrigin, plane.point);
        marks.push(p);
        line(p, [
            p[0] + 0.2 * plane.plane.normal.x,
            p[1] + 0.2 * plane.plane.normal.y,
            p[2] + 0.2 * plane.plane.normal.z,
        ]);
    }
    return { segments, marks };
}
export function drawCastView(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const drawing = castDrawing(world);
    for (const segment of drawing.segments) line(segment.a, segment.b, segment.color, JOINT_LINE);
    const q = world.storage(GlobalTransform).rotation;
    const right = qRotate(
        q.x.get(camera),
        q.y.get(camera),
        q.z.get(camera),
        q.w.get(camera),
        1,
        0,
        0,
    );
    for (const p of drawing.marks) {
        const center = worldToViewport(world, camera, p);
        const unit = worldToViewport(world, camera, p.map((v, i) => v + right[i]!) as Point);
        if (!center || !unit) continue;
        const half = 2 / Math.hypot(unit[0] - center[0], unit[1] - center[1]);
        line(
            p.map((v, i) => v - half * right[i]!) as Point,
            p.map((v, i) => v + half * right[i]!) as Point,
            JOINT_INK,
            JOINT_DOT,
        );
    }
}
