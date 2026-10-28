import { ShapeKind, type World } from "@dylanebert/shallot";
import { type HullData, shapeEdges, shapeMesh } from "@dylanebert/shallot/standard/physics";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import { type ManifoldPose, ManifoldSample, manifoldRotate, manifoldTransform } from "./manifold";

type Point = [number, number, number];
const point = (v: { x: number; y: number; z: number }): Point => [v.x, v.y, v.z];
export function manifoldDrawing(world: World) {
    const s = world.resource(ManifoldSample),
        m = s.manifold;
    const marks: { point: Point; color: number }[] = [],
        segments: { a: Point; b: Point; color: number }[] = [];
    if (!s.scene || !m) return { marks, segments };
    const t = s.scene.startsWith("triangle") ? s.b : s.a,
        n = manifoldRotate(t.q, m.normal),
        f = Math.fround;
    for (let i = 0; i < m.pointCount; i++) {
        const p = m.points[i]!,
            at = manifoldTransform(t, p.point);
        const color = p.separation > 0 ? 0x797369 : INK;
        marks.push({ point: point(at), color });
        segments.push({
            a: point(at),
            b: [f(at.x + f(0.5 * n.x)), f(at.y + f(0.5 * n.y)), f(at.z + f(0.5 * n.z))],
            color: INK,
        });
    }
    return { marks, segments };
}
export function drawManifold(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const { marks, segments } = manifoldDrawing(world);
    for (const { a, b, color } of segments) line(a, b, color, JOINT_LINE);
    for (const mark of marks) drawInkPoint(world, camera, mark.point, line, mark.color, 10);
}
export function manifoldGeometry(world: World) {
    const s = world.resource(ManifoldSample);
    type Geometry = {
        mesh: { vertices: Float32Array; indices: Uint32Array };
        edges: [Point, Point][];
        pose: ManifoldPose;
        role: "ground" | "subject";
    };
    const geometries: Geometry[] = [];
    const smooth = (kind: number, t: ManifoldPose, role: Geometry["role"]) => {
        const q = {
            kind,
            geometry: 0,
            scale: [1, 1, 1] as Point,
            radius: kind === ShapeKind.Sphere ? s.sphere.radius : s.capsule.radius,
            sphereCenter: point(s.sphere.center),
            capsuleCenter1: point(s.capsule.center1),
            capsuleCenter2: point(s.capsule.center2),
        };
        geometries.push({ mesh: shapeMesh(world, q), edges: shapeEdges(world, q), pose: t, role });
    };
    const hull = (h: HullData, t: ManifoldPose, role: Geometry["role"]) => {
        const vertices: number[] = [],
            indices: number[] = [],
            edges: [Point, Point][] = [];
        for (const [i, face] of h.faces.entries()) {
            const normal = h.planes[i]!.normal,
                start = vertices.length / 8;
            let e = face.edge,
                count = 0;
            do {
                const p = h.points[h.edges[e]!.origin]!;
                vertices.push(p.x, p.y, p.z, 0, normal.x, normal.y, normal.z, 0);
                count++;
                e = h.edges[e]!.next;
            } while (e !== face.edge);
            for (let j = 1; j < count - 1; j++) indices.push(start, start + j, start + j + 1);
        }
        for (const [i, e] of h.edges.entries())
            if (i < e.twin)
                edges.push([point(h.points[e.origin]!), point(h.points[h.edges[e.twin]!.origin]!)]);
        geometries.push({
            mesh: { vertices: new Float32Array(vertices), indices: new Uint32Array(indices) },
            edges,
            pose: t,
            role,
        });
    };
    if (!s.scene) return geometries;
    if (s.scene.startsWith("triangle")) {
        const a = point(s.triangle[0]),
            b = point(s.triangle[1]),
            c = point(s.triangle[2]);
        const u = b.map((v, i) => v - a[i]!) as Point,
            v = c.map((v, i) => v - a[i]!) as Point;
        const n: Point = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ],
            length = Math.hypot(...n);
        const vertices = [a, b, c].flatMap((p) => [...p, 0, ...n.map((x) => x / length), 0]);
        geometries.push({
            mesh: {
                vertices: new Float32Array(vertices),
                indices: new Uint32Array([0, 1, 2, 0, 2, 1]),
            },
            edges: [
                [a, b],
                [b, c],
                [c, a],
            ],
            pose: s.a,
            role: "ground",
        });
    } else if (s.hullA) hull(s.hullA, s.a, "ground");
    else smooth(s.scene === "sphereVsSphere" ? ShapeKind.Sphere : ShapeKind.Capsule, s.a, "ground");
    if (s.hullB) hull(s.hullB, s.b, "subject");
    else smooth(s.scene.endsWith("Sphere") ? ShapeKind.Sphere : ShapeKind.Capsule, s.b, "subject");
    return geometries;
}
