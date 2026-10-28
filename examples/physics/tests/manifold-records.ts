import type { World } from "@dylanebert/shallot";
import type { HullData } from "@dylanebert/shallot/standard/physics";
import { type ManifoldPose, ManifoldSample } from "../src/manifold";
import { manifoldDrawing } from "../src/manifold-presentation";
export function manifoldRecords(w: World): number[][] {
    const s = w.resource(ManifoldSample),
        m = s.manifold!;
    const pose = (t: ManifoldPose) => [0, t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s];
    const rows = [
        pose(s.a),
        pose(s.b),
        [
            6,
            m.pointCount,
            ...(m.pointCount
                ? [m.normal.x, m.normal.y, m.normal.z, m.feature, m.squaredDistance]
                : [0, 0, 0, 0, 0]),
            ...m.points
                .slice(0, m.pointCount)
                .flatMap((p) => [
                    p.point.x,
                    p.point.y,
                    p.point.z,
                    p.separation,
                    p.pair.owner1,
                    p.pair.index1,
                    p.pair.owner2,
                    p.pair.index2,
                    p.triangleIndex,
                ]),
        ],
        [
            7,
            s.simplex.metric,
            s.simplex.count,
            ...s.simplex.indexA,
            ...s.simplex.indexB,
            s.sat.separation,
            s.sat.type,
            s.sat.indexA,
            s.sat.indexB,
            s.sat.hit,
        ],
    ];
    if (s.scene.startsWith("triangle"))
        rows.push([5, ...s.triangle.flatMap((v) => [v.x, v.y, v.z])]);
    const sphere = () =>
        rows.push([1, s.sphere.center.x, s.sphere.center.y, s.sphere.center.z, s.sphere.radius]);
    const capsule = () =>
        rows.push([
            2,
            s.capsule.center1.x,
            s.capsule.center1.y,
            s.capsule.center1.z,
            s.capsule.center2.x,
            s.capsule.center2.y,
            s.capsule.center2.z,
            s.capsule.radius,
        ]);
    const hull = (h: HullData, role: number) =>
        rows.push([role, ...h.points.flatMap((v) => [v.x, v.y, v.z])]);
    switch (s.scene) {
        case "sphereVsSphere":
            sphere();
            break;
        case "capsuleVsSphere":
            capsule();
            sphere();
            break;
        case "hullVsSphere":
            hull(s.hullA!, 3);
            sphere();
            break;
        case "triangleVsSphere":
            sphere();
            break;
        case "capsuleVsCapsule":
            capsule();
            break;
        case "capsuleVsHull":
            hull(s.hullA!, 3);
            capsule();
            break;
        case "triangleVsCapsule":
            capsule();
            break;
        case "hullVsHull":
            hull(s.hullA!, 3);
            hull(s.hullB!, 4);
            break;
        case "triangleVsHull":
            hull(s.hullB!, 4);
            break;
    }
    const drawing = manifoldDrawing(w);
    for (const [i, line] of drawing.segments.entries())
        rows.push([8, ...line.a, ...line.b], [9, ...drawing.marks[i]!.point, 10]);
    return rows;
}
