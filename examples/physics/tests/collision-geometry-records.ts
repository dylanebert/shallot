import type { World } from "@dylanebert/shallot";
import { box } from "../src/collision-debug";
import { CollisionGeometry } from "../src/collision-geometry";
export function collisionGeometryRecords(world: World): number[][] {
    const s = world.resource(CollisionGeometry);
    if (s.scene === "hull2D")
        return [
            [
                -1,
                s.source,
                s.input.length,
                s.hull.length,
                s.simplified.length,
                s.targetCount,
                s.pointCount,
                s.drawScale,
                Number(s.drawLabels),
            ],
            ...s.input.map((p) => [0, p.p.x, p.p.y, p.separation, p.originalIndex]),
            ...s.hull.map((p) => [1, p.p.x, p.p.y, p.separation, p.originalIndex]),
            ...s.simplified.map((p) => [2, p.p.x, p.p.y, p.separation, p.originalIndex]),
        ];
    const d = s.cast;
    if (!d) return [];
    const t = s.transform;
    const v = s.translation;
    return [
        [-1, 8, 0, 8, 0],
        ...box(3, 0.25, 3, -0.25).map((p) => [0, p.x, p.y, p.z]),
        ...box(0.5, 0.5, 0.5).map((p) => [1, p.x, p.y, p.z]),
        [2, t.p.x, t.p.y, t.p.z, t.q.v.x, t.q.v.y, t.q.v.z, t.q.s],
        [3, v.x, v.y, v.z, 1, Number(s.canEncroach)],
        [
            4,
            d.point.x,
            d.point.y,
            d.point.z,
            d.normal.x,
            d.normal.y,
            d.normal.z,
            d.fraction,
            d.iterations,
            Number(d.hit),
        ],
    ];
}
