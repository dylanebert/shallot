import type { World } from "@dylanebert/shallot";
import type { HullData, MassData } from "@dylanebert/shallot/standard/physics";
import { Geometry } from "../src/geometry";

const point = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z];
function hullRecord(h: HullData, x: number): number[] {
    return [
        5,
        x,
        0,
        0,
        0,
        0,
        0,
        1,
        h.vertexCount,
        h.edgeCount,
        h.faceCount,
        h.surfaceArea,
        h.volume,
        h.innerRadius,
        ...point(h.center),
        ...point(h.centralInertia.cx),
        ...point(h.centralInertia.cy),
        ...point(h.centralInertia.cz),
        ...h.points.flatMap(point),
        ...h.edges.flatMap((e) => [e.next, e.twin, e.origin, e.face]),
    ];
}
function massRecord(m: MassData): number[] {
    return [
        6,
        m.mass,
        ...point(m.center),
        ...point(m.inertia.cx),
        ...point(m.inertia.cy),
        ...point(m.inertia.cz),
    ];
}
export function geometryRecords(world: World): number[][] {
    const s = world.resource(Geometry),
        rows: number[][] = [];
    if (s.scene === "capsuleMass") rows.push([3, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1]);
    for (const [i, h] of s.hulls.entries())
        if (h) rows.push(hullRecord(h, s.scene === "hullTransform" ? (i === 0 ? -2 : 2) : 0));
    for (const m of s.masses) rows.push(massRecord(m));
    return rows;
}
