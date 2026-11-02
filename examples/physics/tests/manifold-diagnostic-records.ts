import type { World } from "@dylanebert/shallot";
import type { HullData } from "@dylanebert/shallot/standard/physics";
import type { Arc } from "../src/manifold-diagnostics";
import { ManifoldSample } from "../src/manifold-state";

const xyz = (p: { x: number; y: number; z: number }) => [p.x, p.y, p.z];
export function manifoldDiagnosticRecords(world: World): number[][] {
    const s = world.resource(ManifoldSample),
        rows: number[][] = [[-1, Number(s.useCache), s.manualFeature, Number(s.drawPoints)]];
    function hull(h: HullData, role: number) {
        const hash = Array.from({ length: 4 }, (_, i) =>
            Number((h.hash >> BigInt(16 * i)) & 0xffffn),
        );
        rows.push([
            20,
            role,
            h.vertexCount,
            h.edgeCount,
            h.faceCount,
            h.surfaceArea,
            h.volume,
            h.innerRadius,
            ...xyz(h.center),
            ...xyz(h.aabb.lowerBound),
            ...xyz(h.aabb.upperBound),
            ...xyz(h.centralInertia.cx),
            ...xyz(h.centralInertia.cy),
            ...xyz(h.centralInertia.cz),
            ...hash,
        ]);
        for (let i = 0; i < h.vertexCount; i++)
            rows.push([21, role, i, ...xyz(h.points[i]!), h.vertices[i]!.edge]);
        for (let i = 0; i < h.edgeCount; i++) {
            const e = h.edges[i]!;
            rows.push([22, role, i, e.next, e.twin, e.origin, e.face]);
        }
        for (let i = 0; i < h.faceCount; i++) {
            const p = h.planes[i]!;
            rows.push([23, role, i, h.faces[i]!.edge, ...xyz(p.normal), p.offset]);
        }
        rows.push([24, role, ...h.edgeCosines]);
    }
    hull(s.hullA!, 0);
    hull(s.hullB!, 1);
    if (s.culling) {
        const c = s.culling;
        rows.push([
            30,
            Number(c.cylinders),
            c.height,
            c.radius,
            c.sides,
            Number(c.showSpheres),
            Number(c.showCulled),
            Number(c.showNormals),
            c.edgeBound,
            c.seedA,
            c.seedB,
            c.probeA,
            c.probeB,
            c.keptA,
            c.keptB,
            c.sphereKeptA,
            c.sphereKeptB,
            c.separationA,
            c.separationB,
            c.gap,
            c.separatedFeature,
            Number(c.probeApplied),
        ]);
        rows.push([31, ...c.facesA], [32, ...c.facesB], [33, ...c.edgesA], [34, ...c.edgesB]);
    } else if (s.wedge) {
        const w = s.wedge;
        rows.push([
            40,
            w.hullType,
            w.seed & 0xffff,
            w.seed >>> 16,
            w.searchCount,
            Number(w.found),
            Number(w.edgeWins),
            w.faceA,
            w.faceB,
            w.radius,
            ...xyz(w.gaussCenter),
            w.gaussRadius,
            ...xyz(w.centerOffset),
            ...xyz(w.edge.normal),
            w.edge.separation,
            w.edge.edgeA,
            w.edge.edgeB,
        ]);
        const arc = (tag: number, a: Arc) =>
            rows.push([
                tag,
                ...xyz(a.u),
                ...xyz(a.v),
                ...xyz(a.peak),
                a.a,
                a.b,
                a.c,
                a.separationU,
                a.separationV,
                a.outsideDegrees,
                Number(a.inside),
            ]);
        arc(41, w.arcA!);
        arc(42, w.arcB!);
    }
    return rows;
}
