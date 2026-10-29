import { Shape, ShapeKind, type World } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import { VillageState } from "./village";

type Point = [number, number, number];
export function drawVillageView(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
    records = world.resource(VillageState).drawing,
): void {
    for (const r of records) {
        const a = r.slice(1, 4) as Point;
        if (r[0] === 0) drawInkPoint(world, camera, a, line, INK, r[4]);
        else if (r[0] === 1) line(a, r.slice(4, 7) as Point, INK, JOINT_LINE);
        else {
            const [eid] = world.query([Shape]);
            const shape = world.get(eid!, Shape)!;
            for (const [u, v] of shapeEdges(world, {
                ...shape,
                kind: ShapeKind.Sphere,
                sphereCenter: [0, 0, 0],
                radius: r[11]!,
            }))
                line(
                    u.map((x, j) => x + a[j]!) as Point,
                    v.map((x, j) => x + a[j]!) as Point,
                    INK,
                    JOINT_LINE,
                );
        }
    }
}
