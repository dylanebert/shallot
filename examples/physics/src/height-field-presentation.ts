import { Shape, ShapeKind, type World } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import { HeightFieldState } from "./mesh-height-field";

type Point = [number, number, number];
export function heightFieldDrawing(world: World): number[][] {
    const s = world.resource(HeightFieldState);
    if (!s.active) return [];
    const f = Math.fround;
    const v = s.values;
    const origin: Point = [v["ray x"], 4, v["ray z"]];
    const delta: Point = [v["delta x"], -8, v["delta z"]];
    const end = origin.map((p, i) => p + delta[i]!) as Point;
    const records = [
        [0, ...origin, v.radius === 0 ? 6 : 2],
        [0, ...end, v.radius === 0 ? 6 : 2],
        [1, ...origin, ...end],
    ];
    if (v.radius !== 0) {
        const fraction = s.hit ? s.result.fraction : 1;
        const center = origin.map((p, i) => f(p + f(fraction * delta[i]!)));
        records.push([4, ...center, 0, 0, 0, 1, 0, 0, 0, v.radius]);
    }
    if (s.hit) {
        const { point: p, normal: n } = s.result;
        records.push([
            1,
            p.x,
            p.y,
            p.z,
            p.x + f(0.5 * n.x),
            p.y + f(0.5 * n.y),
            p.z + f(0.5 * n.z),
        ]);
        records.push([0, p.x, p.y, p.z, v.radius === 0 ? 10 : 6]);
    }
    return records;
}
export function drawHeightFieldView(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const s = world.resource(HeightFieldState);
    for (const record of heightFieldDrawing(world)) {
        const a = record.slice(1, 4) as Point;
        if (record[0] === 0) drawInkPoint(world, camera, a, line, INK, record[4]);
        else if (record[0] === 1) line(a, record.slice(4, 7) as Point, INK, JOINT_LINE);
        else {
            const shape = world.get(s.ground, Shape)!;
            for (const [p, q] of shapeEdges(world, {
                ...shape,
                kind: ShapeKind.Sphere,
                sphereCenter: [0, 0, 0],
                radius: record[11]!,
            })) {
                line(
                    p.map((v, i) => v + a[i]!) as Point,
                    q.map((v, i) => v + a[i]!) as Point,
                    INK,
                    JOINT_LINE,
                );
            }
        }
    }
}
