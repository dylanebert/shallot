import { Shape, ShapeKind, type World } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { DeterminismState } from "./determinism";
import { JOINT_INK, JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
export function querySpawnDrawing(world: World): { a: Point; b: Point }[] {
    const s = world.resource(DeterminismState);
    if (!s.spawnCount) return [];
    const lines = [{ a: s.origin, b: s.rayPoint }];
    if (s.rayHit)
        lines.push({ a: s.rayPoint, b: s.rayPoint.map((v, i) => v + s.normal[i]!) as Point });
    const corners = Array.from(
        { length: 8 },
        (_, n) => s.lower.map((v, i) => (n & (1 << i) ? s.upper[i]! : v)) as Point,
    );
    for (let i = 0; i < 8; i++)
        for (let axis = 0; axis < 3; axis++)
            if (!(i & (1 << axis))) lines.push({ a: corners[i]!, b: corners[i | (1 << axis)]! });
    const center = s.origin.map((v, i) => v + s.fraction * s.translation[i]!) as Point;
    for (const [a, b] of shapeEdges(world, {
        ...world.get(s.bodies[0]!, Shape)!,
        kind: ShapeKind.Sphere,
        sphereCenter: [0, 0, 0],
        radius: 0.5,
    }))
        lines.push({
            a: a.map((v, i) => v + center[i]!) as Point,
            b: b.map((v, i) => v + center[i]!) as Point,
        });
    return lines;
}
export function drawQuerySpawn(
    world: World,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    for (const { a, b } of querySpawnDrawing(world)) line(a, b, JOINT_INK, JOINT_LINE);
}
