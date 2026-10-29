import { Shape, ShapeKind, type World } from "@dylanebert/shallot";
import { shapeEdges } from "@dylanebert/shallot/standard/physics";
import { CharacterSample, validNormal } from "./character";
import { drawInkPoint, INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";

type Point = [number, number, number];
const CYAN = 0x38afc0;
const ORANGE = 0xff8135;
export function characterDrawing(world: World): { records: number[][]; colors: number[] } {
    const s = world.resource(CharacterSample);
    const records: number[][] = [],
        colors: number[] = [];
    const add = (record: number[], color = INK) => {
        records.push(record);
        colors.push(color);
    };
    const f = Math.fround;
    const p = s.position;
    const capsule = (position: typeof p, color = INK) =>
        add(
            [
                3,
                position.x,
                position.y,
                position.z,
                0,
                0,
                0,
                1,
                s.capsule.center1.x,
                s.capsule.center1.y,
                s.capsule.center1.z,
                s.capsule.center2.x,
                s.capsule.center2.y,
                s.capsule.center2.z,
                s.capsule.radius,
            ],
            color,
        );
    capsule(p);
    for (let i = 0; i < s.planes.count; i++) {
        const r = s.planes.records[i]!.plane;
        const n = r.plane.normal;
        const scale = f(r.plane.offset - s.capsule.radius);
        const at: Point =
            s.scene === "capsulePlane"
                ? [f(p.x + f(scale * n.x)), f(p.y + f(scale * n.y)), f(p.z + f(scale * n.z))]
                : [f(p.x + r.point.x), f(p.y + r.point.y), f(p.z + r.point.z)];
        const color = s.scene === "moverOverlap" && !validNormal(n) ? ORANGE : INK;
        const length = s.scene === "capsulePlane" ? f(0.1) : 0.5;
        add([0, ...at, s.scene === "capsulePlane" ? 5 : 6], color);
        add(
            [
                s.scene === "capsulePlane" ? 1 : 2,
                ...at,
                f(at[0] + f(length * n.x)),
                f(at[1] + f(length * n.y)),
                f(at[2] + f(length * n.z)),
            ],
            color,
        );
    }
    if (s.scene === "moverOverlap") capsule(s.pushed, CYAN);
    return { records, colors };
}
export function drawCharacterView(
    world: World,
    camera: number,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
    records = characterDrawing(world).records,
): void {
    const { colors } = characterDrawing(world);
    const shape = world.get(world.query([Shape])[Symbol.iterator]().next().value!, Shape)!;
    for (let i = 0; i < records.length; i++) {
        const r = records[i]!,
            color = colors[i]!;
        const a = r.slice(1, 4) as Point;
        if (r[0] === 0) drawInkPoint(world, camera, a, line, color, r[4]);
        else if (r[0] === 3) {
            const edges = shapeEdges(world, {
                ...shape,
                kind: ShapeKind.Capsule,
                capsuleCenter1: r.slice(8, 11) as Point,
                capsuleCenter2: r.slice(11, 14) as Point,
                radius: r[14]!,
            });
            for (const [u, v] of edges)
                line(
                    u.map((x, j) => x + a[j]!) as Point,
                    v.map((x, j) => x + a[j]!) as Point,
                    color,
                    JOINT_LINE,
                );
        } else {
            const b = r.slice(4, 7) as Point;
            line(a, b, color, JOINT_LINE);
            if (r[0] === 2) {
                const dx = b[0] - a[0],
                    dy = b[1] - a[1],
                    dz = b[2] - a[2];
                const length = Math.hypot(dx, dy, dz);
                if (length > 0) {
                    const side: Point = Math.abs(dx / length) > 0.5 ? [dy, -dx, 0] : [0, dz, -dy];
                    const norm = Math.hypot(...side);
                    for (const sign of [-1, 1])
                        line(
                            b,
                            b.map(
                                (x, j) =>
                                    x -
                                    0.15 * [dx, dy, dz][j]! +
                                    (sign * 0.075 * length * side[j]!) / norm,
                            ) as Point,
                            color,
                            JOINT_LINE,
                        );
                }
            }
        }
    }
}
