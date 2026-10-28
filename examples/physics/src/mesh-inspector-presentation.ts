import type { World } from "@dylanebert/shallot";
import { INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import { MeshInspectorState } from "./mesh-inspector";

type Point = [number, number, number];
export function meshViewerBounds(world: World): number[][] {
    const s = world.resource(MeshInspectorState);
    if (!s.active || s.values["draw level"] < 0) return [];
    return (s.report?.nodeBounds[s.values["draw level"]] ?? []).map(
        ({ lowerBound: a, upperBound: b }) => [a.x, a.y, a.z, b.x, b.y, b.z],
    );
}
export function drawMeshViewerView(
    world: World,
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    for (const box of meshViewerBounds(world)) {
        const corner = (bits: number): Point => [
            box[bits & 1 ? 3 : 0]!,
            box[bits & 2 ? 4 : 1]!,
            box[bits & 4 ? 5 : 2]!,
        ];
        for (let bits = 0; bits < 8; bits++)
            for (const axis of [1, 2, 4])
                if (!(bits & axis)) line(corner(bits), corner(bits | axis), INK, JOINT_LINE);
    }
}
