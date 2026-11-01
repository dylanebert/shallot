import type { World } from "@dylanebert/shallot";
import { RestitutionState } from "./restitution";

type Point = [number, number, number];
export function restitutionLines(world: World): { a: Point; b: Point }[] {
    const s = world.resource(RestitutionState);
    const line = (x: number, y: number, halfWidth: number) => ({
        a: [x - halfWidth, y, 0] as Point,
        b: [x + halfWidth, y, 0] as Point,
    });
    switch (s.scene) {
        case "singleBoxRestitution":
            return [line(0, 5.5, 4)];
        case "singleSphereRestitution":
            return [line(0, 10.5, 4)];
        case "rotatedBoxRestitution":
            return [line(0, 10.5, 12)];
        case "sphereStackRestitution":
            return [
                line(
                    0,
                    Math.fround(
                        Math.fround(0.5 + s.stack) + Math.fround(2.5 * Math.fround(1 / 60)),
                    ),
                    2,
                ),
            ];
        case "bounceImpulse":
            return [line(0, Math.fround(0.5 + s.height), 2)];
        case "restitutionPropagation":
            return s.apex.flatMap((apex, i) => [
                line(4 * (i - 1), 5.5 + i, 1),
                ...(apex > 0 ? [line(4 * (i - 1), Math.fround(0.5 + i + apex), 1)] : []),
            ]);
        default:
            return [];
    }
}
