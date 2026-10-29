import { INK } from "./ink-presentation";
import { JOINT_LINE } from "./joint-presentation";
import type { Mover } from "./mover";

type Point = [number, number, number];

export function pogoDrawing(mover: Mover): number[][] {
    const ray = mover.pogoRay;
    return ray
        ? [
              [
                  Number(mover.onGround),
                  ray.origin.x,
                  ray.origin.y,
                  ray.origin.z,
                  ray.end.x,
                  ray.end.y,
                  ray.end.z,
              ],
          ]
        : [];
}

export function drawPogoRay(
    records: number[][],
    line: (a: Point, b: Point, color: number, options: typeof JOINT_LINE) => void,
): void {
    const ray = records[0];
    if (ray)
        line(
            ray.slice(1, 4) as Point,
            ray.slice(4, 7) as Point,
            ray[0] ? INK : 0x797369,
            JOINT_LINE,
        );
}
