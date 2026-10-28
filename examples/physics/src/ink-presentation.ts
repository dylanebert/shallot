import { GlobalTransform, qRotate, type World } from "@dylanebert/shallot";
import { worldToViewport } from "@dylanebert/shallot/rendering";

type Point = [number, number, number];
export const INK = 0xeee9df;
export const INK_DOT = { width: 4, depthBias: -1 };

export function drawInkPoint(
    world: World,
    camera: number,
    point: Point,
    line: (a: Point, b: Point, color: number, options: typeof INK_DOT) => void,
): void {
    const q = world.storage(GlobalTransform).rotation;
    const right = qRotate(
        q.x.get(camera),
        q.y.get(camera),
        q.z.get(camera),
        q.w.get(camera),
        1,
        0,
        0,
    );
    const center = worldToViewport(world, camera, point);
    const unit = worldToViewport(world, camera, [
        point[0] + right[0],
        point[1] + right[1],
        point[2] + right[2],
    ]);
    if (!center || !unit) return;
    const half = INK_DOT.width / Math.hypot(unit[0] - center[0], unit[1] - center[1]) / 2;
    line(
        [point[0] - right[0] * half, point[1] - right[1] * half, point[2] - right[2] * half],
        [point[0] + right[0] * half, point[1] + right[1] * half, point[2] + right[2] * half],
        INK,
        INK_DOT,
    );
}
